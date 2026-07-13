"""Audit whether the learned CAD Pair-Pose route is ready to advance.

This is a *route gate*, not a training script.  It prevents an attractive
metric on an easy local-contact split from being mistaken for an end-to-end,
cross-domain CAD assembly result.  It never reads SolidWorks exam files.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
import sys
from typing import Any

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_PURE = Path(r"D:\Model_match_public_data\fusion360_pure_brep_v1")
DEFAULT_PAIR = Path(r"D:\Model_match_public_data\joinable_pose_contact_v3_full")
DEFAULT_TRAINING = Path(r"D:\Model_match_public_data\joinable_pose_contact_v3_full_train")
DEFAULT_STRONG = ROOT / "public_cad_dataset_audit" / "outputs" / "fusion360_strong_contact_pose_v1"
DEFAULT_TRAINING_PYTHON = Path(r"D:\Model_match_envs\joinable_gpu\python.exe")


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _npz_audit(path: Path) -> dict[str, Any]:
    arrays = np.load(path)
    required = {"pair_embedding", "target_pose", "free_dof_mask", "patch_a", "patch_b", "contact_reference"}
    missing = sorted(required.difference(arrays.files))
    if missing:
        return {"status": "invalid", "missing_arrays": missing}
    embedding, pose = arrays["pair_embedding"], arrays["target_pose"]
    dof, contact = arrays["free_dof_mask"], arrays["contact_reference"]
    patterns, counts = np.unique(dof, axis=0, return_counts=True)
    return {
        "status": "ok",
        "examples": int(len(embedding)),
        "array_shapes": {name: list(arrays[name].shape) for name in required},
        "embedding_nonzero_rate": float((np.linalg.norm(embedding, axis=1) > 1e-8).mean()),
        "embedding_std": float(embedding.std()),
        "free_dof_patterns": {
            str(pattern.astype(int).tolist()): int(count)
            for pattern, count in zip(patterns, counts)
        },
        "translation_norm_quantiles": [
            float(value) for value in np.quantile(np.linalg.norm(pose[:, :3], axis=1), (0, .5, .9, .99, 1))
        ],
        "contact_bearing_rate": float((contact[:, 3] > .5).mean()),
    }


def _markdown(report: dict[str, Any]) -> str:
    full = report["full_pair_pose_dataset"]
    test = report["full_training_baseline"].get("test", {})
    strong = report["strong_contact_baseline"].get("overall", {})
    lines = [
        "# 治本路线可行性与数据准入审计",
        "",
        "## 结论",
        "",
        "项目具备训练真正 B-Rep Pair Pose / 接口评分网络的主要正样本、相对 Pose、局部接口面片和 assembly 隔离切分；但当前完整模型的 holdout 分数不足，不能用于 SolidWorks 考试或宣称功能语义正确。",
        "",
        "## 已具备的硬条件",
        "",
        f"- Fusion B-Rep 合同：{report['pure_brep_contract'].get('record_count', 0)} 个记录，{report['pure_brep_contract'].get('joint_supervision_count', 0)} 条 joint supervision，split overlap 为 {report['pure_brep_contract'].get('split_group_overlap_count', 'unknown')}。",
        f"- 完整 Pair Pose 训练集：train/dev/test = {full['train'].get('examples', 0)}/{full['dev'].get('examples', 0)}/{full['test'].get('examples', 0)}；三份 embedding 非零率分别为 {full['train'].get('embedding_nonzero_rate', 0):.3f}/{full['dev'].get('embedding_nonzero_rate', 0):.3f}/{full['test'].get('embedding_nonzero_rate', 0):.3f}。",
        f"- 自由度标签不是全零：训练集中有 {len(full['train'].get('free_dof_patterns', {}))} 类 DOF mask。",
        f"- 局部强接触基线：rank-1 = {strong.get('true_contact_rank1_rate', 0):.3f}，但它只评测真实接触对抗局部扰动。",
        "",
        "## 不能误读的结果",
        "",
        f"- 完整 Pair Pose + 接口评分模型的 assembly-holdout rank-1 仅为 {test.get('score_positive_rank1_rate', 0):.3f}。这是目前唯一应作为完整模型基线使用的数字。",
        "- 强接触集的高分不能替代真实多模式 Pose、同装配干扰接口、跨 CAD 域与多零件闭合测试。",
        "",
        "## 尚未补齐的硬条件",
        "",
        "1. **Pose 等价类**：同一接口的对称旋转、可滑移区间和多种有效装配状态尚未组成一个正例集合；单一 Pose 标签会把其他有效解误作负例。",
        "2. **困难负例**：需要由真实 occurrence Pose 自动构造近接触滑移、翻转、穿透、错误插入深度，以及同装配的错接口负例；不能仅用随机局部噪声。",
        "3. **接口闭合 target**：现有 gap/coverage/normal target 没有记录完整插入长度、包络比例、重复孔阵列一致性等可由 B-Rep 自动量测的 target。",
        "4. **外部考试标签**：当前没有隔离的 SolidWorks 真实装配 Pose 真值，因此不能量化跨 CAD 域泛化，更不能据 case1–5 调参。",
        "5. **GPU 训练环境**：默认 OCCT 环境为 CPU-only；预检会单独探测 GPU 训练环境，二者不得互相覆盖。",
        "",
        "## 允许的下一步",
        "",
        "- 先从 Fusion occurrence 与 B-Rep 自动生成等价 Pose 集和困难负例；不读取 SolidWorks 考试目录。",
        "- 训练 Pair Pose top-k 与 candidate-conditioned Interface Scorer，并在 Fusion assembly holdout 上分别报告 top-k Pose 误差、真实接触排序、错误自动接受率。",
        "- 只有完整训练基线显著超过当前 0.597 rank-1，且外部真值 benchmark 不回退，才接入多零件因子图。",
        "- SolidWorks case1–5 只作为冻结考试；若无真值，结果只能做人工可视化 review。",
    ]
    return "\n".join(lines) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pure-brep", type=Path, default=DEFAULT_PURE)
    parser.add_argument("--pair-dataset", type=Path, default=DEFAULT_PAIR)
    parser.add_argument("--training", type=Path, default=DEFAULT_TRAINING)
    parser.add_argument("--strong-contact", type=Path, default=DEFAULT_STRONG)
    parser.add_argument("--training-python", type=Path, default=DEFAULT_TRAINING_PYTHON,
                        help="Dedicated Python used only for neural training/runtime checks.")
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    pure = _load_json(args.pure_brep / "pure_brep_contract_audit.json")
    full = {split: _npz_audit(args.pair_dataset / f"{split}.npz") for split in ("train", "dev", "test")}
    training = _load_json(args.training / "training_report.json")
    strong = _load_json(args.strong_contact / "strong_contact_holdout.json")
    probe = (
        "import json,torch; print(json.dumps({'torch':torch.__version__,"
        "'cuda_available':bool(torch.cuda.is_available()),'cuda_version':torch.version.cuda,"
        "'device':torch.cuda.get_device_name(0) if torch.cuda.is_available() else None}))"
    )
    try:
        raw = subprocess.check_output(
            [str(args.training_python), "-c", probe], text=True, stderr=subprocess.STDOUT, timeout=30
        )
        runtime = json.loads(raw.strip().splitlines()[-1]) | {"python": str(args.training_python)}
    except Exception as exc:  # pragma: no cover - diagnostic fallback
        runtime = {"status": f"training_python_unavailable:{type(exc).__name__}", "python": str(args.training_python)}
    full_score = float((training.get("test") or {}).get("score_positive_rank1_rate", 0.0))
    gates = {
        "leakage_safe_assembly_split": bool(pure.get("passed") and pure.get("split_group_overlap_count") == 0),
        "nonzero_joinable_embeddings": all(
            full[split].get("embedding_nonzero_rate", 0.0) >= .99 for split in full
        ),
        "dof_supervision_present": len(full["train"].get("free_dof_patterns", {})) >= 3,
        "full_baseline_ready_for_exam": full_score >= .75,
        "cuda_training_ready": bool(runtime.get("cuda_available")),
        "external_solidworks_pose_ground_truth_available": False,
        "functional_semantic_labels_available": False,
    }
    report = {
        "schema_version": "root_cause_route_readiness.v1",
        "scope": "Fusion-only training readiness; SolidWorks cases are excluded from all model data.",
        "pure_brep_contract": pure,
        "full_pair_pose_dataset": full,
        "full_training_baseline": training,
        "strong_contact_baseline": strong,
        "runtime": runtime,
        "gates": gates,
        "verdict": {
            "route_feasible": all(gates[key] for key in (
                "leakage_safe_assembly_split", "nonzero_joinable_embeddings", "dof_supervision_present"
            )),
            "ready_to_claim_case_generalization": False,
            "next_required_work": [
                "build symmetry/continuous-DOF equivalence sets from recorded Fusion occurrences",
                "build geometry-valid-but-wrong hard negatives from same-assembly distractors and measured SE(3) perturbations",
                "add learned/derived interface-closure targets before changing the global solver",
                "create a separate CUDA training environment",
                "obtain or create an evaluation-only SolidWorks ground-truth Pose benchmark",
            ],
        },
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "root_cause_readiness.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    (args.output_dir / "root_cause_readiness.md").write_text(_markdown(report), encoding="utf-8")
    print(json.dumps({"output_dir": str(args.output_dir.resolve()), "gates": gates}, ensure_ascii=False))


if __name__ == "__main__":
    main()
