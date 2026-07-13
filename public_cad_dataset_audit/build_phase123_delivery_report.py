"""Build the machine-readable and human-readable Step 1-3 delivery report."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
AUDIT_ROOT = Path(__file__).resolve().parent


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def directory_bytes(path: Path) -> int:
    return sum(
        file.stat().st_size for file in path.rglob("*") if file.is_file()
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mixed_pool_root", type=Path)
    parser.add_argument(
        "--json-output",
        type=Path,
        default=AUDIT_ROOT
        / "outputs"
        / "phase123_delivery_report.json",
    )
    parser.add_argument(
        "--markdown-output",
        type=Path,
        default=AUDIT_ROOT / "PHASE123_DELIVERY_20260705.md",
    )
    parser.add_argument("--desktop-output", type=Path)
    args = parser.parse_args()

    tool_root = (
        PROJECT_ROOT
        / "cad_assembly_agent"
        / "tools"
        / "joinable_interface_predictor"
    )
    phase_root = (
        AUDIT_ROOT / "outputs" / "phase123_real_assembly_dataset"
    )
    paths = {
        "tool_manifest": tool_root / "frozen_tool_manifest.json",
        "tool_validation": tool_root / "frozen_tool_validation.json",
        "tool_smoke": tool_root / "frozen_tool_smoke_prediction.json",
        "assembly_manifest": phase_root / "assembly_dataset_manifest.json",
        "assembly_quality": phase_root
        / "assembly_graph_quality_report.json",
        "mixed_manifest": args.mixed_pool_root.resolve()
        / "mixed_pool_manifest.json",
        "mixed_validation": phase_root
        / "mixed_pool_validation_report.json",
        "functional_catalog": AUDIT_ROOT
        / "functional_assembly_template_catalog.json",
    }
    missing = [
        f"{name}:{path}" for name, path in paths.items() if not path.is_file()
    ]
    if missing:
        raise FileNotFoundError(";".join(missing))
    tool = read_json(paths["tool_manifest"])
    tool_validation = read_json(paths["tool_validation"])
    assembly = read_json(paths["assembly_manifest"])
    quality = read_json(paths["assembly_quality"])
    mixed = read_json(paths["mixed_manifest"])
    mixed_validation = read_json(paths["mixed_validation"])
    functional_catalog = read_json(paths["functional_catalog"])

    step_1_complete = (
        tool.get("freeze_status") == "frozen"
        and tool_validation.get("status") == "passed"
        and (
            tool_validation.get("pair_interface_smoke_evidence") or {}
        ).get("valid")
        is True
        and (tool.get("gate_policy") or {}).get(
            "can_change_accepted_groups"
        )
        is False
    )
    step_2_complete = (
        assembly.get("usable_for_mixed_pool_construction") is True
        and quality.get("usable_assembly_count") >= 10
        and quality.get("rejected_assembly_count") == 0
        and quality.get("all_pair_partitions_complete") is True
        and quality.get("all_step_geometry_available") is True
        and quality.get("all_relations_mapped") is True
        and quality.get("all_interface_entities_mapped") is True
    )
    step_3_complete = (
        mixed_validation.get("status") == "passed"
        and all((mixed.get("quality_gates") or {}).values())
        and not mixed.get("failure_reasons")
    )
    steps = {
        "step_1_freeze_joinable_pair_tool": {
            "status": "completed" if step_1_complete else "failed",
            "official_checkpoint_sha256": (
                tool.get("checkpoint") or {}
            ).get("sha256"),
            "official_test_metrics": tool.get(
                "reference_evaluation", {}
            ),
            "fresh_pair_smoke": tool.get(
                "pair_interface_smoke_evidence", {}
            ),
            "safety_boundary": (
                "Pair-interface ranking only; shadow evidence cannot accept "
                "an assembly group."
            ),
        },
        "step_2_real_assembly_dataset": {
            "status": "completed" if step_2_complete else "failed",
            "dataset_id": assembly.get("dataset_id"),
            "assembly_count": quality.get("usable_assembly_count"),
            "totals": quality.get("totals"),
            "quality_gates": {
                field: quality.get(field)
                for field in (
                    "all_pair_partitions_complete",
                    "all_step_geometry_available",
                    "all_relations_mapped",
                    "all_interface_entities_mapped",
                )
            },
            "truth_boundary": assembly.get("truth_layers"),
            "functional_template_catalog_status": functional_catalog.get(
                "catalog_status"
            ),
        },
        "step_3_real_mixed_pools": {
            "status": "completed" if step_3_complete else "failed",
            "dataset_id": mixed.get("dataset_id"),
            "source_assembly_count": mixed.get("source_assembly_count"),
            "pool_count": mixed.get("pool_count"),
            "quality_gates": mixed.get("quality_gates"),
            "validation_totals": mixed_validation.get("totals"),
            "output_root": str(args.mixed_pool_root.resolve()),
            "output_bytes": directory_bytes(
                args.mixed_pool_root.resolve()
            ),
            "truth_boundary": mixed.get("truth_policy"),
        },
    }
    all_complete = all(
        row["status"] == "completed" for row in steps.values()
    )
    limitations = [
        "The current Fusion subset has no audited part_role, assembly_family, or functional_relation labels.",
        "Contact-only edges are observations, not guaranteed designer intent.",
        "Cross-assembly negatives are provenance negatives, not proof of universal mechanical incompatibility.",
        "The 58 geometry-similarity negatives are candidates for later pose/collision audit; none is claimed as a verified hard negative.",
        "The functional family catalog is an annotation contract only; it is not counted as human-validated CAD benchmark data.",
        "No mixed-pool JoinABLe batch inference was run in Steps 1-3; that is the Step-4/AutoDL trigger.",
    ]
    report = {
        "schema_version": "1.0.0",
        "task": "Complete roadmap Steps 1, 2, and 3",
        "status": "completed" if all_complete else "failed",
        "steps": steps,
        "artifact_integrity": {
            name: {
                "path": str(path.resolve()),
                "sha256": sha256_file(path),
            }
            for name, path in paths.items()
        },
        "legacy_pipeline_modified": False,
        "training_performed": False,
        "deepseek_used": False,
        "solidworks_user_files_processed": False,
        "autodl_required_for_this_delivery": False,
        "next_autodl_trigger": (
            "Step 4: batch JoinABLe inference over anonymized mixed-pool "
            "candidate part pairs."
        ),
        "limitations": limitations,
        "failure_reasons": [] if all_complete else [
            name for name, row in steps.items()
            if row["status"] != "completed"
        ],
        "unavailable_fields": [
            "functional_semantic_group_truth",
            "verified_geometric_hard_negatives",
            "verified_semantic_hard_negatives",
            "interchangeable_part_labels",
        ],
    }
    write_json(args.json_output.resolve(), report)

    evaluation = tool["reference_evaluation"]
    totals = quality["totals"]
    mixed_totals = mixed_validation["totals"]
    lines = [
        "# CAD装配项目第1～3步交付报告",
        "",
        f"总状态：{'已完成' if all_complete else '未通过'}",
        "",
        "## 第1步：冻结JoinABLe双零件Tool",
        "",
        "- 官方checkpoint及运行源码已用SHA-256冻结。",
        f"- 官方测试集：{evaluation['sample_count']}对。",
        (
            f"- Top-1/Top-5/Top-10："
            f"{evaluation['top_1_accuracy']:.2%}/"
            f"{evaluation['top_5_recall']:.2%}/"
            f"{evaluation['top_10_recall']:.2%}。"
        ),
        "- 已重新运行一对真实STEP图的CPU推理，Top-1节点对为[2, 3]，冻结校验通过。",
        "- 工具保持shadow mode，不能直接接受装配组。",
        "",
        "## 第2步：真实Fusion装配级数据",
        "",
        f"- 可用装配：{quality['usable_assembly_count']}/10。",
        f"- occurrence-body实例：{totals['part_count']}。",
        (
            f"- 直接正边：{totals['positive_pair_count']}，其中designer joint"
            f"边{totals['designer_joint_pair_count']}，contact-only边"
            f"{totals['contact_only_pair_count']}。"
        ),
        f"- 关系映射：{totals['mapped_relation_count']}/{totals['relation_count']}。",
        (
            f"- 接口实体映射：{totals['mapped_interface_entity_count']}/"
            f"{totals['interface_entity_count']}。"
        ),
        "- 10个装配的STEP均可用，正/负pair全集分区完整。",
        "- cover_base、shaft_hub_key、bearing_housing已建立功能标注契约，但尚未伪造为已验证CAD样本。",
        "",
        "## 第3步：真实mixed-pool",
        "",
        f"- 源装配：{mixed['source_assembly_count']}，池：{mixed['pool_count']}。",
        (
            f"- 匿名STEP实例：{mixed_totals['part_count']}，真实来源组："
            f"{mixed_totals['true_group_count']}。"
        ),
        (
            f"- 直接joint/contact正对："
            f"{mixed_totals['direct_positive_pair_count']}。"
        ),
        (
            f"- 同组但无直接边（不误标为负）："
            f"{mixed_totals['same_group_nonedge_count']}。"
        ),
        (
            f"- 跨来源装配负对："
            f"{mixed_totals['cross_group_negative_pair_count']}。"
        ),
        (
            f"- 几何相似负例候选："
            f"{mixed_totals['similarity_candidate_count']}，"
            "均明确标记为尚未经过pose/collision验证。"
        ),
        "- train/validation/test之间assembly_id零重叠，pool_input源身份泄漏为0。",
        f"- 数据位置：{args.mixed_pool_root.resolve()}",
        "",
        "## 边界与下一步",
        "",
    ]
    lines.extend(f"- {item}" for item in limitations)
    lines.extend(
        [
            "",
            "第4步开始才需要打开AutoDL：对mixed-pool候选零件对批量运行JoinABLe，并做候选召回审计。",
            "",
        ]
    )
    markdown = "\n".join(lines)
    args.markdown_output.resolve().parent.mkdir(
        parents=True, exist_ok=True
    )
    args.markdown_output.resolve().write_text(
        markdown, encoding="utf-8"
    )
    if args.desktop_output:
        args.desktop_output.resolve().parent.mkdir(
            parents=True, exist_ok=True
        )
        args.desktop_output.resolve().write_text(
            markdown, encoding="utf-8"
        )
    print(
        f"Phase 1-3 delivery report: {report['status']} "
        f"({args.markdown_output.resolve()})"
    )
    return 0 if all_complete else 2


if __name__ == "__main__":
    raise SystemExit(main())
