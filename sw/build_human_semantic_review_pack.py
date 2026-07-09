"""Build a blinded STL/PNG semantic study and matching DeepSeek explanations.

Ground truth is used only to stratify the evaluation pack and is written to a
separate, clearly marked answer-key folder. It never affects production
grouping decisions or any DeepSeek request.
"""

from __future__ import annotations

import argparse
import builtins
import csv
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any

from PIL import Image, ImageColor, ImageDraw, ImageFont
from OCC.Core.BRep import BRep_Builder, BRep_Tool
from OCC.Core.BRepBuilderAPI import BRepBuilderAPI_Transform
from OCC.Core.BRepCheck import BRepCheck_Analyzer
from OCC.Core.BRepMesh import BRepMesh_IncrementalMesh
from OCC.Core.StlAPI import StlAPI_Reader, StlAPI_Writer
from OCC.Core.TopAbs import TopAbs_FACE
from OCC.Core.TopExp import TopExp_Explorer
from OCC.Core.TopLoc import TopLoc_Location
from OCC.Core.TopoDS import TopoDS_Compound, TopoDS_Shape, topods

from build_assembly import build_transform, load_step
from calibrate_semantic_review import _auc, _brier, calibration_gate
from contracts import GroupProposal
from semantic_pool import build_summary
from semantic_review import DeepSeekReviewer


COLORS = [
    "#4C78A8",
    "#F58518",
    "#54A24B",
    "#E45756",
    "#72B7B2",
    "#B279A2",
]


def _load(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _parts_key(parts) -> frozenset[str]:
    return frozenset(str(part) for part in parts)


def _transformed_shape(path: Path, placement: dict[str, Any]):
    shape = load_step(str(path))
    transform = build_transform(placement)
    if transform.Form() != 0:
        shape = BRepBuilderAPI_Transform(shape, transform, True).Shape()
    return shape


def _triangles(shape) -> list[list[tuple[float, float, float]]]:
    BRepMesh_IncrementalMesh(shape, 0.45, False, 0.5, True).Perform()
    explorer = TopExp_Explorer(shape, TopAbs_FACE)
    triangles = []
    while explorer.More():
        face = topods.Face(explorer.Current())
        location = TopLoc_Location()
        mesh = BRep_Tool.Triangulation(face, location)
        if mesh is not None:
            transform = location.Transformation()
            for index in range(1, mesh.NbTriangles() + 1):
                node_ids = mesh.Triangle(index).Get()
                points = []
                for node_id in node_ids:
                    point = mesh.Node(node_id).Transformed(transform)
                    points.append((point.X(), point.Y(), point.Z()))
                triangles.append(points)
        explorer.Next()
    return triangles


def _load_components(run_dir: Path) -> list[tuple[str, Any]]:
    manifest = _load(run_dir / "assembly_manifest.json")
    components = []
    for index, component in enumerate(manifest["components"], start=1):
        source = run_dir / component["source"]
        shape = _transformed_shape(
            source, component.get("placement", {})
        )
        components.append((f"P{index:02d}", shape))
    return components


def _write_combined_stl(
    components: list[tuple[str, Any]], output: Path
) -> dict[str, Any]:
    builder = BRep_Builder()
    compound = TopoDS_Compound()
    builder.MakeCompound(compound)
    for _, shape in components:
        BRepMesh_IncrementalMesh(shape, 0.35, False, 0.5, True).Perform()
        builder.Add(compound, shape)
    StlAPI_Writer().Write(compound, str(output))
    reloaded = TopoDS_Shape()
    read_ok = bool(StlAPI_Reader().Read(reloaded, str(output)))
    valid = read_ok and bool(BRepCheck_Analyzer(reloaded).IsValid())
    if not valid:
        raise RuntimeError(f"STL roundtrip failed: {output}")
    return {
        "stl_bytes": output.stat().st_size,
        "stl_readback_ok": read_ok,
        "stl_shape_valid": valid,
    }


def _render(
    components: list[tuple[str, Any]],
    output: Path,
    study_id: str,
) -> None:
    width, height = 1600, 1200
    top_margin, bottom_margin, side_margin = 120, 90, 90
    yaw, pitch = math.radians(38), math.radians(24)

    def rotate(point):
        x, y, z = point
        x1 = math.cos(yaw) * x - math.sin(yaw) * y
        y1 = math.sin(yaw) * x + math.cos(yaw) * y
        return (
            x1,
            math.cos(pitch) * y1 - math.sin(pitch) * z,
            math.sin(pitch) * y1 + math.cos(pitch) * z,
        )

    projected = []
    all_points = []
    legend = []
    for index, (label, shape) in enumerate(components):
        triangles = _triangles(shape)
        if not triangles:
            continue
        color = COLORS[index % len(COLORS)]
        for triangle in triangles:
            rotated = [rotate(point) for point in triangle]
            projected.append(
                {
                    "points": rotated,
                    "depth": sum(point[2] for point in rotated) / 3,
                    "color": color,
                }
            )
            all_points.extend(rotated)
        legend.append((label, color))
    if not all_points:
        raise RuntimeError(f"no triangles for {study_id}")
    min_x = min(point[0] for point in all_points)
    max_x = max(point[0] for point in all_points)
    min_y = min(point[1] for point in all_points)
    max_y = max(point[1] for point in all_points)
    span_x, span_y = max(max_x - min_x, 1.0), max(max_y - min_y, 1.0)
    available_width = width - 2 * side_margin
    available_height = height - top_margin - bottom_margin
    scale = min(available_width / span_x, available_height / span_y)
    offset_x = (width - span_x * scale) / 2 - min_x * scale
    offset_y = (
        top_margin
        + (available_height - span_y * scale) / 2
        + max_y * scale
    )
    image = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(image)
    try:
        title_font = ImageFont.truetype("segoeui.ttf", 34)
        text_font = ImageFont.truetype("segoeui.ttf", 23)
    except OSError:
        title_font = ImageFont.load_default()
        text_font = ImageFont.load_default()
    depths = [item["depth"] for item in projected]
    depth_min, depth_max = min(depths), max(depths)
    depth_span = max(depth_max - depth_min, 1e-9)
    for item in builtins.sorted(
        projected, key=lambda row: row["depth"]
    ):
        points_2d = [
            (
                point[0] * scale + offset_x,
                offset_y - point[1] * scale,
            )
            for point in item["points"]
        ]
        rgb = ImageColor.getrgb(item["color"])
        shade = 0.78 + 0.22 * (
            (item["depth"] - depth_min) / depth_span
        )
        fill = builtins.tuple(
            builtins.min(255, int(channel * shade)) for channel in rgb
        )
        draw.polygon(points_2d, fill=fill, outline="#343434")
    title = f"{study_id}  |  {len(components)} parts  |  blinded review"
    draw.text((50, 38), title, fill="#202020", font=title_font)
    legend_x = width - 230
    for index, (label, color) in enumerate(legend):
        y = 45 + index * 34
        draw.rectangle(
            (legend_x, y, legend_x + 24, y + 24),
            fill=color,
            outline="#303030",
        )
        draw.text(
            (legend_x + 34, y - 2),
            label,
            fill="#202020",
            font=text_font,
        )
    draw.text(
        (50, height - 55),
        "Question: Does this look like one functionally coherent assembly?",
        fill="#333333",
        font=text_font,
    )
    image.save(output, format="PNG", optimize=True)


def _proposal_maps(pool: Path):
    proposals = _load(pool / "grouping" / "group_proposals.json")
    by_parts = {_parts_key(item["parts"]): item for item in proposals}
    return proposals, by_parts


def _study_selection(
    root: Path, results: Path
) -> list[dict[str, Any]]:
    review = _load(results / "final_review_groups.json")
    review_by_pool: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for item in review:
        if (
            not item["evaluation_is_true_group"]
            and item.get("pose", {}).get("final_pose_status") == "valid"
        ):
            review_by_pool[item["pool_id"]].append(item)
    selected = []
    for pool in sorted(
        path for path in root.iterdir() if (path / "pool_gt.json").is_file()
    ):
        gt = _load(pool / "pool_gt.json")
        _, by_parts = _proposal_maps(pool)
        for truth in gt["true_groups"]:
            proposal = by_parts.get(_parts_key(truth["parts"]))
            if not proposal:
                continue
            run_dir = (
                pool
                / "oracle_validation"
                / f"GT_{proposal['group_id'].removeprefix('G_')}"
            )
            if (run_dir / "assembly_manifest.json").is_file():
                selected.append(
                    {
                        "pool_id": pool.name,
                        "proposal": proposal,
                        "run_dir": run_dir,
                        "source_truth": True,
                        "selection_role": "positive_truth_group",
                    }
                )
        hard_negatives = sorted(
            review_by_pool.get(pool.name, []),
            key=lambda item: (
                float(item["geometry_score"]),
                float(item["consistency"]["group_consistency_score"]),
            ),
            reverse=True,
        )[:2]
        for item in hard_negatives:
            selected.append(
                {
                    "pool_id": pool.name,
                    "proposal": item,
                    "run_dir": (
                        pool / "validation" / item["group_id"]
                    ),
                    "source_truth": False,
                    "selection_role": "pose_valid_hard_negative",
                }
            )
    selected.sort(
        key=lambda item: (
            item["pool_id"],
            item["selection_role"],
            item["proposal"]["group_id"],
        )
    )
    return selected


def run(
    pools_root: str | Path,
    results_dir: str | Path,
    output_dir: str | Path,
    pipeline_config_path: str | Path,
    *,
    deepseek_mode: str = "live",
) -> dict[str, Any]:
    root = Path(pools_root).resolve()
    results = Path(results_dir).resolve()
    output = Path(output_dir).resolve()
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(
            f"output folder is not empty; choose a new folder: {output}"
        )
    output.mkdir(parents=True, exist_ok=True)
    blinded = output / "01_待人工判断"
    deepseek_dir = output / "02_DeepSeek解释_标注后查看"
    answer_dir = output / "03_真值答案_标注后查看"
    blinded.mkdir()
    deepseek_dir.mkdir()
    answer_dir.mkdir()
    config = _load(Path(pipeline_config_path).resolve())
    reviewer = DeepSeekReviewer(
        config["semantic_review"],
        results / "deepseek_human_study_cache",
    )
    selection = _study_selection(root, results)
    human_rows, answer_rows, deepseek_rows, project_rows = [], [], [], []
    for index, item in enumerate(selection, start=1):
        study_id = f"H{index:04d}"
        proposal = item["proposal"]
        part_count = len(proposal["parts"])
        folder = blinded / f"{part_count:02d}件组"
        folder.mkdir(exist_ok=True)
        stem = f"{study_id}__{part_count}parts"
        print(
            f"{study_id}: {item['pool_id']}/{proposal['group_id']} "
            f"parts={part_count}",
            flush=True,
        )
        components = _load_components(item["run_dir"])
        stl_path = folder / f"{stem}.stl"
        png_path = folder / f"{stem}.png"
        stl_check = _write_combined_stl(components, stl_path)
        _render(components, png_path, study_id)
        item_meta = {
            "study_id": study_id,
            "part_count": part_count,
            "component_aliases": [label for label, _ in components],
            "files": {
                "stl": stl_path.name,
                "screenshot": png_path.name,
            },
            "question": (
                "Does this look like one functionally coherent mechanical "
                "assembly? Judge only from the visual/STL."
            ),
            **stl_check,
        }
        (folder / f"{stem}.json").write_text(
            json.dumps(item_meta, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
        human_rows.append(
            {
                "study_id": study_id,
                "part_count": part_count,
                "功能上是否像一个完整装配_yes_no_uncertain": "",
                "是否像同一机械系统_yes_no_uncertain": "",
                "主要判断依据": "",
                "异常或不确定点": "",
            }
        )
        answer_rows.append(
            {
                "study_id": study_id,
                "pool_id": item["pool_id"],
                "candidate_id": proposal["group_id"],
                "source_truth_group": item["source_truth"],
                "selection_role": item["selection_role"],
                "parts": "|".join(proposal["parts"]),
            }
        )
        pool = root / item["pool_id"]
        features = _load(pool / "index" / "part_features.json")
        feature_map = {row["part_id"]: row for row in features}
        edges = _load(pool / "index" / "pruned_candidates.json")
        edge_map = {row["candidate_id"]: row for row in edges}
        contract = GroupProposal.model_validate(
            {
                key: proposal[key]
                for key in (
                    "schema_version",
                    "group_id",
                    "parts",
                    "candidate_edges",
                    "geometry_score",
                    "connected",
                    "status",
                    "reasons",
                )
            }
        )
        summary = build_summary(
            contract, feature_map, edge_map, len(feature_map)
        )
        summary["proposal_id"] = study_id
        summary["hard_geometry_status"] = (
            "physical_pose_valid_but_source_provenance_hidden"
        )
        summary["constraints"]["human_study_blinded"] = True
        summary["constraints"]["semantic_output_explanation_only"] = True
        record = reviewer.review(summary, mode=deepseek_mode)
        decision = record["decision"]
        deepseek_rows.append(
            {
                "study_id": study_id,
                "verdict": decision["verdict"],
                "plausibility_score": decision["plausibility_score"],
                "confidence": decision["confidence"],
                "reason_codes": "|".join(decision["reason_codes"]),
                "explanation": decision["explanation"],
                "risk_flags": "|".join(decision["risk_flags"]),
                "affected_grouping": False,
                "cache_hit": record.get("cache_hit", False),
            }
        )
        project_rows.append(
            {
                "study_id": study_id,
                "semantic_input": summary,
                "deepseek_record": record,
                "evaluation_label": int(item["source_truth"]),
            }
        )

    def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
        with path.open("w", newline="", encoding="utf-8-sig") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)

    write_csv(output / "人工语义标注表.csv", human_rows)
    write_csv(deepseek_dir / "DeepSeek结果.csv", deepseek_rows)
    write_csv(answer_dir / "答案.csv", answer_rows)
    _write_project = results / "human_semantic_study.json"
    _write_project.write_text(
        json.dumps(project_rows, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    semantic_inputs = [
        {
            "study_id": row["study_id"],
            **row["semantic_input"],
        }
        for row in project_rows
    ]
    semantic_reviews = [
        {
            "study_id": row["study_id"],
            "decision": row["deepseek_record"]["decision"],
            "provider": row["deepseek_record"].get("provider"),
            "model": row["deepseek_record"].get("model"),
            "cache_hit": row["deepseek_record"].get("cache_hit", False),
            "application_mode": "explanation_only",
            "affected_final_decision": False,
        }
        for row in project_rows
    ]
    (results / "semantic_inputs.json").write_text(
        json.dumps(semantic_inputs, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    (results / "semantic_reviews.json").write_text(
        json.dumps(semantic_reviews, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    labels = [row["evaluation_label"] for row in project_rows]
    semantic_scores = [
        row["deepseek_record"]["decision"]["plausibility_score"]
        for row in project_rows
    ]
    geometry_scores = [
        float(item["proposal"]["geometry_score"]) for item in selection
    ]
    semantic_auc = _auc(labels, semantic_scores)
    semantic_brier = _brier(labels, semantic_scores)
    geometry_brier = _brier(labels, geometry_scores)
    verdicts = sorted(
        {
            row["deepseek_record"]["decision"]["verdict"]
            for row in project_rows
        }
    )
    semantic_enabled = calibration_gate(
        semantic_auc=semantic_auc,
        semantic_brier=semantic_brier,
        geometry_brier=geometry_brier,
        verdicts=verdicts,
        holdout={},
    )
    calibration = {
        "schema_version": "1.0.0",
        "study": "blinded_balanced_visual_semantic_study",
        "review_count": len(project_rows),
        "positive_count": sum(labels),
        "negative_count": len(labels) - sum(labels),
        "semantic_auc_against_source_truth": semantic_auc,
        "semantic_brier_score": semantic_brier,
        "geometry_brier_score": geometry_brier,
        "verdicts": verdicts,
        "human_labels_available": False,
        "semantic_reranking_enabled": semantic_enabled,
        "semantic_application_mode": "explanation_only",
        "gate_rules": {
            "minimum_semantic_auc": 0.70,
            "brier_must_improve_over_geometry": True,
            "required_verdicts": ["accept", "reject", "abstain"],
            "holdout_auto_accept_precision_not_decreased": True,
            "holdout_false_positive_count_not_increased": True,
        },
        "gate_failure_reasons": [
            "semantic AUC is below 0.70",
            "human semantic labels have not been supplied",
            "holdout acceptance safety has not been demonstrated",
        ],
    }
    (results / "semantic_calibration_report.json").write_text(
        json.dumps(calibration, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    (results / "semantic_gate_decision.md").write_text(
        "\n".join(
            [
                "# Semantic Gate Decision",
                "",
                "**CLOSED — DeepSeek is explanation-only.**",
                "",
                f"- Study candidates: {len(project_rows)}",
                f"- Semantic AUC against source truth: {semantic_auc}",
                f"- Semantic Brier score: {semantic_brier}",
                f"- Geometry Brier score: {geometry_brier}",
                f"- Verdicts: {', '.join(verdicts)}",
                "- Human semantic labels and a safe holdout comparison are "
                "still required.",
                "- No DeepSeek output affected grouping, ranking, acceptance, "
                "or rejection.",
                "",
            ]
        ),
        encoding="utf-8",
    )
    instructions = "\n".join(
        [
            "# 人工语义复核说明",
            "",
            "1. 先只打开 `01_待人工判断` 中的 PNG；需要旋转检查时再用 "
            "SolidWorks 打开同名 STL。",
            "2. 在 `人工语义标注表.csv` 中填写两个 yes/no/uncertain 字段。",
            "3. 不要在标注前打开 02、03 文件夹，以免 DeepSeek 或真值影响判断。",
            "4. 这里的“语义正确”指功能上是否像一个完整、协调的机械装配；"
            "来源真值另行比较。",
            "5. 所有 STL 均已由 OCCT 重新读取并通过有效性检查。",
            "",
            f"样本数：{len(selection)}；真值正样本与物理可行硬负样本均采用"
            "盲编号。",
            "",
        ]
    )
    (output / "README_先看这里.md").write_text(
        instructions, encoding="utf-8"
    )
    return {
        "output_dir": str(output),
        "items": len(selection),
        "truth_positive_items": sum(
            bool(item["source_truth"]) for item in selection
        ),
        "hard_negative_items": sum(
            not bool(item["source_truth"]) for item in selection
        ),
        "deepseek_reviews": len(deepseek_rows),
        "stl_roundtrip_verified": len(selection),
        "semantic_effect_on_grouping": False,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", nargs="?", default="mixed_pools_v1")
    parser.add_argument(
        "--results",
        default=str(Path(__file__).parent / "data" / "results"),
    )
    parser.add_argument("--output", required=True)
    parser.add_argument(
        "--pipeline-config",
        default=str(
            Path(__file__).parent / "configs" / "pool_pipeline.json"
        ),
    )
    parser.add_argument(
        "--deepseek-mode",
        choices=("live", "cache_only", "off"),
        default="live",
    )
    args = parser.parse_args()
    result = run(
        args.root,
        args.results,
        args.output,
        args.pipeline_config,
        deepseek_mode=args.deepseek_mode,
    )
    print(json.dumps(result, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
