"""Run known_group_assembly on cases 1-5 and score direct edges.

This runner is for the short-term target: recover the direct assembly graph.
Pose status is printed as a diagnostic but does not decide pass/fail here.
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

sys.path.insert(0, "sw")
from known_group_assembly import run_known_group_assembly  # noqa: E402


SW_DIR = Path("sw")

CASES = [
    {
        "case_id": "case_1",
        "dir": SW_DIR / "1",
        "beam": 20,
        "brep_graph_dir": SW_DIR / "1" / "_brep_graphs_enriched",
        "expected_edges": [("flange_part_a.step", "flange_part_b.step")],
        "expected_non_edges": [],
        "desc": "2 flanges",
    },
    {
        "case_id": "case_2",
        "dir": SW_DIR / "2",
        "beam": 20,
        "brep_graph_dir": SW_DIR / "2" / "_brep_graphs_enriched",
        "expected_edges": [
            ("shaft_with_keyway.step", "flange_a_pipe_X_fan15deg.step"),
            ("shaft_with_keyway.step", "flange_b_pipe_Y_fan20deg.step"),
            ("shaft_with_keyway.step", "key.step"),
        ],
        "expected_non_edges": [
            ("flange_a_pipe_X_fan15deg.step", "flange_b_pipe_Y_fan20deg.step"),
            ("flange_a_pipe_X_fan15deg.step", "key.step"),
            ("flange_b_pipe_Y_fan20deg.step", "key.step"),
        ],
        "desc": "shaft + 2 flanges + key",
    },
    {
        "case_id": "case_3",
        "dir": SW_DIR / "3",
        "beam": 20,
        "brep_graph_dir": SW_DIR / "3" / "_brep_graphs_enriched",
        "expected_edges": [
            (
                "01_FAN-CAGE-MODULE-R620-NH.stp",
                "01_FAN-MODULE-SUNON-6056(1).stp",
            )
        ],
        "expected_non_edges": [],
        "desc": "fan cage + fan module",
    },
    {
        "case_id": "case_4",
        "dir": SW_DIR / "4_lightweight",
        "beam": 20,
        "brep_graph_dir": None,
        "expected_edges": [
            ("01-62DC24-MLB-PCBA.stp", "5-rd_rc_a_2rx4_1_ddrv.stp"),
            ("01-62DC24-MLB-PCBA.stp", "Hygon-7400-3D.stp"),
        ],
        "expected_non_edges": [
            ("5-rd_rc_a_2rx4_1_ddrv.stp", "Hygon-7400-3D.stp")
        ],
        "desc": "PCBA + memory + CPU",
    },
    {
        "case_id": "case_5",
        "dir": SW_DIR / "5_lightweight",
        "beam": 20,
        "brep_graph_dir": None,
        "expected_edges": [
            (
                "01-ASSY-CHASSIS-MODULE-R6250H0.stp",
                "01-ASSY-CHASSIS-EAR-L-R620.stp",
            ),
            ("01-ASSY-CHASSIS-MODULE-R6250H0.stp", "5-CRPS1300NC.stp"),
        ],
        "expected_non_edges": [
            ("01-ASSY-CHASSIS-EAR-L-R620.stp", "5-CRPS1300NC.stp")
        ],
        "desc": "chassis + ear + PSU",
    },
]


def norm(pair):
    return tuple(sorted(str(value) for value in pair))


def run_one(info: dict) -> dict:
    case_id = info["case_id"]
    case_dir = info["dir"]
    expected = {norm(edge) for edge in info["expected_edges"]}
    expected_non_edges = {norm(edge) for edge in info["expected_non_edges"]}

    print(f'\n{"=" * 60}', flush=True)
    print(f'{case_id}: {info["desc"]}', flush=True)
    print(f'{"=" * 60}', flush=True)

    started = time.time()
    try:
        out = run_known_group_assembly(
            case_dir,
            beam_width=info["beam"],
            brep_graph_dir=info.get("brep_graph_dir"),
        )
    except Exception as exc:
        elapsed = time.time() - started
        print(f"  [CRASH] {elapsed:.0f}s: {exc}", flush=True)
        return {
            "case_id": case_id,
            "passed": False,
            "error": str(exc),
            "runtime_s": round(elapsed, 1),
        }
    elapsed = time.time() - started

    actual = {
        norm(connection["parts"])
        for connection in out.get("direct_connections", [])
        if "parts" in connection
    }

    true_positive_edges = actual & expected
    false_positive_edges = actual - expected
    false_negative_edges = expected - actual
    false_positive_known_non_edges = actual & expected_non_edges

    tp = len(true_positive_edges)
    fp = len(false_positive_edges)
    fn = len(false_negative_edges)
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = (
        2 * precision * recall / (precision + recall)
        if (precision + recall)
        else 0.0
    )

    pose_status = out.get("pose_status", "?")
    collision = out.get("collision_validation", {})
    collision_count = len(collision.get("collisions", []))
    selected_rank = collision.get("selected_pose_rank")
    checked_pose_count = collision.get("checked_pose_count")

    passed = fp == 0 and fn == 0
    icon = "[OK]" if passed else "[FAIL]"
    print(
        f"  {icon} direct-edges | Pose={pose_status} | "
        f"{elapsed:.0f}s | checked_pose={checked_pose_count} "
        f"selected_rank={selected_rank}",
        flush=True,
    )
    print(f"  Exp={sorted(expected)}", flush=True)
    print(f"  Act={sorted(actual)}", flush=True)
    print(
        f"  TP={tp} FP={fp} FN={fn} | "
        f"P={precision:.2f} R={recall:.2f} F1={f1:.2f}",
        flush=True,
    )
    print(f"  Collision diagnostic={collision_count}", flush=True)
    if false_positive_edges:
        print(f"  [FP] {sorted(false_positive_edges)}", flush=True)
    if false_negative_edges:
        print(f"  [FN] {sorted(false_negative_edges)}", flush=True)
    if false_positive_known_non_edges:
        print(
            f"  [FP non-edge] {sorted(false_positive_known_non_edges)}",
            flush=True,
        )

    return {
        "case_id": case_id,
        "passed": passed,
        "pose_status": pose_status,
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "precision": round(precision, 3),
        "recall": round(recall, 3),
        "f1": round(f1, 3),
        "runtime_s": round(elapsed, 1),
        "collision_count": collision_count,
        "checked_pose_count": checked_pose_count,
        "selected_pose_rank": selected_rank,
        "actual": [list(edge) for edge in actual],
        "fp_list": [list(edge) for edge in false_positive_edges],
        "fn_list": [list(edge) for edge in false_negative_edges],
    }


def main() -> int:
    results = [run_one(info) for info in CASES]

    print(f'\n\n{"=" * 60}')
    print("SUMMARY")
    print(f'{"=" * 60}')
    passed_count = sum(1 for row in results if row.get("passed"))
    print(f"  [OK] {passed_count}/{len(results)} direct-edge tests passed")
    for row in results:
        icon = "[OK]" if row.get("passed") else (
            "[CRASH]" if "error" in row else "[FAIL]"
        )
        print(
            f'  {icon} {row["case_id"]}: '
            f'P={row.get("precision", 0):.2f} '
            f'R={row.get("recall", 0):.2f} '
            f'F1={row.get("f1", 0):.2f} '
            f'pose={row.get("pose_status")} '
            f'{row.get("runtime_s", 0):.0f}s'
        )

    (SW_DIR / "exam_results.json").write_text(
        json.dumps(results, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    return 0 if passed_count == len(results) else 1


if __name__ == "__main__":
    raise SystemExit(main())
