"""Summarise frozen pose ablations with the conservative contact gate."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "sw") not in sys.path:
    sys.path.insert(0, str(ROOT / "sw"))

from learned_joint.pose_acceptance import contact_supported_exact_pose  # noqa: E402


def _run(value: str) -> tuple[str, Path]:
    if "=" not in value:
        raise argparse.ArgumentTypeError("--run requires LABEL=EXACT_JSON")
    label, path = value.split("=", 1)
    result = Path(path)
    if not label or not result.is_file():
        raise argparse.ArgumentTypeError("--run requires an existing exact JSON")
    return label, result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    parser.add_argument("--run", action="append", type=_run, required=True)
    args = parser.parse_args()
    rows: list[dict[str, Any]] = []
    for label, path in args.run:
        payload = json.loads(path.read_text(encoding="utf-8"))
        hypotheses = payload.get("hypotheses") or []
        exact = [str((row.get("exact_validation") or {}).get("status", "not_checked")) for row in hypotheses]
        gates = [contact_supported_exact_pose(row) for row in hypotheses]
        rows.append({
            "label": label,
            "source": str(path.resolve()),
            "hypothesis_count": len(hypotheses),
            "exact_status_counts": {value: exact.count(value) for value in sorted(set(exact))},
            "contact_gate_status_counts": {
                value: sum(gate["status"] == value for gate in gates)
                for value in sorted({gate["status"] for gate in gates})
            },
            "contact_supported_valid_count": sum(gate["status"] == "valid" for gate in gates),
            "separated_occt_valid_count": sum(
                gate["reason"] == "occt_valid_but_selected_edges_are_separated" for gate in gates
            ),
            "gate_audit": gates,
        })
    report = {
        "schema_version": "frozen_pose_ablation_audit.v1",
        "acceptance_rule": "OCCT non-collision plus contact support on every selected constraint edge",
        "runs": rows,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
