"""One-command reproducible conservative D3.5-D7 delivery."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from build_human_semantic_review_pack import run as build_human_pack
from candidate_recall_audit import run_audit
from conservative_pipeline import run as run_conservative


def write_closed_semantic_gate(
    results_dir: str | Path,
    calibration_path: str | Path,
) -> dict[str, object]:
    """Write the mandatory D6 artifacts without inventing semantic evidence.

    The frozen mixed pools do not carry engineering roles, assembly-family
    labels, or functional relations.  In that situation an anonymous geometry
    summary must not be sent to DeepSeek and the semantic gate stays closed.
    """

    results = Path(results_dir).resolve()
    results.mkdir(parents=True, exist_ok=True)
    historical = json.loads(
        Path(calibration_path).resolve().read_text(encoding="utf-8")
    )

    (results / "semantic_inputs.json").write_text(
        "[]\n", encoding="utf-8"
    )
    (results / "semantic_reviews.json").write_text(
        "[]\n", encoding="utf-8"
    )
    report = {
        "schema_version": "1.0.0",
        "semantic_reranking_enabled": False,
        "semantic_application_mode": "explanation_only",
        "structured_functional_fields_available": False,
        "input_count": 0,
        "provider_called": False,
        "historical_semantic_auc": historical.get("semantic_auc"),
        "historical_semantic_brier_score": historical.get(
            "semantic_brier_score"
        ),
        "historical_geometry_brier_score": historical.get(
            "geometry_brier_score"
        ),
        "historical_verdicts": historical.get("verdicts", []),
        "gate_rules": {
            "minimum_semantic_auc": 0.70,
            "brier_must_improve_over_geometry": True,
            "required_verdicts": ["accept", "reject", "abstain"],
            "holdout_auto_accept_precision_not_decreased": True,
            "holdout_false_positive_count_not_increased": True,
        },
        "gate_failure_reasons": [
            "frozen mixed pools lack part_role, interface_type, "
            "assembly_family, and functional_relation fields",
            "historical semantic AUC is below 0.70",
            "historical verdicts do not contain accept/reject/abstain",
            "holdout acceptance safety has not been demonstrated",
        ],
    }
    (results / "semantic_calibration_report.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    (results / "semantic_gate_decision.md").write_text(
        "# Semantic Gate Decision\n\n"
        "**CLOSED — explanation-only; no provider call was made.**\n\n"
        "- Structured semantic inputs: 0\n"
        "- The frozen pools do not contain `part_role`, `interface_type`, "
        "`assembly_family`, or `functional_relation`.\n"
        "- Anonymous geometry summaries are not submitted for semantic "
        "judgment.\n"
        "- DeepSeek cannot change scores, ranking, acceptance, rejection, "
        "or review routing.\n"
        "- Re-open only after the calibration rules in "
        "`semantic_calibration_report.json` pass on a functional holdout.\n",
        encoding="utf-8",
    )
    return report


def main() -> int:
    here = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", default=str(here / "mixed_pools_v1"))
    parser.add_argument(
        "--results", default=str(here / "data" / "results")
    )
    parser.add_argument(
        "--human-pack",
        help="New, empty folder for blinded STL/PNG review artifacts.",
    )
    parser.add_argument(
        "--deepseek-mode",
        choices=("live", "cache_only", "off"),
        default="cache_only",
    )
    args = parser.parse_args()
    pipeline_config = here / "configs" / "pool_pipeline.json"
    conservative_config = (
        here / "configs" / "conservative_pipeline.json"
    )
    semantic_calibration = here / "configs" / "semantic_calibration.json"
    audit = run_audit(args.root, args.results)
    conservative = run_conservative(
        args.root,
        args.results,
        pipeline_config_path=pipeline_config,
        conservative_config_path=conservative_config,
    )
    human = None
    semantic_gate = None
    if args.human_pack:
        human = build_human_pack(
            args.root,
            args.results,
            args.human_pack,
            pipeline_config,
            deepseek_mode=args.deepseek_mode,
        )
    else:
        semantic_gate = write_closed_semantic_gate(
            args.results, semantic_calibration
        )
    print(
        json.dumps(
            {
                "candidate_recall": audit,
                "conservative_metrics": conservative,
                "human_semantic_pack": human,
                "semantic_gate": semantic_gate,
            },
            indent=2,
            ensure_ascii=False,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
