"""CPU-only recovery evaluation for a saved STEP-domain JoinABLe state."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from finetune_step_domain import evaluate, load_split_rows, read_json
from joinable_compat import (
    DEFAULT_CHECKPOINT,
    build_model,
    load_checkpoint,
    write_json,
)


DEFAULT_MANIFEST = Path(
    r"D:\Model_match_public_data\fusion360_joint\domain_adapt_300"
    r"\domain_adaptation_manifest.json"
)


def deltas(before: dict, after: dict) -> dict:
    result = {}
    for label_scope in ("exact", "equivalent"):
        result[label_scope] = {}
        for k in (1, 5, 10, 20):
            field = f"top_{k}_recall"
            result[label_scope][field] = (
                (after[label_scope][field] or 0.0)
                - (before[label_scope][field] or 0.0)
            )
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--base-checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--adapted-state", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    manifest = read_json(args.manifest)
    validation = load_split_rows(manifest, "validation", training=False)
    test = load_split_rows(manifest, "test", training=False)
    checkpoint, official_args = load_checkpoint(args.base_checkpoint)
    model = build_model(checkpoint, official_args).cpu().eval()
    device = torch.device("cpu")
    baseline_validation = evaluate(model, validation, device)
    baseline_test = evaluate(model, test, device)
    adapted = torch.load(
        args.adapted_state, map_location="cpu", weights_only=False
    )
    model.load_state_dict(adapted["state_dict"], strict=True)
    adapted_validation = evaluate(model, validation, device)
    adapted_test = evaluate(model, test, device)
    report = {
        "schema_version": "1.0.0",
        "evaluation_device": "cpu",
        "base_checkpoint": str(args.base_checkpoint),
        "adapted_state": str(args.adapted_state),
        "adapted_epoch": adapted.get("epoch"),
        "baseline_validation": baseline_validation,
        "adapted_validation": adapted_validation,
        "validation_delta": deltas(
            baseline_validation, adapted_validation
        ),
        "baseline_test": baseline_test,
        "adapted_test": adapted_test,
        "test_delta": deltas(baseline_test, adapted_test),
        "failure_reasons": (
            baseline_validation["failures"]
            + adapted_validation["failures"]
            + baseline_test["failures"]
            + adapted_test["failures"]
        ),
        "unavailable_fields": [
            "functional_assembly_validity",
            "mixed_pool_grouping_quality",
        ],
    }
    write_json(args.output, report)
    print(json.dumps(
        {
            "adapted_epoch": report["adapted_epoch"],
            "validation_delta": report["validation_delta"]["exact"],
            "test_delta": report["test_delta"]["exact"],
        },
        indent=2,
    ))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
