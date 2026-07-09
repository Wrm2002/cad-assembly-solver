import json
from pathlib import Path


def write_ground_truth(case_dir, spec, case_id):
    case_dir = Path(case_dir)
    payload = {
        "case_id": case_id,
        "group_size": spec["group_size"],
        "template": spec["template"],
        "dataset_intended_use": spec.get(
            "dataset_intended_use", "geometry_smoke_only"
        ),
        "functional_positive_eligible": bool(
            spec.get("functional_positive_eligible", False)
        ),
        "functional_positive_exclusion_reason": spec.get(
            "functional_positive_exclusion_reason",
            "Legacy primitive dataset is not function-grounded.",
        ),
        "system_class": spec.get("system_class"),
        "parts": [f"part_{index + 1:02d}.step" for index in range(len(spec["parts"]))],
        "true_mates": spec["true_mates"],
        "placements": spec["placements"],
        "parameters": spec["parameters"],
        "hard_negatives": spec["hard_negatives"],
        "part_semantics": spec.get("part_semantics", {}),
        "generator": "SolidWorks API programmatic synthetic CAD",
    }
    path = case_dir / "gt.json"
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    step_dir = case_dir / "step"
    step_dir.mkdir(parents=True, exist_ok=True)
    (step_dir / "gt.json").write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    return path
