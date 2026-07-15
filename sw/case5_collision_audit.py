"""Run bounded OCCT solid collision validation for a case-5 manifest."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from placement_validation import exact_shape_collisions_solid_broadphase


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("manifest", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--max-pairs", type=int, default=96)
    args = parser.parse_args()
    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    audit = exact_shape_collisions_solid_broadphase(
        args.manifest.parent,
        manifest.get("components", []),
        maximum_solid_pair_checks=args.max_pairs,
    )
    audit["candidate_status_rule"] = (
        "Only complete no_collision_detected may be called collision_free; "
        "partial/uncertain validation remains review_required."
    )
    args.output.write_text(json.dumps(audit, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps({
        "status": audit.get("status"),
        "collision_result": audit.get("collision_result"),
        "checked_solid_pair_count": audit.get("checked_solid_pair_count"),
    }, indent=2))


if __name__ == "__main__":
    main()
