"""Isolated PartFeature extraction worker for large real-world STEP files."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from part_index import index_part


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("step_file")
    parser.add_argument("part_id")
    parser.add_argument("output")
    args = parser.parse_args()
    feature = index_part(
        Path(args.step_file).resolve(),
        args.part_id,
    ).model_dump(mode="json")
    output = Path(args.output).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(feature, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "part_id": feature["part_id"],
                "volume": feature["volume"],
                "planar_faces": len(feature["planar_faces"]),
                "cylindrical_faces": len(feature["cylindrical_faces"]),
                "holes": len(feature["holes"]),
                "hole_patterns": len(feature["hole_patterns"]),
            },
            ensure_ascii=False,
        ),
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
