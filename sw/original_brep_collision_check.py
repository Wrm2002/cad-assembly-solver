"""Isolated exact OCCT common-volume audit for an assembly manifest.

Run this as a subprocess: complex vendor STEP files occasionally cause a
native OCCT failure.  A crash or incomplete boolean is reported as uncertain,
never as collision-free.
"""

from __future__ import annotations

import argparse
import itertools
import json
from pathlib import Path


def run(manifest_path: Path, output_path: Path) -> dict:
    from OCC.Core.BRepAlgoAPI import BRepAlgoAPI_Common
    from OCC.Core.BRepGProp import brepgprop
    from OCC.Core.GProp import GProp_GProps
    from OCC.Core.BRepBuilderAPI import BRepBuilderAPI_Transform

    from build_assembly import build_transform, load_step

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    rows = []
    for component in manifest["components"]:
        source = (manifest_path.parent / component["source"]).resolve()
        shape = load_step(str(source))
        moved = BRepBuilderAPI_Transform(shape, build_transform(component.get("placement", {})), True).Shape()
        rows.append((component["id"], moved))
    pairs = []
    for (left_id, left), (right_id, right) in itertools.combinations(rows, 2):
        common = BRepAlgoAPI_Common(left, right)
        common.Build()
        if not common.IsDone():
            pairs.append({"parts": [left_id, right_id], "status": "uncertain", "reason": "occt_common_not_done"})
            continue
        props = GProp_GProps()
        brepgprop.VolumeProperties(common.Shape(), props)
        pairs.append({"parts": [left_id, right_id], "status": "checked", "common_volume_mm3": float(props.Mass())})
    result = {"status": "checked", "pairs": pairs}
    output_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("manifest", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    print(json.dumps(run(args.manifest, args.output), indent=2))


if __name__ == "__main__":
    main()
