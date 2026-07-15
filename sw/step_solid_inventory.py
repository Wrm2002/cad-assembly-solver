"""Read-only per-solid inventory for a STEP model.

The inventory is used to distinguish a component's exterior body, mounting
flange and small insertion tabs before any interface pose is proposed.  It
does not export, heal, fuse or otherwise modify topology.
"""
from __future__ import annotations

import argparse
import json
from hashlib import sha256
from pathlib import Path

from OCC.Core.BRepBndLib import brepbndlib
from OCC.Core.BRepGProp import brepgprop
from OCC.Core.Bnd import Bnd_Box
from OCC.Core.GProp import GProp_GProps
from OCC.Core.IFSelect import IFSelect_RetDone
from OCC.Core.STEPControl import STEPControl_Reader
from OCC.Core.TopAbs import TopAbs_FACE, TopAbs_SOLID
from OCC.Core.TopExp import TopExp_Explorer


def file_hash(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def count_faces(shape) -> int:
    count = 0
    explorer = TopExp_Explorer(shape, TopAbs_FACE)
    while explorer.More():
        count += 1
        explorer.Next()
    return count


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    source = args.source.resolve()
    source_hash = file_hash(source)

    reader = STEPControl_Reader()
    if reader.ReadFile(str(source)) != IFSelect_RetDone:
        raise RuntimeError(f"cannot read STEP: {source}")
    reader.TransferRoots()
    shape = reader.OneShape()

    rows = []
    explorer = TopExp_Explorer(shape, TopAbs_SOLID)
    index = 0
    while explorer.More():
        solid = explorer.Current()
        explorer.Next()
        box = Bnd_Box()
        box.SetGap(0.0)
        brepbndlib.Add(solid, box)
        bounds = [float(value) for value in box.Get()]
        lo, hi = bounds[:3], bounds[3:]
        volume_properties = GProp_GProps()
        surface_properties = GProp_GProps()
        brepgprop.VolumeProperties(solid, volume_properties)
        brepgprop.SurfaceProperties(solid, surface_properties)
        centre = volume_properties.CentreOfMass()
        rows.append(
            {
                "solid_index": index,
                "bbox_min_mm": lo,
                "bbox_max_mm": hi,
                "bbox_extent_mm": [hi[axis] - lo[axis] for axis in range(3)],
                "volume_mm3": max(0.0, float(volume_properties.Mass())),
                "surface_area_mm2": max(0.0, float(surface_properties.Mass())),
                "centroid_mm": [centre.X(), centre.Y(), centre.Z()],
                "face_count": count_faces(solid),
            }
        )
        index += 1

    if file_hash(source) != source_hash:
        raise RuntimeError("source STEP changed during read-only inventory")
    rows.sort(key=lambda row: (-row["volume_mm3"], row["solid_index"]))
    payload = {
        "schema_version": "step_solid_inventory.v1",
        "source": str(source),
        "source_sha256": source_hash,
        "topology_modified": False,
        "solid_count": len(rows),
        "solids": rows,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps({"solid_count": len(rows), "output": str(args.output)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
