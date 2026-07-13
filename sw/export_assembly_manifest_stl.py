"""Export every transformed component in an assembly manifest as one STL."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any


def _resolved_component_source(
    manifest_path: str | Path, component: dict[str, Any]
) -> Path:
    manifest = Path(manifest_path).resolve()
    return (manifest.parent / str(component["source"])).resolve()


def _safe_name(value: str) -> str:
    clean = re.sub(r"[^A-Za-z0-9_.-]+", "_", value).strip("_.")
    return clean or "component"


def export_manifest(
    manifest_path: str | Path,
    output_dir: str | Path,
    *,
    linear_deflection: float = 0.5,
    angular_deflection: float = 0.5,
) -> list[str]:
    manifest_path = Path(manifest_path).resolve()
    output_dir = Path(output_dir).resolve()
    data = json.loads(manifest_path.read_text(encoding="utf-8"))

    from OCC.Core.BRepBuilderAPI import BRepBuilderAPI_Transform
    from OCC.Core.BRepMesh import BRepMesh_IncrementalMesh
    from OCC.Core.StlAPI import StlAPI_Writer

    from build_assembly import build_transform, load_step

    output_dir.mkdir(parents=True, exist_ok=True)
    outputs = []
    for index, component in enumerate(data.get("components", [])):
        source = _resolved_component_source(manifest_path, component)
        if not source.is_file():
            raise FileNotFoundError(f"component STEP not found: {source}")
        shape = load_step(str(source))
        transform = build_transform(component.get("placement", {}))
        transformed = BRepBuilderAPI_Transform(shape, transform, True).Shape()
        BRepMesh_IncrementalMesh(
            transformed,
            float(linear_deflection),
            False,
            float(angular_deflection),
            True,
        ).Perform()
        label = _safe_name(
            str(component.get("label") or component.get("id") or index)
        )
        output = output_dir / f"part_{index:02d}_{label}.stl"
        writer = StlAPI_Writer()
        writer.SetASCIIMode(False)
        if not writer.Write(transformed, str(output)):
            raise RuntimeError(f"failed to write STL: {output}")
        outputs.append(str(output))
    return outputs


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("manifest", type=Path)
    parser.add_argument("output_dir", type=Path)
    parser.add_argument("--linear-deflection", type=float, default=0.5)
    parser.add_argument("--angular-deflection", type=float, default=0.5)
    args = parser.parse_args()
    outputs = export_manifest(
        args.manifest,
        args.output_dir,
        linear_deflection=args.linear_deflection,
        angular_deflection=args.angular_deflection,
    )
    print(
        json.dumps(
            {"output_dir": str(args.output_dir.resolve()), "files": outputs},
            ensure_ascii=False,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
