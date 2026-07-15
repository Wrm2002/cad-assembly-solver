"""Render an assembly manifest with opaque OCCT shaded geometry.

Unlike the legacy Matplotlib STL renderer, this path never periodically drops
triangles from large vendor models.  Every manifest component instance is
loaded and transformed independently, so repeated parts remain visible and
auditable.
"""

from __future__ import annotations

import argparse
import math
import json
from pathlib import Path
from typing import Any


PALETTE = [
    (0.15, 0.43, 0.85),
    (0.10, 0.72, 0.28),
    (0.94, 0.44, 0.06),
    (0.62, 0.20, 0.72),
    (0.08, 0.64, 0.69),
    (0.86, 0.19, 0.26),
    (0.72, 0.60, 0.10),
    (0.34, 0.34, 0.38),
]


def _component_source(
    manifest_path: Path, component: dict[str, Any]
) -> Path:
    return (manifest_path.parent / str(component["source"])).resolve()


def _shape_bbox(shape: Any) -> dict[str, list[float]]:
    from OCC.Core.Bnd import Bnd_Box
    from OCC.Core.BRepBndLib import brepbndlib

    box = Bnd_Box()
    box.SetGap(0.0)
    brepbndlib.Add(shape, box)
    minimum_x, minimum_y, minimum_z, maximum_x, maximum_y, maximum_z = box.Get()
    return {
        "min": [minimum_x, minimum_y, minimum_z],
        "max": [maximum_x, maximum_y, maximum_z],
    }


def _union_bbox(rows: list[dict[str, list[float]]]) -> dict[str, list[float]]:
    return {
        "min": [min(row["min"][axis] for row in rows) for axis in range(3)],
        "max": [max(row["max"][axis] for row in rows) for axis in range(3)],
    }


def _font(size: int):
    from PIL import ImageFont

    for name in ("arial.ttf", "DejaVuSans.ttf"):
        try:
            return ImageFont.truetype(name, size=size)
        except OSError:
            continue
    return ImageFont.load_default()


def _compose_multiview(
    view_paths: list[tuple[str, Path]],
    output_path: Path,
    *,
    title: str,
    legend: list[dict[str, Any]],
) -> None:
    from PIL import Image, ImageDraw

    loaded = [(name, Image.open(path).convert("RGB")) for name, path in view_paths]
    cell_width = max(image.width for _, image in loaded)
    cell_height = max(image.height for _, image in loaded)
    margin = 24
    title_height = 64
    legend_line_height = 30
    legend_height = margin + legend_line_height * max(1, (len(legend) + 1) // 2)
    view_columns = 2
    view_rows = max(1, math.ceil(len(loaded) / view_columns))
    canvas = Image.new(
        "RGB",
        (
            view_columns * cell_width + (view_columns + 1) * margin,
            title_height
            + view_rows * cell_height
            + (view_rows + 1) * margin
            + legend_height,
        ),
        (255, 255, 255),
    )
    draw = ImageDraw.Draw(canvas)
    title_font = _font(28)
    label_font = _font(20)
    legend_font = _font(17)
    draw.text((margin, 16), title, fill=(20, 25, 34), font=title_font)
    for index, (name, image) in enumerate(loaded):
        column = index % 2
        row = index // 2
        x = margin + column * (cell_width + margin)
        y = title_height + margin + row * (cell_height + margin)
        canvas.paste(image, (x, y))
        draw.rectangle(
            (x + 10, y + 10, x + 155, y + 44),
            fill=(255, 255, 255),
            outline=(105, 115, 130),
            width=1,
        )
        draw.text(
            (x + 18, y + 15), name,
            fill=(25, 31, 42), font=label_font,
        )

    legend_y = (
        title_height
        + view_rows * cell_height
        + (view_rows + 1) * margin
    )
    column_width = cell_width + margin
    for index, row in enumerate(legend):
        column = index % 2
        line = index // 2
        x = margin + column * column_width
        y = legend_y + line * legend_line_height
        rgb = tuple(round(255.0 * value) for value in row["color_rgb"])
        draw.rectangle((x, y + 3, x + 22, y + 23), fill=rgb, outline=(40, 40, 40))
        draw.text(
            (x + 32, y),
            f"{row['instance_index']}: {row['label']}",
            fill=(30, 35, 43),
            font=legend_font,
        )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(output_path)
    for _, image in loaded:
        image.close()


def render_manifest(
    manifest_path: str | Path,
    output_path: str | Path,
    *,
    view_width: int = 1200,
    view_height: int = 900,
    audit_path: str | Path | None = None,
    expanded_views: bool = False,
    complete_views: bool = False,
    relationship_focus: bool = False,
    relationship_view: bool = False,
    context_transparency: float = 0.72,
) -> dict[str, Any]:
    from OCC.Core.BRepBuilderAPI import BRepBuilderAPI_Transform
    from OCC.Core.Quantity import Quantity_Color, Quantity_TOC_RGB
    from OCC.Display.OCCViewer import Viewer3d

    from build_assembly import build_transform, load_step

    manifest_path = Path(manifest_path).resolve()
    output_path = Path(output_path).resolve()
    audit_path = (
        Path(audit_path).resolve()
        if audit_path
        else output_path.with_suffix(".render_audit.json")
    )
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    components = list(manifest.get("components") or [])
    if not components:
        raise ValueError("assembly manifest contains no components")

    if not 0.0 <= context_transparency < 1.0:
        raise ValueError("context_transparency must be in [0, 1)")

    transformed_shapes = []
    component_audit = []
    for index, component in enumerate(components):
        source = _component_source(manifest_path, component)
        if not source.is_file():
            raise FileNotFoundError(f"component STEP not found: {source}")
        shape = load_step(str(source))
        transform = build_transform(component.get("placement", {}))
        transformed = BRepBuilderAPI_Transform(shape, transform, True).Shape()
        label = str(component.get("label") or component.get("id") or source.stem)
        color = PALETTE[index % len(PALETTE)]
        bbox = _shape_bbox(transformed)
        transformed_shapes.append((label, transformed, color, bbox))
        component_audit.append({
            "instance_index": index + 1,
            "component_id": component.get("id"),
            "label": label,
            "source": str(source),
            "placement": component.get("placement", {}),
            "color_rgb": list(color),
            "bbox": bbox,
        })

    context_index = None
    if relationship_view and len(transformed_shapes) > 1:
        def bbox_volume(row: tuple[Any, ...]) -> float:
            bbox = row[3]
            return math.prod(
                max(0.0, bbox["max"][axis] - bbox["min"][axis])
                for axis in range(3)
            )

        context_index = max(
            range(len(transformed_shapes)),
            key=lambda index: bbox_volume(transformed_shapes[index]),
        )

    viewer = Viewer3d()
    viewer.Create(
        create_default_lights=True,
        draw_face_boundaries=True,
        phong_shading=True,
        display_glinfo=False,
    )
    viewer.SetSize(int(view_width), int(view_height))
    viewer.set_bg_gradient_color([250, 250, 250], [225, 230, 236])
    viewer.EnableAntiAliasing()
    viewer.SetModeShaded()
    for index, (_, shape, color_rgb, _) in enumerate(transformed_shapes):
        transparency = (
            context_transparency
            if context_index is not None and index == context_index
            else 0.0
        )
        viewer.DisplayShape(
            shape,
            color=Quantity_Color(*color_rgb, Quantity_TOC_RGB),
            transparency=transparency,
            update=False,
        )
        component_audit[index]["transparency"] = transparency
        component_audit[index]["relationship_context"] = index == context_index

    raw_dir = output_path.parent / f"{output_path.stem}_views"
    raw_dir.mkdir(parents=True, exist_ok=True)
    view_methods = [
        ("isometric", viewer.View_Iso),
        ("front", viewer.View_Front),
        ("right", viewer.View_Right),
        ("top", viewer.View_Top),
    ]
    if expanded_views or complete_views:
        def opposite_isometric() -> None:
            viewer.View_Iso()
            camera = viewer.View.Camera()
            direction = camera.Direction()
            from OCC.Core.gp import gp_Dir
            camera.SetDirection(gp_Dir(
                -direction.X(), -direction.Y(), -direction.Z()
            ))
            viewer.View.SetCamera(camera)

        view_methods = [
            ("isometric", viewer.View_Iso),
            ("opposite isometric", opposite_isometric),
            ("front", viewer.View_Front),
            ("rear", viewer.View_Rear),
            ("right", viewer.View_Right),
            ("top", viewer.View_Top),
        ]
        if complete_views:
            view_methods = [
                ("isometric", viewer.View_Iso),
                ("opposite isometric", opposite_isometric),
                ("front", viewer.View_Front),
                ("rear", viewer.View_Rear),
                ("left", viewer.View_Left),
                ("right", viewer.View_Right),
                ("top", viewer.View_Top),
                ("bottom", viewer.View_Bottom),
            ]
    if relationship_focus:
        from OCC.Core.V3d import V3d_XposYposZpos

        view_methods = [
            (
                "component-side isometric",
                lambda: viewer.View.SetProj(V3d_XposYposZpos),
            ),
            ("component-side front", viewer.View_Rear),
        ]
    view_paths = []
    for name, choose_view in view_methods:
        choose_view()
        viewer.FitAll()
        viewer.View.Redraw()
        path = raw_dir / f"{name}.png"
        viewer.ExportToImage(str(path))
        if not path.is_file():
            raise RuntimeError(f"OCCT did not create view image: {path}")
        view_paths.append((name, path))

    title = str(manifest.get("assembly_name") or manifest_path.stem)
    if relationship_view:
        title += " - relationship diagnostic (largest carrier translucent)"
    _compose_multiview(
        view_paths,
        output_path,
        title=title,
        legend=component_audit,
    )
    audit = {
        "schema_version": "occt_manifest_render.v1",
        "manifest": str(manifest_path),
        "output": str(output_path),
        "renderer": "pythonocc Viewer3d shaded",
        "triangle_subsampling": False,
        "relationship_view": relationship_view,
        "context_transparency": (
            context_transparency if context_index is not None else 0.0
        ),
        "component_instance_count": len(component_audit),
        "components": component_audit,
        "assembly_bbox": _union_bbox([row["bbox"] for row in component_audit]),
        "views": [
            {"name": name, "path": str(path)} for name, path in view_paths
        ],
    }
    audit_path.parent.mkdir(parents=True, exist_ok=True)
    audit_path.write_text(
        json.dumps(audit, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return audit


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("manifest", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--view-width", type=int, default=1200)
    parser.add_argument("--view-height", type=int, default=900)
    parser.add_argument("--audit")
    parser.add_argument(
        "--expanded-views",
        action="store_true",
        help="render six directions, including the opposite isometric and rear views",
    )
    parser.add_argument(
        "--complete-views",
        action="store_true",
        help="render two isometrics plus all six orthographic directions",
    )
    parser.add_argument(
        "--relationship-view",
        action="store_true",
        help="make the largest bounding-box component translucent",
    )
    parser.add_argument(
        "--relationship-focus",
        action="store_true",
        help="render only a component-side isometric and component-side front view",
    )
    parser.add_argument("--context-transparency", type=float, default=0.72)
    args = parser.parse_args()
    audit = render_manifest(
        args.manifest,
        args.output,
        view_width=args.view_width,
        view_height=args.view_height,
        audit_path=args.audit,
        expanded_views=args.expanded_views,
        complete_views=args.complete_views,
        relationship_focus=args.relationship_focus,
        relationship_view=args.relationship_view,
        context_transparency=args.context_transparency,
    )
    print(json.dumps({
        "output": audit["output"],
        "component_instance_count": audit["component_instance_count"],
        "audit": str(
            Path(args.audit).resolve()
            if args.audit else Path(audit["output"]).with_suffix(
                ".render_audit.json"
            )
        ),
    }, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
