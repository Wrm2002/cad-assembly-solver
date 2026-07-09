"""Render candidate group parts as a PNG tray for multimodal vision review.

Each part is loaded from its STEP file and rendered from a standard isometric
view. Parts are arranged in a labelled grid suitable for Qwen-VL / GPT-4V input.
"""

from __future__ import annotations

import builtins
import math
from pathlib import Path
from typing import Any

from PIL import Image, ImageColor, ImageDraw, ImageFont
from OCC.Core.BRep import BRep_Tool
from OCC.Core.BRepMesh import BRepMesh_IncrementalMesh
from OCC.Core.TopAbs import TopAbs_FACE
from OCC.Core.TopExp import TopExp_Explorer
from OCC.Core.TopLoc import TopLoc_Location
from OCC.Core.TopoDS import topods

from build_assembly import load_step


COLORS = [
    "#4C78A8", "#F58518", "#54A24B", "#E45756",
    "#72B7B2", "#B279A2", "#EECA3B", "#BAB0AC",
]


def _triangles(shape) -> list[list[tuple[float, float, float]]]:
    BRepMesh_IncrementalMesh(shape, 0.45, False, 0.5, True).Perform()
    explorer = TopExp_Explorer(shape, TopAbs_FACE)
    triangles = []
    while explorer.More():
        face = topods.Face(explorer.Current())
        location = TopLoc_Location()
        mesh = BRep_Tool.Triangulation(face, location)
        if mesh is not None:
            transform = location.Transformation()
            for index in range(1, mesh.NbTriangles() + 1):
                node_ids = mesh.Triangle(index).Get()
                points = []
                for node_id in node_ids:
                    point = mesh.Node(node_id).Transformed(transform)
                    points.append((point.X(), point.Y(), point.Z()))
                triangles.append(points)
        explorer.Next()
    return triangles


def _render_single(
    shape,
    label: str,
    cell_width: int,
    cell_height: int,
    border: int = 4,
) -> Image.Image:
    """Render one part into a fixed-size cell with label."""
    tris = _triangles(shape)
    image = Image.new("RGB", (cell_width, cell_height), "#F5F5F5")
    draw = ImageDraw.Draw(image)
    try:
        font = ImageFont.truetype("segoeui.ttf", 18)
    except OSError:
        font = ImageFont.load_default()

    if not tris:
        draw.text((20, cell_height // 2 - 10), f"[{label}: no faces]", fill="#999", font=font)
        return image

    # Camera: standard isometric-ish view
    yaw = math.radians(38)
    pitch = math.radians(24)

    def rotate(pt):
        x, y, z = pt
        x1 = math.cos(yaw) * x - math.sin(yaw) * y
        y1 = math.sin(yaw) * x + math.cos(yaw) * y
        return (x1, math.cos(pitch) * y1 - math.sin(pitch) * z,
                math.sin(pitch) * y1 + math.cos(pitch) * z)

    all_pts = []
    projected = []
    color = COLORS[0]  # single color per cell
    for tri in tris:
        rotated = [rotate(p) for p in tri]
        projected.append({"points": rotated, "depth": sum(p[2] for p in rotated) / 3, "color": color})
        all_pts.extend(rotated)

    min_x, max_x = min(p[0] for p in all_pts), max(p[0] for p in all_pts)
    min_y, max_y = min(p[1] for p in all_pts), max(p[1] for p in all_pts)
    span_x, span_y = max(max_x - min_x, 1.0), max(max_y - min_y, 1.0)

    margin = border * 2 + 30  # extra top margin for label
    scale = min((cell_width - 2 * border) / span_x, (cell_height - margin - border) / span_y)
    off_x = (cell_width - span_x * scale) / 2 - min_x * scale
    off_y = margin + (cell_height - margin - border - span_y * scale) / 2 + max_y * scale

    depths = [p["depth"] for p in projected]
    d_min, d_max = min(depths), max(depths)
    d_span = max(d_max - d_min, 1e-9)

    for item in builtins.sorted(projected, key=lambda r: r["depth"]):
        pts_2d = [(p[0] * scale + off_x, off_y - p[1] * scale) for p in item["points"]]
        rgb = ImageColor.getrgb(item["color"])
        shade = 0.78 + 0.22 * ((item["depth"] - d_min) / d_span)
        fill = builtins.tuple(min(255, int(c * shade)) for c in rgb)
        draw.polygon(pts_2d, fill=fill, outline="#343434")

    draw.text((border + 4, border + 2), label, fill="#202020", font=font)
    return image


def render_parts_tray(
    step_paths: list[str | Path],
    part_labels: list[str],
    output_path: str | Path,
    *,
    cols: int = 4,
    cell_width: int = 400,
    cell_height: int = 320,
) -> Path:
    """Render a grid of STEP parts as a single PNG image.

    Args:
        step_paths: List of STEP file paths (one per part).
        part_labels: Short labels for each part (e.g., ["P01 shaft", "P02 hub"]).
        output_path: Where to save the PNG.
        cols: Number of columns in the grid.
        cell_width, cell_height: Size of each cell in pixels.

    Returns:
        Path to the rendered PNG.
    """
    n = len(step_paths)
    if n == 0:
        raise ValueError("no parts to render")
    rows = math.ceil(n / cols)

    # Render each part
    cells = []
    for i, (path, label) in enumerate(zip(step_paths, part_labels)):
        try:
            shape = load_step(str(path))
            cell_img = _render_single(shape, label, cell_width, cell_height)
        except Exception as exc:
            # Fallback: blank cell with error
            cell_img = Image.new("RGB", (cell_width, cell_height), "#FFF0F0")
            try:
                font = ImageFont.truetype("segoeui.ttf", 14)
            except OSError:
                font = ImageFont.load_default()
            draw = ImageDraw.Draw(cell_img)
            draw.text((10, cell_height // 2 - 10), f"[{label}]\nerror: {exc}", fill="#C00", font=font)
        cells.append(cell_img)

    # Fill remaining cells in last row with blanks
    remainder = rows * cols - n
    for _ in range(remainder):
        cells.append(Image.new("RGB", (cell_width, cell_height), "#F5F5F5"))

    # Compose grid
    total_width = cols * cell_width
    total_height = rows * cell_height + 60  # extra space for title
    canvas = Image.new("RGB", (total_width, total_height), "white")
    draw = ImageDraw.Draw(canvas)
    try:
        title_font = ImageFont.truetype("segoeui.ttf", 26)
    except OSError:
        title_font = ImageFont.load_default()

    title = f"Candidate Assembly Group  |  {n} parts  |  Parts Tray View"
    draw.text((20, 16), title, fill="#202020", font=title_font)

    for i, cell in enumerate(cells):
        col, row = i % cols, i // cols
        x, y = col * cell_width, row * cell_height + 60
        canvas.paste(cell, (x, y))

    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(output, format="PNG", optimize=True)
    return output


def render_exploded_tray(
    step_paths: list[str | Path],
    part_labels: list[str],
    output_path: str | Path,
    *,
    spacing: int = 60,
) -> Path:
    """Render parts in a horizontally exploded view (like an exploded diagram).

    Each part is rendered at its natural scale with horizontal spacing.
    """
    n = len(step_paths)
    if n == 0:
        raise ValueError("no parts to render")

    cell_w, cell_h = 320, 280
    cells = []
    for path, label in zip(step_paths, part_labels):
        try:
            shape = load_step(str(path))
            cell_img = _render_single(shape, label, cell_w, cell_h)
        except Exception as exc:
            cell_img = Image.new("RGB", (cell_w, cell_h), "#FFF0F0")
            try:
                font = ImageFont.truetype("segoeui.ttf", 14)
            except OSError:
                font = ImageFont.load_default()
            draw = ImageDraw.Draw(cell_img)
            draw.text((10, cell_h // 2 - 10), f"[{label}]\nerror", fill="#C00", font=font)
        cells.append(cell_img)

    total_w = n * cell_w + (n - 1) * spacing + 40
    total_h = cell_h + 80
    canvas = Image.new("RGB", (total_w, total_h), "white")
    draw = ImageDraw.Draw(canvas)
    try:
        title_font = ImageFont.truetype("segoeui.ttf", 26)
    except OSError:
        title_font = ImageFont.load_default()

    draw.text((20, 16), f"Exploded View  |  {n} parts", fill="#202020", font=title_font)
    for i, cell in enumerate(cells):
        x = 20 + i * (cell_w + spacing)
        canvas.paste(cell, (x, 60))

    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(output, format="PNG", optimize=True)
    return output
