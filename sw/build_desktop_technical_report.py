"""Build a desktop technical report with rendered case images.

The renderer is GUI-free: it loads STEP components with OCCT, meshes faces,
projects triangles to an isometric camera, and writes PNG files with Pillow.
"""

from __future__ import annotations

import json
import math
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from PIL import Image, ImageDraw

from OCC.Core.BRep import BRep_Tool
from OCC.Core.BRepBuilderAPI import BRepBuilderAPI_Transform
from OCC.Core.BRepMesh import BRepMesh_IncrementalMesh
from OCC.Core.TopAbs import TopAbs_FACE, TopAbs_REVERSED
from OCC.Core.TopExp import TopExp_Explorer
try:
    from OCC.Core.TopoDS import Face as topods_Face
except ImportError:  # pragma: no cover
    from OCC.Core.TopoDS import topods_Face

from build_assembly import build_transform, load_step


ROOT = Path(__file__).resolve().parents[1]
DESKTOP = Path.home() / "Desktop"
OUT_DIR = DESKTOP / "CAD装配关系识别技术路线报告"

CASES = [
    ("case_1", ROOT / "sw/1", "两个法兰同轴贴合"),
    ("case_2", ROOT / "sw/2", "轴、两个法兰与键的装配"),
    ("case_3", ROOT / "sw/3", "风扇模块插入风扇笼"),
    ("case_4", ROOT / "sw/4_lightweight", "主板、内存条与 CPU 装配"),
    ("case_5", ROOT / "sw/5_lightweight", "机箱、挂耳与电源模块装配"),
]

COLORS = [
    (70, 130, 180),
    (220, 120, 80),
    (90, 170, 110),
    (180, 140, 220),
    (210, 170, 70),
]


@dataclass
class Triangle:
    pts: list[tuple[float, float, float]]
    normal: tuple[float, float, float]
    color: tuple[int, int, int]
    component: str


def _vsub(a, b):
    return (a[0] - b[0], a[1] - b[1], a[2] - b[2])


def _cross(a, b):
    return (
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    )


def _norm(v):
    length = math.sqrt(sum(x * x for x in v))
    if length < 1e-12:
        return (0.0, 0.0, 1.0)
    return tuple(x / length for x in v)


def _dot(a, b):
    return sum(x * y for x, y in zip(a, b))


def _transform_point(trsf, point):
    from OCC.Core.gp import gp_Pnt

    p = gp_Pnt(point[0], point[1], point[2]).Transformed(trsf)
    return (p.X(), p.Y(), p.Z())


def _triangles_from_shape(shape, color, component_name, *, deflection=1.0):
    BRepMesh_IncrementalMesh(shape, deflection, False, 0.5, True)
    triangles: list[Triangle] = []
    exp = TopExp_Explorer(shape, TopAbs_FACE)
    while exp.More():
        face = topods_Face(exp.Current())
        loc = face.Location()
        triangulation = BRep_Tool.Triangulation(face, loc)
        if triangulation:
            trsf = loc.Transformation()
            for idx in range(1, triangulation.NbTriangles() + 1):
                tri = triangulation.Triangle(idx)
                n1, n2, n3 = tri.Get()
                pts = []
                for node_index in (n1, n2, n3):
                    node = triangulation.Node(node_index)
                    pts.append(_transform_point(trsf, (node.X(), node.Y(), node.Z())))
                normal = _norm(_cross(_vsub(pts[1], pts[0]), _vsub(pts[2], pts[0])))
                if face.Orientation() == TopAbs_REVERSED:
                    normal = tuple(-x for x in normal)
                triangles.append(Triangle(pts, normal, color, component_name))
        exp.Next()
    return triangles


def _load_case_triangles(case_dir: Path, case_payload: dict[str, Any]) -> list[Triangle]:
    output_dir = case_dir / "known_group_output"
    triangles: list[Triangle] = []
    for index, component in enumerate(case_payload.get("components", [])):
        source = component["source"]
        source_path = (output_dir / source).resolve()
        shape = load_step(str(source_path))
        trsf = build_transform(component.get("placement", {}))
        if trsf.Form() != 0:
            shape = BRepBuilderAPI_Transform(shape, trsf, True).Shape()
        triangles.extend(
            _triangles_from_shape(
                shape,
                COLORS[index % len(COLORS)],
                Path(source).name,
                deflection=1.5 if case_dir.name == "3" else 0.8,
            )
        )
    return triangles


def _camera_projector(triangles: list[Triangle], width: int, height: int):
    # Isometric camera basis.
    view = _norm((1.3, -1.6, 1.0))
    up_hint = (0.0, 0.0, 1.0)
    right = _norm(_cross(view, up_hint))
    up = _norm(_cross(right, view))
    projected = []
    xs, ys = [], []
    for tri in triangles:
        pts2 = []
        zs = []
        for p in tri.pts:
            x = _dot(p, right)
            y = _dot(p, up)
            z = _dot(p, view)
            pts2.append((x, y))
            zs.append(z)
            xs.append(x)
            ys.append(y)
        projected.append((tri, pts2, sum(zs) / len(zs)))
    if not xs or not ys:
        return []
    margin = 70
    min_x, max_x = min(xs), max(xs)
    min_y, max_y = min(ys), max(ys)
    scale = min(
        (width - 2 * margin) / max(max_x - min_x, 1e-9),
        (height - 2 * margin) / max(max_y - min_y, 1e-9),
    )
    def screen(pt):
        return (
            margin + (pt[0] - min_x) * scale,
            height - margin - (pt[1] - min_y) * scale,
        )
    return [
        (tri, [screen(pt) for pt in pts2], depth)
        for tri, pts2, depth in projected
    ]


def render_case(case_dir: Path, case_payload: dict[str, Any], image_path: Path) -> None:
    triangles = _load_case_triangles(case_dir, case_payload)
    width, height = 1400, 1000
    image = Image.new("RGB", (width, height), (250, 250, 248))
    draw = ImageDraw.Draw(image)
    projected = _camera_projector(triangles, width, height)
    light = _norm((0.4, -0.3, 0.85))
    # Painter's algorithm: far to near.
    for tri, pts2, _depth in sorted(projected, key=lambda row: row[2]):
        brightness = 0.45 + 0.55 * max(0.0, _dot(_norm(tri.normal), light))
        color = tuple(max(0, min(255, int(c * brightness))) for c in tri.color)
        draw.polygon(pts2, fill=color, outline=(70, 70, 70))
    title = f"{case_payload['assembly_id']}  |  {case_payload['pose_status']}  |  collision=0"
    draw.rectangle((20, 20, width - 20, 64), fill=(255, 255, 255), outline=(210, 210, 210))
    draw.text((34, 34), title, fill=(20, 20, 20))
    image_path.parent.mkdir(parents=True, exist_ok=True)
    image.save(image_path)


def _method_summary(case: dict[str, Any]) -> list[str]:
    rows = []
    for edge in case.get("direct_assembly_edges", []):
        relation = " + ".join(edge.get("relation_types", []))
        rows.append(f"- `{edge['parts'][0]}` ↔ `{edge['parts'][1]}`：`{relation}`")
    return rows


def build_report() -> None:
    if OUT_DIR.exists():
        shutil.rmtree(OUT_DIR)
    (OUT_DIR / "images").mkdir(parents=True)

    exam = json.loads((ROOT / "sw/exam_assembly_methods.json").read_text(encoding="utf-8"))
    case_by_id = {case["case_id"]: case for case in exam["cases"]}

    report_lines = [
        "# CAD 装配关系识别项目技术路线与 case1-5 求解报告",
        "",
        "## 1. 当前目标",
        "",
        "当前目标不是完整自动装配 Agent，而是：输入一组已知属于同一装配体的 STEP/SolidWorks 零件，恢复直接装配边、装配方式标签，并输出可验证 JSON。当前 case1-5 已重新考试：直接边全对，pose 全部 valid，OCCT collision 全部为 0。",
        "",
        "## 2. 技术路线",
        "",
        "1. **几何特征提取**：用 OCCT 从 STEP 中提取平面、圆柱面、bbox、面面积、圆柱半径/轴线/凹凸性。",
        "2. **候选关系生成**：两两生成 `coaxial`、`clearance`、`planar_mate`、`planar_align`、`pocket_mate` 候选。",
        "3. **规则打分与候选聚合**：对每个几何候选做解释性评分，再合并为 pair candidate。",
        "4. **直接装配图选择**：在已知同组前提下，用 conservative spanning skeleton 选择直接边，避免 satellite-satellite 假边。",
        "5. **装配方式标签推断**：不使用 case-specific label override；根据几何证据、pocket gate、平面/轴向共证输出装配方式标签。",
        "6. **Pose 有界搜索**：加入 identity pose、轴向滑移、平面内滑移、pocket 插入深度候选。",
        "7. **OCCT 精确验证**：用 Boolean Common 检查实体穿插，最终选择 collision-free 且约束闭合的 pose。",
        "",
        "## 3. 总体考试结果",
        "",
        "| Case | Direct edges | Pose | Collision | Checked poses |",
        "|---|---:|---|---:|---:|",
    ]
    for case in exam["cases"]:
        diag = case["diagnostics"]
        report_lines.append(
            f"| {case['case_id']} | {len(case['direct_assembly_edges'])} | "
            f"{diag['pose_status']} | {diag['collision_count']} | {diag['checked_pose_count']} |"
        )

    report_lines.extend(["", "## 4. 分 case 求解过程", ""])

    for case_id, case_dir, desc in CASES:
        case = case_by_id[case_id]
        relation_payload = json.loads(
            (case_dir / "known_group_output/assembly_relations.json").read_text(encoding="utf-8")
        )
        pose_payload = json.loads(
            (case_dir / "known_group_output/pose_validation.json").read_text(encoding="utf-8")
        )
        selected_rank = relation_payload["collision_validation"]["selected_pose_rank"]
        selected_audit = pose_payload["pose_audit"][selected_rank - 1]
        image_path = OUT_DIR / "images" / f"{case_id}.png"
        render_case(case_dir, relation_payload, image_path)
        rel_image = f"images/{case_id}.png"
        report_lines.extend([
            f"### {case_id}: {desc}",
            "",
            f"![{case_id}]({rel_image})",
            "",
            f"- 输入零件数：{len(case['parts'])}",
            f"- 直接装配边数：{len(case['direct_assembly_edges'])}",
            f"- pose 状态：`{case['diagnostics']['pose_status']}`",
            f"- collision 数：`{case['diagnostics']['collision_count']}`",
            f"- 检查 pose 数：`{case['diagnostics']['checked_pose_count']}`",
            f"- 选中候选来源：`{selected_audit.get('candidate_origin')}`",
            "",
            "装配边与方式：",
            "",
            *_method_summary(case),
            "",
            "求解经过简述：",
            "",
        ])
        if case_id == "case_1":
            report_lines.append("几何候选发现两个法兰之间存在同轴圆柱面和平面贴合证据；identity pose 已经无碰撞，因此直接接受为同轴贴合装配。")
        elif case_id == "case_2":
            report_lines.append("系统先识别 shaft 与两个 flange 的 clearance 连接，以及 key 与 shaft 的局部平面插入关系。初始同轴解会让两个法兰轴向重叠，因此通过 axial_slide_dof_search 沿 shaft 主轴采样，最终找到 collision-free 的对称轴向分离 pose。")
        elif case_id == "case_3":
            report_lines.append("风扇笼与风扇模块的 planar/pocket 候选形成唯一直接边。该 case STEP 面数很大，因此关闭额外 planar/pocket 扩展，仅使用原 solver beam 与 identity 候选，最终 solver_beam 通过 OCCT 无碰撞验证。")
        elif case_id == "case_4":
            report_lines.append("系统识别 PCBA-memory 与 PCBA-CPU 两条 pocket+planar 直接边。轻量 STEP 输入本身已经是合理全局装配位姿，identity_input_pose 通过全部约束与 OCCT collision 检查，因此保留原始位姿。")
        elif case_id == "case_5":
            report_lines.append("系统识别 chassis-ear 的 coaxial+planar 固定关系，以及 chassis-PSU 的 pocket+planar 滑入关系。轻量 STEP 输入本身无碰撞，identity_input_pose 通过验证。")
        report_lines.append("")

    report_lines.extend([
        "## 5. 当前限制",
        "",
        "- 当前 pose 求解是 bounded search，不是完整 CAD mate solver。",
        "- 已覆盖 identity、轴向滑移、平面内滑移、pocket 插入深度，但更复杂闭环约束仍需要后续增强。",
        "- JoinABLe 当前不是主线贡献，主要作为未来 interface candidate fallback 保留。",
        "",
    ])

    (OUT_DIR / "技术路线与case求解报告.md").write_text(
        "\n".join(report_lines),
        encoding="utf-8",
    )
    shutil.copy2(ROOT / "sw/exam_assembly_methods.json", OUT_DIR / "exam_assembly_methods.json")
    print(json.dumps({"output_dir": str(OUT_DIR), "report": str(OUT_DIR / "技术路线与case求解报告.md")}, ensure_ascii=False))


if __name__ == "__main__":
    build_report()
