from pathlib import Path
from PIL import Image, ImageDraw, ImageFont


OUT = Path(r"C:\Users\11049\Desktop\STEP自动装配_通用求解流程图.png")
W, H = 1800, 2400
BG = "#F7FAFC"
INK = "#17324D"
MUTED = "#486581"
BLUE = "#DDEEFF"
BLUE_BORDER = "#2774AE"
GREEN = "#DDF5E8"
GREEN_BORDER = "#248A55"
AMBER = "#FFF0CC"
AMBER_BORDER = "#B7791F"
RED = "#FFE2E2"
RED_BORDER = "#C53030"
GREY = "#ECF1F5"
GREY_BORDER = "#627D98"


def font(size, bold=False):
    name = "msyhbd.ttc" if bold else "msyh.ttc"
    path = Path(r"C:\Windows\Fonts") / name
    if not path.exists():
        path = Path(r"C:\Windows\Fonts\simhei.ttf")
    return ImageFont.truetype(str(path), size)


F_TITLE = font(50, True)
F_SUB = font(25)
F_BOX = font(30, True)
F_DETAIL = font(24)
F_LABEL = font(22, True)


def centered_multiline(draw, box, lines, main_font=F_BOX, detail_font=F_DETAIL, fill=INK):
    x1, y1, x2, y2 = box
    items = []
    for idx, line in enumerate(lines):
        f = main_font if idx == 0 else detail_font
        bbox = draw.textbbox((0, 0), line, font=f)
        items.append((line, f, bbox[3] - bbox[1]))
    total = sum(h for _, _, h in items) + 12 * (len(items) - 1)
    y = (y1 + y2 - total) / 2
    for line, f, h in items:
        bbox = draw.textbbox((0, 0), line, font=f)
        x = (x1 + x2 - (bbox[2] - bbox[0])) / 2
        draw.text((x, y), line, font=f, fill=fill)
        y += h + 12


def box(draw, rect, lines, fill, outline):
    draw.rounded_rectangle(rect, radius=26, fill=fill, outline=outline, width=5)
    centered_multiline(draw, rect, lines)


def diamond(draw, center, size, lines, fill, outline):
    cx, cy = center
    w, h = size
    pts = [(cx, cy - h // 2), (cx + w // 2, cy), (cx, cy + h // 2), (cx - w // 2, cy)]
    draw.polygon(pts, fill=fill, outline=outline)
    draw.line(pts + [pts[0]], fill=outline, width=5)
    centered_multiline(draw, (cx - w * .32, cy - h * .25, cx + w * .32, cy + h * .25), lines, F_BOX, F_DETAIL)


def arrow(draw, start, end, color=BLUE_BORDER, width=6, label=None, label_at=0.5):
    draw.line([start, end], fill=color, width=width)
    ex, ey = end
    sx, sy = start
    if abs(ex - sx) >= abs(ey - sy):
        pts = [(ex, ey), (ex - 20 if ex > sx else ex + 20, ey - 11), (ex - 20 if ex > sx else ex + 20, ey + 11)]
    else:
        pts = [(ex, ey), (ex - 11, ey - 20 if ey > sy else ey + 20), (ex + 11, ey - 20 if ey > sy else ey + 20)]
    draw.polygon(pts, fill=color)
    if label:
        lx = sx + (ex - sx) * label_at
        ly = sy + (ey - sy) * label_at
        draw.rounded_rectangle((lx - 40, ly - 18, lx + 40, ly + 18), radius=10, fill=BG)
        bbox = draw.textbbox((0, 0), label, font=F_LABEL)
        draw.text((lx - (bbox[2]-bbox[0])/2, ly - (bbox[3]-bbox[1])/2 - 2), label, font=F_LABEL, fill=color)


def main():
    img = Image.new("RGB", (W, H), BG)
    draw = ImageDraw.Draw(img)
    title = "面对一个新 STEP 装配任务：通用求解流程"
    sub = "目标：优先避免错误自动接受；几何可行不等于装配来源正确"
    draw.text((W / 2 - draw.textbbox((0, 0), title, font=F_TITLE)[2] / 2, 54), title, font=F_TITLE, fill=INK)
    draw.text((W / 2 - draw.textbbox((0, 0), sub, font=F_SUB)[2] / 2, 126), sub, font=F_SUB, fill=MUTED)

    left, right = 360, 1440
    flow = [
        (210, ["1. 输入零件池", "原始 STEP；不依赖来源编号、颜色或文件名"], BLUE, BLUE_BORDER),
        (410, ["2. 提取 B-Rep 几何", "平面、圆柱轴、孔、槽、导向边、止挡、局部包络"], BLUE, BLUE_BORDER),
        (610, ["3. 高召回接口候选", "轴-孔、孔阵列、平面贴合、键槽、导轨与止挡"], BLUE, BLUE_BORDER),
        (810, ["4. 生成 Pose 分支", "正反插入、镜像、相位和孔阵列映射"], BLUE, BLUE_BORDER),
        (1010, ["5. 刚体位姿优化 T=(R,t)", "共轴、贴合、孔距、相位、插深、止挡残差"], BLUE, BLUE_BORDER),
    ]
    rects = []
    for y, lines, fill, border in flow:
        r = (left, y, right, y + 125)
        rects.append(r)
        box(draw, r, lines, fill, border)
    for a, b in zip(rects, rects[1:]):
        arrow(draw, ((a[0] + a[2]) // 2, a[3]), ((b[0] + b[2]) // 2, b[1]))

    d1 = (900, 1260)
    diamond(draw, d1, (900, 250), ["6. 几何验证", "明确干涉、残差、可进入性、碰撞检查状态"], AMBER, AMBER_BORDER)
    arrow(draw, (900, rects[-1][3]), (900, 1135))

    reject = (85, 1197, 420, 1325)
    unresolved = (1380, 1197, 1715, 1325)
    box(draw, reject, ["Rejected", "明确碰撞或残差不合格"], RED, RED_BORDER)
    box(draw, unresolved, ["Unresolved", "候选不足或接口不可识别"], GREY, GREY_BORDER)
    arrow(draw, (450, 1260), (420, 1260), RED_BORDER, label="失败", label_at=.5)
    arrow(draw, (1350, 1260), (1380, 1260), GREY_BORDER, label="不足", label_at=.5)

    audit = (360, 1435, 1440, 1560)
    box(draw, audit, ["7. 多证据与组级一致性审计", "独立证据数、关键接口覆盖、弱单接口、与更完整组的冲突"], AMBER, AMBER_BORDER)
    arrow(draw, (900, 1385), (900, 1435), AMBER_BORDER)

    d2 = (900, 1745)
    diamond(draw, d2, (980, 270), ["8. 保守自动接受门控", "Pose valid + 无明确干涉 + ≥2 独立几何证据 + 无冲突"], AMBER, AMBER_BORDER)
    arrow(draw, (900, 1560), (900, 1610))

    accepted = (1020, 1905, 1715, 2035)
    review = (85, 1905, 780, 2035)
    box(draw, review, ["Review", "证据不足、碰撞不确定或语义风险；交给人工复核"], AMBER, AMBER_BORDER)
    box(draw, accepted, ["Accepted", "高置信自动接受；保留全部证据与残差"], GREEN, GREEN_BORDER)
    arrow(draw, (425, 1745), (425, 1905), AMBER_BORDER, label="否", label_at=.42)
    arrow(draw, (1375, 1745), (1375, 1905), GREEN_BORDER, label="是", label_at=.42)

    loop_y = 2140
    draw.rounded_rectangle((180, loop_y, 1620, loop_y + 140), radius=22, fill="#EAF2F8", outline=BLUE_BORDER, width=3)
    centered_multiline(draw, (200, loop_y, 1600, loop_y + 140), ["复核反馈：补充局部导向/止挡/孔阵列证据，或确认不应成组", "回到候选生成重新计算；不把人工结论写死为某个零件的固定坐标"], F_BOX, F_DETAIL)
    arrow(draw, (430, 2035), (430, 2140), AMBER_BORDER)
    # Return loop: bottom → left edge → candidate stage
    draw.line([(180, loop_y + 70), (110, loop_y + 70), (110, 672), (360, 672)], fill=BLUE_BORDER, width=5)
    draw.polygon([(360, 672), (338, 661), (338, 683)], fill=BLUE_BORDER)

    footer = "自动接受只是高置信工程建议；任何不确定性均保留为 review / unresolved，而不是强行完整分组。"
    bbox = draw.textbbox((0, 0), footer, font=F_SUB)
    draw.text(((W - (bbox[2]-bbox[0]))/2, H - 76), footer, font=F_SUB, fill=MUTED)

    img.save(OUT, quality=95)
    print(OUT)


if __name__ == "__main__":
    main()
