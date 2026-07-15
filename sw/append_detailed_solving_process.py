from __future__ import annotations

import hashlib
import zipfile
from pathlib import Path

from lxml import etree


REPORT = Path(r"C:\Users\11049\Desktop\STEP自动装配系统_技术工作报告.docx")
TEMP = REPORT.with_name(REPORT.stem + "_with_process.tmp.docx")


def media_hashes(path: Path) -> dict[str, str]:
    with zipfile.ZipFile(path) as archive:
        return {
            member: hashlib.sha256(archive.read(member)).hexdigest()
            for member in archive.namelist()
            if member.startswith("word/media/")
        }


W = "http://schemas.openxmlformats.org/wordprocessingml/2006/main"
NS = {"w": W}


def w(tag: str):
    return etree.Element(f"{{{W}}}{tag}")


def prop(parent, tag: str, val: str | None = None):
    node = etree.SubElement(parent, f"{{{W}}}{tag}")
    if val is not None:
        node.set(f"{{{W}}}val", val)
    return node


def make_paragraph(text: str, *, style: str | None = None, heading=False, page_break_before=False):
    p = w("p")
    ppr = etree.SubElement(p, f"{{{W}}}pPr")
    if style:
        prop(ppr, "pStyle", style)
    if page_break_before:
        prop(ppr, "pageBreakBefore")
    spacing = prop(ppr, "spacing")
    spacing.set(f"{{{W}}}after", "100" if heading else "80")
    spacing.set(f"{{{W}}}line", "300" if not heading else "276")
    spacing.set(f"{{{W}}}lineRule", "auto")
    r = etree.SubElement(p, f"{{{W}}}r")
    rpr = etree.SubElement(r, f"{{{W}}}rPr")
    fonts = prop(rpr, "rFonts")
    fonts.set(f"{{{W}}}ascii", "Calibri")
    fonts.set(f"{{{W}}}hAnsi", "Calibri")
    fonts.set(f"{{{W}}}eastAsia", "Microsoft YaHei")
    if heading:
        prop(rpr, "b")
        color = prop(rpr, "color")
        color.set(f"{{{W}}}val", "1F4E79")
        size = prop(rpr, "sz")
        size.set(f"{{{W}}}val", "30" if style == "Heading1" else "24")
    else:
        size = prop(rpr, "sz")
        size.set(f"{{{W}}}val", "21")
    text_node = etree.SubElement(r, f"{{{W}}}t")
    text_node.text = text
    return p


def append_process(body):
    content = [
        ("附录 A 详细求解过程与判定边界", "Heading1", True, True),
        (
        "本附录记录当前系统从原始 STEP 零件到候选姿态与最终分层结果的实际求解流程。它强调几何证据与可审计性，不把零件文件名、来源编号、颜色或人工观察结果编码成求解规则。"
        , None, False, False),
        ("A.1 输入、特征提取与候选接口", "Heading2", True, False),
        ("步骤一：读取原始 STEP 的 B-Rep；保留零件的局部坐标系和原始拓扑，不通过布尔融合或拓扑重建来修改原模型。", None, False, False),
        ("步骤二：从每个零件提取可用于装配的稳定几何：圆柱面/圆孔轴线及半径、平面法向与边界、孔中心、槽口方向、局部包围盒、薄壁折弯面、导向边和端部止挡。", None, False, False),
        ("步骤三：生成高召回的候选接口对，而不是先给每个零件指定唯一配对。候选包括孔—孔、轴—孔、平面—平面、键槽—键槽、导轨—导向面及局部止挡。弱单接口候选保留给 review，不直接自动接受。", None, False, False),
        ("对于超大厂家模型，几何代理只用于初筛和初始化：用包围盒、主方向、候选导向面与止挡面求一个低成本初始位姿，最后仍将刚体变换回填到原始 STEP 进行渲染与验证。", None, False, False),
        ("A.2 刚体 Pose 求解", "Heading2", True, False),
        ("每个零件以刚体变换 T=(R,t) 表示，其中 R 是旋转、t 是平移。求解不从网络直接输出唯一姿态，而是由接口假设生成有限个离散朝向，再连续优化平移和剩余角度。", None, False, False),
        ("步骤一：建立载体零件坐标系；针对每个候选接口计算局部坐标框架，例如圆柱轴线给出插入轴，平面法向给出接触方向，槽长方向给出相位参考。", None, False, False),
        ("步骤二：枚举几何固有的离散歧义：同轴的正反插入、平面对贴的正反法向、孔阵列的循环相位以及键槽的可行旋转。每个分支都保留来源与残差，避免只保留一个不可解释的 Top1。", None, False, False),
        ("步骤三：对每个分支最小化多项残差：轴线距离与夹角、接触面间隙、孔中心对应残差、键槽相位误差、插入深度/止挡误差和局部碰撞惩罚。若一个零件带有已锁定的附属件，使用同一刚体变换传播其姿态。", None, False, False),
        ("步骤四：对优化后的候选执行几何检查。OCCT 精确 common-volume 能完成时用于否定明确干涉；若原生布尔超时、崩溃或仅能在代理上完成，状态记为 uncertain/review，而不是标成 collision-free。", None, False, False),
        ("A.3 多证据门控与结果分层", "Heading2", True, False),
        ("Pose 成功只说明存在一个物理可行的几何摆放，不说明这些零件真实来自同一装配。因此系统把分数排序与自动接受解耦：final_score 仅排序，不能单独决定 accepted。", None, False, False),
        ("步骤一：统计证据数量和独立证据数量。独立证据必须来自不同约束族，例如“共轴 + 端面贴合 + 键槽相位”是三类证据；单孔或单平面偶然匹配不是。", None, False, False),
        ("步骤二：检查接口覆盖与组级一致性：是否存在合理的承载零件—从属零件结构；当前小组是否阻断了更完整的候选；是否只是在一个弱接触上过拟合。", None, False, False),
        ("步骤三：采用保守输出：同时满足无明确干涉、pose valid、几何分数阈值、至少两项独立证据、组级一致性阈值、非弱单接口且无全局冲突的候选才可 accepted；否则转入 review、rejected 或 unresolved。语义模型当前仅解释复核风险，不参与门控。", None, False, False),
        ("A.4 case1—case4 的逐案求解", "Heading2", True, False),
        ("case1（法兰/轴）：首先从中心孔和外圆柱面提取同轴假设；再以两端法兰面的接触方向和轴向深度消除平移自由度；最后以孔阵列的对称性做一致性检查。因共轴与平面贴合相互独立，结果可作为基础自动装配的正向样例。", None, False, False),
        ("case2（轴、键、双法兰）：先以轴与两法兰的圆柱接口建立共同主轴；然后使两法兰内侧端面贴合，确定轴向位置；再把键放入轴键槽，并以两件法兰的键槽相位消除同轴旋转歧义。该过程避免了“轴在中心但法兰面未贴合”或“法兰贴合但键错相”的单约束假解。", None, False, False),
        ("case3（风扇模块/载体）：从外壳承载结构和模块局部外形建立候选插入方向；以连接器一侧、局部止挡和外壳包络共同筛选可行姿态；再用无明显干涉与多视图检查确认模块没有在镜像位置或错误端部。它说明流程并不局限于回转体。", None, False, False),
        ("case4（主板、CPU、内存条）：以主板作为载体。CPU 单独通过插座平面、边界方向和中心位置求解；内存条通过插槽长轴、卡槽方向、插入深度与局部接触求解。两个从属件分别锁定后再合并验证，因此修正 CPU 朝向不会破坏内存条已确定的插槽姿态。", None, False, False),
        ("A.5 case5 的完整求解链与当前边界", "Heading2", True, False),
        ("case5 的难点不是单一刚体变换，而是大型机箱的复杂拓扑、PSU 的局部舱位、耳件的折弯导向几何及孔阵列证据在不同候选之间出现冲突。", None, False, False),
        ("步骤一：机箱作为载体，先用其外形、开口、局部边缘和可进入空间筛选 PSU 的安装区域；不使用整体模型的布尔融合来改写机箱。", None, False, False),
        ("步骤二：对 PSU 提取包围盒、安装面、风扇/接口面、边缘台阶以及可见圆孔；对机箱提取局部侧壁、候选孔、导向边、端部止挡和可容纳包络。先以“进入方向 + 边缘/止挡”生成局部舱位候选。", None, False, False),
        ("步骤三：在每个局部舱位内进行孔阵列匹配：比较孔中心的刚体变换、孔距关系、孔径兼容、安装面共面性和残差。当前保留的最佳孔阵列候选为三对孔中心对应，平均残差约 0.246 mm。", None, False, False),
        ("步骤四：耳件单独以折弯导向面、局部槽口和端部止挡建立插入候选；只有当其与机箱的孔/面/边至少构成两类独立证据时，才允许与 PSU 共同进入更大的组级闭合。", None, False, False),
        ("步骤五：由于当前孔阵列候选与部分边缘贴合候选尚未形成同一组满足条件的约束闭合，case5 仍标记为 review。报告中的图展示的是当前最优复核候选，而非硬编码的最终装配真值。下一步应在导向/止挡限定的局部 ROI 内重新联合拟合安装面、孔阵列和耳件，而不是放宽阈值或强行输出 accepted。", None, False, False),
        ("A.6 可复核输出与避免硬编码", "Heading2", True, False),
        ("每个候选应保留：候选接口类型、生成分支、几何特征、约束残差、碰撞检查状态、独立证据数、组级一致性、拒绝/复核原因和最终类别。这样当人工指出“孔没有对齐”或“部件在机箱外侧”时，系统修正的是相应的几何约束与验证规则，而不是把某个 case 的坐标写死到 JSON。", None, False, False),
        ("因此，case1—case4 的成功结果可以作为已验证的求解能力；case5 则用于暴露现有几何证据在复杂钣金装配中的不足，并推动候选召回、局部接口识别与保守 review 机制改进。", None, False, False),
    ]
    sect_pr = body.find("w:sectPr", namespaces=NS)
    insertion_index = body.index(sect_pr) if sect_pr is not None else len(body)
    for text, style, heading, page_break in content:
        body.insert(insertion_index, make_paragraph(text, style=style, heading=heading, page_break_before=page_break))
        insertion_index += 1


def main():
    if not REPORT.exists():
        raise FileNotFoundError(REPORT)
    before = media_hashes(REPORT)
    with zipfile.ZipFile(REPORT, "r") as source:
        document_xml = source.read("word/document.xml")
        root = etree.fromstring(document_xml)
        body = root.find("w:body", namespaces=NS)
        append_process(body)
        updated_xml = etree.tostring(root, xml_declaration=True, encoding="UTF-8", standalone=True)
        with zipfile.ZipFile(TEMP, "w") as target:
            for info in source.infolist():
                payload = updated_xml if info.filename == "word/document.xml" else source.read(info.filename)
                target.writestr(info, payload)
    after = media_hashes(TEMP)
    if before != after:
        TEMP.unlink(missing_ok=True)
        raise RuntimeError("Embedded image bytes changed; refusing to overwrite the user-edited report.")
    TEMP.replace(REPORT)
    print(f"updated: {REPORT}")
    print(f"images preserved: {len(before)}")


if __name__ == "__main__":
    main()
