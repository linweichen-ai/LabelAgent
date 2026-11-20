#!/usr/bin/env python3
"""生成中文单栏、IEEE Transactions 论文结构风格的 LabelAgent 技术报告。"""

from __future__ import annotations

from pathlib import Path

from PIL import Image, ImageDraw, ImageFont
from docx import Document
from docx.enum.table import WD_CELL_VERTICAL_ALIGNMENT, WD_TABLE_ALIGNMENT
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from docx.shared import Inches, Pt, RGBColor


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "docs" / "deliverables"
ASSETS = OUT / "ieee-cn-assets"
OUTPUT = OUT / "LabelAgent_中文单栏IEEE_Transactions风格技术论文_20261120.docx"


def font(run, size=10.5, *, bold=False, italic=False, family="宋体"):
    run.font.name = family
    run._element.rPr.rFonts.set(qn("w:eastAsia"), family)
    run.font.size = Pt(size)
    run.font.bold = bold
    run.font.italic = italic
    run.font.color.rgb = RGBColor(0, 0, 0)


def page_number(paragraph):
    paragraph.alignment = WD_ALIGN_PARAGRAPH.CENTER
    run = paragraph.add_run()
    begin = OxmlElement("w:fldChar"); begin.set(qn("w:fldCharType"), "begin")
    instruction = OxmlElement("w:instrText"); instruction.set(qn("xml:space"), "preserve"); instruction.text = "PAGE"
    end = OxmlElement("w:fldChar"); end.set(qn("w:fldCharType"), "end")
    run._r.extend([begin, instruction, end]); font(run, 8, family="Times New Roman")


def keep(paragraph, next_only=False):
    ppr = paragraph._p.get_or_add_pPr()
    ppr.append(OxmlElement("w:keepNext" if next_only else "w:keepLines"))


def paragraph(doc, text, *, indent=True, size=11.3, space_after=5, bold_lead=None):
    p = doc.add_paragraph()
    p.alignment = WD_ALIGN_PARAGRAPH.JUSTIFY
    p.paragraph_format.line_spacing = 1.18
    p.paragraph_format.space_before = Pt(0)
    p.paragraph_format.space_after = Pt(space_after)
    if indent:
        p.paragraph_format.first_line_indent = Pt(21)
    if bold_lead and text.startswith(bold_lead):
        font(p.add_run(bold_lead), size, bold=True)
        font(p.add_run(text[len(bold_lead):]), size)
    else:
        font(p.add_run(text), size)
    keep(p)
    return p


def primary(doc, number, title):
    p = doc.add_paragraph()
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p.paragraph_format.space_before = Pt(10)
    p.paragraph_format.space_after = Pt(5)
    prefix = f"{number}. " if number else ""
    font(p.add_run(f"{prefix}{title}"), 14, bold=True, family="微软雅黑")
    keep(p, next_only=True)
    return p


def secondary(doc, letter, title):
    p = doc.add_paragraph()
    p.paragraph_format.space_before = Pt(6)
    p.paragraph_format.space_after = Pt(3)
    font(p.add_run(f"{letter}. {title}"), 12.3, bold=True, family="微软雅黑")
    keep(p, next_only=True)
    return p


def formula(doc, text, note=None):
    p = doc.add_paragraph()
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p.paragraph_format.space_before = Pt(4)
    p.paragraph_format.space_after = Pt(2)
    font(p.add_run(text), 11, italic=True, family="Times New Roman")
    if note:
        q = doc.add_paragraph()
        q.alignment = WD_ALIGN_PARAGRAPH.CENTER
        q.paragraph_format.space_after = Pt(4)
        font(q.add_run(note), 9)


def caption(doc, text, *, table=False):
    p = doc.add_paragraph()
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p.paragraph_format.space_before = Pt(4 if table else 1)
    p.paragraph_format.space_after = Pt(2 if table else 5)
    font(p.add_run(text), 9, bold=table)
    keep(p, next_only=table)
    return p


def figure(doc, path, label, width=6.7, note=None):
    p = doc.add_paragraph()
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p.paragraph_format.space_before = Pt(4)
    p.paragraph_format.space_after = Pt(0)
    p.add_run().add_picture(str(path), width=Inches(width))
    keep(p, next_only=True)
    caption(doc, label)
    if note:
        q = doc.add_paragraph()
        q.alignment = WD_ALIGN_PARAGRAPH.CENTER
        q.paragraph_format.line_spacing = 1.2
        q.paragraph_format.space_after = Pt(9)
        font(q.add_run(note), 8.5, italic=True)


def _cell_border(cell, *, top=None, bottom=None):
    tc_pr = cell._tc.get_or_add_tcPr()
    borders = tc_pr.first_child_found_in("w:tcBorders")
    if borders is None:
        borders = OxmlElement("w:tcBorders")
        tc_pr.append(borders)
    for edge, value in (("top", top), ("bottom", bottom)):
        if value is None:
            continue
        node = borders.find(qn(f"w:{edge}"))
        if node is None:
            node = OxmlElement(f"w:{edge}")
            borders.append(node)
        node.set(qn("w:val"), "single")
        node.set(qn("w:sz"), str(value))
        node.set(qn("w:space"), "0")
        node.set(qn("w:color"), "000000")


def table(doc, label, headers, rows, widths):
    caption(doc, label, table=True)
    t = doc.add_table(rows=1, cols=len(headers))
    t.alignment = WD_TABLE_ALIGNMENT.CENTER
    t.autofit = False
    layout = OxmlElement("w:tblLayout"); layout.set(qn("w:type"), "fixed"); t._tbl.tblPr.append(layout)
    for column, width in zip(t.columns, widths):
        column.width = Inches(width)
    for index, text in enumerate(headers):
        cell = t.rows[0].cells[index]
        cell.width = Inches(widths[index])
        cell.vertical_alignment = WD_CELL_VERTICAL_ALIGNMENT.CENTER
        cell.text = ""
        p = cell.paragraphs[0]; p.alignment = WD_ALIGN_PARAGRAPH.CENTER
        p.paragraph_format.space_before = Pt(3); p.paragraph_format.space_after = Pt(3)
        font(p.add_run(text), 10.2, bold=True, family="微软雅黑")
        _cell_border(cell, top=10, bottom=7)
    for row_index, values in enumerate(rows):
        cells = t.add_row().cells
        for index, text in enumerate(values):
            cell = cells[index]
            cell.width = Inches(widths[index])
            cell.vertical_alignment = WD_CELL_VERTICAL_ALIGNMENT.CENTER
            cell.text = ""
            p = cell.paragraphs[0]
            p.alignment = WD_ALIGN_PARAGRAPH.LEFT
            p.paragraph_format.space_after = Pt(0)
            p.paragraph_format.space_before = Pt(2); p.paragraph_format.space_after = Pt(2)
            font(p.add_run(str(text)), 10)
    for cell in t.rows[-1].cells:
        _cell_border(cell, bottom=10)
    return t


def cn_font(size, bold=False):
    for path in (
        Path(r"C:\Windows\Fonts\msyhbd.ttc" if bold else r"C:\Windows\Fonts\msyh.ttc"),
        Path(r"C:\Windows\Fonts\simhei.ttf"),
    ):
        if path.exists():
            return ImageFont.truetype(str(path), size)
    return ImageFont.load_default()


def prepare_assets():
    ASSETS.mkdir(parents=True, exist_ok=True)
    shared_assets = [
        OUT / "manual-assets/closed-loop-architecture.png",
        OUT / "manual-assets/vlm-speed-chart.png",
        OUT / "manual-assets/training-status.png",
    ]
    if not all(path.exists() for path in shared_assets):
        from build_labelagent_progress_manual import architecture_diagram, speed_chart, training_chart

        architecture_diagram()
        speed_chart()
        training_chart()

    # 真坡/假坡并列图，图内说明全部中文。
    sources = [
        ROOT / "artifacts/new-dataset-fast-vlm5090-slope-round331/camera0.png",
        ROOT / "artifacts/new-dataset-fast-vlm5090-slope-round331/camera3.png",
    ]
    panels = []
    target_h = 690
    for path in sources:
        image = Image.open(path).convert("RGB")
        ratio = target_h / image.height
        panels.append(image.resize((int(image.width * ratio), target_h), Image.Resampling.LANCZOS))
    canvas = Image.new("RGB", (sum(i.width for i in panels), target_h + 65), "white")
    x = 0
    for image in panels:
        canvas.paste(image, (x, 65)); x += image.width
    draw = ImageDraw.Draw(canvas)
    draw.text((70, 14), "真实坡板：保留", fill=(20, 110, 60), font=cn_font(34, True))
    draw.text((panels[0].width + 70, 14), "ego 重叠假坡：抑制并转人工", fill=(170, 70, 20), font=cn_font(34, True))
    slope = ASSETS / "斜坡正负样例.png"; canvas.save(slope, optimize=True)

    # Round335 frame89 四联图，去除原英文标题并重新加中文图内说明。
    source = Image.open(ROOT / "artifacts/pallet-lidar-instance-sam-round335/instance-lidar-sam-review.jpg").convert("RGB")
    panel_h = source.height // 4
    raw = source.crop((0, panel_h * 2 + 62, source.width, panel_h * 3))
    tile_w = raw.width // 4
    lidar = Image.new("RGB", (raw.width, raw.height + 62), "white")
    lidar.paste(raw, (0, 62))
    draw = ImageDraw.Draw(lidar)
    labels = ["点云正负提示", "原机器候选", "点云引导 SAM 结果", "两者差异"]
    for index, label in enumerate(labels):
        box = draw.textbbox((0, 0), label, font=cn_font(30, True))
        width = box[2] - box[0]
        draw.text((index * tile_w + (tile_w - width) / 2, 12), label, fill=(20, 35, 50), font=cn_font(30, True))
    lidar_path = ASSETS / "点云提示frame89四联图.png"; lidar.save(lidar_path, optimize=True)
    return slope, lidar_path


def build():
    slope_asset, lidar_asset = prepare_assets()
    doc = Document()
    section = doc.sections[0]
    section.page_width = Inches(8.5)
    section.page_height = Inches(11)
    section.top_margin = Inches(0.7)
    section.bottom_margin = Inches(0.7)
    section.left_margin = Inches(0.8)
    section.right_margin = Inches(0.8)
    section.header_distance = Inches(0.28)
    section.footer_distance = Inches(0.3)
    header = section.header.paragraphs[0]
    header.alignment = WD_ALIGN_PARAGRAPH.CENTER
    font(header.add_run("LabelAgent 项目技术论文内部评审稿 · 2026年11月"), 8.5)
    page_number(section.footer.paragraphs[0])

    normal = doc.styles["Normal"]
    normal.font.name = "宋体"; normal._element.rPr.rFonts.set(qn("w:eastAsia"), "宋体")
    normal.font.size = Pt(11.3); normal.font.color.rgb = RGBColor(0, 0, 0)

    title = doc.add_paragraph(); title.alignment = WD_ALIGN_PARAGRAPH.CENTER
    title.paragraph_format.space_before = Pt(16); title.paragraph_format.space_after = Pt(8)
    font(title.add_run("LabelAgent：面向工业场景的\n安全门控多模态闭环自动标注系统"), 22, bold=True, family="微软雅黑")
    author = doc.add_paragraph(); author.alignment = WD_ALIGN_PARAGRAPH.CENTER
    author.paragraph_format.space_after = Pt(2)
    font(author.add_run("内部评审稿（作者与单位待定）"), 11, family="微软雅黑")
    note = doc.add_paragraph(); note.alignment = WD_ALIGN_PARAGRAPH.CENTER
    note.paragraph_format.space_after = Pt(12)
    font(note.add_run("证据冻结日期：2026年11月20日"), 9, italic=True)

    abstract = doc.add_paragraph(); abstract.alignment = WD_ALIGN_PARAGRAPH.JUSTIFY
    abstract.paragraph_format.left_indent = Inches(0.25); abstract.paragraph_format.right_indent = Inches(0.25)
    abstract.paragraph_format.space_after = Pt(5); abstract.paragraph_format.line_spacing = 1.08
    font(abstract.add_run("摘  要—"), 10.8, bold=True)
    font(abstract.add_run(
        "工业自动标注不仅要求模型能够识别目标并产生精细掩码，还必须解决数据身份、异步任务、模型拥塞、人工复核、训练集污染和版本回滚等系统问题。本文提出 LabelAgent，一套由可提示分割模型、视觉语言模型、执行 Agent、点云几何约束与人工门禁共同组成的闭环自动标注系统。系统将数据集、文件、相机、帧、载入代次和任务号共同绑定，避免旧请求覆盖当前结果；同时严格区分机器候选、人工真值、训练批准与部署批准。在双 RTX 5090 配对实测中，小图 VLM 通道将 camera1 总耗时由 91.500 s 降至 25.688 s，将 camera2 由 70.204 s 降至 13.125 s。栈板 v5 轮廓清理在 150 帧、1,316 个候选的只读审计中未出现策略违规。针对远距离薄栈板，三维实例点提示将一个吞入竖直背景的 7,697 像素掩码收缩为 583 像素，但 4 个困难目标仍无一通过完整自动替换门禁。上述结果证明了工作台响应速度、故障隔离和几何约束能力的提升，而不能替代独立人工真值上的准确率结论。"
    ), 10.8)
    terms = doc.add_paragraph(); terms.alignment = WD_ALIGN_PARAGRAPH.JUSTIFY
    terms.paragraph_format.left_indent = Inches(0.25); terms.paragraph_format.right_indent = Inches(0.25)
    terms.paragraph_format.space_after = Pt(5)
    font(terms.add_run("关键词—"), 10.8, bold=True)
    font(terms.add_run("自动标注；闭环学习；工业感知；多模态融合；可提示分割；视觉语言模型；安全门控"), 10.8)
    practice = doc.add_paragraph(); practice.alignment = WD_ALIGN_PARAGRAPH.JUSTIFY
    practice.paragraph_format.left_indent = Inches(0.25); practice.paragraph_format.right_indent = Inches(0.25)
    practice.paragraph_format.space_after = Pt(10)
    font(practice.add_run("实践说明—"), 10.8, bold=True)
    font(practice.add_run("本项目最重要的工程结论是：一个看起来合理或分数很高的掩码，并不自动等于可用于训练的标注。系统必须展示中间证据、显式排队模型请求、拒绝过期结果，并在人工确认后才允许候选进入真值与训练流程。"), 10.8)

    figure(doc, OUT / "manual-assets/closed-loop-architecture.png", "图1  LabelAgent 安全门控自动标注闭环", 6.6)

    primary(doc, "I", "引言")
    paragraph(doc, "可提示分割基础模型已经能够通过文本、点、框或示例图像产生通用掩码[1]–[3]。然而，在工厂机器人数据中，真正困难的问题并不只是“能不能分出来”，而是如何确认当前处理的是哪一批数据、哪一路相机和哪一帧；如何在模型繁忙时保持工作台可用；如何让 VLM 发现新类别；以及如何保证机器结果不会绕过人工复核直接进入训练。")
    paragraph(doc, "LabelAgent 面向上述系统性问题构建。其前端负责数据身份、逐帧操作、候选叠加与人工复核；后端依次调用 SAM3、VLM 与 Agent；点云、标定和时序信息作为可选几何证据；最终只有通过人工和训练门禁的数据才允许进入后训练与灰度回灌。斜坡虽然在下游可与地面合并为可通行区域，但在发现和评测阶段仍保留独立语义，以便观察困难坡面的召回和误检。")
    paragraph(doc, "本文的贡献不是宣称模型已经实现全自动真值生成，而是给出一套证据边界明确的工程方案：1）以文件载入代次与任务号约束异步结果；2）以 4090 稳定主路由和 5090 小图快速通道兼顾速度与回退；3）以候选、真值、训练批准和部署批准的状态隔离避免自我强化；4）以栈板边界、斜坡保护和点云实例提示验证多模态优化；5）完整保留未通过门禁的方法和原因。")

    primary(doc, "II", "问题定义与设计目标")
    secondary(doc, "A", "问题定义")
    paragraph(doc, "设当前工作台状态由数据集身份 d、文件 f、相机 c、帧 i、文件载入代次 g 和任务号 q 共同确定。任一模型结果 r 只有在返回时仍与当前状态完全一致，才允许显示或写入候选区。否则结果被视为过期，不得覆盖用户后续选择。")
    paragraph(doc, "简单来说，模型结果只有在数据集、文件、相机、帧、载入代次和任务号都没有变化时才会被接受；只要其中一项不一致，系统就拒绝这条旧结果。", indent=False)
    paragraph(doc, "自动标注结果分为四个相互隔离的状态：机器候选、人工真值、训练批准和部署批准。机器候选可被显示、删除或改类，但不能自动晋级为人工真值；人工真值也必须在训练集与冻结测试集隔离后才允许训练；训练完成的模型仍需外部门禁通过后才能部署。")
    secondary(doc, "B", "设计目标")
    paragraph(doc, "系统设计同时追求五个目标：工作台操作连续可恢复；新旧数据集与四路相机身份清晰；类别覆盖和中间调用可见；模型加速不改变安全复核路径；所有优化都有输入哈希、状态回读和回滚记录。对于没有人工像素真值的实验，只报告延迟、候选规模、几何一致性和门禁结果，不报告像素准确率。")

    primary(doc, "III", "系统架构与实现")
    secondary(doc, "A", "数据身份与异步一致性")
    paragraph(doc, "数据切换、快速拖动时间轴、区域编辑、清空候选、保存真值、设置参考帧、停止任务和恢复结果均绑定文件载入代次。时间轴遵循“最后一次选择生效”，模型返回后还要核对当前图像摘要。画布尺寸或分栏变化后，前端重新计算坐标并读回当前帧，避免推理结果漂浮在原图之外。")
    secondary(doc, "B", "SAM3、VLM 与 Agent 协作")
    paragraph(doc, "SAM3 负责产生实例轮廓；VLM 负责场景描述、已有类别发现、未知 label 提议、空间冲突解释和高风险候选复查；Agent 只能执行保守增删、改类建议或转人工操作。页面同时展示 SAM3 分割、VLM 复查输入、VLM 决策输出和 Agent 标注结果；若某阶段没有真实调用，页面明确显示无记录，而不是用空数组伪装成功。")
    secondary(doc, "C", "共享模型槽与故障恢复")
    paragraph(doc, "Local VLM busy 的本质是共享模型槽正在处理其他请求。当前工作台不再直接失败，而是显示排队状态并进行受控重试。精确响应缓存、参考帧发现预取、相邻帧预取和服务端发现结果复用共同减少重复请求。任何缓存命中必须同时匹配图像摘要、标签表、配置、相机、帧和文件载入代次。")
    figure(doc, ROOT / "artifacts/dataset-coverage-browser-acceptance-round324-current/new-dataset-class-coverage.png", "图2  当前工作台的数据身份、候选叠加与逐类别覆盖反馈", 6.6)

    doc.add_page_break()
    primary(doc, "IV", "数据集与实验协议")
    secondary(doc, "A", "数据来源")
    table(doc, "表1  数据集、用途与证据边界", ["数据来源", "用途", "边界"], [
        ["批次A的12个完整目录", "2,014个标签帧；数据加载与训练准备", "编号009为空壳并跳过"],
        ["Capture 20261103", "152帧、四相机、LiDAR；端到端与点云实验", "现有机器候选不是人工像素真值"],
        ["批次B 003/004", "问题包单图抽样与里程异常复核", "已知里程中断；不得静默修补入训练"],
        ["train72", "后训练候选池", "需人工签核后方可训练"],
        ["test24", "冻结评测集", "永久禁止回流调参"],
    ], [1.45, 3.0, 2.25])
    secondary(doc, "B", "实验原则")
    paragraph(doc, "所有线上或离线实验均保存输入摘要、模型健康状态、执行参数、输出摘要和正式状态前后哈希。浏览器验收除观察页面外，还检查 HTTP 状态、当前文件和帧、候选数量、真值写入次数及坐标读回。点云实验明确禁止读取 labels、ali_label 和 wilddet3d-labels 等参考目录。")
    secondary(doc, "C", "评价指标")
    paragraph(doc, "工作台指标包括首屏可用时间、点击到候选时间、任务恢复和过期结果拒绝；候选几何指标包括修改比例、结构孔保留率、组件合并和相邻实例重叠；点云指标包括可形成提示的目标数、请求成功数、HTTP 安全拒绝数、选中掩码像素和完整门禁通过数。")
    paragraph(doc, "所有速度对比都使用同一路相机和配对输入，避免把不同样本难度误当成加速效果。", indent=False)

    primary(doc, "V", "工作台稳定性与推理效率")
    secondary(doc, "A", "工作台稳定性")
    paragraph(doc, "当前版本已经实现候选先显示、VLM 与 Agent 后台复核。用户不再需要等待整个链路结束后才能看到掩码；后台失败也不会抹掉已经生成的候选。快速翻帧、切数据、重试和恢复均受到任务与文件载入代次约束。画布叠加实测最大坐标偏差为 0.015625 像素。Round333 完整回归包含 Python 1,125 项与 186 项子测试、JavaScript 148 项，正式状态逐字段回读未发现真值、候选或参考帧越权变化。")
    secondary(doc, "B", "双5090小图VLM加速")
    paragraph(doc, "在配对 camera1 请求上，小图 VLM 通道将总耗时从 91.500 s 降至 25.688 s，下降 71.93%；camera2 从 70.204 s 降至 13.125 s，下降 81.30%。参考场景后台预取耗时 17.405 s，其中 VLM 发现为 15.898 s；正式点击到候选仅 2.622 s。大图 1920×1536 的 5090 路线虽然从 55.813 s 降到 36.093 s，但改变了高风险候选的复核路径，因此没有部署。")
    figure(doc, OUT / "manual-assets/vlm-speed-chart.png", "图3  双RTX 5090小图VLM通道的配对实测耗时", 6.3)
    table(doc, "表2  主要效率结果", ["路径", "优化前", "优化后", "变化"], [
        ["camera1 小图VLM", "91.500 s", "25.688 s", "下降71.93%"],
        ["camera2 小图VLM", "70.204 s", "13.125 s", "下降81.30%"],
        ["参考发现预取后的正式点击", "需重复发现", "2.622 s", "严格绑定缓存命中"],
        ["大图5090候选路线", "55.813 s", "36.093 s", "因安全路径变化而拒绝"],
    ], [2.0, 1.35, 1.35, 2.0])

    primary(doc, "VI", "分割专项结果")
    secondary(doc, "A", "栈板边界清理")
    paragraph(doc, "栈板 v5 后处理针对木纹造成的碎洞和浅凹口进行图像引导修补，同时保留长叉孔、板缝和塑料网格等真实结构。150帧正式候选只读审计共包含1,316个候选，其中1,126个为栈板；1,040个历史栈板候选仍会被v5调整。全量最大修改比例为2.9408%，最低结构孔保留率为89.5911%，仅发生3次受控组件合并，策略违规数为0。该结果证明迁移约束可执行，但不等于像素准确率。")
    figure(doc, ROOT / "artifacts/pallet-boundary-live-round334/before-after.jpg", "图4  历史栈板候选使用v5清理前后的只读对照", 6.15, "绿色为候选叠加；本图未修改人工真值、训练批准或模型权重。")
    secondary(doc, "B", "斜坡与可通行区域")
    paragraph(doc, "斜坡判断不依赖颜色。系统综合使用VLM场景语义、SAM3文本或点提示、画面空间位置、与floor的连接关系、候选宽高比、ego重叠和必要时的LiDAR几何。camera0真实坡板保持不变；camera3假坡因99.9%被高置信ego覆盖而退出自动叠加并转人工。下游规划可以将slope与floor并入可通行区域，但评测仍保留slope类别。")
    figure(doc, slope_asset, "图5  真实坡板保留与ego重叠假坡抑制", 6.6)
    secondary(doc, "C", "开放类别与其他机器人")
    paragraph(doc, "当VLM明确判断存在other robot而基线SAM没有结果时，系统可追加autonomous pallet truck提示，最多保留SAM分数最高的3个候选，再交由8B模型执行高风险盲审。该补漏证明新label可以进入候选流程，但仍不能绕过人工门禁或自动写入训练集。")
    figure(doc, ROOT / "artifacts/other-robot-fallback-round310-positive-v2/00-1789897560182495905-fallback.jpg", "图6  远处other robot补漏候选示例", 5.6, "该结果保持机器候选状态，未自动晋级为人工真值。")

    primary(doc, "VII", "点云优化方法与实验")
    secondary(doc, "A", "方法")
    paragraph(doc, "点云不直接替代SAM3，而是提供不依赖颜色的空间约束。旧版远处栈板提示先把离地0.04–0.45 m的点投影到图像，再在图像平面膨胀并求连通域，容易把不同深度但投影相邻的结构连接起来。新版只用图像位置关联冻结候选和种子，实例成员由LiDAR XY平面0.18 m邻接聚类决定。")
    paragraph(doc, "一个三维簇至少需要2个命中候选的种子；若第一和第二簇命中数相同，则按歧义失败关闭；簇最大跨度不得超过1.8 m。选中簇提供正点，远离该簇的地面点提供地面负点，候选上方的高点提供竖直背景负点。SAM结果仍需通过候选覆盖、面积扩张、主轴方向、相邻实例重叠以及全部正负点检查。")
    secondary(doc, "B", "实测结果")
    paragraph(doc, "在Round321同一组4个冻结困难栈板上，新方法仍有3个目标能够形成提示。三组可用实例分别包含4、10和18个三维簇点，每组获得4个地面负点和2个竖直背景负点。共运行9次点提示SAM，6次返回掩码、3次被HTTP 400安全拒绝，其他错误为0。没有目标通过完整自动替换门禁。")
    table(doc, "表3  三维实例点提示与旧提示的选中掩码对比", ["目标", "旧掩码像素", "三维提示像素", "变化", "完整门禁"], [
        ["camera0 frame39", "1,685", "1,426", "−15.37%", "未通过"],
        ["camera0 frame35", "2,181", "1,500", "−31.22%", "未通过"],
        ["camera1 frame89", "7,697", "583", "−92.43%", "未通过"],
        ["camera2 frame64", "无可靠种子", "无可靠种子", "失败关闭", "未通过"],
    ], [1.5, 1.35, 1.45, 1.2, 1.2])
    point_result = paragraph(doc, "camera1 frame89的旧结果以约0.8789的SAM高分扩张为7,697像素竖直背景块；新版保守单锚点结果缩到583像素，并同时满足正点覆盖、地面负点不被吞入和竖直背景负点不被吞入。该结果证明背景泄漏得到抑制，但没有人工像素真值，不能据此宣称IoU或召回提高。")
    point_result.paragraph_format.space_before = Pt(5)
    figure(doc, lidar_asset, "图7  camera1 frame89的三维点提示、旧候选、SAM结果与差异", 6.7)
    secondary(doc, "C", "点云方案的适用边界")
    paragraph(doc, "点云适合三类优化：低矮栈板实例分离；基于局部平面法向、高度梯度和平面残差的斜坡发现；结合离地动态点簇与ego排除的其他机器人候选。其前提是标定、时间同步和点云可见性可信。批次B 003与004存在里程中断，涉及这两个包的时序补偿必须单独标记，不能静默修复后混入训练。")

    primary(doc, "VIII", "后训练与微调状态")
    secondary(doc, "A", "VLM后训练")
    paragraph(doc, "train72与test24已创建，分别包含72张和24张图像，文件名与内容哈希零交集。2B VLM LoRA完成两步冒烟、一轮62步和两轮62步训练，权重张量审计确认发生真实更新。一轮候选在受控dev8上将合法结构化输出由0/8提高到8/8；但冻结test24候选与8B教师在slope上仅18/24一致，而且教师一致率不等于人工准确率，因此2B模型没有部署。")
    secondary(doc, "B", "SAM3微调")
    paragraph(doc, "SAM3目前使用预训练权重、工程提示、ROI、几何门禁和轮廓后处理，尚未完成项目人工像素真值上的微调。主要原因是96张CVAT任务仍缺少完整人工COCO导出和逐图签核。下一步优先只训练mask decoder或采用轻量LoRA，训练样本覆盖木纹孔洞、远处薄栈板、局部坡板、普通地面困难负样本和ego重叠假坡；test24永久只评测。")
    figure(doc, OUT / "manual-assets/training-status.png", "图8  当前训练与部署状态", 6.2)

    primary(doc, "IX", "消融、负结果与部署决策")
    paragraph(doc, "系统保留未通过实验，因为这些结果明确了安全边界。任何只改善单项速度或召回、却改变误检和人工复核路径的方法，都不允许替换当前主路。")
    doc.add_page_break()
    table(doc, "表4  已尝试但未部署的主要路线", ["路线", "观察到的改善", "阻断原因"], [
        ["大图5090快速路由", "55.813 s降至36.093 s", "人体与控制台混合候选的复核路径改变"],
        ["激进斜坡提示", "开发集召回提高", "普通地面误报为坡"],
        ["相邻帧光流", "运输515个候选", "完整面积、主轴和邻居门禁0个通过"],
        ["远处栈板ROI扩张", "少数候选面积变大", "吞入背景或相邻实例；0/16通过"],
        ["第一版LiDAR点提示", "7次正常返回", "高分结果仍扩成竖直背景；0/4通过"],
        ["2B VLM LoRA替换8B", "速度候选成立", "斜坡与other robot外部门禁未通过"],
    ], [1.8, 2.05, 2.85])
    paragraph(doc, "上述负结果表明，继续增加提示词、放宽面积阈值或只追求SAM分数并不能稳定提高自动标注质量。更合理的路线是补充独立人工像素真值、采用几何上可解释的正负提示，并在相同冻结样本上同时检查召回、误检、实例粘连、结构孔保留和延迟。")

    primary(doc, "X", "局限与后续闭环计划")
    secondary(doc, "A", "当前局限")
    paragraph(doc, "目前最强证据集中在工作台一致性、模型请求可用性、延迟和候选变换安全性，而不是独立生产像素精度。部分斜坡和其他机器人结果来自同批次开发样本或助手复核；历史候选由多个算法版本生成；点云对远距离目标较稀疏；两问题包的里程中断限制时序补偿。")
    secondary(doc, "B", "下一步执行顺序")
    paragraph(doc, "第一步，在生产状态副本中完成历史栈板候选迁移演练，验证候选ID、类别、人工真值、参考帧和训练批准不变；第二步，完成train72与test24的人工复核、导出和签核；第三步，在独立像素真值上比较SAM3轻量微调、点云提示和当前主路；第四步，把工厂实时数据接入身份识别、自动候选、新label发现、Agent复核、人工门禁、训练评测和灰度回灌。")
    paragraph(doc, "仿真数据可以补充稀有坡度、遮挡和远距离组合，但必须与真实数据分层报告。任何新模型只有在冻结测试集和新厂区数据上同时满足召回、误检、延迟、跨相机一致性和可回滚部署条件后，才允许灰度替换现有主路。")

    primary(doc, "XI", "结论")
    paragraph(doc, "LabelAgent说明工业自动标注本质上是一个带状态、带安全边界的系统问题，而不是一次模型推理。当前系统已经实现明确的数据身份、稳定的浏览器操作、可见的模型排队、可回退的VLM加速、保守的候选后处理和审计记录。三维实例点提示显著缩小了一个已知背景泄漏，但没有通过自动替换门禁。因而当前最准确的结论是：平台响应速度、故障隔离和多模态候选约束得到改善，人工真值闭环和独立精度验证仍是下一阶段核心。")
    paragraph(doc, "复现声明：本文结果对应代码分支 codex/workbench-stability-20261117，与中文图文说明书、Round335点云结构化证据和生成脚本一同交付。机器候选、人工真值、训练批准与部署批准在数据和流程上保持分离。", indent=False, size=9.5)

    primary(doc, "", "参考文献")
    refs = [
        "[1] A. Kirillov 等：《分割一切》，IEEE/CVF 国际计算机视觉会议，2023年，第4015–4026页。",
        "[2] N. Ravi 等：《SAM 2：面向图像与视频的分割一切》，国际学习表征会议，2025年。",
        "[3] Meta AI：《SAM 3：基于概念的分割一切》，arXiv:2511.16719，2025年。",
        "[4] P. Wang 等：《Qwen2-VL：增强视觉语言模型对任意分辨率世界的感知》，arXiv:2409.12191，2024年。",
        "[5] M. A. Fischler 与 R. C. Bolles：《随机采样一致性：用于图像分析与自动制图的模型拟合范式》，ACM通信，第24卷第6期，第381–395页，1981年。",
        "[6] IEEE作者中心：《期刊论文结构》与《IEEE论文模板》，访问日期：2026年11月20日。",
    ]
    for ref in refs:
        p = doc.add_paragraph(); p.alignment = WD_ALIGN_PARAGRAPH.JUSTIFY
        p.paragraph_format.left_indent = Pt(21); p.paragraph_format.first_line_indent = Pt(-21)
        p.paragraph_format.space_after = Pt(2); font(p.add_run(ref), 9.5)

    doc.save(OUTPUT)
    return OUTPUT


if __name__ == "__main__":
    print(build())
