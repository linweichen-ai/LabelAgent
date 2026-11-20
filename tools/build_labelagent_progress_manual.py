#!/usr/bin/env python3
"""Build the illustrated LabelAgent progress and operating manual."""

from __future__ import annotations

import hashlib
import re
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont
from docx import Document
from docx.enum.section import WD_SECTION
from docx.enum.table import WD_ALIGN_VERTICAL, WD_TABLE_ALIGNMENT
from docx.enum.text import WD_ALIGN_PARAGRAPH, WD_BREAK
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from docx.shared import Inches, Pt, RGBColor


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "docs" / "deliverables"
TEMP = OUT / "manual-assets"
HANDOFF = ROOT / "docs" / "WORKBENCH_STABILITY_HANDOFF_20261117.md"

TITLE = "LabelAgent 项目进展与使用说明书"
SUBTITLE = "工作台稳定性 新数据集 自动标注 斜坡 栈板与后训练"
VERSION = "版本 2026 11 20"


def font(size: int, bold: bool = False):
    path = Path(r"C:\Windows\Fonts\msyhbd.ttc" if bold else r"C:\Windows\Fonts\msyh.ttc")
    return ImageFont.truetype(str(path), size=size)


def set_run_font(run, name="Microsoft YaHei", size=None, bold=None, color="000000"):
    run.font.name = name
    run._element.get_or_add_rPr().rFonts.set(qn("w:eastAsia"), name)
    run._element.rPr.rFonts.set(qn("w:ascii"), name)
    run._element.rPr.rFonts.set(qn("w:hAnsi"), name)
    if size is not None:
        run.font.size = Pt(size)
    if bold is not None:
        run.bold = bold
    run.font.color.rgb = RGBColor.from_string(color)


def set_cell_shading(cell, fill):
    tc_pr = cell._tc.get_or_add_tcPr()
    shd = tc_pr.find(qn("w:shd"))
    if shd is None:
        shd = OxmlElement("w:shd")
        tc_pr.append(shd)
    shd.set(qn("w:fill"), fill)


def set_cell_margins(cell, top=100, start=120, bottom=100, end=120):
    tc = cell._tc
    tc_pr = tc.get_or_add_tcPr()
    tc_mar = tc_pr.first_child_found_in("w:tcMar")
    if tc_mar is None:
        tc_mar = OxmlElement("w:tcMar")
        tc_pr.append(tc_mar)
    for margin, value in (("top", top), ("start", start),
                          ("bottom", bottom), ("end", end)):
        node = tc_mar.find(qn(f"w:{margin}"))
        if node is None:
            node = OxmlElement(f"w:{margin}")
            tc_mar.append(node)
        node.set(qn("w:w"), str(value))
        node.set(qn("w:type"), "dxa")


def set_table_borders(table, color="D9D9D9", size="6"):
    tbl_pr = table._tbl.tblPr
    borders = tbl_pr.first_child_found_in("w:tblBorders")
    if borders is None:
        borders = OxmlElement("w:tblBorders")
        tbl_pr.append(borders)
    for edge in ("top", "left", "bottom", "right", "insideH", "insideV"):
        tag = f"w:{edge}"
        element = borders.find(qn(tag))
        if element is None:
            element = OxmlElement(tag)
            borders.append(element)
        element.set(qn("w:val"), "single")
        element.set(qn("w:sz"), size)
        element.set(qn("w:space"), "0")
        element.set(qn("w:color"), color)


def set_repeat_header(row):
    tr_pr = row._tr.get_or_add_trPr()
    tbl_header = OxmlElement("w:tblHeader")
    tbl_header.set(qn("w:val"), "true")
    tr_pr.append(tbl_header)


def remove_paragraph_borders(paragraph):
    p_pr = paragraph._p.get_or_add_pPr()
    borders = p_pr.find(qn("w:pBdr"))
    if borders is not None:
        p_pr.remove(borders)


def add_table(doc, headers, rows, widths=None):
    table = doc.add_table(rows=1, cols=len(headers))
    table.alignment = WD_TABLE_ALIGNMENT.CENTER
    table.autofit = False
    set_table_borders(table)
    header = table.rows[0]
    set_repeat_header(header)
    for index, value in enumerate(headers):
        cell = header.cells[index]
        set_cell_shading(cell, "1F4E78")
        set_cell_margins(cell)
        cell.vertical_alignment = WD_ALIGN_VERTICAL.CENTER
        paragraph = cell.paragraphs[0]
        paragraph.alignment = WD_ALIGN_PARAGRAPH.CENTER
        run = paragraph.add_run(str(value))
        set_run_font(run, size=9, bold=True, color="FFFFFF")
        if widths:
            cell.width = Inches(widths[index])
    for row_index, values in enumerate(rows):
        row = table.add_row()
        for index, value in enumerate(values):
            cell = row.cells[index]
            set_cell_margins(cell)
            cell.vertical_alignment = WD_ALIGN_VERTICAL.CENTER
            if row_index % 2:
                set_cell_shading(cell, "F2F6FA")
            paragraph = cell.paragraphs[0]
            paragraph.alignment = (WD_ALIGN_PARAGRAPH.CENTER if index == 0
                                   else WD_ALIGN_PARAGRAPH.LEFT)
            run = paragraph.add_run(str(value))
            set_run_font(run, size=8.5)
            if widths:
                cell.width = Inches(widths[index])
    doc.add_paragraph().paragraph_format.space_after = Pt(0)
    return table


def add_para(doc, text="", *, bold_lead=None, style=None, align=None,
             space_after=6, keep=False):
    paragraph = doc.add_paragraph(style=style)
    if align is not None:
        paragraph.alignment = align
    paragraph.paragraph_format.space_after = Pt(space_after)
    paragraph.paragraph_format.line_spacing = 1.25
    paragraph.paragraph_format.keep_with_next = keep
    if bold_lead and text.startswith(bold_lead):
        first = paragraph.add_run(bold_lead)
        set_run_font(first, size=10.5, bold=True)
        rest = paragraph.add_run(text[len(bold_lead):])
        set_run_font(rest, size=10.5)
    else:
        run = paragraph.add_run(text)
        set_run_font(run, size=10.5)
    return paragraph


def add_bullets(doc, items):
    for item in items:
        paragraph = doc.add_paragraph(style="List Bullet")
        paragraph.paragraph_format.left_indent = Inches(.22)
        paragraph.paragraph_format.first_line_indent = Inches(-.12)
        paragraph.paragraph_format.space_after = Pt(4)
        paragraph.paragraph_format.line_spacing = 1.2
        run = paragraph.add_run(item)
        set_run_font(run, size=10.2)


def add_steps(doc, items):
    for index, item in enumerate(items, 1):
        paragraph = doc.add_paragraph()
        paragraph.paragraph_format.left_indent = Inches(.24)
        paragraph.paragraph_format.first_line_indent = Inches(-.24)
        paragraph.paragraph_format.space_after = Pt(6)
        number = paragraph.add_run(f"{index}  ")
        set_run_font(number, size=10.5, bold=True, color="1F4E78")
        run = paragraph.add_run(item)
        set_run_font(run, size=10.5)


def fit_image(source: Path) -> Path:
    digest = hashlib.sha256(source.read_bytes()).hexdigest()[:16]
    target = TEMP / f"{digest}.jpg"
    if target.exists():
        return target
    TEMP.mkdir(parents=True, exist_ok=True)
    with Image.open(source) as image:
        image = image.convert("RGB")
        image.thumbnail((1800, 1300), Image.Resampling.LANCZOS)
        image.save(target, "JPEG", quality=84, optimize=True)
    return target


def add_figure(doc, source: Path, caption: str, note: str | None = None,
               page_break=False):
    if page_break:
        doc.add_page_break()
    image = fit_image(source)
    with Image.open(image) as opened:
        width, height = opened.size
    max_w, max_h = 6.7, 7.0
    ratio = min(max_w / width, max_h / height)
    shown_w = width * ratio
    paragraph = doc.add_paragraph()
    paragraph.alignment = WD_ALIGN_PARAGRAPH.CENTER
    paragraph.paragraph_format.keep_with_next = True
    paragraph.add_run().add_picture(str(image), width=Inches(shown_w))
    cap = doc.add_paragraph()
    cap.alignment = WD_ALIGN_PARAGRAPH.CENTER
    cap.paragraph_format.space_after = Pt(3 if note else 8)
    cap.paragraph_format.keep_with_next = bool(note)
    run = cap.add_run(caption)
    set_run_font(run, size=9, bold=True)
    if note:
        n = doc.add_paragraph()
        n.alignment = WD_ALIGN_PARAGRAPH.CENTER
        n.paragraph_format.space_after = Pt(8)
        run = n.add_run(note)
        set_run_font(run, size=8.5, color="595959")


def architecture_diagram() -> Path:
    path = TEMP / "closed-loop-architecture.png"
    canvas = Image.new("RGB", (1800, 980), "white")
    draw = ImageDraw.Draw(canvas)
    title_font, box_font, small_font = font(54, True), font(30, True), font(24)
    draw.text((70, 50), "LabelAgent 自动标注闭环", fill="#111111", font=title_font)
    boxes = [
        ("工厂数据接入", "MCAP 图片 LiDAR 标定"),
        ("数据集识别", "批次 文件 相机 帧强绑定"),
        ("SAM3 候选", "分割实例与几何轮廓"),
        ("VLM 发现与复查", "新类别 语义 风险解释"),
        ("Agent 修改", "保守增删 改类与复核队列"),
        ("人工门禁", "确认真值或保留候选"),
        ("训练与评测", "train 与 test 永久隔离"),
        ("模型回灌", "门禁通过后灰度部署"),
    ]
    positions = []
    for row in range(2):
        for col in range(4):
            x = 70 + col * 425
            y = 180 + row * 360
            positions.append((x, y, x + 350, y + 210))
    for index, ((heading, detail), pos) in enumerate(zip(boxes, positions)):
        fill = "#E8F1F8" if index < 6 else "#EEF5E8"
        draw.rounded_rectangle(pos, radius=22, fill=fill, outline="#1F4E78", width=4)
        x1, y1, x2, y2 = pos
        draw.text((x1 + 25, y1 + 38), heading, fill="#0F2940", font=box_font)
        draw.multiline_text((x1 + 25, y1 + 105), detail, fill="#354A5F",
                            font=small_font, spacing=8)
    arrows = [(0, 1), (1, 2), (2, 3), (3, 4), (4, 5), (5, 6), (6, 7)]
    for a, b in arrows:
        ax1, ay1, ax2, ay2 = positions[a]
        bx1, by1, bx2, by2 = positions[b]
        if ay1 == by1:
            start, end = (ax2 + 8, (ay1 + ay2) // 2), (bx1 - 8, (by1 + by2) // 2)
        else:
            start, end = ((ax1 + ax2) // 2, ay2 + 8), ((bx1 + bx2) // 2, by1 - 8)
        draw.line([start, end], fill="#1F4E78", width=6)
        ex, ey = end
        draw.polygon([(ex, ey), (ex - 16, ey - 12), (ex - 16, ey + 12)], fill="#1F4E78")
    draw.arc((120, 620, 1700, 950), start=5, end=175, fill="#2E7D32", width=6)
    draw.text((680, 900), "新数据和人工纠错持续回流 但必须经过训练门禁", fill="#2E7D32", font=small_font)
    canvas.save(path)
    return path


def speed_chart() -> Path:
    path = TEMP / "vlm-speed-chart.png"
    canvas = Image.new("RGB", (1500, 850), "white")
    draw = ImageDraw.Draw(canvas)
    draw.text((60, 35), "5090 小图 VLM 加速前后", fill="#111111", font=font(48, True))
    values = [("camera1", 91.500, 25.688), ("camera2", 70.204, 13.125)]
    colors = ("#9FBAD0", "#1F78B4")
    for row, (name, before, after) in enumerate(values):
        y = 210 + row * 280
        draw.text((60, y + 50), name, fill="#222222", font=font(32, True))
        for idx, (label, value) in enumerate((("加速前", before), ("加速后", after))):
            yy = y + idx * 100
            width = int(value / 100 * 1050)
            draw.rounded_rectangle((280, yy, 280 + width, yy + 62), radius=12, fill=colors[idx])
            draw.text((300 + width, yy + 10), f"{label} {value:.3f} 秒",
                      fill="#222222", font=font(26))
    draw.text((60, 760), "camera1 下降 71.93%  camera2 下降 81.30%  候选几何保持一致",
              fill="#354A5F", font=font(28))
    canvas.save(path)
    return path


def training_chart() -> Path:
    path = TEMP / "training-status.png"
    canvas = Image.new("RGB", (1600, 780), "white")
    draw = ImageDraw.Draw(canvas)
    draw.text((60, 35), "训练与部署状态", fill="#111111", font=font(48, True))
    rows = [
        ("VLM 2B LoRA", "已完成隔离后训练", "未部署", "训练62张  dev8 评测  test24只评测"),
        ("正式 8B VLM", "持续在生产运行", "已部署", "4090主路由  5090小图快速通道"),
        ("SAM3", "尚未完成像素微调", "等待真值", "缺少通过人工门禁的像素级掩码"),
        ("历史栈板候选", "v5迁移预检通过", "未批量写入", "1040个旧候选待可回滚迁移"),
    ]
    for i, (name, state, deploy, note) in enumerate(rows):
        y = 145 + i * 145
        draw.rounded_rectangle((60, y, 1540, y + 112), radius=16,
                               fill="#F2F6FA", outline="#D9D9D9", width=3)
        draw.text((90, y + 18), name, fill="#0F2940", font=font(28, True))
        draw.text((420, y + 18), state, fill="#1F4E78", font=font(26))
        deploy_color = "#2E7D32" if deploy == "已部署" else "#B36B00"
        draw.text((840, y + 18), deploy, fill=deploy_color, font=font(26, True))
        draw.text((90, y + 65), note, fill="#595959", font=font(23))
    canvas.save(path)
    return path


def linked_images():
    text = HANDOFF.read_text(encoding="utf-8")
    pattern = re.compile(r"\[([^\]]+)\]\(([^\)]+\.(?:png|jpg|jpeg))\)", re.I)
    result = []
    for label, relative in pattern.findall(text):
        path = HANDOFF.parent / relative
        if path.is_file():
            result.append((label, path))
    return result


def build() -> Path:
    OUT.mkdir(parents=True, exist_ok=True)
    TEMP.mkdir(parents=True, exist_ok=True)
    doc = Document()
    section = doc.sections[0]
    section.page_width = Inches(8.5)
    section.page_height = Inches(11)
    section.top_margin = Inches(.7)
    section.bottom_margin = Inches(.7)
    section.left_margin = Inches(.75)
    section.right_margin = Inches(.75)

    styles = doc.styles
    for name in ("Normal", "Title", "Subtitle", "Heading 1", "Heading 2", "Heading 3"):
        style = styles[name]
        style.font.name = "Microsoft YaHei"
        style._element.rPr.rFonts.set(qn("w:eastAsia"), "Microsoft YaHei")
        style.font.color.rgb = RGBColor(0, 0, 0)
    styles["Normal"].font.size = Pt(10.5)
    styles["Title"].font.size = Pt(26)
    styles["Title"].font.bold = True
    styles["Subtitle"].font.size = Pt(13)
    styles["Heading 1"].font.size = Pt(18)
    styles["Heading 1"].font.bold = True
    styles["Heading 1"].paragraph_format.space_before = Pt(12)
    styles["Heading 1"].paragraph_format.space_after = Pt(8)
    styles["Heading 1"].paragraph_format.keep_with_next = True
    styles["Heading 2"].font.size = Pt(13.5)
    styles["Heading 2"].font.bold = True
    styles["Heading 2"].paragraph_format.space_before = Pt(9)
    styles["Heading 2"].paragraph_format.space_after = Pt(6)
    styles["Heading 2"].paragraph_format.keep_with_next = True
    styles["Heading 3"].font.size = Pt(11.5)
    styles["Heading 3"].font.bold = True

    # Cover
    doc.add_paragraph().paragraph_format.space_after = Pt(80)
    title = doc.add_paragraph(style="Title")
    title.alignment = WD_ALIGN_PARAGRAPH.CENTER
    title.add_run(TITLE)
    remove_paragraph_borders(title)
    subtitle = doc.add_paragraph(style="Subtitle")
    subtitle.alignment = WD_ALIGN_PARAGRAPH.CENTER
    subtitle.add_run(SUBTITLE)
    add_para(doc, VERSION, align=WD_ALIGN_PARAGRAPH.CENTER, space_after=28)
    add_figure(doc, architecture_diagram(), "图 1  LabelAgent 自动标注闭环总览")
    add_para(doc, "适用对象：项目负责人、演示人员、标注复核人员和后续开发人员",
             align=WD_ALIGN_PARAGRAPH.CENTER, space_after=3)
    add_para(doc, "生产入口：http://127.0.0.1:8765",
             align=WD_ALIGN_PARAGRAPH.CENTER, space_after=3)
    add_para(doc, "同步基线：GitHub 与 CNB 提交 c8e6311  本地补充 Round334 只读预检",
             align=WD_ALIGN_PARAGRAPH.CENTER, space_after=0)
    doc.add_page_break()

    doc.add_heading("文档结论", level=1)
    add_para(doc, "当前工作台已经从能运行推进到可稳定演示和可追溯使用。新旧数据集可以区分，四路相机可以逐帧推理，候选能够叠加回原图，VLM 与 Agent 的输入输出可查看，参考帧预取和5090小图通道显著降低等待。斜坡仍按独立语义发现，但下游可以与地面统一为可通行区域；真正坡板可以保留，普通地面与本车重叠造成的假坡已增加保护。")
    add_para(doc, "现阶段仍不能把机器候选称为人工真值。SAM3尚未完成基于人工像素掩码的微调；2B VLM LoRA后训练已经真实执行，但外部门禁暴露斜坡召回不足，因此没有部署。栈板v5边界清理对新推理已生效，历史保存候选仍有1,040个可进一步优化，当前正在走可回滚迁移流程。")
    add_table(doc, ["模块", "当前状态", "实际作用", "边界"], [
        ["工作台", "已部署并通过浏览器验收", "数据切换 推理 叠加 复核和恢复可用", "仍需避免低磁盘长批次"],
        ["SAM3", "正式服务在线", "产生实例掩码和点提示分割", "尚未做人工像素真值微调"],
        ["VLM", "8B正式 5090小图加速", "发现类别 解释风险 复查候选", "大图5090路线未通过语义门禁"],
        ["Agent", "保守执行并保留复核队列", "修补 候选过滤 改类建议", "不会自动写成人工真值"],
        ["斜坡", "正负样本保护已上线", "保留真坡 抑制ego重叠假坡", "远处局部坡仍是难点"],
        ["栈板", "v5新候选已上线", "去木纹碎洞 平滑凹口 保留叉孔", "旧候选尚未全部迁移"],
        ["后训练", "2B LoRA离线完成", "验证后训练链路与速度候选", "未部署 不代表人工精度"],
    ], widths=[1.0, 1.65, 2.35, 1.75])

    doc.add_heading("目录", level=1)
    add_bullets(doc, [
        "一  系统是什么以及当前达到的效果",
        "二  工作台操作说明",
        "三  过去主要改动与作用",
        "四  斜坡 栈板 other robot专项结果",
        "五  微调与后训练现状",
        "六  新数据集与跨数据验收",
        "七  已否决路线与保留原因",
        "八  当前限制和下一步计划",
        "九  全部图像证据附录",
    ])

    doc.add_page_break()
    doc.add_heading("一 系统是什么以及当前达到的效果", level=1)
    add_para(doc, "LabelAgent不是单独调用SAM3，而是一条SAM3 VLM Agent 人工门禁联合流水线。SAM3负责画轮廓，VLM负责识别画面中出现的语义和风险，Agent按照保守规则修改或转人工，最终只有人工确认的结果才能成为真值或训练输入。")
    add_bullets(doc, [
        "工厂数据进入后，系统先识别具体数据文件、相机、帧和载入代次，避免旧窗口写到新数据。",
        "参考帧准备阶段会生成场景描述并预取类别；正式点击时复用结果，减少重复等待。",
        "候选先显示在原图上，后台VLM与Agent继续复核，用户不需要一直看着0%的进度。",
        "任何机器候选默认都是candidate only，不会自动冒充人工完成，也不会自动进入训练。",
        "训练集与测试集永久隔离，test24只用于评测，不能因效果不佳回流调参。",
    ])
    add_figure(doc, speed_chart(), "图 2  双卡5090小图VLM通道的真实配对耗时")

    doc.add_heading("二 工作台操作说明", level=1)
    doc.add_heading("二 一 打开并检查工作台", level=2)
    add_steps(doc, [
        "打开 http://127.0.0.1:8765。顶部应显示本机运行，页面可以看到人工标注真值和自动标注候选两个主入口。",
        "点击右上角文件夹或切换数据，确认当前数据名称。旧验证集会标成CVAT Task3，新实采数据会标成Capture 20261103，1104批次003和004只标为问题包单图抽样。",
        "选择相机和帧。快速拖动时间轴时以最后一次选择为准，旧图片和旧请求不会覆盖当前帧。",
        "运行前检查图像、参考帧、场景描述、当前帧号和本次处理数量。场景描述过期时先等待自动恢复。",
    ])
    add_figure(doc, ROOT / "artifacts/dataset-coverage-browser-acceptance-round324-current/new-dataset-class-coverage.png",
               "图 3  新数据集与逐类别覆盖页面")
    add_figure(doc, ROOT / "artifacts/dataset-coverage-browser-acceptance-round324-current/manual-inference-shortcut.png",
               "图 4  从人工复核区返回自动推理的快捷入口")

    doc.add_heading("二 二 运行一帧或多帧自动标注", level=2)
    add_steps(doc, [
        "先选择参考帧。参考帧必须经过当前人工复核门禁，历史标注不能直接当作有效参考。",
        "在自动标注候选页选择开始帧与处理数量。演示时建议先处理1帧，确认结果后再增加轮数或帧数。",
        "点击运行后，候选会先发布并叠加到原图，VLM和Agent后台继续复核。页面会显示当前阶段，而不是长时间停留在0。",
        "查看SAM3分割、GPT复查输入、GPT决策输出和Agent标注结果。没有真实调用时页面会明确写暂无记录，不再用空数组伪装成功。",
        "对候选进行人工复核。只有保存当前真值并通过人工门禁后，才可能成为参考帧或后续训练输入。",
    ])
    add_para(doc, "轮数怎么理解：", bold_lead="轮数怎么理解：")
    add_para(doc, "推理轮数表示Agent最多可以观察和修正多少次，不等于一定会执行满。重复无效操作、退化多边形或结果已经稳定时会提前停止。演示建议1至2轮，复杂遮挡可增加，但轮数越多等待越长，也可能增加不必要修改，因此仍由安全门禁限制。")

    doc.add_heading("二 三 如何判断结果是否可信", level=2)
    add_table(doc, ["页面状态", "含义", "应做什么"], [
        ["机器候选", "SAM3和Agent产生的建议", "查看轮廓与类别 不直接当真值"],
        ["需复核", "类别盲审或保护规则发现风险", "人工判断保留 删除或改类"],
        ["人工真值", "已经由人工保存的标注", "仍要看当前复核状态是否有效"],
        ["参考帧可用", "真值和当前人工复核回执匹配", "可以用于后续自动起稿"],
        ["训练批准", "训练门禁全部通过", "当前大多数机器候选并未达到"],
    ], widths=[1.2, 2.5, 2.8])

    doc.add_heading("二 四 常见故障处理", level=2)
    add_table(doc, ["现象", "现在的解释", "处理方式"], [
        ["Local VLM is busy", "共享模型槽正在处理其他请求", "保持页面等待 系统会排队重试 不要连续重复点击"],
        ["进度长期为0", "旧版本可能未确认任务或模型忙", "当前版本会显示检查 提交 发现 生成 复查等阶段 若无变化先刷新健康状态"],
        ["结果不在原图上", "可能是历史结果读回或画布坐标失配", "当前版本会在载入后读回当前帧并重新绘制 拖动后仍应贴合"],
        ["GPT输入或输出为空", "该帧没有实际进入对应阶段或记录过期", "重新运行当前帧 页面应显示完整Prompt 图像和结构化输出"],
        ["切数据后看到旧框", "旧异步请求返回过晚", "当前版本按载入代次和最后一次选择拒绝旧结果"],
        ["磁盘不足", "运行前空间门禁阻止长批次", "减少帧数 清理可恢复缓存 不删除真值与原始数据"],
    ], widths=[1.35, 2.5, 2.65])

    doc.add_heading("三 过去主要改动与作用", level=1)
    doc.add_heading("三 一 工作台稳定性", level=2)
    add_bullets(doc, [
        "按需只加载当前帧，避免一次把整段几何塞进浏览器。效果是进入标注模式和翻帧明显更快。",
        "所有数据加载、跳帧、区域编辑、清空候选、保存真值、设置参考帧和停止任务都绑定文件载入代次。效果是旧窗口无法写入新数据。",
        "快速拖动时间轴采用最后一次操作生效。效果是网络乱序时最终画面仍停在用户最后选中的帧。",
        "任务号、状态、失败重试、结果恢复和页面轮询全部绑定同一任务。效果是旧任务不再覆盖新任务，也不会把历史失败当成当前失败。",
        "候选先显示、后台复核。效果是用户先看到可用结果，再等待VLM和Agent长尾，而不是一直卡在0。",
        "拖动分栏和画布缩放后重新计算坐标并读回原图。实测最大叠加偏差0.015625像素。",
    ])

    doc.add_heading("三 二 VLM速度和可用性", level=2)
    add_bullets(doc, [
        "增加精确响应缓存、跨重启缓存、拥塞计数和固定推理线程，减少重复请求与并发崩溃。",
        "参考帧停留预取、相邻帧预取和最新的服务端发现结果复用，正式点击时不再重复做同一场景发现。",
        "双卡5090增加独立NF4小图通道，主4090保留为稳定回退。camera1总耗时下降71.93%，camera2下降81.30%。",
        "1920乘1536大图5090路线虽然更快，但改变了高风险人体候选的复核路径，因此明确拒绝部署。",
        "Local VLM busy从直接失败改成可见的排队等待，并保留主模型回退。",
    ])
    add_figure(doc, ROOT / "artifacts/new-dataset-fast-vlm5090-round330/camera1.png",
               "图 5  camera1使用5090小图通道后的工作台结果")
    add_figure(doc, ROOT / "artifacts/new-dataset-fast-vlm5090-round330/camera2.png",
               "图 6  camera2使用5090小图通道后的工作台结果")

    doc.add_heading("三 三 数据集与类别覆盖", level=2)
    add_bullets(doc, [
        "文件列表增加明确数据身份，不再把老CVAT camera0至3和新Capture相机混在一起。",
        "批次B 003和004只作为单图问题包抽样展示，不冒充完整MCAP或完整标注集。",
        "逐类别卡片显示已生成数量与需复核数量，可以直接看哪些已覆盖、哪些仍需要人工。",
        "没有参考帧时，floor和slope的大范围扩张受到比例门禁，避免把人物或整片背景吞进地面。",
        "other robot只有在VLM明确发现且基线SAM没有结果时才触发补漏词，结果仍走高风险盲审。",
    ])

    doc.add_heading("三 四 栈板轮廓", level=2)
    add_bullets(doc, [
        "v1至v2解决普通纹理小孔，超预算时只保留安全填孔，不强行平滑。",
        "v4针对近景木栈板密集木纹碎洞，减少内部零碎空洞。",
        "v5加入图像引导的边缘凹口修补，并限制组件合并和总修改比例。",
        "长叉孔、板缝和塑料栈板网格作为结构孔保留，防止把真实结构抹平。",
        "新推理已自动使用v5；历史候选尚有1,040个需要可回滚迁移。",
    ])
    add_figure(doc, ROOT / "artifacts/pallet-current-production-round298/frame-110-result-overlay.png",
               "图 7  当前生产第110帧栈板候选叠加")
    add_figure(doc, ROOT / "artifacts/pallet-boundary-live-round334/before-after.jpg",
               "图 8  历史栈板候选使用v5前后对照",
               "只读重放结果 未修改人工真值或训练批准")

    doc.add_heading("四 斜坡 栈板和其他机器人专项结果", level=1)
    doc.add_heading("四 一 斜坡不依赖颜色", level=2)
    add_para(doc, "斜坡判断不是寻找橙色区域。当前链路综合使用VLM场景判断、SAM3文本或点提示、画面空间位置、与floor的连接关系、候选宽高比例、ego重叠和必要时的LiDAR几何证据。颜色只能作为图像内容的一部分，不能单独决定类别。")
    add_para(doc, "项目约定中，slope最终可以和floor统一成可通行区域，但发现阶段仍分别保留类别。这样既能满足通行规划，也能继续评估坡板召回和误检。")
    add_bullets(doc, [
        "camera0真实坡保持不变。",
        "camera3假坡99.9%被高置信ego覆盖，已移出自动叠加并转人工复核。",
        "VLM对边缘局部坡板仍可能漏报，空间点提示可以让SAM3正确分割，主要瓶颈是自动定位而非SAM3本体。",
        "更激进的全局斜坡提示虽然提升召回，但把普通地面误报成坡，因此未部署。",
    ])
    add_figure(doc, ROOT / "artifacts/new-dataset-fast-vlm5090-slope-round331/camera0.png",
               "图 9  camera0真实坡保留结果")
    add_figure(doc, ROOT / "artifacts/new-dataset-fast-vlm5090-slope-round331/camera3.png",
               "图 10  camera3假坡保护后的结果")

    doc.add_heading("四 二 other robot补漏", level=2)
    add_para(doc, "基线robot提示对远处搬运机器人召回不足。系统现在只在VLM明确判断存在other robot且基线SAM为零时，追加autonomous pallet truck提示；最多保留SAM分数最高的3个，再由8B做高风险盲审。")
    add_figure(doc, ROOT / "artifacts/other-robot-fallback-round310-positive-v2/00-1789897560182495905-fallback.jpg",
               "图 11  other robot补漏候选示例",
               "补漏候选不会自动晋级为真值")

    doc.add_heading("五 微调与后训练现状", level=1)
    add_figure(doc, training_chart(), "图 12  模型训练与部署状态")
    doc.add_heading("五 一 VLM后训练已经真实执行", level=2)
    add_bullets(doc, [
        "train72与test24已在CVAT创建，分别为72张和24张，文件名与内容哈希零交集。",
        "train72先采集8B机器教师候选；70条可用，2条cargo floor冲突留在复核队列。",
        "2B Qwen3 VL LoRA完成两步冒烟、一轮62步和两轮62步训练，权重张量审计确认真实更新。",
        "一轮LoRA在dev8上从基础模型0比8合法结构化输出提升到8比8；两轮候选在冻结test24上与8B教师180比192一致，生成中位约6.54秒。",
        "但slope只有18比24教师一致，且机器教师一致率不是人工准确率，因此2B候选没有部署。",
    ])
    doc.add_heading("五 二 SAM3微调尚未启动", level=2)
    add_para(doc, "SAM3当前使用正式预训练模型与工程级提示、裁剪、几何门禁和后处理。真正的SAM3微调需要人工确认的像素掩码，而当前96张CVAT任务仍缺少完整人工COCO导出和逐图签核。为了避免机器预标注自我强化，训练门禁保持关闭。")
    add_para(doc, "准确结论是：我们做过VLM LoRA后训练，但未部署；没有完成SAM3像素微调。下一步先完成人工复核与导出，再做只解冻mask decoder或LoRA式小规模适配，test24永久只评测。")

    doc.add_heading("六 新数据集与跨数据验收", level=1)
    add_bullets(doc, [
        "Capture 20261103真实MCAP包含152帧和四路相机，当前版本四路均能完成SAM3 VLM Agent 原图叠加闭环。",
        "批次B 003和004单图抽样已加入正式文件列表，分别得到17个和14个候选，并完成八类反馈。",
        "新数据的候选仍保持candidate only，不写人工真值，不自动进入训练。",
        "正式浏览器验收确认文件库10项可见，149帧存在机器候选，逐类别状态完整，页面和HTTP错误为零。",
    ])
    for idx, camera in enumerate((0, 1, 2, 3), 13):
        source = ROOT / f"artifacts/new-dataset-cross-view-round326/camera{camera}.png"
        add_figure(doc, source, f"图 {idx}  Capture新数据camera{camera}端到端结果")

    doc.add_heading("七 已否决路线与保留原因", level=1)
    add_table(doc, ["尝试", "结果", "为什么不部署"], [
        ["大图5090快速路由", "55.813秒降至36.093秒", "人体控制台混合候选从需复核变成直接保留 安全路线改变"],
        ["更激进的斜坡VLM提示", "召回25%升至75%", "精确率降至50% 普通地面误报"],
        ["多图static cache", "部分请求更快", "语义不完全一致或触发超长动态回退"],
        ["batch等于2类别盲审", "一帧节省约1.75秒", "另一帧把困难栈板从人工复核改成高置信保留"],
        ["远处栈板ROI扩张", "少数候选面积变大", "会吞入背景或相邻栈板 最终安全替换0比16"],
        ["相邻帧光流补全", "运输515个候选", "没有候选同时满足面积 主轴和相邻实例门禁"],
        ["LiDAR点提示补远栈板", "7次正常返回", "高分候选仍扩成竖直背景 安全替换0比4"],
        ["2B LoRA直接替换8B", "速度候选通过", "斜坡和other robot外部门禁未通过"],
    ], widths=[1.7, 2.0, 2.9])

    doc.add_heading("八 当前限制和下一步计划", level=1)
    doc.add_heading("八 一 当前限制", level=2)
    add_bullets(doc, [
        "历史旧帧仍可能显示v5之前的栈板轮廓，不能据此判断新推理算法没有生效。",
        "SAM3尚未经过项目人工像素真值微调；目前改善主要来自提示、ROI、几何门禁和轮廓清理。",
        "2B VLM后训练候选没有部署，正式发现模型仍以8B为主。",
        "斜坡的边缘局部视角、远处栈板完整性和远处其他机器人仍需要更多人工真值。",
        "磁盘空间曾多次接近门禁，已经通过可逆压缩和回放迁移恢复，但长批次前仍要先检查空间。",
    ])
    doc.add_heading("八 二 已探索方法总览", level=2)
    add_table(doc, ["方向", "已经尝试的做法", "当前结论"], [
        ["SAM3提示", "多同义文本 点提示 网格提示 正负点", "点提示有效 自动空间定位仍需加强"],
        ["ROI", "单栈板裁剪 多尺度裁剪 四窗口坡板回退", "近景有效 远处容易吞背景 必须保留几何门禁"],
        ["时序", "相邻帧光流 双向一致性", "可运输候选 但远处栈板没有通过完整性门禁"],
        ["点云", "LiDAR去地面 低矮点正提示 地面负提示", "提示生成可行 第一版安全扩展0比4 未部署"],
        ["VLM", "8B发现 逐候选盲审 2B LoRA后训练", "8B正式 2B速度候选因斜坡召回不足未部署"],
        ["几何清理", "孔洞填充 轮廓简化 图像引导凹口修复", "v5已用于新推理 历史候选待迁移"],
        ["跨相机", "四相机独立验收 平面投影诊断", "适合做一致性证据 不能用错误标定强行合并"],
        ["开放类别", "VLM新label发现 other robot补漏", "可以发现新语义 仍需人工门禁后才能入训练"],
    ], widths=[1.0, 3.0, 2.6])

    doc.add_heading("八 三 点云优化方案", level=2)
    add_para(doc, "点云值得继续做，但它的角色是给SAM3与VLM提供空间约束，而不是直接替代图像分割。我们已经用LiDAR、标定和冻结候选生成过正负提示：斜坡开发集出现过可用候选，远处栈板第一版则因为背景扩张而没有通过安全门禁。下一版必须先提升点选择与实例边界，而不是放宽SAM分数。")
    add_bullets(doc, [
        "栈板：去除主地面，在低高度带聚类；聚类核心投影为正点，地面、相邻栈板和竖直背景投影为负点。",
        "斜坡：计算局部平面法向、连续高度梯度、平面残差和与地面的连通性；只有多项一致时才给SAM提示，完全不依赖颜色。",
        "其他机器人：结合离地动态点簇、相邻帧运动和ego包围区排除形成ROI，再由VLM确认类别。",
        "跨相机：同一三维点投影到四路相机时检查一致性，避免单相机局部遮挡造成误判。",
        "时序：涉及已知里程中断的批次B 003与004时必须单独标记，不自动修补异常位姿后混入训练。",
        "验收：必须在独立人工真值上同时改善召回和误检，并保持候选ID、真值与训练隔离。",
    ])
    doc.add_heading("八 四 下一步执行顺序", level=2)
    add_steps(doc, [
        "在生产状态副本上执行历史栈板候选全量迁移，逐字段验证候选ID 类别 真值 参考帧和训练批准不变。",
        "备份正式状态后迁移1至3个高变化帧并人工查看，再分批迁移其余旧候选；每帧都保留旧版本和回滚记录。",
        "完成train72与test24人工复核、导出和签核；test24永久只评测。",
        "基于人工像素真值做SAM3小规模微调，优先mask decoder或轻量适配，使用栈板边界和坡板困难样本。",
        "继续保持4090稳定主路由与5090小图快速通道，任何新加速只有语义和复核路径完全一致才部署。",
        "将工厂实时数据接入数据集识别、自动标注、新类别VLM发现、Agent复核、人工门禁、训练评测和灰度回灌，形成完整闭环。",
    ])

    doc.add_page_break()
    doc.add_heading("九 全部图像证据附录", level=1)
    add_para(doc, "本附录收录项目进展记录中已经引用的全部33张历史图像，并补充本说明书正文中的最新图像。图像中的机器候选不等同于人工真值；部分图片用于记录被否决的实验路线。")
    used = set()
    first_appendix_figure = True
    for _, path in linked_images():
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        if digest in used:
            continue
        used.add(digest)
        add_figure(doc, path, f"附图  {path.name}",
                   page_break=not first_appendix_figure)
        first_appendix_figure = False

    footer = section.footer
    p = footer.paragraphs[0]
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    run = p.add_run("LabelAgent 项目进展与使用说明书  2026 11 20")
    set_run_font(run, size=8, color="666666")

    output = OUT / "LabelAgent_项目进展与使用说明书_20261120.docx"
    doc.save(output)
    return output


if __name__ == "__main__":
    print(build())
