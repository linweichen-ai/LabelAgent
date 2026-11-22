#!/usr/bin/env python3
"""Build an IEEE Transactions-style internal LabelAgent manuscript."""

from __future__ import annotations

from pathlib import Path

from PIL import Image, ImageDraw, ImageFont
from docx import Document
from docx.enum.section import WD_SECTION
from docx.enum.table import WD_CELL_VERTICAL_ALIGNMENT, WD_TABLE_ALIGNMENT
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from docx.shared import Inches, Pt, RGBColor


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "docs" / "deliverables"
ASSETS = OUT / "ieee-assets"
OUTPUT = OUT / "LabelAgent_IEEE_Transactions_Style_Manuscript_20261120.docx"


def set_font(run, size=10, bold=False, italic=False, name="Times New Roman"):
    run.font.name = name
    run._element.rPr.rFonts.set(qn("w:eastAsia"), name)
    run.font.size = Pt(size)
    run.font.bold = bold
    run.font.italic = italic
    run.font.color.rgb = RGBColor(0, 0, 0)


def set_columns(section, number=2, space_twips=360):
    sect_pr = section._sectPr
    existing = sect_pr.find(qn("w:cols"))
    if existing is None:
        existing = OxmlElement("w:cols")
        sect_pr.append(existing)
    existing.set(qn("w:num"), str(number))
    existing.set(qn("w:space"), str(space_twips))
    existing.set(qn("w:equalWidth"), "1")


def add_page_number(paragraph):
    paragraph.alignment = WD_ALIGN_PARAGRAPH.CENTER
    run = paragraph.add_run()
    begin = OxmlElement("w:fldChar")
    begin.set(qn("w:fldCharType"), "begin")
    instruction = OxmlElement("w:instrText")
    instruction.set(qn("xml:space"), "preserve")
    instruction.text = "PAGE"
    end = OxmlElement("w:fldChar")
    end.set(qn("w:fldCharType"), "end")
    run._r.extend([begin, instruction, end])
    set_font(run, 8)


def keep(paragraph, next_only=False):
    ppr = paragraph._p.get_or_add_pPr()
    node = OxmlElement("w:keepNext" if next_only else "w:keepLines")
    ppr.append(node)


def body(doc, text, *, indent=True):
    p = doc.add_paragraph()
    p.alignment = WD_ALIGN_PARAGRAPH.JUSTIFY
    p.paragraph_format.space_before = Pt(0)
    p.paragraph_format.space_after = Pt(0)
    p.paragraph_format.line_spacing = 1.0
    if indent:
        p.paragraph_format.first_line_indent = Inches(0.14)
    set_font(p.add_run(text), 9.5)
    keep(p)
    return p


def primary(doc, number, title):
    p = doc.add_paragraph()
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p.paragraph_format.space_before = Pt(6)
    p.paragraph_format.space_after = Pt(2)
    prefix = f"{number}. " if number else ""
    set_font(p.add_run(f"{prefix}{title.upper()}"), 10, bold=False)
    keep(p, next_only=True)
    return p


def secondary(doc, letter, title):
    p = doc.add_paragraph()
    p.paragraph_format.space_before = Pt(4)
    p.paragraph_format.space_after = Pt(1)
    set_font(p.add_run(f"{letter}. {title}"), 10, italic=True)
    keep(p, next_only=True)
    return p


def caption(doc, text, *, table=False):
    p = doc.add_paragraph()
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p.paragraph_format.space_before = Pt(1 if not table else 3)
    p.paragraph_format.space_after = Pt(3 if not table else 1)
    set_font(p.add_run(text), 8)
    keep(p, next_only=table)
    return p


def figure(doc, path, label, width=3.35):
    p = doc.add_paragraph()
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p.paragraph_format.space_before = Pt(3)
    p.paragraph_format.space_after = Pt(0)
    p.add_run().add_picture(str(path), width=Inches(width))
    keep(p, next_only=True)
    caption(doc, label)


def table(doc, label, headers, rows, widths=None):
    caption(doc, label, table=True)
    t = doc.add_table(rows=1, cols=len(headers))
    t.alignment = WD_TABLE_ALIGNMENT.CENTER
    t.autofit = False
    layout = OxmlElement("w:tblLayout")
    layout.set(qn("w:type"), "fixed")
    t._tbl.tblPr.append(layout)
    if widths:
        for column, width in zip(t.columns, widths):
            column.width = Inches(width)
    for index, value in enumerate(headers):
        cell = t.rows[0].cells[index]
        cell.vertical_alignment = WD_CELL_VERTICAL_ALIGNMENT.CENTER
        cell.text = ""
        p = cell.paragraphs[0]
        p.alignment = WD_ALIGN_PARAGRAPH.CENTER
        set_font(p.add_run(value), 7.4, bold=True)
        if widths:
            cell.width = Inches(widths[index])
    for row in rows:
        cells = t.add_row().cells
        for index, value in enumerate(row):
            cells[index].vertical_alignment = WD_CELL_VERTICAL_ALIGNMENT.CENTER
            cells[index].text = ""
            p = cells[index].paragraphs[0]
            p.alignment = WD_ALIGN_PARAGRAPH.LEFT
            p.paragraph_format.space_after = Pt(0)
            set_font(p.add_run(str(value)), 7.2)
            if widths:
                cells[index].width = Inches(widths[index])
    for row in t.rows:
        for cell in row.cells:
            tc_pr = cell._tc.get_or_add_tcPr()
            margins = tc_pr.first_child_found_in("w:tcMar")
            if margins is None:
                margins = OxmlElement("w:tcMar")
                tc_pr.append(margins)
            for side in ("top", "left", "bottom", "right"):
                node = OxmlElement(f"w:{side}")
                node.set(qn("w:w"), "45")
                node.set(qn("w:type"), "dxa")
                margins.append(node)
    return t


def switch_columns(doc, number):
    section = doc.add_section(WD_SECTION.CONTINUOUS)
    section.page_width = Inches(8.5)
    section.page_height = Inches(11)
    section.top_margin = Inches(0.62)
    section.bottom_margin = Inches(0.62)
    section.left_margin = Inches(0.65)
    section.right_margin = Inches(0.65)
    section.header.is_linked_to_previous = True
    section.footer.is_linked_to_previous = True
    set_columns(section, number, 360)
    return section


def wide_table(doc, label, headers, rows, widths):
    switch_columns(doc, 1)
    result = table(doc, label, headers, rows, widths)
    switch_columns(doc, 2)
    return result


def _font(size, bold=False):
    candidates = [
        Path(r"C:\Windows\Fonts\timesbd.ttf" if bold else r"C:\Windows\Fonts\times.ttf"),
        Path(r"C:\Windows\Fonts\arialbd.ttf" if bold else r"C:\Windows\Fonts\arial.ttf"),
    ]
    for path in candidates:
        if path.exists():
            return ImageFont.truetype(str(path), size)
    return ImageFont.load_default()


def make_assets():
    ASSETS.mkdir(parents=True, exist_ok=True)
    # Slope true/false pair.
    left = Image.open(ROOT / "artifacts/new-dataset-fast-vlm5090-slope-round331/camera0.png").convert("RGB")
    right = Image.open(ROOT / "artifacts/new-dataset-fast-vlm5090-slope-round331/camera3.png").convert("RGB")
    target_h = 720
    panels = []
    for image in (left, right):
        ratio = target_h / image.height
        panels.append(image.resize((int(image.width * ratio), target_h), Image.Resampling.LANCZOS))
    slope = Image.new("RGB", (sum(i.width for i in panels), target_h), "white")
    x = 0
    for image in panels:
        slope.paste(image, (x, 0)); x += image.width
    slope_path = ASSETS / "slope-true-false.png"
    slope.save(slope_path, optimize=True)

    # Round335 frame89 panel: points/current/proposed/difference.
    review = Image.open(ROOT / "artifacts/pallet-lidar-instance-sam-round335/instance-lidar-sam-review.jpg").convert("RGB")
    panel_h = review.height // 4
    lidar = review.crop((0, panel_h * 2, review.width, panel_h * 3))
    lidar_path = ASSETS / "lidar-frame89-panel.png"
    lidar.save(lidar_path, optimize=True)

    # English closed-loop architecture.
    architecture = Image.new("RGB", (1600, 620), "white")
    draw = ImageDraw.Draw(architecture)
    title_font, box_font, small_font = _font(42, True), _font(27, True), _font(21)
    draw.text((55, 25), "Safety-Gated Closed-Loop Annotation", fill=(15, 45, 70), font=title_font)
    labels = [
        ("Factory Data", "Images, LiDAR, calibration"),
        ("Dataset Identity", "File, camera, frame binding"),
        ("Promptable Masks", "SAM3 candidate generation"),
        ("Semantic Review", "VLM label and risk discovery"),
        ("Agent Action", "Bounded edit or review queue"),
        ("Human Gate", "Confirm, relabel, or reject"),
        ("Train and Test", "Audited train; frozen test"),
        ("Staged Return", "Deploy only after gates"),
    ]
    positions = []
    for row in range(2):
        for col in range(4):
            x, y = 55 + col * 385, 115 + row * 225
            positions.append((x, y, x + 330, y + 145))
    for index, ((heading, detail), box) in enumerate(zip(labels, positions)):
        fill = (232, 242, 249) if index < 6 else (238, 247, 234)
        draw.rounded_rectangle(box, radius=18, fill=fill, outline=(22, 91, 132), width=4)
        draw.text((box[0] + 18, box[1] + 24), heading, fill=(15, 45, 70), font=box_font)
        draw.text((box[0] + 18, box[1] + 82), detail, fill=(55, 70, 80), font=small_font)
    for row in range(2):
        for col in range(3):
            a, b = positions[row * 4 + col], positions[row * 4 + col + 1]
            y = (a[1] + a[3]) // 2
            draw.line((a[2] + 8, y, b[0] - 12, y), fill=(22, 91, 132), width=5)
            draw.polygon([(b[0] - 12, y), (b[0] - 30, y - 10), (b[0] - 30, y + 10)], fill=(22, 91, 132))
    draw.line((220, 520, 1380, 520), fill=(43, 125, 65), width=5)
    draw.polygon([(220, 520), (244, 506), (244, 534)], fill=(43, 125, 65))
    draw.text((555, 540), "Reviewed data returns only through training gates", fill=(43, 125, 65), font=small_font)
    architecture_path = ASSETS / "ieee-architecture.png"
    architecture.save(architecture_path, optimize=True)

    # English latency chart.
    speed = Image.new("RGB", (1200, 520), "white")
    draw = ImageDraw.Draw(speed)
    draw.text((35, 25), "Paired VLM Latency on Dual RTX 5090", fill=(15, 45, 70), font=_font(34, True))
    values = [("Camera 1", 91.500, 25.688), ("Camera 2", 70.204, 13.125)]
    scale = 7.6
    for row, (name, before, after) in enumerate(values):
        y = 135 + row * 170
        draw.text((35, y + 30), name, fill=(20, 30, 40), font=_font(25, True))
        draw.rounded_rectangle((190, y, 190 + before * scale, y + 46), radius=8, fill=(170, 195, 215))
        draw.rounded_rectangle((190, y + 70, 190 + after * scale, y + 116), radius=8, fill=(22, 116, 176))
        draw.text((205 + before * scale, y + 8), f"Before {before:.3f} s", fill=(30, 40, 50), font=_font(21))
        draw.text((205 + after * scale, y + 78), f"After {after:.3f} s", fill=(30, 40, 50), font=_font(21))
    speed_path = ASSETS / "ieee-vlm-speed.png"
    speed.save(speed_path, optimize=True)

    training = Image.new("RGB", (1200, 470), "white")
    draw = ImageDraw.Draw(training)
    draw.text((35, 20), "Training and Deployment State", fill=(15, 45, 70), font=_font(34, True))
    rows = [
        ("2B VLM LoRA", "Training completed", "Not deployed", (238, 246, 252)),
        ("8B VLM", "Stable production route", "Deployed", (235, 247, 237)),
        ("SAM3 pixel adaptation", "Reviewed masks incomplete", "Not started", (252, 244, 230)),
        ("Historical pallet migration", "Read-only audit passed", "Not yet applied", (245, 240, 250)),
    ]
    for index, (name, state, decision, fill) in enumerate(rows):
        y = 95 + index * 86
        draw.rounded_rectangle((35, y, 1165, y + 66), radius=12, fill=fill, outline=(160, 170, 180), width=2)
        draw.text((55, y + 17), name, fill=(20, 30, 40), font=_font(23, True))
        draw.text((390, y + 17), state, fill=(45, 55, 65), font=_font(22))
        draw.text((930, y + 17), decision, fill=(120, 70, 10) if "Not" in decision else (25, 115, 55), font=_font(22, True))
    training_path = ASSETS / "ieee-training-status.png"
    training.save(training_path, optimize=True)
    return slope_path, lidar_path, architecture_path, speed_path, training_path


def build():
    slope_path, lidar_path, architecture_path, speed_path, training_path = make_assets()
    doc = Document()
    first = doc.sections[0]
    first.page_width = Inches(8.5)
    first.page_height = Inches(11)
    first.top_margin = Inches(0.62)
    first.bottom_margin = Inches(0.62)
    first.left_margin = Inches(0.65)
    first.right_margin = Inches(0.65)
    set_columns(first, 1)
    for section in doc.sections:
        section.header_distance = Inches(0.25)
        section.footer_distance = Inches(0.28)

    normal = doc.styles["Normal"]
    normal.font.name = "Times New Roman"
    normal._element.rPr.rFonts.set(qn("w:eastAsia"), "Times New Roman")
    normal.font.size = Pt(9.5)
    normal.font.color.rgb = RGBColor(0, 0, 0)

    header = first.header.paragraphs[0]
    header.alignment = WD_ALIGN_PARAGRAPH.CENTER
    set_font(header.add_run("IEEE TRANSACTIONS-STYLE INTERNAL MANUSCRIPT — NOVEMBER 2026"), 7.5)
    add_page_number(first.footer.paragraphs[0])

    title = doc.add_paragraph()
    title.alignment = WD_ALIGN_PARAGRAPH.CENTER
    title.paragraph_format.space_after = Pt(5)
    set_font(title.add_run(
        "LabelAgent: A Safety-Gated Multimodal Pipeline for Closed-Loop Industrial Image Annotation"
    ), 20)
    authors = doc.add_paragraph()
    authors.alignment = WD_ALIGN_PARAGRAPH.CENTER
    authors.paragraph_format.space_after = Pt(1)
    set_font(authors.add_run("Anonymous Authors for Internal Review"), 11)
    affiliation = doc.add_paragraph()
    affiliation.alignment = WD_ALIGN_PARAGRAPH.CENTER
    affiliation.paragraph_format.space_after = Pt(6)
    set_font(affiliation.add_run(
        "Author names and affiliations are intentionally withheld in this internal manuscript."
    ), 8, italic=True)

    abstract = doc.add_paragraph()
    abstract.alignment = WD_ALIGN_PARAGRAPH.JUSTIFY
    abstract.paragraph_format.left_indent = Inches(0.35)
    abstract.paragraph_format.right_indent = Inches(0.35)
    abstract.paragraph_format.space_after = Pt(3)
    set_font(abstract.add_run("Abstract—"), 9, bold=True, italic=True)
    set_font(abstract.add_run(
        "Industrial image annotation systems must combine open-vocabulary recognition, precise masks, "
        "sensor geometry, operator review, and reproducible model updates without allowing machine "
        "candidates to silently become training truth. This paper presents LabelAgent, a safety-gated "
        "pipeline that couples promptable segmentation, a vision-language model, an execution agent, "
        "LiDAR-derived geometric cues, and explicit human approval. The deployed workbench binds every "
        "asynchronous result to dataset identity, camera, frame, file-generation state, and task state. "
        "On paired dual-RTX-5090 trials, the small-image vision-language path reduced end-to-end latency "
        "from 91.500 to 25.688 s on camera 1 and from 70.204 to 13.125 s on camera 2, while preserving "
        "the reviewed decision path. A conservative pallet-boundary procedure passed a 150-frame "
        "candidate-only audit with zero policy violations. For difficult distant pallets, a 3-D "
        "instance-aware LiDAR prompt reduced a known vertical-background mask from 7,697 to 583 pixels, "
        "but no candidate passed the complete automatic replacement gate. These results demonstrate "
        "improved responsiveness and failure containment rather than production-level segmentation "
        "accuracy. The system therefore retains human truth, frozen evaluation data, and deployment "
        "approval as separate states and defines a path toward reviewed post-training and closed-loop use."
    ), 9, italic=True)

    terms = doc.add_paragraph()
    terms.alignment = WD_ALIGN_PARAGRAPH.JUSTIFY
    terms.paragraph_format.left_indent = Inches(0.35)
    terms.paragraph_format.right_indent = Inches(0.35)
    terms.paragraph_format.space_after = Pt(3)
    set_font(terms.add_run("Index Terms—"), 9, bold=True, italic=True)
    set_font(terms.add_run(
        "Automated annotation, closed-loop learning, industrial perception, multimodal fusion, "
        "promptable segmentation, safety gates, vision-language models."
    ), 9, italic=True)

    note = doc.add_paragraph()
    note.alignment = WD_ALIGN_PARAGRAPH.JUSTIFY
    note.paragraph_format.left_indent = Inches(0.35)
    note.paragraph_format.right_indent = Inches(0.35)
    note.paragraph_format.space_after = Pt(5)
    set_font(note.add_run("Note to Practitioners—"), 9, bold=True, italic=True)
    set_font(note.add_run(
        "The main engineering lesson is that a fast mask is not automatically a valid annotation. "
        "LabelAgent exposes intermediate evidence, queues model contention, rejects stale browser "
        "results, and requires explicit review before a candidate can become reference or training data. "
        "LiDAR is used as a geometric constraint rather than an automatic semantic oracle."
    ), 9, italic=True)

    # Full-width overview before entering the two-column body.
    p = doc.add_paragraph()
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p.add_run().add_picture(str(architecture_path), width=Inches(6.7))
    caption(doc, "Fig. 1. Safety-gated LabelAgent loop. Machine candidates remain separated from human truth and approved training data.")

    body_section = doc.add_section(WD_SECTION.CONTINUOUS)
    body_section.page_width = Inches(8.5)
    body_section.page_height = Inches(11)
    body_section.top_margin = Inches(0.62)
    body_section.bottom_margin = Inches(0.62)
    body_section.left_margin = Inches(0.65)
    body_section.right_margin = Inches(0.65)
    body_section.header.is_linked_to_previous = True
    body_section.footer.is_linked_to_previous = True
    set_columns(body_section, 2, 360)

    primary(doc, "I", "Introduction")
    body(doc, "Foundation segmentation models can produce useful masks from text, point, box, or exemplar prompts [1]–[3]. In an industrial data pipeline, however, segmentation is only one part of the problem. A usable system must identify the active dataset and sensor view, recover from model contention, expose semantic review, preserve operator edits, and prevent machine output from being recycled as truth without approval.")
    body(doc, "LabelAgent addresses this broader systems problem. The pipeline combines a deployed promptable segmenter, an 8B vision-language model (VLM), a conservative execution agent, optional LiDAR geometry, and a human review gate. The workbench is designed for four-camera forklift data and open-set factory scenes containing pallets, persons, floor, slopes, ego structure, cargo-floor regions, vehicles, and other robots.")
    body(doc, "The contribution of this manuscript is an evidence-bounded engineering account. First, it describes task and file-generation binding that prevents stale asynchronous results from overwriting the active frame. Second, it reports paired latency improvements obtained with a dedicated dual-RTX-5090 small-image VLM route. Third, it evaluates conservative pallet contour cleanup and a 3-D instance-aware LiDAR prompt. Fourth, it records negative experiments and explains why they were not deployed. No result is presented as independent segmentation accuracy unless human pixel truth exists.")

    primary(doc, "II", "System Architecture")
    secondary(doc, "A", "Data and State Model")
    body(doc, "Each loaded asset is represented by a dataset identity, source file, camera topic, frame index, and monotonically changing file-generation token. Long-running SAM, VLM, and agent operations also carry a task identifier. A result is accepted only when these identifiers still match the active workbench. This contract handles rapid timeline dragging, data switching, retry, stop, and result recovery without allowing an older request to overwrite a newer selection.")
    secondary(doc, "B", "Segmentation, Semantic Review, and Agent Action")
    body(doc, "The segmenter proposes instance masks. The VLM discovers visible categories, explains spatial conflicts, and reviews high-risk candidates. The agent applies bounded edits or routes a candidate to review. Candidate-only, human-truth, approved-for-training, and deployed states are represented separately. Consequently, a plausible mask can be displayed immediately while semantic review continues, but it cannot silently become ground truth.")
    secondary(doc, "C", "Closed-Loop Training Boundary")
    body(doc, "The intended loop is factory acquisition, sensor and dataset identification, candidate generation, VLM discovery of known or unseen labels, agent modification, human review, audited training, frozen evaluation, and staged deployment. Simulation data may supplement rare slopes and occlusions, but simulated and real performance must be reported separately. The evaluation-only test24 split is never recycled into training.")

    figure(doc, ROOT / "artifacts/dataset-coverage-browser-acceptance-round324-current/new-dataset-class-coverage.png",
           "Fig. 2. Current workbench view with explicit dataset identity, per-class coverage, and candidate overlays.")

    primary(doc, "III", "Safety-Gated Implementation")
    secondary(doc, "A", "Asynchronous Consistency")
    body(doc, "Every destructive or state-changing action is bound to the active file generation. The same rule applies to candidate import, region editing, reference preparation, task stop, retry, and result recovery. Timeline navigation follows last-selection-wins semantics. Canvas coordinates are recomputed after layout changes, and the current frame is read back before overlay rendering.")
    secondary(doc, "B", "VLM Availability and Latency")
    body(doc, "The VLM path uses visible queueing instead of immediate failure when a shared model slot is busy. Exact response caches, adjacent-frame prefetch, reference-scene prefetch, and a dual-RTX-5090 NF4 small-image route reduce repeated computation. The stable RTX-4090 route remains available for fallback. A 1920 × 1536 fast route was rejected because it changed the review disposition of a human-console mixed candidate despite being faster.")
    figure(doc, speed_path,
           "Fig. 3. Paired small-image VLM latency before and after the dual-RTX-5090 route.")
    secondary(doc, "C", "Candidate and Truth Isolation")
    body(doc, "Machine masks are rendered as candidates and remain eligible for deletion or relabeling. Only explicit operator confirmation can create human truth. Any geometric post-processing that changes a saved candidate invalidates its previous class audit and reopens review. Migration tools operate frame by frame, archive the prior candidate revision, verify idempotence, and fail closed when dataset or artifact bindings differ.")

    wide_table(doc, "TABLE I\nDATASETS AND EVIDENCE BOUNDARIES",
          ["Source", "Use", "Boundary"], [
              ["Batch A, 12 complete dirs", "2,014 labelled frames", "Directory 009 excluded"],
              ["Capture 20261103", "152 frames, four cameras, LiDAR", "Candidates are not pixel truth"],
              ["Batch B 003/004", "Problem-package probes", "Odometry interruption flagged"],
              ["train72", "Post-training candidate pool", "Requires human approval"],
              ["test24", "Frozen evaluation", "Never used for tuning"],
          ], widths=[1.7, 2.25, 2.55])

    primary(doc, "IV", "Experimental Protocol")
    secondary(doc, "A", "Evaluation Philosophy")
    body(doc, "Engineering acceptance uses paired inputs, immutable hashes, browser readback, model health, and state-difference checks. Candidate-only experiments may measure latency, mask size, geometric consistency, or gate outcomes, but they do not claim precision, recall, or intersection-over-union without independent human truth. A method is deployed only when it preserves the existing safety path and passes both positive and hard-negative cases.")
    secondary(doc, "B", "Hardware and Services")
    body(doc, "The stable segmentation and VLM paths use dual RTX 4090 GPUs, while a dedicated dual RTX 5090 route accelerates bounded small-image VLM requests. GPU experiments are one-shot and release resources after completion. The browser workbench runs locally and uses separate model health endpoints. No long-running GPU occupancy process is required for the reported measurements.")
    secondary(doc, "C", "Metrics")
    body(doc, "Latency is measured from paired request start to candidate availability. Geometry audits report candidate counts, modification ratios, component merges, structural-hole retention, and policy violations. LiDAR experiments report prompt-ready targets, successful requests, HTTP safety rejections, selected mask pixels, and complete-gate acceptance. State invariance is verified by SHA-256 before and after an offline experiment.")

    primary(doc, "V", "Results")
    secondary(doc, "A", "Workbench Responsiveness")
    body(doc, "On paired camera-1 requests, the small-image VLM path reduced latency from 91.500 to 25.688 s, a 71.93% reduction. Camera 2 decreased from 70.204 to 13.125 s, an 81.30% reduction. Reference discovery prefetch required 17.405 s in the background, including 15.898 s for VLM discovery, after which the formal click-to-candidate delay was 2.622 s. The cache is bound to the image digest, labels, configuration, camera, frame, and file generation, and therefore does not reuse semantically mismatched results.")

    wide_table(doc, "TABLE II\nPAIRED LATENCY RESULTS",
          ["Path", "Before", "After", "Reduction"], [
              ["Camera 1 VLM", "91.500 s", "25.688 s", "71.93%"],
              ["Camera 2 VLM", "70.204 s", "13.125 s", "81.30%"],
              ["Prefetched formal click", "N/A", "2.622 s", "Bound cache hit"],
          ], widths=[1.8, 1.25, 1.25, 2.0])

    secondary(doc, "B", "Pallet Boundary Cleanup")
    body(doc, "The v5 pallet procedure fills texture holes and shallow image-supported notches while retaining long fork slots and structural gaps. A read-only audit of 150 production frames contained 1,316 candidates, including 1,126 pallets. Of these, 1,040 historical pallet candidates would change under v5. The maximum change ratio was 2.9408%, the minimum structural retention was 89.5911%, only three controlled component merges occurred, and zero policy violations were observed. This result validates migration safety constraints, not pixel accuracy.")
    figure(doc, ROOT / "artifacts/pallet-boundary-live-round334/before-after.jpg",
           "Fig. 4. Read-only examples of historical pallet candidates before and after v5 cleanup; no human truth was changed.")

    secondary(doc, "C", "Slope and Traversable Area")
    body(doc, "Slope discovery does not rely on color. The deployed path combines scene semantics, promptable masks, spatial position, floor connectivity, aspect ratio, ego overlap, and optional LiDAR geometry. A true ramp in camera 0 was preserved, while a camera-3 false slope with 99.9% high-confidence ego overlap was removed from automatic overlay and routed to review. Downstream planning may union slope and floor as traversable area, while evaluation preserves the slope label to measure difficult ramp behavior.")
    figure(doc, slope_path,
           "Fig. 5. Slope handling: a visible ramp is preserved (left), while an ego-overlap false ramp is suppressed (right).")

    secondary(doc, "D", "3-D Instance-Aware LiDAR Prompts")
    body(doc, "The first LiDAR pallet experiment used image-space connectivity for low-height projected points and produced a 7,697-pixel high-score mask that expanded into vertical background. The revised method uses 0.18-m connected components in the LiDAR XY plane, requires an unambiguous seed majority, and adds separate ground-ring and tall-background negative points. Three of four frozen targets produced prompts. Six of nine SAM calls returned masks and three were rejected with HTTP 400. No target passed the full automatic replacement gate.")
    body(doc, "For camera 1 frame 89, the selected conservative result decreased from 7,697 to 583 pixels (92.43%) while covering the positive point and excluding all ground and tall-background negatives. This is evidence of background-leak suppression, not evidence of higher segmentation accuracy. The candidate remains offline and requires pixel-level human review.")
    figure(doc, lidar_path,
           "Fig. 6. Camera-1 frame 89: 3-D-derived prompts, frozen candidate, selected SAM output, and difference view.")

    wide_table(doc, "TABLE III\nLIDAR PROMPT ABLATION ON FROZEN DIFFICULT PALLETS",
          ["Target", "Old px", "3-D px", "Change", "Full gate"], [
              ["Cam. 0, f39", "1,685", "1,426", "−15.37%", "Fail"],
              ["Cam. 0, f35", "2,181", "1,500", "−31.22%", "Fail"],
              ["Cam. 1, f89", "7,697", "583", "−92.43%", "Fail"],
              ["Cam. 2, f64", "No seed", "No seed", "Fail closed", "Fail"],
          ], widths=[1.35, 1.0, 1.0, 1.45, 1.3])

    secondary(doc, "E", "Post-Training Status")
    body(doc, "A 2B VLM LoRA path was executed on train72 candidates and improved legal structured output on a controlled development set. It was not deployed because slope agreement on the frozen test candidate set remained insufficient, and teacher agreement is not human accuracy. A separate SAM3 projection-filtered pseudo-truth run reached 81 optimizer steps, but frozen test24 mean union IoU decreased from 0.969289 to 0.963695 and one catastrophic frame was recorded; the candidate was stopped and not deployed. Human-reviewed pixel-mask fine-tuning has not started because the signed export remains incomplete. This distinction prevents pseudo-label adaptation from being misreported as validated human-supervised improvement.")
    figure(doc, training_path,
           "Fig. 7. Training and deployment state. Completed training does not imply deployment approval.")

    primary(doc, "VI", "Negative Results and Ablations")
    body(doc, "Negative experiments were retained because they define the safe operating region. A full-resolution RTX-5090 route reduced one request from 55.813 to 36.093 s but changed a high-risk review outcome and was rejected. More aggressive slope prompting increased development recall but produced ordinary-floor false positives. Temporal optical-flow transport moved 515 candidates but none satisfied area, principal-axis, and neighbor-instance gates. Distant-pallet ROI expansion achieved larger masks but accepted zero of 16 targets under the complete safety criteria.")
    wide_table(doc, "TABLE IV\nREJECTED OR NONDEPLOYED ROUTES",
          ["Route", "Observed benefit", "Blocking failure"], [
              ["Full-image 5090", "Lower latency", "Review path changed"],
              ["Aggressive slope prompt", "Higher dev recall", "Floor false positives"],
              ["Optical flow", "515 transported", "0 complete-gate passes"],
              ["Distant-pallet ROI", "Larger masks", "Background/neighbor leakage"],
              ["2B VLM LoRA", "Faster candidate", "Slope/open-set gate failed"],
          ], widths=[1.7, 2.1, 2.6])

    primary(doc, "VII", "Limitations and Future Work")
    body(doc, "The current evidence is strongest for workbench consistency, latency, and bounded candidate transformations. It is not an independent estimate of production segmentation accuracy. Several results use assistant-reviewed development examples, and historical candidates were generated under multiple versions. The next acceptance step is therefore a reviewed pixel-mask set that includes wood-grain pallet holes, thin distant pallets, partial ramps, ordinary-floor hard negatives, ego-overlap false slopes, and distant robots.")
    body(doc, "Point-cloud work will proceed in three stages. First, inference-only geometry will produce positive and negative prompts without changing model weights. Second, reviewed depth, height, and normal maps may be added to a lightweight mask-decoder adaptation. Third, temporally compensated clusters may be evaluated only on sequences with verified odometry. Batch-B packages 003 and 004 contain known odometry interruptions and must remain explicitly flagged rather than silently repaired into training data.")
    body(doc, "The deployment gate requires simultaneous improvement in reviewed recall and false positives, unchanged candidate and truth identities, stable cross-camera projection, bounded latency, and a reversible release. Until these conditions are met, the incumbent route remains active.")

    primary(doc, "VIII", "Conclusion")
    body(doc, "LabelAgent demonstrates that industrial auto-annotation is a stateful safety problem rather than a single-model inference problem. The deployed system now provides explicit dataset identity, stable browser behavior, visible model contention, fast bounded VLM routing, conservative candidate post-processing, and auditable recovery. LiDAR-derived 3-D prompts materially reduced a documented background leak but did not pass the automatic replacement gate. The correct conclusion is therefore improved responsiveness and failure containment, with a reproducible path toward human-reviewed post-training—not a claim of autonomous ground-truth generation.")

    primary(doc, "", "References")
    references = [
        "[1] A. Kirillov et al., “Segment Anything,” in Proc. IEEE/CVF Int. Conf. Comput. Vis. (ICCV), 2023, pp. 4015–4026.",
        "[2] N. Ravi et al., “SAM 2: Segment Anything in Images and Videos,” in Proc. Int. Conf. Learn. Representations (ICLR), 2025.",
        "[3] Meta AI, “SAM 3: Segment Anything with Concepts,” arXiv:2511.16719, 2025.",
        "[4] P. Wang et al., “Qwen2-VL: Enhancing Vision-Language Model’s Perception of the World at Any Resolution,” arXiv:2409.12191, 2024.",
        "[5] M. A. Fischler and R. C. Bolles, “Random sample consensus: A paradigm for model fitting with applications to image analysis and automated cartography,” Commun. ACM, vol. 24, no. 6, pp. 381–395, 1981.",
        "[6] IEEE Author Center, “Structure Your Article” and “IEEE Article Templates,” accessed Nov. 20, 2026.",
    ]
    for item in references:
        p = doc.add_paragraph()
        p.paragraph_format.left_indent = Inches(0.18)
        p.paragraph_format.first_line_indent = Inches(-0.18)
        p.paragraph_format.space_after = Pt(1)
        p.alignment = WD_ALIGN_PARAGRAPH.JUSTIFY
        set_font(p.add_run(item), 8)

    # Internal evidence note is deliberately separate from scholarly references.
    primary(doc, "", "Internal Reproducibility Statement")
    body(doc, "The reported workbench, latency, pallet, slope, and LiDAR results are bound to repository branch codex/workbench-stability-20261117. The illustrated Chinese manual and the structured Round335 evidence accompany this manuscript. Factory images are retained under the project’s existing access controls. Machine candidates, human truth, training approval, and deployment approval remain distinct states.", indent=False)

    doc.save(OUTPUT)
    return OUTPUT


if __name__ == "__main__":
    print(build())
