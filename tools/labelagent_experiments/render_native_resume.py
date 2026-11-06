"""Render only saved native-resume evidence; never synthesize model outcomes.

The sole outputs are assets/, videos/, index.html and media_verification.json
under outputs/labelagent_native_resume_20261107.  Source photos, masks, JSON
responses, old runs and final_summary.json are read-only inputs. Videos are
labelled stage replays, not screen recordings or fresh Agent inference.
"""
import collections
import hashlib
import html
import json
import sys
from fractions import Fraction
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / 'outputs/labelagent_native_resume_20261107'
WORK = ROOT / 'task_workspaces/labelagent_native_resume_20261107'
INC = ROOT / 'outputs/labelagent_context_comparison_20261105'
VIDEO_NAMES = (
    '01_原图结构与实际坐标规划',
    '02_实际SAM请求与完整返回',
    '03_原门槛复核与实例支持',
    '04_保留基线与本轮结论',
)


def main():
    import av
    import cv2
    import numpy as np
    from PIL import Image, ImageDraw, ImageFont, ImageOps

    if not (OUT / 'final_summary.json').exists():
        raise FileNotFoundError('Wait for final_summary.json; no final outcome is inferred by the renderer')
    if (OUT / 'media_verification.json').exists():
        raise FileExistsError('Media already rendered; retain the verified artifact')
    source_hashes = {}

    def remember(path):
        path = Path(path)
        source_hashes[str(path.resolve())] = hashlib.sha256(path.read_bytes()).hexdigest()
        return path

    def read(path, fallback=None):
        path = Path(path)
        if not path.exists():
            return fallback
        return json.loads(remember(path).read_text(encoding='utf-8'))

    def picture(path):
        return Image.open(remember(path)).convert('RGB')

    def write_json(path, value):
        path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')

    summary = read(OUT / 'final_summary.json')
    cfg = read(WORK / 'config_comparison.json') or read(WORK / 'config.json')
    if not cfg or not cfg.get('source_snapshot'):
        raise ValueError('A bound source_snapshot is required to rasterize actual saved shapes')
    sys.path.insert(0, cfg['source_snapshot'])
    from source.agent.edit_agent import _rasterize

    assets, videos = OUT / 'assets', OUT / 'videos'
    assets.mkdir(exist_ok=True)
    videos.mkdir(exist_ok=True)
    font_path = 'C:/Windows/Fonts/msyh.ttc'
    fonts = {size: ImageFont.truetype(font_path, size) for size in (20, 22, 24, 27, 34, 38)}
    groups = collections.OrderedDict()
    sequences = {name: [] for name in VIDEO_NAMES}
    cards = []
    notices = []
    raw_evidence = []
    candidate_masks = []

    def wrapped(text, font, max_width):
        rows = []
        for paragraph in str(text).splitlines() or ['']:
            line = ''
            for char in paragraph:
                if line and font.getlength(line + char) > max_width:
                    rows.append(line)
                    line = char
                else:
                    line += char
            rows.append(line)
        return rows

    def paint_lines(draw, lines, xy, font, fill, width, line_height, limit=None):
        rows = [line for text in lines for line in wrapped(text, font, width)]
        if limit is not None and len(rows) > limit:
            rows = rows[:limit - 1] + ['完整原文见网页对应的“实际保存响应”。']
        x, y = xy
        for line in rows:
            draw.text((x, y), line, font=font, fill=fill)
            y += line_height
        return y

    def register(path, group, video, seconds=3, sources=(), note=''):
        groups.setdefault(group, []).append(path)
        sequences[video].append((path, seconds))
        cards.append({'path': path.relative_to(OUT).as_posix(), 'group': group,
                      'source_files': [str(Path(p)) for p in sources], 'note': note})

    def panel(title, items, stem, notes=(), group='', video=None, seconds=3, sources=()):
        canvas = Image.new('RGB', (1920, 1080), '#142436')
        draw = ImageDraw.Draw(canvas)
        paint_lines(draw, [title], (34, 24), fonts[34], '#ffffff', 1850, 43, 2)
        top, region_height = 175, 590
        cell_width = 1832 // max(1, len(items))
        for index, (label, image) in enumerate(items):
            x = 44 + index * cell_width
            paint_lines(draw, [label], (x, 118), fonts[22], '#9fe9cf', cell_width - 26, 27, 2)
            photo = ImageOps.contain(image.convert('RGB'), (cell_width - 26, region_height))
            canvas.paste(photo, (x + (cell_width - 26 - photo.width) // 2,
                                 top + (region_height - photo.height) // 2))
        paint_lines(draw, list(notes), (34, 810), fonts[22], '#e0ebf3', 1850, 30, 7)
        draw.text((34, 1045), '实际保存证据 · 仅可视化 · 候选不是人工真值', font=fonts[20], fill='#a6bacd')
        path = assets / (stem + '.jpg')
        canvas.save(path, quality=94)
        register(path, group, video or VIDEO_NAMES[0], seconds, sources, '\n'.join(map(str, notes)))
        return path

    def text_card(title, lines, stem, group, video, seconds=5):
        canvas = Image.new('RGB', (1920, 1080), '#142436')
        draw = ImageDraw.Draw(canvas)
        paint_lines(draw, [title], (40, 30), fonts[38], '#ffffff', 1830, 48, 2)
        paint_lines(draw, lines, (45, 155), fonts[27], '#e0ebf3', 1825, 44, 18)
        draw.text((40, 1035), '从本轮保存文件读取；缺少的结果明确标为未执行或未提供', font=fonts[20], fill='#a6bacd')
        path = assets / (stem + '.jpg')
        canvas.save(path, quality=94)
        register(path, group, video, seconds, note='\n'.join(map(str, lines)))
        return path

    def overlay(image, mask, color=(0, 225, 165)):
        rgb = np.array(image.convert('RGB')).copy()
        mask = np.asarray(mask) != 0
        if mask.shape != rgb.shape[:2]:
            raise ValueError('Saved mask must have the exact source-photo geometry')
        rgb[mask] = (.55 * rgb[mask] + .45 * np.array(color)).astype('uint8')
        return Image.fromarray(rgb)

    def cropped(image, box):
        x1, y1, x2, y2 = map(int, box)
        if not 0 <= x1 < x2 <= image.width or not 0 <= y1 < y2 <= image.height:
            raise ValueError('Invalid saved crop box; no clipping of coordinates for display')
        return image.crop((x1, y1, x2, y2))

    def mask_box(mask, padding=32):
        yy, xx = np.where(mask)
        if not len(xx):
            return [0, 0, mask.shape[1], mask.shape[0]]
        return [max(0, int(xx.min()) - padding), max(0, int(yy.min()) - padding),
                min(mask.shape[1], int(xx.max()) + padding + 1),
                min(mask.shape[0], int(yy.max()) + padding + 1)]

    def full_photo(case):
        return picture(INC / 'inputs' / (case + '.png'))

    def old_pallet_union(case, width, height):
        saved = read(INC / 'C_hybrid' / case / 'result.json')
        if saved is None:
            raise FileNotFoundError('The retained baseline must remain inspectable')
        masks = [_rasterize(s, width, height) for s in saved['shapes'] if s.get('label') == 'pallet']
        return np.maximum.reduce(masks) if masks else np.zeros((height, width), np.uint8)

    # Every saved planning answer, including schema or seed-binding failures.
    planning = read(OUT / 'planning/progress.json', [])
    plan_done = read(OUT / 'planning/completed.json')
    seen_seed = set()
    for number, row in enumerate(planning, 1):
        folder = Path(row['folder'])
        source_file, cue_file = folder / 'SAM_source_input.png', folder / (row['cue'] + '.png')
        source = picture(source_file)
        cue = picture(cue_file)
        decision = row.get('decision', {})
        valid = row.get('valid_prompt') is True
        status = '坐标绑定通过；仍需后续完整实例验收' if valid else '未通过坐标/单物体绑定，不计为有效提示'
        raw_evidence.append({'heading': f"规划 {number:02d} / seed {row['index']:02d} / {row['cue']}",
                             'source': str(folder / (row['cue'] + '_decision.json')), 'value': row})
        notes = [status,
                 '开放物体命名：' + str(decision.get('object', '没有有效模型物体答案')) + '；范围：' + str(decision.get('instance_scope', '未提供')),
                 '模型证据：' + str(decision.get('evidence', row.get('error', '未提供'))),
                 '拒绝原因：' + str(row.get('rejection') or '无坐标拒绝；不等于语义或面积合格'),
                 '原始正点：' + json.dumps(decision.get('positive_points'), ensure_ascii=False) + '；原始负点：' + json.dumps(decision.get('negative_points'), ensure_ascii=False)]
        panel(f"seed {row['index']:02d} / {row['case']} / {row['cue']}：本轮真实规划响应",
              [('无定位标记的原始照片', source), ('模型实际收到的定位视图', cue)],
              f'规划响应_{number:02d}', notes, '实际物体命名与来源坐标绑定', VIDEO_NAMES[0], 3,
              [source_file, cue_file, folder / (row['cue'] + '_decision.json')])
        if row['index'] not in seen_seed:
            seen_seed.add(row['index'])
            d0, d1 = folder / 'detail0.png', folder / 'detail1.png'
            if d0.exists() and d1.exists():
                panel(f"seed {row['index']:02d}：同一物体规划的原始全貌与两份细节",
                      [('原始方形照片', source), ('实际左部细节输入', picture(d0)), ('实际右部细节输入', picture(d1))],
                      f'原图与细节_seed{row["index"]:02d}',
                      ['本轮实际输入的三份图像；细节不作为另一套点坐标系。',
                       '全貌覆盖完整旧种子，细节仅辅助观察；没有把候选物体名称渲染到输入图片中。'],
                      '原始全貌与实际细节输入', VIDEO_NAMES[0], 3, [source_file, d0, d1])
        if valid and row.get('points'):
            binding = row['binding']; box = binding['source_box']; array = np.array(source).copy()
            for point in row['points']:
                x, y = int(point['x'] - box[0]), int(point['y'] - box[1])
                if not 0 <= x < source.width or not 0 <= y < source.height:
                    raise ValueError('Validated point lies outside actual source photo; stop media verification')
                cv2.circle(array, (x, y), max(3, round(source.width / 150)),
                           (40, 255, 90) if point['label'] == 1 else (255, 40, 170), -1)
            panel(f"seed {row['index']:02d} / {row['cue']}：实际绑定到原图的点",
                  [('未修改的原始照片', source), ('有效正负点的位置（仅显示）', Image.fromarray(array))],
                  f'来源绑定点_{number:02d}',
                  ['绿点为正点，紫点为负点；坐标来自模型返回，经严格映射后显示。',
                   '没有夹取、吸附、按面积预算挪点；通过坐标检查不代表完整托盘已识别。'],
                  '通过来源绑定的实际提示点', VIDEO_NAMES[0], 3, [source_file])
    if not planning:
        notices.append('规划阶段没有保存响应；不展示或推断模型已完成规划。')
        text_card('结构规划尚无可回放模型响应', [notices[-1]], '规划_未执行',
                  '未执行边界', VIDEO_NAMES[0])

    # Every actual SAM request and its unmodified return, not just survivors.
    sam_rows = read(OUT / 'seam_generation/progress.json', [])
    sam_done = read(OUT / 'seam_generation/completed.json')
    for number, row in enumerate(sam_rows, 1):
        folder = Path(row['folder']); input_file = folder / 'input.png'
        source = picture(input_file); prompt = read(folder / 'prompt.json', {}); response = read(folder / 'response.json')
        if response is None:
            notices.append(str(folder) + ' 没有 response.json；本次请求不呈现返回掩码。')
            response = {}
        mask = _rasterize(response, source.width, source.height) if response.get('parts') else np.zeros((source.height, source.width), np.uint8)
        display = np.array(overlay(source, mask, (240, 105, 65))).copy()
        for point in prompt.get('points', []):
            x, y = int(point['x']), int(point['y'])
            if not 0 <= x < source.width or not 0 <= y < source.height:
                raise ValueError('Actual SAM prompt point is out of bounds')
            cv2.circle(display, (x, y), max(3, round(source.width / 150)),
                       (40, 255, 90) if point['label'] == 1 else (255, 40, 170), -1)
        notes = ['实际完整返回的着色显示；绿正点、紫负点；没有按面积裁小输出。',
                 '技术有效完整输出：' + str(row.get('technically_valid', '未提供')) + '；返回掩码像素：' + str(int(mask.sum())),
                 '拒绝或警告：' + str(row.get('rejection') or row.get('warnings') or response.get('rejected_reason') or '本步骤无拒绝，仍需原验收'),
                 '原始响应、提示点及失败结果保留；SAM 非空返回不等于识别到托盘。']
        panel(f"实际 SAM 请求 {number:02d} / {row.get('case')} / seed {row.get('index')} / {row.get('cue', '')}",
              [('实际 SAM 原始输入', source), ('完整原始返回与真实提示点', Image.fromarray(display))],
              f'SAM完整返回_{number:02d}', notes, '全部真实SAM请求与原始返回', VIDEO_NAMES[1], 4,
              [input_file, folder / 'prompt.json', folder / 'response.json'])
        raw_evidence.append({'heading': f'SAM 请求 {number:02d} 的保存记录', 'source': str(folder / 'response.json'),
                             'value': {'request_record': row, 'prompt': prompt,
                                       'return_metadata': {k: v for k, v in response.items() if k not in ('points', 'parts')}}})
        blank_file, blank_response_file = folder / 'synthetic_blank_input.png', folder / 'synthetic_blank_response.json'
        if blank_file.exists() and blank_response_file.exists():
            blank = picture(blank_file); bresp = read(blank_response_file)
            bm = _rasterize(bresp, blank.width, blank.height) if bresp.get('parts') else np.zeros((blank.height, blank.width), np.uint8)
            panel('合成空白诊断：同一提示的实际返回，不能计为真实检出',
                  [('合成空白实际输入', blank), ('合成空白上的原始返回', overlay(blank, bm, (235, 95, 55)))],
                  f'合成空白诊断_{number:02d}',
                  [f'非空返回：{bool(bresp.get("parts"))}；掩码像素：{int(bm.sum())}。',
                   '此为功能诊断，明确不是真实场景、训练样本或真实托盘结果。'],
                  '明确标记的合成空白诊断', VIDEO_NAMES[1], 4, [blank_file, blank_response_file])
    if not sam_rows:
        line = '本轮没有保存真实 SAM 点提示请求；不得把准备输入当作分割结果。'
        if sam_done:
            line += ' 保存状态：' + str(sam_done.get('status', '未提供'))
        text_card('本轮 SAM 阶段的执行边界', [line], 'SAM_无请求记录', '未执行边界', VIDEO_NAMES[1])

    # Whole mapped candidates and the original incumbent in identical display crops.
    hypotheses = read(OUT / 'seam_generation/all_hypotheses.json', [])
    for number, row in enumerate(hypotheses, 1):
        source = full_photo(row['case']); mask = _rasterize(row['shape'], source.width, source.height)
        baseline = old_pallet_union(row['case'], source.width, source.height)
        box = row.get('source_box') or mask_box(mask)
        extra = int(((mask != 0) & (baseline == 0)).sum())
        candidate_masks.append({'key': row['key'], 'case': row['case'], 'pixels': int(mask.sum()),
                                'new_union_pixels_display_audit': extra,
                                'source_mask_sha256': hashlib.sha256(mask.tobytes()).hexdigest()})
        panel(f"完整新候选 {number:02d} / {row['case']} / {row['key'][:12]}",
              [('同一原始照片', cropped(source, box)),
               ('完整模型候选（未裁面积）', cropped(overlay(source, mask), box)),
               ('原有前景（仅显示保留基线）', cropped(overlay(source, baseline, (100, 165, 255)), box))],
              f'完整候选与原基线_{number:02d}',
              [f'候选完整掩码 {int(mask.sum())} 像素；相对保留托盘并集新增 {extra} 像素。',
               '本图是原始输出的可视化，不对掩码裁剪、相交、删连通块或填补。',
               '完整模型输出仍需物体语义、独立实例支持、人员保护与原面积验收；不能仅凭点覆盖采用。'],
              '新完整候选与保留基线', VIDEO_NAMES[1], 4,
              [INC / 'inputs' / (row['case'] + '.png'), Path(row['source_prediction'])])

    selected = read(OUT / 'instance_generation/selection.json')
    verification = read(OUT / 'instance_generation/verification_progress.json', {}).get('rows', [])
    for number, row in enumerate(verification, 1):
        candidate = row['row']; source = full_photo(candidate['case'])
        mask = _rasterize(candidate['shape'], source.width, source.height); box = mask_box(mask)
        decisions = row.get('neutral', [])
        neutral_text = ' / '.join(str(r.get('object', r)) for r in decisions) or '未执行中性物体复核'
        panel('原门槛下的候选复核 / ' + candidate['key'][:12],
              [('原始实景', cropped(source, box)), ('完整候选，只用于观察', cropped(overlay(source, mask), box))],
              f'原门槛复核_{number:02d}',
              ['原始中性物体回答：' + neutral_text,
               f"中性语义通过：{row.get('neutral_passed', '未提供')}；单实例通过：{row.get('single_instance_passed', '未提供')}；可见范围通过：{row.get('extent_passed', '未提供')}。",
               f"本项最终通过：{row.get('passed', '未提供')}；新调用与输入绑定复用需查阅本轮请求账目。",
               '只显示原保存答案与验收状态，不把复用回答写成新增推理。'],
              '原预算与语义验收证据', VIDEO_NAMES[2], 4)
        raw_evidence.append({'heading': '候选复核 ' + candidate['key'][:12],
                             'source': str(OUT / 'instance_generation/verification_progress.json'),
                             'value': {k: v for k, v in row.items() if k != 'row'}})
    if selected is not None:
        reasons = collections.Counter(r.get('reason', '未提供原因') for r in selected.get('rejected_whole_masks', []))
        counts = {k: len(v) for k, v in selected.get('selected', {}).items()}
        lines = ['保存的完整候选入选数量：' + json.dumps(counts, ensure_ascii=False),
                 '整项拒绝原因及数量：'] + [f'{reason}：{count}' for reason, count in reasons.items()]
        lines += [f"语义暂扣记录：{len(selected.get('held_semantics', []))}。",
                  '面积拒绝可能早于语义检查；被面积挡住不等于已证实是真托盘。',
                  '所有拒绝保持完整原输出，不以裁掉像素来通过预算。']
        text_card('完整候选的原标准筛选记录', lines, '完整候选原标准筛选', '原预算与语义验收证据', VIDEO_NAMES[2])
    else:
        text_card('原标准筛选尚无保存结果', ['未发现 selection.json；不得宣称新候选已通过或已被最终拒绝。'],
                  '候选筛选_未执行', '未执行边界', VIDEO_NAMES[2])

    audit_folder = OUT / 'instance_support_audit_corrected'
    if not (audit_folder / 'completed.json').exists():
        audit_folder = OUT / 'instance_support_audit'
    audit = read(audit_folder / 'completed.json')
    for number, file in enumerate(sorted(audit_folder.glob('*.jpg')), 1):
        image = picture(file)
        if image.size != (1920, 1080):
            raise ValueError('The saved support-diagnostic card must already be 1920x1080')
        output_file = assets / f'独立实例支持诊断_{number:02d}.jpg'
        image.save(output_file, quality=95)
        register(output_file, '冻结后的实例支持诊断', VIDEO_NAMES[2], 5, [file],
                 '直接保存本轮修正后的完整诊断图，避免嵌套缩小文字；参考只用于冻结后评分，不是人工二维真值。')
    if audit is not None:
        raw_evidence.append({'heading': '冻结后的独立实例支持完整记录',
                             'source': str(audit_folder / 'completed.json'), 'value': audit})

    # Baseline is explicitly labelled old evidence; no old frames become new calls.
    for cam in (0, 1):
        case = f'camera{cam}_frame76'; source = full_photo(case)
        baseline = old_pallet_union(case, source.width, source.height)
        panel(case + '：保留版本的原始照片与托盘前景',
              [('原始照片（同图工程实验）', source), ('原有托盘掩码；不是本轮新推理', overlay(source, baseline, (100, 165, 255)))],
              case + '_保留基线',
              ['明确复用：这是原保留版本的托盘前景，用于对照，不计为新生成或新复核。',
               '是否逐像素保持、是否新增候选与是否重跑 Agent，须以本轮保护收据和总结为准。'],
              '明确标记的保留版本对照', VIDEO_NAMES[3], 5,
              [INC / 'inputs' / (case + '.png'), INC / 'C_hybrid' / case / 'result.json'])

    declared = [
        '本轮保存总结中的保留托盘支持：' + str(summary.get('retained_pallet_support', '未提供')),
        '收敛结果来源：' + str(summary.get('retained_convergence', '未提供')),
        '新增 Agent 实际运行：' + str(summary.get('fresh_agent_runs', '未提供')),
        '是否采用新候选：' + str(summary.get('new_candidate_adopted', '未提供')),
        '剩余托盘漏检：' + str(summary.get('remaining_pallet_misses', '未提供')),
        '原标注写入次数：' + str(summary.get('annotation_writes', '未提供')),
    ]
    conclusions = summary.get('conclusions', [])
    limitations = summary.get('limitations', [])
    if isinstance(conclusions, str):
        conclusions = [conclusions]
    if isinstance(limitations, str):
        limitations = [limitations]
    text_card('本轮实际结果：按保存总结展示', declared, '本轮结果声明', '本轮结论与证据边界', VIDEO_NAMES[3], 6)
    for index in range(0, len(conclusions), 8):
        text_card('本轮结论 / ' + str(index // 8 + 1), conclusions[index:index + 8],
                  '本轮结论_' + str(index // 8 + 1), '本轮结论与证据边界', VIDEO_NAMES[3], 6)
    boundary = list(limitations) + ['视频是实际保存证据的阶段回放，非连续录屏。',
                                  '定位标记、着色与点标记只用于展示，没有改变输入源图、模型返回或掩码。',
                                  '同图开发诊断与功能对照不等于独立密集准确率或上线验收。']
    for index in range(0, len(boundary), 8):
        text_card('证据边界与未完成项 / ' + str(index // 8 + 1), boundary[index:index + 8],
                  '证据边界_' + str(index // 8 + 1), '本轮结论与证据边界', VIDEO_NAMES[3], 6)

    video_receipts = []
    for name, stages in sequences.items():
        if not stages:
            raise AssertionError('Each declared replay requires actual evidence or an explicit missing-stage card')
        path = videos / (name + '.mp4'); pending = path.with_suffix('.pending.mp4')
        container = av.open(str(pending), 'w', format='mp4', options={'movflags': '+faststart'})
        stream = container.add_stream('libx264', rate=10); stream.width = 1920; stream.height = 1080
        stream.pix_fmt = 'yuv420p'; stream.codec_context.thread_count = 2
        stream.options = {'crf': '23', 'preset': 'veryfast'}
        count = 0
        for path_image, seconds in stages:
            photo = ImageOps.contain(Image.open(path_image).convert('RGB'), (1880, 1000))
            canvas = Image.new('RGB', (1920, 1080), '#142436')
            canvas.paste(photo, ((1920 - photo.width) // 2, 4))
            ImageDraw.Draw(canvas).text((30, 1030), '实际保存阶段回放 · 非连续录屏 · 旧结果复用已明确标记',
                                       font=fonts[22], fill='white')
            frame = av.VideoFrame.from_image(canvas)
            for _ in range(seconds * 10):
                frame.pts = count; frame.time_base = Fraction(1, 10); frame.pict_type = av.video.frame.PictureType.NONE
                for packet in stream.encode(frame):
                    container.mux(packet)
                count += 1
        for packet in stream.encode():
            container.mux(packet)
        container.close(); pending.replace(path)
        decoded = 0
        with av.open(str(path)) as video:
            for frame in video.decode(video=0):
                if frame.width != 1920 or frame.height != 1080:
                    raise AssertionError('Unexpected decoded video dimensions')
                decoded += 1
        if decoded != count:
            raise AssertionError('Video frame count differs from actual stage sequence')
        receipt = {'path': path.relative_to(OUT).as_posix(), 'seconds': count / 10,
                   'frames': decoded, 'sha256': hashlib.sha256(path.read_bytes()).hexdigest(),
                   'width': 1920, 'height': 1080, 'kind': 'saved_stage_replay_not_screen_recording',
                   'stages': [{'path': p.relative_to(OUT).as_posix(), 'seconds': s} for p, s in stages]}
        video_receipts.append(receipt)
        print(json.dumps({k: v for k, v in receipt.items() if k != 'stages'}, ensure_ascii=False), flush=True)

    # Check every generated image and every source input again before success.
    image_receipts = []
    for card in cards:
        path = OUT / card['path']
        with Image.open(path) as im:
            im.load()
            if im.size != (1920, 1080):
                raise AssertionError('A diagnostic card is not 1920x1080')
        image_receipts.append({**card, 'sha256': hashlib.sha256(path.read_bytes()).hexdigest(),
                               'width': 1920, 'height': 1080})
    unchanged = all(hashlib.sha256(Path(p).read_bytes()).hexdigest() == digest for p, digest in source_hashes.items())
    if not unchanged:
        raise AssertionError('A source artifact changed during media rendering')

    page = ['<!doctype html><html lang="zh"><meta charset="utf-8">',
            '<meta name="viewport" content="width=device-width,initial-scale=1">',
            '<title>原图坐标续测：完整基座与独立实例证据</title>',
            '<style>body{max-width:1450px;margin:24px auto;padding:20px;font:17px/1.7 "Microsoft YaHei",sans-serif;background:#eff4f8;color:#142436}section{background:white;padding:24px;border-radius:12px;margin:20px 0}img,video{display:block;width:100%;height:auto}pre{white-space:pre-wrap;overflow-wrap:anywhere}summary{cursor:pointer}a{color:#005c89}p{margin:12px 0}.note{border-left:5px solid #137c6c;padding-left:16px}</style>',
            '<h1>原图坐标续测：完整基座与独立实例证据</h1>',
            '<section><p class="note">全部图片源于本轮实际保存输入、模型回答与完整返回；保留基线明确标为旧结果。视频为保存阶段回放，非连续录屏。</p><pre>',
            html.escape('\n'.join(declared + list(conclusions) + list(limitations))), '</pre></section>',
            '<section><h2>本轮原始记录与核验</h2>']
    for rel in ['final_summary.json', 'optimization_plan.json', 'planning/completed.json',
                'planning/frozen_plan.json', 'planning/request_receipts.json', 'planning/progress.json',
                'seam_generation/completed.json', 'seam_generation/progress.json',
                'seam_generation/all_hypotheses.json', 'instance_generation/completed.json',
                'instance_generation/selection.json', audit_folder.relative_to(OUT).as_posix() + '/completed.json',
                'acceptance_checks.json', 'original_preservation.json', 'whole_mask_preservation.json',
                'complete_generated_mask_proof.json', 'initial_seed_reuse_proof.json', 'media_verification.json']:
        if rel == 'media_verification.json' or (OUT / rel).exists():
            page.append(f'<p><a href="{rel}">{html.escape(rel)}</a></p>')
    page.append('</section><section><h2>4 段实际保存阶段回放</h2>')
    for receipt in video_receipts:
        page.append(f'<p>{html.escape(receipt["path"])} · {receipt["seconds"]} 秒 · 全部 {receipt["frames"]} 帧已解码</p><video controls preload="metadata" src="{receipt["path"]}"></video>')
    page.append('</section>')
    for group, paths in groups.items():
        page.append('<section><h2>' + html.escape(group) + '</h2>')
        for path in paths:
            relative = path.relative_to(OUT).as_posix()
            page.append(f'<p>{html.escape(path.stem)}</p><a href="{relative}"><img loading="lazy" alt="{html.escape(path.stem, quote=True)}" src="{relative}"></a>')
        page.append('</section>')
    page.append('<section><h2>实际保存响应（可展开）</h2>')
    for item in raw_evidence:
        page.append('<details><summary>' + html.escape(item['heading']) + '</summary><pre>' +
                    html.escape(json.dumps(item['value'], ensure_ascii=False, indent=2)) + '</pre></details>')
    page.append('</section></html>')
    (OUT / 'index.html').write_text('\n'.join(page), encoding='utf-8')
    receipt = {'passed': True, 'pictures': len(image_receipts), 'videos': video_receipts,
               'fully_decoded_video_frames': sum(r['frames'] for r in video_receipts),
               'image_receipts': image_receipts, 'all_source_files_unchanged_during_render': unchanged,
               'source_file_sha256': source_hashes, 'native_planning_rows': len(planning),
               'planning_stage_completed': plan_done, 'actual_SAM_request_rows': len(sam_rows),
               'SAM_stage_completed': sam_done, 'new_complete_hypothesis_rows': len(hypotheses),
               'candidate_display_geometry': candidate_masks, 'notices': notices,
               'instance_support_diagnostic_source': audit_folder.relative_to(OUT).as_posix(),
               'mask_or_annotation_writes': 0, 'synthetic_model_outputs': 0,
               'screen_recording': False, 'browser_playback_verification_required_separately': True}
    write_json(OUT / 'media_verification.json', receipt)
    print(json.dumps({'passed': True, 'pictures': len(image_receipts), 'videos': len(video_receipts),
                      'decoded_frames': receipt['fully_decoded_video_frames'], 'output': str(OUT)}, ensure_ascii=False), flush=True)


if __name__ == '__main__':
    main()
