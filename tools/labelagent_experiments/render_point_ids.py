"""Render only saved point-ID evidence; never synthesize model outcomes.

The sole outputs are assets/, videos/, index.html and media_verification.json
under outputs/labelagent_point_ids_20261107.  Source photos, masks, JSON
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
OUT = ROOT / 'outputs/labelagent_point_ids_20261107'
WORK = ROOT / 'task_workspaces/labelagent_point_ids_20261107'
INC = ROOT / 'outputs/labelagent_context_comparison_20261105'
VIDEO_NAMES = (
    '01_原图与固定编号点选择',
    '02_实际SAM请求与完整返回',
    '03_原门槛复核与实例支持',
    '04_保留基线与本轮结论',
    '05_邻帧RGB关联诊断',
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
        if video in sequences:
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
        register(path, group, VIDEO_NAMES[0] if video is None else video, seconds, sources, '\n'.join(map(str, notes)))
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

    # The two actual inputs are clean photo + same-photo point-ID menu.
    planning = read(OUT / 'planning/progress.json', [])
    plan_done = read(OUT / 'planning/completed.json')
    for number, row in enumerate(planning, 1):
        folder = Path(row['folder'])
        source_file, clean_file, cue_file = folder / 'SAM_source_input.png', folder / 'clean.png', folder / (row['cue'] + '.png')
        source = picture(source_file); clean = picture(clean_file)
        cue = picture(cue_file)
        decision = row.get('decision', {})
        valid = row.get('valid_prompt') is True
        status = '固定编号点绑定通过；仍需后续完整实例验收' if valid else '未通过编号点/单物体绑定，不计为有效提示'
        raw_evidence.append({'heading': f"规划 {number:02d} / seed {row['index']:02d} / {row['cue']}",
                             'source': str(folder / (row['cue'] + '_decision.json')), 'value': row})
        notes = [status,
                 '开放物体命名：' + str(decision.get('object', '没有有效模型物体答案')) + '；范围：' + str(decision.get('instance_scope', '未提供')),
                 '模型证据：' + str(decision.get('evidence', row.get('error', '未提供'))),
                 '拒绝原因：' + str(row.get('rejection') or '无编号绑定拒绝；不等于语义或面积合格'),
                 '模型原始正点编号：' + json.dumps(decision.get('positive_ids'), ensure_ascii=False) + '；负点编号：' + json.dumps(decision.get('negative_ids'), ensure_ascii=False)]
        panel(f"seed {row['index']:02d} / {row['case']} / {row['cue']}：本轮真实编号点选择",
              [('模型输入1：同源无标记照片', clean), ('模型输入2：同源定位与固定点编号', cue)],
              f'编号点规划响应_{number:02d}', notes, '实际物体命名与固定编号点选择', VIDEO_NAMES[0], 3,
              [source_file, clean_file, cue_file, folder / (row['cue'] + '_decision.json')])
        if valid and row.get('points'):
            binding = row['binding']; box = binding['source_box']; array = np.array(source).copy()
            by_id = {p['id']: p for p in binding['point_catalog']}
            expected = [(by_id[ident]['x'], by_id[ident]['y'], label)
                        for name, label in [('positive_ids', 1), ('negative_ids', 0)]
                        for ident in decision.get(name, [])]
            actual = [(p['x'], p['y'], p['label']) for p in row['points']]
            if expected != actual:
                raise ValueError('Saved actual SAM points differ from exact selected-ID mapping')
            for point in row['points']:
                x, y = int(point['x'] - box[0]), int(point['y'] - box[1])
                if not 0 <= x < source.width or not 0 <= y < source.height:
                    raise ValueError('Validated point lies outside actual source photo; stop media verification')
                cv2.circle(array, (x, y), max(3, round(source.width / 150)),
                           (40, 255, 90) if point['label'] == 1 else (255, 40, 170), -1)
            panel(f"seed {row['index']:02d} / {row['cue']}：编号精确映射后的实际点",
                  [('未修改的原始照片', source), ('按固定编号查表得到的正负点', Image.fromarray(array))],
                  f'编号精确映射点_{number:02d}',
                  ['绿点为正点，紫点为负点；模型只选择编号，像素位置来自请求前冻结的目录。',
                   '没有让模型直接生成坐标，也没有补点、替换点、按面积预算挪点；绑定通过不等于完整实例合格。'],
                  '编号精确映射后的实际提示点', VIDEO_NAMES[0], 3, [source_file])
    if not planning:
        notices.append('规划阶段没有保存响应；不展示或推断模型已完成规划。')
        text_card('结构规划尚无可回放模型响应', [notices[-1]], '规划_未执行',
                  '未执行边界', VIDEO_NAMES[0])

    # These are real planner responses on explicitly synthetic blank controls.
    planner_controls = read(OUT / 'planner_controls/progress.json', [])
    planner_controls_done = read(OUT / 'planner_controls/completed.json')
    for number, row in enumerate(planner_controls, 1):
        folder = Path(row['folder']); clean_file = folder / 'clean.png'; cue_file = folder / (row['cue'] + '.png')
        decision = row.get('decision', {})
        panel(f"合成灰白对照 {number:02d} / {row.get('kind')} / {row['cue']}：真实模型回答",
              [('合成无标记输入（不是实景）', picture(clean_file)),
               ('同一合成图的固定编号点输入', picture(cue_file))],
              f'编号规划_合成灰白对照_{number:02d}',
              ['独立功能对照：这是合成灰白照片上的实际规划请求，不能计入真实场景托盘检出。',
               '物体原答：' + str(decision.get('object', '无有效回答')) + '；范围：' + str(decision.get('instance_scope', '未提供')),
               '正点编号：' + json.dumps(decision.get('positive_ids'), ensure_ascii=False) + '；负点编号：' + json.dumps(decision.get('negative_ids'), ensure_ascii=False),
               f"声称托盘：{row.get('claims_pallet', '未提供')}；弃权且点列表为空：{row.get('abstains_with_empty_points', '未提供')}；编号绑定有效：{row.get('valid_prompt', '未提供')}。",
               '未因对照表现改变原语义/完整性/纯度/面积条件；结果只用于诊断与保守拦截。'],
              '合成灰白上的真实规划负对照', VIDEO_NAMES[0], 4, [clean_file, cue_file])
        raw_evidence.append({'heading': '合成灰白编号规划对照 ' + str(number),
                             'source': str(OUT / 'planner_controls/progress.json'), 'value': row})
        if row.get('valid_prompt') and row.get('points'):
            source_file = folder / 'SAM_source_input.png'; source = picture(source_file)
            box = row['binding']['source_box']; selected_points = np.array(source).copy()
            catalog = {point['id']: point for point in row['binding']['point_catalog']}
            expected = [(catalog[ident]['x'], catalog[ident]['y'], label)
                        for field, label in [('positive_ids', 1), ('negative_ids', 0)]
                        for ident in decision.get(field, [])]
            if expected != [(p['x'], p['y'], p['label']) for p in row['points']]:
                raise ValueError('Synthetic-control IDs do not map exactly to saved points')
            for point in row['points']:
                x, y = int(point['x'] - box[0]), int(point['y'] - box[1])
                if not 0 <= x < source.width or not 0 <= y < source.height:
                    raise ValueError('Synthetic-control selected point escaped its source crop')
                cv2.circle(selected_points, (x, y), max(3, round(source.width / 150)),
                           (40, 255, 90) if point['label'] == 1 else (255, 40, 170), -1)
            panel(f"合成对照 {number:02d} / {row['cue']}：编号可绑定，却没有真实物体",
                  [('无真实物体的合成输入', source), ('错误物体声明产生的有效编号点', Image.fromarray(selected_points))],
                  f'合成对照_有效编号假点_{number:02d}',
                  ['这是规划器在合成灰白输入上选择的点，只有编号和坐标绑定有效，不能证明物体存在。',
                   '没有向 SAM 发送此对照点集；图中无分割掩码，不能当作真实目标或检测结果。',
                   '即使没有声称托盘，仍然可能虚构单个物体；空白图上的物体幻觉尚未解决。'],
                  '合成对照中的有效编号假点', VIDEO_NAMES[0], 4, [source_file])

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
               f"本项最终通过：{row.get('passed', '未提供')}；本轮新语义成功响应：{summary.get('new_semantic_successful_responses', '未提供')}，历史输入绑定复用不计为新调用。",
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

    # Same-camera RGB tracking is a separate diagnostic, never a final mask.
    temporal_folder = OUT / 'temporal_audit'
    temporal = read(temporal_folder / 'completed.json')
    temporal_gallery_files = 0
    if temporal is not None:
        if temporal.get('status') != 'complete_RGB_diagnostic_only':
            raise ValueError('Temporal RGB diagnostics are still incomplete')
        if temporal.get('new_masks') != 0 or temporal.get('object_identity_confirmations') != 0:
            raise ValueError('Do not render changed temporal semantics as a no-mask diagnostic')
        temporal_readback = read(temporal_folder / 'independent_readback.json')
        if not temporal_readback or temporal_readback.get('passed') is not True:
            raise ValueError('Independent RGB diagnostic readback must pass before publishing media')
        read(temporal_folder / 'image_readback.json')
        read(temporal_folder / 'protected_inputs_readback.json')
        read(temporal_folder / 'visual_observations.json')
        for row in temporal.get('rows', []):
            folder = temporal_folder / f"seed{row['index']:02d}"
            read(folder / 'tracking.json')
            crop_bindings = read(folder / 'display_crop_binding.json')
            if any(binding.get('mask_warped') is not False for binding in crop_bindings):
                raise ValueError('No warped mask may be shown as a final segmentation')
            target_by_frame = {target['frame']: target for target in row['targets']}
            combined_file = folder / 'three_frame_comparison.jpg'
            combined = picture(combined_file)
            if combined.size != (1920, 1080):
                raise ValueError('A native temporal comparison card must be 1920x1080')
            output_file = assets / f"邻帧RGB_seed{row['index']:02d}_三帧完整对照.jpg"
            combined.save(output_file, quality=95)
            register(output_file, '同相机邻帧RGB关联：完整三帧对照', VIDEO_NAMES[4], 6,
                     [combined_file], '仅原图与真实RGB跟踪点；不转移掩码，不确认对象身份，不作最终分割。')
            temporal_gallery_files += 1
            for frame in (74, 76, 78):
                if frame == 76:
                    metric = f"原始帧：{row['initial_points']} 个图像特征点；这是RGB跟踪起点，不是三维参考点。"
                    anchor = '原始锚点仅定位旧种子；旧种子不是人工实例真值。'
                else:
                    target = target_by_frame[frame]
                    metric = f"有效RGB对应 {target['valid_points']}/{target['initial_points']}（{target['valid_fraction']:.1%}）；图像关联诊断可用：{target['diagnostic_image_correspondence_available']}。"
                    anchor = f"锚点通过：{target['anchor_valid']}；对象身份确认：{target['object_identity_confirmed']}；完整基座确认：{target['more_complete_visible_base_confirmed']}。"
                crop_info = next(binding for binding in crop_bindings if binding['frame'] == frame)
                for suffix, label in [('source', '无标记原始照片裁块'), ('tracks', '实际RGB跟踪点（不是SAM提示）')]:
                    source_file = folder / f'frame{frame}_{suffix}.png'
                    source = picture(source_file)
                    panel(f"邻帧RGB / seed {row['index']:02d} / {row['case']} / 帧{frame} / {label}",
                          [(label, source)], f"邻帧RGB_seed{row['index']:02d}_frame{frame}_{suffix}",
                          [metric, anchor,
                           '裁块定位依据：' + str(crop_info.get('delta_source')) + '；裁块因图像边缘重定位：' + str(crop_info.get('display_crop_repositioned_to_image_edge')) + '。',
                           '点对应仅由同相机RGB图像估计；不读取三维参考点/标签/面积/标定，不把任何warp当作最终mask。'],
                          '邻帧原图与真实RGB跟踪点图库', False, 3, [source_file])
                    temporal_gallery_files += 1
        if temporal_gallery_files != temporal['pictures']:
            raise ValueError('Not every saved temporal diagnostic picture reached the gallery')
        text_card('邻帧RGB诊断的实际范围与限制',
                  [f"{temporal['seeds']} 个旧种子、{temporal['pairs']} 对同相机邻帧；完整展示 {temporal_gallery_files} 张实际保存图。",
                   f"达到图像关联诊断条件：{temporal['diagnostic_supported_pairs']}/{temporal['pairs']}；锚点有效：{temporal['anchor_valid_pairs']}/{temporal['pairs']}。",
                   f"对象身份确认：{temporal['object_identity_confirmations']}；完整基座确认：{temporal['complete_base_confirmations']}；新增mask：{temporal['new_masks']}。",
                   '图像关联数与原Agent的6/6收敛是不同指标；不得混用或宣称恢复了漏检。',
                   '原图、跟踪失败、锚点失败和裁块定位均保留；本诊断不生成最终标注、不转移掩码。'],
                  '邻帧RGB_证据范围', '邻帧诊断范围与限制', VIDEO_NAMES[4], 7)
        raw_evidence.append({'heading': '同相机邻帧RGB诊断完整记录',
                             'source': str(temporal_folder / 'completed.json'), 'value': temporal})
    else:
        text_card('邻帧RGB诊断尚无完成收据', ['没有 temporal_audit/completed.json；不展示成功关联或实例恢复。'],
                  '邻帧RGB_未执行', '未执行边界', VIDEO_NAMES[4])

    contract = read(OUT / 'semantic_contract_audit.json')
    if contract is not None:
        scope = contract.get('scope', {})
        text_card('原语义规则与叠层归属：冻结后的只读证据核查',
                  [contract.get('conclusion', '未提供审计结论'),
                   '用户规则：每个托盘分别标注，堆叠时每层一个实例；不能合并整摞或改类别换取通过。',
                   '原42/43是两个薄Pallet框，几何关系与叠层相容；不据此虚构第76帧的二维像素边界。',
                   '第0帧参考的批准仅适用于对应参考，不能延伸为第76帧42/43的身份确认。',
                   f"本项只读审计：模型调用 {scope.get('model_calls', '未提供')}，新增mask {scope.get('new_masks', '未提供')}，阈值变动 {scope.get('threshold_changes', '未提供')}，标注写入 {scope.get('annotation_writes', '未提供')}。"],
                  '语义规则与叠层关系_只读证据', '原语义规则与二维身份边界', VIDEO_NAMES[3], 7)

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
    conclusions = [str(line).replace('通过严格坐标绑定', '通过严格编号点绑定') for line in conclusions]
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
            '<title>固定编号点与邻帧归属：完整基座与独立实例证据</title>',
            '<style>body{max-width:1450px;margin:24px auto;padding:20px;font:17px/1.7 "Microsoft YaHei",sans-serif;background:#eff4f8;color:#142436}section{background:white;padding:24px;border-radius:12px;margin:20px 0}img,video{display:block;width:100%;height:auto}pre{white-space:pre-wrap;overflow-wrap:anywhere}summary{cursor:pointer}a{color:#005c89}p{margin:12px 0}.note{border-left:5px solid #137c6c;padding-left:16px}</style>',
            '<h1>固定编号点与邻帧归属：完整基座与独立实例证据</h1>',
            '<section><p class="note">全部图片源于本轮实际保存输入、模型回答与完整返回；保留基线明确标为旧结果。视频为保存阶段回放，非连续录屏。</p><pre>',
            html.escape('\n'.join(declared + list(conclusions) + list(limitations))), '</pre></section>',
            '<section><h2>本轮原始记录与核验</h2>']
    for rel in ['final_summary.json', 'optimization_plan.json', 'planning/completed.json',
                'planning/frozen_plan.json', 'planning/request_receipts.json', 'planning/progress.json',
                'planner_controls/completed.json', 'planner_controls/progress.json',
                'seam_generation/completed.json', 'seam_generation/progress.json',
                'seam_generation/all_hypotheses.json', 'instance_generation/completed.json',
                'instance_generation/selection.json', audit_folder.relative_to(OUT).as_posix() + '/completed.json',
                'acceptance_checks.json', 'original_preservation.json', 'whole_mask_preservation.json',
                'complete_generated_mask_proof.json', 'initial_seed_reuse_proof.json',
                'temporal_audit/completed.json', 'temporal_audit/independent_readback.json',
                'temporal_audit/visual_observations.json', 'semantic_contract_audit.json', 'media_verification.json']:
        if rel == 'media_verification.json' or (OUT / rel).exists():
            page.append(f'<p><a href="{rel}">{html.escape(rel)}</a></p>')
    page.append('</section><section><h2>' + str(len(video_receipts)) + ' 段实际保存阶段回放</h2>')
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
               'source_file_sha256': source_hashes, 'point_ID_planning_rows': len(planning),
               'planner_synthetic_control_rows': len(planner_controls),
               'planner_controls_completed': planner_controls_done,
               'planning_input_images_per_request': 2,
               'model_selects_IDs_not_coordinates': True,
               'temporal_original_gallery_pictures': temporal_gallery_files,
               'temporal_diagnostic_completed': temporal,
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
