"""Locate an indicated original target before sending any category request."""
import hashlib
import json

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from tools.evaluate_target_cards import fit_panel

GRID = 9
CELLS = tuple(f'{chr(65 + row)}{col + 1}' for row in range(GRID) for col in range(GRID))


def cell_box(cell, width, height):
    if cell not in CELLS or min(width, height) < GRID:
        raise ValueError('A supported source grid cell required')
    row, col = ord(cell[0]) - 65, int(cell[1:]) - 1
    return (col * width // GRID, row * height // GRID, (col + 1) * width // GRID, (row + 1) * height // GRID)


def validate_target(target):
    if not isinstance(target, np.ndarray) or target.dtype != bool or target.ndim != 2 or not target.any() or min(target.shape) < GRID:
        raise ValueError('Nonempty source-bound boolean target required')


def target_cells(target):
    validate_target(target); height, width = target.shape
    return [cell for cell in CELLS if location_ok(cell, target)]


def location_ok(cell, target):
    validate_target(target)
    if cell not in CELLS:
        return False
    x1, y1, x2, y2 = cell_box(cell, target.shape[1], target.shape[0])
    return bool(target[y1:y2, x1:x2].any())


def target_anchor(target):
    validate_target(target); yy, xx = np.nonzero(target)
    index = int(np.argmin((xx - xx.mean()) ** 2 + (yy - yy.mean()) ** 2))
    return int(xx[index]), int(yy[index])


def marked_source(source, target):
    source = source.convert('RGB')
    validate_target(target)
    if target.shape != (source.height, source.width):
        raise ValueError('Target and source dimensions differ')
    marked = source.copy(); draw = ImageDraw.Draw(marked)
    contours, _ = cv2.findContours(target.astype(np.uint8), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    for contour in contours:
        points = [tuple(p) for p in contour.reshape(-1, 2).tolist()]
        if len(points) >= 2:
            draw.line(points + points[:1], fill=(255, 0, 255), width=max(2, round(source.width / 640)))
    return marked


def localization_view(source, target):
    marked = marked_source(source, target); width, height = marked.size
    fitted, fit = fit_panel(marked, 960, 960)
    picture = Image.new('RGB', (1024, 1024), (127, 127, 127)); picture.paste(fitted, (32, 32))
    draw = ImageDraw.Draw(picture); font = ImageFont.load_default(size=16)
    ox, oy = 32 + fit['offset'][0], 32 + fit['offset'][1]
    sx, sy = fit['display_size'][0] / width, fit['display_size'][1] / height
    display = lambda p: (round(ox + p[0] * sx), round(oy + p[1] * sy))
    for cell in CELLS:
        x1, y1, x2, y2 = cell_box(cell, width, height)
        left, top = display((x1, y1)); right, bottom = display((x2, y2))
        draw.rectangle((left, top, right - 1, bottom - 1), outline=(190, 190, 190), width=1)
        box = draw.textbbox((left + 3, top + 2), cell, font=font)
        draw.rectangle(box, fill=(30, 30, 30)); draw.text((left + 3, top + 2), cell, font=font, fill=(255, 255, 255))
    tip = target_anchor(target)
    tail = (round(tip[0] + (.15 * width if tip[0] < width / 2 else -.15 * width)),
            round(tip[1] + (.12 * height if tip[1] < height / 2 else -.12 * height)))
    tail = (max(1, min(width - 2, tail[0])), max(1, min(height - 2, tail[1])))
    cv = np.array(picture)
    cv2.arrowedLine(cv, display(tail), display(tip), (255, 150, 0), thickness=3, tipLength=.08)
    return Image.fromarray(cv), {'grid': GRID, 'arrow_tip': list(tip), 'display_arrow_tip': list(display(tip)),
                     'grid_font_display_pixels': 16, 'whole_given_source_retained': True,
                     'acceptable_cells_from_input_geometry': target_cells(target), 'fit': fit,
                     'class_or_expected_cell_rendered_as_answer': False}


def classification_views(source, target, selected_cell):
    if not location_ok(selected_cell, target):
        raise ValueError('Localization failed: do not prepare or send class requests')
    marked = marked_source(source, target); width, height = marked.size
    yy, xx = np.nonzero(target); bounds = (int(xx.min()), int(yy.min()), int(xx.max()) + 1, int(yy.max()) + 1)
    x1, y1, x2, y2 = cell_box(selected_cell, width, height)
    focus = (min(x1, bounds[0]), min(y1, bounds[1]), max(x2, bounds[2]), max(y2, bounds[3]))
    row, col = ord(selected_cell[0]) - 65, int(selected_cell[1:]) - 1
    context = (max(0, col - 1) * width // GRID, max(0, row - 1) * height // GRID,
               min(GRID, col + 2) * width // GRID, min(GRID, row + 2) * height // GRID)
    context = (min(context[0], bounds[0]), min(context[1], bounds[1]), max(context[2], bounds[2]), max(context[3], bounds[3]))
    assert int(target[focus[1]:focus[3], focus[0]:focus[2]].sum()) == int(target.sum())
    whole = marked.copy(); ImageDraw.Draw(whole).rectangle((focus[0], focus[1], focus[2] - 1, focus[3] - 1), outline=(0, 220, 255), width=3)
    focused, surrounding = marked.crop(focus), marked.crop(context)
    overview, fit = fit_panel(focused, 512, 512)
    linked = Image.new('RGB', (1536, 512), (127, 127, 127)); draw = ImageDraw.Draw(linked)
    font = ImageFont.load_default(size=14); panels = []
    for i, (title, picture) in enumerate(zip(('CELL', 'CONTEXT', 'WHOLE'), (focused, surrounding, whole))):
        panel, info = fit_panel(picture, 480, 480); left = 24 + 504 * i
        linked.paste(panel, (left, 16)); draw.text((left + 4, 0), title, font=font, fill=(240, 240, 240)); panels.append(info)
    return {'overview': overview, 'linked': linked}, {'selected_cell': selected_cell, 'focus_box': list(focus),
        'context_box': list(context), 'original_target_pixels': int(target.sum()),
        'retained_target_pixels': int(target[focus[1]:focus[3], focus[0]:focus[2]].sum()),
        'isolated_gray_target_panel_removed': True, 'target_not_shrunk': True, 'fit': fit, 'panels': panels}


def localization_payload(fixture, asset, case):
    raw, mime = asset
    if hashlib.sha256(raw).hexdigest() != case['localization_image']['sha256']:
        raise ValueError('Localization asset differs')
    import base64
    schema = {'type': 'object', 'properties': {'cell': {'type': 'string', 'enum': [*CELLS, 'uncertain']},
        'reason': {'type': 'string', 'minLength': 1, 'maxLength': 100}}, 'required': ['cell', 'reason'], 'additionalProperties': False}
    return {'model': fixture['model'], 'temperature': 0, 'max_tokens': 192, 'image_token_budget': 1024,
        'messages': [{'role': 'user', 'content': [{'type': 'text', 'text': fixture['localization_prompt']},
            {'type': 'image_url', 'image_url': {'url': 'data:' + mime + ';base64,' + base64.b64encode(raw).decode('ascii')}}]}],
        'response_format': {'type': 'json_schema', 'json_schema': {'name': 'target_location', 'schema': schema, 'strict': True}}}


def validated_location(raw):
    meta = raw.get('local_vlm', {})
    choice = raw['choices'][0]
    if meta.get('backend') != 'transformers_local_cuda' or meta.get('image_count') != 1 or choice.get('finish_reason') != 'stop':
        raise ValueError('A complete real CUDA localization response required')
    parsed = json.loads(choice['message']['content'])
    if set(parsed) != {'cell', 'reason'} or parsed['cell'] not in (*CELLS, 'uncertain') or not isinstance(parsed['reason'], str) or not 1 <= len(parsed['reason']) <= 100:
        raise ValueError('Invalid localization output')
    return parsed
