"""Category images depend only on the already verified source region."""
import cv2
import numpy as np
from PIL import Image, ImageDraw

from tools.evaluate_target_cards import fit_panel
from tools.grounded_target_views import target_anchor, validate_target


def focused_target_views(source, target, verified_box):
    validate_target(target)
    if target.shape != (source.height, source.width):
        raise ValueError('Source geometry differs')
    x1, y1, x2, y2 = verified_box
    if not (0 <= x1 < x2 <= source.width and 0 <= y1 < y2 <= source.height):
        raise ValueError('Invalid verified source box')
    local_mask = target[y1:y2, x1:x2]
    if int(local_mask.sum()) != int(target.sum()):
        raise ValueError('Verified region must retain the entire original target')
    # Crop first. Rendering and cue geometry below cannot access outside pixels.
    photo = source.crop((x1, y1, x2, y2)).convert('RGB')
    fitted, fit = fit_panel(photo, 512, 512)
    sx, sy = fit['display_size'][0] / photo.width, fit['display_size'][1] / photo.height
    ox, oy = fit['offset']
    display = lambda p: (round(ox + p[0] * sx), round(oy + p[1] * sy))
    tip = target_anchor(local_mask)
    tail = (round(tip[0] + (.3 * photo.width if tip[0] < photo.width / 2 else -.3 * photo.width)),
            round(tip[1] + (.3 * photo.height if tip[1] < photo.height / 2 else -.3 * photo.height)))
    tail = (max(1, min(photo.width - 2, tail[0])), max(1, min(photo.height - 2, tail[1])))
    pointed = np.array(fitted)
    cv2.arrowedLine(pointed, display(tail), display(tip), (255, 150, 0), thickness=3, tipLength=.1)
    boxed = fitted.copy(); draw = ImageDraw.Draw(boxed)
    yy, xx = np.nonzero(local_mask)
    left, top = display((int(xx.min()), int(yy.min())))
    right, bottom = display((int(xx.max()) + 1, int(yy.max()) + 1))
    length = max(6, min(18, (right - left) // 4, (bottom - top) // 3))
    for x, y, dx, dy in ((left, top, 1, 1), (right, top, -1, 1), (left, bottom, 1, -1), (right, bottom, -1, -1)):
        draw.line([(x + dx * length, y), (x, y), (x, y + dy * length)], fill=(255, 0, 255), width=3)
    return {'overview': Image.fromarray(pointed), 'linked': boxed}, {
        'verified_source_box': list(verified_box), 'source_region_size': list(photo.size),
        'original_target_pixels': int(target.sum()), 'retained_target_pixels': int(local_mask.sum()),
        'pointer_tip_in_original_target': bool(local_mask[tip[1], tip[0]]),
        'both_views_use_only_verified_region_pixels': True, 'whole_scene_and_neighbor_panels_removed': True,
        'original_annotation_geometry_changed': False, 'cues': ['foreground_pointer', 'target_bbox_corners'], 'fit': fit}
