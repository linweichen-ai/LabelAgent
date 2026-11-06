"""Preserve source context while pointing inside the proposed visible contour.

Geometry is derived from a proposed mask, not human localization truth. The
source-only inspection is reused unchanged as a separate diagnostic. Its
visibility, point and feature description never enter classification requests.
"""
import hashlib
import json

import cv2
import numpy as np
from PIL import Image, ImageDraw

from tools.evaluate_target_cards import fit_panel
from tools.evaluate_vlm_grounding import build_payload
from tools.grounded_target_views import target_anchor
from tools.target_first import grounding_payload


COMMON = (
    'Two images show exactly the SAME original photograph crop. Image 1 is clean '
    'source evidence without guides. Image 2 repeats that source with a complete '
    'MAGENTA outline of the selected visible surface. Use the outline only to '
    'locate the source surface in image 1; guide lines are not objects. '
    'Verify the source pixels. If image 1 is only a uniform empty '
    'field, return uncertain even when image 2 has an outline or arrow. '
)
TARGETS = {
    'overview': (
        'Classify the physical surface at the ORANGE ARROW TIP inside the MAGENTA '
        'contour. The tip is inside the proposed visible surface, not on its '
        'bounding-box edge. Follow the actual source surface and directly '
        'connected structure; do not switch to nearby cargo or machinery. '
    ),
    'linked': (
        'Classify the physical surface enclosed by the MAGENTA contour, not the '
        'surrounding bounding box or a nearby prominent object. Follow the '
        'actual source surface and directly connected structure; do not switch '
        'to nearby cargo or machinery. '
    ),
}
DEFINITIONS = (
    'A pallet is a manufactured load-bearing transport platform, including a '
    'visible deck, block, runner or base. A person is a human body or part with '
    'visible anatomical, clothing or body continuity. Curvature or a machine '
    'silhouette alone does not establish a person; nearby cargo alone does not '
    'establish a pallet. If the selected source surface cannot be distinguished '
    'or has no evidence of either class, return uncertain. Return category and '
    'one short reason describing the selected physical surface.'
)


def render(source, mask, source_box):
    """Draw only complete contours and an interior pointer on a fixed RGB crop."""
    source = source.convert('RGB')
    if (not isinstance(mask, np.ndarray) or mask.dtype != bool
            or mask.shape != (source.height, source.width) or not mask.any()):
        raise ValueError('Nonempty source-bound boolean mask required')
    if len(source_box) != 4 or any(type(v) is not int for v in source_box):
        raise ValueError('Integer source crop required')
    x1, y1, x2, y2 = source_box
    if not (0 <= x1 < x2 <= source.width and 0 <= y1 < y2 <= source.height):
        raise ValueError('Source crop outside photograph')
    cropped = mask[y1:y2, x1:x2]
    if int(cropped.sum()) != int(mask.sum()):
        raise ValueError('Source crop would discard proposed target pixels')
    # Zero padding treats the image edge as a real boundary, including when the
    # selected region fills its source crop. DIST_L2/5 is a fixed approximation.
    padded = np.pad(cropped.astype(np.uint8), 1, mode='constant', constant_values=0)
    distances = cv2.distanceTransform(padded, cv2.DIST_L2, 5)[1:-1, 1:-1]
    maximum = float(distances.max())
    yy, xx = np.nonzero(cropped)
    cy, cx = float(yy.mean()), float(xx.mean())
    ty, tx = np.nonzero(distances == maximum)
    selected = int(np.argmin((tx - cx) ** 2 + (ty - cy) ** 2))
    ax, ay = int(tx[selected]), int(ty[selected])
    assert cropped[ay, ax] and maximum > 0
    old_x, old_y = target_anchor(mask)
    old_distance = float(distances[old_y - y1, old_x - x1])

    clean, fit = fit_panel(source.crop(tuple(source_box)), 512, 512)
    outlined = clean.copy()
    draw = ImageDraw.Draw(outlined)
    sx = fit['display_size'][0] / cropped.shape[1]
    sy = fit['display_size'][1] / cropped.shape[0]
    ox, oy = fit['offset']
    def display(x, y):
        return (round(ox + (x + .5) * sx - .5), round(oy + (y + .5) * sy - .5))
    contours, _ = cv2.findContours(cropped.astype(np.uint8), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    for contour in contours:
        points = [display(int(x), int(y)) for x, y in contour.reshape(-1, 2)]
        if len(points) >= 2:
            draw.line(points + [points[0]], fill=(255, 0, 255), width=2)
        elif points:
            draw.point(points[0], fill=(255, 0, 255))
    tip = display(ax, ay)
    tail = (round(tip[0] + (92 if tip[0] < 256 else -92)),
            round(tip[1] + (72 if tip[1] < 256 else -72)))
    tail = tuple(max(2, min(509, value)) for value in tail)
    arrowed = np.array(outlined)
    cv2.arrowedLine(arrowed, tail, tip, (255, 150, 0), thickness=2,
                   line_type=cv2.LINE_8, tipLength=.12)
    metadata = dict(
        source_box=list(source_box), fit=fit, original_target_pixels=int(mask.sum()),
        retained_target_pixels=int(cropped.sum()), anchor_source=[ax + x1, ay + y1],
        display_arrow_tip=list(tip), old_anchor_source=[old_x, old_y],
        max_distance=maximum, old_anchor_distance=old_distance,
        distance_method='cv2 DIST_L2 maskSize=5 with one zero border',
        tie_break='nearest proposed-mask centroid, then row-major order',
        all_proposed_target_pixels_retained=True, contour_width=2,
        contour_count=len(contours), filled_overlay=False,
        class_or_ground_truth_used=False, original_source_crop_unchanged=True,
        proposed_mask_not_human_localization_truth=True)
    return {'overview': Image.fromarray(arrowed), 'linked': outlined}, metadata


def classify_payload(fixture, assets, case, view, grounding):
    if view not in TARGETS or len(case['views'][view]) != 2:
        raise ValueError('Two matched clean/outlined source images required')
    # Signature compatibility only: the fallible old inspection is not evidence
    # for this new query. Never propagate its visibility, point or descriptions.
    prompt = COMMON + TARGETS[view] + DEFINITIONS
    neutral = dict(categories=fixture['categories'], temperature=0, max_tokens=192,
                   prompts={'query': prompt})
    body, _ = build_payload(neutral, assets, {'views': {'query': case['views'][view]}}, 'query', fixture['model'])
    body['response_format']['json_schema']['schema']['properties']['reason'] = {
        'type': 'string', 'pattern': '[ -~]{1,100}'}
    # No inspection fields, class guesses, or truth enter this stage.
    digest = hashlib.sha256(json.dumps(body, sort_keys=True, separators=(',', ':')).encode()).hexdigest()
    return body, digest
