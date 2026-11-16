"""Reduce contour salience while retaining geometry, context and pointer alpha."""
import cv2
import numpy as np
from PIL import Image
from tools.contour_visibility import render
from tools.evaluate_target_cards import fit_panel
from tools.faint_pointer import faint_pointer


def render_contour_alpha(source, mask, source_box, alpha=.25):
    if not 0 < alpha <= 1:
        raise ValueError('Contour alpha must lie in (0,1]')
    opaque, meta = render(source, mask, source_box)
    clean, fit = fit_panel(source.convert('RGB').crop(tuple(source_box)), 512, 512)
    assert fit == meta['fit']
    linked = Image.blend(clean, opaque['linked'], alpha)
    tip = tuple(meta['display_arrow_tip'])
    tail = (round(tip[0] + (92 if tip[0] < 256 else -92)),
            round(tip[1] + (72 if tip[1] < 256 else -72)))
    tail = tuple(max(2, min(509, v)) for v in tail)
    arrowed = np.array(linked)
    cv2.arrowedLine(arrowed, tail, tip, (255, 150, 0), thickness=2,
                   line_type=cv2.LINE_8, tipLength=.12)
    overview = faint_pointer(linked, Image.fromarray(arrowed))
    # The alpha=1 control must reconstruct the existing quarter-arrow input.
    if alpha == 1:
        assert np.array_equal(np.array(linked), np.array(opaque['linked']))
        assert np.array_equal(np.array(overview), np.array(faint_pointer(opaque['linked'], opaque['overview'])))
    meta = dict(meta, contour_alpha=alpha, arrow_alpha=.25,
                clean_source_unchanged=True, geometry_unchanged=True,
                class_or_expected_answer_used=False)
    return dict(overview=overview, linked=linked), meta
