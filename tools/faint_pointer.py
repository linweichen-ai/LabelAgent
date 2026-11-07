"""Change only arrow opacity; retain opaque original contour and source crop."""
from PIL import Image

ARROW_ALPHA=0.25

def faint_pointer(contour_only,arrowed):
    if contour_only.mode!='RGB' or arrowed.mode!='RGB' or contour_only.size!=arrowed.size:
        raise ValueError('Matched RGB locator panels required')
    return Image.blend(contour_only,arrowed,ARROW_ALPHA)
