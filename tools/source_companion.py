"""Expose unchanged source pixels separately from drawn location guides."""
from tools.ascii_target_recheck import ascii_card_payload
from tools.evaluate_target_cards import fit_panel

PREFIX = ('Two images show exactly the SAME source crop. Image 1 is the photograph without any location markers. '
          'Image 2 repeats it with the target location marked. Use image 2 only to locate the target in image 1. '
          'An arrow or magenta corner is not physical evidence. If image 1 contains only a uniform empty field, '
          'there is no object even if image 2 has markers: return uncertain. Otherwise classify the physical object '
          'at that SAME location from image 1; do not switch to another nearby object. ')

def unmarked_companion(source, source_box):
    if len(source_box) != 4 or any(type(v) is not int for v in source_box):
        raise ValueError('An exact integer source box is required')
    x1,y1,x2,y2=source_box
    if not (0 <= x1 < x2 <= source.width and 0 <= y1 < y2 <= source.height):
        raise ValueError('Source box is outside the photograph')
    return fit_panel(source.crop(tuple(source_box)),512,512)

def source_companion_payload(fixture, assets, case, view, guard=False):
    if not guard and len(case['views'][view]) != 2:
        raise ValueError('Exactly one clean source and one marked query required')
    return ascii_card_payload(fixture,assets,case,view,guard)
