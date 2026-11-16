"""Classify the physical host of a marked part, not a cue or isolated dark patch."""
from tools.reference_semantics import reason_first_payload

PROMPT='Identify the PHYSICAL OBJECT that the marked visible part belongs to, using the photograph around the arrow tip or magenta corners. Do not classify the drawn marker or merely the color/shape of isolated pixels. A marked part can be a recess, opening or occluded piece of an identifiable object: for a pallet, a fork-entry opening framed by a deck/supports or a visible deck, block or runner belongs to the transport base; for a person, a sleeve, hand or limb with visible body/clothing continuity belongs to the human. The entire object need not be visible. Use only a coherent visible physical structure and the marked part connection, not a different nearby object. Cargo alone, floor, a cast shadow, reflection or isolated dark patch with no identifiable host are uncertain. A uniform blank, with or without drawn arrows/corners, has no host object and is uncertain. First describe the visible part-to-object connection in a short reason, then give category: pallet, person or uncertain. If no host can be established, abstain. Return reason BEFORE category. Do not invent missing structures.'

def part_object_payload(fixture,assets,case,view,guard=False):
    return reason_first_payload(fixture,assets,case,view,guard)
