"""Query-only, evidence-first open-set review without outside class filters."""
from tools.reference_semantics import reason_first_payload

PROMPT='This is an annotation REVIEW crop, not a guaranteed positive object. There may be no physical target. First describe the actual photographic evidence at the arrow tip or inside the magenta corners; ignore all drawn guides as object appearance. Then choose a category consistent with that evidence. A uniform gray or white area remains empty even when arrows or boxes are drawn on it: uncertain. Reflections, shadows, an isolated indistinct patch, floor, cargo and machine parts without a recognizable target are uncertain. A person needs identifiable human anatomy or visible clothing/body continuity; a curved dark shape alone is not a person. A pallet is a manufactured transport base: deck, support block or runner, including a visible part clearly belonging to such a base. Do not require the entire pallet, but nearby cargo alone does not prove a pallet. Do not invent anatomy or platform structure. If your evidence cannot identify the part, choose uncertain rather than guessing. Write a short reason FIRST and category LAST. Return fields in order: reason, category.'

def cautious_payload(fixture,assets,case,view,guard=False):
    return reason_first_payload(fixture,assets,case,view,guard)
