"""Distinguish an empty clean scene from a low-texture selected component.

This is a global prompt candidate, not an engineering override of raw labels.
It retains the incumbent locator, image order, bytes and ordinary class schema.
"""
import hashlib,json
from tools.source_last import source_last_payload as BASE_PAYLOAD

SCENE_ABSENCE_INSTRUCTION=(
    'First inspect clean image 2 as a whole before using the locator. Do not '
    'assume a proposed detection exists. If the clean view consists only of a '
    'solid background or a smooth lighting gradient, with no physical object '
    'edges, joints, texture or connected structure, return uncertain regardless '
    'of any guide silhouette in image 1. This is a whole-clean-scene absence '
    'check: do not mistake a low-texture component within a real object scene '
    'for an empty scene. When physical objects are present, map the locator to '
    'the same selected source surface and its directly connected structure. '
    'A component can be small, smooth or partly occluded and still belong to '
    'a pallet or person under the original class definitions. Do not switch '
    'to another nearby object. In the reason, name structure actually visible '
    'at the selected source surface, not the colored guide.'
)

def scene_absence_payload(fixture,assets,case,view,guard=False):
    body,digest=BASE_PAYLOAD(fixture,assets,case,view,guard)
    if guard:return body,digest
    body=json.loads(json.dumps(body))
    body['messages'][0]['content'][0]['text']+='\n\n'+SCENE_ABSENCE_INSTRUCTION
    return body,hashlib.sha256(json.dumps(body,sort_keys=True,separators=(',',':')).encode()).hexdigest()
