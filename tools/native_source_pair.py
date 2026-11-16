"""Native selected-source neighborhood first, clean context second; no guides."""
import copy,hashlib,json
from tools.native_context import native_context_payload as BASE_NATIVE
from tools.contour_visibility import DEFINITIONS

COMMON=(
    'Two unmarked views come from the SAME source photograph. Image 1 is a '
    'native-photo neighborhood around the selected interior point; image 2 '
    'provides its wider unmarked context. Use image 1 as the selected source '
    'focus and image 2 only for directly connected physical structure. Do not '
    'switch to a separate nearby object merely because it dominates image 2. '
    'Do not assume a proposed object exists. If both source views provide only '
    'a solid background or smooth lighting gradient without physical object '
    'structure, return uncertain. Neutral layout padding is not an object. '
    'A small or low-texture component within a real object scene is not by '
    'itself evidence that the whole scene is empty. '
)
QUESTIONS={
    'overview':('Classify the selected physical surface shown in image 1. '
                'Use image 2 to resolve its actual material and attached support or body structure. '),
    'linked':('Identify the selected source component in image 1, then classify '
              'the real object it directly belongs to using image 2. Do not select a different adjacent object. '),
}

def native_source_pair_payload(fixture,assets,case,view,guard=False):
    body,digest=BASE_NATIVE(fixture,assets,case,view,guard)
    if guard:return body,digest
    parts=body['messages'][0]['content']
    body['messages'][0]['content']=[dict(type='text',text=COMMON+QUESTIONS[view]+DEFINITIONS),copy.deepcopy(parts[2]),copy.deepcopy(parts[3])]
    return body,hashlib.sha256(json.dumps(body,sort_keys=True,separators=(',',':')).encode()).hexdigest()
