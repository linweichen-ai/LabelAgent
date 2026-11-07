"""Matched-budget guide ablation: identical text, native crop and context."""
import copy,hashlib,json
from tools.native_context import native_context_payload as BASE_NATIVE
from tools.contour_visibility import DEFINITIONS

COMMON=(
    'Three views refer to the SAME source photograph. Image 1 is a context view '
    'that may contain a locator guide; any guide only points to source material '
    'and is not physical object evidence. Image 2 is an unmarked native-photo '
    'neighborhood around the selected interior point. Image 3 is unmarked context. '
    'Use actual material and directly connected structure in the source views. '
)
QUESTIONS={
    'overview':('Classify the selected physical surface covered by the native neighborhood in image 2. '
                'Locate the same source surface in the context; do not switch to nearby cargo, machinery or other people. '),
    'linked':('Classify the real object whose selected visible surface is shown in image 2. '
              'Use directly connected physical structure across the source views; a guide silhouette alone cannot establish class. '),
}


def paired_guide_payload(fixture,assets,case,view,guard=False):
    body,digest=BASE_NATIVE(fixture,assets,case,view,guard)
    if guard:return body,digest
    body['messages'][0]['content'][0]['text']=COMMON+QUESTIONS[view]+DEFINITIONS
    if not fixture['painted_locator']:
        body['messages'][0]['content'][1]=copy.deepcopy(body['messages'][0]['content'][3])
    return body,hashlib.sha256(json.dumps(body,sort_keys=True,separators=(',',':')).encode()).hexdigest()
