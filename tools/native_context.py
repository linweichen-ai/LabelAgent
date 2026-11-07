"""Add unmarked native source pixels without replacing locator or context."""
import base64,hashlib,json
from PIL import Image
from tools.source_last import source_last_payload as BASE_PAYLOAD
from tools.contour_visibility import TARGETS, DEFINITIONS
from tools.evaluate_vlm_grounding import validated_response as BASE_VALIDATOR
from tools.evaluate_target_first import validate_backend as BASE_BACKEND

COMMON=(
    'Three images refer to the SAME original photograph. Image 1 is the original '
    'locator with a complete MAGENTA contour of the selected visible surface. '
    'Image 2 is an unmarked native-photo neighborhood around the same selected '
    'interior point. Image 3 is the original unmarked context crop. These are '
    'overlapping views; the neighborhood may be tighter or wider than the old '
    'context crop. Use image 1 only to locate the physical surface in images 2 '
    'and 3. Use actual source material and directly connected structure; colored '
    'guide lines, arrows and their silhouettes are not physical objects. '
    'If the clean source views do not provide physical target evidence, return uncertain. '
)


def native_window(source,anchor,side=128):
    if source.mode!='RGB' or type(side) is not int or side<1:
        raise ValueError('RGB source and positive integer side required')
    if len(anchor)!=2 or any(type(v) is not int for v in anchor):
        raise ValueError('Integer source anchor required')
    ax,ay=anchor;w,h=source.size
    if not(0<=ax<w and 0<=ay<h):raise ValueError('Anchor outside original source')
    span=min(side,w,h);x=max(0,min(w-span,ax-span//2));y=max(0,min(h-span,ay-span//2))
    box=[x,y,x+span,y+span];crop=source.crop(tuple(box))
    return crop.resize((512,512),Image.Resampling.BICUBIC),dict(native_crop_box=box,native_crop_side=span,anchor_in_crop=[ax-x,ay-y],
        raw_crop_rgb_sha256=hashlib.sha256(crop.tobytes()).hexdigest(),no_padding_or_guides_added=True,resampling='PIL BICUBIC')


def native_context_payload(fixture,assets,case,view,guard=False):
    if guard:return BASE_PAYLOAD(fixture,assets,case,view,True)
    refs=case['views'][view]
    if len(refs)!=3:raise ValueError('Clean context, locator and native patch refs required')
    original_case=dict(case,views={view:refs[:2]})
    body,_=BASE_PAYLOAD(fixture,assets,original_case,view)
    patch=refs[2];raw,mime=assets[patch['path']]
    if hashlib.sha256(raw).hexdigest()!=patch['sha256']:raise ValueError('Native patch bytes changed')
    body['messages'][0]['content'][0]['text']=COMMON+TARGETS[view]+DEFINITIONS
    body['messages'][0]['content'].insert(2,dict(type='image_url',image_url=dict(url='data:'+mime+';base64,'+base64.b64encode(raw).decode())))
    return body,hashlib.sha256(json.dumps(body,sort_keys=True,separators=(',',':')).encode()).hexdigest()


def validate_response(raw,categories,image_count):
    actual=raw.get('local_vlm',{}).get('image_count')
    return BASE_VALIDATOR(raw,categories,3 if actual==3 else image_count)


def validate_backend(raw,fixture,image_count,guard=False):
    actual=raw.get('local_vlm',{}).get('image_count')
    return BASE_BACKEND(raw,fixture,3 if actual==3 else image_count,guard)
