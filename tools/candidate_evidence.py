"""Append label-blind candidate crops to real production VLM review calls."""
from contextlib import contextmanager
import hashlib
import io
import json

from PIL import Image,ImageDraw,ImageFont
from source.agent import workflow
from tools.evaluate_instance_focus import largest_mask_candidate,render_instance_views


def candidate_evidence_images(image,shapes):
    size=Image.open(io.BytesIO(image)).size
    target=largest_mask_candidate(shapes,*size)
    views,geometry=render_instance_views(image,target,padding=.1)
    font=ImageFont.load_default(size=48)
    encoded=[]
    for view,title in zip(views,('selected pixels','source crop','mask outline')):
        canvas=Image.new('RGB',(max(640,view.width),view.height+72),(245,245,245))
        canvas.paste(view,((canvas.width-view.width)//2,72))
        ImageDraw.Draw(canvas).text((12,8),f'ID {target["id"]} - {title}',fill=(0,0,0),font=font)
        buffer=io.BytesIO();canvas.save(buffer,'JPEG',quality=90)
        encoded.append(buffer.getvalue())
    metadata={'candidate_id':target['id'],'original_crop_box':geometry['original_crop_box'],
        'source_crop_size':[geometry['width'],geometry['height']],
        'header_font_pixels':48,'current_class_text_rendered':False,
        'image_sha256':[hashlib.sha256(raw).hexdigest() for raw in encoded]}
    return tuple(encoded),metadata


@contextmanager
def scoped_evidence_policy(policy,agent,on_bundle):
    if policy not in {'baseline','candidate_crops'}:raise ValueError('Unknown evidence policy')
    if policy=='baseline':yield;return
    original=workflow.chat
    def chat(config,instruction,image,extra_images=(),response_schema=None):
        if agent.shapes and 1+len(extra_images)+3<=8:
            images,metadata=candidate_evidence_images(image,agent.shapes)
            first=2+len(extra_images)
            hint=(f'\nAdditional images {first},{first+1},{first+2} show one CURRENT candidate, in order: '
                'its selected pixels on gray, its original photo crop, and its neutral mask outline. '
                'Their large header ID links to candidates, not reference instances. Current class metadata may be wrong: '
                'verify the object appearance in these photographs against its current label. '
                'Do not infer the object class from outline color or from a label alone. '
                'Reclassify only a visually supported wrong class; leave correct candidates unchanged. '
                'Inspect other candidates in the original target images as before. '
                'All tool coordinates still use the original target frame, not crop coordinates. '
                'Crop binding: '+json.dumps({key:metadata[key] for key in ('candidate_id','original_crop_box')},separators=(',',':')))
            on_bundle({'appended':True,'base_image_count':1+len(extra_images),'appended_images':3,**metadata})
            return original(config,instruction+hint,image,extra_images=(*extra_images,*images),response_schema=response_schema)
        on_bundle({'appended':False,'base_image_count':1+len(extra_images),'appended_images':0})
        return original(config,instruction,image,extra_images=extra_images,response_schema=response_schema)
    workflow.chat=chat
    try:yield
    finally:workflow.chat=original
