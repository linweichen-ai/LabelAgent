"""Fresh local visual class observations supplied to real VLM tool decisions.

No observation is converted into an operation here. The production review model
must choose all tools, and every review performs two new classifier requests.
"""
from contextlib import contextmanager
import hashlib
import io
import json
import time
import urllib.request
import uuid

from source.agent import workflow
from tools.connect_local_vlm import loopback_url
from tools.evaluate_instance_focus import largest_mask_candidate,render_instance_views,consensus_category
from tools.evaluate_vlm_grounding import build_payload,validated_response


def classify_online(config,image,shape,template,on_call):
    views,geometry=render_instance_views(image,shape,padding=.1)
    refs=[];assets={}
    for index,view in enumerate(views):
        encoded=io.BytesIO();view.save(encoded,'JPEG',quality=90);raw=encoded.getvalue();key=str(index)
        refs.append({'path':key,'sha256':hashlib.sha256(raw).hexdigest()});assets[key]=(raw,'image/jpeg')
    case={'views':{'single':refs[:1],'context':refs}}
    categories=[]
    opener=urllib.request.build_opener(urllib.request.ProxyHandler({}))
    for kind in ('single','context'):
        body,fingerprint=build_payload(template,assets,case,kind,config['llm_model'])
        event={'id':uuid.uuid4().hex,'status':'running','view':kind,'input_fingerprint':fingerprint,
            'image_sha256':[ref['sha256'] for ref in case['views'][kind]],'current_class_sent_to_classifier':False}
        on_call(event)
        started=time.monotonic()
        try:
            request=urllib.request.Request(loopback_url(config['llm_url'])+'/chat/completions',json.dumps(body).encode(),headers={'Content-Type':'application/json'})
            with opener.open(request,timeout=300) as response:raw=json.load(response)
            parsed=validated_response(raw,template['categories'],len(case['views'][kind]))
            event.update(status='succeeded',raw_response=raw,parsed=parsed,seconds=round(time.monotonic()-started,3))
            categories.append(parsed['category'])
        except Exception as error:
            event.update(status='failed',error_type=type(error).__name__)
            raise
        finally:on_call(event)
    return {'candidate_id':shape['id'],'single_observed_class':categories[0],'context_observed_class':categories[1],
        'two_view_consensus':consensus_category(*categories),'original_crop_box':geometry['original_crop_box'],
        'fresh_classifier_requests':2,'cached_class_responses_used':0}


@contextmanager
def scoped_online_observation(policy,agent,template,on_call,on_observation):
    if policy not in {'baseline','online_observation'}:raise ValueError('Unknown observation policy')
    if policy=='baseline':yield;return
    original=workflow.chat
    def chat(config,instruction,image,extra_images=(),response_schema=None):
        if agent.shapes:
            target=largest_mask_candidate(agent.shapes,agent.width,agent.height)
            observation=classify_online(config,image,target,template,on_call)
            observation['current_label_from_candidate']=target['label']
            on_observation(observation)
            hint=('\nFocused tool observation: '+json.dumps(observation,separators=(',',':'))
                +'\nThese are fallible fresh image observations, not reference annotations. '
                 'Compare this current candidate label with the observed classes and the target photographs. '
                 'Use the existing tools only for a visually supported defect; preserve correct candidates. '
                 'The observation is not an operation and does not choose a tool for you. '
                 'Choose the normal review JSON yourself. A reclassification to the same label makes no change.')
            return original(config,instruction+hint,image,extra_images=extra_images,response_schema=response_schema)
        return original(config,instruction,image,extra_images=extra_images,response_schema=response_schema)
    workflow.chat=chat
    try:yield
    finally:workflow.chat=original
