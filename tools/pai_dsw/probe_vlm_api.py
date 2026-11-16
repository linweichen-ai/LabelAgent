"""Probe a loopback OpenAI-compatible VLM without changing LabelAgent settings.

Works with native vLLM; does not require the old custom /health or local_vlm
metadata. A passed probe establishes API/image/JSON compatibility, not accuracy
or independent proof of the physical GPU backend.
"""
import argparse
import base64
import getpass
import hashlib
import json
import mimetypes
import os
from pathlib import Path
import time
from urllib.parse import urlparse
from urllib.request import build_opener,ProxyHandler,Request


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--base',default='http://127.0.0.1:18878/v1')
    p.add_argument('--model',default='labelagent-vlm32b')
    p.add_argument('--image',type=Path,action='append',required=True)
    p.add_argument('--output',type=Path,required=True)
    args=p.parse_args()
    parsed=urlparse(args.base)
    if (parsed.scheme!='http' or parsed.hostname not in {'127.0.0.1','localhost','::1'}
        or parsed.username or parsed.password or parsed.query or parsed.fragment or not parsed.path.endswith('/v1')):
        p.error('Use the loopback endpoint reached through your authenticated SSH tunnel')
    if not 1<=len(args.image)<=8:p.error('Provide one to eight real images')
    if args.output.exists():p.error('Choose a fresh output file; existing receipts are not overwritten')
    key=os.environ.get('LABELAGENT_VLM_KEY') or getpass.getpass('VLM API key (hidden): ')
    if not key:p.error('An API key is required')
    opener=build_opener(ProxyHandler({}))
    def request(suffix,body=None):
        data=None if body is None else json.dumps(body).encode('utf-8')
        req=Request(args.base.rstrip('/')+suffix,data=data,headers={'Content-Type':'application/json','Authorization':'Bearer '+key})
        with opener.open(req,timeout=300) as r:return json.load(r)
    models=request('/models')
    if args.model not in {m['id'] for m in models.get('data',[])}:raise ValueError('Requested model alias is not served')
    content=[{'type':'text','text':'Inspect the supplied photographs. Return a JSON object with visible_objects (short object class names), image_count (number of input pictures), and reason (one short sentence of visible evidence). A blank image may have no objects. Do not invent hidden objects.'}]
    images=[]
    for path in args.image:
        raw=path.read_bytes();mime=mimetypes.guess_type(path.name)[0]
        if mime not in {'image/png','image/jpeg','image/webp'}:raise ValueError('Use PNG/JPEG/WebP images')
        content.append({'type':'image_url','image_url':{'url':f'data:{mime};base64,'+base64.b64encode(raw).decode('ascii')}})
        images.append({'sha256':hashlib.sha256(raw).hexdigest(),'bytes':len(raw)})
    schema={'type':'object','properties':{
        'visible_objects':{'type':'array','maxItems':6,'items':{'type':'string','minLength':1,'maxLength':40}},
        'image_count':{'type':'integer','minimum':1,'maximum':8},
        'reason':{'type':'string','minLength':1,'maxLength':180}},
        'required':['visible_objects','image_count','reason'],'additionalProperties':False}
    body={'model':args.model,'temperature':0,'max_tokens':384,
        'messages':[{'role':'user','content':content}],
        'response_format':{'type':'json_schema','json_schema':{'name':'labelagent_probe','strict':True,'schema':schema}}}
    start=time.monotonic();raw=request('/chat/completions',body);elapsed=round(time.monotonic()-start,3)
    choice=raw['choices'][0]
    try:answer=json.loads(choice['message']['content'])
    except (ValueError,TypeError):answer=None
    valid=(raw.get('model')==args.model and choice.get('finish_reason')=='stop' and isinstance(answer,dict)
        and set(answer)=={'visible_objects','image_count','reason'} and type(answer['image_count']) is int
        and answer['image_count']==len(args.image) and isinstance(answer['visible_objects'],list)
        and len(answer['visible_objects'])<=6 and all(isinstance(s,str) and 1<=len(s)<=40 for s in answer['visible_objects'])
        and isinstance(answer['reason'],str) and 1<=len(answer['reason'])<=180)
    receipt={'api_image_json_compatible':valid,'base':args.base,'model':args.model,'input_images':images,
        'seconds':elapsed,'answer':answer,'usage':raw.get('usage'),'finish_reason':choice.get('finish_reason'),
        'settings_modified':False,'full_agent_verified':False,'annotation_accuracy_verified':False,
        'physical_CUDA_backend_attested_by_this_HTTP_probe':False,'raw_response':raw}
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(receipt,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    print(json.dumps({k:receipt[k] for k in ('api_image_json_compatible','model','seconds','answer')},ensure_ascii=False))
    if not valid:raise SystemExit('Probe did not meet the strict image/JSON requirements; settings were not changed')


if __name__=='__main__':main()
