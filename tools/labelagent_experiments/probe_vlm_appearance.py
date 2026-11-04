"""Actual source-crop controls: can the VLM name visible foreground without masks?"""
import json,base64,io,time,urllib.request,urllib.error,hashlib
from pathlib import Path
from PIL import Image
ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/'outputs/labelagent_hybrid_repair_20261104/vlm_appearance_controls'
def main():
    OUT.mkdir(exist_ok=True)
    source=ROOT/'outputs/labelagent_hybrid_comparison_v3_20261104/inputs/camera0_frame76.png'
    image=Image.open(source).convert('RGB')
    cases=[('visible_robot',(641,398,929,665)),('visible_floor',(250,850,700,1200)),('visible_person',(1500,850,1920,1500))]
    plan={'purpose':'three exposed visually selected diagnostic crops, not a held-out accuracy set','source_sha256':hashlib.sha256(source.read_bytes()).hexdigest(),'cases':cases,'mask_overlays':False,'annotation_writes':0}
    (OUT/'plan.json').write_text(json.dumps(plan,indent=2)+'\n',encoding='utf-8');rows=[]
    for name,box in cases:
        file=OUT/(name+'.png');crop=image.crop(box);crop.save(file)
        body={'model':'labelagent-vlm','temperature':0,'max_tokens':128,
            'messages':[{'role':'user','content':[{'type':'text','text':'Describe the main visible object or surface at the center of this uncolored photo in one short English sentence. Do not infer from any mask; no masks are provided. Return JSON with object.'},
            {'type':'image_url','image_url':{'url':'data:image/png;base64,'+base64.b64encode(file.read_bytes()).decode()}}]}],
            'response_format':{'type':'json_schema','json_schema':{'name':'appearance','strict':True,'schema':{'type':'object','properties':{'object':{'type':'string','minLength':1,'maxLength':120}},'required':['object'],'additionalProperties':False}}}}
        for attempt in range(100):
            try:
                request=urllib.request.Request('http://127.0.0.1:18868/v1/chat/completions',json.dumps(body).encode(),headers={'Content-Type':'application/json'})
                with urllib.request.urlopen(request,timeout=180) as response:r=json.load(response)
                break
            except urllib.error.HTTPError as error:
                text=error.read().decode()
                if error.code!=503 or 'Another model request' not in text:raise
                time.sleep(2)
        else:raise RuntimeError('Model remained busy')
        rows.append({'case':name,'response':r['choices'][0]['message']['content'],'usage':r['usage']})
        (OUT/'responses.json').write_text(json.dumps(rows,indent=2)+'\n',encoding='utf-8');print(json.dumps(rows[-1]),flush=True)
if __name__=='__main__':main()
