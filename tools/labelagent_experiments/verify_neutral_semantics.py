import json,sys,io
from pathlib import Path
from PIL import Image
from diagnose_semantic_contract import request,save
from semantic_evidence_gate import CAPTION_PROMPT,CAPTION_SCHEMA,pallet_evidence
ROOT=Path(__file__).resolve().parents[2];out=ROOT/'outputs/labelagent_semantic_recovery_20261105/neutral_controls';out.mkdir(exist_ok=True)
if (out/'completed.json').exists():raise FileExistsError('Preserve controls')
raw=Image.open(ROOT/'outputs/labelagent_preserved_model_comparison_20261105/inputs/camera0_frame76.png').convert('RGB')
cases={'gray':Image.new('RGB',(256,256),'gray'),'white':Image.new('RGB',(256,256),'white'),'black':Image.new('RGB',(256,256),'black'),
 'red':Image.new('RGB',(256,256),'red'),'raw_floor':raw.crop((250,850,700,1200)),'raw_robot':raw.crop((641,398,929,665)),
 'raw_person':raw.crop((1540,1050,1920,1536)), 'pallet_stack':raw.crop((1470,420,1675,545)), 'wood_pallet_base':raw.crop((1110,1000,1410,1080))}
rows=[]
for name,im in cases.items():
 im.save(out/(name+'.png'));file=out/(name+'.json');r=json.loads(file.read_text(encoding='utf-8')) if file.exists() else request(im,CAPTION_PROMPT,CAPTION_SCHEMA);save(file,r)
 row={'case':name,'decision':r['decision'],'pallet_evidence':pallet_evidence(r['decision']),'usage':r['usage'],'runtime':r['runtime']};rows.append(row);save(out/'progress.json',rows);print(json.dumps(row,ensure_ascii=False),flush=True)
save(out/'completed.json',{'status':'complete','rows':rows,'negative_controls_passed':all(not r['pallet_evidence'] for r in rows if r['case'] not in ['pallet_stack','wood_pallet_base']),'positive_diagnostic_recall':sum(r['pallet_evidence'] for r in rows if r['case'] in ['pallet_stack','wood_pallet_base']),'positive_diagnostics':2,'not_generalization':True})
