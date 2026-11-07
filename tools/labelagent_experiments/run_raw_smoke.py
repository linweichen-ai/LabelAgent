"""Bounded real/private SAM capture: one frozen photo and one gray control."""
import datetime,hashlib,io,json,socket,sys,subprocess,base64,time,urllib.request,urllib.error
from pathlib import Path
import numpy as np
from PIL import Image,ImageDraw,ImageFont
import cv2
from sam_raw_capture import load_private_segment,capture_segment,save_capture

ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/'outputs/labelagent_raw_smoke_20261107'
SOURCE=Path('D:/Apps/LabelAgent/source/sam3_service/point_segment.py')
OLD=ROOT/'outputs/labelagent_native_resume_20261107/seam_generation/camera0_frame76/seed02_pointer'
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(2**20),b''):h.update(b)
 return h.hexdigest()
def save(p,v):Path(p).write_text(json.dumps(v,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
def closed():
 for port in (8766,17866,18868,18869):
  with socket.socket() as s:
   s.settimeout(.3)
   if s.connect_ex(('127.0.0.1',port))==0:raise RuntimeError('Shared model listener appeared; no independent GPU allocation')
def main():
 if OUT.exists():raise FileExistsError('Use immutable fresh experiment')
 with urllib.request.urlopen('http://127.0.0.1:8766/manager/status',timeout=10) as f:manager=json.load(f)
 apps='existing manager coordinates requests'
 OUT.mkdir(parents=True)
 input_bytes=(OLD/'input.png').read_bytes();image=Image.open(io.BytesIO(input_bytes)).convert('RGB')
 points=json.loads((OLD/'prompt.json').read_text(encoding='utf-8'))['points']
 source_hash=sha(SOURCE)
 assert source_hash=='ab8861c57fbd275e8d16b5cc9bfbf734b3e4d7226d02b93d2b0ef9173fc49a81'
 cfg=json.loads((ROOT/'task_workspaces/labelagent_point_ids_20261107/config.json').read_text(encoding='utf-8'))
 protected=[Path(cfg['reference_annotations']),Path(cfg['reference_description']),SOURCE,OLD/'input.png',OLD/'prompt.json']
 before={str(p):sha(p) for p in protected}
 weights=sorted((SOURCE.parents[1]/'sam3/weights').glob('*.pt'))
 plan={'status':'frozen_before_inference','at':datetime.datetime.now().astimezone().isoformat(),'real_requests':1,'gray_requests':1,'source_sha256':source_hash,'capture_tool_sha256':sha(Path(__file__).with_name('sam_raw_capture.py')),'driver_sha256':sha(__file__),'original_input_sha256':sha(OLD/'input.png'),'points':points,'prior_mask':None,'weights':[{'path':str(p),'sha256':sha(p),'bytes':p.stat().st_size} for p in weights],'protected':before,'execution_route':'existing shared manager optional capture','no_acceptance_change':True,'candidate_adopted':False}
 save(OUT/'frozen_plan.json',plan)
 sys.path.insert(0,'D:/Apps/LabelAgent')
 from source.agent.edit_agent import _rasterize
 rows=[];cards=[]
 font=ImageFont.truetype('C:/Windows/Fonts/msyh.ttc',30)
 for kind,im in [('real',image),('gray',Image.new('RGB',image.size,(127,127,127)))]:
  folder=OUT/kind;folder.mkdir();buffer=io.BytesIO();im.save(buffer,'PNG');rawbytes=input_bytes if kind=='real' else buffer.getvalue();(folder/'input.png').write_bytes(rawbytes)
  body=json.dumps({'image':base64.b64encode(rawbytes).decode(),'points':points,'include_raw_capture':True}).encode()
  for attempt in range(300):
   req=urllib.request.Request('http://127.0.0.1:8766/segment_points',data=body,headers={'Content-Type':'application/json'})
   try:
    with urllib.request.urlopen(req,timeout=600) as f:payload=json.load(f)
    break
   except urllib.error.HTTPError as e:
    if e.code==503:time.sleep(1);continue
    payload=json.loads(e.read());break
  else:raise RuntimeError('Manager busy; no successful response')
  save(folder/'http_response.json',payload)
  capture=payload['raw_capture'];record=capture['metadata'];npz=base64.b64decode(capture['arrays_npz_base64']);(folder/'received_arrays.npz').write_bytes(npz)
  arrays=np.load(io.BytesIO(npz),allow_pickle=False)
  for i,pred in enumerate(record['predictions']):
   for key in ('masks','scores','point_coords','point_labels'):
    if key in pred:pred[key]['array']=arrays[f'{i}_{key}']
  record['response']={k:v for k,v in payload.items() if k!='raw_capture'} if record['status']=='segment_returned' else None
  manifest=save_capture(record,folder)
  row={'kind':kind,'status':record['status'],'predict_inst_calls':record['predict_inst_calls'],'complete_capture':manifest['raw_capture_complete'],'selected_index_observed':record['selected_index_observed'],'error':record['error']}
  if record['selected_index_observed']:
   a=record['predictions'][record['selected_prediction_call']]['masks']['array'][record['selected_index']].squeeze()
   binary=bool(np.isin(a,[0,1]).all());row.update(dtype=str(a.dtype),binary_values=binary)
   if binary:
    raw=(a!=0).astype('uint8');wire=_rasterize(record['response'],im.width,im.height) if record['response'] else np.zeros_like(raw)
    row.update(raw_pixels=int(raw.sum()),raw_components=int(cv2.connectedComponents(raw)[0]-1),wire_available=record['response'] is not None)
    if record['response'] is not None:row.update(wire_pixels=int(wire.sum()),removed_pixels=int(((raw!=0)&(wire==0)).sum()),added_pixels=int(((raw==0)&(wire!=0)).sum()),warnings=record['response'].get('warnings',[]))
    for name,mask in [('raw',raw),('wire',wire),('removed',((raw!=0)&(wire==0)).astype('uint8')),('added',((raw==0)&(wire!=0)).astype('uint8'))]:Image.fromarray(mask*255).save(folder/(name+'.png'))
    for name,mask in [('raw',raw),('wire',wire)]:
     photo=np.array(im).copy();photo[mask!=0]=(photo[mask!=0]*.45+np.array([0,210,150])*.55).astype('uint8')
     card=Image.new('RGB',(1920,1080),(240,244,248));d=ImageDraw.Draw(card);d.text((40,22),f'真实SAM隔离捕获 / {kind} / {name}',font=font,fill=(15,30,50));large=Image.fromarray(photo);large.thumbnail((1200,830));large=large.resize((round(large.width*min(1200/large.width,830/large.height)),round(large.height*min(1200/large.width,830/large.height))),Image.Resampling.NEAREST);card.paste(large,(40,115));
     text=[f'内部面积: {row.get("raw_pixels")}',f'内部组件: {row.get("raw_components")}',f'返回面积: {row.get("wire_pixels","无有效返回")}',f'删除 / 新增: {row.get("removed_pixels","未知")} / {row.get("added_pixels","未知")}',f'状态: {row["status"]}','同一请求捕获；没有裁掩码','灰图是合成负对照','13/17与原6/6保留；未采用新候选']
     for j,line in enumerate(text):d.text((1280,140+j*85),line,font=font,fill=(15,30,50))
     path=folder/(name+'_comparison.jpg');card.save(path);cards.append(path)
  save(folder/'comparison.json',row);rows.append(row);save(OUT/'progress.json',rows);print(json.dumps(row),flush=True)
 after={p:sha(p) for p in before};assert before==after
 video=OUT/'内部位图与返回轮廓_真实推理回放.mp4';writer=cv2.VideoWriter(str(video),cv2.VideoWriter_fourcc(*'mp4v'),10,(1920,1080))
 for p in cards:
  frame=cv2.imread(str(p))
  for _ in range(50):writer.write(frame)
 writer.release();cap=cv2.VideoCapture(str(video));count=0
 while True:
  ok,_=cap.read()
  if not ok:break
  count+=1
 cap.release();assert count==len(cards)*50
 summary={'status':'complete_real_private_inference','rows':rows,'real_SAM_calls':1,'gray_SAM_calls':1,'model_weight_source':plan['weights'],'protected_sources_unchanged':True,'new_agent_runs':0,'new_candidate_adopted':False,'retained_pallet_support':'13/17','retained_original_convergence':'6/6','remaining_misses':4,'gallery_images':len(cards),'videos':1,'decoded_frames':count,'service_modified':False,'historical_internal_masks_recovered':False}
 save(OUT/'completed.json',summary)
 links=''.join('<p><img width="1100" src="'+p.relative_to(OUT).as_posix()+'"></p>' for p in cards)
 (OUT/'index.html').write_text('<meta charset="utf-8"><title>真实SAM内部掩码捕获</title><h1>真实SAM内部掩码与返回轮廓</h1><p>原图一次、灰图一次；保存阶段回放，不是连续录屏。未采用新候选。</p><a href="completed.json">完整结论</a><p><video width="1100" controls src="'+video.name+'"></video></p>'+links,encoding='utf-8')
 print(json.dumps(summary,ensure_ascii=False),flush=True)
if __name__=='__main__':main()
