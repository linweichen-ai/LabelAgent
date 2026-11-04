"""One real SAM request on enlarged raw-photo crops of approved floor gaps."""
import base64,hashlib,io,json,math
from pathlib import Path
import cv2,numpy as np
from PIL import Image

def png(image):
 stream=io.BytesIO();image.save(stream,'PNG');return stream.getvalue()

def build_atlas(image,prior,gaps,points,cell=256,pad=16):
 if not gaps or len(gaps)>31 or len(points)>32:raise ValueError('Original reviewed gap/point cap exceeded')
 source=Image.open(io.BytesIO(image)).convert('RGB');w,h=source.size
 columns=math.ceil(math.sqrt(len(gaps)));rows=math.ceil(len(gaps)/columns)
 atlas=Image.new('RGB',(columns*cell,rows*cell),(96,96,96));atlas_prior=np.zeros((rows*cell,columns*cell),np.uint8)
 bindings=[];mapped_points=[]
 for index,gap in enumerate(gaps):
  x1,y1,x2,y2=gap['bbox'];box=[max(0,x1-pad),max(0,y1-pad),min(w,x2+pad+1),min(h,y2+pad+1)]
  cw,ch=box[2]-box[0],box[3]-box[1];scale=min((cell-16)/cw,(cell-16)/ch);tw,th=max(1,round(cw*scale)),max(1,round(ch*scale))
  ax=(index%columns)*cell+(cell-tw)//2;ay=(index//columns)*cell+(cell-th)//2
  atlas.paste(source.crop(box).resize((tw,th),Image.Resampling.LANCZOS),(ax,ay))
  atlas_prior[ay:ay+th,ax:ax+tw]=cv2.resize(prior[box[1]:box[3],box[0]:box[2]],(tw,th),interpolation=cv2.INTER_NEAREST)
  inside=[p for p in points if p['label']==1 and gap['mask'][int(p['y']),int(p['x'])]]
  if not inside:raise ValueError('Approved gap has no original, budgeted positive point')
  p=inside[0];mx=ax+min(tw-1,max(0,int((p['x']-box[0]+.5)*tw/cw)));my=ay+min(th-1,max(0,int((p['y']-box[1]+.5)*th/ch)))
  mapped_points.append({'x':mx,'y':my,'label':1})
  bindings.append({'hole_id':gap['id'],'source_box':box,'atlas_box':[ax,ay,ax+tw,ay+th],
     'original_point':p,'atlas_point':mapped_points[-1],'source_gap_pixels':int(gap['mask'].sum())})
 if len(mapped_points)>32:raise ValueError('Atlas point cap exceeded')
 return png(atlas),png(Image.fromarray(atlas_prior*255)),mapped_points,bindings

def infer_atlas(config,sam_request,image,prior,gaps,points):
 from source.agent.edit_agent import _rasterize,_shape_from_mask
 atlas,atlas_prior,atlas_points,bindings=build_atlas(image,prior,gaps,points)
 binding={'image_sha256':hashlib.sha256(image).hexdigest(),'atlas_sha256':hashlib.sha256(atlas).hexdigest(),
  'prior_sha256':hashlib.sha256(atlas_prior).hexdigest(),'original_points':points,'actual_atlas_points':atlas_points,'bindings':bindings}
 key=hashlib.sha256(json.dumps(binding,sort_keys=True).encode()).hexdigest();folder=Path(config['sam_gap_atlas_cache']);folder.mkdir(parents=True,exist_ok=True)
 response_file=folder/(key+'_response.json');cached=response_file.exists()
 (folder/(key+'_input.png')).write_bytes(atlas);(folder/(key+'_prior.png')).write_bytes(atlas_prior)
 (folder/(key+'_binding.json')).write_text(json.dumps(binding,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
 if cached:response=json.loads(response_file.read_text(encoding='utf-8'))
 else:
  response=sam_request(config,'/segment_points',atlas,points=atlas_points,mask=base64.b64encode(atlas_prior).decode())
  response_file.write_text(json.dumps(response,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
 if not response.get('parts'):raise ValueError('Real atlas SAM response has no mask')
 aw,ah=Image.open(io.BytesIO(atlas)).size;raw=_rasterize({'parts':response['parts']},aw,ah)
 mapped=np.zeros_like(prior);rows=[]
 for gap,b in zip(gaps,bindings):
  x1,y1,x2,y2=b['source_box'];ax1,ay1,ax2,ay2=b['atlas_box']
  local=cv2.resize(raw[ay1:ay2,ax1:ax2],(x2-x1,y2-y1),interpolation=cv2.INTER_NEAREST)
  projected=np.zeros_like(prior);projected[y1:y2,x1:x2]=local
  hit=int((projected&gap['mask']).sum());rows.append({'hole_id':gap['id'],'target_pixels':int(gap['mask'].sum()),'SAM_covered_pixels':hit,'fully_covered':hit==int(gap['mask'].sum())})
  mapped |= projected & gap['mask']
 shape=_shape_from_mask(mapped,'floor',1)
 result={**response,'parts':shape['parts'] if shape else [],'atlas_receipt':{'key':key,'cached':cached,'real_requests':0 if cached else 1,'original_cumulative_points':len(points),'actual_atlas_points':len(atlas_points),'rows':rows}}
 (folder/(key+'_mapped.png')).write_bytes(png(Image.fromarray(mapped*255)))
 (folder/(key+'_receipt.json')).write_text(json.dumps(result['atlas_receipt'],ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
 return result
