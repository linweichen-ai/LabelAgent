"""One true cropped SAM call; full-coordinate points remain under the old cap."""
import base64,hashlib,io,json
from pathlib import Path
import cv2,numpy as np
from PIL import Image
def png(im):
 b=io.BytesIO();im.save(b,'PNG');return b.getvalue()
def spatial_sam(config,sam,image,prior,gaps,points):
 from source.agent.edit_agent import _rasterize,_shape_from_mask
 source=Image.open(io.BytesIO(image)).convert('RGB');w,h=source.size
 if not gaps or len(points)>32:raise ValueError('Invalid original crop prompt budget')
 # Context padding is fixed in advance, not chosen using reference points.
 x1=max(0,min(g['bbox'][0] for g in gaps)-96);y1=max(0,min(g['bbox'][1] for g in gaps)-96)
 x2=min(w,max(g['bbox'][2] for g in gaps)+97);y2=min(h,max(g['bbox'][3] for g in gaps)+97)
 active=[p for p in points if x1<=p['x']<x2 and y1<=p['y']<y2]
 # Only a real, already budgeted floor anchor is routed. Never invent a point.
 mapped=[dict(p,x=p['x']-x1,y=p['y']-y1) for p in active]
 raw=png(source.crop((x1,y1,x2,y2)));mask=png(Image.fromarray(prior[y1:y2,x1:x2]*255))
 binding={'source_sha256':hashlib.sha256(image).hexdigest(),'crop':[x1,y1,x2,y2], 'image_sha256':hashlib.sha256(raw).hexdigest(),'prior_sha256':hashlib.sha256(mask).hexdigest(),'original_points':points,'actual_points':mapped,'gap_ids':[g['id'] for g in gaps]}
 key=hashlib.sha256(json.dumps(binding,sort_keys=True).encode()).hexdigest();cache=Path(config['sam_spatial_cache']);cache.mkdir(parents=True,exist_ok=True)
 file=cache/(key+'_response.json');cached=file.exists()
 (cache/(key+'_input.png')).write_bytes(raw);(cache/(key+'_prior.png')).write_bytes(mask)
 (cache/(key+'_binding.json')).write_text(json.dumps(binding,indent=2)+'\n',encoding='utf-8')
 if cached:response=json.loads(file.read_text(encoding='utf-8'))
 else:
  response=sam(config,'/segment_points',raw,points=mapped,mask=base64.b64encode(mask).decode())
  file.write_text(json.dumps(response,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
 local=_rasterize(response,x2-x1,y2-y1);full=np.zeros_like(prior);full[y1:y2,x1:x2]=local
 Image.fromarray(full*255).save(cache/(key+'_raw_full.png'))
 shape=_shape_from_mask(full,'floor',1)
 return {**response,'parts':shape['parts'] if shape else [],'spatial_receipt':{'key':key,'crop':binding['crop'],'cached':cached,'original_point_count':len(points),'actual_point_count':len(mapped)}}

def choose_early_hard_fills(holes,assessments,operations,geometry,width,height,compiler):
 ground={r['hole_id'] for r in assessments if r['semantic']=='ground'}
 groups={}
 for h in holes:
  if h['id'] in ground:groups.setdefault(h['shape_id'],[]).append(h)
 if not groups:return operations[:3]
 _,group=max(groups.items(),key=lambda row:(len(row[1]),-row[0]))
 if len(group)<=3:return operations[:3]
 # Thin structures suffer most from downsampling; prioritize two complete
 # independently approved gaps for the two individually counted direct edits.
 def narrowness(g):
  dist=cv2.distanceTransform(g['mask'].astype('uint8'),cv2.DIST_L2,5)
  return (float(dist.max()),g['area'],g['id'])
 ordered=sorted(group,key=narrowness);hard=ordered[:2];rest=ordered[2:]
 fills=[next(op for op in operations if op.get('hole_id')==g['id']) for g in hard]
 batch=compiler(rest,assessments,geometry,width,height,set())
 return fills+([batch] if batch else [])

def choose_context_first(holes,assessments,operations,geometry,width,height,compiler):
 ground={r['hole_id'] for r in assessments if r['semantic']=='ground'}
 groups={}
 for h in holes:
  if h['id'] in ground:groups.setdefault(h['shape_id'],[]).append(h)
 if not groups:return operations[:3]
 _,group=max(groups.items(),key=lambda row:(len(row[1]),-row[0]))
 if len(group)<=3:return operations[:3]
 # Deep interior cutouts compete with a dominant floor prior. Boundary
 # cutouts have nearby outside context. This fixed ranking uses only pixels.
 def rank(g):
  x1,y1,x2,y2=g['bbox'];return (-min(x1,y1,width-1-x2,height-1-y2),-g['area'],g['id'])
 ordered=sorted(group,key=rank);fills=[next(op for op in operations if op.get('hole_id')==g['id']) for g in ordered[:2]]
 batch=compiler(ordered[2:],assessments,geometry,width,height,set())
 return fills+([batch] if batch else [])
