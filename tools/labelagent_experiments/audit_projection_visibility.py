"""Hash-bound camera/LiDAR join, ray-box occlusion and neighborhood evidence."""
import json,sys,hashlib,io
from pathlib import Path
from projection_audit_core import ROOT,read,environment,frame_data,cameras,other_box_front_hits
def save(p,v):p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(v,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
def main():
 cfg,ev=environment();import numpy as np,cv2
 from PIL import Image,ImageDraw,ImageFont,ImageOps
 from mcap.reader import make_reader
 out=ROOT/'outputs/labelagent_projection_recovery_20261105/projection_visibility';out.mkdir(parents=True,exist_ok=True)
 if (out/'completed.json').exists():raise FileExistsError('Preserve projection audit')
 dataset=Path(cfg['dataset']);manifest=read(dataset/'import_manifest.json');cams=cameras(dataset,manifest,ev);data=frame_data(dataset,manifest,76,ev);stamp=int(data['stamp']);records={(r['frame'],r['topic']):r for r in manifest['records']};targetids=[42,47,48,49];font=ImageFont.truetype('C:/Windows/Fonts/msyh.ttc',25);small=ImageFont.truetype('C:/Windows/Fonts/msyh.ttc',18);colors={42:(255,70,70),47:(60,240,90),48:(250,190,40),49:(70,180,255)}
 join=[]
 with Path(manifest['mcap_file']).open('rb') as f:
  for schema,channel,message in make_reader(f).iter_messages(start_time=stamp,end_time=stamp+1):
   r=records[76,channel.topic];assert message.log_time==stamp and message.publish_time==stamp and hashlib.sha256(message.data).hexdigest()==r['sha256'];join.append({'topic':channel.topic,'log_time_ns':message.log_time,'publish_time_ns':message.publish_time,'payload_hash_verified':True,'time_origin':'original filename; importer synthesizes MCAP log/publish timestamps'})
 assert len(join)==5
 image_metadata=[]
 for cam in range(4):
  r=records[76,f'/capture/camera_{cam}/image'];raw=ev.load_image(dataset,r)
  with Image.open(io.BytesIO(raw)) as im:image_metadata.append({'camera':cam,'size':list(im.size),'info_keys':list(im.info),'exif_fields':list(im.getexif().keys()),'hash_verified':True})
 _,fields,_,_,_=ev.read_pcd(dataset/'source/lidar'/f'{stamp}.pcd');time_fields=[f for f in fields if f['name'].lower() in ['timestamp','time','t','offset_time']]
 # Preserve numeric calibration validation separately from actual geometric accuracy.
 calib=[]
 for cam,c in cams.items():
  r=c['transform'][:3,:3];recovered=(data['xyz'][:100]@r.T+c['transform'][:3,3]);hom=np.column_stack([recovered,np.ones(len(recovered))])@np.linalg.inv(c['transform']).T
  calib.append({'camera':cam,'determinant':float(np.linalg.det(r)),'orthonormal_error':float(np.linalg.norm(r.T@r-np.eye(3))),'roundtrip_max_m':float(np.abs(hom[:,:3]-data['xyz'][:100]).max()),'provided_numeric_calibration_valid':True,'independent_reprojection_accuracy_verified':False})
 visibility=[];point_details=[];sensitivity=[]
 for cam,camera in cams.items():
  source,pixels=ev.project_visible(data['xyz'],camera);xyz=data['xyz']@camera['transform'][:3,:3].T+camera['transform'][:3,3];z=xyz[:,2];good=z>.2;allidx=np.flatnonzero(good);uv=xyz[good,:2]/z[good,None]*camera['intrinsics'][:2]+camera['intrinsics'][2:];inside=(uv[:,0]>=0)&(uv[:,0]<1919.5)&(uv[:,1]>=0)&(uv[:,1]<1535.5);ap=np.floor(uv[inside]+.5).astype('int32');ai=allidx[inside];depth=np.full((1536,1920),np.inf,np.float32);np.minimum.at(depth,(ap[:,1],ap[:,0]),z[ai].astype('float32'));nearby=cv2.erode(depth,np.ones((25,25),np.uint8),borderType=cv2.BORDER_CONSTANT,borderValue=float('inf'))
  for identifier in targetids:
   eligible=next((i for i,b in enumerate(data['boxes']) if b['obj_id']==identifier),None)
   if eligible is None:continue
   sel=data['owner'][source]==eligible+1;idx=source[sel];pp=pixels[sel];front=other_box_front_hits(data,camera,idx);near=nearby[pp[:,1],pp[:,0]] if len(pp) else np.array([]);closer=(z[idx]-near)>.15
   proxies=[{'point_source_index':int(i),'raw_pcd_row':int(data['raw_pcd_rows'][i]),'pixel':p.tolist(),'depth_m':float(z[i]),'nearby12px_min_depth_m':float(n),'nearby_closer_than_15cm':bool(q),'other_reference_box_in_front':b} for i,p,n,q,b in zip(idx,pp,near,closer,front)]
   point_details.extend({'camera':cam,'object_id':identifier,**p} for p in proxies)
   row={'camera':cam,'object_id':identifier,'original8px_15cm_visible_points':len(idx),'front_other_box_proxy_points':sum(bool(p) for p in front),'nearer_lidar_in_12px_proxy_points':int(closer.sum()),'occluding_object_ids':sorted({b['object_id'] for v in front for b in v}),'proxies_are_not_visibility_truth':True};visibility.append(row)
  for cell in [1,4,8,16]:
   for slack in [0.,.15]:
    src,px=ev.project_visible(data['xyz'],camera,cell_size=cell,depth_slack=slack)
    counts={str(identifier):int(np.count_nonzero(data['owner'][src]==next((i+1 for i,b in enumerate(data['boxes']) if b['obj_id']==identifier),-999))) for identifier in targetids}
    sensitivity.append({'camera':cam,'cell':cell,'slack_m':slack,'target_point_counts':counts,'original_metric_setting':cell==8 and slack==.15,'diagnostic_only_never_acceptance':True})
 # Real neighboring frames, their own PCD/labels/calibration, no synthetic time shift.
 temporal=[];source_hashes={};pose_raw=ev.checked_source(dataset,manifest,'poses_aligned_motion.txt').decode('utf-8');pose_stamps={line.split(',')[0] for line in pose_raw.splitlines() if line and not line.startswith('#')}
 for frame in [74,75,76,77,78]:
  fd=frame_data(dataset,manifest,frame,ev);ts=int(fd['stamp']);row={'frame':frame,'timestamp_ns':ts,'relative_to76_ms':(ts-stamp)/1e6,'pose_same_filename_timestamp_present':fd['stamp'] in pose_stamps,'cameras':[]}
  for cam,camera in cams.items():
   r=records[frame,f'/capture/camera_{cam}/image'];raw=ev.load_image(dataset,r);source_hashes[r['source_file']]=r['sha256'];image=Image.open(io.BytesIO(raw)).convert('RGB');render=np.array(image).copy();src,px=ev.project_visible(fd['xyz'],camera);counts={}
   for identifier in targetids:
    ownerid=next((i+1 for i,b in enumerate(fd['boxes']) if b['obj_id']==identifier),-999);sel=fd['owner'][src]==ownerid;uv=px[sel];counts[str(identifier)]=len(uv)
    for x,y in uv:cv2.circle(render,(int(x),int(y)),4,colors[identifier],-1)
    if ownerid>0:
     box=fd['boxes'][ownerid-1]['box'];corners=box.corners@camera['transform'][:3,:3].T+camera['transform'][:3,3]
     if np.all(corners[:,2]>.2):
      pp=corners[:,:2]/corners[:,2,None]*camera['intrinsics'][:2]+camera['intrinsics'][2:]
      for a in range(8):
       for b in range(a):
        # Adjacent signed unit-cube corners differ in exactly one coordinate.
        local=(box.corners-box.center)@box.rotation
        if np.count_nonzero(abs(local[a]-local[b])>1e-5)==1 and np.max(abs(pp[[a,b]]))<1e6:cv2.line(render,tuple(np.round(pp[a]).astype(int)),tuple(np.round(pp[b]).astype(int)),colors[identifier],1)
    if frame==76 and len(uv):
     x1,y1=uv.min(0);x2,y2=uv.max(0);rect=(max(0,int(x1)-80),max(0,int(y1)-80),min(1920,int(x2)+81),min(1536,int(y2)+81));canvas=Image.new('RGB',(1600,960),'#142436');d=ImageDraw.Draw(canvas);d.text((25,20),f'相机{cam} / 对象{identifier} / 原图与原投影对照',font=font,fill='white')
     for col,im in enumerate([image.crop(rect),Image.fromarray(render).crop(rect)]):im=ImageOps.contain(im,(760,720));canvas.paste(im,(20+col*800+(760-im.width)//2,100+(720-im.height)//2))
     vv=next(v for v in visibility if v['camera']==cam and v['object_id']==identifier);d.text((25,840),f"原指标可见点{len(uv)}；前方三维框代理{vv['front_other_box_proxy_points']}；12像素内更近点{vv['nearer_lidar_in_12px_proxy_points']}",font=small,fill='white');d.text((25,895),'框体和邻域只能提示遮挡风险，不能当作相机深度真值；没有删参考点或改验收。',font=small,fill='#e1d0b6');canvas.save(out/f'camera{cam}_obj{identifier}_冲突核查.jpg',quality=93)
   canvas=Image.new('RGB',(1920,1080),'#142436');d=ImageDraw.Draw(canvas);d.text((25,20),f'真实邻帧{frame} / camera{cam} / 相对帧76 {(ts-stamp)/1e6:.3f}ms',font=font,fill='white');im=ImageOps.contain(Image.fromarray(render),(1840,880));canvas.paste(im,((1920-im.width)//2,80));d.text((25,985),'红42、绿47、黄48、蓝49：本帧原始参考框与原8像素/0.15米可见点；非二维真值。',font=small,fill='white');d.text((25,1025),'文件同名时间对齐不能证明真实曝光同步；此处只展示原数据，不进行时间重配或标注更改。',font=small,fill='#c4dae8');canvas.save(out/f'frame{frame}_camera{cam}_原投影.jpg',quality=92);row['cameras'].append({'camera':cam,'target_visible_points':counts,'filename_time_equals_lidar':r['timestamp_ns']==ts})
  temporal.append(row);save(out/'progress.json',{'completed_neighborhood_frames':frame-73});print(json.dumps({'frame':frame,'cameras':4}),flush=True)
 conclusion={'status':'complete','projection_sources_hash_verified':True,'MCAP_frame76_messages_verified':join,'image_metadata':image_metadata,'PCD_fields':data['pcd_fields'],'PCD_time_fields':[f['name'] for f in time_fields],'calibration':calib,'visibility':visibility,'zbuffer_sensitivity':sensitivity,'neighbor_frames':temporal,'neighborhood_source_hashes':source_hashes,
  'confirmed':'original filenames,imported frame mapping,MCAP payload/timestamps and numeric calibration checks agree; original8px/.15m visibility only tests sampled LiDAR cells,not dense camera occlusion',
  'not_confirmed':['physical camera exposure vs LiDAR clock alignment','absolute extrinsic reprojection accuracy without independent2D correspondences','proxy occlusion as dense image truth'],
  'original_metric_and_references_changed':False,'annotation_writes':0,'new_generation_reference_access':False}
 save(out/'target_point_depth_lineage.json',point_details);save(out/'completed.json',conclusion);print(json.dumps({'projection_audit_complete':True,'visibility':visibility,'source_images':len(source_hashes)},ensure_ascii=False),flush=True)
if __name__=='__main__':main()
