"""Visualize references and geometry counts; never render solver-selected masks."""
import json,shutil,hashlib
from pathlib import Path
import numpy as np,cv2,av
from PIL import Image,ImageDraw,ImageFont,ImageOps
from projection_audit_core import ROOT,read
OUT=ROOT/'outputs/labelagent_point_separability_20261107';INC=ROOT/'outputs/labelagent_context_comparison_20261105'
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def save(p,v):Path(p).write_text(json.dumps(v,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
def main():
 result=read(OUT/'completed.json');assets=OUT/'assets';assets.mkdir(exist_ok=True);font=ImageFont.truetype('C:/Windows/Fonts/msyh.ttc',27);small=ImageFont.truetype('C:/Windows/Fonts/msyh.ttc',22);photos=[];plots=[];cropchecks=[]
 for row in result['rows']:
  cam,oid=row['camera'],row['object_id'];data=read(OUT/f'camera{cam}_obj{oid}_frozen_reference_points.json');px=np.array(data['pixels']);own=np.array(data['owners']);target=own==data['target_owner'];xy=px[target];box=(max(0,int(xy[:,0].min())-90),max(0,int(xy[:,1].min())-90),min(1920,int(xy[:,0].max())+91),min(1536,int(xy[:,1].max())+91));source=Image.open(INC/'inputs'/f'camera{cam}_frame76.png').convert('RGB');crop=source.crop(box);native=assets/f'camera{cam}_obj{oid}_原始裁块.png';crop.save(native);np.testing.assert_array_equal(np.asarray(Image.open(native)),np.asarray(source)[box[1]:box[3],box[0]:box[2]]);cropchecks.append({'camera':cam,'object_id':oid,'box':box,'native_crop_exact':True,'source_sha256':sha(INC/'inputs'/f'camera{cam}_frame76.png')})
  marked=source.copy();d=ImageDraw.Draw(marked)
  for (x,y),t in zip(px,target):d.ellipse((x-3,y-3,x+3,y+3),fill='#ff453a' if t else '#2b9dff')
  canvas=Image.new('RGB',(1920,1080),'#eff3f7');d=ImageDraw.Draw(canvas);d.text((30,25),f'相机{cam} / 参考对象{oid}：原图与原投影（不是二维真值）',font=font,fill='#10283c')
  for col,im in enumerate((crop,marked.crop(box))):
   scale=min(900/im.width,780/im.height);im=im.resize((round(im.width*scale),round(im.height*scale)),Image.Resampling.NEAREST);canvas.paste(im,(30+960*col,130))
  d.text((30,940),f'红：目标参考点 {target.sum()}；蓝：其余PAL参考点。完全沿用8像素/0.15米可见性。',font=small,fill='#10283c');d.text((30,1000),'只作可见性核查；不据参考点切掩码、选提示或更改类别。',font=small,fill='#10283c');p=assets/f'camera{cam}_obj{oid}_原图对照.jpg';canvas.save(p);photos.append(p)
  canvas=Image.new('RGB',(1920,1080),'#eff3f7');d=ImageDraw.Draw(canvas);d.text((30,25),f'对象{oid}：参考点几何可分性宽松诊断',font=font,fill='#10283c')
  for j,r in enumerate(row['radii']):
   y=160+180*j;d.text((60,y),f'容差半径 {r["radius"]}：{r["target_points_reachable_without_any_other_PAL"]}/{r["target_points"]} 个目标点存在无其他PAL命中的位置',font=font,fill='#10283c');d.text((60,y+65),f'完整整数求解状态 {r["solver_status"]}；最低支持需 {r["minimum_required_target_hits"]} 点；原纯度要求67%。',font=small,fill='#10283c')
  for j,line in enumerate(('允许零碎、互不连接的像素；忽略图像、完整基座、面积和人员，因此不是有效分割。','结论：没有发现像素容差导致的不可分证明；不能由此确认可见实例或投影准确。','求解器选择的像素没有导出为掩码，也不会送进SAM/Agent。','13/17与原6/6保留，四个漏检未恢复。')):d.text((60,790+j*65),line,font=small,fill='#10283c')
  p=assets/f'camera{cam}_obj{oid}_几何诊断.jpg';canvas.save(p);plots.append(p)
 videos=[]
 for name,paths in [('01_四对象原图与投影核查.mp4',photos),('02_几何可分性诊断解释.mp4',plots)]:
  p=OUT/name
  with av.open(str(p),'w') as container:
   stream=container.add_stream('libx264',rate=10);stream.width=1920;stream.height=1080;stream.pix_fmt='yuv420p'
   for card in paths:
    a=np.asarray(Image.open(card).convert('RGB'))
    for _ in range(50):
     for packet in stream.encode(av.VideoFrame.from_ndarray(a,format='rgb24')):container.mux(packet)
   for packet in stream.encode():container.mux(packet)
  cap=cv2.VideoCapture(str(p));count=0
  while True:
   ok,_=cap.read()
   if not ok:break
   count+=1
  cap.release();assert count==200;videos.append({'path':name,'frames':count,'sha256':sha(p)})
 media={'passed':True,'pictures':8,'videos':videos,'fully_decoded_frames':400,'native_crops':cropchecks,'no_solver_mask_rendered_or_adopted':True,'model_calls':0};save(OUT/'media_verification.json',media)
 text='''四个漏检的参考点几何可分性诊断

本轮仅使用已冻结参考点作诊断，没有模型调用或新掩码。沿用8像素/0.15米可见点、0/3/6容差、0.5支持和0.67纯度，检查允许任意不连通像素、忽略实际图像/完整对象/面积/人员的宽松数学问题。四目标×三半径共12项均可行，而且每个目标点都有不命中其他PAL参考的像素位置。因此没有得到“原像素容差使四目标必然不可分”的证据；这也不能证明真实可见基座可分或标定正确。

允许的零碎点岛不是完整实例。求解器选择的像素不导出为mask、不参与生成、不覆盖原标签、不计作召回提升。必须继续依靠图像单层结构、主体定位和真实遮挡核查，不能用参考点切分得到高纯度。

独立栅格穷举、小样本完全重合、分离、重复参考点等5项CPU检查通过；12个整数解的实际命中集合已重算。原六组结果SHA前后相同，原标注与验收不变，仍13/17和原6/6、剩4。8张诊断卡、4份原始裁块、2段20秒阶段回放（400帧）均回读。该结果不提供新的2D人工真值或密集分割准确率。
''';(OUT/'本轮结论.txt').write_text(text,encoding='utf-8')
 html='<meta charset="utf-8"><h1>四个漏检：原图、投影与几何可分性</h1><p>参考点诊断；零模型调用、零新增掩码。没有恢复实例，13/17与原6/6保留。</p><a href="completed.json">完整诊断</a><p>以下视频为保存阶段回放，不是连续录屏。</p>'
 html+=''.join('<p><video width="1100" controls src="'+v['path']+'"></video></p>' for v in videos)+''.join('<p><img width="1100" src="'+p.relative_to(OUT).as_posix()+'"></p>' for p in photos+plots);(OUT/'index.html').write_text(html,encoding='utf-8')
 repo=ROOT/'task_workspaces/labelagent_realtime_handoff_20261104'
 for name in ('point_separability.py','run_point_separability.py','finish_point_separability.py'):shutil.copy2(Path(__file__).with_name(name),repo/'tools/labelagent_experiments'/name)
 shutil.copy2(Path(__file__).with_name('test_point_separability.py'),repo/'tests/test_point_separability.py')
 public={'status':result['status'],'cases':[{'camera':r['camera'],'object_id':r['object_id'],'radii':[{k:q[k] for k in ('radius','target_points','minimum_required_target_hits','target_points_reachable_without_any_other_PAL','solver_status','relaxation_feasible','optimal_other_hits')} for q in r['radii']]} for r in result['rows']],'model_calls':0,'new_masks':0,'adopted':False,'retained_support':'13/17','retained_original_Agent':'6/6','remaining_misses':4,'CPU_tests':5,'not_real_segmentation_or_dense_truth':True}
 save(repo/'docs/test-results/point-separability.json',public);(repo/'docs/test-results/point-separability.md').write_text(text,encoding='utf-8')
 print(json.dumps({'pictures':8,'videos':2,'frames':400,'not_new_segmentation':True}),flush=True)
if __name__=='__main__':main()
