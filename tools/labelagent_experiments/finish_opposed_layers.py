"""Report complete opposed-point SAM outputs; no acceptance shortcuts."""
import json,hashlib,shutil
from pathlib import Path
import numpy as np,cv2,av
from PIL import Image,ImageDraw,ImageFont
import run_opposed_layers as experiment
base=experiment.base;ROOT=experiment.ROOT;OUT=experiment.OUT
def main():
 plan=base.read(OUT/'frozen_plan.json');generated=base.read(OUT/'generation_completed.json');audit=base.read(OUT/'support_area_audit.json');assets=OUT/'assets';assets.mkdir(exist_ok=True);font=ImageFont.truetype('C:/Windows/Fonts/msyh.ttc',28);small=ImageFont.truetype('C:/Windows/Fonts/msyh.ttc',22);cards=[];checks=[]
 for row in generated['rows']:
  folder=Path(row['folder']);meta=base.read(folder/'metadata.json');arrays=np.load(folder/'all_arrays.npz',allow_pickle=False);i=meta['selected_prediction_call'];raw=(arrays[f'{i}_masks'][meta['selected_index']].squeeze()!=0).astype('uint8');full=np.load(folder/'complete_full_mask.npy',allow_pickle=False);x1,y1,x2,y2=row['box'];assert np.array_equal(full[y1:y2,x1:x2],raw) and int(full.sum())==int(raw.sum());np.testing.assert_array_equal(arrays[f'{i}_point_coords'],[[p['x'],p['y']] for p in row['points']]);np.testing.assert_array_equal(arrays[f'{i}_point_labels'],[p['label'] for p in row['points']]);assert meta['image_sha256']==base.sha(row['input'])
  original=Image.open(row['input']).convert('RGB');im=original.copy();draw=ImageDraw.Draw(im)
  for p in row['points']:draw.ellipse((p['x']-3,p['y']-3,p['x']+3,p['y']+3),fill='#ffcc00' if p['label'] else '#ff00ff')
  rgb=np.asarray(original).copy();rgb[raw!=0]=(.5*rgb[raw!=0]+.5*np.array([0,210,170])).astype('uint8');candidate=next(r for r in audit['candidates'] if r['key']==row['key']);radius3=next(c for c in audit['cases'] if c['case']==row['case'])['radii'][1];ownership=next(r for r in radius3['new_mask_ownership'] if r['key']==row['key'])['ownership'];target=next((r for r in ownership if r['object_id']==42),None);neighbor=next((r for r in ownership if r['object_id']==43),None)
  for label,photo in [('实际正负点',im),('完整内部输出',Image.fromarray(rgb))]:
   c=Image.new('RGB',(1920,1080),'#eff3f7');d=ImageDraw.Draw(c);d.text((35,25),f'{row["side"]} / {label} / RGB层结构分离试验',font=font,fill='#152d42');photo=photo.resize((850,850),Image.Resampling.NEAREST);c.paste(photo,(35,130));notes=['黄点：正点；紫点：负点',f'完整内部面积 {row["raw_pixels"]}',f'新增并集 {candidate["extra_pixels"]}',f'目标42命中 {target["points"] if target else 0}/16',f'目标42纯度 {target["purity"] if target else 0:.3f}',f'邻43命中 {neighbor["points"] if neighbor else 0}/50','原支持门槛未通过','未裁掩码，未采用','13/17与原6/6保留']
   for j,note in enumerate(notes):d.text((1010,150+j*80),note,font=font,fill='#152d42')
   p=assets/(row['side']+'_'+label+'.jpg');c.save(p);cards.append(p)
  checks.append({'key':row['key'],'actual_prompt_matches_frozen':True,'whole_internal_mask_exact':True,'selected_dtype':meta['predictions'][i]['masks']['dtype'],'all_arrays_sha256':base.sha(folder/'all_arrays.npz')})
 assert all(base.sha(p)==h for p,h in plan['protected'].items())
 video=OUT/'上下侧正负点与完整输出.mp4'
 with av.open(str(video),'w') as container:
  stream=container.add_stream('libx264',rate=10);stream.width=1920;stream.height=1080;stream.pix_fmt='yuv420p'
  for p in cards:
   image=np.asarray(Image.open(p).convert('RGB'))
   for _ in range(50):
    for packet in stream.encode(av.VideoFrame.from_ndarray(image,format='rgb24')):container.mux(packet)
  for packet in stream.encode():container.mux(packet)
 cap=cv2.VideoCapture(str(video));count=0
 while True:
  ok,_=cap.read()
  if not ok:break
  count+=1
 cap.release();assert count==len(cards)*50
 summary={'status':'complete_original_support_gate_failed','real_SAM_calls':generated['unique_real_SAM_calls'],'new_masks':len(generated['rows']),'new_Agent_runs':0,'new_VLM_calls':0,'semantic_review_not_run_reason':'Both new hypotheses already fail original remaining-instance support; semantic requirements remain mandatory before any adoption','retained_support':'13/17','retained_original_Agent':'6/6','remaining_misses':4,'adopted':False,'pictures':len(cards),'videos':1,'frames':count,'prompt_mask_readback':checks,'all_original_six_and_reference_bytes_unchanged':True,'proposal_rule_change':'Opposite-side negatives may fall inside unconfirmed search seed; no accepted/original mask changed, no final threshold changed'}
 base.save(OUT/'final_summary.json',summary);base.save(OUT/'media_verification.json',{'passed':True,'pictures':len(cards),'videos':[{'path':video.name,'frames':count,'sha256':base.sha(video)}],'fully_decoded_frames':count,'complete_native_outputs_verified':True})
 text='''相邻层正负点分离试验

同8个图像搜索种子，固定RGB边缘配对及第一几何线对，上/下四分之一处各作一次目标，另一侧放负点，最多16请求。有效提示2组，实际SAM2次。唯一改变是允许负点落在“未确认搜索种子”内；原已接受前景、原标注和全部最终门槛保持。

完整内部输出分别4571、1849像素，新增并集40、0，面积合格、未冲突人员。上侧在radius3支持旧邻43为48/50，纯度48/49=98%，但目标42仅1/16；下侧目标42为4/16，纯度4/11=36.4%。均未补回实例，0/3/6全匹配不变。不将邻居纯度上升计作剩余目标成功。两份均已失败于原支持门槛，未调用后续VLM/Agent；语义/单基座/范围等要求未解除，也没有通过声明。

全部原始数组与选中索引保留，提示实际载荷与冻结点一致，完整内部mask原样映射全图。没有水平裁剪、组件删除、面积倒推或根据参考点选提示。4图及20秒阶段回放200帧完整解码。仍13/17、原6/6，剩4，未采用新候选。这是复用捕获的开发对比，不是密集2D真值或泛化验证。
''';(OUT/'本轮结论.txt').write_text(text,encoding='utf-8');html='<meta charset="utf-8"><h1>相邻层正负点分离</h1><p>2次真实SAM；完整内部位图均未通过剩余实例支持。没有采用、没有新Agent运行。</p><a href="support_area_audit.json">原支持/面积对比</a> · <a href="final_summary.json">结论</a><p>保存阶段回放，非连续录屏。</p><video width="1100" controls src="'+video.name+'"></video>'+''.join('<p><img width="1100" src="'+p.relative_to(OUT).as_posix()+'"></p>' for p in cards);(OUT/'index.html').write_text(html,encoding='utf-8')
 repo=ROOT/'task_workspaces/labelagent_realtime_handoff_20261104'
 for name in ('run_opposed_layers.py','finish_opposed_layers.py'):shutil.copy2(Path(__file__).with_name(name),repo/'tools/labelagent_experiments'/name)
 public={k:v for k,v in summary.items() if k!='prompt_mask_readback'};base.save(repo/'docs/test-results/opposed-layers.json',public);(repo/'docs/test-results/opposed-layers.md').write_text(text,encoding='utf-8')
 print(json.dumps(public,ensure_ascii=False),flush=True)
if __name__=='__main__':main()
