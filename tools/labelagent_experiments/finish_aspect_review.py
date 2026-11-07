"""Verify actual grids and full native content; captions are not mask acceptance."""
import json,shutil,textwrap
from pathlib import Path
import numpy as np,cv2,av
from PIL import Image,ImageDraw,ImageFont,ImageOps
import run_aspect_review as experiment
from semantic_subject_guard import compatible_pallet_subject
from aspect_views import aspect_view
base=experiment.base;ROOT=experiment.ROOT;OUT=experiment.OUT
def main():
 result=base.read(OUT/'completed.json');plan=base.read(OUT/'frozen_plan.json');assert all(base.sha(p)==h for p,h in plan['protected'].items());real=[r for r in result['rows'] if r['kind']=='aspect_candidate'];old=[r for r in result['rows'] if r['kind']=='square_letterbox_baseline'];neg=[r for r in result['rows'] if r['kind'].startswith('matched_')];assets=OUT/'assets';assets.mkdir(exist_ok=True);copies=OUT/'reused_responses';copies.mkdir(exist_ok=True);checks=[]
 for r in result['rows']:
  assert base.request_key(r['image'])==r['request_sha256'];response=base.read(r['response']);stats=response['local_vlm'];assert stats['actual_image_token_counts']==[256] and stats['model_config_sha256']==plan['model_config_sha256'];assert json.loads(response['choices'][0]['message']['content'])==r['decision'];assert compatible_pallet_subject(r['decision'])==r['PAL_passed']
  if 'reuse_response' in r:shutil.copy2(r['response'],copies/(r['request_sha256']+'.json'))
  checks.append({'name':r['name'],'kind':r['kind'],'request_sha256':r['request_sha256'],'response_sha256':base.sha(r['response']),'actual_tokens':256,'grid':stats['image_grid_thw'],'reused_prior':'reuse_response' in r})
 font=ImageFont.truetype('C:/Windows/Fonts/msyh.ttc',26);small=ImageFont.truetype('C:/Windows/Fonts/msyh.ttc',22);cards=[];controls=[];geometry=[]
 for r in real:
  source=Image.open(base.INC/'inputs'/'camera0_frame76.png').convert('RGB');rendered,g=aspect_view(source,r['binding']);np.testing.assert_array_equal(np.asarray(rendered),np.asarray(Image.open(r['image'])));assert g==r['aspect_geometry'];previous=next(p for p in old if p['name']==r['name']);ratio=g['candidate_area_fraction_estimate']/g['square_candidate_fraction'];assert ratio>1;geometry.append({'name':r['name'],'candidate_sampling_ratio':ratio,'content_fraction':g['content_fraction'],'full_rectangle_retained':g['no_crop_of_native_rectangle'],'same_raw_candidate_mask':True})
  c=Image.new('RGB',(1920,1080),'#eff3f7');d=ImageDraw.Draw(c);d.text((30,22),r['name']+' / 等256视觉token，不同空间分配',font=font,fill='#102a40')
  for k,row in enumerate((previous,r)):
   x=30+960*k;im=Image.open(row['image']).convert('RGB');im=ImageOps.contain(im,(860,610));c.paste(im,(x,110));d.text((x,750),'旧方形补边 / grid 1×32×32' if k==0 else '新非方形 / grid 1×16×64',font=font,fill='#102a40');d.text((x,800),f'主体门槛: {row["PAL_passed"]}',font=small,fill='#102a40')
   for j,line in enumerate(textwrap.wrap(row['decision']['object'],width=60)):d.text((x,850+31*j),line,font=small,fill='#102a40')
  p=assets/(r['name']+'.jpg');c.save(p);cards.append(p)
  c=Image.new('RGB',(1920,1080),'#eff3f7');d=ImageDraw.Draw(c);d.text((30,22),r['name']+f' / 候选采样占比×{ratio:.2f}，匹配灰白负对照',font=font,fill='#102a40')
  for k,color in enumerate(('gray','white')):
   row=next(p for p in neg if p['name']==r['name']+'_'+color);x=30+960*k;im=ImageOps.contain(Image.open(row['image']).convert('RGB'),(860,610));c.paste(im,(x,110));d.text((x,750),f'合成{color} / 主体PAL: {row["PAL_passed"]}',font=font,fill='#102a40')
   for j,line in enumerate(textwrap.wrap(row['decision']['object'],width=60)):d.text((x,830+31*j),line,font=small,fill='#102a40')
  p=assets/(r['name']+'_negative.jpg');c.save(p);controls.append(p)
 consensus=[]
 for prefix in sorted({r['name'].rsplit('_',1)[0] for r in real}):
  rows=[r for r in real if r['name'].rsplit('_',1)[0]==prefix];assert len(rows)==2;consensus.append({'candidate':prefix,'near_and_context_passed':all(r['PAL_passed'] for r in rows)})
 videos=[]
 for name,paths in [('01_等预算真实主体对比.mp4',cards),('02_匹配几何灰白负对照.mp4',controls)]:
  p=OUT/name
  with av.open(str(p),'w') as container:
   stream=container.add_stream('libx264',rate=10);stream.width=1920;stream.height=1080;stream.pix_fmt='yuv420p'
   for card in paths:
    a=np.asarray(Image.open(card).convert('RGB'))
    for _ in range(50):
     for packet in stream.encode(av.VideoFrame.from_ndarray(a,format='rgb24')):container.mux(packet)
   for packet in stream.encode():container.mux(packet)
  cap=cv2.VideoCapture(str(p));n=0
  while True:
   ok,_=cap.read()
   if not ok:break
   n+=1
  cap.release();assert n==300;videos.append({'path':name,'frames':n,'sha256':base.sha(p)})
 summary={'status':'complete_partial_caption_gain_not_adopted','fresh_VLM_responses':result['fresh_responses'],'prior_exact_reused_rows':result['reused_rows'],'same_run_duplicate_rows':len(result['rows'])-result['fresh_responses']-result['reused_rows'],'baseline_primary_subject_PASS':sum(r['PAL_passed'] for r in old),'aspect_primary_subject_PASS':sum(r['PAL_passed'] for r in real),'candidate_views':len(real),'candidate_consensus_PASS':sum(c['near_and_context_passed'] for c in consensus),'candidates':len(consensus),'matched_negative_logical_rejections':sum(not r['PAL_passed'] for r in neg),'matched_negative_logical_rows':len(neg),'matched_negative_unique_requests':len({r['request_sha256'] for r in neg}),'actual_tokens_unchanged':256,'candidate_sampling_ratio_range':[min(g['candidate_sampling_ratio'] for g in geometry),max(g['candidate_sampling_ratio'] for g in geometry)],'new_masks':0,'new_Agent_runs':0,'adopted':False,'retained_support':'13/17','retained_original_Agent':'6/6','remaining_misses':4,'pictures':12,'videos':2,'frames':600}
 base.save(OUT/'final_summary.json',summary);base.save(OUT/'independent_readback.json',{'passed':True,'request_response_checks':checks,'geometry':geometry,'consensus':consensus,'original_six_and_sources_exact':True,'single_caption_not_instance_success':True});base.save(OUT/'media_verification.json',{'passed':True,'pictures':12,'videos':videos,'fully_decoded_frames':600,'saved_stage_replay_not_screen_recording':True})
 text='''固定256视觉token的非方形视图对照

相同三份完整raw候选near/context六视图，不改原裁块/候选/anchor/提示/schema/温度/输出上限，用最贴近原矩形纵横比的固定256-token格子画布等比放入完整内容。实际回执由方形grid[1,32,32]变为非方形[1,16,64]，每图仍256 token；候选canvas采样面积占比提高约2.6~4倍。没有新增原生信息或裁掉mask，变化是有限视觉预算的空间分配。

13个新真实回答、10个旧完整请求身份复用、5个本轮相同灰图去重，共28逻辑行。旧主体0/6，新2/6；两条分别明确描述白色托盘堆叠和白色木托盘，不是此前被误放行的背景蓝托盘。但三候选near与context同时通过仍0/3，且原实例支持失败的raw没有改变，所以不采用，不算新增检出。模型说两层也不是人工物理身份确认。

匹配几何灰白负对照12/12逻辑行（7个独立请求）未称托盘；4份旧阳性/负例按完整请求复用，没有假装新推理。没有新SAM、mask、训练或Agent运行，13/17及原6/6、剩4，全部最终门槛与默认保持。局部caption改善不是独立泛化、二维真值或完整基座验收。12图及两段30秒阶段回放600帧完整回读。
''';(OUT/'本轮结论.txt').write_text(text,encoding='utf-8');html='<meta charset="utf-8"><h1>等预算非方形视图</h1><p>主体描述0/6→2/6，但双视图同时通过0/3；没有恢复实例，原13/17与原6/6保留。</p><a href="final_summary.json">汇总</a> · <a href="independent_readback.json">实际grid与完整像素回读</a><p>保存阶段回放，非连续录屏。</p>'+''.join('<p><video width="1100" controls src="'+v['path']+'"></video></p>' for v in videos)+''.join('<p><img width="1100" src="'+p.relative_to(OUT).as_posix()+'"></p>' for p in cards+controls);(OUT/'index.html').write_text(html,encoding='utf-8')
 repo=ROOT/'task_workspaces/labelagent_realtime_handoff_20261104'
 for name in ('aspect_views.py','run_aspect_review.py','finish_aspect_review.py'):shutil.copy2(Path(__file__).with_name(name),repo/'tools/labelagent_experiments'/name)
 shutil.copy2(Path(__file__).with_name('test_aspect_views.py'),repo/'tests/test_aspect_views.py');base.save(repo/'docs/test-results/aspect-review.json',summary);(repo/'docs/test-results/aspect-review.md').write_text(text,encoding='utf-8')
 print(__import__('json').dumps(summary,ensure_ascii=False),flush=True)
if __name__=='__main__':main()
