"""Recompute paired request identity and render actual caption outcomes."""
import hashlib,json,shutil,textwrap
from pathlib import Path
import numpy as np,cv2,av
from PIL import Image,ImageDraw,ImageFont
from run_visual_budget import ROOT,OUT,read,save,sha,payload
def main():
 result=read(OUT/'completed.json');plan=read(OUT/'frozen_plan.json');assert len(result['rows'])==20
 protected=plan['protected'];assert all(sha(p)==h for p,h in protected.items());pairs=[];assets=OUT/'assets';assets.mkdir(exist_ok=True);font=ImageFont.truetype('C:/Windows/Fonts/msyh.ttc',26);small=ImageFont.truetype('C:/Windows/Fonts/msyh.ttc',23);cards=[];checks=[]
 for a,b in zip(result['rows'][::2],result['rows'][1::2]):
  assert a['name']==b['name'] and a['sha256']==b['sha256'] and a['budget']==256 and b['budget']==1024
  raw=Path(a['image']).read_bytes();pa=payload(raw,256);pb=payload(raw,1024);assert hashlib.sha256(json.dumps(pa,sort_keys=True).encode()).hexdigest()==a['request_sha256'];assert hashlib.sha256(json.dumps(pb,sort_keys=True).encode()).hexdigest()==b['request_sha256'];pa.pop('image_token_budget');pb.pop('image_token_budget');assert pa==pb
  for row in (a,b):
   response=read(row['response']);meta=response['local_vlm'];assert meta['actual_image_token_counts']==[row['budget']] and meta['effective_image_token_budget']==row['budget'];assert json.loads(response['choices'][0]['message']['content'])==row['decision'];assert meta['model_config_sha256']==row['model_config_sha256']
  checks.append({'name':a['name'],'only_budget_field_differs':True,'actual_tokens':[256,1024],'same_image_sha256':a['sha256'],'raw_responses_readback':True});pairs.append((a,b))
  canvas=Image.new('RGB',(1920,1080),'#eff3f7');d=ImageDraw.Draw(canvas);d.text((30,22),a['name']+' / 同图像字节，仅预算不同',font=font,fill='#142e43');im=Image.open(a['image']).convert('RGB').resize((740,740),Image.Resampling.LANCZOS);canvas.paste(im,(30,130));d.text((30,920),'1024图为相同原生裁块插值，不新增原生细节。',font=small,fill='#142e43')
  for k,row in enumerate((a,b)):
   x=820+535*k;d.text((x,145),f'预算/实际token: {row["budget"]}',font=font,fill='#142e43');d.text((x,215),f'托盘主体门槛: {row["PAL_subject_passed"]}',font=small,fill='#142e43')
   for j,line in enumerate(textwrap.wrap(row['decision']['object'],width=32)):d.text((x,300+42*j),line,font=small,fill='#142e43')
   if row['kind']=='control':d.text((x,780),f'对照预期PAL: {row["expected_PAL"]}',font=small,fill='#142e43')
  d.text((820,930),'原图、掩码、默认服务和Agent轮数均未更改。',font=small,fill='#142e43');p=assets/(a['name']+'.jpg');canvas.save(p);cards.append(p)
 videos=[]
 for name,paths in [('01_候选主体指代预算对比.mp4',cards[:6]),('02_阳性与灰白负对照.mp4',cards[6:])]:
  p=OUT/name
  with av.open(str(p),'w') as container:
   stream=container.add_stream('libx264',rate=10);stream.width=1920;stream.height=1080;stream.pix_fmt='yuv420p'
   for card in paths:
    image=np.asarray(Image.open(card).convert('RGB'))
    for _ in range(50):
     for packet in stream.encode(av.VideoFrame.from_ndarray(image,format='rgb24')):container.mux(packet)
   for packet in stream.encode():container.mux(packet)
  cap=cv2.VideoCapture(str(p));n=0
  while True:
   ok,_=cap.read()
   if not ok:break
   n+=1
  cap.release();assert n==len(paths)*50;videos.append({'path':p.name,'frames':n,'sha256':sha(p)})
 controls={str(b):sum(r['passed'] for r in result['controls'][str(b)]) for b in (256,1024)};candidates={str(b):sum(r['PAL_subject_passed'] for r in result['rows'] if r['budget']==b and r['kind']!='control') for b in (256,1024)}
 verification={'passed':True,'paired_requests':checks,'successful_responses':20,'only_field_changed':'image_token_budget','model_config_same':len({r['model_config_sha256'] for r in result['rows']})==1,'native_sources_and_original_six_unchanged':True,'default_budget_restored_to':result['default_budget_after'],'candidate_PAL_counts':candidates,'controls_pass_counts':controls,'candidate_adopted':False};save(OUT/'independent_readback.json',verification)
 save(OUT/'media_verification.json',{'passed':True,'pictures':len(cards),'videos':videos,'fully_decoded_frames':sum(v['frames'] for v in videos),'source_bytes_and_decisions_readback':True,'stage_replay_not_screen_recording':True})
 text='''同内容图片的视觉token预算受控对比

10张图片×256/1024视觉token，20次真实回答。输入照片从相同原生裁块和相同完整mask视图范围重建为1024像素；两组图像字节完全一致，仅image_token_budget不同。与更早512像素请求不是字节相同的复用。1024图插值不新增原生信息，测试的是更多视觉采样/token是否改善描述，不能称为拍摄质量提升。

每个响应独立核对实际token确为256或1024，模型配置摘要相同、温度0和输出上限256/提示/schema一致，服务默认预算在实验后仍256。六个候选视图在两组均0/6通过托盘主体门槛；原木托盘阳性在256组正确、1024组改成没有肯定托盘词的模糊表面描述。四项控制256为4/4、1024为3/4，灰白及机器人负对照均保持非托盘。更大预算未改善候选对象指代并损失阳性，拒绝采用。

原始响应和错误回答全部保留；没有新SAM、掩码、训练或Agent运行，没有改变默认预算和最终验收。原输入/参考/六组结果SHA保持，仍13/17及原6/6、剩4。10张对比卡和两段30/20秒保存阶段回放（500帧）均完整解码。本对比属于同采集工程诊断，不是独立泛化或密集分割精度。
''';(OUT/'本轮结论.txt').write_text(text,encoding='utf-8')
 html='<meta charset="utf-8"><h1>单请求视觉预算对照：不采用</h1><p>20次真实回答，候选两组均0/6，控制4/4降至3/4。1024图不新增原生细节。13/17及原6/6保留。</p><a href="completed.json">原始汇总</a> · <a href="independent_readback.json">请求与响应回读</a><p>视频是已保存阶段回放，不是连续录屏。</p>'
 html+=''.join('<p><video width="1100" controls src="'+v['path']+'"></video></p>' for v in videos)+''.join('<p><img width="1100" src="'+p.relative_to(OUT).as_posix()+'"></p>' for p in cards);(OUT/'index.html').write_text(html,encoding='utf-8')
 repo=ROOT/'task_workspaces/labelagent_realtime_handoff_20261104'
 for name in ('run_visual_budget.py','finish_visual_budget.py'):shutil.copy2(Path(__file__).with_name(name),repo/'tools/labelagent_experiments'/name)
 public={'status':'complete_rejected_budget_candidate','successful_responses':20,'candidate_PAL_counts':candidates,'candidate_views_per_arm':6,'control_pass_counts':controls,'controls_per_arm':4,'actual_token_counts_verified':True,'same_image_bytes_between_arms':True,'interpolated_inputs_not_new_native_detail':True,'default_budget_after':256,'new_SAM_calls':0,'new_Agent_runs':0,'adopted':False,'retained_support':'13/17','retained_original_Agent':'6/6','remaining_misses':4,'pictures':10,'videos':2}
 save(repo/'docs/test-results/visual-budget.json',public);(repo/'docs/test-results/visual-budget.md').write_text(text,encoding='utf-8')
 home=ROOT/'deliverables/LabelAgent_实时交接/09_补充成果/视觉预算受控对比_20261107';assert not home.exists();shutil.copytree(OUT,home);code=home/'复现工具';code.mkdir()
 for name in ('run_visual_budget.py','finish_visual_budget.py','target_identity_policy.py','semantic_evidence_gate.py','semantic_subject_guard.py'):shutil.copy2(Path(__file__).with_name(name),code/name)
 (code/'说明.txt').write_text('迁机需重绑定ROOT、原数据/冻结环境和既有本地服务。不是独立运行镜像；不得重复已成功请求。',encoding='utf-8')
 files=[{'path':p.relative_to(home).as_posix(),'sha256':sha(p)} for p in home.rglob('*') if p.is_file()];assert all(sha(home/f['path'])==f['sha256'] for f in files);save(home/'交接SHA256.json',{'files':files})
 save(ROOT/'task_workspaces/labelagent_handoff_control_20261104/visual_budget_bundle_verification.json',{'passed':True,'bundle':str(home),'files':len(files),'pictures':len(cards),'videos':2,'frames':500,'successful_responses':20,'actual_token_receipts_verified':True,'original_six_and_sources_unchanged':True});print(json.dumps(public,ensure_ascii=False),flush=True)
if __name__=='__main__':main()
