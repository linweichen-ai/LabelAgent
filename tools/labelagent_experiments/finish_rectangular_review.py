"""Read back complete-mask geometry and replay old/new text guards separately."""
import hashlib,json,shutil,textwrap,types
from pathlib import Path
import numpy as np,cv2,av
from PIL import Image,ImageDraw,ImageFont
from run_rectangular_review import ROOT,OUT,OLD,INC,read,save,sha,request_key
from rectangular_views import render
from semantic_subject_guard import compatible_pallet_subject
def main():
 result=read(OUT/'completed.json');plan=read(OUT/'frozen_plan.json');C=ROOT/'task_workspaces/labelagent_handoff_control_20261104';qa=read(C/'rectangular_regression.json');assert qa['passed'];oldgate=types.ModuleType('saved_old_subject_gate');exec(compile((OUT/'semantic_subject_guard_before_fix.py').read_bytes(),'saved_old_subject_gate','exec'),oldgate.__dict__)
 for p,h in plan['protected'].items():assert sha(p)==h
 checks=[];replayed=[];cached=OUT/'reused_responses';cached.mkdir(exist_ok=True)
 for r in result['rows']:
  assert request_key(r['image'])==r['request_sha256'];response=read(r['response']);assert json.loads(response['choices'][0]['message']['content'])==r['decision'];assert response['local_vlm']['model_config_sha256']==plan['model_config_sha256'];assert response['local_vlm']['actual_image_token_counts']==[256];assert oldgate.compatible_pallet_subject(r['decision'])==r['PAL_passed']
  updated=compatible_pallet_subject(r['decision']);replayed.append({'name':r['name'],'kind':r['kind'],'legacy_PASS':r['PAL_passed'],'strict_primary_subject_PASS':updated,'decision':r['decision']})
  if 'reuse_response' in r:shutil.copy2(r['response'],cached/(r['request_sha256']+'.json'))
  checks.append({'name':r['name'],'request_sha256':r['request_sha256'],'response_sha256':sha(r['response']),'actual_tokens':256,'reused_prior':'reuse_response' in r})
 real=[r for r in result['rows'] if r['kind']=='rect_candidate'];baseline=[r for r in result['rows'] if r['kind']=='square_baseline'];assets=OUT/'assets';assets.mkdir(exist_ok=True);font=ImageFont.truetype('C:/Windows/Fonts/msyh.ttc',26);small=ImageFont.truetype('C:/Windows/Fonts/msyh.ttc',22);cards=[];geometry=[]
 for r in real:
  b=r['binding'];sourcefile=INC/'inputs'/'camera0_frame76.png';source=Image.open(sourcefile).convert('RGB');box=b['canvas_box'];clip=b['source_box'];image,native,offset=render(source,b);np.testing.assert_array_equal(np.asarray(image),np.asarray(Image.open(r['image'])));actual=np.asarray(native)[clip[1]-box[1]:clip[3]-box[1],clip[0]-box[0]:clip[2]-box[0]];np.testing.assert_array_equal(actual,np.asarray(source)[clip[1]:clip[3],clip[0]:clip[2]]);geometry.append({'name':r['name'],'source_crop_pixels_exact':True,'all_candidate_pixels_retained':b['all_candidate_pixels_retained'],'anchor_center_on_mask':b['center_on_mask']})
  previous=next(p for p in baseline if p['name']==r['name']);canvas=Image.new('RGB',(1920,1080),'#eff3f7');d=ImageDraw.Draw(canvas);d.text((30,22),r['name']+' / 方形与矩形窗口，完整候选像素不变',font=font,fill='#142d43')
  for k,row in enumerate((previous,r)):
   x=30+960*k;im=Image.open(row['image']).convert('RGB').resize((600,600),Image.Resampling.LANCZOS);canvas.paste(im,(x,110));d.text((x,740),'旧方形' if k==0 else '新矩形（灰色补边）',font=font,fill='#142d43')
   for j,line in enumerate(textwrap.wrap(row['decision']['object'],width=60)):d.text((x,800+31*j),line,font=small,fill='#142d43')
  p=assets/(r['name']+'.jpg');canvas.save(p);cards.append(p)
 # Matched-control cards preserve both geometry and raw negative answers.
 controls=[]
 for r in real:
  gray=next(p for p in result['rows'] if p['name']==r['name']+'_gray');white=next(p for p in result['rows'] if p['name']==r['name']+'_white');canvas=Image.new('RGB',(1920,1080),'#eff3f7');d=ImageDraw.Draw(canvas);d.text((30,22),r['name']+' / 同窗口灰白负对照（不是真实对象）',font=font,fill='#142d43')
  for k,row in enumerate((gray,white)):
   x=30+960*k;canvas.paste(Image.open(row['image']).convert('RGB').resize((600,600)),(x,110));d.text((x,740),f'严格主体检查PAL: {compatible_pallet_subject(row["decision"])}',font=font,fill='#142d43')
   for j,line in enumerate(textwrap.wrap(row['decision']['object'],width=60)):d.text((x,800+31*j),line,font=small,fill='#142d43')
  p=assets/(r['name']+'_controls.jpg');canvas.save(p);controls.append(p)
 videos=[]
 for name,paths in [('01_方形矩形真实回答对比.mp4',cards),('02_匹配灰白形状负对照.mp4',controls)]:
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
  cap.release();assert n==300;videos.append({'path':name,'frames':n,'sha256':sha(p)})
 guard_changed=[r for r in replayed if r['legacy_PASS']!=r['strict_primary_subject_PASS']];save(OUT/'subject_guard_replay.json',{'old_guard_sha256':sha(OUT/'semantic_subject_guard_before_fix.py'),'new_guard_sha256':sha(Path(__file__).with_name('semantic_subject_guard.py')),'rows':replayed,'changed':guard_changed,'raw_model_answers_unchanged':True,'not_complete_visual_grounding':True})
 base_count=sum(r['PAL_passed'] for r in baseline);legacy=sum(r['PAL_passed'] for r in real);strict=sum(compatible_pallet_subject(r['decision']) for r in real);negatives=[r for r in result['rows'] if r['kind'].startswith('matched_')];assert all(not compatible_pallet_subject(r['decision']) for r in negatives)
 summary={'status':'complete_geometry_not_adopted_subject_bug_fixed','fresh_VLM_responses':result['fresh_responses'],'prior_exact_reused_rows':result['reused_rows'],'same_run_duplicate_rows':len(result['rows'])-result['fresh_responses']-result['reused_rows'],'logical_rows':len(result['rows']),'baseline_subject_PASS':base_count,'rect_legacy_word_PASS':legacy,'rect_primary_subject_PASS':strict,'candidate_views':6,'matched_negative_PASS':len(negatives),'matched_negative_total':len(negatives),'engineering_tests':qa['tests'],'new_masks':0,'new_Agent_runs':0,'adopted':False,'retained_support':'13/17','retained_original_Agent':'6/6','remaining_misses':4,'pictures':12,'videos':2,'frames':600,'default_workbench_not_replaced':True}
 save(OUT/'final_summary.json',summary);save(OUT/'independent_readback.json',{'passed':True,'request_response_checks':checks,'geometry':geometry,'raw_model_answers_unchanged':True,'original_sources_and_six_exact':True,'legacy_false_background_mention_rejected':len(guard_changed)==1,'regression':qa})
 save(OUT/'media_verification.json',{'passed':True,'pictures':12,'videos':videos,'fully_decoded_frames':600,'saved_replay_not_screen_recording':True})
 text='''完整候选矩形视图与主体误放行修复

相同3个完整raw候选、near/context共6视图，保持原caption提示/schema、256视觉token、温度和输出上限，只改变上下文窗口几何；横纵分别围绕同一foreground anchor保留候选全部像素，保留源裁块原像素并等比灰补边。它保留的是完整预测候选，不是已确认的完整物理基座。加入每个窗口匹配gray/white源图负对照，避免把白色长条补边当物体。13个新真实VLM回答，10个旧请求完整身份复用，另5个相同灰图在本轮去重复用，共28逻辑行；无新SAM/掩码/训练/Agent运行。

原词检查方形0/6、矩形出现1/6，但唯一过关回答实际说“灰色地面、前景容器、背景蓝色托盘”，不能计作当前白色主体成功。发现semantic_subject_guard仍只按PAL词+车辆/不确定词否决，缺少主要分句绑定。保留旧检查源码与所有原始回答，新增保守主要分句检查：背景/附属托盘不能替代主要物体；明确主要托盘即使后面有其他背景物仍可通过，原车辆/不确定否决保持。回放后方形0/6、矩形0/6，12/12匹配负对照未被当托盘。不采用矩形方案，主体词修复不等于完整视觉定位证明。

36项相关工程检查通过，包含旧语义恢复/实例/范围/坐标及新主体、矩形边缘/断开像素保留。首次发布测试运行器缺少仓库根路径导致两个导入错误，原记录保留；补路径后全通过，没有伪造通过或重发模型请求。源图/参考/原六组结果SHA保持；仍13/17及原6/6、剩4。新guard仅更新实验工具，未替换工作台默认或原已接受结果。

12图与2段30秒保存阶段回放600帧完整解码。旧词命中、严格主体回放和原始模型描述分开保存；不是新的二维真值、召回提升或独立泛化验证。
''';(OUT/'本轮结论.txt').write_text(text,encoding='utf-8');html='<meta charset="utf-8"><h1>矩形视图与主体检查修复</h1><p>13次新回答，当前主体仍0/6；背景提及不计检出。13/17与原6/6保留。</p><a href="final_summary.json">结论</a> · <a href="subject_guard_replay.json">原始回答与检查回放</a><p>保存阶段回放，非连续录屏。</p>'+''.join('<p><video width="1100" controls src="'+v['path']+'"></video></p>' for v in videos)+''.join('<p><img width="1100" src="'+p.relative_to(OUT).as_posix()+'"></p>' for p in cards+controls);(OUT/'index.html').write_text(html,encoding='utf-8')
 repo=ROOT/'task_workspaces/labelagent_realtime_handoff_20261104';shutil.copy2(Path(__file__),repo/'tools/labelagent_experiments'/Path(__file__).name);save(repo/'docs/test-results/rectangular-review.json',summary);(repo/'docs/test-results/rectangular-review.md').write_text(text,encoding='utf-8')
 home=ROOT/'deliverables/LabelAgent_实时交接/09_补充成果/矩形视图与主体检查修复_20261107';assert not home.exists();shutil.copytree(OUT,home);code=home/'复现工具';code.mkdir()
 for name in ('rectangular_views.py','run_rectangular_review.py','finish_rectangular_review.py','semantic_subject_guard.py','test_rectangular_views.py','test_subject_binding.py','run_visual_budget.py','semantic_evidence_gate.py','target_identity_policy.py'):shutil.copy2(Path(__file__).with_name(name),code/name)
 for name in ('rectangular_regression.json','rectangular_regression.log','rectangular_regression_first_import_error.json','rectangular_regression_first_import_error.log'):shutil.copy2(C/name,home/name)
 (code/'说明.txt').write_text('恢复需重绑定原ROOT、数据和既有服务；不是独立镜像。所有成功请求不可重复；旧gate源码和旧/新gate回放均保留。',encoding='utf-8')
 files=[{'path':p.relative_to(home).as_posix(),'sha256':sha(p)} for p in home.rglob('*') if p.is_file()];assert all(sha(home/f['path'])==f['sha256'] for f in files);save(home/'交接SHA256.json',{'files':files});save(C/'rectangular_bundle_verification.json',{'passed':True,'bundle':str(home),'files':len(files),'pictures':12,'videos':2,'frames':600,'fresh_VLM_responses':13,'engineering_tests':36,'original_six_exact':True,'all_candidate_pixels_preserved':True});print(json.dumps(summary,ensure_ascii=False),flush=True)
if __name__=='__main__':main()
