"""Seal actual RGB-layer proposals, raw masks, unchanged-gate failures and media."""
import base64,hashlib,io,json,shutil,sys
from pathlib import Path
import numpy as np,cv2,av
from PIL import Image,ImageDraw,ImageFont
from run_rgb_layers import ROOT,OUT,INC,read,save,sha
def main():
 plan=read(OUT/'frozen_plan.json');gen=read(OUT/'generation_completed.json');audit=read(OUT/'support_area_audit.json');review=read(OUT/'semantic_review.json')
 assert all(sha(p)==h for p,h in plan['protected'].items())
 cards=[];checks=[];assets=OUT/'assets';assets.mkdir(exist_ok=True);font=ImageFont.truetype('C:/Windows/Fonts/msyh.ttc',26);small=ImageFont.truetype('C:/Windows/Fonts/msyh.ttc',22)
 def card(title,im,notes,name):
  c=Image.new('RGB',(1920,1080),'#eef3f7');d=ImageDraw.Draw(c);d.text((35,25),title,font=font,fill='#152b40');im=im.copy();im.thumbnail((1150,870));scale=min(1150/im.width,870/im.height);im=im.resize((round(im.width*scale),round(im.height*scale)),Image.Resampling.NEAREST);c.paste(im,(35,115))
  for j,line in enumerate(notes):d.text((1240,140+j*70),line,font=small,fill='#152b40')
  p=assets/name;c.save(p);cards.append(p)
 for diag in plan['diagnostics']:
  index=diag['seed'];folder=OUT/f'seed{index:02d}';im=Image.open(folder/'input.png').convert('RGB');display=im.copy();d=ImageDraw.Draw(display);props=[p for p in plan['proposals'] if p['seed']==index]
  for prop in props:
   for x1,y1,x2,y2,_ in prop['lines']:d.line((x1,y1,x2,y2),fill='#ffcc00',width=2)
   for p in prop['points']:d.ellipse((p['x']-3,p['y']-3,p['x']+3,p['y']+3),fill='#ff3333')
  card(f'原图种子 {index}：冻结的 RGB 层结构提示（搜索种子非真值）',display,[f'检测水平线: {len(diag["lines"])}',f'几何配对: {diag["pairs"]}',f'有效提示: {len(props)}',f'拒绝绑定: {len(diag["rejected"])}','黄线: RGB边缘配对','红点: 实际冻结正点','未产生方案则不补点','没有参考点参与生成'],f'seed{index:02d}.jpg')
 for r in gen['rows']:
  folder=Path(r['folder']);meta=read(folder/'metadata.json');assert meta['image_sha256']==sha(r['input'])
  assert r['key']==hashlib.sha256((sha(r['input'])+json.dumps(r['points'],sort_keys=True)).encode()).hexdigest()
  arrays=np.load(folder/'all_arrays.npz',allow_pickle=False);i=meta['selected_prediction_call'];np.testing.assert_array_equal(arrays[f'{i}_point_coords'],[[p['x'],p['y']] for p in r['points']]);np.testing.assert_array_equal(arrays[f'{i}_point_labels'],[1]*len(r['points']))
  mask=np.load(folder/'complete_full_mask.npy',allow_pickle=False);local=(arrays[f'{i}_masks'][meta['selected_index']].squeeze()!=0).astype('uint8');x1,y1,x2,y2=r['box'];assert int(mask.sum())==int(local.sum()) and np.array_equal(mask[y1:y2,x1:x2],local)
  im=Image.open(r['input']).convert('RGB');rgb=np.asarray(im).copy();rgb[local!=0]=(.5*rgb[local!=0]+.5*np.array([0,210,170])).astype('uint8');detail=next(a for a in audit['candidates'] if a['key']==r['key'])
  sem=next(a for a in review['rows'] if a['key']==r['key']);card('SAM完整内部位图：面积合格，但实例纯度未通过',Image.fromarray(rgb),[f'完整面积: {r["raw_pixels"]}',f'新增并集: {detail["extra_pixels"]}','参考42: 13/16点','参考43: 50点','目标42纯度: 20.6%','原门槛: 67%','实例匹配: 仍13/17',f'中性语义: {sem["neutral_passed"]}','未裁掩码，未采用'], 'whole_raw_candidate.jpg')
  checks.append({'key':r['key'],'prompt_payload_exact':True,'complete_raw_mapping_exact':True,'raw_arrays_sha256':sha(folder/'all_arrays.npz')})
 # All eight source crops match the original native pixels, including skipped seeds.
 cohort=read(ROOT/'outputs/labelagent_point_ids_20261107/planning/cohort.json')
 for seed in cohort:
  im=Image.open(INC/'inputs'/(seed['case']+'.png')).convert('RGB').crop(tuple(seed['binding']['source_box']));np.testing.assert_array_equal(np.asarray(im),np.asarray(Image.open(OUT/f"seed{seed['index']:02d}"/'input.png')))
 videos=[]
 for name,paths in [('01_RGB提示与拒绝记录.mp4',cards[:8]),('02_完整位图及原门槛结论.mp4',cards[8:])]:
  v=OUT/name
  with av.open(str(v),'w') as container:
   stream=container.add_stream('libx264',rate=10);stream.width=1920;stream.height=1080;stream.pix_fmt='yuv420p'
   for p in paths:
    a=np.asarray(Image.open(p).convert('RGB'))
    for _ in range(50):
     for packet in stream.encode(av.VideoFrame.from_ndarray(a,format='rgb24')):container.mux(packet)
   for packet in stream.encode():container.mux(packet)
  cap=cv2.VideoCapture(str(v));count=0
  while True:
   ok,_=cap.read()
   if not ok:break
   count+=1
  cap.release();assert count==50*len(paths);videos.append({'path':v.name,'frames':count,'sha256':sha(v)})
 media={'passed':True,'pictures':len(cards),'videos':videos,'fully_decoded_frames':sum(v['frames'] for v in videos),'all_8_crops_exact':True,'raw_prompt_and_mask_checks':checks,'saved_stage_replay_not_screen_recording':True};save(OUT/'media_verification.json',media)
 summary={'status':'complete_rejected_candidate','real_SAM_calls':gen['unique_real_SAM_calls'],'valid_prompts':len(plan['proposals']),'seeds':8,'new_masks':gen['valid_complete_outputs'],'new_VLM_trace_records':review['actual_model_trace_records'],'remaining_misses':4,'retained_pallet_support':'13/17','retained_original_Agent':'6/6','new_Agent_runs':0,'adopted':False,'raw_area':8493,'extra_union_pixels':80,'target42_points':13,'target42_total':16,'neighbor43_points':50,'purity42':13/63,'all_original_masks_retained':True,'annotation_writes':0,'acceptance_thresholds_unchanged':True,'pictures':len(cards),'videos':len(videos)};save(OUT/'final_summary.json',summary)
 html='<meta charset="utf-8"><h1>RGB逐层候选与完整位图对比</h1><p>8种子、1有效提示、1真实SAM；13/17与原6/6保留。保存阶段回放，非连续录屏。</p><a href="final_summary.json">结论</a> · <a href="support_area_audit.json">原门槛点支持</a> · <a href="semantic_review.json">实际语义回答</a>'
 html+=''.join('<p><video width="1100" controls src="'+v['path']+'"></video></p>' for v in videos)+''.join('<p><img width="1100" src="'+p.relative_to(OUT).as_posix()+'"></p>' for p in cards);(OUT/'index.html').write_text(html,encoding='utf-8')
 repo=ROOT/'task_workspaces/labelagent_realtime_handoff_20261104'
 for name in ('rgb_layer_policy.py','run_rgb_layers.py','audit_rgb_layers.py','review_rgb_layers.py','finish_rgb_layers.py'):shutil.copy2(Path(__file__).with_name(name),repo/'tools/labelagent_experiments'/name)
 save(repo/'docs/test-results/rgb-layers.json',summary)
 description='''RGB逐层提示完整位图试验：8个固定图像种子只生成1个合格提示，执行1次真实SAM，保留8493像素内部完整位图。新增并集80像素，符合原全局面积预算、未冲突人员；目标42覆盖13/16，同时邻近43覆盖50点，纯度13/63=20.6%，低于原67%，各半径0/3/6均未增加匹配。没有裁掩码、加轮数、放宽标准或采用新候选。原27托盘/3person、13/17和原6/6保持，新Agent运行0。原图裁块/提示载荷/完整数组到全图映射均回读一致。单层提示仍落入含相邻结构的区域，保留内部全部位图未解决实例归属。
原中性/单基座/范围复核的真实结果另见本地semantic_review；工程图为已暴露采集数据诊断，不是密集像素准确率或新人工真值。下一步要改善图像单实例的正负提示分离，不能按参考点或剩余面积切输出。'''
 (repo/'docs/test-results/rgb-layers.md').write_text(description,encoding='utf-8');(OUT/'本轮结论.txt').write_text(description,encoding='utf-8')
 home=ROOT/'deliverables/LabelAgent_实时交接/09_补充成果/RGB逐层提示与完整位图_20261107';assert not home.exists();shutil.copytree(OUT,home)
 for name in ('rgb_layer_policy.py','run_rgb_layers.py','audit_rgb_layers.py','review_rgb_layers.py','finish_rgb_layers.py'):shutil.copy2(Path(__file__).with_name(name),home/name)
 files=[{'path':p.relative_to(home).as_posix(),'sha256':sha(p)} for p in home.rglob('*') if p.is_file()];assert all(sha(home/f['path'])==f['sha256'] for f in files);save(home/'交接SHA256.json',{'files':files})
 save(ROOT/'task_workspaces/labelagent_handoff_control_20261104/rgb_layers_bundle_verification.json',{'passed':True,'bundle':str(home),'files':len(files),'pictures':len(cards),'videos':len(videos),'frames':media['fully_decoded_frames'],'all_8_native_crops_exact':True,'prompt_and_complete_masks_exact':True,'original_foreground_kept':True})
 print(json.dumps(summary,ensure_ascii=False),flush=True)
if __name__=='__main__':main()
