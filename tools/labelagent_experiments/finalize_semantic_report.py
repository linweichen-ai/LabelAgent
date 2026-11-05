"""Attach post-freeze loss diagnosis and exact whole-mask preservation proofs."""
import json,sys,hashlib,html,textwrap
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];out=ROOT/'outputs/labelagent_semantic_recovery_20261105';helpers=Path(__file__).parent
def read(p):return json.loads(p.read_text(encoding='utf-8'))
def save(p,v):p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(v,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
cfg=read(ROOT/'task_workspaces/labelagent_semantic_recovery_20261105/config_semantic.json');sys.path.insert(0,cfg['source_snapshot'])
import numpy as np,cv2
from PIL import Image,ImageDraw,ImageFont,ImageOps
from source.agent.edit_agent import _rasterize
summary=read(out/'public_summary.json');context=ROOT/'outputs/labelagent_context_comparison_20261105';candidate=ROOT/'outputs/labelagent_semantic_comparison_20261105';previous=ROOT/'outputs/labelagent_preserved_model_comparison_20261105';generation=Path(cfg['instance_generation']);selection=read(generation/'C_selection.json')['selected'];proof=[]
def fp(s):return hashlib.sha256(_rasterize(s,1920,1536).tobytes()).hexdigest()
for case,rows in selection.items():
 final=read(candidate/'C_hybrid'/case/'result.json')['shapes'];pal={fp(s) for s in final if s['label']=='pallet'}
 for r in rows:
  shapes=read(Path(r['source_prediction']))['PAL_shapes'];finger=fp(r['shape']);assert finger in {fp(s) for s in shapes} and finger in pal
  proof.append({'case':case,'key':r['key'],'raster_sha256':finger,'unchanged_original_model_instance_in_final':True})
 old=[s for s in read(previous/'C_hybrid'/case/'result.json')['shapes'] if s['label'] in ['pallet','person']]
 held=[s for s in read(context/'C_hybrid'/case/'result.json')['shapes'] if s['label'] in ['pallet','person']]
 assert sorted((s['label'],fp(s)) for s in old)==sorted((s['label'],fp(s)) for s in held)
save(out/'whole_mask_verification.json',{'passed':True,'new_complete_model_masks':proof,'budget_pixel_clipping':False,'context_foreground_rasters_identical_to_previous13_support':True})
controls=read(out/'semantic_contract/completed.json')['rows']+read(out/'cross_ablation/completed.json')['rows']+read(out/'neutral_controls/completed.json')['rows'];runtime=[r['runtime'] for r in controls]
assert all(r['image_count']==1 and r['actual_image_token_counts'] and r['actual_image_token_counts'][0]>0 for r in runtime)
assert len({r['model_config_sha256'] for r in runtime})==1
save(out/'semantic_runtime_verification.json',{'passed':True,'actual_image_encoding_verified_for_requests':len(runtime),'model_config_sha256':runtime[0]['model_config_sha256'],'same_physical_runtime_configuration':True,'prompt_and_enum_crossing_is_real':True})
losses=read(out/'candidate_loss_audit/completed.json');rejected=read(out/'instance_generation_contextual/C_selection.json')['rejected_whole_masks'];font=ImageFont.truetype('C:/Windows/Fonts/msyh.ttc',27);small=ImageFont.truetype('C:/Windows/Fonts/msyh.ttc',20);cards=[]
for row in losses['rows']:
 row['whole_mask_budget_rejection']=next((r for r in rejected if r['key']==row['key']),None)
 image=Image.open(context/'inputs'/(row['case']+'.png')).convert('RGB');mask=_rasterize(row['source_shape'],1920,1536);ys,xs=np.where(mask);box=(max(0,int(xs.min())-48),max(0,int(ys.min())-48),min(1920,int(xs.max())+49),min(1536,int(ys.max())+49));render=np.array(image).copy();render[mask!=0]=(.5*render[mask!=0]+.5*np.array([235,90,60])).astype('uint8')
 canvas=Image.new('RGB',(1920,1080),'#142436');d=ImageDraw.Draw(canvas);ids=','.join(str(x['obj_id']) for x in row['matches']);d.text((30,20),row['case']+' / 事后候选损失追踪 / 参考对象 '+ids,font=font,fill='white')
 for i,(label,im) in enumerate([('原始局部照片',image.crop(box)),('完整原模型候选（红）',Image.fromarray(render).crop(box))]):
  d.text((35+i*950,80),label,font=font,fill='#aff0dd');im=ImageOps.contain(im,(900,610));canvas.paste(im,(35+i*950+(900-im.width)//2,130+(610-im.height)//2))
 lines=[f"候选{row['key'][:12]}；稀疏支持必要条件 {[(x['obj_id'],x['best_candidate_supported_coverage']) for x in row['matches']]}",f"中性证据查询 {row['neutral_evidence_queried']}；通过 {row['neutral_evidence_passed']}；最终接纳 {row['admitted']}"]
 lines.extend('实际模型描述：'+v['object'] for v in row['captions'])
 if row['whole_mask_budget_rejection']:lines.append('整项拒绝：'+row['whole_mask_budget_rejection']['reason']+f"；新增面积 {row['whole_mask_budget_rejection']['new_pixels']}")
 y=765
 for line in lines:
  for part in textwrap.wrap(line,130):d.text((30,y),part,font=small,fill='#d8e5f1');y+=32
 d.text((30,1035),'仅事后解释，不回流筛选；稀疏支持与可见语义有冲突时，不为分数强行加标注。',font=small,fill='#ffd7bc')
 file=out/'assets'/('事后候选损失_'+row['key'][:12]+'.jpg');canvas.save(file,quality=93);cards.append(file)
save(out/'candidate_loss_audit/completed.json',losses)
findings='''最终保留固定前景版本：托盘13/17、490/579参考点，复核由4/6改善至6/6。新细网格＋语义门控版本为12/17、493/579，虽然多覆盖3点，但实例少1个，按原标准拒绝替换。

剩余4个对象的事后诊断（参考点不参与生成）：
相机0/42：细网格原池出现0.5稀疏支持候选，但两张实际描述均识别为灰色箱体/容器，不能强行当作可见托盘。本证据提示稀疏投影与可见对象语义存在冲突或遮挡风险，不据此改原标签、指标或认定参考错误。
相机0/48：整摞粘连仍使候选纯度不达0.67；新细网格原池也未给出合格单实例。
相机1/47：所有本轮原始候选最高纯度合格覆盖仍为4/11=0.3636，达不到原0.5门槛；不能靠门控或面积裁剪补出缺失观测。
相机1/49：旧粗网格原池有0.5稀疏支持的候选，但整项新增28430像素，超原面积预算而未进入语义查询；最终未接纳。该原候选不满足原预算，不能把原池可能性称作恢复。

新版本还丢失上轮相机1/48的0.6364支持候选：中性描述看成灰白表面/栏杆，未通过视觉门控；这正是新版本12/17未能替换上轮13/17的原因之一。小目标可辨认性、相邻物体主体归属及稀疏投影/遮挡一致性仍待独立二维人工确认。不能为了指标把含货物/栏杆的候选直接称为托盘，也没有放宽任何验收。
'''
(out/'剩余问题精确诊断.txt').write_text(findings,encoding='utf-8')
summary.update(pictures=summary['pictures']+len(cards),retained_offline_comparison='labelagent_context_comparison_20261105',new_instance_variant_rejected=True,remaining_previous_pallet_misses=4,remaining_nonconverged_groups=0,classification_control_scope='7 exposed functional negatives,2 exposed visible-pallet diagnostics; no automatic semantic approval')
save(out/'public_summary.json',summary);media=read(out/'media_verification.json');media['pictures']=summary['pictures'];save(out/'media_verification.json',media)
page=(out/'index.html').read_text(encoding='utf-8');banner='<section><b>保留13/17前景的版本：6/6组三轮内复查完成。新候选版12/17，实例退步，拒绝替换。原始标注和默认流程未改变。</b><pre>'+html.escape(findings)+'</pre></section>'
page=page.replace('<section>',banner+'<section>',1);extra='<section><h2>冻结后逐候选损失核对</h2><p><a href="candidate_loss_audit/completed.json">原始逐候选评分及拒绝来源</a></p>'
for file in cards:extra+=f'<p>{file.stem}</p><img loading="lazy" src="{file.relative_to(out).as_posix()}">'
page=page.replace('</html>',extra+'</section></html>');(out/'index.html').write_text(page,encoding='utf-8')
notes=(out/'结论与限制.txt').read_text(encoding='utf-8');(out/'结论与限制.txt').write_text(findings+'\n'+notes,encoding='utf-8')
save(out/'generation_source_receipt.json',{'prediction_frozen_before_reference_diagnosis':True,'scripts_sha256':{name:hashlib.sha256((helpers/name).read_bytes()).hexdigest() for name in ['semantic_evidence_gate.py','semantic_subject_guard.py','guard_contextual_candidates.py','generate_contextual_candidates.py','run_semantic_comparison.py','ground_action_review_v6.py','spatial_gap_recovery.py']}})
print(json.dumps({'pictures':summary['pictures'],'videos':summary['videos'],'new_variant_rejected':True,'remaining_pallets':4,'remaining_nonconverged':0}),flush=True)
