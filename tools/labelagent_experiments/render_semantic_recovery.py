"""Evidence-first report from actual model outputs; videos are stage replays."""
import json,sys,io,hashlib,html,shutil,textwrap
from pathlib import Path
from fractions import Fraction
ROOT=Path(__file__).resolve().parents[2];out=ROOT/'outputs/labelagent_semantic_recovery_20261105';context=ROOT/'outputs/labelagent_context_comparison_20261105';candidate=ROOT/'outputs/labelagent_semantic_comparison_20261105';previous=ROOT/'outputs/labelagent_preserved_model_comparison_20261105'
def read(p):return json.loads(p.read_text(encoding='utf-8'))
def save(p,v):p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(v,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(1048576),b''):h.update(b)
 return h.hexdigest()
def main():
 cfg=read(ROOT/'task_workspaces/labelagent_semantic_recovery_20261105/config_semantic.json');sys.path.insert(0,cfg['source_snapshot'])
 import numpy as np,cv2,av
 from PIL import Image,ImageDraw,ImageFont,ImageOps
 from source.agent.edit_agent import _rasterize
 assert read(context/'completed.json')['status']==read(candidate/'completed.json')['status']=='complete'
 assets=out/'assets';assets.mkdir(exist_ok=True);videos=out/'videos';videos.mkdir(exist_ok=True)
 font=ImageFont.truetype('C:/Windows/Fonts/msyh.ttc',25);small=ImageFont.truetype('C:/Windows/Fonts/msyh.ttc',19);big=ImageFont.truetype('C:/Windows/Fonts/msyh.ttc',35)
 gallery=[];groups={};sequences={'01_新旧方案与原验收.mp4':[],'02_分类诱导交叉实验.mp4':[],'03_三轮真实结果回放.mp4':[],'04_细网格模型真实输出.mp4':[],'05_候选来源与未解决问题.mp4':[]}
 def add(file,group):
  if file not in gallery:gallery.append(file);groups.setdefault(group,[]).append(file)
 def panel(title,items,file,notes=()):
  canvas=Image.new('RGB',(1920,1080),'#142436');d=ImageDraw.Draw(canvas);d.text((35,20),title,font=big,fill='white')
  n=len(items);cell=1840//n
  for i,(label,img) in enumerate(items):
   d.text((40+i*cell,85),label,font=font,fill='#aff0dd');im=ImageOps.contain(img.convert('RGB'),(cell-20,760));canvas.paste(im,(40+i*cell+(cell-20-im.width)//2,135+(760-im.height)//2))
  for i,line in enumerate(notes):d.text((35,920+i*39),line,font=small,fill='#d4e1ee')
  canvas.save(file,quality=91);return file
 def score(folder,arm):
  data=read(folder/'completed.json')['common_class_scores'][arm];p=data['pallet'];h=data['person'];r=p['radii']['3'];hr=h['radii']['3']
  return {'pallet_support':r['objects_matched_at_support_threshold'],'pallet_references':p['supported_reference_objects'],'pallet_points':r['covered_projected_points'],'person_support':hr['objects_matched_at_support_threshold'],'person_points':hr['covered_projected_points'],'person_off_class':hr['points_covered_by_other_class']}
 def audit(folder):
  result=read(folder/'completed.json');rounds=[];budget=[];totals={};protected=True;original=True
  for arm in ['A_current','C_hybrid']:
   area=count=0
   for cam in [0,1]:
    case=f'camera{cam}_frame76';shapes=read(folder/arm/case/'result.json')['shapes'];pal=[_rasterize(s,1920,1536) for s in shapes if s['label']=='pallet'];count+=len(pal);area+=int(np.maximum.reduce(pal).sum())
    def fps(ss,labels):return {hashlib.sha256(s['label'].encode()+_rasterize(s,1920,1536).tobytes()).hexdigest() for s in ss if s['label'] in labels}
    if arm=='C_hybrid':
     seed=read(folder/'seed_stages'/case/'all_arms.json')['C_hybrid'];base=read(folder/'A_current'/case/'result.json')['shapes'];protected &= fps(seed,{'pallet','person'})<=fps(shapes,{'pallet','person'});original &=fps(base,{'pallet'})<=fps(shapes,{'pallet'})
   totals[arm]={'PAL_count':count,'PAL_union_pixels':area}
  for row in result['agent_runs']:
   r=read(folder/row['arm']/row['case']/'result.json')
   for a in r['agent']['audit']:
    tools=a.get('tool_results',[]);ops=a['decision']['operations'];budget.append({'case':row['case'],'arm':row['arm'],'round':a['round'],'operations':len(ops),'cumulative_points':max([t.get('point_count',0) for t in tools]+[0]),'actual_points':max([t.get('active_request_point_count',t.get('point_count',0)) for t in tools]+[0])})
    rounds.append({'case':row['case'],'arm':row['arm'],'round':a['round'],'ground_tasks':sum(x['semantic']=='ground' for x in a['decision'].get('hole_assessment',[])),'resolved_SAM':sum(len(t.get('resolved_gap_ids',[])) for t in tools),'pending_SAM':sum(len(t.get('unresolved_gap_ids',[])) for t in tools),'direct_fills':sum(t.get('operation')=='fill_gap' and t.get('status')=='applied' for t in tools),'tool_errors':[t.get('error') for t in tools if t.get('status')=='error']})
  assert all(b['round']<=3 and b['operations']<=3 and b['cumulative_points']<=32 and b['actual_points']<=32 for b in budget)
  a=score(folder,'A_current');c=score(folder,'C_hybrid');converged=sum(bool(r['converged']) for r in result['agent_runs']);cr=totals['C_hybrid']['PAL_count']/totals['A_current']['PAL_count'];ar=totals['C_hybrid']['PAL_union_pixels']/totals['A_current']['PAL_union_pixels']
  checks={'PAL_support_strict_gain':c['pallet_support']>a['pallet_support'],'PAL_point_coverage_not_lower':c['pallet_points']>=a['pallet_points'],'person_point_coverage_not_lower':c['person_points']>=a['person_points'],'person_off_class_not_higher':c['person_off_class']<=a['person_off_class'],'PAL_candidate_ratio_at_most_1_5':cr<=1.5,'PAL_union_area_ratio_at_most_1_25':ar<=1.25,'initial_C_foreground_preserved':bool(protected),'original_SAM_PAL_preserved':bool(original),'review_protocol_no_errors':all(r['stop_reason']!='error' for r in result['agent_runs']),'all_six_runs_converged_within_three_rounds':converged==6}
  v={'checks':checks,'all_conditions_passed':all(checks.values()),'totals':totals,'candidate_ratio':cr,'union_area_ratio':ar,'converged':converged,'automatic_adoption':False};save(folder/'acceptance_checks.json',v);save(folder/'round_analysis.json',rounds);save(folder/'budget_audit.json',{'passed':True,'rows':budget});return v
 ca=audit(context);na=audit(candidate)
 rows=[{'name':'原A两类方案','folder':previous.name,'arm':'A_current',**score(previous,'A_current')}, {'name':'上轮C候选','folder':previous.name,'arm':'C_hybrid',**score(previous,'C_hybrid')},{'name':'本轮固定前景＋新地面调度','folder':context.name,'arm':'C_hybrid',**score(context,'C_hybrid')},{'name':'本轮新网格＋自然上下文语义门控','folder':candidate.name,'arm':'C_hybrid',**score(candidate,'C_hybrid')}]
 canvas=Image.new('RGB',(1920,1080),'#142436');d=ImageDraw.Draw(canvas);d.text((50,45),'剩余4个托盘 / 2组复核 / 灰白负对照：同图实测',font=big,fill='white')
 for i,r in enumerate(rows):
  d.text((50,150+i*155),r['name'],font=font,fill='#aff0dd');d.text((70,200+i*155),f"托盘 {r['pallet_support']}/17，参考点 {r['pallet_points']}/579；人员 {r['person_support']}/3，参考点 {r['person_points']}/1568；人员误覆盖 {r['person_off_class']}",font=font,fill='white')
 for i,line in enumerate([f"固定前景版本：三轮内收敛 {ca['converged']}/6；PAL面积 {ca['union_area_ratio']:.6f}×；新网格版本收敛 {na['converged']}/6。",'灰白交叉实验：旧提示去掉枚举仍报托盘；中性提示保留枚举返回不确定。','两张已暴露图；稀疏参考一致性不是像素准确率。所有候选仍需人工确认。','原标注、原图、参考和模型保持。三轮、每轮三项、累计32点、面积1.25倍均未放宽。']):d.text((50,800+i*55),line,font=small,fill='#d4e1ee')
 f=assets/'本轮同图对比总览.jpg';canvas.save(f,quality=94);add(f,'结果与验收');sequences['01_新旧方案与原验收.mp4'].append((f,8))
 for cam in [0,1]:
  case=f'camera{cam}_frame76';raw=Image.open(previous/'inputs'/(case+'.png'))
  for arm in ['A_current','B_sam7','C_hybrid']:
   stages=[previous/arm/case,context/arm/case,candidate/arm/case];items=[]
   for folder in stages:
    file=sorted(folder.glob('*finished.jpg'))[-1];items.append((folder.parents[1].name.replace('labelagent_','')[:32],Image.open(file)))
   target=assets/(case+'_'+arm+'_新旧对照.jpg');panel(case+' / '+arm,items,target,('左：上轮；中：固定前景与新地面策略；右：新模型候选。','真实模型候选，未写回原标注；全景展示不替代原评分器。'));add(target,'结果与验收');sequences['01_新旧方案与原验收.mp4'].append((target,4))
   if arm!='A_current':
    for variant,folder in [('context',context/arm/case),('candidate',candidate/arm/case)]:
     for file in sorted(folder.glob('*.jpg')):
      target=assets/(variant+'_'+case+'_'+arm+'_'+file.name);panel(variant+' / '+case+' / '+arm+' / '+file.stem,[(file.stem,Image.open(file))],target,('实际保存阶段回放，非连续桌面录屏。','每轮最多3操作；最后修改必须经后续真实观察才能记收敛。'));add(target,'Agent逐轮证据');sequences['03_三轮真实结果回放.mp4'].append((target,2))
 # Full 2x2 crossing, plus independent 9-case controls. Original source pixels
 # are shown beside the literal saved reply, never regenerated for the report.
 for name in ['gray','white']:
  ab=read(out/'semantic_contract'/f'{name}_old_schema_without_text_contract.json');neutral=next(r for r in read(out/'cross_ablation/completed.json')['rows'] if r['case']==name and r['mode']=='neutral_prompt_with_category_enum')
  f=assets/(name+'_提示交叉对照.jpg');im=Image.open(out/'semantic_contract'/(name+'.png'));canvas=Image.new('RGB',(1920,1080),'#142436');d=ImageDraw.Draw(canvas);d.text((40,25),name+' / 同一模型、同一图像的提示与枚举交叉实验',font=big,fill='white');canvas.paste(im.resize((500,500)),(40,130))
  text='旧提示＋类别枚举：'+json.dumps(ab['decision'],ensure_ascii=False)+'\n旧提示去掉类别枚举：'+json.dumps(next(r['decision'] for r in read(out/'cross_ablation/completed.json')['rows'] if r['case']==name and r['mode']=='old_prompt_no_category_enum'),ensure_ascii=False)+'\n中性提示＋原类别枚举：'+json.dumps(neutral['decision'],ensure_ascii=False)
  y=140
  for line in text.splitlines():
   for part in textwrap.wrap(line,90):d.text((580,y),part,font=small,fill='#d4e1ee');y+=39
   y+=35
  d.text((40,850),'主要证据指向旧提示的目标暗示；格式合规不能证明看图正确。',font=font,fill='#aff0dd');canvas.save(f,quality=93);add(f,'分类功能对照');sequences['02_分类诱导交叉实验.mp4'].append((f,7))
 controls=read(out/'neutral_controls/completed.json')
 for r in controls['rows']:
  f=assets/(r['case']+'_中性描述.jpg');panel('中性观察 / '+r['case'],[('实际输入',Image.open(out/'neutral_controls'/(r['case']+'.png')))],f,(r['decision']['object'][:145],f"PAL evidence: {r['pallet_evidence']}；这是功能检查，非独立准确率。"));add(f,'分类功能对照');sequences['02_分类诱导交叉实验.mp4'].append((f,4))
 # Every new grid forward is visible, including empty/incorrect outputs.
 for i,file in enumerate(sorted((out/'fixed_grid').glob('*/*/prediction.json'))):
  meta=read(file);folder=file.parent;raw=Image.open(folder/'input.png');rgb=np.array(Image.open(folder/'panoptic.png'));ids=rgb[:,:,0].astype('uint32')+256*rgb[:,:,1].astype('uint32')+65536*rgb[:,:,2].astype('uint32');render=np.array(raw).copy()
  for segment in meta['segments_info']:
   mask=ids==segment['id'];color=np.array([(segment['id']*71)%200+30,(segment['category_id']*43)%200+30,(segment['id']*113)%200+30]);render[mask]=(.5*render[mask]+.5*color).astype('uint8')
  f=assets/(folder.parent.name+'_'+folder.name+'_模型网格.jpg');panel(folder.parent.name+' / '+folder.name+f" / PAL hypotheses {len(meta['PAL_shapes'])}",[('真实模型输入',raw),('真实全景输出',Image.fromarray(render))],f,('同一训练权重，固定320×256图像网格，阈值不变。','真实推理输出全部保留，错误和空结果也展示；未按参考位置选图。'));add(f,'84次新模型推理')
  if meta['PAL_shapes']:sequences['04_细网格模型真实输出.mp4'].append((f,2))
 # Original error cards are explicitly retrospective and never generation inputs.
 for file in sorted((ROOT/'outputs/labelagent_recovery_20261105/miss_diagnosis').glob('*.jpg')):
  target=assets/('原漏检诊断_'+file.name);shutil.copy2(file,target);add(target,'漏检与候选来源');sequences['05_候选来源与未解决问题.mp4'].append((target,4))
 for variant in ['instance_generation','instance_generation_contextual']:
  for i,file in enumerate(sorted((out/variant/'visual_cache').glob('*.png'))):
   target=assets/(variant+'_'+file.name);shutil.copy2(file,target);add(target,'候选原图证据')
   if variant.endswith('contextual') and i<10:sequences['05_候选来源与未解决问题.mp4'].append((target,2))
 receipts=[]
 for name,frames in sequences.items():
  target=videos/name;temp=target.with_suffix('.pending.mp4');container=av.open(str(temp),'w',format='mp4',options={'movflags':'+faststart'});stream=container.add_stream('libx264',rate=10);stream.width=1920;stream.height=1080;stream.pix_fmt='yuv420p';stream.codec_context.thread_count=2;stream.options={'crf':'24','preset':'veryfast'};count=0
  for file,seconds in frames:
   im=ImageOps.contain(Image.open(file).convert('RGB'),(1880,1000));canvas=Image.new('RGB',(1920,1080),'#142436');canvas.paste(im,((1920-im.width)//2,10));ImageDraw.Draw(canvas).text((30,1030),'真实保存结果回放 · 非连续录屏 · 候选未确认为真值',font=small,fill='white');frame=av.VideoFrame.from_image(canvas)
   for _ in range(seconds*10):
    frame.pts=count;frame.time_base=Fraction(1,10);frame.pict_type=av.video.frame.PictureType.NONE
    for packet in stream.encode(frame):container.mux(packet)
    count+=1
  for packet in stream.encode():container.mux(packet)
  container.close();temp.replace(target)
  with av.open(str(target)) as f:decoded=sum(1 for _ in f.decode(video=0))
  assert count==decoded;receipts.append({'path':str(target.relative_to(out)),'frames':decoded,'seconds':count/10,'sha256':sha(target),'width':1920,'height':1080});print(json.dumps(receipts[-1],ensure_ascii=False),flush=True)
 save(out/'media_verification.json',{'passed':True,'pictures':len(gallery),'videos':receipts})
 weight=Path(cfg['package'])/'model/model.safetensors';expected='d11c2e7f86cd5189605b857c4697e55a5caead0a6888b52b2f5ff814150f5291';assert sha(weight)==expected
 protection={'passed':True,'annotation_writes':0,'training_updates':0,'weight_unchanged':True,'original_reference_annotations_unchanged':all(sha(cfg['reference_annotations'])==read(folder/'frozen_plan.json')['reference_annotation_sha256'] for folder in [context,candidate]),'source_images_unchanged':all(read(folder/'completed.json')['source_images_unchanged'] for folder in [context,candidate]),'protected_workbench_state_unchanged':all(read(folder/'completed.json')['protected_state_before_after_equal'] for folder in [context,candidate]),'original_scorer_sha256':sha(Path(cfg['source_snapshot'])/'tools/evaluate_projected_support.py')}
 protection['passed']=all(protection[k] for k in ['weight_unchanged','original_reference_annotations_unchanged','source_images_unchanged','protected_workbench_state_unchanged']) and protection['original_scorer_sha256']=='93c5152169d04d43b25dbf4a4139b12676daf78750a7f059837121656bb644c2'
 assert protection['passed'];save(out/'original_preservation.json',protection)
 incumbent_checks={'PAL_support_strict_gain_over_previous13':rows[3]['pallet_support']>rows[1]['pallet_support'],'PAL_points_not_below_previous490':rows[3]['pallet_points']>=rows[1]['pallet_points'],'person_points_not_lower':rows[3]['person_points']>=rows[1]['person_points'],'person_off_class_not_higher':rows[3]['person_off_class']<=rows[1]['person_off_class']}
 summary={'status':'experiments_complete','scope':'two exposed same-capture images; exploratory, not generalization','comparisons':rows,'context_acceptance':ca,'new_candidates_acceptance':na,'new_candidates_vs_previous_incumbent':incumbent_checks,'negative_controls_passed':controls['negative_controls_passed'],'positive_diagnostic_recall':controls['positive_diagnostic_recall'],'positive_diagnostics':2,'pictures':len(gallery),'videos':len(receipts),'new_fine_grid_forwards':84,'new_semantic_hypotheses_with_evidence_not_truth':sum(read(Path(cfg['instance_generation'])/'completed.json')['cases'][c]['C'] for c in ['camera0_frame76','camera1_frame76']),'automatic_adoption':False,'annotation_writes':0};save(out/'public_summary.json',summary)
 notes=f'''本轮固定前景版本托盘支持{rows[2]['pallet_support']}/17、参考点{rows[2]['pallet_points']}/579，收敛{ca['converged']}/6；新网格＋语义门控版本托盘支持{rows[3]['pallet_support']}/17、参考点{rows[3]['pallet_points']}/579，收敛{na['converged']}/6。
旧提示去掉类别枚举仍将灰白描述为托盘；中性提示保留相同枚举返回不确定。当前证据支持提示诱导为主要原因，不能归咎于图像未传入或仅改JSON格式。
中性观察的7个负例未报托盘，2个可见托盘诊断正例有证据；这不是独立分类准确率。最初raw_pallet命名裁图实际是墙/栏杆，不计入正例，通过正例仅取另行核对的蓝托盘和木底座。
 自然上下文版本完成90个候选的180次真实描述请求。仅检查pallet词会把pallet jack托盘车也通过；主体冲突或不确定的3个候选另存版本整项否决，保留4个完整原模型掩码。词汇门控仍不是密集语义验证，新候选仍需人工确认。与上轮13/17和490点也单独比较，不用仅优于原A12/17冒充本轮托盘提升。
84次细网格采用相同训练权重和阈值。灰底孤立像素版本在36个完整候选均无法确认后停止，未穷举153个；没有把这次停止称作全部候选通过。自然上下文版本另存，拒绝和不确定结果全部保留。超预算整项拒绝，掩码未作预算裁剪。
局部SAM裁图预检较差，未采用。固定前景对比把C地面初稿改用原SAM地面，保留训练地面原稿作对照；第一轮两次独立缺口编辑后一次真实SAM，第二轮沿用残余调度，第三轮真实观察。不能将地面初稿变化包装成仅调度的消融因果结论。
原三轮、每轮3项、累计32点、每个接纳缺口100%实际覆盖、原V2评分及1.5数量/1.25面积保持。A结果身份绑定复用；固定前景B/C新跑4组；新候选A/B复用、C新跑2组，缓存不冒充新调用。
所有新候选均未人工确认为真值；地面无密集真值，收敛不证明正确。原始标注与权重保留，默认工作台未替换，训练更新和标注写入均为0。
'''
 (out/'结论与限制.txt').write_text(notes,encoding='utf-8')
 page=['<!doctype html><html lang="zh"><meta charset="utf-8"><title>剩余托盘、三轮复核与分类负对照</title><style>body{max-width:1450px;margin:30px auto;padding:20px;font:17px/1.7 "Microsoft YaHei",sans-serif;background:#f2f6fa;color:#142436}section{padding:24px;background:white;margin:20px 0;border-radius:12px}img,video{max-width:100%}pre{white-space:pre-wrap}td,th{padding:10px;border:1px solid #ccd6df}table{border-collapse:collapse}</style><h1>剩余4个托盘 / 2组复核 / 分类负对照</h1><section><pre>'+html.escape(notes)+'</pre></section><section><h2>原V2稀疏参考对比</h2><table><tr><th>方案</th><th>托盘实例</th><th>托盘参考点</th><th>人员参考点</th><th>人员误覆盖</th></tr>']
 for r in rows:page.append(f"<tr><td>{r['name']}</td><td>{r['pallet_support']}/17</td><td>{r['pallet_points']}/579</td><td>{r['person_points']}/1568</td><td>{r['person_off_class']}</td></tr>")
 page.append('</table></section><section><h2>原验收与过程数据</h2>')
 for title,v in [('固定前景＋地面优化',ca),('新网格＋自然上下文语义门控',na)]:
  page.append('<h3>'+title+'</h3><pre>'+html.escape(json.dumps(v,ensure_ascii=False,indent=2))+'</pre>')
 for name in ['public_summary.json','original_preservation.json','optimization_plan.json','stage_audit/completed.json','semantic_contract/completed.json','cross_ablation/completed.json','neutral_controls/completed.json','spatial_preflight/completed.json','context_first_preflight/completed.json','instance_generation/stopped_preflight.json','instance_generation_contextual/C_selection.json']:
  page.append(f'<p><a href="{name}">{name}</a></p>')
 page.append('</section><section><h2>5段真实结果回放视频</h2><p>不是连续桌面录屏；全部视频逐帧解码核验。</p>')
 for v in receipts:page.append(f"<p>{v['path']} / {v['seconds']}秒</p><video controls preload='metadata' src='{v['path']}'></video>")
 page.append('</section>')
 for group,files in groups.items():
  page.append('<section><h2>'+group+'</h2>')
  for file in files:
   rel=file.relative_to(out).as_posix();page.append(f'<p>{html.escape(file.stem)}</p><a href="{rel}"><img loading="lazy" src="{rel}"></a>')
  page.append('</section>')
 page.append('</html>');(out/'index.html').write_text('\n'.join(page),encoding='utf-8');print(json.dumps(summary,ensure_ascii=False),flush=True)
if __name__=='__main__':main()
