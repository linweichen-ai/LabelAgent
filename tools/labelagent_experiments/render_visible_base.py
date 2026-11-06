"""Real source views, failed coordinates, complete masks and retained runs."""
from pathlib import Path
helper=Path(__file__).with_name('render_instance_separation.py');template=helper.read_text(encoding='utf-8').replace('labelagent_instance_separation_20261105','labelagent_visible_base_20261107').replace('labelagent_instance_separation_comparison_20261105','labelagent_visible_base_comparison_20261107')
prefix=template[:template.index(' # Every original forward shown')];suffix=template[template.index(' receipts=[]'):]
body=r'''
 # All16 actual first-variant answers, including invalid source points.
 for i,row in enumerate(read(out/'planning/progress.json')):
  folder=Path(row['folder']);decision=row.get('decision',{});file=assets/f'物体结构规划_{i+1:02d}.jpg';notes=[str(decision.get('object'))+' / '+str(decision.get('instance_scope')),str(row.get('rejection') or '提示点通过来源绑定，不等于通过实例验收。'),'模型原始正点 '+str(decision.get('positive_points'))+'；负点 '+str(decision.get('negative_points'))]
  panel(f"seed{row['index']:02d} / {row['cue']} / 实际物体结构规划",[('模型实际收到的全貌',Image.open(folder/(row['cue']+'.png'))),('原始左部细节',Image.open(folder/'detail0.png')),('原始右部细节',Image.open(folder/'detail1.png'))],file,notes);add(file,'16次真实结构规划与坐标拒绝','01_同次推理与未竞争查询',3)
 # V2 is preparation, never shown as a successful new model answer.
 folder=out/'native_inputs/seed01';old=out/'planning/seed01';file=assets/'真实方形输入准备_共享模型忙未执行.jpg';panel('坐标输入修正：仅准备，未取得新模型回答',[('已失败首版输入',Image.open(old/'pointer.png')),('真实方形源图输入准备',Image.open(folder/'pointer.png'))],file,('修正版明确归一化坐标、去掉全貌留白；因共享模型持续繁忙没有执行成功。','原失败响应保留，不重解释、不夹取、不吸附坐标，不中断其他模型任务。'));add(file,'输入准备与执行边界','01_同次推理与未竞争查询',5)
 for p in sorted((out/'aspect_model_generation').glob('*/*/prediction.json')):
  folder=p.parent;r=read(p);rgb=np.array(Image.open(folder/'actual_padded_input.png').convert('RGB'));enc=np.asarray(Image.open(folder/'panoptic_padded.png')).astype('int32');ids=enc[:,:,0]+256*enc[:,:,1]+65536*enc[:,:,2]
  for seg in r['segments_info']:
   m=ids==seg['id'];color=np.array([(seg['id']*73)%220+20,(seg['id']*109)%220+20,(seg['id']*137)%220+20]);rgb[m]=(.5*rgb[m]+.5*color).astype('uint8')
  file=assets/f"同权重保持比例_{r['case']}_{r['index']:02d}.jpg";panel(r['case']+f" / seed{r['index']:02d} / 实际CPU模型输入",[('原始照片区域',Image.open(folder/'source_input.png')),('保持比例的实际448×576输入',Image.open(folder/'actual_padded_input.png')),('同次原配置分割ID着色',Image.fromarray(rgb))],file,(f"PAL查询{r['PAL_queries']}；可无损映射完整候选{r['accepted_complete_hypotheses']}。",'灰边是输入填充，不是实景；查询只要在填充有阳性像素就整项拒绝。'));add(file,'8次真实CPU同权重推理','02_可见接缝与无先验SAM',3)
 for i,row in enumerate(read(out/'aspect_model_generation/all_hypotheses.json')):
  source=Image.open(inc/'inputs'/(row['case']+'.png'));m=_rasterize(row['shape'],1920,1536);b=row['source_box'];file=assets/f'新完整模型查询_{i+1:02d}.jpg';panel(row['case']+f" / 完整查询{row['query_index']}",[('原始照片',source.crop(tuple(b))),('完整模型输出，未裁面积',overlay(source,m).crop(tuple(b)))],file,(f"类别得分{row['score']:.4f}；填充区阳性像素0。",'完整映射不等于单个可见基座；最终仍需原面积、语义、纯度和支持条件。'));add(file,'9条新完整查询记录','02_可见接缝与无先验SAM',3)
 for i,row in enumerate(read(out/'text_generation/progress.json')):
  folder=Path(row['folder']);r=read(folder/'response.json');file=assets/f'完整基座文字提示_{i+1:02d}.jpg';panel(row['case']+f" / seed{row['index']:02d} / 实际SAM文字请求",[('实际SAM源图',Image.open(folder/'input.png')),('原返回：没有有效掩码，保持照片',Image.open(folder/'input.png'))],file,('固定文字提示：single pallet base including visible deck and support feet。',str(r.get('diagnostics',{})),'返回0不作为成功检出；没有以手工轮廓补出结果。'));add(file,'8次真实SAM文字试跑','03_单实例正负对照与矛盾拦截',3)
 for f in sorted((out/'area_diagnostic').glob('*.jpg')):add(f,'完整旧候选的新增面积归属','04_剩余四实例的支持核查',4)
 for f in sorted((out/'instance_support_audit').glob('*.jpg')):add(f,'冻结后的实例纯度诊断','04_剩余四实例的支持核查',4)
 for i,row in enumerate(read(out/'instance_generation/verification_progress.json')['rows']):
  r=row['row'];photo=Image.open(inc/'inputs'/(r['case']+'.png'));m=_rasterize(r['shape'],1920,1536);ys,xs=np.where(m);b=[max(0,int(xs.min())-40),max(0,int(ys.min())-40),min(1920,int(xs.max())+41),min(1536,int(ys.max())+41)];file=assets/f'预算内原语义复核_{i+1:02d}.jpg';panel('原预算内候选 / '+r['key'][:8],[('原始照片',photo.crop(tuple(b))),('完整候选掩码',overlay(photo,m).crop(tuple(b)))],file,([d['object'] for d in row['neutral']],f"原中性双视图门槛通过：{row['neutral_passed']}；未追加。",'旧同输入响应复用；只有新输入实际新增语义请求，调用账目单列。'));add(file,'原门槛下的语义拒绝','03_单实例正负对照与矛盾拦截',3)
 for cam in [0,1]:
  case=f'camera{cam}_frame76'
  for f in sorted((comparison/'C_hybrid'/case).glob('*.jpg')):
   file=assets/(case+'_复用旧阶段_'+f.name);panel(case+' / 复用的旧真实阶段 '+f.stem,[(f.stem,Image.open(f))],file,('本轮未接纳新实例，全部7类初始掩码与旧运行逐项一致，复用原真实阶段。','本图不是新复核或连续录屏；本轮新Agent模型调用0，原6/6不受改写。'));add(file,'原真实阶段的身份绑定复用','05_三轮复测与前景完整保护',2)
 lines=['本轮最终保持13/17托盘、490/579点，人员3/3与1349/1568点，原6/6结果保留。','5/6纯度候选：完整8289像素，单一连通部分；新增3132都在主体中，预算只剩209.25。','不能靠去掉离散噪点让该原候选过关；没有按面积差产生负点或裁输出。','16次结构规划有时能说出木托盘，但全部坐标绑定失败，不交给SAM。','真实方形/归一化坐标修正因共享模型忙未执行；不把输入准备当成推理。','8次同权重CPU前向得到9条完整查询；8次固定完整基座SAM文字提示均返回0。','新完整查询虽最多覆盖13/16点，仍未满足原实例纯度；没有新增可接纳实例。','全部候选冻结后才诊断参考；三轮、每轮3项、累计32点和原面积/数量不变。','全部初始掩码相同，六组旧真实结果逐项复用；不是新测得6/6，也未加轮数。','原标注写入0，训练更新0，默认替换0；完整失败输入和输出均保留。']
 canvas=Image.new('RGB',(1920,1080),'#142436');d=ImageDraw.Draw(canvas);d.text((30,25),'完整可见基座与原面积预算：本轮结论',font=large,fill='white');y=130
 for line in lines:
  for t in textwrap.wrap(line,94):d.text((30,y),t,font=font,fill='#dfebf4');y+=48
  y+=20
 file=assets/'最终原标准对比.jpg';canvas.save(file,quality=94);add(file,'最终结论','05_三轮复测与前景完整保护',8);seq['04_剩余四实例的支持核查'].append((file,6))
'''
suffix=suffix.replace('原始候选、失败v1与保守修正v2均保留。','失败物体规划、未执行的坐标修正、真实新模型输出与缓存复用边界均保留。')
start=suffix.index(" for name in ['public_summary.json'");end=suffix.index("page.append('</section><section><h2>5段真实阶段回放",start)
suffix=suffix[:start]+" for name in ['public_summary.json','acceptance_checks.json','optimization_plan.json','area_diagnostic/completed.json','planning/completed.json','planning/protocol_deviation.json','planning_native/failure.json','aspect_model_generation/completed.json','text_generation/completed.json','instance_generation/completed.json','instance_support_audit/completed.json','whole_mask_preservation.json','original_preservation.json','complete_generated_mask_proof.json','initial_seed_reuse_proof.json','budget_audit.json','本轮结论.md']:page.append(f'<p><a href=\"{name}\">{name}</a></p>')\n "+suffix[end:]
exec(compile(prefix+body+suffix,__file__,'exec'),{'__name__':'__main__','__file__':__file__})
