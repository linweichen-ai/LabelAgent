"""Source photographs and complete outputs; no invented segmentation media."""
import json,sys,hashlib,html,textwrap
from pathlib import Path
from fractions import Fraction
ROOT=Path(__file__).resolve().parents[2];out=ROOT/'outputs/labelagent_instance_separation_20261105';comparison=ROOT/'outputs/labelagent_instance_separation_comparison_20261105';inc=ROOT/'outputs/labelagent_context_comparison_20261105'
def read(p):return json.loads(p.read_text(encoding='utf-8'))
def save(p,v):p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(v,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
def main():
 s=read(out/'public_summary.json');cfg=read(ROOT/'task_workspaces/labelagent_instance_separation_20261105/config_comparison.json');sys.path.insert(0,cfg['source_snapshot'])
 import numpy as np,cv2,av
 from PIL import Image,ImageDraw,ImageFont,ImageOps
 from source.agent.edit_agent import _rasterize
 assets=out/'assets';assets.mkdir(exist_ok=True);(out/'videos').mkdir(exist_ok=True);font=ImageFont.truetype('C:/Windows/Fonts/msyh.ttc',25);small=ImageFont.truetype('C:/Windows/Fonts/msyh.ttc',19);large=ImageFont.truetype('C:/Windows/Fonts/msyh.ttc',34)
 groups={};seq={k:[] for k in ['01_同次推理与未竞争查询','02_可见接缝与无先验SAM','03_单实例正负对照与矛盾拦截','04_剩余四实例的支持核查','05_三轮复测与前景完整保护']}
 def add(file,group,video,seconds=2):groups.setdefault(group,[]).append(file);seq[video].append((file,seconds))
 def panel(title,items,file,notes=()):
  canvas=Image.new('RGB',(1920,1080),'#142436');d=ImageDraw.Draw(canvas);d.text((28,20),title,font=large,fill='white');cw=1840//len(items)
  for j,(label,im) in enumerate(items):
   d.text((40+j*cw,85),label,font=font,fill='#a7e8d5');im=ImageOps.contain(im.convert('RGB'),(cw-24,730));canvas.paste(im,(40+j*cw+(cw-24-im.width)//2,135+(730-im.height)//2))
  y=900
  for note in notes:
   for line in textwrap.wrap(str(note),100):d.text((30,y),line,font=small,fill='#d6e8f4');y+=32
  canvas.save(file,quality=93);return file
 def overlay(im,mask,color=(0,225,165)):
  rgb=np.array(im.convert('RGB')).copy();rgb[mask!=0]=(.55*rgb[mask!=0]+.45*np.array(color)).astype('uint8');return Image.fromarray(rgb)
 # Every original forward shown, including no-PAL and discarded-query crops.
 for i,p in enumerate(sorted((out/'query_generation').glob('*/*/prediction.json'))):
  folder=p.parent;file=assets/f'模型同次推理_{i+1:02d}.jpg';record=read(p);photo=Image.open(folder/'input.png').convert('RGB');encoded=np.asarray(Image.open(folder/'panoptic.png')).astype('int32');ids=encoded[:,:,0]+256*encoded[:,:,1]+65536*encoded[:,:,2];painted=np.array(photo).copy()
  for seg in record['segments_info']:
   sid=seg['id'];m=ids==sid;color=np.array([(sid*73+40)%230+20,(sid*109+30)%230+20,(sid*137+10)%230+20]);painted[m]=(.5*painted[m]+.5*color).astype('uint8')
  panel('同次推理：原图与原竞争后结果',[('实际模型输入',photo),('原配置分割（按原分割ID着色）',Image.fromarray(painted))],file,('每个原预测区域配对32/96像素上下文；同一已训练权重，无新训练。','着色仅供可视化，不改变原分割ID；不把查询数量当作独立实例数。'));add(file,'16次配对模型推理','01_同次推理与未竞争查询')
 for i,row in enumerate(read(out/'query_generation/all_hypotheses.json')):
  box=row['source_box'];source=Image.open(inc/'inputs'/(row['case']+'.png'));mask=_rasterize(row['shape'],1920,1536);file=assets/f'完整查询候选_{i+1:02d}.jpg';panel(row['case']+f" / 原始查询{row['query_index']} / 未裁预算",[('原始输入',source.crop(tuple(box))),('完整查询掩码',overlay(source,mask).crop(tuple(box)))],file,(f"模型得分{row['score']:.4f}；仅为候选，不代表通过纯度、面积或语义检查。",'查询原二值掩码与提交轮廓逐像素一致；复杂度超限整项拒绝。'));add(file,'30个完整模型查询','01_同次推理与未竞争查询')
 # All 15 real requests, not just seven valid outputs.
 for i,row in enumerate(read(out/'seam_generation/progress.json')):
  folder=Path(row['folder']);im=Image.open(folder/'input.png').convert('RGB');prompt=read(folder/'prompt.json');r=read(folder/'response.json');rgb=np.array(im).copy()
  if r.get('parts'):rgb=np.array(overlay(im,_rasterize(r,im.width,im.height),(235,95,55)))
  for p in prompt['points']:cv2.circle(rgb,(int(p['x']),int(p['y'])),5,(40,255,90) if p['label']==1 else (255,40,170),-1)
  file=assets/f'可见接缝SAM_{i+1:02d}.jpg';panel(row['case']+f' / 接缝分离尝试{i+1}',[('实际SAM输入',im),('全部原始返回与实际提示点',Image.fromarray(rgb))],file,(f"绿正点、紫负点；无先验掩码；有效完整输出：{row['technically_valid']}。",str(row.get('rejection') or row.get('warnings') or '实际返回完整保留，不能按面积裁小。'),'相邻层的负点用于分离；没有输入三维参考位置。'));add(file,'15次真实SAM分离试跑','02_可见接缝与无先验SAM')
  if (folder/'synthetic_blank_response.json').exists():
   blank=Image.open(folder/'synthetic_blank_input.png');br=read(folder/'synthetic_blank_response.json');render=overlay(blank,_rasterize(br,blank.width,blank.height)) if br.get('parts') else blank;file=assets/f'合成空白SAM诊断_{i+1:02d}.jpg';panel('合成空白负对照：不是实景结果',[('真实场景返回',Image.fromarray(rgb)),('相同提示、无先验、合成空白返回',render)],file,('仅用于检查提示依赖；不得计入真实托盘检出或真实场景数据。',));add(file,'SAM合成对照','02_可见接缝与无先验SAM')
 # Actual bound views; display both the raw enum and conservative final gate.
 controls=read(out/'instance_generation_consistency/functional_controls/completed.json')['rows']
 for r in controls:
  folder=out/'instance_generation_consistency/functional_controls'/r['name'];e=read(folder/'structure_evidence.json')
  for j,row in enumerate(e['rows']):
   im=Image.open(Path(row['file']).with_suffix('.png'));decision=row['decision'];file=assets/(r['name']+'_'+row['cue']+'_实例对照.jpg');panel('功能对照 '+r['name']+' / '+row['cue'],[('原始功能裁图',Image.open(folder/'source.png')),('模型实际收到的定位视图',im)],file,(json.dumps(decision,ensure_ascii=False),f"矛盾拦截后的功能检查通过：{r['passed']}；暴露过的功能样本不是独立准确率。"));add(file,'单实例功能对照与矛盾拦截','03_单实例正负对照与矛盾拦截',3)
 for i,row in enumerate(read(out/'instance_generation_consistency/verification_progress.json')['rows']):
  shape=row['row']['shape'];case=row['row']['case'];source=Image.open(inc/'inputs'/(case+'.png'));mask=_rasterize(shape,1920,1536);yy,xx=np.where(mask);b=[max(0,int(xx.min())-32),max(0,int(yy.min())-32),min(1920,int(xx.max())+33),min(1536,int(yy.max())+33)];file=assets/f'预算内候选拒绝证据_{i+1:02d}.jpg';panel('进入原面积预算的候选 / '+str(i+1),[('原始照片',source.crop(tuple(b))),('候选完整掩码',overlay(source,mask).crop(tuple(b)))],file,([d['object'] for d in row['neutral']],f"原中性双视图语义通过：{row['neutral_passed']}；不追加。"));add(file,'预算内候选原语义拒绝','03_单实例正负对照与矛盾拦截',3)
 for file in sorted((out/'instance_support_audit').glob('*.jpg')):add(file,'冻结后的四实例支持核查','04_剩余四实例的支持核查',5)
 for cam in [0,1]:
  case=f'camera{cam}_frame76';old=sorted((inc/'C_hybrid'/case).glob('*finished.jpg'))[-1];new=sorted((comparison/'C_hybrid'/case).glob('*finished.jpg'))[-1];file=assets/(case+'_完整前景对照.jpg');panel(case+' / 原有前景逐像素保留',[('保留13/17版本',Image.open(old)),('这次原三轮复测',Image.open(new))],file,('全部原27托盘与3人员掩码保持；没有加轮数、裁掩码或调验收。','四组A/B复用原同输入结果；C两组从相同原始地面种子真实重跑。'));add(file,'完整前景保护','05_三轮复测与前景完整保护',4)
  for f in sorted((comparison/'C_hybrid'/case).glob('*.jpg')):
   file=assets/(case+'_Agent_'+f.name);panel(case+' / 实际保存阶段 '+f.stem,[(f.stem,Image.open(f))],file,('实际保存结果回放，非连续录屏。','三轮上限、每轮3操作、累计32点；收敛不能证明像素标注准确。'));add(file,'实际Agent阶段','05_三轮复测与前景完整保护')
 # One readable final card, with a strict instance outcome foremost.
 lines=[f"保留版本：13/17托盘，490/579点，人员1349/1568点，6/6收敛。",f"这次完整复测：{s['new_pallet_support']}/17托盘，{s['new_pallet_points']}/579点，{s['converged_groups']}/6收敛。",'16次模型配对推理：30个完整查询；15次真实无先验SAM：7个完整返回。','6个候选进入原剩余209.25像素预算，但均未通过原有中性语义复核。','其余完整候选按原面积/人员/重复约束整项拒绝，没有裁小输出。','修复单实例复核矛盾：解释含“stack/多个”时，即使状态写single也不放行。','正负功能对照是同一10次已保存响应重新解析，未增加Agent轮数或模型复核。',f"原严格实例增益通过：{s['new_candidate_adopted_as_offline_incumbent']}；仍4个漏检，不宣称已恢复。",'原标注写入0，训练更新0；同图工程实验，不代表独立准确率或上线验收。']
 canvas=Image.new('RGB',(1920,1080),'#142436');d=ImageDraw.Draw(canvas);d.text((35,25),'遮挡粘连：独立实例尝试与原验收结果',font=large,fill='white');y=130
 for line in lines:
  for t in textwrap.wrap(line,94):d.text((35,y),t,font=font,fill='#dfecf5');y+=50
  y+=20
 file=assets/'最终同图对比.jpg';canvas.save(file,quality=94);add(file,'最终结果','04_剩余四实例的支持核查',8);seq['05_三轮复测与前景完整保护'].append((file,6))
 receipts=[]
 for name,frames in seq.items():
  file=out/'videos'/(name+'.mp4');temp=file.with_suffix('.pending.mp4');container=av.open(str(temp),'w',format='mp4',options={'movflags':'+faststart'});stream=container.add_stream('libx264',rate=10);stream.width=1920;stream.height=1080;stream.pix_fmt='yuv420p';stream.codec_context.thread_count=2;stream.options={'crf':'24','preset':'veryfast'};count=0
  for picture,seconds in frames:
   im=ImageOps.contain(Image.open(picture).convert('RGB'),(1880,1000));canvas=Image.new('RGB',(1920,1080),'#142436');canvas.paste(im,((1920-im.width)//2,5));ImageDraw.Draw(canvas).text((30,1030),'真实保存结果回放 · 非连续录屏 · 候选不是人工真值',font=small,fill='white');frame=av.VideoFrame.from_image(canvas)
   for _ in range(seconds*10):
    frame.pts=count;frame.time_base=Fraction(1,10);frame.pict_type=av.video.frame.PictureType.NONE
    for packet in stream.encode(frame):container.mux(packet)
    count+=1
  for packet in stream.encode():container.mux(packet)
  container.close();temp.replace(file)
  with av.open(str(file)) as video:decoded=sum(1 for _ in video.decode(video=0))
  assert decoded==count;receipts.append({'path':file.relative_to(out).as_posix(),'seconds':count/10,'frames':decoded,'sha256':hashlib.sha256(file.read_bytes()).hexdigest(),'width':1920,'height':1080});print(json.dumps(receipts[-1],ensure_ascii=False),flush=True)
 pictures=sum(map(len,groups.values()));s.update(pictures=pictures,videos=5,fully_decoded_video_frames=sum(r['frames'] for r in receipts));save(out/'public_summary.json',s);save(out/'media_verification.json',{'passed':True,'pictures':pictures,'videos':receipts})
 notes='\n'.join(lines)+'\n\n全部视频来自实际输入、模型返回和保存阶段的回放，不是连续录屏。功能正负对照已暴露，不能当作独立泛化评测。三维参考仅在最终预测冻结后用于诊断。原始候选、失败v1与保守修正v2均保留。\n';(out/'结论与限制.txt').write_text(notes,encoding='utf-8')
 page=['<!doctype html><html lang="zh"><meta charset="utf-8"><title>遮挡与粘连：独立实例复测</title><style>body{max-width:1450px;margin:30px auto;padding:20px;font:17px/1.7 "Microsoft YaHei",sans-serif;background:#f2f6fa;color:#142436}section{background:white;padding:24px;border-radius:12px;margin:20px 0}img,video{max-width:100%}pre{white-space:pre-wrap}</style><h1>遮挡与粘连：独立实例复测</h1><section><pre>'+html.escape(notes)+'</pre></section><section><h2>原标准验收与逐像素保护</h2>']
 for name in ['public_summary.json','acceptance_checks.json','optimization_plan.json','instance_support_audit/completed.json','query_generation/completed.json','seam_generation/completed.json','instance_generation/completed.json','instance_generation_consistency/completed.json','instance_generation_consistency/functional_controls/completed.json','instance_generation_consistency/selection.json','whole_mask_preservation.json','original_preservation.json','budget_audit.json','round_analysis.json']:page.append(f'<p><a href="{name}">{name}</a></p>')
 page.append('</section><section><h2>5段真实阶段回放</h2>')
 for r in receipts:page.append(f"<p>{r['path']} · {r['seconds']}秒</p><video controls preload='metadata' src='{r['path']}'></video>")
 page.append('</section>')
 for group,files in groups.items():
  page.append('<section><h2>'+group+'</h2>')
  for file in files:page.append(f'<p>{html.escape(file.stem)}</p><a href="{file.relative_to(out).as_posix()}"><img loading="lazy" src="{file.relative_to(out).as_posix()}"></a>')
  page.append('</section>')
 page.append('</html>');(out/'index.html').write_text('\n'.join(page),encoding='utf-8');print(json.dumps(s,ensure_ascii=False),flush=True)
if __name__=='__main__':main()
