"""Actual point lineage, projection evidence, SAM attempts and bounded review."""
import json,sys,hashlib,shutil,html,textwrap
from pathlib import Path
from fractions import Fraction
ROOT=Path(__file__).resolve().parents[2];out=ROOT/'outputs/labelagent_projection_recovery_20261105';comparison=ROOT/'outputs/labelagent_projection_wire_comparison_20261105';inc=ROOT/'outputs/labelagent_context_comparison_20261105'
def read(p):return json.loads(p.read_text(encoding='utf-8'))
def save(p,v):p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(v,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
def main():
 summary=read(out/'public_summary.json');cfg=read(ROOT/'task_workspaces/labelagent_projection_recovery_20261105/config_wire_comparison.json');sys.path.insert(0,cfg['source_snapshot'])
 import numpy as np,cv2,av
 from PIL import Image,ImageDraw,ImageFont,ImageOps
 from source.agent.edit_agent import _rasterize
 assets=out/'assets';assets.mkdir(exist_ok=True);videos=out/'videos';videos.mkdir(exist_ok=True);font=ImageFont.truetype('C:/Windows/Fonts/msyh.ttc',26);small=ImageFont.truetype('C:/Windows/Fonts/msyh.ttc',19);large=ImageFont.truetype('C:/Windows/Fonts/msyh.ttc',36)
 gallery=[];groups={};seq={name:[] for name in ['01_逐点增减与实例匹配.mp4','02_四相机五邻帧投影.mp4','03_遮挡与原生时间证据.mp4','04_48次SAM实例试跑.mp4','05_三轮复核与完整掩码保护.mp4']};seen=set()
 def add(file,group):
  if file not in gallery:gallery.append(file);groups.setdefault(group,[]).append(file)
 def panel(title,items,file,notes=()):
  canvas=Image.new('RGB',(1920,1080),'#142436');d=ImageDraw.Draw(canvas);d.text((30,20),title,font=large,fill='white');cell=1840//len(items)
  for i,(label,img) in enumerate(items):
   d.text((40+i*cell,85),label,font=font,fill='#a7e8d5');im=ImageOps.contain(img.convert('RGB'),(cell-20,740));canvas.paste(im,(40+i*cell+(cell-20-im.width)//2,130+(740-im.height)//2))
  y=900
  for note in notes:
   for line in textwrap.wrap(note,135):d.text((30,y),line,font=small,fill='#d3e3ef');y+=34
  canvas.save(file,quality=92);return file
 # All source audit figures are shown, not cherry-picked by support gain.
 for folder,group,video in [('point_instance_audit','逐点归属与匹配','01_逐点增减与实例匹配.mp4'),('projection_visibility','原投影与遮挡风险','02_四相机五邻帧投影.mp4')]:
  for file in sorted((out/folder).glob('*.jpg')):
   add(file,group)
   if folder=='projection_visibility' and '_冲突核查' in file.name:seq['03_遮挡与原生时间证据.mp4'].append((file,4))
   else:seq[video].append((file,3 if folder=='point_instance_audit' else 2))
 # Exact original point-owner counts; no point deletion or scoring adjustment.
 audit=read(out/'point_instance_audit/completed.json');canvas=Image.new('RGB',(1920,1080),'#142436');d=ImageDraw.Draw(canvas);d.text((40,30),'为什么多覆盖3点却少支持1个实例',font=large,fill='white');y=130
 lines=['原保留方案：490/579点，13/17实例。被拒替换方案：493/579点，12/17实例。','逐点核对：新增11点，丢失8点；相机0新增8，相机1新增3并丢失8。','新增11点：相机0的对象36 +5、47 +1、48 +1、50 +1；相机1的对象28 +3。','其中多数对象原本已获支持；相机0对象48的额外1点仍未通过原0.67纯度。','丢失8点：相机1对象48 -7、对象49 -1。对象48由7/11支持降为0/11。','对象48在匹配前已失去合格候选，属于候选资格丢失，不是匹配算法抢名额。','并集覆盖点数不要求来自单一实例；实例支持还要满足纯度和一对一条件。','原半径3、纯度0.67、支持0.5、最低5点及全部参考点保持。']
 for line in lines:
  for piece in textwrap.wrap(line,95):d.text((40,y),piece,font=font,fill='#e0ebf3');y+=50
  y+=15
 file=assets/'11点新增与8点丢失_精确归属.jpg';canvas.save(file,quality=94);add(file,'逐点归属与匹配');seq['01_逐点增减与实例匹配.mp4'].insert(0,(file,8))
 timing=read(out/'timing_audit/completed.json');vis=read(out/'projection_visibility/completed.json');canvas=Image.new('RGB',(1920,1080),'#142436');d=ImageDraw.Draw(canvas);d.text((40,30),'投影、遮挡和时间：已确认与尚无证据',font=large,fill='white');y=140
 lines=['已确认：源文件、标签帧号、MCAP五路消息内容和同名时间戳一致；标定数值有效。','但MCAP日志/发布时间由导入器从文件名填写，不能证明硬件曝光同步。','五个邻帧的PCD虽然有time列，但全部为0；PNG也没有曝光时间元数据。','因此无法从这份数据恢复逐点扫描时间，也不能确认原去畸变状态。','相机0/42：原指标保留16点；16点的12像素邻域内均有更近点，提示遮挡风险。','邻域中较近点与目标深度差中位数约3米，但邻域不是同一相机像素的深度真值。','射线与参考三维框的相交只是框体代理；近处框12几乎与所有目标射线相交。','不能把框体当作完全不透明实体，也没有据此删点、重配时间或修改验收。','物理同步偏差、绝对外参误差和3D参考错误仍无法用现有证据独立确认。']
 for line in lines:
  for piece in textwrap.wrap(line,100):d.text((40,y),piece,font=font,fill='#e0ebf3');y+=52
  y+=14
 file=assets/'投影与原生时间_证据边界.jpg';canvas.save(file,quality=94);add(file,'原投影与遮挡风险');seq['03_遮挡与原生时间证据.mp4'].insert(0,(file,8))
 # All48 actual source requests remain visible, including warned/rejected outputs.
 for variant in ['instance_generation','instance_generation_blue']:
  for i,row in enumerate(read(out/variant/'sam_progress.json')):
   folder=Path(row['folder']);image=Image.open(folder/'input.png').convert('RGB');prompt=read(folder/'prompt.json');response=read(folder/'response.json');render=np.array(image).copy();points=prompt['points']
   if response.get('parts'):
    mask=_rasterize(response,image.width,image.height);render[mask!=0]=(.5*render[mask!=0]+.5*np.array([235,95,55])).astype('uint8')
   for p in points:cv2.circle(render,(int(p['x']),int(p['y'])),5,(30,255,100) if p['label']==1 else (255,60,180),-1)
   file=assets/(variant+'_'+str(i+1).zfill(2)+'_实际SAM点提示.jpg');notes=[f"{row['case']}；正点绿、负点紫；候选有效返回 {row['technically_valid']}。",str(row.get('rejection') or row.get('warnings') or '返回原始轮廓完整保留；有效返回不等于语义正确。'),'只按原预测区域和图像材质生成；没有输入参考对象位置。超预算整项拒绝。'];panel(variant+' / '+str(i+1)+' / 完整实际SAM结果',[('原始模型输入',image),('原始返回掩码与提示点',Image.fromarray(render))],file,notes);add(file,'48次真实SAM实例试跑');seq['04_48次SAM实例试跑.mp4'].append((file,2))
 # Original neutral photos are deduplicated by bytes, not selected by accuracy.
 for variant in ['instance_generation','instance_generation_blue','instance_generation_wire_exact']:
  for file in sorted((out/variant/'visual_cache').glob('*.png')):
   key=hashlib.sha256(file.read_bytes()).hexdigest()
   if key in seen:continue
   seen.add(key);target=assets/('原图语义证据_'+key[:16]+'.png');shutil.copy2(file,target);add(target,'中性图像复核证据')
 failed=read(out/'wire_geometry_failed_variant.json');canvas=Image.new('RGB',(1920,1080),'#142436');d=ImageDraw.Draw(canvas);d.text((40,30),'轮廓转换逐像素检查：失败保留与无损修正',font=large,fill='white');y=150
 for line in ['首次追加版本在轮廓重提取中丢18像素、另加3像素，未通过完整输出保持检查。','问题来自候选转换步骤，原有13/17全部前景未改变；该失败版本完整留档。','修正版直接平移SAM原始所有part和hole，不再把掩码重新提取为轮廓。','复用原48次推理，不增加SAM前向；逐像素对照原响应，任何差异即拒绝。','所有原27个托盘掩码与人员掩码逐像素保持；没有预算裁剪。','原面积122302；原上限122511.25；追加空间仅209.25像素。']:
  for piece in textwrap.wrap(line,95):d.text((40,y),piece,font=font,fill='#e0ebf3');y+=55
  y+=20
 file=assets/'完整轮廓保护_失败与修正.jpg';canvas.save(file,quality=94);add(file,'完整前景保护');seq['05_三轮复核与完整掩码保护.mp4'].append((file,7))
 for cam in [0,1]:
  case=f'camera{cam}_frame76';old=sorted((inc/'C_hybrid'/case).glob('*finished.jpg'))[-1];new=sorted((comparison/'C_hybrid'/case).glob('*finished.jpg'))[-1];file=assets/(case+'_保留前景对比.jpg');panel(case+' / 原13实例前景与完整追加试跑',[('保留版本',Image.open(old)),('独立追加试跑',Image.open(new))],file,('左图全部原托盘和人员掩码在右图逐像素保持。','相同原SAM地面种子、三轮上限、每轮3操作、累计32点；收敛不等于地面真值。'));add(file,'完整前景保护');seq['05_三轮复核与完整掩码保护.mp4'].append((file,4))
  for f in sorted((comparison/'C_hybrid'/case).glob('*.jpg')):
   target=assets/(case+'_Agent_'+f.name);panel(case+' / 实际Agent阶段 '+f.stem,[(f.stem,Image.open(f))],target,('真实保存结果回放，不是连续录屏。','最后修改仍须下一轮观察确认；所有未完成情况按原规则保留。'));add(target,'三轮Agent真实阶段');seq['05_三轮复核与完整掩码保护.mp4'].append((target,2))
 canvas=Image.new('RGB',(1920,1080),'#142436');d=ImageDraw.Draw(canvas);d.text((40,35),'本轮最终对比：保留13/17与6/6，原标准不变',font=large,fill='white');lines=[f"保留版本：13/17，490/579点，人员1349/1568点，人员误覆盖0，三轮6/6。",f"完整追加试跑：{summary['new_pallet_support']}/17，{summary['new_pallet_points']}/579点，人员{summary['new_person_points']}/1568点，误覆盖{summary['person_cross_class_points']}。",f"三轮收敛{summary['converged_groups']}/6；完整新增{summary['new_candidate_additions']}项；新增并集面积{summary['union_extra_pixels']}像素。",f"面积比例{summary['union_area_ratio']:.6f}，原上限1.25；严格超过13实例：{summary['new_pallet_support']>13}。",'参考点指标是稀疏投影一致性，不是像素准确率；不删除冲突点，不修改标签。','新增参考点与新增实例不同：旧被拒版本多3点，却丢失对象48的一对一支持。','所有新候选仍需人工确认；默认没有安装替换。',f"本轮结论：{'满足新候选工程比较项，仍需独立人工验收' if summary['new_candidate_adopted_as_offline_incumbent'] else '新实例支持没有严格提升，保留原13/17前景和6/6结果'}。"]
 y=160
 for line in lines:
  for piece in textwrap.wrap(line,100):d.text((40,y),piece,font=font,fill='#dceaf3');y+=55
  y+=20
 file=assets/'最终同图对比.jpg';canvas.save(file,quality=94);add(file,'最终结论');seq['01_逐点增减与实例匹配.mp4'].append((file,6));seq['05_三轮复核与完整掩码保护.mp4'].append((file,8))
 receipts=[]
 for name,frames in seq.items():
  path=videos/name;temp=path.with_suffix('.pending.mp4');container=av.open(str(temp),'w',format='mp4',options={'movflags':'+faststart'});stream=container.add_stream('libx264',rate=10);stream.width=1920;stream.height=1080;stream.pix_fmt='yuv420p';stream.codec_context.thread_count=2;stream.options={'crf':'24','preset':'veryfast'};count=0
  for file,seconds in frames:
   im=ImageOps.contain(Image.open(file).convert('RGB'),(1880,1000));canvas=Image.new('RGB',(1920,1080),'#142436');canvas.paste(im,((1920-im.width)//2,5));ImageDraw.Draw(canvas).text((30,1030),'真实模型/原始投影结果回放 · 非连续录屏 · 候选不是人工真值',font=small,fill='white');frame=av.VideoFrame.from_image(canvas)
   for _ in range(seconds*10):
    frame.pts=count;frame.time_base=Fraction(1,10);frame.pict_type=av.video.frame.PictureType.NONE
    for packet in stream.encode(frame):container.mux(packet)
    count+=1
  for packet in stream.encode():container.mux(packet)
  container.close();temp.replace(path)
  with av.open(str(path)) as f:decoded=sum(1 for _ in f.decode(video=0))
  assert decoded==count;receipts.append({'path':path.relative_to(out).as_posix(),'seconds':count/10,'frames':decoded,'sha256':hashlib.sha256(path.read_bytes()).hexdigest(),'width':1920,'height':1080});print(json.dumps(receipts[-1],ensure_ascii=False),flush=True)
 save(out/'media_verification.json',{'passed':True,'pictures':len(gallery),'videos':receipts});summary.update(pictures=len(gallery),videos=len(receipts),fully_decoded_video_frames=sum(r['frames'] for r in receipts));save(out/'public_summary.json',summary)
 notes=f'''本轮保留13/17与6/6前景结果。完整追加试跑为{summary['new_pallet_support']}/17、{summary['new_pallet_points']}/579点，收敛{summary['converged_groups']}/6；新候选采用：{summary['new_candidate_adopted_as_offline_incumbent']}。原始标注未写入，默认未替换。
旧493点却12实例：逐点新增11、丢失8，其中相机1对象48丢7点、对象49丢1点；对象48的纯度合格支持由7/11降0，在匹配之前已失去资格。新增点多数来自已支持对象，一对一支持不能由并集点数替代。
源文件/MCAP绑定正确，但原生五帧PCD time全0、PNG无曝光元数据。物理曝光同步、原去畸变和绝对外参精度不能从现有数据独立确认。8像素/.15米可见性只筛采样LiDAR栅格，无法保证相机稠密遮挡。相机0/42的16点附近12像素范围内都有较近LiDAR，中位深度差约3米，提示遮挡/跨物体风险；射线框相交仍只是代理，不能据此删点或断言参考错误。
两个实例试跑共48次真实SAM；第一候选转换曾丢18加3像素，整版未通过完整响应保持检查。修正版直接平移全部原轮廓/孔洞，复用原推理，逐像素保持；不裁掩码，不放宽面积。原27托盘全部保留，剩余原预算209.25像素。
本报告{len(gallery)}图、5段视频来自真实输入、模型返回与保存阶段；视频是结果回放，不是连续桌面录屏。原指标、三轮、每轮3操作、累计32点、人员保护及原始标注保持。
''';(out/'结论与限制.txt').write_text(notes,encoding='utf-8')
 page=['<!doctype html><html lang="zh"><meta charset="utf-8"><title>剩余托盘：逐点归属、投影遮挡与完整实例追加</title><style>body{max-width:1450px;margin:30px auto;padding:20px;font:17px/1.7 "Microsoft YaHei",sans-serif;background:#f2f6fa;color:#142436}section{background:white;padding:24px;border-radius:12px;margin:20px 0}img,video{max-width:100%}pre{white-space:pre-wrap}</style><h1>剩余托盘：逐点归属、投影遮挡与完整实例追加</h1><section><pre>'+html.escape(notes)+'</pre></section>']
 page.append('<section><h2>原验收与逐像素保护</h2><pre>'+html.escape(json.dumps(read(out/'acceptance_checks.json'),ensure_ascii=False,indent=2))+'</pre>')
 for name in ['point_instance_audit/completed.json','point_instance_audit/point_lineage.json','projection_visibility/completed.json','projection_visibility/target_point_depth_lineage.json','timing_audit/completed.json','wire_geometry_failed_variant.json','instance_generation_wire_exact/all_wire_mapping_checks.json','instance_generation_wire_exact/selection.json','whole_mask_preservation.json','original_preservation.json','budget_audit.json','round_analysis.json']:page.append(f'<p><a href="{name}">{name}</a></p>')
 page.append('</section><section><h2>5段真实结果回放</h2>')
 for r in receipts:page.append(f"<p>{r['path']} · {r['seconds']}秒</p><video controls preload='metadata' src='{r['path']}'></video>")
 page.append('</section>')
 for group,files in groups.items():
  page.append('<section><h2>'+group+'</h2>')
  for file in files:page.append(f'<p>{html.escape(file.stem)}</p><a href="{file.relative_to(out).as_posix()}"><img loading="lazy" src="{file.relative_to(out).as_posix()}"></a>')
  page.append('</section>')
 page.append('</html>');(out/'index.html').write_text('\n'.join(page),encoding='utf-8');print(json.dumps(summary,ensure_ascii=False),flush=True)
if __name__=='__main__':main()
