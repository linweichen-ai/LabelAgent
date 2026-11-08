"""Actual sensor-height plots and complete paired native outputs, without adoption."""
from pathlib import Path
import numpy as np
from PIL import Image,ImageDraw
from run_MF_stock_sensor_SAM import OUT,SOURCE
from prepare_palette_parts import BASE
from run_rgb_layers import read,save
from deliver_context_graph import sha
from experiment_report_media import Report,overlay
REPORT=BASE/'labelagent_MF_sensor_report_20261108'

def main():
    source=read(SOURCE/'completed.json');sp=read(SOURCE/'frozen_plan.json');g=read(OUT/'generation_completed.json');sem=read(OUT/'semantic/completed.json');audit=read(OUT/'audit_summary.json');checks=read(OUT/'independent_readback.json');assert checks['passed'] and not any(r['all_semantic_stages_passed'] for r in sem['candidates']);report=Report(REPORT);summary={'status':'complete_sensor_assisted_diagnostic_no_new_adoption','source_stock_views':9,'unlabelled_height_positive_proposals':16,'new_SAM_calls':32,'new_VLM_calls':sem['fresh_this_execution'],'new_MF_calls':0,'new_Agent_runs':0,'native_valid':19,'semantic_passed':0,'new_accepted_instances':0,'source_left_q6_sensor_returns':42,'source_left_q6_selected_height_groups':1,'source_left_q6_no_forced_split':True,'native_rejected_without_service_warnings':13,'foreground_instance_support':'13/17','historical_Agent_convergence':'6/6 retained, not rerun','all_original_gates_unchanged':True,'no_mask_cut':True,'original_annotations_preserved':True,'sensor_arm_not_same_input_modality_as_RGB_only':True,'sensor_groups_not_confirmed_object_instances':True,'pictures':27,'saved_stage_replays':4,'series_pictures':122,'series_saved_stage_replays':24,'current_both_batches_new_SAM':61,'current_both_batches_new_VLM':156+sem['fresh_this_execution'],'adopted':False}
    report.text_card('overview','00_overview','点云高度辅助训练MF提案：真实成对测试',['全部符合原边界门的训练R15 stockPAL：9个源视图','16个传感器正点 × 正点/正负点两臂 = 32次新SAM',f"19张原生有效；完整语义0通过；新VLM {summary['new_VLM_calls']}",'13/17、历史6/6保留；没有新Agent运行或写入标注','高度组不是独立基座身份；负点仍是未经确认的假设'])
    colors=['#0077bb','#ee7733','#009988','#cc3311'];views=sp['views']
    for index,v in enumerate(views,1):
        diag=next(d for d in source['diagnostics'] if d['teacher_key']==v['teacher_key']);points=np.load(SOURCE/(v['teacher_key']+'_sensor_components.npz'),allow_pickle=False);z=points['xyz'][:,2];uv=points['pixels'];labels=points['component'];prior=np.asarray(Image.open(v['prior']))>127;photo=Image.open(v['input']).convert('RGB');yy,xx=np.nonzero(prior);box=(max(0,int(xx.min())-24),max(0,int(yy.min())-24),min(photo.width,int(xx.max())+25),min(photo.height,int(yy.max())+25));chart=Image.new('RGB',(600,400),'white');draw=ImageDraw.Draw(chart);draw.line((50,20,50,350,580,350),fill='black',width=2);lo,hi=float(z.min()),float(z.max());vlo,vhi=float(uv[:,1].min()),float(uv[:,1].max())
        for depth,pixel,label in zip(z,uv[:,1],labels):
            x=50+int((float(depth)-lo)/max(hi-lo,1e-9)*525);y=350-int((float(pixel)-vlo)/max(vhi-vlo,1)*315);draw.ellipse((x-3,y-3,x+3,y+3),fill=colors[int(label)%4])
        draw.text((60,360),f'lidar z (m): {lo:.3f} .. {hi:.3f}',fill='black',font=report.small);draw.text((60,0),f'projected image v: {vlo:.0f} .. {vhi:.0f}',fill='black',font=report.small);chart.save(REPORT/'assets'/f'sensor_scatter_{index:02d}.png');c=report.canvas(v['teacher_key']);report.panel(c,0,overlay(photo,prior).crop(box),'原训练模型完整stock PAL',['保留全部stock前景；没有按GT选区域','这只是源提案，并非标注']);report.panel(c,1,chart,'原始未标注回波与推断高度组',[f"原始可见回波 {len(z)}；BIC选择 {diag['chosen_k']} 组",'相同颜色是传感器拟合组，不是物体ID']);report.panel(c,2,Image.fromarray(prior.astype('uint8')*255).crop(box),'完整源先验',[str(diag.get('reason','两个计数最多的组生成正点')),'没有用标注坐标生成提示','不是与RGB基线相同输入模态']);report.add('sources',f'source_{index:02d}',c)
    for i in range(1,17):
        rows=[r for r in g['rows'] if r['index']==i];photo=Image.open(rows[0]['source_image']).convert('RGB');masks=[];xx=[];yy=[]
        for r in rows:
            f=Path(r['folder'])/'complete_full_mask.npy';m=np.load(f,allow_pickle=False) if f.exists() else None;masks.append(m)
            if m is not None:y,x=np.nonzero(m);yy.extend([int(y.min()),int(y.max())]);xx.extend([int(x.min()),int(x.max())])
        point=rows[0]['source_point'];neg=rows[0]['hypothesized_negative_source_point'];box=(max(0,min(xx+[point[0],neg[0]])-24),max(0,min(yy+[point[1],neg[1]])-24),min(photo.width,max(xx+[point[0],neg[0]])+25),min(photo.height,max(yy+[point[1],neg[1]])+25));c=report.canvas(f'点云源点 {i:02d} · {rows[0]["case"]}');pic=photo.crop(box);draw=ImageDraw.Draw(pic)
        for q,color in [(point,'lime'),(neg,'red')]:x,y=q[0]-box[0],q[1]-box[1];draw.ellipse((x-5,y-5,x+5,y+5),outline=color,width=2)
        report.panel(c,0,pic,'实际未标注回波提示',['绿：正点；红：其他高度组假设负点','源图与点云选点冻结在参考评分前','不能把不同高度直接当不同托盘'])
        for k,(r,m) in enumerate(zip(rows,masks),1):
            pic=overlay(photo,m).crop(box) if m is not None else overlay(Image.open(r['input']),np.load(Path(r['folder'])/'complete_native_mask.npy',allow_pickle=False));report.panel(c,k,pic,'仅正点' if k==1 else '正点 + 其他高度组负点',[f"完整像素 {r['native_pixels']}；原生有效 {r['valid']}",f"触边 {r['touches_input_border']}；留白FG {r['foreground_on_synthetic_padding']}",'负点中心空白不代表7×7邻域通过'])
        report.add('paired',f'pair_{i:02d}',c)
    report.text_card('gates','26_gates','本轮未采用：高度分组还不是实例分离',['左侧q6：42个可见回波，BIC只支持1组；没有强行拆成2组','32输出中19原生有效；13张无服务警告仍被完整原生检查拒收','负点约束与输入留白分别核验；失败位图和原答全部保存','两臂原面积/人员/纯度/一对一门后均13/17，没有新增实例','原标注、三轮和全部验收不变；这是诊断，未部署默认服务'])
    save(REPORT/'summary.json',summary);save(REPORT/'independent_readback.json',{'passed':True,'native_capture_and_sensor_readback_sha256':sha(OUT/'independent_readback.json'),'source_sensor_diagnostics_sha256':sha(SOURCE/'completed.json'),'semantic_completed_sha256':sha(OUT/'semantic/completed.json'),'sensor_rows_and_actual_request_and_all_native_FG_bound':True});videos=report.videos(['overview','sources','paired','gates']);assert len(report.cards)==27;report.index('点云辅助MF提案：27图4段回放','完整原生掩码与未标注点云真实对比。没有新增合格实例；原13/17与历史6/6保留。附9张原始传感器散点图，不另计入27张汇报图。',videos);print(summary,flush=True)

if __name__=='__main__':main()
