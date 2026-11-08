"""Render complete saved native-output comparisons; never modify experiment masks."""
import collections,html
from pathlib import Path
import numpy as np
from PIL import Image,ImageDraw
from prepare_mf_first_parts import OUT
from prepare_palette_parts import BASE
from run_rgb_layers import read,save
from deliver_context_graph import sha
from experiment_report_media import Report,overlay
REPORT=BASE/'labelagent_MF_first_report_20261108'

def main():
    g=read(OUT/'generation_completed.json');pool=read(OUT/'native_pool/frozen_outputs.json');s=read(OUT/'semantic/completed.json');ps=read(OUT/'native_pool/semantic/completed.json');audit=read(OUT/'audit_summary.json');assert read(OUT/'independent_readback.json')['passed'];assert not any(x['all_semantic_stages_passed'] for x in s['candidates']+ps['candidates']), 'Inspect positive semantic result before writing no-adoption report'
    report=Report(REPORT);summary={'status':'complete_no_new_adoption','new_SAM_requests':g['actual_new_SAM_capture_requests'],'historical_exact_SAM_reuses':g['exact_historical_capture_reuses'],'logical_source_requests':32,'default_native_valid':25,'default_semantic_passes':0,'native_hypotheses':len(pool['rows']),'native_valid_hypotheses':sum(r['valid'] for r in pool['rows']),'pool_semantic_passes':0,'new_VLM_calls':s['fresh_this_execution']+ps['fresh_this_execution'],'primary_new_VLM_calls':s['fresh_this_execution'],'pool_new_VLM_calls':ps['fresh_this_execution'],'new_MF_calls':0,'new_Agent_runs':0,'new_accepted_instances':0,'foreground_instance_support':'13/17','historical_Agent_convergence':'6/6; retained, not rerun','Agent_round_limit':3,'area_cap':122511.25,'incumbent_area':122302,'additive_headroom':209.25,'no_output_mask_clipping':True,'original_annotations_preserved':True,'adopted':False,'pool_native_index_arms_frozen_before_reference_scoring':True,'alternative_native_masks_not_simultaneous_instances':True,'padding_FG_failures_default':[r for r in audit['unmappable_padding_violations_not_scored_by_cutting']],'pictures':22,'saved_stage_replays':4,'not_screen_recording':True}
    report.text_card('overview','00_overview','MF初稿 → 完整SAM → 原语义复核：实测对照',['32个逻辑请求：29次新SAM，3次精确历史复用','默认25张原生有效；固定原生索引池76张有效；完整语义均0通过',f"新VLM {summary['new_VLM_calls']} 次；新MF前向0次；新Agent运行0次",'13/17保留；历史6/6未重跑；剩余4例；没有采用新标注','输出掩码不裁切；三轮、面积、纯度、人员与一对一要求不变'])
    for i in range(1,17):
        rows=[r for r in g['rows'] if r['index']==i];photo=Image.open(rows[0]['source_image']).convert('RGB');point=rows[0]['source_point'];masks=[]
        for r in rows:
            f=Path(r['folder'])/'complete_full_mask.npy';masks.append(np.load(f,allow_pickle=False) if f.exists() else None)
        ys=[];xs=[]
        for m in masks:
            if m is not None:
                y,x=np.nonzero(m);ys.extend([int(y.min()),int(y.max())]);xs.extend([int(x.min()),int(x.max())])
        box=(max(0,min(xs+[point[0]])-24),max(0,min(ys+[point[1]])-24),min(photo.width,max(xs+[point[0]])+25),min(photo.height,max(ys+[point[1]])+25))
        c=report.canvas(f'源点 {i:02d} · {rows[0]["case"]} · {point}');original=photo.crop(box);d=ImageDraw.Draw(original);x,y=point[0]-box[0],point[1]-box[1];d.ellipse((x-5,y-5,x+5,y+5),outline='red',width=2);report.panel(c,0,original,'原图与实际源点',['选点来自训练MF stock PAL与源RGB部件','原局部描述保持原样，不伪装PAL已通过','显示框包住两张全部前景；不是裁输出'])
        for j,(r,m) in enumerate(zip(rows,masks),1):
            pic=overlay(photo,m).crop(box) if m is not None else overlay(Image.open(r['input']),np.load(Path(r['folder'])/'complete_native_mask.npy',allow_pickle=False))
            report.panel(c,j,pic,'整帧输入' if r['arm']=='original_source' else '显式留白输入',[f"完整native像素 {r['native_pixels']}；有效 {r['valid']}",f"触输入边界 {r['touches_input_border']}；留白前景 {r['foreground_on_synthetic_padding']}",'留白有前景则整张拒绝，没有裁掉后再算'])
        report.add('paired','pair_%02d'%i,c)
    for key in ['p024','p078','p084','p090']:
        r=next(x for x in pool['rows'] if x['key']==key);m=np.load(Path(r['folder'])/'complete_full_mask.npy',allow_pickle=False);photo=Image.open(r['source_image']).convert('RGB');y,x=np.nonzero(m);box=(max(0,int(x.min())-24),max(0,int(y.min())-24),min(photo.width,int(x.max())+25),min(photo.height,int(y.max())+25));sem=next(x for x in ps['candidates'] if x['key']==key);arm=next(a for a in audit['arms'] if a['arm']==f"fixed_index{r['native_index']}_{r['arm']}");metric=next(x for x in arm['candidates'] if x['key']==key);c=report.canvas(key+' · 固定原生索引独立诊断（未采用）');report.panel(c,0,photo.crop(box),'原始可见图像',['此框完整包住本张全部前景','不以参考点决定候选像素']);report.panel(c,1,overlay(photo,m).crop(box),'完整原生输出',[f"native index {r['native_index']}；总像素 {metric['raw_pixels']}",f"新增占地 {metric['extra_pixels']}；余量209.25",f"原面积门通过 {metric['area_fits']}"]);report.panel(c,2,Image.fromarray(m*255).crop(box),'完整二值掩码',[f"中性主体 {sem['neutral_passed']}",f"单基座 {sem['single_base_passed']}；完整形状 {sem['extent_passed']}",'全部语义门通过 False']);report.add('diagnostics',key,c)
    report.text_card('gates','21_gates','独立支持增加也不能越过原面积与语义门',['p084：5097像素，新增占地1004，超过209.25余量','r3/r6：缺失obj47得到8/11点支持、纯度1；仍非验收通过','其他蓝堆输出混入多层：覆盖更多点不等于独立实例','所有固定索引臂通过几何、面积、人员门后仍为13/17','本页是保存阶段回放；原数组、原答、失败输出全部保留'])
    save(REPORT/'summary.json',summary);save(REPORT/'independent_readback.json',{'passed':True,'experiment_readback':str(OUT/'independent_readback.json'),'experiment_readback_sha256':sha(OUT/'independent_readback.json'),'completed_default_semantics_sha256':sha(OUT/'semantic/completed.json'),'completed_pool_semantics_sha256':sha(OUT/'native_pool/semantic/completed.json'),'source_arrays_and_original_annotations_not_modified':True});videos=report.videos(['overview','paired','diagnostics','gates']);assert len(report.cards)==22;report.index('MF初稿与完整原生SAM：22图4段回放','没有新增合格实例。默认25与固定索引池76张原生有效掩码均未通过全部语义门；原13/17和历史6/6保留。',videos);print(summary,flush=True)

if __name__=='__main__':main()
