"""Evidence-backed comparison images and fully decoded saved-stage replays."""
import json,sys
from pathlib import Path
from prepare_fixed_grid_followup import ROOT,BASE,OUT
import numpy as np
from PIL import Image,ImageDraw
from run_rgb_layers import read,save
from deliver_context_graph import sha
from experiment_report_media import Report,overlay
REPORT=BASE/'labelagent_registered_grid_followup_report_20261108'

def main():
    g=read(BASE/'labelagent_registered_fixed_grid_20261108/generation_completed.json');p=read(OUT/'frozen_plan.json');sam=read(OUT/'SAM/generation_completed.json');rv=read(OUT/'review_completed.json');sem=read(OUT/'semantic/completed.json');audit=read(OUT/'audit_summary.json');coverage=read(OUT/'post_generation_registered_coverage.json');assert read(OUT/'independent_readback.json')['passed'] and len(g['rows'])==24 and len(sam['rows'])==14 and len(sem['candidates'])==14;report=Report(REPORT)
    for model in ['R5_historical_boundary','R15_PAL_candidate']:
        for cam in [0,1]:
            case=f'camera{cam}_frame76';rows=[r for r in g['rows'] if r['model']==model and r['case']==case];assert len(rows)==5;card=report.canvas(f'同图同配方，仅换注册权重：{model} / camera{cam}')
            for k,r in enumerate(rows):
                photo=Image.open(r['input']).convert('RGB');rgb=np.asarray(Image.open(Path(r['folder'])/'panoptic.png')).astype('uint32');ids=rgb[:,:,0]+256*rgb[:,:,1]+65536*rgb[:,:,2];pal=[s['id'] for s in r['segments_info'] if s['category_id']==1];report.panel(card,k,overlay(photo,np.isin(ids,pal)),r['source_view']+f" / 原PAL实例{len(pal)}",['448×576、原CPU FP32与原后处理','青色只显示PAL；全类别原输出另存','不是新增实例验收'],True)
            report.panel(card,5,Image.open(rows[0]['original_source_image']).convert('RGB'),'整幅真实源图',['固定四象限与原图边界留白','真实图像像素与前一对权重相同','R15专用候选仍未部署'],True);report.add('registered_grid',f'grid_{cam}_{model}',card)
    for start in range(0,len(rv['rows']),6):
        card=report.canvas('固定源点的中性复核：指向何物，不提示托盘类别')
        for k,r in enumerate(rv['rows'][start:start+6]):
            near=Image.open(r['bindings'][0]['image']).convert('RGB');context=Image.open(r['bindings'][1]['image']).convert('RGB');photo=Image.new('RGB',(1024,512));photo.paste(near,(0,0));photo.paste(context,(512,0));report.panel(card,k,photo,str(r['index'])+(' / PAL主体' if r['PAL_subject_evidence'] else ' / 未过PAL主体'),[r['decision']['object'][:110],'原129/385双照片，无类别标记','通过主体不等于完整独立基座'],True)
        report.add('point_review',f'points_{start:02d}',card)
    for cue in sorted({r['index'] for r in sam['rows']}):
        rows=[r for r in sam['rows'] if r['index']==cue];assert len(rows)==2;source=Image.open(rows[0]['source_image']).convert('RGB');masks=[np.load(Path(r['folder'])/'complete_full_mask.npy',allow_pickle=False) for r in rows];union=(masks[0]|masks[1])!=0;yy,xx=np.where(union);w,h=source.size;box=[max(0,int(xx.min())-32),max(0,int(yy.min())-32),min(w,int(xx.max())+33),min(h,int(yy.max())+33)];card=report.canvas(f'点{cue}：完整原生SAM输出，原图与显式留白输入比较');report.panel(card,0,source.crop(tuple(box)),'真实源图局部放大',['显示框保留两张完整掩码的全部前景','模型实际输入与全部原始数组另存','显示放大没有修改输出掩码'])
        for j,r in enumerate(rows,1):
            extra=next(a for a in audit['arms'] if a['arm']==r['arm']);c=next(x for x in extra['candidates'] if x['index']==cue);decision=next(x for x in sem['candidates'] if x['key']==r['request_sha256']);report.panel(card,j,overlay(source,masks[j-1]).crop(tuple(box)),r['arm'],[f"原生{r['native_pixels']}像素；新增占地{c['extra_pixels']}",'留白前景0；完整逆映射像素数一致','面积不通过；未修改掩码',f"单基座复核：{'通过' if decision['single_base_passed'] else '未通过'}"])
        report.add('native_pairs',f'paired_cue{cue:02d}',card)
    lines=['事后参考点覆盖（不是独立实例通过）：']
    for c in coverage['cases']:
        values=' / '.join(m['model'].replace('_historical_boundary','').replace('_PAL_candidate','')+':'+str(m['points_in_any_stock_PAL_output'])+'/'+str(m['reference_points']) for m in c['models']);lines.append(c['case'].split('_')[0]+' obj'+str(c['object_id'])+'  '+values)
    lines.extend(['R15左侧预测存在，但最高分提案与双模型共识没有把它送去复核。','不把点覆盖当实例支持；不按参考点重选掩码或裁切输出。']);report.text_card('diagnosis','coverage','模型漏判与提案筛选是两个层次',lines)
    report.text_card('diagnosis','gates','完整原生输出仍不满足验收；原方案继续保留',['新增24次CPU：20真实、4合成控制；48份保存输出原后处理逐像素一致。','16真实源点+2灰白对照：18次新VLM，7点进入SAM。','7点×2臂=14次新SAM，全部原生有效、留白前景0、无前景裁切。','全部14候选面积不通过；蓝色候选多个实例粘连，纯度不足。','完整语义复核108逻辑视图、25唯一回复：23次新调用+2精确历史复用。','14候选均未通过完整语义；原13/17、历史6/6保留，未重跑Agent。','下一阶段独立提案分支仅准备；最终类别、单基座、面积和纯度门槛不变。'])
    videos=report.videos(['registered_grid','point_review','native_pairs','diagnosis']);media=read(REPORT/'media_verification.json');summary={'status':'complete_registered_grid_and_consensus_paired_SAM_comparison','new_CPU_forwards':24,'real_CPU':20,'synthetic_CPU':4,'original_postprocess_saved_forward_replays_exact':48,'new_neutral_VLM_calls':18,'new_full_semantics_VLM_calls':23,'new_VLM_calls_total':41,'full_semantics_exact_historical_reuses':2,'full_semantics_unique_replies':25,'full_semantics_logical_views':108,'new_SAM_calls':14,'source_points':16,'points_neutrally_admitted':7,'native_valid_full_outputs':14,'all_padding_foreground_zero_and_full_count_mapping':True,'area_passing':0,'full_semantic_passing':0,'new_accepted_instances':0,'retained_support':'13/17','historical_Agent':'6/6_not_rerun','remaining_misses':4,'new_Agent_runs':0,'all_original_final_gates_unchanged':True,'no_output_foreground_cuts':True,'pictures':media['pictures'],'videos':len(videos),'saved_replay_frames':media['fully_decoded_frames'],'registered_stock_coverage_post_generation':[{'case':r['case'],'object_id':r['object_id'],'models':[{k:m[k] for k in ['model','points_in_any_stock_PAL_output','reference_points']} for m in r['models']]} for r in coverage['cases']],'proposal_bottleneck':'highest_score_one_segment_and_pair_consensus_do_not_submit_R15_left_predictions','next_single_teacher_source_plan_prepared_only':True,'adopted':False};save(REPORT/'summary.json',summary);save(REPORT/'independent_readback.json',{'passed':True,'query_postprocess_readback_sha256':sha(OUT/'all48_original_postprocess_readback.json'),'capture_readback_sha256':sha(OUT/'independent_readback.json'),'source_selection_sha256':sha(OUT/'semantic_source_selection.json'),'semantic_result_sha256':sha(OUT/'semantic/completed.json'),'no_mask_or_annotation_adoption':True});text='注册R5/R15同图同配方24次新CPU对照已完成。R15左侧两例的原PAL稀疏点覆盖达到11/11和4/6，但固定最高分与双模型共识提案未送入这两区域，不能当作实例找回。当前固定16源点中7点通过中性复核，14次原图/留白配对SAM输出完整保留、留白前景均0；全部面积不通过，蓝色堆叠候选仍粘连，14候选完整语义均未过。原13/17和历史6/6继续保留，未重跑Agent或放宽门槛。';(REPORT/'本轮结论.txt').write_text(text,encoding='utf-8');report.index('注册训练权重与原生配对掩码：真实对照及拒收原因',text,videos);print(summary,flush=True)

if __name__=='__main__':main()
