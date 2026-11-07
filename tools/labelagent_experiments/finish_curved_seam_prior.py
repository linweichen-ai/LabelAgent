"""Saved RGB seams, priors, untouched native masks and strict comparisons."""
from pathlib import Path
import numpy as np
from PIL import Image
from run_rgb_layers import ROOT,read,save,sha
from experiment_report_media import Report,overlay
import run_curved_seam_prior as trial
OUT=trial.BASE/'labelagent_curved_seam_report_20261108'
def main():
    assert read(trial.OUT/'independent_readback.json')['passed'];p=read(trial.OUT/'frozen_plan.json');g=read(trial.OUT/'generation_completed.json');assert g['actual_new_SAM_calls']==8 and all(sha(f)==h for f,h in p['protected'].items());report=Report(OUT);checks=[]
    audits={arm:read(trial.OUT/'audits'/arm/'support_area_audit.json') for arm in ['original','curved']}
    for d in audits.values():assert all(r['old_matches']==r['geometry_area_person_valid_matches']==r['unfiltered_all_matches'] for c in d['cases'] for r in c['radii'])
    for parent in p['parents']:
        n=parent['parent'];rows=[r for r in g['rows'] if r['parent']==n];assert len(rows)==2;original=next(r for r in rows if r['arm']=='original');curved=next(r for r in rows if r['arm']=='curved');photo=Image.open(original['input']).convert('RGB');card=report.canvas(f'原图动态路径先验 {n}：不是返回结果、没有参考点参与生成')
        folder=trial.OUT/'parents'/f'{n:02d}';report.panel(card,0,Image.open(folder/'source_RGB_seams.png').convert('RGB'),'原RGB边缘路径 / 原正点',['黄线为上路径，青线为下路径','路径按图像梯度、平滑及源正点计算','未读取标注或验收面积预算'])
        for k,row in enumerate([original,curved],1):
            m=np.asarray(Image.open(row['prior']))>0;report.panel(card,k,overlay(photo,m),'原完整源候选先验' if k==1 else '新原图路径先验',[f'提示先验像素{int(m.sum())}','同一张原图、同一个既有正点','仅作为SAM输入，未当作分割答案'])
        report.add('source',f'source_{n:02d}',card);card=report.canvas(f'完整SAM原生返回 {n}：先验变化、独立实例仍未增加')
        report.panel(card,0,photo,'实际提交的未标记原照片',[parent['case'],f'源点{parent["source_positive"]}','已有PAL主体复核复用，未新增语义调用','完整单基座语义尚未确认'])
        for k,row in enumerate([original,curved],1):
            x1,y1,x2,y2=row['box'];full=np.load(Path(row['folder'])/'complete_full_mask.npy',allow_pickle=False);mask=full[y1:y2,x1:x2];prior=np.asarray(Image.open(row['prior']))>0;a=next(r for r in audits[row['arm']]['candidates'] if r['index']==row['index']);r3=next(r for c in audits[row['arm']]['cases'] if c['case']==row['case'] for r in c['radii'] if r['radius']==3);own=next(r['ownership'] for r in r3['new_mask_ownership'] if r['key']==row['key']);notes=[f"完整原生面积{row['raw_pixels']} / 面积通过{a['area_fits']}",f"模型在先验外新增{int(((mask!=0)&~prior).sum())}像素；无输出裁切"]
            notes += [f"obj{v['object_id']} {v['points']}点 / 纯度{v['purity']:.3f}" for v in own[:4]]
            report.panel(card,k,overlay(photo,mask),'原先验 → 完整原生输出' if k==1 else '曲线路径 → 完整原生输出',notes);checks.append({'index':row['index'],'native_pixels_added_outside_prior':int(((mask!=0)&~prior).sum()),'native_pixels_removed_from_prior':int((prior&(mask==0)).sum()),'complete_native_mask_untouched':True})
        report.add('native',f'native_{n:02d}',card)
    binding=read(ROOT/'outputs/labelagent_point_ids_20261107/incumbent_binding.json')['files'];assert all(sha(ROOT/f)==h for f,h in binding.items())
    report.text_card('native','summary','曲线边缘先验仍未解决粘连，全部候选保留并拒收',['新增8次真实SAM：4组相同RGB/原正点的先验对照','全部8个接口返回和原生提示约束通过；最终实例支持仍无新增','蓝色叠层目标覆盖仍为26/27，纯度从0.313到0.351，未达0.67','曲线先验仍保留大部分整摞；原图强外缘没有定位独立层间界限','另一个旧实例支持从16/24降至10/24，先验变小不等于更正确','13/17与历史6/6保留，原始标注和所有验收门槛不变','本轮0次Agent；历史6/6未重跑，视频为实际保存阶段回放'])
    report.text_card('native','limits','下一步应识别层间界限，不能用参考点或面积反推输出',['先验由源模型候选及原图边缘产生，属于未确认假设','当前路径只受原正点与外框限制，仍会追随整摞较强外缘','将检查由独立源候选锚点约束层间搜索，保持可复核的图像证据','所有返回像素保持，不能以削去邻层作为最终分割修复','完整基座语义没有通过；PAL主体初筛不算实例确认','没有采用任何新候选，不把同批结果称为泛化提升'])
    videos=report.videos(['source','native']);summary={'status':'complete_rejected_curved_source_seam_prior','actual_new_SAM_requests':8,'actual_new_VLM_calls':0,'exact_prior_positive_reviews_reused':4,'prior_pairs':4,'native_masks_retained':8,'original_radii':[0,3,6],'all_original_gates_unchanged':True,'retained_support':'13/17','retained_historical_Agent':'6/6','historical_Agent_not_rerun':True,'new_Agent_runs':0,'remaining_misses':4,'new_accepted_instances':0,'all_8_native_point_guards_passed':True,'no_output_mask_clipping':True,'adopted':False,'target48_paired_r3_points':[26,26],'target48_paired_r3_purity':[26/83,26/74],'old49_paired_r3_points':[16,10],'pictures':len(report.cards),'videos':len(videos),'saved_stage_replay_not_screen_recording':True};save(OUT/'summary.json',summary);save(OUT/'independent_readback.json',{'passed':True,'source_trial_readback':sha(trial.OUT/'independent_readback.json'),'all_source_protected_exact':True,'all_six_original_results_exact':True,'native_prior_deltas':checks});text='4组先验对照共新增8次SAM。曲线边缘先验没有新增独立实例：蓝色叠层仍覆盖26/27，但混入邻层，纯度0.313→0.351，低于原0.67；另一旧实例点支持16/24→10/24。先验变小没有证明更正确，全部原生返回完整保存、未裁输出、未采用。原13/17与历史6/6保持，历史Agent未重跑，仍剩4个。0次新增VLM；PAL正点主体复核按原答复用，未认定完整基座语义通过。视频为保存阶段回放。';(OUT/'本轮结论.txt').write_text(text,encoding='utf-8');report.index('原图曲线边缘先验与完整原生掩码对比',text,videos);print(summary,flush=True)
if __name__=='__main__':main()
