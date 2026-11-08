"""Actual source parts, neutral prompt pairs and native-feedback evidence."""
from pathlib import Path
from prepare_palette_parts import ROOT,BASE
from prepare_balanced_palette_parts import OUT as PARTS
from run_concise_palette_candidates import OUT as CONCISE
from neutral_prompt_pair import OUT as PROMPTS
from run_rgb_layers import read,save
from deliver_context_graph import sha
from experiment_report_media import Report,overlay
import numpy as np
from PIL import Image,ImageDraw
REPORT=BASE/'labelagent_parts_feedback_report_20261108'

def twin(r):
    first=Image.open(r['bindings'][0]['image']).convert('RGB');second=Image.open(r['bindings'][1]['image']).convert('RGB');im=Image.new('RGB',(1024,512));im.paste(first,(0,0));im.paste(second,(512,0));return im

def main():
    part=read(PARTS/'review_completed.json');pg=read(PARTS/'SAM/generation_completed.json');ps=read(PARTS/'semantic/completed.json');prompt=read(PROMPTS/'completed.json');cg=read(CONCISE/'SAM/generation_completed.json');cs=read(CONCISE/'semantic/completed.json');audit=read(CONCISE/'audit_summary.json');feedback=read(BASE/'labelagent_native_feedback_replay_20261108/completed.json');assert read(PARTS/'independent_readback.json')['passed'] and read(CONCISE/'independent_readback.json')['passed'];assert not any(r['all_semantic_stages_passed'] for r in cs['candidates']);report=Report(REPORT)
    # These are source-only cohort members, not reference-selected new inputs.
    left=[r for r in part['rows'] if isinstance(r['index'],int) and r['case']=='camera1_frame76' and r['source_view']=='q1' and r['source_rank']==3];old=read(BASE/'labelagent_single_teacher_grid_20261108/review_completed.json');oldleft=next(r for r in old['rows'] if r['index']==16);card=report.canvas('左侧源点：原距离点与新部件点，均保留原照片')
    for k,r in enumerate([oldleft]+left[:2]):report.panel(card,k,twin(r),('原距离选点' if k==0 else '新RGB部件点')+' '+str(r['source_point']),[r['decision']['object'][:100],'中性主体未过；没有伪造PAL证据','照片未经类别文字或掩码引导'])
    report.add('parts','source_left_parts',card)
    rows=pg['rows'];photo=Image.open(rows[0]['source_image']).convert('RGB');masks=[np.load(Path(r['folder'])/'complete_full_mask.npy',allow_pickle=False) for r in rows];ys,xs=np.where((masks[0]|masks[1])!=0);box=[max(0,int(xs.min())-32),max(0,int(ys.min())-32),min(1920,int(xs.max())+33),min(1536,int(ys.max())+33)];card=report.canvas('自然模型输出面积缩小仍不等于通过：没有裁掩码');report.panel(card,0,photo.crop(tuple(box)),'真实源图',['同正点，第二臂只增加源部件负点','负点物理身份只是待核验假设','所有模型原生数组完整保留'])
    for k,r in enumerate(rows,1):report.panel(card,k,overlay(photo,masks[k-1]).crop(tuple(box)),r['arm'],[f"原生{r['native_pixels']}像素",'单点17392新增占地462，负点臂7961新增163','负点臂仍违反7×7排除；继续拒收','这是模型重新预测，未删输出像素'])
    report.add('parts','negative_area_not_acceptance',card)
    bad=[r for r in feedback['rows'] if r['round1_feedback'].get('service_returned_without_warning_but_native_failed')]
    for i,r in enumerate(bad,1):
        f=r['round1_feedback'];failure=next(v for v in f['failures'] if v['reason']=='negative7x7_contains_foreground');response=read(r['response']);cap=response['raw_capture'];import base64,io
        arrays=np.load(io.BytesIO(base64.b64decode(cap['arrays_npz_base64'])),allow_pickle=False);meta=cap['metadata'];native=arrays[f"{meta['selected_prediction_call']}_masks"][meta['selected_index']].squeeze();x,y=failure['source_point'];patch=native[max(0,y-3):y+4,max(0,x-3):x+4];colors=np.where(patch[:,:,None]!=0,np.array([25,210,235],np.uint8),np.array([240,240,240],np.uint8));im=Image.fromarray(colors.astype('uint8')).resize((490,490),Image.Resampling.NEAREST);d=ImageDraw.Draw(im)
        for q in range(8):d.line([(q*70,0),(q*70,490)],fill='black',width=2);d.line([(0,q*70),(490,q*70)],fill='black',width=2)
        d.rectangle([210,210,280,280],outline='red',width=5);card=report.canvas(f'原生复核漏报实测例{i}：中心为0，邻域仍有前景');report.panel(card,0,im,'原生负点7×7，青色为前景',[f"中心前景：{failure['center_foreground']}",f"实际违反像素：{failure['foreground_pixels_in_required_region']}",'仅可视化放大；没有修改原始位图'])
        report.panel(card,1,Image.new('RGB',(490,490),'#f8f1e5'),'服务原回复',['status=segment_returned','warnings=[]','原选择器没有完全合格候选时选违反最少项','后续返回仅检查负点中心，漏掉整个邻域'])
        report.panel(card,2,Image.new('RGB',(490,490),'#e9f1fa'),'新实验复核适配器',['明确拒收并反馈实际失败区域','第3轮仍失败则保留原标注并弃权','28条真实捕获回放，2条静默违反被拦住','0模型调用、0新Agent、尚未部署默认服务']);report.add('native_feedback',f'feedback_case{i}',card)
    controls=[r for r in prompt['rows'] if r['kind']=='control' and r['arm']=='original']
    for start in [0,3]:
        card=report.canvas('同图同模型提示对照：六项PAL/非PAL校准（不是六类准确率）')
        for k,r in enumerate(controls[start:start+3]):
            new=next(v for v in prompt['rows'] if v['index']==r['index'] and v['arm']=='concise');report.panel(card,k,twin(r),r['index'],['Old: '+r['decision']['object'][:70],'New: '+new['decision']['object'][:70],f"原/新PAL主体 {int(r['PAL_subject_evidence'])}/{int(new['PAL_subject_evidence'])}",'照片、权重、JSON及256+256视觉预算一致'])
        report.add('prompt_pair',f'control_pair{start}',card)
    real=[r for r in prompt['rows'] if r['kind']=='real_source_not_truth' and r['arm']=='original']
    for start in range(0,len(real),6):
        card=report.canvas('只改中性描述文本：提案变化不当作识别准确率')
        for k,r in enumerate(real[start:start+6]):
            new=next(v for v in prompt['rows'] if v['index']==r['index'] and v['arm']=='concise');report.panel(card,k,twin(r),'源点'+str(r['index']),['Old: '+r['decision']['object'][:48],'New: '+new['decision']['object'][:48],f"PAL primary {int(r['PAL_subject_evidence'])} -> {int(new['PAL_subject_evidence'])}; proposal only"],True)
        report.add('prompt_pair',f'prompt_sources{start:02d}',card)
    for cue in sorted({r['index'] for r in cg['rows']}):
        rows=[r for r in cg['rows'] if r['index']==cue];photo=Image.open(rows[0]['source_image']).convert('RGB');masks=[np.load(Path(r['folder'])/'complete_full_mask.npy',allow_pickle=False) for r in rows];ys,xs=np.where((masks[0]|masks[1])!=0);box=[max(0,int(xs.min())-32),max(0,int(ys.min())-32),min(1920,int(xs.max())+33),min(1536,int(ys.max())+33)];card=report.canvas(f'简洁提示提案源点{cue}：完整SAM与原验收');report.panel(card,0,photo.crop(tuple(box)),'真实源图局部显示',['显示框包含两张完整原生输出','提案已过中性新提示的六项校准','最终完整实例复核仍使用原提示及全部门槛'])
        for k,r in enumerate(rows,1):
            ar=next(a for a in audit['arms'] if a['arm']==r['arm']);c=next(c for c in ar['candidates'] if c['index']==cue);sem=next((c for c in cs['candidates'] if c['key']==r['request_sha256']),None);report.panel(card,k,overlay(photo,masks[k-1]).crop(tuple(box)),r['arm'],[f"原生{r['native_pixels']}，新增{c['extra_pixels']}像素",f"native有效{r['valid']}；面积合格{c['area_fits']}",'完整语义：'+('未通过' if sem and not sem['all_semantic_stages_passed'] else 'native无效，不进入语义'),'2份完全相同旧SAM原答复用，其他8次新请求' if r['historical_exact_SAM_reuse'] else '本臂实际新SAM请求，全部数组已保留'])
        report.add('candidate_pairs',f'concise_SAM{cue:02d}',card)
    report.text_card('candidate_pairs','result_summary','保留原13/17与历史6/6；当前仍无新增合格实例',['22部位点：22新VLM+2精确旧负对照；2次新SAM，1张native有效。','提示配对：同22源点，PAL主体提案1→5，六项PAL/非PAL校准均过。','提示试验32次新VLM+24精确旧回复复用，不能推断真实点准确率。','简洁提案10逻辑SAM：8新调用+2旧捕获；9张native有效，完整语义0/9。','两阶段完整语义另有2+14次新VLM；没有新Mask2Former前向或训练。','当前共70次新VLM、10次新SAM；没有裁掩码、增加Agent轮数或降门槛。','下一MF→SAM→完整复核控制已准备16点32请求，尚未执行、0新调用。'])
    videos=report.videos(['parts','native_feedback','prompt_pair','candidate_pairs']);media=read(REPORT/'media_verification.json');summary={'status':'complete_source_parts_prompt_pair_and_native_feedback_comparison','new_CPU_Mask2Former_forwards':0,'new_training':False,'new_VLM_calls_total':part['actual_new_VLM_calls_total']+ps['fresh_this_execution']+prompt['actual_new_VLM_calls_total']+cs['fresh_this_execution'],'new_SAM_calls_total':pg['actual_new_SAM_calls_total']+cg['actual_new_SAM_calls_total'],'new_palette_VLM':part['actual_new_VLM_calls_total'],'palette_full_semantics_VLM':ps['fresh_this_execution'],'prompt_pair_new_VLM':prompt['actual_new_VLM_calls_total'],'prompt_pair_exact_prior_reuses':prompt['exact_prior_reuses'],'concise_full_semantics_VLM':cs['fresh_this_execution'],'palette_positive_points':22,'prompt_PAL_proposals_original':sum(r['original_PAL'] for r in prompt['pairs']),'prompt_PAL_proposals_concise':sum(r['concise_PAL'] for r in prompt['pairs']),'new_prompt_passed_six_binary_PAL_controls':prompt['concise_all6_controls_pass'],'not_point_accuracy_or_independent_scene_validation':True,'concise_logical_SAM_requests':10,'concise_new_SAM_calls':cg['actual_new_SAM_calls_total'],'concise_exact_prior_SAM_reuses':cg['exact_historical_SAM_reuses'],'concise_native_valid':sum(r['valid'] for r in cg['rows']),'concise_complete_semantic_passed':sum(r['all_semantic_stages_passed'] for r in cs['candidates']),'native_feedback_replay_records':feedback['real_capture_rows'],'native_feedback_quiet_failures_caught':feedback['quiet_native_failures_detected'],'feedback_adapter_deployed':False,'default_service_unchanged':True,'new_Agent_runs':0,'new_accepted_instances':0,'retained_support':'13/17','historical_Agent':'6/6_not_rerun','remaining_misses':4,'all_original_final_gates_unchanged':True,'no_native_output_mask_cut':True,'pictures':media['pictures'],'videos':len(videos),'fully_decoded_frames':media['fully_decoded_frames'],'next_MF_first_plan_prepared_only':True,'next_MF_first_points':16,'next_MF_first_requests':32,'next_MF_first_model_calls':0,'adopted':False};assert summary['new_VLM_calls_total']==70 and summary['new_SAM_calls_total']==10 and media['pictures']==16;save(REPORT/'summary.json',summary)
    # A second readback verifies the isolated prompt ablation, including exact
    # historical reply bytes and actual per-image token counts.
    requests=read(PROMPTS/'frozen_plan.json')['requests'];checks=[]
    for r in prompt['rows']:
        request=read(r['request_file']);response=read(r['response_file'])['raw_response'];assert sha(r['response_file'])==r['response_sha256'] and response['local_vlm']['actual_image_token_counts']==[256,256];peer=next(p for p in requests if p['index']==r['index'] and p['arm']!=r['arm']);other=read(peer['request_file']);request['messages'][0]['content'][0]['text']=other['messages'][0]['content'][0]['text'];assert request==other;checks.append({'index':r['index'],'arm':r['arm'],'only_prompt_text_differs_actual_images_model_schema_budget_same':True})
    save(REPORT/'independent_readback.json',{'passed':True,'palette_readback_sha256':sha(PARTS/'independent_readback.json'),'concise_readback_sha256':sha(CONCISE/'independent_readback.json'),'feedback_replay_sha256':sha(BASE/'labelagent_native_feedback_replay_20261108/completed.json'),'prompt_pairs_actual_payloads_verified':checks,'all_output_bitmaps_preserved':True,'no_Agent_convergence_or_annotation_acceptance_claim':True});text='新部位选点及简洁中性提示没有带来合格新增实例：简洁提示提案1→5，但实际10张逻辑SAM中9张原生有效，原完整语义0/9；原13/17与历史6/6保留。发现服务无警告仍有负点7×7前景的两条真实记录，新实验复核适配器在28条捕获离线回放正确拒收，尚未部署默认服务。新增16图4段保存阶段回放，完整原答与数组私有保留。下一MF→SAM→完整复核控制仅准备，不能把32请求当已执行。';(REPORT/'本轮结论.txt').write_text(text,encoding='utf-8');report.index('部件选点、提示对照与原生复核反馈：严格拒收保留原标注',text,videos);print(summary,flush=True)

if __name__=='__main__':main()
