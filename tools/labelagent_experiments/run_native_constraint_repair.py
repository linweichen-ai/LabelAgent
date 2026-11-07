"""Bounded native-constraint repair using model counterexamples, never mask edits."""
import hashlib,shutil,sys
from pathlib import Path
import cv2,numpy as np
from run_rgb_layers import ROOT,read,save,sha
from native_prompt_guard import evaluate
import run_trained_point_sam as sam
BASE=Path('D:/AI/Codex/Home/visualizations/2026/11/04/01a1a56c-0d6f-7d61-8326-d3bbd344d2df');SOURCE=BASE/'labelagent_mask_graph_core_20261108';OUT=BASE/'labelagent_native_constraint_repair_20261108'

def prepare():
    if OUT.exists():raise FileExistsError('Preserve original-bound repair trial')
    g=read(SOURCE/'SAM/generation_completed.json');p=read(SOURCE/'frozen_plan.json');assert len(g['rows'])==7;protected={str(SOURCE/'SAM/generation_completed.json'):sha(SOURCE/'SAM/generation_completed.json'),str(SOURCE/'frozen_plan.json'):sha(SOURCE/'frozen_plan.json'),str(Path(__file__)):sha(__file__),str(Path(sam.__file__)):sha(sam.__file__),str(Path(__file__).with_name('native_prompt_guard.py')):sha(Path(__file__).with_name('native_prompt_guard.py'))};catalogue={}
    for folder in [ROOT/'outputs/labelagent_aspect_context_cues_20261108',BASE/'labelagent_highres_context_cues_20261108']:
        for r in read(folder/'SAM/generation_completed.json')['rows']:catalogue[folder.name+'/'+r['sam_request_sha256']]=r
    cases=[]
    for r in g['rows']:
        detail=next(x for x in p['details'] if x['parent']==r['parent_rank']);x1,y1,x2,y2=r['box'];owners={}
        for n in detail['neighbor_hypotheses']:
            q=n['negative_source_point'];path=Path(catalogue[n['uid']]['folder'])/'complete_full_mask.npy';protected[str(path)]=sha(path);owners[f'{q[0]-x1},{q[1]-y1}']=str(path)
        for f in [r['input'],r['raw_array_file'],r['metadata_file'],str(Path(r['folder'])/'complete_full_mask.npy')]:protected[f]=sha(f)
        cases.append({'initial':r,'negative_source_neighbor_masks':owners})
    OUT.mkdir();save(OUT/'frozen_plan.json',{'cases':cases,'protected':protected,'original_max_SAM_requests_per_case':3,'max_cumulative_point_annotations_per_case':32,'stronger_total_point_annotations_across_all_cases':32,'initial_point_annotations':sum(len(c['initial']['points']) for c in cases),'max_new_negative_counterexamples_per_round':3,'candidate_rule':'Foreground violation pixels within original-radius3 negative neighborhood, with full original-radius3 source-neighbor interior and no conflicting positive; sparse7x7 coverage priority','no_reference_or_area_for_selection':True,'no_mask_pixel_changes':True,'all_existing_points_and_labels_preserved':True,'first_request_exact_saved_capture_reused_not_repeated':True,'native_constraint_stop_not_Agent_or_instance_acceptance':True,'new_Agent_runs':0,'Agent_round_limit':3,'all_original_final_gates_required':True});print({'cases':len(cases),'initial_saved_requests_reused':len(cases),'initial_points':sum(len(c['initial']['points']) for c in cases),'max_requests_per_case':3,'total_point_budget':32,'new_model_calls':0},flush=True)

def run():
    if (OUT/'completed.json').exists():raise FileExistsError('Preserve completed constraint repair')
    p=read(OUT/'frozen_plan.json');assert all(sha(f)==h for f,h in p['protected'].items());results=[];fresh=0;total_points=p['initial_point_annotations']
    for case in p['cases']:
        current=case['initial'];owners=dict(case['negative_source_neighbor_masks']);rounds=[];per_case_points=0;stop=''
        for round_id in range(1,4):
            a=np.load(current['raw_array_file'],allow_pickle=False);mask=a[f"{current['prediction_call']}_masks"][current['native_index']].squeeze();guard=evaluate(mask,current['points']);per_case_points+=len(current['points']);rounds.append({'round':round_id,'row':current,'native_guard':guard,'exact_initial_capture_reuse':round_id==1});save(OUT/'progress.json',{'finished_cases':results,'current_case':case['initial']['index'],'rounds':rounds,'actual_new_SAM_requests':fresh,'total_point_annotations':total_points})
            if guard['valid']:stop='native_constraints_passed_not_instance_acceptance';break
            if round_id==3:stop='original_three_request_limit';break
            if any(q['label']==1 and not q['valid'] for q in guard['points']):stop='positive_violation_not_repaired_by_negative_points';break
            x1,y1,x2,y2=current['box'];candidates=[];positives=[q for q in current['points'] if q['label']==1]
            for violation in guard['points']:
                if violation['valid'] or violation['label']!=0:continue
                old=current['points'][violation['index']];owner=owners.get(f"{old['x']},{old['y']}")
                if owner is None:continue
                neighbor=np.load(owner,allow_pickle=False)[y1:y2,x1:x2].astype('uint8');core=cv2.erode(neighbor,np.ones((7,7),np.uint8),borderType=cv2.BORDER_CONSTANT,borderValue=0)!=0;oy,ox=old['y'],old['x']
                for yy in range(max(0,oy-3),min(mask.shape[0],oy+4)):
                    for xx in range(max(0,ox-3),min(mask.shape[1],ox+4)):
                        if not mask[yy,xx] or not core[yy,xx] or any(max(abs(xx-q['x']),abs(yy-q['y']))<=3 for q in positives):continue
                        if any(q['x']==xx and q['y']==yy for q in current['points']):continue
                        priority=min((xx-q['x'])**2+(yy-q['y'])**2 for q in current['points'] if q['label']==0);candidates.append((priority,yy,xx,owner))
            additions=[]
            for _,yy,xx,owner in sorted(candidates,key=lambda z:(-z[0],z[1],z[2])):
                if any(max(abs(xx-q['x']),abs(yy-q['y']))<=3 for q in additions):continue
                if len(additions)==3:break
                if total_points+len(current['points'])+len(additions)+1>32 or per_case_points+len(current['points'])+len(additions)+1>32:continue
                additions.append({'x':xx,'y':yy,'label':0});owners[f'{xx},{yy}']=owner
            if not additions:stop='no_source_supported_counterexample_within_original_budget';break
            points=current['points']+additions;fields=['index','case','box','input','source_image','source_point','teacher_key','source_parent_uid','source_parent_cue','parent_rank','arm'];row={k:current[k] for k in fields};row.update(points=points,exact_previous_capture=None,repair_round=round_id+1,original_source_case=case['initial']['index'],new_model_derived_negative_points=additions,not_reference_or_budget_mask_generation=True);row['sam_request_sha256']=sam.key(row);folder=OUT/f"case{row['index']:02d}_round{round_id+1}";file=folder/'generation_completed.json'
            if not file.exists():
                folder.mkdir(exist_ok=True);save(folder/'frozen_plan.json',{'rows':[row],'protected':{str(OUT/'frozen_plan.json'):sha(OUT/'frozen_plan.json'),str(Path(__file__)):sha(__file__),str(current['raw_array_file']):sha(current['raw_array_file']),str(Path(sam.__file__)):sha(sam.__file__)},'source_sha256':sam.SOURCE_SHA,'max_new_calls':1,'same_original_positive_and_all_existing_negatives':True,'no_mask_editing':True});sam.OUT=folder;sam.run()
            generated=read(file);assert len(generated['rows'])==1;current=generated['rows'][0];fresh+=generated['actual_new_SAM_calls'];total_points+=len(points)
        results.append({'source_case':case['initial']['index'],'rounds':rounds,'stop_reason':stop,'per_case_point_annotations':per_case_points,'native_constraints_passed':rounds[-1]['native_guard']['valid'],'no_instance_or_Agent_convergence_claim':True})
    assert total_points<=32 and all(len(c['rounds'])<=3 and c['per_case_point_annotations']<=32 for c in results);save(OUT/'completed.json',{'status':'complete_bounded_model_counterexample_repair','cases':results,'actual_new_SAM_requests':fresh,'exact_initial_requests_reused':len(results),'total_point_annotations':total_points,'native_constraint_cases_passed':sum(c['native_constraints_passed'] for c in results),'cases_count':len(results),'new_VLM_calls':0,'new_Agent_runs':0,'all_full_native_masks_preserved':True,'not_adopted':True,'not_full_Agent_or_instance_acceptance':True});print({'new_SAM_requests':fresh,'native_constraint_passed':sum(c['native_constraints_passed'] for c in results),'cases':len(results),'total_points':total_points,'new_Agent_runs':0},flush=True)

if __name__=='__main__':prepare() if sys.argv[1]=='prepare' else run()
