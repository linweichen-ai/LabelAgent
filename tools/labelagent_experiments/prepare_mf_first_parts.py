"""MF -> SAM -> whole-instance Agent review, preserving all final gates.

Earlier per-point captions are recorded faithfully as uncertain/failed source
evidence. They are NOT fabricated as PAL. This new proposal-stage control does
not veto a trained stock-PAL proposal before SAM supplies its object geometry.
"""
import base64,datetime,hashlib,shutil
from pathlib import Path
from prepare_palette_parts import BASE
from prepare_balanced_palette_parts import OUT as PARENT
from run_rgb_layers import read,save
from deliver_context_graph import sha
OUT=BASE/'labelagent_MF_first_parts_20261108'

def main():
    if OUT.exists():raise FileExistsError('Preserve MF-first proposal protocol')
    if shutil.disk_usage(BASE).free<192*2**20:raise RuntimeError('Original192MiB preparation reserve')
    p=read(PARENT/'frozen_plan.json');rv=read(PARENT/'review_completed.json');grid=BASE/'labelagent_registered_fixed_grid_20261108';g=read(grid/'generation_completed.json');groups={};protected={str(PARENT/'frozen_plan.json'):sha(PARENT/'frozen_plan.json'),str(PARENT/'review_completed.json'):sha(PARENT/'review_completed.json'),str(grid/'generation_completed.json'):sha(grid/'generation_completed.json'),str(Path(__file__)):sha(__file__)}
    for r in p['rows']:
        for logical in r['logical_sources']:
            key=(logical['case'],logical['model'],logical['teacher_key'],logical['stock_segment']['id']);group=groups.setdefault(key,[])
            if not any(x['source_point']==logical['source_point'] for x in group):group.append(logical|{'source_review_index':r['index']})
    order={'q1':0,'q2':1,'q3':2,'q4':3,'whole':4};ordered=sorted(groups.items(),key=lambda pair:(order[pair[1][0]['source_view']],pair[0][0],pair[1][0]['source_rank'],pair[0][1]));candidates=[];seen=set()
    for level in [0,1]:
        for _,group in ordered:
            group=sorted(group,key=lambda r:(r['source_point'][1],r['source_point'][0]))
            if level>=len(group):continue
            r=group[level];key=(r['case'],*r['source_point'])
            if key in seen or len(candidates)>=16:continue
            seen.add(key);source_review=next(x for x in rv['rows'] if x['index']==r['source_review_index']);candidates.append(r|{'index':len(candidates)+1,'original_neutral_point_PAL_evidence':source_review['PAL_subject_evidence'],'original_neutral_point_decision':source_review['decision'],'original_neutral_response':source_review['response_file'],'original_neutral_response_sha256':source_review['response_sha256'],'neutral_caption_is_not_a_final_mask_decision':True})
    OUT.mkdir();requests=[]
    for r in candidates:
        teacher=next(t for t in g['rows'] if t['model']==r['model'] and t['teacher_key']==r['teacher_key']);assert r['teacher_stock_PAL_verified'];ax,ay=r['source_point']
        for arm in ['original_source','explicit_padded_view']:
            if arm=='original_source':image=r['source_image'];point=[ax,ay];virtual=[0,0,1920,1536];observed=[0,0,1920,1536]
            else:image=teacher['input'];virtual=teacher['source_virtual_box'];point=[ax-virtual[0],ay-virtual[1]];observed=teacher['observed_region_in_input']
            points=[{'x':point[0],'y':point[1],'label':1}];body={'image':base64.b64encode(Path(image).read_bytes()).decode(),'points':points,'include_raw_capture':True};key=hashlib.sha256(__import__('json').dumps(body,sort_keys=True).encode()).hexdigest();file=OUT/'prepared_requests'/(key+'_PRIVATE.json');save(file,body);requests.append(r|{'arm':arm,'input':image,'input_sha256':sha(image),'points':points,'virtual_box':virtual,'observed_rectangle':observed,'request_file':str(file),'request_sha256':key});protected[image]=sha(image);protected[str(file)]=sha(file);protected[r['original_neutral_response']]=r['original_neutral_response_sha256']
    assert len(requests)<=32;save(OUT/'frozen_plan.json',{'at':datetime.datetime.now().astimezone().isoformat(),'rows':requests,'source_points':candidates,'protected':protected,'proposal_source':'Original trained registered Mask2Former accepted stock PAL, unchanged class and native/stock boundary gates; RGB part positions pre-existing and frozen','all_previous_point_captions_kept_without_false_PAL_flag':True,'new_control_order':'MF stock proposal -> complete native SAM capture -> unchanged whole-instance semantic/area/person/purity/visibility/matching review','removed_early_point_caption_veto_is_proposal_order_only':True,'final_neutral_and_gray_white_single_base_extent_gates_all_required':True,'original13_identity_and_r0_3_6_protection_required_before_any_adoption':True,'Agent_round_limit':3,'Agent_point_limit':32,'max_source_points':16,'max_SAM_requests':32,'SAM_arms':['original_source','explicit_padded_view'],'mapping':'Save entire native bitmap; reject ANY foreground on synthetic padding; reject original native input-border and prompt violations; inverse-map ALL foreground with equal count, no clipping','no_NULL_or_non_PAL_query_promoted':True,'model_config_sha256':p['model_config_sha256'],'new_MF_calls':0,'new_Agent_runs':0,'new_training':False,'new_model_calls_at_preparation':0,'adopted':False});save(OUT/'prepared.json',{'status':'prepared_not_executed','source_points':len(candidates),'logical_SAM_requests':len(requests),'original_point_caption_passed':sum(r['original_neutral_point_PAL_evidence'] for r in candidates),'original_point_caption_failed_or_uncertain':sum(not r['original_neutral_point_PAL_evidence'] for r in candidates),'new_model_calls':0});print(read(OUT/'prepared.json'),flush=True)

if __name__=='__main__':main()
