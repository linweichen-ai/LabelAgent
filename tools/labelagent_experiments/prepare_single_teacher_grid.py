"""Source-only broader registered-model proposals; acceptance gates unchanged.

The completed pair-consensus arm remains the control. This additional offline
proposal arm tests up to three original accepted PAL segments per registered
model/view without requiring a second model to agree before neutral review.
It changes proposal ranking/consensus jointly, not final acceptance thresholds.
"""
import datetime,hashlib,sys
from pathlib import Path
import prepare_fixed_grid_followup as parent
from prepare_fixed_grid_followup import BASE,ROOT
import cv2,numpy as np
from PIL import Image
from run_rgb_layers import read,save
from deliver_context_graph import sha
from m2f_query_replay import replay,probability
OUT=BASE/'labelagent_single_teacher_grid_20261108'

def prepare():
    if OUT.exists():raise FileExistsError('Preserve new offline proposal protocol')
    baseline=parent.OUT;grid=BASE/'labelagent_registered_fixed_grid_20261108';p=read(grid/'frozen_plan.json');g=read(grid/'generation_completed.json');assert read(baseline/'all48_original_postprocess_readback.json')['passed'] and g['new_CPU_model_forwards']==24;OUT.mkdir();parent.OUT=OUT;protected={str(Path(__file__)):sha(__file__),str(grid/'frozen_plan.json'):sha(grid/'frozen_plan.json'),str(grid/'generation_completed.json'):sha(grid/'generation_completed.json'),str(baseline/'all48_original_postprocess_readback.json'):sha(baseline/'all48_original_postprocess_readback.json'),str(Path(parent.__file__)):sha(parent.__file__)};candidates=[];vetoes=[]
    for r in g['rows']:
        if r['is_synthetic_control']:continue
        f=Path(r['folder']);assert all(sha(f/n)==h for n,h in r['artifacts'].items());photo=Image.open(r['input']).convert('RGB');a=np.load(f/'raw_queries.npz',allow_pickle=False);ids,segs,_,winner=replay(a['class_logits'],a['mask_logits'],(photo.height,photo.width),p['postprocessing']);assert segs==r['segments_info'];protected[r['input']]=sha(r['input']);protected[r['original_source_image']]=sha(r['original_source_image']);protected[str(f/'raw_queries.npz')]=r['artifacts']['raw_queries.npz'];pal=sorted([s for s in segs if s['category_id']==1],key=lambda s:(-s['score'],s['id']))[:3]
        for rank,s in enumerate(pal,1):
            selected=ids==s['id'];q=np.unique(winner[selected]);assert len(q)==1;q=int(q[0]);native=probability(a['mask_logits'],q,(photo.height,photo.width))>=.5;border=lambda m:bool(m[0].any() or m[-1].any() or m[:,0].any() or m[:,-1].any())
            if border(native) or border(selected):vetoes.append({'model':r['model'],'view':r['teacher_key'],'rank':rank,'reason':'unchanged_native_or_stock_input_border'});continue
            distance=cv2.distanceTransform(np.pad(selected.astype('uint8'),1),cv2.DIST_L2,5)[1:-1,1:-1];x1,y1,x2,y2=r['observed_region_in_input'];observed=np.zeros(selected.shape,bool);observed[y1:y2,x1:x2]=True;ys=np.where(selected)[0];lower=int(ys.min())+2*(int(ys.max())-int(ys.min())+1)//3
            for policy in ['maximum_distance','lowest_third_maximum_distance']:
                scores=distance.copy();scores[~observed]=0
                if policy.startswith('lowest'):scores[:lower]=0
                if scores.max()<=0:continue
                y,x=map(int,np.unravel_index(scores.argmax(),scores.shape));ax=x+r['source_virtual_box'][0];ay=y+r['source_virtual_box'][1];assert 0<=ax<1920 and 0<=ay<1536;candidates.append({'cohort':'single_teacher_registered_proposal_only','model':r['model'],'case':r['case'],'teacher_key':r['teacher_key'],'source_view':r['source_view'],'source_image':r['original_source_image'],'padded_input':r['input'],'source_virtual_box':r['source_virtual_box'],'observed_region_in_input':r['observed_region_in_input'],'input_point':[x,y],'source_point':[ax,ay],'policy':policy,'stock_segment':s,'query':q,'source_rank':rank,'teacher_stock_PAL_verified':True,'both_model_consensus_required_for_this_proposal_arm':False,'not_a_final_mask_or_truth':True})
    order={'whole':0,'q1':1,'q2':2,'q3':3,'q4':4};candidates.sort(key=lambda r:(r['source_rank'],order[r['source_view']],r['policy'],r['case'],r['model']));rows=[];seen={}
    for r in candidates:
        k=(r['case'],*r['source_point'])
        if k in seen:rows[seen[k]]['logical_sources'].append(r);continue
        if len(rows)>=32:vetoes.append(r|{'reason':'32_unique_source_point_limit'});continue
        index=len(rows)+1;photo=Image.open(r['source_image']).convert('RGB');views=parent.caption_views(photo,r['source_point'],OUT/'inputs'/f'cue{index:02d}');seen[k]=len(rows);rows.append(r|views|{'index':index,'logical_sources':[r],'source_partition_index':index})
    controls=[]
    for name in ['gray','white']:
        file=ROOT/'outputs/labelagent_semantic_recovery_20261105/neutral_controls'/(name+'.png');photo=Image.open(file).convert('RGB');protected[str(file)]=sha(file);views=parent.caption_views(photo,[photo.width//2,photo.height//2],OUT/'controls'/name);controls.append(views|{'index':'control_'+name,'known_no_real_object':True,'name':name})
    for r in rows+controls:
        protected[r['request_file']]=sha(r['request_file'])
        for b in r['bindings']:protected[b['image']]=b['sha256']
    save(OUT/'frozen_plan.json',{'at':datetime.datetime.now().astimezone().isoformat(),'rows':rows,'controls':controls,'protected':protected,'model_config_sha256':read(baseline/'frozen_plan.json')['model_config_sha256'],'parent_consensus_arm_unchanged':True,'changed_proposal_rules':['up_to3_original_accepted_PAL_segments_per_registered_model_view','do_not_require_second_model_PAL_at_point_before_neutral_review'],'joint_proposal_change_not_single_factor_causal_claim':True,'selection_reads_only_original_predictions_and_RGB_not_reference_coordinates':True,'same_original_native_border_and_all_final_gates':True,'final_class_semantics_single_base_extent_area_purity_visibility_matching_all_still_required':True,'max_unique_real_point_reviews':32,'max_unique_SAM_point_candidates':16,'max_paired_SAM_calls':32,'Agent_round_limit':3,'new_Agent_runs':0,'adopted':False,'vetoes':vetoes,'protocol_informed_by_previous_diagnostics_not_independent_validation':True});save(OUT/'prepared.json',{'status':'prepared_not_executed','source_points':len(rows),'controls':2,'new_model_calls':0,'new_training':False,'original_annotation_writes':0});print(read(OUT/'prepared.json'),flush=True)

if __name__=='__main__':prepare()
