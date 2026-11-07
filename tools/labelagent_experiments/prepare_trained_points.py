"""CPU replay of original stock outputs to prepare image-only SAM point cues."""
import datetime,json,os,shutil,sys
from pathlib import Path
import numpy as np
from PIL import Image
from run_rgb_layers import ROOT,read,save,sha

OUT=Path(os.environ.get('LABELAGENT_TRAINED_POINTS_OUTPUT',str(ROOT/'outputs/labelagent_trained_points_20261108')))
OLD=ROOT/'outputs/labelagent_hybrid_repair_20261104/query_diagnosis';LATER=ROOT/'outputs/labelagent_retained_teacher_20261107/query_diagnosis';PARENT=ROOT/'outputs/labelagent_soft_prior_20261108'
PACK=ROOT/'outputs/mask2former_single_recovery_20261104/最新单模型候选包'


def freeze():
    if OUT.exists():raise FileExistsError('Preserve model-derived point study')
    if shutil.disk_usage(OUT.parent).free<256*2**20:raise RuntimeError('Output space guard')
    parent=read(PARENT/'frozen_plan.json');teachers=[];seen=set();protected={str(PARENT/'frozen_plan.json'):sha(PARENT/'frozen_plan.json'),str(PACK/'inference_config.json'):sha(PACK/'inference_config.json'),str(Path(__file__)):sha(__file__),str(Path(__file__).with_name('trained_point_policy.py')):sha(Path(__file__).with_name('trained_point_policy.py')),str(Path(__file__).with_name('m2f_query_replay.py')):sha(Path(__file__).with_name('m2f_query_replay.py'))}
    for p in parent['proposals']:
        if p['kind']!='real' or p['teacher_key'] in seen:continue
        seen.add(p['teacher_key']);teachers.append({k:p[k] for k in ['teacher_key','case','source_view','query','box','input','prior','logits']})
        for key in ['input','prior','logits']:protected[p[key]]=sha(p[key])
        for base in [OLD,LATER]:
            folder=base/p['case']/p['source_view']
            for name in ['raw_queries.npz','panoptic.png','diagnosis.json']:protected[str(folder/name)]=sha(folder/name)
    parameters=read(PACK/'inference_config.json')['postprocessing'];assert parameters['promote_null_pallet'] is False;OUT.mkdir();save(OUT/'frozen_plan.json',{'at':datetime.datetime.now().astimezone().isoformat(),'teachers':teachers,'protected':protected,'stock_postprocess_parameters':parameters,'old_teacher_sha256':parent['teacher_checkpoint_sha256'],'later_teacher_sha256':read(ROOT/'outputs/labelagent_retained_teacher_20261107/frozen_plan.json')['candidate_sha256'],'max_two_point_cues_per_parent':True,'no_new_Mask2Former_forwards_or_training':True,'no_stock_threshold_change_or_null_promotion':True,'no_posthoc_reference_selected_diagnostic_masks_used':True,'both_original_stock_class_maps_for_point_filter_only':True,'all_final_output_masks_must_remain_complete':True,'no_reference_points_labels_pose_depth_calibration_or_area_inputs':True,'Agent_round_limit':3,'new_Agent_runs':0,'prepared_only_no_SAM_or_VLM_calls':True});print({'frozen_teachers':len(teachers),'cached_views_each_model':len({(p['case'],p['source_view']) for p in teachers}),'new_model_forwards':0},flush=True)


def prepare():
    if (OUT/'prepared.json').exists():raise FileExistsError('Point cues already prepared')
    import psutil
    if psutil.virtual_memory().available<1500*2**20:raise RuntimeError('1.5GiB available CPU RAM guard; no replay started')
    import torch
    from m2f_query_replay import replay
    from trained_point_policy import points
    torch.set_num_threads(2);plan=read(OUT/'frozen_plan.json');assert all(sha(p)==h for p,h in plan['protected'].items());maps={};checks=[];proposals=[];diagnostics=[]
    for teacher in plan['teachers']:
        key=(teacher['case'],teacher['source_view']);meta=read(OLD/key[0]/key[1]/'diagnosis.json');vx1,vy1,vx2,vy2=meta['crop'];size=(vy2-vy1,vx2-vx1)
        if key not in maps:
            both=[]
            for tag,base in [('old',OLD),('later',LATER)]:
                folder=base/key[0]/key[1];a=np.load(folder/'raw_queries.npz',allow_pickle=False);ids,segs,queries,_=replay(a['class_logits'],a['mask_logits'],size,plan['stock_postprocess_parameters']);rgb=np.asarray(Image.open(folder/'panoptic.png')).astype('uint32');saved=rgb[:,:,0]+256*rgb[:,:,1]+65536*rgb[:,:,2];np.testing.assert_array_equal(ids,saved);pal=np.isin(ids,[s['id'] for s in segs if s['category_id']==1]);both.append(pal);checks.append({'model':tag,'case':key[0],'view':key[1],'stock_panoptic_exact':True,'class_map_from_raw_queries_only':True})
            maps[key]=both
        x1,y1,x2,y2=teacher['box'];ys=slice(y1-vy1,y2-vy1);xs=slice(x1-vx1,x2-vx1);logits=np.load(teacher['logits'],allow_pickle=False);parent=np.asarray(Image.open(teacher['prior']).convert('L'))>127;rows,diag=points(logits,parent,maps[key][0][ys,xs],maps[key][1][ys,xs]);diagnostics.append({'teacher':teacher['teacher_key'],**diag})
        for row in rows:proposals.append({**teacher,**row,'source_point':[row['points'][0]['x']+x1,row['points'][0]['y']+y1],'no_mask_prior_for_later_SAM':True,'new_SAM_call_executed':False})
    assert all(sha(p)==h for p,h in plan['protected'].items());save(OUT/'prepared.json',{'status':'CPU_only_trained_point_cues_prepared','rows':proposals,'diagnostics':diagnostics,'stock_replay_checks':checks,'point_cues':len(proposals),'new_Mask2Former_forwards':0,'new_SAM_calls':0,'new_VLM_calls':0,'new_masks':0,'reference_inputs':False,'not_annotation_or_instance_acceptance':True});print({'prepared_point_cues':len(proposals),'exact_stock_replays':len(checks),'new_model_forwards':0,'new_SAM_or_VLM_calls':0},flush=True)


if __name__=='__main__':freeze() if sys.argv[1]=='freeze' else prepare()
