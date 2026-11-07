"""Explicit MF evaluation adapters and genuine SAM pools, after source freeze."""
import hashlib,json
from pathlib import Path
import numpy as np
from PIL import Image
from run_rgb_layers import ROOT,read,save,sha
from native_prompt_guard import evaluate
import audit_soft_prior as audit
OUT=ROOT/'outputs/labelagent_aspect_context_cues_20261108'

def audit_rows(name,rows,provenance):
    folder=OUT/'audits'/name
    if folder.exists():raise FileExistsError('Preserve completed zoom comparison arm')
    save(folder/'generation_completed.json',{'status':'complete','rows':rows,'provenance':provenance,'not_mask_generation_or_selection_from_reference':True});audit.OUT=folder;audit.main();result=read(folder/'support_area_audit.json');return {'arm':name,'case_r3_matches':{c['case']:next(r for r in c['radii'] if r['radius']==3)['geometry_area_person_valid_matches'] for c in result['cases']},'candidates':len(result['candidates']),'valid_area_person_masks':sum(r['valid'] and r['area_fits'] and not r['person_conflict'] for r in result['candidates'])}

def main():
    if (OUT/'audit_summary.json').exists():raise FileExistsError('Preserve zoom audit')
    plan=read(OUT/'frozen_plan.json');assert all(sha(f)==h for f,h in plan['protected'].items());review=read(OUT/'review_completed.json');generated=read(OUT/'SAM/generation_completed.json');pool_plan=read(OUT/'native_pool_source_plan.json');assert sha(OUT/'SAM/frozen_plan.json')==pool_plan['SAM_frozen_plan_sha256'];arms=[];mf_by_arm={};pool=[];default=[]
    for cue in plan['rows']:
        neutral=next(r['PAL_subject_evidence'] for r in review['rows'] if r['partition']==cue['index']);x1,y1,x2,y2=cue['box'];native=np.load(cue['MF_native_full_mask'],allow_pickle=False);assert hashlib.sha256(native.tobytes()).hexdigest()==cue['MF_native_mask_sha256'];stock=np.asarray(Image.open(cue['MF_stock_mask']))!=0
        for mode,local in [('native',native[y1:y2,x1:x2]),('stock',stock.astype('uint8'))]:
            folder=OUT/'MF_evaluation_adapters'/f"{cue['model']}_cue{cue['index']:02d}_{mode}";folder.mkdir(parents=True,exist_ok=True);full=np.zeros((1536,1920),np.uint8);full[y1:y2,x1:x2]=local;assert full.sum()==local.sum();np.save(folder/'complete_full_mask.npy',full);prediction='MF_native_probability' if mode=='native' else 'MF_original_stock';arrayfile=folder/'evaluation_bitmap_adapter.npz';np.savez_compressed(arrayfile,**{prediction+'_masks':local[None]});meta={'source_type':prediction,'evaluation_adapter_only_not_SAM_capture':True,'original_MF_query':cue['query'],'checkpoint_sha256':cue['model_sha256'],'original_native_mask_source':cue['MF_native_full_mask'],'original_stock_mask_source':cue['MF_stock_mask'],'no_threshold_or_pixels_changed_by_evaluation_adapter':True};save(folder/'metadata.json',meta);key=f"MF_{cue['model']}_cue{cue['index']:02d}_{mode}";row={**cue,'key':key,'kind':'real','source_type':prediction,'folder':str(folder),'raw_array_file':str(arrayfile),'metadata_file':str(folder/'metadata.json'),'prediction_call':prediction,'native_index':0,'mask_sha256':hashlib.sha256(full.tobytes()).hexdigest(),'raw_pixels':int(full.sum()),'valid':bool(neutral),'neutral_source_point_PAL':neutral,'not_SAM_source_selection':True};mf_by_arm.setdefault(cue['model']+'_'+mode,[]).append(row)
    for row in generated['rows']:
        arrays=np.load(row['raw_array_file'],allow_pickle=False);meta=read(row['metadata_file']);x1,y1,x2,y2=row['box'];source_hash=row['mask_sha256']
        for call,prediction in enumerate(meta['predictions']):
            masks=arrays[f'{call}_masks'];assert hashlib.sha256(masks.tobytes()).hexdigest()==prediction['masks']['sha256']
            for index,native in enumerate(masks):
                local=np.asarray(native).squeeze();assert np.isin(local,[0,1]).all();local=(local!=0).astype('uint8');folder=OUT/'native_pool'/f"cue{row['index']:02d}_call{call}_index{index}";folder.mkdir(parents=True,exist_ok=True);full=np.zeros((1536,1920),np.uint8);full[y1:y2,x1:x2]=local;assert full.sum()==local.sum();np.save(folder/'complete_full_mask.npy',full);Image.fromarray(local*255).save(folder/'raw.png');guard=evaluate(local,row['points']);border=bool(local[0].any() or local[-1].any() or local[:,0].any() or local[:,-1].any());selected=call==meta['selected_prediction_call'] and index==meta['selected_index'];r={**row,'key':row['sam_request_sha256']+f'_{call}_{index}','kind':'real','folder':str(folder),'raw_array_file':row['raw_array_file'],'metadata_file':row['metadata_file'],'prediction_call':call,'native_index':index,'mask_sha256':hashlib.sha256(full.tobytes()).hexdigest(),'raw_pixels':int(full.sum()),'source_selected':selected,'native_prompt_guard':guard,'touches_border':border,'valid':bool(guard['valid'] and not border),'native_score':float(arrays[f'{call}_scores'][index]),'pool_hypothesis_not_original_selected_output':not selected};pool.append(r)
                if selected:assert r['mask_sha256']==source_hash;default.append(r)
    save(OUT/'native_pool_frozen_outputs.json',{'rows':pool,'source_plan_sha256':sha(OUT/'native_pool_source_plan.json'),'all_native_masks_retained_before_reference_evaluation':True,'no_cherry_pick_by_support_or_area':True,'no_adoption_or_human_truth':True});assert len(default)==len(generated['rows']) and len(pool)==3*len(default)
    for name,rows in mf_by_arm.items():arms.append(audit_rows('MF_'+name,rows,{'source':'MF_native_or_original_stock_output','no_new_model_forward':True,'separate_checkpoint_and_output_type_arms':True}))
    arms.append(audit_rows('SAM_source_default',default,{'source':'original service selected index','actual_unique_new_SAM_calls':generated['actual_new_SAM_calls']}))
    # Each alternative arm contains at most one output per original request.
    # Never count all three alternatives simultaneously as three annotations.
    for index in [0,1,2]:
        rows=[r for r in pool if r['native_index']==index];assert len(rows)==len(default);arms.append(audit_rows(f'SAM_fixed_native_index{index}',rows,{'source':'fixed alternative index across same three requests','original_source_selected_flags_retained':True,'not_a_reference_selected_mask_or_deployed_change':True}))
    save(OUT/'audit_summary.json',{'status':'complete_post_freeze_MF_and_SAM_pool_comparison','arms':arms,'actual_MF_CPU_forwards':12,'actual_unique_new_SAM_calls':generated['actual_new_SAM_calls'],'new_model_calls_during_audit':0,'original_source_default_distinct_from_alternatives':True,'no_unselected_candidate_adopted_by_reference_score':True,'adopted':False});print(json.dumps({'arms':arms,'new_model_calls':0},ensure_ascii=False),flush=True)

if __name__=='__main__':main()
