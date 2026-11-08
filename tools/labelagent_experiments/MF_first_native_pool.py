"""Freeze all original SAM fixed-index hypotheses before reference scoring.

Source-selected outputs remain the primary control. Alternative index arms
are separate proposal diagnostics, never three simultaneous annotations.
"""
import hashlib,shutil,sys
from pathlib import Path
from prepare_mf_first_parts import OUT
from prepare_palette_parts import np
from run_rgb_layers import read,save
from deliver_context_graph import sha
from native_review_feedback import describe_native,SOURCE_SHA
POOL=OUT/'native_pool'

def main():
    resume=len(sys.argv)>1 and sys.argv[1]=='resume'
    if (POOL/'frozen_outputs.json').exists() or (POOL.exists() and not resume):raise FileExistsError('Preserve native pool; explicit verified partial resume only')
    if (OUT/'audits').exists():raise RuntimeError('Pool must be frozen before reference scoring')
    p=read(OUT/'frozen_plan.json');execution=read(OUT/'execution_plan.json');g=read(OUT/'generation_completed.json');assert execution['alternative_diagnostic_arms_predeclared']==[0,1,2];assert all(sha(f)==h for f,h in p['protected'].items());protected={str(OUT/'generation_completed.json'):sha(OUT/'generation_completed.json'),str(OUT/'execution_plan.json'):sha(OUT/'execution_plan.json'),str(Path(__file__)):sha(__file__)};existing={};rows=[]
    for r in g['rows']:
        file=Path(r['folder'])/'complete_full_mask.npy'
        if file.exists():existing[(r['case'],r['full_mask_sha256'])]=file
    for r in g['rows']:
        meta=read(r['metadata_file']);assert meta['source_sha256']==SOURCE_SHA;arrays=np.load(r['raw_array_file'],allow_pickle=False);call=meta['selected_prediction_call'];masks=arrays[f'{call}_masks'];assert hashlib.sha256(masks.tobytes()).hexdigest()==meta['predictions'][call]['masks']['sha256'];protected[r['raw_array_file']]=sha(r['raw_array_file']);protected[r['metadata_file']]=sha(r['metadata_file'])
        for index,raw in enumerate(masks):
            native=(raw.squeeze()!=0).astype('uint8');feedback=describe_native(native,r['points']);x1,y1,x2,y2=r['observed_rectangle'];outside=native.copy();outside[y1:y2,x1:x2]=0;pad=int(outside.sum());key=f'p{len(rows)+1:03d}';primary=index==meta['selected_index'];valid=feedback['native_prompt_passed'] and pad==0
            if primary:assert valid==r['valid']
            row=r|{'key':key,'prediction_call':call,'native_index':index,'source_selected':primary,'source_default_selected_index':meta['selected_index'],'native_hypothesis_not_API_selected_output':not primary,'native_score':float(arrays[f'{call}_scores'][index]),'native_pixels':int(native.sum()),'native_value_sha256':hashlib.sha256(native.tobytes()).hexdigest(),'native_prompt_guard':feedback['native_prompt_guard'],'touches_input_border':feedback['touches_input_border'],'foreground_on_synthetic_padding':pad,'valid':valid,'original_capture_folder':r['folder'],'original_request_sha256':r['request_sha256']}
            # No full-frame bitmap is manufactured for a padding violation.
            if valid:
                vx,vy,_,_=r['virtual_box'];ax,ay,bx,by=x1+vx,y1+vy,x2+vx,y2+vy;assert 0<=ax<bx<=1920 and 0<=ay<by<=1536;full=np.zeros((1536,1920),'uint8');full[ay:by,ax:bx]=native[y1:y2,x1:x2];assert int(full.sum())==int(native.sum());digest=hashlib.sha256(full.tobytes()).hexdigest();folder=POOL/'candidates'/key;folder.mkdir(parents=True,exist_ok=resume);file=folder/'complete_full_mask.npy';previous=existing.get((r['case'],digest))
                if file.exists():np.testing.assert_array_equal(full,np.load(file,allow_pickle=False))
                else:
                    if shutil.disk_usage(OUT).free<48*2**20:raise RuntimeError('Original48MiB bitmap output reserve')
                    if previous:shutil.copy2(previous,file);np.testing.assert_array_equal(full,np.load(file,allow_pickle=False))
                    else:np.save(file,full)
                existing[(r['case'],digest)]=file
                row.update(folder=str(folder),full_mask_sha256=digest,all_native_foreground_preserved_in_source=True,storage_exact_independent_bitmap_copy=str(previous) if previous else None);protected[str(file)]=sha(file)
            rows.append(row)
    assert len(rows)==96;save(POOL/'frozen_outputs.json',{'status':'complete_source_frozen_native_index_hypotheses','rows':rows,'protected':protected,'predeclared_alternative_indices':[0,1,2],'all_source_native_valid_selected_before_reference_scoring':True,'reference_inputs_read':False,'no_selection_by_area_purity_or_support':True,'primary_source_selected_control_kept':True,'never_mix_three_alternatives_from_same_request_as_three_annotations':True,'model_config_sha256':p['model_config_sha256'],'no_new_model_calls':True,'new_Agent_runs':0,'adopted':False});print({'raw_native_hypotheses':len(rows),'all_native_valid':sum(r['valid'] for r in rows),'valid_nondefault':sum(r['valid'] and not r['source_selected'] for r in rows),'padding_failures':sum(r['foreground_on_synthetic_padding']>0 for r in rows),'new_model_calls':0},flush=True)

if __name__=='__main__':main()
