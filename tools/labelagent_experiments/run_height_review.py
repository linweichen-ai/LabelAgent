"""Unchanged neutral reviewer; explicit single-source SAM proposal contract."""
import os,sys,shutil
from pathlib import Path
from run_rgb_layers import ROOT,read,save,sha
import run_base_material as reviewer
import run_trained_point_sam as sam
OUT=ROOT/'outputs/labelagent_height_review_20261108'

def initialize():
    file=OUT/'execution_plan.json';cache=OUT/'exact_cache_catalogue/completed.json'
    if not file.exists():
        sources=[Path(os.environ['LABELAGENT_PAIRED_CUE_OUTPUT'])/'completed.json']+[ROOT/'outputs'/n/f for n,f in [('labelagent_trained_point_review_20261108','completed.json'),('labelagent_base_material_20261108','review_completed.json'),('labelagent_base_blue_surface_20261108','review_completed.json'),('labelagent_zoom_cues_20261108','review_completed.json'),('labelagent_registry_zoom_pool_20261108','review_completed.json')]];rows=[]
        for p in sources:rows+=read(p)['rows']
        save(cache,{'rows':rows,'source_sha256':{str(p):sha(p) for p in sources}});save(file,{'driver_sha256':sha(__file__),'reviewer_sha256':sha(reviewer.__file__),'source_plan_sha256':sha(OUT/'frozen_plan.json'),'cache_sha256':sha(cache),'no_fabricated_stock_consensus':True})
    p=read(file);assert p['driver_sha256']==sha(__file__) and p['reviewer_sha256']==sha(reviewer.__file__) and p['source_plan_sha256']==sha(OUT/'frozen_plan.json') and p['cache_sha256']==sha(cache);reviewer.OUT=OUT;reviewer.PAIR=cache.parent

def prepare_SAM():
    target=OUT/'SAM'
    if target.exists():raise FileExistsError('Preserve prepared sensor SAM protocol')
    if shutil.disk_usage(OUT).free<96*2**20:raise RuntimeError('Original96MiB preparation reserve')
    p=read(OUT/'frozen_plan.json');d=read(OUT/'review_completed.json');rows=[];admissions=[]
    for r in p['rows']:
        rr=next(v for v in d['rows'] if v['partition']==r['index']);admissions.append({'cue':r['index'],'neutral_positive_PAL':rr['PAL_subject_evidence'],'unlabelled_height_is_not_confirmed_instance_identity':True})
        if rr['PAL_subject_evidence']:rows.append({**r,'sam_request_sha256':sam.key(r),'exact_previous_capture':None})
    protected={str(OUT/'frozen_plan.json'):sha(OUT/'frozen_plan.json'),str(OUT/'review_completed.json'):sha(OUT/'review_completed.json'),str(Path(__file__)):sha(__file__),str(Path(sam.__file__)):sha(sam.__file__)}
    w=read(ROOT/'outputs/labelagent_registry_zoom_pool_20261108/SAM_weights_identity.json');weight=Path(w['path']);assert weight.stat().st_size==w['bytes'] and weight.stat().st_mtime_ns==w['mtime_ns'];save(OUT/'weights_same_stat_binding.json',{'previous_streamed_weight_receipt':str(ROOT/'outputs/labelagent_registry_zoom_pool_20261108/SAM_weights_identity.json'),'previous_receipt_sha256':sha(ROOT/'outputs/labelagent_registry_zoom_pool_20261108/SAM_weights_identity.json'),'current_size_and_mtime_exact':True,'not_a_fresh_weight_hash':True});protected[str(OUT/'weights_same_stat_binding.json')]=sha(OUT/'weights_same_stat_binding.json')
    for r in rows:protected[r['input']]=sha(r['input'])
    target.mkdir();save(target/'frozen_plan.json',{'rows':rows,'source_admissions':admissions,'protected':protected,'source_sha256':sam.SOURCE_SHA,'max_new_calls':len(rows),'point_only_no_prior':True,'original_source_selected_index_only':True,'source_evidence':'MF source prior plus unlabelled sensor height component plus unchanged neutral positive RGB subject','negative_is_unconfirmed_height_component_hypothesis':True,'no_reference_or_area_for_inputs':True,'no_mask_clipping':True,'all_original_acceptance_gates_unchanged':True,'new_Agent_rounds':0,'Agent_round_limit':3});print({'neutral_accepted':len(rows),'new_SAM_calls':0},flush=True)

if __name__=='__main__':
    initialize()
    if sys.argv[1]=='review':reviewer.review()
    elif sys.argv[1]=='prepare_SAM':prepare_SAM()
    else:sam.OUT=OUT/'SAM';sam.run()
