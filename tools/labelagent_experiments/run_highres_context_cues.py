"""Typed single-teacher proposals, exact neutral cache and genuine SAM reuse."""
import datetime,hashlib,os,shutil,sys
from pathlib import Path
from run_rgb_layers import ROOT,read,save,sha
import run_base_material as reviewer
import run_trained_point_sam as sam
OUT=Path('D:/AI/Codex/Home/visualizations/2026/11/04/01a1a56c-0d6f-7d61-8326-d3bbd344d2df/labelagent_highres_context_cues_20261108')
WEIGHTS=Path('D:/Apps/LabelAgent/source/sam3/weights/sam3.pt')

def initialize():
    file=OUT/'execution_plan.json';cache=OUT/'exact_cache_catalogue/completed.json'
    if not file.exists():
        sources=[Path(os.environ['LABELAGENT_PAIRED_CUE_OUTPUT'])/'completed.json',ROOT/'outputs/labelagent_trained_point_review_20261108/completed.json',ROOT/'outputs/labelagent_base_material_20261108/review_completed.json',ROOT/'outputs/labelagent_base_blue_surface_20261108/review_completed.json',ROOT/'outputs/labelagent_zoom_cues_20261108/review_completed.json',ROOT/'outputs/labelagent_registry_zoom_pool_20261108/review_completed.json',ROOT/'outputs/labelagent_height_review_20261108/review_completed.json',ROOT/'outputs/labelagent_aspect_context_cues_20261108/review_completed.json'];rows=[]
        for p in sources:rows+=read(p)['rows']
        save(cache,{'rows':rows,'source_files_sha256':{str(p):sha(p) for p in sources}});save(file,{'driver_sha256':sha(__file__),'reviewer_sha256':sha(reviewer.__file__),'sam_runner_sha256':sha(sam.__file__),'source_plan_sha256':sha(OUT/'frozen_plan.json'),'cache_catalogue_sha256':sha(cache),'SAM_admission_type':'one calibrated source teacher PAL plus unchanged neutral subject guard, without fabricated two-teacher consensus metadata','deduplicate_exact_source_image_and_points_before_network':True,'original_acceptance_gates_not_changed':True})
    p=read(file);assert p['driver_sha256']==sha(__file__) and p['reviewer_sha256']==sha(reviewer.__file__) and p['sam_runner_sha256']==sha(sam.__file__) and p['source_plan_sha256']==sha(OUT/'frozen_plan.json') and p['cache_catalogue_sha256']==sha(cache);reviewer.OUT=OUT;reviewer.PAIR=cache.parent

def prepare_SAM():
    target=OUT/'SAM'
    if target.exists():raise FileExistsError('Preserve prepared source pool SAM')
    if shutil.disk_usage(OUT).free<96*2**20:raise RuntimeError('Original raw output reserve')
    plan=read(OUT/'frozen_plan.json');review=read(OUT/'review_completed.json');assert all(sha(f)==h for f,h in plan['protected'].items());unique={};admissions=[]
    for cue in plan['rows']:
        decision=next(r for r in review['rows'] if r['partition']==cue['index']);accepted=decision['PAL_subject_evidence'];admissions.append({'cue':cue['index'],'model':cue['model'],'PAL_subject_evidence':accepted})
        if not accepted:continue
        assert cue['stock_segment']['category_id']==1 and cue['stock_segment']['score']>.2;k=sam.key(cue)
        if k not in unique:unique[k]={**cue,'sam_request_sha256':k,'exact_previous_capture':None,'logical_source_cues_sharing_exact_request':[]}
        unique[k]['logical_source_cues_sharing_exact_request'].append(cue['index'])
    # Hash weights in bounded chunks; never load a3.4GiB byte buffer merely to hash.
    h=hashlib.sha256()
    with WEIGHTS.open('rb') as f:
        for b in iter(lambda:f.read(4*2**20),b''):h.update(b)
    assert h.hexdigest()=='9999e2341ceef5e136daa386eecb55cb414446a00ac2b55eb2dfd2f7c3cf8c9e'
    weight_receipt=OUT/'SAM_weights_identity.json';save(weight_receipt,{'path':str(WEIGHTS),'bytes':WEIGHTS.stat().st_size,'mtime_ns':WEIGHTS.stat().st_mtime_ns,'sha256':h.hexdigest(),'stream_hash_chunk_bytes':4*2**20})
    protected={str(OUT/'frozen_plan.json'):sha(OUT/'frozen_plan.json'),str(OUT/'review_completed.json'):sha(OUT/'review_completed.json'),str(OUT/'execution_plan.json'):sha(OUT/'execution_plan.json'),str(Path(__file__)):sha(__file__),str(Path(sam.__file__)):sha(sam.__file__),str(weight_receipt):sha(weight_receipt)};history=[]
    for pattern in ['labelagent_*/generation_completed.json','labelagent_*/SAM/generation_completed.json']:
        for file in (ROOT/'outputs').glob(pattern):
            for old in read(file).get('rows',[]):
                if old.get('input') and old.get('points') and old.get('folder') and (Path(old['folder'])/'response.json').is_file() and Path(old['input']).is_file():history.append(old)
    for r in unique.values():
        protected[r['input']]=sha(r['input'])
        for old in history:
            if old['points']!=r['points'] or old.get('box')!=r['box'] or sha(old['input'])!=sha(r['input']):continue
            f=Path(old['folder'])/'response.json';raw=read(f);meta=raw.get('raw_capture',{}).get('metadata',{})
            if meta.get('source_sha256')!=sam.SOURCE_SHA or meta.get('prior_mask_sha256') is not None or meta.get('experimental_box_prompt') or meta.get('diagnostic_prior_mode') or not meta.get('raw_capture_complete') or not meta.get('selected_index_observed'):continue
            if sam.key(old)!=r['sam_request_sha256']:continue
            r['exact_previous_capture']=str(f);protected[str(f)]=sha(f);break
    target.mkdir();save(target/'frozen_plan.json',{'rows':list(unique.values()),'source_admissions':admissions,'protected':protected,'source_sha256':sam.SOURCE_SHA,'max_new_calls':sum(r['exact_previous_capture'] is None for r in unique.values()),'source_evidence_type':'specified calibrated teacher stock PAL and neutral source subject, native nonborder','never_fabricate_legacy_both_stock_flag':True,'point_only_no_prior':True,'original_source_selected_index_only':True,'no_reference_or_area_for_inputs':True,'no_mask_clipping':True,'new_Agent_rounds':0,'Agent_round_limit':3});print({'accepted_logical_source_cues':sum(a['PAL_subject_evidence'] for a in admissions),'unique_SAM_requests':len(unique),'exact_cache_hits':sum(r['exact_previous_capture'] is not None for r in unique.values()),'new_SAM_calls':0},flush=True)

def run_SAM():
    w=read(OUT/'SAM_weights_identity.json');assert WEIGHTS.stat().st_size==w['bytes'] and WEIGHTS.stat().st_mtime_ns==w['mtime_ns'];sam.OUT=OUT/'SAM';sam.run()

if __name__=='__main__':
    initialize();{'review':reviewer.review,'prepare_SAM':prepare_SAM,'run_SAM':run_SAM}[sys.argv[1]]()
