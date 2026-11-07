"""Neutral review of new source anchors, then complete native SAM hypotheses."""
import os,shutil,sys
from pathlib import Path
from run_rgb_layers import ROOT,read,save,sha
import run_base_material as reviewer
import run_trained_point_sam as sam
OUT=Path('D:/AI/Codex/Home/visualizations/2026/11/04/01a1a56c-0d6f-7d61-8326-d3bbd344d2df/labelagent_mask_graph_collision_20261108')
HIGH=OUT.with_name('labelagent_highres_context_cues_20261108')

def initialize():
    file=OUT/'execution_plan.json';cache=OUT/'exact_cache_catalogue/completed.json'
    if not file.exists():
        sources=[Path(os.environ['LABELAGENT_PAIRED_CUE_OUTPUT'])/'completed.json',HIGH/'review_completed.json']+[ROOT/'outputs'/n/f for n,f in [('labelagent_trained_point_review_20261108','completed.json'),('labelagent_base_material_20261108','review_completed.json'),('labelagent_base_blue_surface_20261108','review_completed.json'),('labelagent_zoom_cues_20261108','review_completed.json'),('labelagent_registry_zoom_pool_20261108','review_completed.json'),('labelagent_height_review_20261108','review_completed.json'),('labelagent_aspect_context_cues_20261108','review_completed.json')]];rows=[]
        for p in sources:rows+=read(p)['rows']
        save(cache,{'rows':rows,'sources_sha256':{str(p):sha(p) for p in sources}});save(file,{'driver_sha256':sha(__file__),'reviewer_sha256':sha(reviewer.__file__),'source_plan_sha256':sha(OUT/'frozen_plan.json'),'cache_sha256':sha(cache),'existing_positive_evidence_bound_new_positives_must_all_pass_original_neutral_guard':True,'negative_collision_points_are_proposal_hypotheses_not_truth':True})
    p=read(file);assert p['driver_sha256']==sha(__file__) and p['reviewer_sha256']==sha(reviewer.__file__) and p['source_plan_sha256']==sha(OUT/'frozen_plan.json') and p['cache_sha256']==sha(cache);reviewer.OUT=OUT;reviewer.PAIR=cache.parent

def prepare_SAM():
    target=OUT/'SAM'
    if target.exists():raise FileExistsError('Preserve graph hypotheses')
    if shutil.disk_usage(OUT).free<96*2**20:raise RuntimeError('Original96MiB preparation reserve')
    p=read(OUT/'frozen_plan.json');d=read(OUT/'review_completed.json');assert all(sha(f)==h for f,h in p['protected'].items());rows=[];admissions=[]
    for r in p['rows']:
        extra=[x for x in d['rows'] if x['partition']==r['index']];assert len(extra)==len(r['new_positive_points']);admitted=bool(r['parent_source_neutral_PAL_bound'] and all(x['PAL_subject_evidence'] for x in extra));admissions.append({'cue':r['index'],'arm':r['arm'],'source_parent':r['source_parent_uid'],'all_new_positive_subjects_PAL':admitted,'negative_identity_unconfirmed':True})
        if admitted:rows.append({**r,'sam_request_sha256':sam.key(r),'exact_previous_capture':None})
    assert sum(len(r['points']) for r in rows)<=32
    wfile=HIGH/'SAM_weights_identity.json';w=read(wfile);weights=Path(w['path']);assert weights.stat().st_size==w['bytes'] and weights.stat().st_mtime_ns==w['mtime_ns'];protected={str(OUT/'frozen_plan.json'):sha(OUT/'frozen_plan.json'),str(OUT/'review_completed.json'):sha(OUT/'review_completed.json'),str(Path(__file__)):sha(__file__),str(Path(sam.__file__)):sha(sam.__file__),str(wfile):sha(wfile)}
    history=[]
    for root in [ROOT/'outputs',OUT.parent]:
        for pattern in ['labelagent_*/generation_completed.json','labelagent_*/SAM/generation_completed.json']:
            for file in root.glob(pattern):
                for old in read(file).get('rows',[]):
                    if old.get('points') and old.get('input') and old.get('folder'):history.append(old)
    for r in rows:
        protected[r['input']]=sha(r['input'])
        for old in history:
            if old['points']!=r['points'] or old.get('box')!=r['box'] or not Path(old['input']).is_file() or sha(old['input'])!=sha(r['input']):continue
            f=Path(old['folder'])/'response.json'
            if not f.is_file():continue
            meta=read(f).get('raw_capture',{}).get('metadata',{})
            if meta.get('source_sha256')!=sam.SOURCE_SHA or meta.get('prior_mask_sha256') is not None or meta.get('experimental_box_prompt') or not meta.get('raw_capture_complete') or not meta.get('selected_index_observed'):continue
            if sam.key(old)!=r['sam_request_sha256']:continue
            r['exact_previous_capture']=str(f);protected[str(f)]=sha(f);break
    target.mkdir();save(target/'frozen_plan.json',{'rows':rows,'source_admissions':admissions,'protected':protected,'source_sha256':sam.SOURCE_SHA,'max_new_calls':sum(r['exact_previous_capture'] is None for r in rows),'all_point_annotations':sum(len(r['points']) for r in rows),'cumulative_point_budget':32,'same_original_parent_positive_preserved':True,'new_positive_points_all_neutrally_reviewed':True,'negative_graph_identity_is_unconfirmed_hypothesis':True,'sampling_region_not_sent_as_prior_or_output':True,'no_mask_prior':True,'original_selected_index_required':True,'no_reference_or_budget_for_generation':True,'no_output_mask_clipping':True,'all_original_acceptance_gates_unchanged':True,'new_Agent_rounds':0,'Agent_round_limit':3});print({'admitted_hypotheses':len(rows),'exact_previous_reuses':sum(r['exact_previous_capture'] is not None for r in rows),'planned_points':sum(len(r['points']) for r in rows),'new_SAM_calls':0},flush=True)

if __name__=='__main__':
    initialize()
    if sys.argv[1]=='review':reviewer.review()
    elif sys.argv[1]=='prepare_SAM':prepare_SAM()
    else:
        w=read(HIGH/'SAM_weights_identity.json');p=Path(w['path']);assert p.stat().st_size==w['bytes'] and p.stat().st_mtime_ns==w['mtime_ns'];sam.OUT=OUT/'SAM';sam.run()
