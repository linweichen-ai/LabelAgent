"""Exact cached neutral review and deduplicated point-SAM of frozen zoom cues."""
import os,sys
from pathlib import Path
from run_rgb_layers import ROOT,read,save,sha
import run_base_material as reviewer
import run_trained_point_sam as sam
OUT=ROOT/'outputs/labelagent_zoom_cues_20261108'

def initialize():
    file=OUT/'execution_plan.json';cache=OUT/'exact_cache_catalogue/completed.json'
    if not file.exists():
        sources=[Path(os.environ['LABELAGENT_PAIRED_CUE_OUTPUT'])/'completed.json',ROOT/'outputs/labelagent_trained_point_review_20261108/completed.json',ROOT/'outputs/labelagent_base_material_20261108/review_completed.json',ROOT/'outputs/labelagent_base_blue_surface_20261108/review_completed.json'];rows=[]
        for p in sources:rows+=read(p)['rows']
        save(cache,{'rows':rows,'source_files_sha256':{str(p):sha(p) for p in sources}});save(file,{'driver_sha256':sha(__file__),'reviewer_sha256':sha(reviewer.__file__),'sam_runner_sha256':sha(sam.__file__),'source_plan_sha256':sha(OUT/'frozen_plan.json'),'cache_catalogue_sha256':sha(cache),'only_source_conditions_not_reference_for_admission':True})
    p=read(file);assert p['driver_sha256']==sha(__file__) and p['reviewer_sha256']==sha(reviewer.__file__) and p['sam_runner_sha256']==sha(sam.__file__) and p['source_plan_sha256']==sha(OUT/'frozen_plan.json') and p['cache_catalogue_sha256']==sha(cache);reviewer.OUT=OUT;reviewer.PAIR=cache.parent

def prepare_SAM():
    p=read(OUT/'frozen_plan.json');r=read(OUT/'review_completed.json');unique={};admissions=[]
    for cue in p['rows']:
        reviews=[x for x in r['rows'] if x['partition']==cue['index']];assert len(reviews)==1;admitted=reviews[0]['PAL_subject_evidence'];admissions.append({'cue':cue['index'],'model':cue['model'],'PAL_subject_evidence':admitted})
        if not admitted:continue
        k=sam.key(cue)
        if k not in unique:unique[k]={**cue,'PAL_subject_evidence':True,'logical_zoom_cues_sharing_exact_SAM_input':[]}
        unique[k]['logical_zoom_cues_sharing_exact_SAM_input'].append(cue['index'])
    save(OUT/'proposal_adapter/completed.json',{'rows':list(unique.values()),'source_admissions':admissions,'neutral_review_sha256':sha(OUT/'review_completed.json'),'exact_same_input_deduplicated_before_network':True,'MF_native_prior_not_sent_in_point_only_arm':True});sam.OUT=OUT/'SAM';sam.REVIEW=OUT/'proposal_adapter';sam.prepare();print({'neutral_accepted_logical_cues':sum(a['PAL_subject_evidence'] for a in admissions),'unique_SAM_requests':len(unique),'new_SAM_calls':0},flush=True)

def main():
    initialize()
    if sys.argv[1]=='review':reviewer.review()
    elif sys.argv[1]=='prepare_SAM':prepare_SAM()
    else:sam.OUT=OUT/'SAM';sam.REVIEW=OUT/'proposal_adapter';sam.run()

if __name__=='__main__':main()
