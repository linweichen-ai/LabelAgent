"""Proposal-only concise prompt arm; every original final gate is preserved."""
import copy,datetime,os,shutil,sys
from pathlib import Path
from prepare_palette_parts import ROOT,BASE
from prepare_balanced_palette_parts import OUT as PARENT
from neutral_prompt_pair import OUT as PAIR,PROMPT
from run_rgb_layers import read,save
from deliver_context_graph import sha
import run_palette_parts as source
OUT=BASE/'labelagent_concise_palette_candidates_20261108'

def prepare():
    if OUT.exists():raise FileExistsError('Preserve concise proposal adapter')
    if shutil.disk_usage(BASE).free<96*2**20:raise RuntimeError('Original96MiB reserve')
    pair=read(PAIR/'completed.json');assert pair['concise_all6_controls_pass'];parent=read(PARENT/'frozen_plan.json');rows=[];reviews=[];protected=copy.deepcopy(parent['protected']);protected.update({str(PAIR/'frozen_plan.json'):sha(PAIR/'frozen_plan.json'),str(PAIR/'completed.json'):sha(PAIR/'completed.json'),str(Path(__file__)):sha(__file__),str(Path(source.__file__)):sha(source.__file__)})
    for r in parent['rows']:
        reviewed=next(v for v in pair['rows'] if v['arm']=='concise' and v['index']==r['index']);row=r|{k:reviewed[k] for k in ['bindings','request_file','request_sha256']};rows.append(row);reviews.append(row|{k:reviewed[k] for k in ['decision','PAL_subject_evidence','response_file','response_sha256']});protected[reviewed['request_file']]=sha(reviewed['request_file']);protected[reviewed['response_file']]=reviewed['response_sha256']
    controls=[]
    for name in ['gray','white']:
        reviewed=next(v for v in pair['rows'] if v['index']=='control_'+name and v['arm']=='concise');r=reviewed|{'known_no_real_object':True};controls.append(r);reviews.append(r);protected[r['request_file']]=sha(r['request_file']);protected[r['response_file']]=r['response_sha256']
    OUT.mkdir();plan=copy.deepcopy(parent);plan.update(rows=rows,controls=controls,protected=protected,proposal_variant='Exact paired concise-neutral prompt from completed22-source-point study; no relabelled ground truth',all6_prompt_controls_verified_before_proposals=True,prompt_change_only_for_proposal_admission=True,all_original_final_full_mask_semantics_unchanged=True,new_model_calls_this_adapter=0);save(OUT/'frozen_plan.json',plan);save(OUT/'review_completed.json',{'rows':reviews,'plain_controls_reject_PAL':True,'source_all6_concise_controls_pass':True,'all_actual_reviews_already_completed_at':str(PAIR/'completed.json'),'actual_new_VLM_calls_total':0,'source_prompt_trial_actual_new_calls':pair['actual_new_VLM_calls_total'],'adapter_no_new_network_calls':True,'new_Agent_runs':0});print({'source_points':len(rows),'concise_proposal_admitted':sum(r['PAL_subject_evidence'] for r in reviews if not r.get('known_no_real_object')),'new_model_calls_this_adapter':0},flush=True)

def prepare_SAM():
    source.OUT=OUT;source.api.OUT=OUT;source.prepare_SAM();file=OUT/'SAM/frozen_plan.json';p=read(file);save(OUT/'SAM/preparation_history/before_exact_palette_capture_cache.json',p);old=read(PARENT/'SAM/generation_completed.json');cache={r['request_sha256']:r for r in old['rows']};reuses=0
    for r in p['rows']:
        prior=cache.get(r['request_sha256'])
        if prior:
            response=Path(prior['folder'])/'response.json';d=read(response);m=d['raw_capture']['metadata'];assert r['points']==prior['points'] and sha(r['input'])==sha(prior['input']) and m['source_sha256']==source.api.SOURCE_SHA and m['prior_mask_sha256'] is None;r['exact_previous_capture']=str(response);p['protected'][str(response)]=sha(response);reuses+=1
    p['protected'][str(Path(__file__))]=sha(__file__);p['max_new_SAM_calls']=len({r['request_sha256'] for r in p['rows'] if not r['exact_previous_capture']});p['exact_parent_palette_capture_reuses']=reuses;save(file,p);print({'logical_SAM_requests':len(p['rows']),'exact_parent_capture_reuses':reuses,'max_new_SAM_calls':p['max_new_SAM_calls']},flush=True)

def main():
    stage=sys.argv[1]
    if stage=='prepare':prepare();return
    if stage=='prepare_SAM':prepare_SAM();return
    source.OUT=OUT;source.api.OUT=OUT
    if stage=='SAM':source.SAM();return
    import audit_palette_parts as audit
    audit.OUT=OUT
    if stage=='prepare_semantics':audit.prepare_semantics()
    elif stage=='audit':
        original_body=audit.api.body
        def concise_body(first,second):
            body=original_body(first,second);body['messages'][0]['content'][0]['text']=PROMPT;return body
        audit.api.body=concise_body;audit.audit()
    else:
        import review_box_layers as reviewer
        reviewer.OUT=OUT;reviewer.S=OUT/'semantic';reviewer.run()

if __name__=='__main__':main()
