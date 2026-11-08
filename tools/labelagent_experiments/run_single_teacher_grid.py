"""Separate proposal arm; unchanged neutral reviewer and native SAM decoder."""
import datetime,hashlib,json,os,sys
from pathlib import Path
from prepare_single_teacher_grid import OUT,ROOT
from run_rgb_layers import read,save
from deliver_context_graph import sha
import run_fixed_grid_followup as original

def prepare_SAM():
    target=OUT/'SAM'
    if target.exists():raise FileExistsError('Preserve single-teacher SAM plan')
    p=original.initialize();d=read(OUT/'review_completed.json');assert d['plain_controls_reject_PAL'];selected=[];requests=[]
    for r in p['rows']:
        rr=next(v for v in d['rows'] if v['index']==r['index']);assert r['teacher_stock_PAL_verified'] and r['both_model_consensus_required_for_this_proposal_arm'] is False
        if not rr['PAL_subject_evidence']:continue
        if len(selected)>=16:break
        selected.append(r)
        for arm in ['original_source','explicit_padded_view']:
            if arm=='original_source':image=r['source_image'];point=r['source_point'];virtual=[0,0,1920,1536];observed=[0,0,1920,1536]
            else:image=r['padded_input'];point=r['input_point'];virtual=r['source_virtual_box'];observed=r['observed_region_in_input']
            import base64
            payload={'image':base64.b64encode(Path(image).read_bytes()).decode(),'points':[{'x':point[0],'y':point[1],'label':1}],'include_raw_capture':True};key=hashlib.sha256(json.dumps(payload,sort_keys=True).encode()).hexdigest();file=target/'prepared_requests'/(key+'_PRIVATE.json');save(file,payload);requests.append(r|{'arm':arm,'input':image,'input_sha256':sha(image),'points':payload['points'],'virtual_box':virtual,'observed_rectangle':observed,'request_file':str(file),'request_sha256':key,'neutral_response':rr['response_file'],'neutral_response_sha256':rr['response_sha256']})
    assert len(requests)<=32 and sha(original.SOURCE)==original.SOURCE_SHA
    old=ROOT/'outputs/labelagent_registry_zoom_pool_20261108/SAM_weights_identity.json';w=read(old);weight=Path(w['path']);assert weight.stat().st_size==w['bytes'] and weight.stat().st_mtime_ns==w['mtime_ns'];save(target/'weights_same_stat_binding.json',{'original_streamed_identity':str(old),'original_receipt_sha256':sha(old),'current_size_mtime_exact':True,'not_a_new_full_weight_hash':True});protected={str(OUT/'frozen_plan.json'):sha(OUT/'frozen_plan.json'),str(OUT/'review_completed.json'):sha(OUT/'review_completed.json'),str(Path(__file__)):sha(__file__),str(Path(original.__file__)):sha(original.__file__),str(original.SOURCE):original.SOURCE_SHA,str(target/'weights_same_stat_binding.json'):sha(target/'weights_same_stat_binding.json')}
    for r in requests:protected[r['input']]=r['input_sha256'];protected[r['request_file']]=sha(r['request_file']);protected[r['neutral_response']]=r['neutral_response_sha256']
    save(target/'frozen_plan.json',{'rows':requests,'protected':protected,'actual_point_candidates':len(selected),'max_new_SAM_calls':len({r['request_sha256'] for r in requests}),'source_proposal_only_without_pair_consensus':True,'not_a_fabricated_both_stock_maps_flag':True,'point_only_no_prior':True,'source_selected_index_only':True,'no_mask_clipping':True,'all_final_gates_unchanged':True,'Agent_round_limit':3,'new_Agent_runs':0});print({'prepared_point_candidates':len(selected),'logical_paired_requests':len(requests),'new_calls':0},flush=True)

def main():
    original.OUT=OUT;execution=OUT/'single_teacher_execution_plan.json';hashes={str(Path(__file__)):sha(__file__),str(Path(original.__file__)):sha(original.__file__)}
    if not execution.exists():save(execution,{'protected':hashes,'source_plan_sha256':sha(OUT/'frozen_plan.json'),'no_default_or_acceptance_changes':True})
    e=read(execution);assert e['protected']==hashes and e['source_plan_sha256']==sha(OUT/'frozen_plan.json')
    import psutil
    phase=sys.argv[1];file=OUT/(phase+'_runtime_process.json');state={'pid':os.getpid(),'create_time':psutil.Process().create_time(),'at':datetime.datetime.now().astimezone().isoformat(),'phase':phase,'status':'running'}
    if file.exists():
        prior=read(file)
        try:
            process=psutil.Process(prior['pid']);live=process.is_running() and abs(process.create_time()-prior['create_time'])<.01
        except psutil.NoSuchProcess:live=False
        if live:raise RuntimeError('Exact prior phase still live; poll instead of restart')
    save(file,state)
    try:{'review':original.review,'prepare_SAM':prepare_SAM,'SAM':original.SAM}[phase]();state['status']='complete'
    except BaseException as error:state.update(status='failed',error=str(error));raise
    finally:save(file,state|{'last_at':datetime.datetime.now().astimezone().isoformat()})

if __name__=='__main__':main()
