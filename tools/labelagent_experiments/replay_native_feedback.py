"""Real-capture replay only; never starts Agent rounds or modifies a service."""
from pathlib import Path
from prepare_palette_parts import ROOT,BASE
from run_rgb_layers import read,save
from deliver_context_graph import sha
from native_review_feedback import review_response
OUT=BASE/'labelagent_native_feedback_replay_20261108'

def main():
    if OUT.exists():raise FileExistsError('Preserve real native-feedback replay')
    rows=[];protected={str(Path(__file__)):sha(__file__),str(Path(__file__).with_name('native_review_feedback.py')):sha(Path(__file__).with_name('native_review_feedback.py'))}
    for name in ['labelagent_fixed_grid_followup_20261108','labelagent_single_teacher_grid_20261108','labelagent_balanced_palette_parts_20261108','labelagent_source_gap_instances_20261108']:
        file=BASE/name/'SAM/generation_completed.json';g=read(file);protected[str(file)]=sha(file)
        for r in g['rows']:
            response=Path(r['folder'])/'response.json';protected[str(response)]=sha(response);reply=read(response);first=review_response(reply,r['points'],1);last=review_response(reply,r['points'],3);assert first['native_prompt_passed']==last['native_prompt_passed'] and last['remaining_rounds']==0 and first['keep_current_annotation'] and not first['annotation_write_permitted'];rows.append({'cohort':name,'cue':r['index'],'arm':r.get('arm'),'response':str(response),'response_sha256':sha(response),'round1_feedback':first,'round3_feedback':last,'no_real_Agent_or_model_call':True})
    assert all(sha(f)==h for f,h in protected.items());result={'status':'complete_offline_native_feedback_adapter_replay','rows':rows,'real_capture_rows':len(rows),'native_passed':sum(r['round1_feedback']['native_prompt_passed'] for r in rows),'quiet_native_failures_detected':sum(r['round1_feedback'].get('service_returned_without_warning_but_native_failed',False) for r in rows),'all_at_limit_failures_abstain':all(r['round3_feedback']['next_action']=='abstain_at_limit' for r in rows if not r['round3_feedback']['native_prompt_passed']),'new_model_calls':0,'new_Agent_runs':0,'annotation_writes':0,'deployed':False,'not_an_Agent_convergence_result':True,'protected':protected};save(OUT/'completed.json',result);print({k:v for k,v in result.items() if k not in ['rows','protected']},flush=True)

if __name__=='__main__':main()
