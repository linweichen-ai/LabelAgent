"""Before network, keep parent measurements separate from fresh SAM outputs."""
from pathlib import Path
from run_rgb_layers import read,save,sha
import run_mask_graph_collision as runner
import run_trained_point_sam as sam

if __name__=='__main__':
    runner.initialize();runner.prepare_SAM();file=runner.OUT/'SAM/frozen_plan.json';before=read(file);history=runner.OUT/'proposal_adapter/parent_provenance_plan.json';save(history,before);fields=['folder','metadata_file','raw_array_file','mask_sha256','raw_pixels','prediction_call','native_index','valid','source_selected','new_SAM_call','response_sha256','all_native_arrays_retained','low_res_logits_present','touches_border','native_prompt_guard','status'];rows=[]
    for row in before['rows']:
        original_key=sam.key(row);measurements={k:row[k] for k in fields if k in row};clean={k:v for k,v in row.items() if k not in fields};clean['source_parent_measurements_not_new_predictions']=measurements;assert sam.key(clean)==original_key==clean['sam_request_sha256'];rows.append(clean)
    after={**before,'rows':rows,'proposal_provenance_adapter':'Inherited parent output fields retained only under source_parent_measurements_not_new_predictions, before any new SAM request','no_input_image_points_or_request_bytes_changed':True};after['protected'].update({str(Path(__file__)):sha(__file__),str(history):sha(history)});save(file,after);save(runner.OUT/'proposal_adapter/readback.json',{'passed':True,'source_parent_fields_separated_before_new_predictions':True,'exact_request_hashes_unchanged':True,'new_model_calls':0});print({'prepared_hypotheses':len(rows),'point_count':sum(len(r['points']) for r in rows),'parent_measurements_separate':True,'new_model_calls':0},flush=True)
