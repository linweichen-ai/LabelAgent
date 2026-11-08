"""Explicitly bind unchanged readback/semantics to the separate source cohort."""
import sys
from pathlib import Path
from prepare_single_teacher_grid import OUT
from run_rgb_layers import save
from deliver_context_graph import sha

def main():
    stage=sys.argv[1]
    if stage=='prepare_semantics':
        import review_fixed_grid_instances as wrapper
        wrapper.OUT=OUT;wrapper.prepare()
        save(OUT/'explicit_audit_binding.json',{'cohort':str(OUT),'wrapper_sha256':sha(__file__),'original_semantic_adapter_sha256':sha(wrapper.__file__),'all_valid_native_outputs_selected_before_new_support_area_audit':True,'no_old_cohort_result_overwritten':True})
    elif stage=='audit':
        import audit_fixed_grid_candidates as audit
        audit.OUT=OUT;audit.main()
    else:
        import review_box_layers as reviewer
        reviewer.OUT=OUT;reviewer.S=OUT/'semantic';reviewer.run()

if __name__=='__main__':main()
