"""Resume derived audit on another volume without changing sources or gates."""
from pathlib import Path
from run_rgb_layers import read,save,sha
import audit_aspect_context_cues as engine
OUT=Path('D:/AI/Codex/Home/visualizations/2026/11/04/01a1a56c-0d6f-7d61-8326-d3bbd344d2df/labelagent_aspect_context_audit_20261108')
if __name__=='__main__':
    save(OUT/'audit_execution_binding.json',{'driver_sha256':sha(__file__),'original_unchanged_audit_engine_sha256':sha(engine.__file__),'input_snapshot_sha256':{name:sha(OUT/name) for name in ['frozen_plan.json','review_completed.json','native_pool_source_plan.json','SAM/frozen_plan.json','SAM/generation_completed.json']},'only_derivative_output_directory_changed':True,'no_model_or_mask_source_mutation':True});engine.OUT=OUT;engine.main()
