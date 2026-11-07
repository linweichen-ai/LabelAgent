"""Same frozen max-two source stock rule, on native-aspect real context."""
from pathlib import Path
from run_rgb_layers import ROOT,read,save,sha
import prepare_zoom_cues as runner
Z=Path('D:/AI/Codex/Home/visualizations/2026/11/04/01a1a56c-0d6f-7d61-8326-d3bbd344d2df/labelagent_aspect_context_teacher_20261108')
OUT=ROOT/'outputs/labelagent_aspect_context_cues_20261108'

if __name__=='__main__':
    runner.Z=Z;runner.OUT=OUT;runner.main();p=read(OUT/'frozen_plan.json');p['protected'][str(Path(__file__))]=sha(__file__);p['matched_source_followup_max_two_rule_unchanged']=True;p['new_context_and_aspect_are_joint_input_change_not_isolated_shape_causality']=True;save(OUT/'frozen_plan.json',p)
