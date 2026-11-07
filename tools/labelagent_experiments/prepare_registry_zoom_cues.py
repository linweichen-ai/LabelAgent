"""Original source ranking and stock checks for the two registry checkpoints."""
from pathlib import Path
from run_rgb_layers import ROOT,read,save,sha
import prepare_zoom_cues as runner
OUT=ROOT/'outputs/labelagent_registry_zoom_cues_20261108'

if __name__=='__main__':
    runner.Z=ROOT/'outputs/labelagent_registry_zoom_teacher_20261108';runner.OUT=OUT;runner.main();p=read(OUT/'frozen_plan.json');p['protected'][str(Path(__file__))]=sha(__file__);p['registry_checkpoint_scope']='R5 historical weights and R15 unadopted candidate with common experimental448x576 inputs, not default recipe reproduction';save(OUT/'frozen_plan.json',p)
