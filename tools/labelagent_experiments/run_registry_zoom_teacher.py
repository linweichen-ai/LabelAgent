"""Same six native ROIs and CPU recipe, two additional local registered weights."""
import sys
from pathlib import Path
from run_rgb_layers import ROOT,read,save,sha
import run_zoom_teacher as runner
OUT=ROOT/'outputs/labelagent_registry_zoom_teacher_20261108'
MODELS=[{'name':'R5_historical_boundary','path':str(ROOT/'task_workspaces/cvat_mask2former_r5_20261103/runs/epoch_20/checkpoint'),'sha256':'c6255fc5e7d2676991288cd3a216c0366e61b5b833012c7f7ce51d9db80eb716'},{'name':'R15_PAL_candidate','path':str(ROOT/'task_workspaces/cvat_mask2former_r15_20261104/best_model'),'sha256':'4b98939d0b6208feac55029f956632d16effbee584e65f8aaddda3ad3e3a65f9'}]

def main():
    runner.OUT=OUT
    if sys.argv[1]=='prepare':
        for model in MODELS:
            p=Path(model['path']);assert sha(p/'model.safetensors')==model['sha256'];assert read(p/'config.json')==read(runner.PACK/'model/config.json') and read(p/'preprocessor_config.json')==read(runner.PACK/'model/preprocessor_config.json')
        runner.prepare();p=read(OUT/'frozen_plan.json');p['models']=MODELS;p['protected'][str(Path(__file__))]=sha(__file__)
        for model in MODELS:
            for name in ['model.safetensors','config.json','preprocessor_config.json']:f=Path(model['path'])/name;p['protected'][str(f)]=sha(f)
        registry=ROOT/'task_workspaces/cvat_mask2former_r15_20261104/recommended_pipeline.json';p['protected'][str(registry)]=sha(registry);p.update(model_scope='R5 historical boundary checkpoint and R15 unadopted PAL candidate, no default deployment change',common_experimental_input_height_width=[448,576],R5_original_recommended_input=[384,512],not_reproduction_of_R5_native_384_recipe=True,common_original_postprocess_unchanged_from_current_trial=True,earlier_two_checkpoint_comparison_kept_separate=True);save(OUT/'frozen_plan.json',p)
    else:
        p=read(OUT/'frozen_plan.json');assert p['protected'][str(Path(__file__))]==sha(__file__);runner.run()

if __name__=='__main__':main()
