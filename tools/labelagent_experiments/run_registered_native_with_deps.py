"""Bind the existing project dependency directory, preserving the frozen driver."""
import sys
from pathlib import Path
from run_rgb_layers import ROOT,read,save,sha
import run_registered_native_baseline as trial
if __name__=='__main__':
    cfg=read(ROOT/'task_workspaces/labelagent_point_ids_20261107/config.json');sys.path.insert(0,cfg['dependency_dir'])
    file=trial.OUT/'engineering_failures/dependency_binding.json'
    if not file.exists():save(file,{'initial_failure':'transformers absent from bare service venv; occurred before model loading or any forward','project_dependency_dir':cfg['dependency_dir'],'wrapper_sha256':sha(__file__),'frozen_driver_sha256':sha(trial.__file__),'no_model_inputs_thresholds_or_driver_changed':True,'new_model_calls_before_fix':0})
    trial.run()
