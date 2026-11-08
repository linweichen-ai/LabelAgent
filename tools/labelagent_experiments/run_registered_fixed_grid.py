"""Additional registered checkpoints on the identical frozen whole-frame grid.

This controlled448x576 comparison is separate from the actual R5 native
recommendation at384x512. No reference coordinates choose views or cues.
"""
import copy,datetime,os,sys,traceback
from pathlib import Path
from run_rgb_layers import ROOT,read,save
from deliver_context_graph import sha
import fixed_grid_cpu_engine as engine
import run_registry_zoom_teacher as registry
BASE=engine.OUT.parent;OUT=BASE/'labelagent_registered_fixed_grid_20261108'
def prepare():
    if OUT.exists():raise FileExistsError('Preserve registered fixed-grid protocol')
    source=engine.OUT/'frozen_plan.json';p=copy.deepcopy(read(source));p['models']=copy.deepcopy(registry.MODELS);old_model_dirs=[Path(r['path']) for r in read(source)['models']];protected={f:h for f,h in p['protected'].items() if not any(Path(f).parent==d for d in old_model_dirs)}
    protected.update({str(source):sha(source),str(Path(__file__)):sha(__file__),str(Path(registry.__file__)):sha(registry.__file__)})
    for r in p['models']:
        path=Path(r['path']);assert sha(path/'model.safetensors')==r['sha256'];assert read(path/'config.json')==read(Path(p['processor_used'])/'config.json') and read(path/'preprocessor_config.json')==read(Path(p['processor_used'])/'preprocessor_config.json')
        for name in ['model.safetensors','config.json','preprocessor_config.json']:protected[str(path/name)]=sha(path/name)
    for name in ['config.json','preprocessor_config.json']:
        file=Path(p['processor_used'])/name;protected[str(file)]=sha(file)
    p.update(protected=protected,model_scope='R5 historical registered boundary and R15 offline unadopted specialist candidate; same grid, tensors, original calibrated postprocessing as prior d11c/6820 comparison',actual_R5_native_384_512_trial_separate=True,not_reproduction_of_registered_native_deployment_recipe=True,only_weights_changed_relative_to_completed_fixed_grid=True,max_new_CPU_forwards=24,real_scene_forwards=20,synthetic_control_forwards=4,no_reference_inputs=True,new_training=False,new_Agent_runs=0,adopted=False);OUT.mkdir();save(OUT/'frozen_plan.json',p);save(OUT/'prepared.json',{'status':'prepared_not_executed','max_CPU_forwards':24,'real':20,'synthetic_controls':4,'new_model_calls':0});print({'prepared_registered_weights':2,'source_views':12,'max_CPU_forwards':24,'new_calls':0},flush=True)
def run():
    if (OUT/'generation_completed.json').exists():raise FileExistsError('Complete registered grid, do not repeat')
    import psutil
    runtime=OUT/'runtime_process.json'
    if runtime.exists():
        old=read(runtime)
        try:
            process=psutil.Process(old['pid']);alive=process.is_running() and abs(process.create_time()-old['create_time'])<.01
        except psutil.NoSuchProcess:alive=False
        if alive:raise RuntimeError('The exact previous registered grid process is still live; poll it rather than restart')
    for raw in OUT.glob('queries/*/*/raw_queries.npz'):
        if not (raw.parent/'complete.json').exists():raise RuntimeError('A raw query capture lacks completion metadata: preserve and diagnose before resuming; never repeat its forward')
    state={'pid':os.getpid(),'create_time':psutil.Process().create_time(),'started_at':datetime.datetime.now().astimezone().isoformat(),'status':'running','source_plan_sha256':sha(OUT/'frozen_plan.json'),'driver_sha256':sha(__file__),'GPU_calls':0,'max_CPU_forwards':24};save(runtime,state);engine.OUT=OUT
    try:engine.run();state['status']='complete'
    except BaseException as e:state.update(status='failed',error=str(e));save(OUT/'runtime_failure.json',state|{'traceback':traceback.format_exc()});raise
    finally:save(runtime,state|{'last_at':datetime.datetime.now().astimezone().isoformat()})
if __name__=='__main__':prepare() if sys.argv[1]=='prepare' else run()
