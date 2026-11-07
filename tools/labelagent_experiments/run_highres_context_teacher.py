"""Same source pixels/context/weights/thresholds; double model input resolution."""
import copy,datetime,shutil,sys
from pathlib import Path
from run_rgb_layers import read,save,sha
import highres_context_cpu_engine as runner
LOW=Path('D:/AI/Codex/Home/visualizations/2026/11/04/01a1a56c-0d6f-7d61-8326-d3bbd344d2df/labelagent_aspect_context_teacher_20261108');OUT=runner.OUT

def prepare():
    if OUT.exists():raise FileExistsError('Preserve frozen high-resolution source trial')
    if shutil.disk_usage(OUT.parent).free<512*2**20:raise RuntimeError('Stronger512MiB high-resolution output reserve')
    original=LOW/'frozen_plan.json';p=copy.deepcopy(read(original));assert all(sha(f)==h for f,h in p['protected'].items()) and read(LOW/'generation_completed.json')['new_CPU_model_forwards']==12;p['at']=datetime.datetime.now().astimezone().isoformat();p['protected'].update({str(original):sha(original),str(LOW/'generation_completed.json'):sha(LOW/'generation_completed.json'),str(LOW/'actual_geometry_readback.json'):sha(LOW/'actual_geometry_readback.json'),str(Path(__file__)):sha(__file__),str(Path(runner.__file__)):sha(runner.__file__)})
    geometry=read(LOW/'actual_geometry_readback.json')['rows']
    for view in p['views']:
        actual=next(r for r in geometry if r['teacher_key']==view['teacher_key']);view['previous_inherited_scale_diagnostics_not_used']={k:view.pop(k) for k in ['vertical_object_scale_ratio_vs_old_view','horizontal_object_scale_ratio_vs_old_view']};view['source_context_scale_vs_historical_quarter_xy']=actual['actual_scale_vs_historical_quarter_xy'];view['matched_low_resolution_input']=view['input'];view['model_input_height_width']=[896,1152];view['no_new_sensor_pixel_detail_created']=True
    p.update(input_height_width=[896,1152],matched_low_resolution_input_height_width=[448,576],same_source_context_pixels_and_native_aspect=True,only_model_input_resolution_doubled_each_axis=True,no_additional_source_detail_or_training=True,max_new_CPU_forwards=12,startup_available_RAM_GiB_min=3,running_available_RAM_GiB_min=2,no_reference_or_budget_for_inputs=True,new_Agent_rounds=0,Agent_round_limit=3,all_final_gates_unchanged=True,followup_proposals_predeclared='At most two stock PAL candidates per model/ROI by original score then ID; original class/mask/overlap gates unchanged, reject native and stock ROI-border; all raw100 queries retained');OUT.mkdir();save(OUT/'frozen_plan.json',p);print({'frozen_same_source_ROIs':6,'trained_weights':2,'model_input':[896,1152],'max_new_CPU_forwards':12,'new_model_calls':0},flush=True)

if __name__=='__main__':prepare() if sys.argv[1]=='prepare' else runner.run()
