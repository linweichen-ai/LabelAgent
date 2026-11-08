"""CPU replay of the actual R15 recommendation: R5 boundary,384x512,no specialist.

Only the original two inference functions and geometry helper are executed;
evaluation and reference-loading functions are not imported or called. Hooks
observe actual forwards and preserve every raw query without changing output.
"""
import os
os.environ['CUDA_VISIBLE_DEVICES']='-1'
os.environ['HF_HUB_OFFLINE']='1';os.environ['TRANSFORMERS_OFFLINE']='1'
import ast,hashlib,inspect,shutil,sys
from pathlib import Path
from run_rgb_layers import ROOT,read,save
from deliver_context_graph import sha
import run_fixed_grid_teacher as previous
OUT=previous.OUT.with_name('labelagent_registered_native_baseline_20261108')
CARD=ROOT/'task_workspaces/cvat_mask2former_r15_20261104/recommended_pipeline.json'
ORIGINAL=ROOT/'task_workspaces/cvat_mask2former_r6_20261103/evaluate_holdout.py'
GEOMETRY=ROOT/'task_workspaces/cvat_mask2former_20261103/prepare_dataset.py'
def functions(path,names):
    text=path.read_text('utf-8-sig');tree=ast.parse(text);found={n.name:ast.get_source_segment(text,n) for n in tree.body if isinstance(n,ast.FunctionDef) and n.name in names};assert set(found)==set(names);return '\n\n'.join(found[n] for n in names)
def prepare():
    if OUT.exists():raise FileExistsError('Preserve actual registered-recipe trial')
    card=read(CARD);assert card['use_specialist'] is False and card['boundary_input']==[384,512];checkpoint=Path(card['boundary_checkpoint']);assert sha(checkpoint/'model.safetensors')==card['boundary_sha256'];protected={str(p):sha(p) for p in [Path(__file__),CARD,ORIGINAL,GEOMETRY,ROOT/'task_workspaces/cvat_mask2former_r15_20261104/export.py']+[checkpoint/name for name in ['model.safetensors','config.json','preprocessor_config.json']]}
    if shutil.disk_usage(OUT.parent).free<256*2**20:raise RuntimeError('Full capture reserve')
    from PIL import Image
    OUT.mkdir();rows=[]
    for cam in [0,1]:
        case=f'camera{cam}_frame76';file=ROOT/'outputs/labelagent_context_comparison_20261105/inputs'/(case+'.png');protected[str(file)]=sha(file);rows.append({'case':case,'input':str(file),'synthetic_control':False})
    for name,value in [('gray',127),('white',255)]:
        file=OUT/'controls'/(name+'.png');file.parent.mkdir(exist_ok=True);Image.new('RGB',(1920,1536),(value,)*3).save(file);protected[str(file)]=sha(file);rows.append({'case':'synthetic_control_'+name,'input':str(file),'synthetic_control':True})
    code=functions(ORIGINAL,['memory_safe_postprocess','predict']);geometry=functions(GEOMETRY,['geometry']);save(OUT/'frozen_plan.json',{'rows':rows,'protected':protected,'model':str(checkpoint),'model_sha256':card['boundary_sha256'],'model_source_sha256':read(previous.OUT/'frozen_plan.json')['model_source_sha256'],'input_height_width':card['boundary_input'],'specialist_active':False,'specialist_candidate_not_deployed':True,'original_inference_functions':code,'original_geometry_function':geometry,'exact_inference_function_source_sha256':hashlib.sha256(code.encode()).hexdigest(),'original_postprocessing':{'object_score':.5,'mask_score':.5,'overlap':.8,'no_null_promotion':True},'source_recipe':'Current R15 recommendation and export default with use_specialist=false, actual R5 boundary checkpoint at384x512; not the older R5 export288x384 or prior uniform448x576 comparison','max_new_CPU_forwards':4,'CPU_FP32':True,'CPU_threads':2,'startup_RAM_GiB_min':3,'runtime_RAM_GiB_min':2,'actual_source_forwards_observed_without_output_patch':True,'no_reference_coordinates_or_annotations_read_for_generation':True,'no_new_training':True,'new_GPU_calls':0,'new_Agent_runs':0,'all_original_final_acceptance_gates_unchanged':True,'adopted':False});print({'prepared':4,'real_sources':2,'plain_controls':2,'actual_recommended_boundary':'R5','input':[384,512],'R15_specialist':False,'new_calls':0},flush=True)
def run():
    if (OUT/'generation_completed.json').exists():raise FileExistsError('Do not repeat native recipe')
    p=read(OUT/'frozen_plan.json');assert all(sha(f)==h for f,h in p['protected'].items());import numpy as np,psutil,torch
    from PIL import Image
    from transformers import Mask2FormerForUniversalSegmentation,Mask2FormerImageProcessor
    assert sha(inspect.getfile(Mask2FormerForUniversalSegmentation))==p['model_source_sha256']
    if psutil.virtual_memory().available<3*2**30:raise RuntimeError('Original3GiB startup guard')
    torch.set_num_threads(2);torch.set_num_interop_threads(1);psutil.Process().nice(psutil.BELOW_NORMAL_PRIORITY_CLASS);scope={'np':np,'torch':torch,'F':torch.nn.functional,'Image':Image};exec(compile(p['original_geometry_function'],str(GEOMETRY),'exec'),scope);exec(compile(p['original_inference_functions'],str(ORIGINAL),'exec'),scope);processor=Mask2FormerImageProcessor.from_pretrained(p['model'],local_files_only=True);model=None;rows=[];fresh=0
    for r in p['rows']:
        folder=OUT/'captures'/r['case'];complete=folder/'complete.json'
        if complete.exists():
            old=read(complete);assert all(sha(folder/n)==h for n,h in old['artifacts'].items());rows.append(old);continue
        if (folder/'raw_queries.npz').exists():raise RuntimeError('Forward capture exists without completion; preserve and replay postprocessing, never repeat the forward')
        if psutil.virtual_memory().available<2*2**30:raise RuntimeError('Original2GiB runtime guard')
        if model is None:model=Mask2FormerForUniversalSegmentation.from_pretrained(p['model'],local_files_only=True).cpu().eval();assert all(t.device.type=='cpu' and t.dtype==torch.float32 for t in model.parameters())
        folder.mkdir(parents=True,exist_ok=True);observed={};count=[0]
        def before(module,args,kwargs):
            tensor=kwargs['pixel_values'];assert list(tensor.shape)==[1,3,384,512] and tensor.device.type=='cpu';observed.update(actual_input_tensor_sha256=hashlib.sha256(tensor.detach().numpy().tobytes()).hexdigest(),actual_input_tensor_shape=list(tensor.shape))
        def after(module,args,output):
            count[0]+=1;assert count[0]==1;classes=output.class_queries_logits[0].detach().float().cpu().numpy();masks=output.masks_queries_logits[0].detach().float().cpu().numpy();assert classes.shape==(100,8) and masks.shape==(100,96,128);np.savez_compressed(folder/'raw_queries.npz',class_logits=classes,mask_logits=masks);save(folder/'forward_observation.json',observed|{'new_CPU_forward_observed':True,'model_sha256':p['model_sha256'],'all_queries_captured_before_original_postprocessing':True})
        first=model.register_forward_pre_hook(before,with_kwargs=True);second=model.register_forward_hook(after)
        try:ids,segs=scope['predict'](model,processor,Image.open(r['input']).convert('RGB'),[384,512],'cpu')
        finally:first.remove();second.remove()
        assert count[0]==1;fresh+=1;rgb=np.stack([ids%256,ids//256%256,ids//65536%256],-1).astype('uint8');Image.fromarray(rgb).save(folder/'panoptic.png');row=r|observed|{'folder':str(folder),'model_sha256':p['model_sha256'],'actual_new_CPU_forward':True,'segments_info':segs,'stock_PAL_instances':sum(s['category_id']==1 for s in segs),'source_original_function_not_output_rewritten':True,'artifacts':{n:sha(folder/n) for n in ['raw_queries.npz','panoptic.png','forward_observation.json']}};save(complete,row);rows.append(row);save(OUT/'progress.json',{'rows':rows,'new_forwards_this_execution':fresh});print({'case':r['case'],'PAL_instances':row['stock_PAL_instances'],'all_class_segments':len(segs),'new_CPU':True},flush=True)
    assert len(rows)==4 and all(sha(f)==h for f,h in p['protected'].items());save(OUT/'generation_completed.json',{'status':'complete_actual_registered_recipe_CPU_not_instance_acceptance','rows':rows,'actual_new_CPU_forwards':4,'fresh_this_execution':fresh,'real_sources':2,'plain_controls':2,'raw_queries_preserved':400,'GPU_calls':0,'new_training':False,'new_Agent_runs':0,'adopted':False});print({'new_CPU':4,'actual_recipe_complete':True,'not_instance_acceptance':True},flush=True)
if __name__=='__main__':prepare() if sys.argv[1]=='prepare' else run()
