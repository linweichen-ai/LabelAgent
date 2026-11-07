"""Paired same-input CPU proposal trial of existing retained 6820 weights."""
import os
os.environ['CUDA_VISIBLE_DEVICES']='-1'
os.environ['HF_HUB_OFFLINE']='1'
import hashlib,inspect,json,sys,time
from pathlib import Path
from run_rgb_layers import ROOT,read,save,sha
from audit_m2f_frozen_queries import OUT as AUDIT,PACK

OUT=ROOT/'outputs/labelagent_retained_teacher_20261107'
PROTOCOL=ROOT/'task_workspaces/mask2former_other_core_local_probe_20261107/protocol.json'


def prepare():
    if OUT.exists():raise FileExistsError('Preserve immutable teacher comparisons')
    old=read(AUDIT/'frozen_plan.json');p=read(PROTOCOL);model=Path(p['checkpoint']);assert sha(model/'model.safetensors')==p['checkpoint_sha256']=='6820dd6ef5cb5557550e7877c2e7f904bfcd504353180f9c01c505c5ce0ed2e2';assert read(model/'config.json')==read(PACK/'model/config.json');assert read(model/'preprocessor_config.json')==read(PACK/'model/preprocessor_config.json')
    OUT.mkdir();protected={str(PROTOCOL):sha(PROTOCOL),str(model/'model.safetensors'):sha(model/'model.safetensors'),str(model/'config.json'):sha(model/'config.json'),str(model/'preprocessor_config.json'):sha(model/'preprocessor_config.json'),**old['protected']}
    save(OUT/'frozen_plan.json',{'driver_sha256':sha(__file__),'model':str(model),'candidate_sha256':p['checkpoint_sha256'],'old_model_sha256':'d11c2e7f86cd5189605b857c4697e55a5caead0a6888b52b2f5ff814150f5291','views':old['views'],'protected':protected,'same_model_source_sha256':p['transformers_sha256'],'same_semantic_configs':True,'processor_used':str(PACK/'model'),'input_height_width':[448,576],'only_weights_change':True,'CPU_threads':2,'initial_available_RAM_GiB_min':3,'running_available_RAM_GiB_min':2,'max_new_forwards':10,'new_training':False,'no_shared_service_changes':True,'checkpoint_selection_for_other_training_unchanged':True,'pending_training_checkpoints_not_used':True,'candidate_adopted':False})
    print({'retained_weight_verified':True,'max_CPU_forwards':10},flush=True)


def run():
    if (OUT/'generation_completed.json').exists():raise FileExistsError('Already completed; never repeat successful forwards')
    plan=read(OUT/'frozen_plan.json');assert sha(__file__)==plan['driver_sha256'] and all(sha(p)==h for p,h in plan['protected'].items())
    cfg=read(ROOT/'task_workspaces/labelagent_point_ids_20261107/config.json');sys.path.insert(0,cfg['dependency_dir']);import torch,numpy as np,psutil
    from PIL import Image
    from transformers import Mask2FormerForUniversalSegmentation,Mask2FormerImageProcessor
    from m2f_query_replay import replay
    source=Path(inspect.getfile(Mask2FormerForUniversalSegmentation));assert sha(source)==plan['same_model_source_sha256']
    if psutil.virtual_memory().available<3*2**30:raise RuntimeError('3GiB startup RAM guard; no model loaded')
    torch.set_num_threads(2);torch.set_num_interop_threads(1);psutil.Process().nice(psutil.BELOW_NORMAL_PRIORITY_CLASS)
    model=Mask2FormerForUniversalSegmentation.from_pretrained(plan['model'],local_files_only=True).cpu().eval();assert all(p.device.type=='cpu' for p in model.parameters());processor=Mask2FormerImageProcessor.from_pretrained(PACK/'model',local_files_only=True);parameters=read(PACK/'inference_config.json')['postprocessing'];count=[0];hook=model.register_forward_pre_hook(lambda *_:count.__setitem__(0,count[0]+1));rows=[];started=time.monotonic()
    for view in plan['views']:
        folder=OUT/'query_diagnosis'/view['case']/view['view'];file=folder/'complete.json'
        if file.exists():
            row=read(file);assert all(sha(folder/k)==v for k,v in row['artifacts'].items());rows.append(row);continue
        if psutil.virtual_memory().available<2*2**30:raise RuntimeError('2GiB runtime RAM guard; successful forwards retained')
        if (OUT/'STOP').exists():raise RuntimeError('Explicit STOP')
        folder.mkdir(parents=True,exist_ok=True);source=Image.open(view['source']).convert('RGB');crop=source.crop(tuple(view['source_box']));native=crop.resize((576,448),Image.Resampling.BILINEAR);inputs=processor(images=native,do_resize=False,return_tensors='pt');input_hash=hashlib.sha256(inputs['pixel_values'].numpy().tobytes()).hexdigest();t=time.monotonic()
        with torch.inference_mode():prediction=model(**inputs)
        classes=prediction.class_queries_logits[0].float().numpy();masks=prediction.masks_queries_logits[0].float().numpy();assert classes.shape==(100,8);np.savez_compressed(folder/'raw_queries.npz',class_logits=classes,mask_logits=masks);ids,segs,queries,_=replay(classes,masks,(crop.height,crop.width),parameters);rgb=np.stack([ids%256,ids//256%256,ids//65536%256],-1).astype('uint8');Image.fromarray(rgb).save(folder/'panoptic.png');row={**view,'folder':str(folder),'old_folder':view['folder'],'input_sha256':sha(view['source']),'actual_input_tensor_sha256':input_hash,'same_PIL_resize_and_processor':True,'class_PAL_retained':sum(q['class_gate'] and q['top_class_id']==0 for q in queries),'postprocess_PAL_instances':sum(s['category_id']==1 for s in segs),'segments_info':segs,'seconds':time.monotonic()-t,'model_forward_calls':1,'artifacts':{n:sha(folder/n) for n in ['raw_queries.npz','panoptic.png']}};save(folder/'diagnosis.json',{'case':view['case'],'view':view['view'],'crop':view['source_box'],'input_sha256':sha(view['source']),'candidate_weight_sha256':plan['candidate_sha256'],'old_same_input_reused':True});save(file,row);rows.append(row);save(OUT/'progress.json',{'rows':rows,'fresh_forwards_this_execution':count[0]});print({'case':view['case'],'view':view['view'],'PAL_queries':row['class_PAL_retained'],'PAL_instances':row['postprocess_PAL_instances']},flush=True);del prediction,inputs
    hook.remove();assert len(rows)==10 and all(sha(p)==h for p,h in plan['protected'].items());save(OUT/'generation_completed.json',{'status':'complete_cpu_teacher_comparison_generation','rows':rows,'unique_model_forwards':10,'fresh_forwards_this_execution':count[0],'new_training':False,'candidate_adopted':False,'elapsed_seconds_this_execution':time.monotonic()-started,'old_model_and_incumbent_protected':True})


if __name__=='__main__':prepare() if sys.argv[1]=='prepare' else run()
