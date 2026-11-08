"""Same original CPU inference on full frames, fixed quadrants and plain controls."""
import os
os.environ['CUDA_VISIBLE_DEVICES']='-1'
os.environ['HF_HUB_OFFLINE']='1'
os.environ['TRANSFORMERS_OFFLINE']='1'
import datetime,gc,hashlib,inspect,json,shutil,sys,time
from pathlib import Path
from run_rgb_layers import ROOT,read,save,sha
OUT=Path('D:/AI/Codex/Home/visualizations/2026/11/04/01a1a56c-0d6f-7d61-8326-d3bbd344d2df/labelagent_fixed_grid_teacher_20261108')
PACK=ROOT/'outputs/mask2former_single_recovery_20261104/最新单模型候选包'
PARENTS=Path('D:/AI/Codex/Home/visualizations/2026/11/04/01a1a56c-0d6f-7d61-8326-d3bbd344d2df/labelagent_trained_points_20261108/frozen_plan.json')
RET=ROOT/'outputs/labelagent_retained_teacher_20261107'

def prepare():
    if OUT.exists():raise FileExistsError('Preserve zoom teacher study')
    if shutil.disk_usage(OUT.parent).free<192*2**20:raise RuntimeError('192MiB raw-query output reserve')
    previous=read(RET/'frozen_plan.json');models=[{'name':'old','path':str(PACK/'model'),'sha256':'d11c2e7f86cd5189605b857c4697e55a5caead0a6888b52b2f5ff814150f5291'},{'name':'retained','path':previous['model'],'sha256':previous['candidate_sha256']}];parents=read(PARENTS)['teachers'];config=read(PACK/'inference_config.json');assert config['input_height_width']==[448,576] and config['postprocessing']['promote_null_pallet'] is False;protected={str(PARENTS):sha(PARENTS),str(RET/'frozen_plan.json'):sha(RET/'frozen_plan.json'),str(RET/'generation_completed.json'):sha(RET/'generation_completed.json'),str(PACK/'inference_config.json'):sha(PACK/'inference_config.json'),str(Path(__file__)):sha(__file__),str(Path(__file__).with_name('m2f_query_replay.py')):sha(Path(__file__).with_name('m2f_query_replay.py'))};views=[]
    for model in models:
        path=Path(model['path']);assert sha(path/'model.safetensors')==model['sha256'];assert read(path/'config.json')==read(PACK/'model/config.json') and read(path/'preprocessor_config.json')==read(PACK/'model/preprocessor_config.json')
        for name in ['model.safetensors','config.json','preprocessor_config.json']:protected[str(path/name)]=sha(path/name)
    for p in parents:
        source=ROOT/'outputs/labelagent_context_comparison_20261105/inputs'/(p['case']+'.png');protected[str(source)]=sha(source);protected[p['input']]=sha(p['input']);old=ROOT/'outputs/labelagent_hybrid_repair_20261104/query_diagnosis'/p['case']/p['source_view'];meta=read(old/'diagnosis.json');old_box=meta['crop'];x1,y1,x2,y2=p['box'];views.append({**p,'source_image':str(source),'old_source_box':old_box,'source_ROI_size':[y2-y1,x2-x1],'original_fixed_input':[448,576],'vertical_object_scale_ratio_vs_old_view':(old_box[3]-old_box[1])/(y2-y1),'horizontal_object_scale_ratio_vs_old_view':(old_box[2]-old_box[0])/(x2-x1),'old_raw_view':str(old),'retained_raw_view':str(RET/'query_diagnosis'/p['case']/p['source_view'])})
    assert len(views)==6;OUT.mkdir();save(OUT/'frozen_plan.json',{'at':datetime.datetime.now().astimezone().isoformat(),'models':models,'views':views,'protected':protected,'model_source_sha256':previous['same_model_source_sha256'],'processor_used':str(PACK/'model'),'input_height_width':[448,576],'postprocessing':config['postprocessing'],'max_new_CPU_forwards':12,'CPU_threads':2,'startup_available_RAM_GiB_min':3,'running_available_RAM_GiB_min':2,'no_shared_service_or_GPU_changes':True,'new_training':False,'pending_checkpoints_not_used':True,'no_reference_or_budget_for_inputs':True,'old_quarter_views_are_historical_controls_not_new_forwards':True,'old_checkpoint_historical_backend_not_claimed_identical_to_current_CPU':True,'retained_historical_CPU_source_and_processor_verified':True,'followup_proposals_predeclared':'At most two accepted stock PAL segments per model/ROI ranked by original class score then segment ID, reject native ROI border before any reference-based choice; all raw queries and stock outputs retained','new_Agent_rounds':0,'Agent_round_limit':3,'all_final_gates_unchanged':True});print({'frozen_native_ROIs':6,'trained_weights':2,'max_new_CPU_forwards':12,'new_training':False,'no_GPU_calls':True},flush=True)

def run():
    if (OUT/'generation_completed.json').exists():raise FileExistsError('Do not repeat successful CPU forwards')
    p=read(OUT/'frozen_plan.json');assert all(sha(f)==h for f,h in p['protected'].items());cfg=read(ROOT/'task_workspaces/labelagent_point_ids_20261107/config.json');sys.path.insert(0,cfg['dependency_dir']);tmp=OUT/'worker_tmp';tmp.mkdir(exist_ok=True);os.environ['TEMP']=os.environ['TMP']=str(tmp)
    import numpy as np,psutil,torch
    from PIL import Image
    from transformers import Mask2FormerForUniversalSegmentation,Mask2FormerImageProcessor
    from m2f_query_replay import replay
    if psutil.virtual_memory().available<3*2**30:raise RuntimeError('3GiB startup RAM guard; no model loaded')
    assert sha(Path(inspect.getfile(Mask2FormerForUniversalSegmentation)))==p['model_source_sha256'];torch.set_num_threads(2);torch.set_num_interop_threads(1);psutil.Process().nice(psutil.BELOW_NORMAL_PRIORITY_CLASS);processor=Mask2FormerImageProcessor.from_pretrained(p['processor_used'],local_files_only=True);rows=[];fresh=0;started=time.monotonic();cache=read(RET/'generation_completed.json')['rows']
    for model_plan in p['models']:
        model=None;counter=[0]
        for view in p['views']:
            folder=OUT/'queries'/model_plan['name']/view['teacher_key'];file=folder/'complete.json'
            if file.exists():
                old=read(file);assert old['model_sha256']==model_plan['sha256'] and all(sha(folder/n)==h for n,h in old['artifacts'].items());rows.append(old);continue
            if (OUT/'STOP').exists():raise RuntimeError('Explicit STOP')
            if psutil.virtual_memory().available<2*2**30:raise RuntimeError('2GiB runtime RAM guard; completed outputs retained')
            if shutil.disk_usage(OUT).free<48*2**20:raise RuntimeError('Raw query output reserve; completed forwards retained')
            crop=Image.open(view['input']).convert('RGB');source=Image.open(view['source_image']).convert('RGB');np.testing.assert_array_equal(np.asarray(source.crop(tuple(view['box']))),np.asarray(crop));native=crop.resize((576,448),Image.Resampling.BILINEAR);inputs=processor(images=native,do_resize=False,return_tensors='pt');tensor_hash=hashlib.sha256(inputs['pixel_values'].numpy().tobytes()).hexdigest();assert tuple(inputs['pixel_values'].shape)==(1,3,448,576);folder.mkdir(parents=True,exist_ok=True);found=None
            if model_plan['name']=='retained':
                for old in cache:
                    if old['actual_input_tensor_sha256']==tensor_hash:
                        previous=Path(old['folder']);assert all(sha(previous/n)==h for n,h in old['artifacts'].items());found=previous/'raw_queries.npz';break
            t=time.monotonic()
            if found:
                shutil.copy2(found,folder/'raw_queries.npz');a=np.load(folder/'raw_queries.npz',allow_pickle=False);classes=a['class_logits'];masks=a['mask_logits'];new_call=False
            else:
                if model is None:
                    model=Mask2FormerForUniversalSegmentation.from_pretrained(model_plan['path'],local_files_only=True).cpu().eval();assert all(x.device.type=='cpu' and x.dtype==torch.float32 for x in model.parameters());hook=model.register_forward_pre_hook(lambda *_:counter.__setitem__(0,counter[0]+1))
                before=counter[0]
                with torch.inference_mode():prediction=model(**inputs)
                assert counter[0]==before+1;classes=prediction.class_queries_logits[0].float().numpy();masks=prediction.masks_queries_logits[0].float().numpy();assert classes.shape==(100,8) and masks.shape[0]==100;np.savez_compressed(folder/'raw_queries.npz',class_logits=classes,mask_logits=masks);new_call=True;fresh+=1
            ids,segs,queries,_=replay(classes,masks,(crop.height,crop.width),p['postprocessing']);rgb=np.stack([ids%256,ids//256%256,ids//65536%256],-1).astype('uint8');Image.fromarray(rgb).save(folder/'panoptic.png');save(folder/'query_class_records.json',queries);row={**view,'model':model_plan['name'],'model_sha256':model_plan['sha256'],'folder':str(folder),'actual_input_tensor_sha256':tensor_hash,'actual_input_tensor_shape':list(inputs['pixel_values'].shape),'source_ROI_sha256':sha(view['input']),'raw_mask_logits_shape':list(masks.shape),'segments_info':segs,'PAL_queries':sum(q['top_class_id']==0 and q['class_gate'] for q in queries),'stock_PAL_instances':sum(s['category_id']==1 for s in segs),'new_forward_call':new_call,'exact_tensor_cache_reuse':str(found) if found else None,'CPU_only_FP32':True,'seconds':time.monotonic()-t,'artifacts':{n:sha(folder/n) for n in ['raw_queries.npz','panoptic.png','query_class_records.json']}};save(file,row);rows.append(row);save(OUT/'progress.json',{'rows':rows,'fresh_CPU_forwards_this_execution':fresh});print({'model':row['model'],'ROI':view['teacher_key'],'PAL_queries':row['PAL_queries'],'stock_PAL_instances':row['stock_PAL_instances'],'cached':not new_call,'seconds':round(row['seconds'],2)},flush=True);del inputs,classes,masks,ids
            if not found:del prediction
            gc.collect()
        if model is not None:hook.remove();del model;gc.collect()
    assert len(rows)==24 and all(sha(f)==h for f,h in p['protected'].items());save(OUT/'generation_completed.json',{'status':'complete_CPU_zoom_teacher_raw_queries','rows':rows,'new_CPU_model_forwards':sum(r['new_forward_call'] for r in rows),'fresh_CPU_forwards_this_execution':fresh,'exact_tensor_cache_reuses':sum(not r['new_forward_call'] for r in rows),'new_training':False,'GPU_calls':0,'new_Agent_runs':0,'adopted':False,'seconds_this_execution':time.monotonic()-started});print({'completed_views':len(rows),'new_CPU_model_forwards':sum(r['new_forward_call'] for r in rows),'new_training':False},flush=True)

if __name__=='__main__':prepare() if sys.argv[1]=='prepare' else run()
