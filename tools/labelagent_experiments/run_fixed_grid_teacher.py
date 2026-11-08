"""Full source frames and fixed quadrants repair the ROI-cohort blind spot.

The grid and sampling rules do not depend on reference coordinates. Border
padding is explicit, never invented scene content. Any future SAM output must
retain every native foreground pixel in observed source coordinates or be
rejected wholesale. Original final acceptance criteria remain unchanged.
"""
import hashlib,shutil,sys
from pathlib import Path
import numpy as np
from PIL import Image
from run_rgb_layers import ROOT,read,save,sha
import fixed_grid_cpu_engine as engine
OUT=engine.OUT
def prepare():
    if OUT.exists():raise FileExistsError('Preserve whole-frame fixed-grid protocol')
    if shutil.disk_usage(OUT.parent).free<512*2**20:raise RuntimeError('Full24-query-output reserve')
    previous=ROOT/'outputs/labelagent_zoom_teacher_20261108/frozen_plan.json';old=read(previous);protected={str(previous):sha(previous),str(Path(__file__)):sha(__file__),str(Path(engine.__file__)):sha(engine.__file__),str(Path(__file__).with_name('run_zoom_teacher.py')):sha(Path(__file__).with_name('run_zoom_teacher.py')),str(Path(__file__).with_name('m2f_query_replay.py')):sha(Path(__file__).with_name('m2f_query_replay.py'))};models=old['models'];processor=Path(old['processor_used'])
    for model in models:
        path=Path(model['path']);assert sha(path/'model.safetensors')==model['sha256']
        for name in ['model.safetensors','config.json','preprocessor_config.json']:protected[str(path/name)]=sha(path/name)
        assert read(path/'config.json')==read(processor/'config.json') and read(path/'preprocessor_config.json')==read(processor/'preprocessor_config.json')
    OUT.mkdir();views=[];pad=32;halo=64
    for cam in [0,1]:
        case=f'camera{cam}_frame76';original=ROOT/'outputs/labelagent_context_comparison_20261105/inputs'/(case+'.png');photo=Image.open(original).convert('RGB');w,h=photo.size;assert (w,h)==(1920,1536);protected[str(original)]=sha(original);canvas=Image.new('RGB',(w+2*pad,h+2*pad),(127,)*3);canvas.paste(photo,(pad,pad));folder=OUT/'inputs'/case;folder.mkdir(parents=True);canonical=folder/'explicit_padded_source.png';canvas.save(canonical);protected[str(canonical)]=sha(canonical);np.testing.assert_array_equal(np.asarray(canvas)[pad:pad+h,pad:pad+w],np.asarray(photo))
        cores=[('whole',[0,0,w,h]),('q1',[0,0,w//2,h//2]),('q2',[w//2,0,w,h//2]),('q3',[0,h//2,w//2,h]),('q4',[w//2,h//2,w,h])]
        for name,core in cores:
            if name=='whole':virtual=[-pad,-pad,w+pad,h+pad]
            else:
                x1,y1,x2,y2=core;virtual=[max(-pad,x1-halo),max(-pad,y1-halo),min(w+pad,x2+halo),min(h+pad,y2+halo)]
            box=[virtual[0]+pad,virtual[1]+pad,virtual[2]+pad,virtual[3]+pad];observed=[max(0,virtual[0]),max(0,virtual[1]),min(w,virtual[2]),min(h,virtual[3])];real_local=[observed[0]-virtual[0],observed[1]-virtual[1],observed[2]-virtual[0],observed[3]-virtual[1]];file=folder/(name+'.png');crop=canvas.crop(tuple(box));crop.save(file);protected[str(file)]=sha(file);views.append({'teacher_key':case+'_'+name,'case':case,'source_view':name,'source_image':str(canonical),'original_source_image':str(original),'original_source_sha256':sha(original),'input':str(file),'box':box,'source_ROI_size':[crop.height,crop.width],'source_original_core':core,'source_virtual_box':virtual,'observed_source_box':observed,'observed_region_in_input':real_local,'explicit_frame_padding':pad,'real_internal_tile_halo':halo,'is_synthetic_control':False,'sampling_does_not_read_references':True})
    for name,value in [('gray',127),('white',255)]:
        folder=OUT/'controls'/name;folder.mkdir(parents=True);canvas=Image.new('RGB',(1984,1600),(127,)*3);canvas.paste(Image.new('RGB',(1920,1536),(value,)*3),(pad,pad));file=folder/'input.png';canvas.save(file);protected[str(file)]=sha(file);views.append({'teacher_key':'control_'+name,'case':'synthetic_control_'+name,'source_view':'whole','source_image':str(file),'input':str(file),'box':[0,0,1984,1600],'source_ROI_size':[1600,1984],'observed_region_in_input':[pad,pad,1920+pad,1536+pad],'explicit_frame_padding':pad,'is_synthetic_control':True,'known_no_real_object':True,'never_a_real_case_or_mask_acceptance_result':True})
    assert len(views)==12
    plan={'models':models,'views':views,'protected':protected,'model_source_sha256':old['model_source_sha256'],'processor_used':str(processor),'input_height_width':[448,576],'postprocessing':old['postprocessing'],'max_new_CPU_forwards':24,'real_scene_forwards':20,'synthetic_control_forwards':4,'CPU_threads':2,'startup_available_RAM_GiB_min':3,'running_available_RAM_GiB_min':2,'no_shared_service_or_GPU_changes':True,'new_training':False,'no_GT_pose_calibration_depth_or_acceptance_area_for_grid_or_followup':True,'protocol_changes_are_joint_not_single_factor_ablation':['whole-frame and fixed quadrant coverage','64px real context halo at interior seams','32px explicit gray padding at physical frame boundary'],'all_original_final_gates_unchanged':True,'Agent_round_limit':3,'new_Agent_runs':0,'adopted':False,'followup_frozen_before_CPU':{'stock_segment_ranking':'one highest score accepted PAL stock segment per model/source view, score then ID ties; gray/white controls reported separately','source_point_rules':['maximum-distance interior on observed source pixels','maximum-distance source point in lowest third of accepted stock PAL bounding height, sampling only; complete query/output never trimmed'],'source_geometry_guards':'both original stock panoptics must predict PAL at each source point; native frame input border test unchanged; only source-observed points, no point on synthetic padding; source selection round-robin across fixed tiles then source cue policies, maximum32 unique point reviews','review':'original neutral dual unmarked source photographs at129/385, same model/prompt/schema/256+256 tokens; original matched controls and full single-base/extent gates still required','max_unique_SAM_point_candidates':16,'paired_SAM_arms':'same observed unmarked RGB and point, original source crop vs explicit padded input; at most32 one-point requests across both arms, original source model selection','foreground_mapping_requirement':'Save full native padded bitmap first. Reject any foreground on synthetic padding; inverse-map every native foreground pixel into the observed original frame with identical foreground count. Never cut a foreground pixel to fit frame/area/budget. Require original native input-border, points, area, person, visibility, purity, independent matching and whole-base semantics.'}}
    save(OUT/'frozen_plan.json',plan);save(OUT/'prepared.json',{'status':'prepared_not_executed','planned_CPU_forwards':24,'real_scene_forwards':20,'plain_control_forwards':4,'new_model_calls':0,'fixed_grid_covers_whole_original_frames':True,'all_original_criteria_unchanged':True});print({'prepared_real_views':10,'synthetic_views':2,'trained_weights':2,'max_new_CPU_forwards':24,'GPU_calls':0,'reference_inputs':False},flush=True)
if __name__=='__main__':prepare() if sys.argv[1]=='prepare' else engine.run()
