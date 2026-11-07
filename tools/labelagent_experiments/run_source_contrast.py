"""Matched source photometric inputs; original coordinates/full outputs preserved."""
import shutil,sys
from pathlib import Path
import cv2,numpy as np
from PIL import Image
from run_rgb_layers import ROOT,read,save,sha
import run_trained_point_sam as sam
OUT=ROOT/'outputs/labelagent_source_contrast_20261108'

def prepare():
    if OUT.exists():raise FileExistsError('Preserve fixed photometric source trial')
    if shutil.disk_usage(ROOT).free<96*2**20:raise RuntimeError('Original96MiB preparation reserve')
    previous=ROOT/'outputs/labelagent_two_sided_layers_20261108/SAM/generation_completed.json';base=read(previous);rows=[];protected={str(previous):sha(previous),str(Path(__file__)):sha(__file__),str(Path(sam.__file__)):sha(sam.__file__)};OUT.mkdir()
    for policy in ['source_LAB_CLAHE_clip2_grid8','source_unsharp_sigma1_amount1']:
        for original in base['rows']:
            source=Image.open(original['input']).convert('RGB');a=np.asarray(source)
            if policy.startswith('source_LAB'):
                lab=cv2.cvtColor(a,cv2.COLOR_RGB2LAB);lab[:,:,0]=cv2.createCLAHE(clipLimit=2.0,tileGridSize=(8,8)).apply(lab[:,:,0]);result=cv2.cvtColor(lab,cv2.COLOR_LAB2RGB)
            else:result=cv2.addWeighted(a,2.0,cv2.GaussianBlur(a,(0,0),1.0),-1.0,0)
            folder=OUT/'inputs'/policy/f"source{original['source_partition_index']:02d}";folder.mkdir(parents=True);file=folder/'actual_model_input.png';Image.fromarray(result).save(file);assert Image.open(file).size==source.size
            row={k:original[k] for k in ['case','box','prior','source_image','points','source_point','teacher_key','source_partition_index']};row.update(index=len(rows)+1,input=str(file),original_untransformed_input=original['input'],original_baseline_folder=original['folder'],photometric_policy=policy,no_geometric_change=True,original_source_selected_rule=True,exact_previous_capture=None);row['sam_request_sha256']=sam.key(row);rows.append(row)
            for f in [str(file),original['input'],original['source_image']]:protected[f]=sha(f)
    wpath=ROOT/'outputs/labelagent_registry_zoom_pool_20261108/SAM_weights_identity.json';w=read(wpath);weights=Path(w['path']);assert weights.stat().st_size==w['bytes'] and weights.stat().st_mtime_ns==w['mtime_ns'];protected[str(wpath)]=sha(wpath)
    save(OUT/'frozen_plan.json',{'rows':rows,'protected':protected,'source_sha256':sam.SOURCE_SHA,'same_original_five_source_points_and_image_geometry':True,'all_original_untransformed_images_and_baseline_masks_preserved':True,'input_pixels_photometrically_transformed_not_original_photo':True,'no_reference_for_inputs':True,'no_mask_prior':True,'no_output_mask_clipping':True,'policy_grid_fixed_before_inference':True,'opencv_version':cv2.__version__,'max_new_calls':6,'original_selected_index_required':True,'source_neutral_evidence_is_previous_original_RGB_not_newly_confirmed_transformed_images':True,'same_weight_stat_as_previous_stream_hash_not_fresh_weight_hash':True,'new_Agent_rounds':0,'Agent_round_limit':3,'all_original_acceptance_gates_required':True});print({'source_images':3,'fixed_photometric_arms':2,'max_new_SAM_calls':6,'new_calls_this_stage':0},flush=True)

def run():
    w=read(ROOT/'outputs/labelagent_registry_zoom_pool_20261108/SAM_weights_identity.json');p=Path(w['path']);assert p.stat().st_size==w['bytes'] and p.stat().st_mtime_ns==w['mtime_ns'];sam.OUT=OUT;sam.run()

if __name__=='__main__':{'prepare':prepare,'run':run}[sys.argv[1]]()
