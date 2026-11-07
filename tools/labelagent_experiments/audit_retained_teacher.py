"""Verify retained-weight outputs against stock postprocessing, then old gates."""
import hashlib,importlib.util,json,sys
from pathlib import Path
import numpy as np,torch
from PIL import Image
import audit_m2f_frozen_queries as core
from trial_retained_m2f import OUT as TRIAL
from run_rgb_layers import ROOT,read,save,sha

core.OUT=TRIAL/'audit';core.OLD=TRIAL/'query_diagnosis'


def prepare():
    completed=read(TRIAL/'generation_completed.json');assert completed['unique_model_forwards']==10;plan=read(TRIAL/'frozen_plan.json');torch.set_num_threads(2);p=core.PACK/'calibrated_postprocess.py';spec=importlib.util.spec_from_file_location('independent_frozen_postprocessor',p);m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);checks=[]
    for row in completed['rows']:
        folder=Path(row['folder']);a=np.load(folder/'raw_queries.npz',allow_pickle=False);x1,y1,x2,y2=row['source_box'];ids,segs=m.postprocess({'class_queries_logits':torch.tensor(a['class_logits'])[None],'masks_queries_logits':torch.tensor(a['mask_logits'])[None]},(y2-y1,x2-x1),read(core.PACK/'inference_config.json')['postprocessing']);rgb=np.asarray(Image.open(folder/'panoptic.png')).astype('uint32');stored=rgb[:,:,0]+rgb[:,:,1]*256+rgb[:,:,2]*65536;assert np.array_equal(ids,stored) and segs==row['segments_info'];checks.append({'case':row['case'],'view':row['view'],'stock_panoptic_pixels_and_segments_exact':True})
    save(TRIAL/'independent_stock_postprocess.json',{'passed':True,'views':checks,'original_postprocessor_sha256':sha(p),'new_model_forwards':0});core.prepare();f=core.OUT/'frozen_plan.json';p=read(f);p.update(actual_cache_model_sha256=plan['candidate_sha256'],actual_wrapper_sha256=sha(__file__),cache_generation_forward_count=10,new_forwards_during_audit=0);p['protected'].update({str(TRIAL/'frozen_plan.json'):sha(TRIAL/'frozen_plan.json'),str(TRIAL/'generation_completed.json'):sha(TRIAL/'generation_completed.json'),str(Path(plan['model'])/'model.safetensors'):plan['candidate_sha256'],str(TRIAL/'independent_stock_postprocess.json'):sha(TRIAL/'independent_stock_postprocess.json')});save(f,p)


if __name__=='__main__':
    if sys.argv[1]=='prepare':prepare()
    else:
        p=read(core.OUT/'frozen_plan.json');assert sha(__file__)==p['actual_wrapper_sha256'];core.run()
