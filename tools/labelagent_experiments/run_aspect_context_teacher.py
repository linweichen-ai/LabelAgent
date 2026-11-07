"""Expand source RGB context to model aspect without shrinking the original ROI."""
import copy,datetime,math,shutil,sys
from pathlib import Path
import numpy as np
from PIL import Image
from run_rgb_layers import ROOT,read,save,sha
import run_zoom_teacher as runner
OUT=Path('D:/AI/Codex/Home/visualizations/2026/11/04/01a1a56c-0d6f-7d61-8326-d3bbd344d2df/labelagent_aspect_context_teacher_20261108')

def expanded(box,size):
    x1,y1,x2,y2=box;width,height=size;w=x2-x1;h=y2-y1;neww=max(w,math.ceil(h*576/448));newh=max(h,math.ceil(w*448/576))
    if neww>width or newh>height:raise ValueError('No native-context expansion fits; do not synthesize pixels or shrink original ROI')
    left=max(0,min(width-neww,(x1+x2-neww)//2));top=max(0,min(height-newh,(y1+y2-newh)//2));result=[left,top,left+neww,top+newh];assert left<=x1 and top<=y1 and result[2]>=x2 and result[3]>=y2;return result

def prepare():
    if OUT.exists():raise FileExistsError('Preserve frozen native-aspect source trial')
    if shutil.disk_usage(OUT.parent).free<192*2**20:raise RuntimeError('Original192MiB raw-query output reserve')
    original=ROOT/'outputs/labelagent_zoom_teacher_20261108/frozen_plan.json';p=copy.deepcopy(read(original));assert all(sha(f)==h for f,h in p['protected'].items());p['protected'][str(original)]=sha(original);p['protected'][str(Path(__file__))]=sha(__file__);p['at']=datetime.datetime.now().astimezone().isoformat();OUT.mkdir();views=[]
    for old in p['views']:
        photo=Image.open(old['source_image']).convert('RGB');box=expanded(old['box'],photo.size);folder=OUT/'inputs'/old['teacher_key'];folder.mkdir(parents=True);file=folder/'actual_source_context.png';photo.crop(tuple(box)).save(file);np.testing.assert_array_equal(np.asarray(Image.open(file).convert('RGB')),np.asarray(photo)[box[1]:box[3],box[0]:box[2]])
        row={k:v for k,v in old.items() if k not in ['prior','logits']};row.update(input=str(file),box=box,matched_previous_ROI_box=old['box'],matched_previous_ROI_input=old['input'],source_ROI_size=[box[3]-box[1],box[2]-box[0]],original_ROI_entirely_contained=True,new_source_pixels_only_no_padding_or_reconstruction=True,geometric_shape_change_ratio=(box[2]-box[0])/(box[3]-box[1])/(576/448));views.append(row);p['protected'][str(file)]=sha(file)
    p['views']=views;p['source_context_policy']='Keep original ROI entirely, expand only real RGB context around same center to model576:448 ratio, bounded by full source image';p['same_original_weights_CPU_FP32_input_shape_postprocessor_and_gates']=True;p['matched_previous_six_direct_stretch_views']=str(original);p['no_reference_or_budget_for_inputs']=True;p['new_Agent_rounds']=0;p['new_model_calls_this_prepare']=0;save(OUT/'frozen_plan.json',p);print({'native_context_inputs':len(views),'max_new_CPU_forwards':12,'new_calls_this_stage':0,'output':str(OUT)},flush=True)

if __name__=='__main__':
    if sys.argv[1]=='prepare':prepare()
    else:runner.OUT=OUT;runner.run()
