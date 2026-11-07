"""Freeze unchanged neutral RGB review of unlabelled-sensor positive points."""
import hashlib,json
from pathlib import Path
from PIL import Image
from run_rgb_layers import ROOT,read,save,sha
from paired_source_cue import body
SENSOR=ROOT/'outputs/labelagent_unlabelled_height_bands_20261108';OUT=ROOT/'outputs/labelagent_height_review_20261108'

def main():
    if OUT.exists():raise FileExistsError('Preserve height-source review cohort')
    source=read(SENSOR/'completed.json');plan=read(SENSOR/'frozen_plan.json');assert all(sha(f)==h for f,h in plan['protected'].items());OUT.mkdir();protected={str(SENSOR/'completed.json'):sha(SENSOR/'completed.json'),str(SENSOR/'frozen_plan.json'):sha(SENSOR/'frozen_plan.json'),str(Path(__file__)):sha(__file__)};rows=[]
    for r in source['rows']:
        x,y=r['source_point'];photo=Image.open(r['source_image']).convert('RGB');bindings=[];folder=OUT/'candidates'/f"cue{r['index']:02d}";folder.mkdir(parents=True)
        for name,side in [('near',129),('context',385)]:
            half=side//2;box=[x-half,y-half,x+half+1,y+half+1];clip=[max(0,box[0]),max(0,box[1]),min(photo.width,box[2]),min(photo.height,box[3])];im=Image.new('RGB',(side,side),(127,127,127));im.paste(photo.crop(tuple(clip)),(clip[0]-box[0],clip[1]-box[1]));f=folder/(name+'.png');im.resize((512,512),Image.Resampling.LANCZOS).save(f);bindings.append({'image':str(f),'sha256':sha(f),'side':side,'source_canvas_box':box,'clipped_box':clip})
        request=body(*[b['image'] for b in bindings]);key=hashlib.sha256(json.dumps(request,sort_keys=True).encode()).hexdigest();file=OUT/'prepared_requests'/(key+'_PRIVATE.json');save(file,request);v={'new_point':r['points'][0],'source_point':r['source_point'],'bindings':bindings,'request_file':str(file),'request_sha256':key};rows.append({**r,'new_point_views':[v],'unreviewed_negative_is_hypothesis_only':True});protected[r['input']]=sha(r['input']);protected[r['source_image']]=sha(r['source_image'])
    model=read(ROOT/'outputs/labelagent_registry_zoom_pool_20261108/frozen_plan.json')['model_config_sha256'];save(OUT/'frozen_plan.json',{'rows':rows,'protected':protected,'model_config_sha256':model,'same_neutral_prompt_and_source_near_context_sizes':True,'no_reference_for_generation':True,'not_same_input_modality_as_RGB_baseline':True,'all_original_gates_unchanged':True,'new_Agent_runs':0,'Agent_round_limit':3});save(OUT/'prepared.json',{'rows':rows});print({'frozen_positive_views':len(rows),'new_model_calls':0},flush=True)

if __name__=='__main__':main()
