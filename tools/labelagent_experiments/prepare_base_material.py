"""Freeze source-only edge failures and material-band prompts before review."""
import datetime,hashlib,json,os,shutil,sys
from pathlib import Path
import numpy as np
from PIL import Image
from run_rgb_layers import ROOT,read,save,sha
from base_structure_points import propose as edges
from base_material_band import propose as material
from paired_source_cue import body
OUT=ROOT/'outputs/labelagent_base_material_20261108';OLD=ROOT/'outputs/labelagent_trained_partition_20261108'

def main():
    if OUT.exists():raise FileExistsError('Preserve source-only base study')
    if shutil.disk_usage(OUT.parent).free<96*2**20:raise RuntimeError('96MiB output reserve')
    original=read(OLD/'partition_plan.json');review=read(ROOT/'outputs/labelagent_trained_point_review_20261108/frozen_plan.json');rows=[];diagnostics=[];protected={str(OLD/'partition_plan.json'):sha(OLD/'partition_plan.json')}
    for name in ['base_structure_points.py','base_material_band.py','prepare_base_material.py','paired_source_cue.py']:protected[str(Path(__file__).with_name(name))]=sha(Path(__file__).with_name(name))
    OUT.mkdir()
    for r in original['rows']:
        photo=Image.open(r['input']).convert('RGB');parent=np.asarray(Image.open(r['prior']).convert('L'))>127;anchor=[r['points'][0]['x'],r['points'][0]['y']];edge,ed=edges(np.asarray(photo),parent,anchor);p,diag=material(np.asarray(photo),parent,anchor);diagnostics.append({'partition':r['index'],'edge_proposal':edge,'edge_diagnostics':ed,'material_proposal':p,'material_diagnostics':diag})
        for f in [r['input'],r['prior'],r['source_image']]:protected[f]=sha(f)
        if p is None:continue
        extra=p['points'];points=r['points']+extra
        if len({(q['x'],q['y']) for q in points})!=len(points):continue
        x1,y1,x2,y2=r['box'];folder=OUT/'inputs'/f"partition{r['index']:02d}";folder.mkdir(parents=True);source=Image.open(r['source_image']).convert('RGB');views=[]
        for i,q in enumerate(extra):
            ax,ay=q['x']+x1,q['y']+y1;bindings=[]
            for name,side in [('near',129),('context',385)]:
                half=side//2;box=[ax-half,ay-half,ax+half+1,ay+half+1];clip=[max(0,box[0]),max(0,box[1]),min(source.width,box[2]),min(source.height,box[3])];canvas=Image.new('RGB',(side,side),(127,127,127));canvas.paste(source.crop(tuple(clip)),(clip[0]-box[0],clip[1]-box[1]));file=folder/f'point{i+1}_{name}.png';canvas.resize((512,512),Image.Resampling.LANCZOS).save(file);bindings.append({'image':str(file),'sha256':sha(file),'source_canvas_box':box,'clipped_box':clip,'side':side})
            request=body(*[b['image'] for b in bindings]);k=hashlib.sha256(json.dumps(request,sort_keys=True).encode()).hexdigest();file=OUT/'prepared_requests'/(k+'_PRIVATE.json');save(file,request);views.append({'new_point':q,'source_point':[ax,ay],'bindings':bindings,'request_file':str(file),'request_sha256':k})
        rows.append({**r,'source_partition_index':r['index'],'original_two_points':r['points'],'points':points,'new_positive_points':extra,'source_edge_candidate':edge,'source_material_diagnostics':diag,'new_point_views':views,'not_human_truth':True})
    save(OUT/'frozen_plan.json',{'at':datetime.datetime.now().astimezone().isoformat(),'rows':rows,'diagnostics':diagnostics,'protected':protected,'model_config_sha256':review['model_config_sha256'],'actual_image_token_counts':[256,256],'logical_new_point_views':sum(len(r['new_point_views']) for r in rows),'max_unique_new_VLM_calls':len({v['request_sha256'] for r in rows for v in r['new_point_views']}),'each_new_point_must_pass_neutral_subject_review_before_SAM':True,'edge_failure_not_silently_replaced':True,'same_original_input_positive_negative_and_native_selection':True,'max_points_per_hypothesis':4,'no_reference_or_area_for_generation':True,'no_output_mask_clipping':True,'Agent_round_limit':3,'new_Agent_rounds':0});print({'proposed':len(rows),'point_views':sum(len(r['new_point_views']) for r in rows),'unique_payloads':len({v['request_sha256'] for r in rows for v in r['new_point_views']}),'edge_proposals':sum(d['edge_proposal'] is not None for d in diagnostics),'new_model_calls':0},flush=True)

if __name__=='__main__':main()
