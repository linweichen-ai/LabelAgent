"""Post-freeze source stock coverage witness, never a proposal selector."""
import sys
from pathlib import Path
from prepare_fixed_grid_followup import BASE,OUT,COHORTS,ROOT
import numpy as np
from PIL import Image
from run_rgb_layers import read,save
from deliver_context_graph import sha
from projection_audit_core import frame_data,cameras

def main():
    target=OUT/'post_generation_registered_coverage.json'
    if target.exists():raise FileExistsError('Preserve completed diagnostic')
    frozen=read(OUT/'frozen_plan.json');assert read(OUT/'all48_original_postprocess_readback.json')['passed'] and all(sha(f)==h for f,h in frozen['protected'].items());cfg=read(ROOT/'task_workspaces/labelagent_point_ids_20261107/config.json');sys.path[:0]=[cfg['source_snapshot'],str(Path(cfg['source_snapshot'])/'tools'),cfg['dependency_dir']]
    import evaluate_projected_support as ev
    evaluator=Path(cfg['source_snapshot'])/'tools/evaluate_projected_support.py';assert sha(evaluator)=='93c5152169d04d43b25dbf4a4139b12676daf78750a7f059837121656bb644c2';protected={cfg['reference_annotations']:sha(cfg['reference_annotations']),cfg['reference_description']:sha(cfg['reference_description'])};dataset=Path(cfg['dataset']);manifest=read(dataset/'import_manifest.json');data=frame_data(dataset,manifest,76,ev);cams=cameras(dataset,manifest,ev);results=[]
    for cam,targets in [(0,[42,48]),(1,[47,49])]:
        case=f'camera{cam}_frame76';indices,pixels=ev.project_visible(data['xyz'],cams[cam]);owners=data['owner'][indices]
        for oid in targets:
            index=next(i for i,b in enumerate(data['boxes']) if b['obj_id']==oid);points=pixels[owners==index+1];models=[]
            for cohort,folder in COHORTS:
                p=read(folder/'frozen_plan.json');g=read(folder/'generation_completed.json')
                for model in [m['name'] for m in p['models']]:
                    covered=np.zeros(len(points),bool);views=[]
                    for r in g['rows']:
                        if r['case']!=case or r['model']!=model:continue
                        x1,y1,x2,y2=r['observed_source_box'];inside=(points[:,0]>=x1)&(points[:,0]<x2)&(points[:,1]>=y1)&(points[:,1]<y2);ii=np.flatnonzero(inside);local=points[inside]-np.array(r['source_virtual_box'][:2]);rgb=np.asarray(Image.open(Path(r['folder'])/'panoptic.png')).astype('uint32');ids=rgb[:,:,0]+256*rgb[:,:,1]+65536*rgb[:,:,2];assigned=ids[local[:,1],local[:,0]];pal=[s['id'] for s in r['segments_info'] if s['category_id']==1];hits=np.isin(assigned,pal);covered[ii]|=hits;views.append({'view':r['source_view'],'visible_reference_points_inside_observed_RGB':int(inside.sum()),'stock_PAL_hits':int(hits.sum()),'stock_PAL_instance_count':len(pal),'per_stock_PAL_reference_hits':[{'id':s,'hits':int((assigned==s).sum())} for s in pal]})
                    models.append({'cohort':cohort,'model':model,'reference_points':len(points),'points_in_any_stock_PAL_output':int(covered.sum()),'views':views})
            results.append({'case':case,'object_id':oid,'models':models})
    assert all(sha(f)==h for f,h in protected.items());result={'status':'complete_post_freeze_all_stock_predictions_coverage_diagnostic','cases':results,'source_point_plan_already_frozen_sha256':sha(OUT/'frozen_plan.json'),'GT_and_evaluator_unchanged':protected,'evaluator_sha256':sha(evaluator),'not_instance_acceptance_or_dense_human_truth':True,'no_diagnostic_point_or_segment_export_for_new_proposals':True,'new_model_calls':0,'adopted':False};save(target,result);print([(r['case'],r['object_id'],[(m['model'],m['points_in_any_stock_PAL_output'],m['reference_points']) for m in r['models']]) for r in results],flush=True)

if __name__=='__main__':main()
