"""Post-generation stock-class witness; never exported as source proposals."""
import sys
from pathlib import Path
import numpy as np
from run_rgb_layers import ROOT,read,save,sha
from projection_audit_core import frame_data,cameras
import run_fixed_grid_teacher as grid
import deliver_anchor_gap_D as delivery
def main():
    plan=read(grid.OUT/'frozen_plan.json');g=read(grid.OUT/'generation_completed.json');assert g['new_CPU_model_forwards']==24;cfg=read(ROOT/'task_workspaces/labelagent_point_ids_20261107/config.json');sys.path[:0]=[cfg['source_snapshot'],str(Path(cfg['source_snapshot'])/'tools'),cfg['dependency_dir']]
    import evaluate_projected_support as ev
    from PIL import Image
    assert sha(Path(cfg['source_snapshot'])/'tools/evaluate_projected_support.py')=='93c5152169d04d43b25dbf4a4139b12676daf78750a7f059837121656bb644c2';dataset=Path(cfg['dataset']);manifest=read(dataset/'import_manifest.json');data=frame_data(dataset,manifest,76,ev);cams=cameras(dataset,manifest,ev);results=[]
    for cam,targets in [(0,[42,48]),(1,[47,49])]:
        case=f'camera{cam}_frame76';indices,pixels=ev.project_visible(data['xyz'],cams[cam]);owners=data['owner'][indices]
        for oid in targets:
            index=next(i for i,b in enumerate(data['boxes']) if b['obj_id']==oid);points=pixels[owners==index+1];models=[]
            for model in ['old','retained']:
                covered=np.zeros(len(points),bool);details=[]
                for row in g['rows']:
                    if row['case']!=case or row['model']!=model:continue
                    x1,y1,x2,y2=row['observed_source_box'];inside=(points[:,0]>=x1)&(points[:,0]<x2)&(points[:,1]>=y1)&(points[:,1]<y2);indices=np.flatnonzero(inside);local=points[inside]+32-np.array(row['box'][:2]);rgb=np.asarray(Image.open(Path(row['folder'])/'panoptic.png')).astype('uint32');ids=rgb[:,:,0]+256*rgb[:,:,1]+65536*rgb[:,:,2];assigned=ids[local[:,1],local[:,0]];pal=[s['id'] for s in row['segments_info'] if s['category_id']==1];hits=np.isin(assigned,pal);covered[indices]|=hits;details.append({'view':row['source_view'],'reference_points_inside_observed_RGB':int(inside.sum()),'stock_PAL_hits':int(hits.sum()),'stock_PAL_instance_count':row['stock_PAL_instances']})
                models.append({'model':model,'original_visible_reference_points':len(points),'points_in_any_stock_PAL_output':int(covered.sum()),'views':details})
            results.append({'case':case,'object_id':oid,'models':models})
    result={'status':'complete_diagnostic_after_all_grid_queries_frozen','cases':results,'generation_sha256':sha(grid.OUT/'generation_completed.json'),'no_generation_coordinate_or_query_selection_exports':True,'source_followup_rules_were_frozen_before_this_diagnostic':True,'camera1_left_targets_stock_PAL_coverage_still_zero':all(m['points_in_any_stock_PAL_output']==0 for c in results if c['case']=='camera1_frame76' for m in c['models']),'stock_count_one_in_camera1_q1_is_not_recovery_of_left_targets':True,'interpretation':'Full grid repairs source-image coverage but not trained PAL proposals at the two left targets. Original source predictions and postprocessing stay unchanged; do not confuse a tile-wide PAL count with target recovery. Reference projections are the old sparse diagnostic, not dense human truth.'}
    target=delivery.FULL/'diagnostics/fixed_grid_stock_coverage.json';save(target,result);copy=target.with_name('audit_fixed_grid_stock_coverage.py');copy.write_bytes(Path(__file__).read_bytes());save(delivery.C_DISPLAY/'固定网格源候选覆盖诊断.json',result);save(delivery.base.C/'fixed_grid_stock_coverage_20261108.json',result)
    note='<p>补充核验：新网格虽覆盖了整幅原图，camera1左侧两例在两套权重的原PAL输出中仍为0命中；左上源框的PAL总数1不等于这两例恢复。<a href="固定网格源候选覆盖诊断.json">查看事后覆盖核验</a></p>';index=delivery.C_DISPLAY/'index.html';text=index.read_text('utf-8');assert note not in text;index.write_text(text.replace('</h1>','</h1>'+note,1),encoding='utf-8');save(delivery.C_DISPLAY/'交接SHA256.json',{'files':[{'path':p.relative_to(delivery.C_DISPLAY).as_posix(),'sha256':sha(p),'bytes':p.stat().st_size} for p in delivery.C_DISPLAY.rglob('*') if p.is_file() and p.name!='交接SHA256.json']});save(delivery.FULL/'交接SHA256.json',{'files':[{'path':p.relative_to(delivery.FULL).as_posix(),'sha256':sha(p),'bytes':p.stat().st_size} for p in delivery.FULL.rglob('*') if p.is_file() and p.name!='交接SHA256.json']});print({'camera1_left_still_zero':result['camera1_left_targets_stock_PAL_coverage_still_zero'],'cases':[(c['case'],c['object_id'],[m['points_in_any_stock_PAL_output'] for m in c['models']]) for c in results]},flush=True)
if __name__=='__main__':main()
