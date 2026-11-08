"""Post-generation reference witness for omitted source-cohort regions.

Reference coordinates are diagnostic only and are not exported as proposals.
The next generator must cover the whole frame on a fixed grid.
"""
import sys
from pathlib import Path
import numpy as np
from run_rgb_layers import ROOT,read,save,sha
import finish_anchor_gap_report as report
from projection_audit_core import frame_data,cameras
def main():
    cfg=read(ROOT/'task_workspaces/labelagent_point_ids_20261107/config.json');sys.path[:0]=[cfg['source_snapshot'],str(Path(cfg['source_snapshot'])/'tools'),cfg['dependency_dir']]
    import evaluate_projected_support as ev
    assert sha(Path(cfg['source_snapshot'])/'tools/evaluate_projected_support.py')=='93c5152169d04d43b25dbf4a4139b12676daf78750a7f059837121656bb644c2';dataset=Path(cfg['dataset']);manifest=read(dataset/'import_manifest.json');data=frame_data(dataset,manifest,76,ev);cams=cameras(dataset,manifest,ev);teacher=report.BASE/'labelagent_aspect_context_teacher_20261108/frozen_plan.json';plan=read(teacher);rows=[]
    for cam,targets in [(0,[42,48]),(1,[47,49])]:
        case=f'camera{cam}_frame76';indices,pix=ev.project_visible(data['xyz'],cams[cam]);owner=data['owner'][indices];views=[v for v in plan['views'] if v['case']==case]
        for oid in targets:
            i=next(i for i,b in enumerate(data['boxes']) if b['obj_id']==oid);points=pix[owner==i+1];inside=np.zeros(len(points),bool);detail=[]
            for v in views:
                x1,y1,x2,y2=v['box'];hit=(points[:,0]>=x1)&(points[:,0]<x2)&(points[:,1]>=y1)&(points[:,1]<y2);inside|=hit;detail.append({'teacher_key':v['teacher_key'],'source_box':v['box'],'visible_reference_points_inside':int(hit.sum())})
            rows.append({'case':case,'object_id':oid,'visible_reference_points':len(points),'points_inside_any_recent_six_teacher_ROIs':int(inside.sum()),'source_teacher_ROIs':detail,'point_span_xy_minmax_for_diagnostic_only':[points.min(0).tolist(),points.max(0).tolist()],'not_generation_inputs_or_dense_2D_truth':True})
    result={'status':'complete_post_generation_source_cohort_coverage_witness','source_teacher_plan_sha256':sha(teacher),'cases':rows,'camera1_left_two_targets_not_covered_by_recent_six_teacher_protocol':all(r['points_inside_any_recent_six_teacher_ROIs']==0 for r in rows if r['case']=='camera1_frame76'),'scope':'Recent six native-aspect/high-resolution source ROI protocols and dependent source mask graph/gap experiments; not the original full-frame baseline','next_source_sampling_rule':'Whole original frames and fixed four quadrants for both cameras, with declared real context and explicit boundary padding; no reference-dependent crop or seed selection','all_reference_coordinates_diagnostic_only':True};save(report.OUT/'source_cohort_coverage.json',result);print(result,flush=True)
if __name__=='__main__':main()
