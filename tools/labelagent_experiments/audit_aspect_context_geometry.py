"""Derive transform facts from actual crops; flag inherited diagnostic fields."""
import hashlib
from pathlib import Path
import numpy as np
from PIL import Image
from run_rgb_layers import ROOT,read,save,sha
OUT=Path('D:/AI/Codex/Home/visualizations/2026/11/04/01a1a56c-0d6f-7d61-8326-d3bbd344d2df/labelagent_aspect_context_teacher_20261108')

def main():
    if (OUT/'actual_geometry_readback.json').exists():raise FileExistsError('Preserve geometry readback')
    p=read(OUT/'frozen_plan.json');assert all(sha(f)==h for f,h in p['protected'].items());rows=[]
    for view in p['views']:
        source=Image.open(view['source_image']).convert('RGB');actual=Image.open(view['input']).convert('RGB');x1,y1,x2,y2=view['box'];ox1,oy1,ox2,oy2=view['matched_previous_ROI_box'];np.testing.assert_array_equal(np.asarray(actual),np.asarray(source)[y1:y2,x1:x2]);assert x1<=ox1 and y1<=oy1 and x2>=ox2 and y2>=oy2;previous=Image.open(view['matched_previous_ROI_input']).convert('RGB');np.testing.assert_array_equal(np.asarray(actual)[oy1-y1:oy2-y1,ox1-x1:ox2-x1],np.asarray(previous));fx=576/actual.width;fy=448/actual.height;oldfx=576/previous.width;oldfy=448/previous.height;old_box=view['old_source_box'];facts={'teacher_key':view['teacher_key'],'source_box':view['box'],'matched_previous_box':view['matched_previous_ROI_box'],'actual_size':[actual.height,actual.width],'actual_model_resize_factors_xy':[fx,fy],'actual_model_anisotropy_y_over_x':fy/fx,'previous_direct_stretch_factors_xy':[oldfx,oldfy],'previous_anisotropy_y_over_x':oldfy/oldfx,'actual_scale_vs_historical_quarter_xy':[(old_box[2]-old_box[0])/actual.width,(old_box[3]-old_box[1])/actual.height],'stale_inherited_scale_fields_not_used_for_inference_or_reporting':{k:view[k] for k in ['vertical_object_scale_ratio_vs_old_view','horizontal_object_scale_ratio_vs_old_view']},'entire_previous_ROI_pixels_preserved_exactly':True,'no_synthetic_pixels_or_geometric_source_transform':True};rows.append(facts)
    save(OUT/'actual_geometry_readback.json',{'passed':True,'rows':rows,'known_metadata_issue':'The preparation wrapper inherited old historical-quarter scale diagnostic fields; those were not model inputs. Actual boxes and source crops are correct and immutable. Use this independently derived table, not inherited scale fields, for reports.','frozen_plan_not_rewritten_after_inference':True,'extra_context_and_aspect_are_joint_change_not_isolated_causality':True,'new_model_calls':0});print({'actual_source_crops_and_model_factors_verified':len(rows),'inherited_scale_fields_flagged_not_used':True},flush=True)

if __name__=='__main__':main()
