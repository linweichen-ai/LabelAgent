"""Keep RGB midpoint heights; use already frozen source-material width only."""
import sys
from pathlib import Path
from run_rgb_layers import ROOT,read,save,sha
from source_layer_box_policy import propose as vertical_box
import run_source_layer_boxes as runner
OUT=ROOT/'outputs/labelagent_source_surface_boxes_20261108'

def source_surface_box(points,width,height):
    base,diag=vertical_box(points,width,height)
    if base is None:return None,diag
    source=read(runner.PREVIOUS/'SAM/generation_completed.json');row=next(r for r in source['rows'] if r['points']==points and r['box'][2]-r['box'][0]==width and r['box'][3]-r['box'][1]==height);lo,hi=row['source_material_diagnostics']['component_x_extent'];base[0]=float(max(0,lo-2));base[2]=float(min(width-1,hi+2))
    if any(not(base[0]<=q['x']<=base[2] and base[1]<=q['y']<=base[3]) for q in points if q['label']==1):return None,{**diag,'reason':'A frozen source positive lies outside RGB material-width box'}
    return base,{**diag,'horizontal_rule':'Already frozen RGB blue-component x extent with two source pixel margin, not reference or budget','source_component_x_extent':[lo,hi],'box_is_prompt_not_output_clip':True}

def main():
    runner.OUT=OUT;runner.propose=source_surface_box
    if sys.argv[1]=='prepare':
        runner.prepare();p=read(OUT/'frozen_plan.json');p['protected'][str(Path(__file__))]=sha(__file__);p['only_horizontal_box_range_differs_from_full_width_box_trial']=True;p['all_source_points_vertical_bounds_and_images_preserved']=True;save(OUT/'frozen_plan.json',p)
    else:
        p=read(OUT/'frozen_plan.json');assert p['protected'][str(Path(__file__))]==sha(__file__);runner.run()

if __name__=='__main__':main()
