"""Add opposite-side RGB instance hypotheses, never reference-derived points."""
import datetime,os,sys
from pathlib import Path
from run_rgb_layers import ROOT,read,save,sha
import run_trained_point_sam as sam
OUT=ROOT/'outputs/labelagent_two_sided_layers_20261108'
PREVIOUS=ROOT/'outputs/labelagent_base_blue_surface_20261108'
INITIAL=ROOT/'outputs/labelagent_trained_point_review_20261108/completed.json'

def prepare():
    if OUT.exists():raise FileExistsError('Preserve two-sided hypothesis study')
    old=read(PREVIOUS/'SAM/generation_completed.json');cues=[r for r in read(INITIAL)['rows'] if r['PAL_subject_evidence']];rows=[];diagnostics=[]
    for r in old['rows']:
        positive=r['points'][0];negative=next(q for q in r['points'] if q['label']==0);x1,y1,x2,y2=r['box'];above=negative['y']<positive['y'];eligible=[]
        for q in cues:
            if q['case']!=r['case'] or sha(q['source_image'])!=sha(r['source_image']):continue
            sx,sy=q['source_point'];x,y=sx-x1,sy-y1;dy=y-positive['y']
            if not(0<=x<x2-x1 and 0<=y<y2-y1) or abs(dy)<8:continue
            if (dy>0)!=above:continue
            if any((x-p['x'])**2+(y-p['y'])**2<9 for p in r['points']):continue
            eligible.append((abs(dy),(x-positive['x'])**2+dy**2,q['index'],x,y))
        if not eligible:diagnostics.append({'partition':r['source_partition_index'],'reason':'No source-reviewed PAL cue on opposite vertical side'});continue
        _,_,index,x,y=min(eligible);new={'x':x,'y':y,'label':0};rows.append({**r,'points':r['points']+[new],'original_four_points':r['points'],'opposite_side_negative_point':new,'opposite_side_source_cue_index':index,'negative_is_other_instance_hypothesis_not_background_truth':True,'new_annotation_or_human_truth':False,'new_VLM_call_required':False});diagnostics.append({'partition':r['source_partition_index'],'new_source_cue':index,'opposite_side_negative_point':new})
    assert len(rows)==3 and all(len(r['points'])==5 for r in rows);OUT.mkdir();protected={str(PREVIOUS/'SAM/generation_completed.json'):sha(PREVIOUS/'SAM/generation_completed.json'),str(INITIAL):sha(INITIAL),str(Path(__file__)):sha(__file__),str(Path(sam.__file__)):sha(sam.__file__)}
    for r in rows:protected[r['input']]=sha(r['input']);protected[r['source_image']]=sha(r['source_image'])
    save(OUT/'frozen_plan.json',{'at':datetime.datetime.now().astimezone().isoformat(),'rows':rows,'diagnostics':diagnostics,'protected':protected,'generation_rule':'Keep complete earlier four-point source proposal; add nearest vertically opposite, >=8px separated previously neutrally reviewed PAL cue inside unchanged original input','all_previous_proposal_points_preserved':True,'no_reference_coordinates_labels_depth_poses_or_area_read_by_generator':True,'layer_hypothesis_context':'posthoc prior reference box metadata inspection motivated instance granularity diagnosis; this is development not independent validation','new_source_cues_not_human_truth':True,'maximum_points_per_proposal':5,'new_Agent_rounds':0,'original_Agent_round_limit':3,'no_new_VLM_or_Mask2Former_calls':True,'no_output_mask_clipping':True});save(OUT/'proposal_adapter/completed.json',{'rows':rows,'frozen_plan_sha256':sha(OUT/'frozen_plan.json'),'only_previously_reviewed_source_cues_used':True});sam.OUT=OUT/'SAM';sam.REVIEW=OUT/'proposal_adapter';sam.prepare();print({'two_sided_hypotheses':len(rows),'new_VLM_calls':0,'new_SAM_calls':0},flush=True)

def run():
    p=read(OUT/'frozen_plan.json');assert all(sha(f)==h for f,h in p['protected'].items());sam.OUT=OUT/'SAM';sam.REVIEW=OUT/'proposal_adapter';sam.run()

if __name__=='__main__':prepare() if sys.argv[1]=='prepare' else run()
