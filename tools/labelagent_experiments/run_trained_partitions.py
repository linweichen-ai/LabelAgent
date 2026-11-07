"""Cross-teacher point hypotheses, complete native masks, no output trimming."""
import datetime,hashlib,os,sys
from pathlib import Path
from run_rgb_layers import ROOT,read,save,sha
import run_trained_point_sam as sam

OUT=ROOT/'outputs/labelagent_trained_partition_20261108'
ORIGINAL=ROOT/'outputs/labelagent_trained_point_review_20261108/completed.json'

def prepare():
    if OUT.exists():raise FileExistsError('Preserve partition hypotheses')
    original=read(ORIGINAL);cues=[r for r in original['rows'] if r['PAL_subject_evidence']];rows=[]
    for positive in cues:
        x1,y1,x2,y2=positive['box']
        for other in cues:
            if positive['case']!=other['case'] or positive['teacher_key']==other['teacher_key']:continue
            if sha(positive['source_image'])!=sha(other['source_image']):continue
            x,y=other['source_point'];px,py=positive['source_point']
            if not(x1<=x<x2 and y1<=y<y2) or (x-px)**2+(y-py)**2<16**2:continue
            points=[dict(positive['points'][0]),{'x':x-x1,'y':y-y1,'label':0}];rows.append({**positive,'index':len(rows)+1,'positive_source_index':positive['index'],'negative_source_index':other['index'],'points':points,'negative_source_point':[x,y],'different_instance_hypothesis_not_confirmed_background':True,'new_annotation_or_human_truth':False})
    assert len(rows)==6 and all(len(r['points'])==2 for r in rows)
    OUT.mkdir();protected={str(ORIGINAL):sha(ORIGINAL),str(Path(__file__)):sha(__file__),str(Path(sam.__file__)):sha(sam.__file__)}
    save(OUT/'partition_plan.json',{'at':datetime.datetime.now().astimezone().isoformat(),'protected':protected,'rows':rows,'rule':'all ordered semantically screened source cues from distinct teacher queries, same source photograph, other point inside unchanged original input and >=16px away','point_labels_are_unconfirmed_partition_hypotheses':True,'no_reference_points_or_area_read_by_generation':True,'no_input_expansion_or_output_clipping':True,'maximum_points_per_hypothesis':2,'new_Agent_rounds':0,'original_Agent_round_limit':3})
    save(OUT/'proposal_adapter/completed.json',{'rows':rows,'partition_plan_sha256':sha(OUT/'partition_plan.json'),'not_final_instance_acceptance':True});sam.OUT=OUT/'SAM';sam.REVIEW=OUT/'proposal_adapter';sam.prepare();print({'partition_hypotheses':len(rows),'new_model_calls':0},flush=True)

def run():
    p=read(OUT/'partition_plan.json');assert all(sha(f)==h for f,h in p['protected'].items());sam.OUT=OUT/'SAM';sam.REVIEW=OUT/'proposal_adapter';sam.run();assert all(sha(f)==h for f,h in p['protected'].items())

if __name__=='__main__':prepare() if sys.argv[1]=='prepare' else run()
