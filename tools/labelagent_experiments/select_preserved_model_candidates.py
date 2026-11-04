"""Keep original trained hypotheses when SAM/review safety checks fail."""
import sys,json,hashlib,shutil
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];work=ROOT/'task_workspaces/labelagent_recovery_20261105';source=ROOT/'outputs/labelagent_recovery_20261105/instance_generation_independent_model';out=source.parent/'instance_generation_preserved_model'
def read(p):return json.loads(p.read_text(encoding='utf-8'))
def save(p,v):p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(v,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
def main():
 cfg=read(work/'config.json');sys.path.insert(0,cfg['source_snapshot'])
 from source.agent.edit_agent import _rasterize
 from preserved_model_policy import select_complete_masks
 if out.exists():raise FileExistsError('Preserve candidate bank')
 out.mkdir();pool={};baselines={};people={}
 for cam in [0,1]:
  case=f'camera{cam}_frame76';seed=read(ROOT/'outputs/labelagent_recovery_comparison_20261105/A_current'/case/'result.json')['shapes']
  baselines[case]=[_rasterize(s,1920,1536) for s in seed if s['label']=='pallet'];people[case]=[_rasterize(s,1920,1536) for s in seed if s['label']=='person'];proposals={}
  for file in sorted((source.parent/'fixed_grid'/case).glob('*/prediction.json')):
   for shape in read(file)['PAL_shapes']:
    mask=_rasterize(shape,1920,1536);key=hashlib.sha256(case.encode()+mask.tobytes()).hexdigest();row={'case':case,'key':key,'shape':shape,'mask':mask,'trained_score':float(shape['score']),'source_prediction':str(file.relative_to(source.parent)),
      'review_status':'human_review_required; failed visual controls cannot authorize overwriting trained instances','SAM_output_advisory_only':True}
    if key not in proposals or row['trained_score']>proposals[key]['trained_score']:proposals[key]=row
  pool[case]=list(proposals.values());save(out/case/'all_raw_model_hypotheses.json',[{k:v for k,v in r.items() if k!='mask'} for r in pool[case]])
 selected,rejected,summary=select_complete_masks(pool,baselines,people)
 for case in baselines:
  save(out/case/'C_children.json',[r['shape'] for r in selected[case]]);save(out/case/'B_children.json',read(source/case/'B_children.json'))
 save(out/'C_selection.json',{'selected':selected,'rejected_whole_masks':rejected,'summary':summary,'no_pixel_clipping':True})
 for name in ['refinement_progress.json','blank_controls.json','model_trace.json']:
  shutil.copy2(source/name,out/name)
 shutil.copytree(source/'visual_cache',out/'visual_cache')
 save(out/'generation_protocol.json',{'scope':'known-frame engineering recovery; no independent accuracy claim','source':'same32 fixed-grid model forwards, all raw instance hypotheses before early filters',
  'mask_overwrite_policy':'failed review negative controls prevent automatic SAM mask replacement; original trained shapes kept as complete unreviewed candidates',
  'ranking':'trained score * (1-maximum IoU with original predicted PAL masks), deterministic key ties; no reference labels/points',
  'budgets':'per-camera count <=1.5 baseline and union<=1.25 baseline, therefore original global limits also hold',
  'person_guard':'whole proposal rejected on overlap with radius3 dilated original predicted person mask',
  'NMS':'original IoU0.5 and overlap0.8 only between newly admitted proposals, after safety and budget checks; original SAM masks retained and not used to veto distinct trained alternatives',
  'no_new_inference':True,'prior_SAM_and_review_traces_advisory':True,'whole_masks_only':True,'automatic_adoption':False})
 save(out/'completed.json',{'status':'complete','references_used_for_selection':False,'annotation_writes':0,'whole_masks_only':True,'model_trace_records':len(read(out/'model_trace.json')),'fresh_model_requests':0,
  'cases':{c:{a:len(read(out/c/(a+'_children.json'))) for a in ['B','C']} for c in baselines},'blank_controls_passed':read(out/'blank_controls.json')['all_passed'],'raw_model_instances_preserved':True})
 print(json.dumps(read(out/'completed.json')),flush=True)
if __name__=='__main__':main()
