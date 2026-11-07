"""Read-only post-freeze attribution of 1,000 already inferred query slots."""
import datetime,hashlib,json,sys
from pathlib import Path
import cv2,numpy as np,psutil,torch
from PIL import Image
from run_rgb_layers import ROOT,INC,read,save,sha
from projection_audit_core import frame_data,cameras
from m2f_query_replay import probability,replay

OUT=ROOT/'outputs/labelagent_m2f_query_replay_20261107'
OLD=ROOT/'outputs/labelagent_hybrid_repair_20261104/query_diagnosis'
PACK=ROOT/'outputs/mask2former_single_recovery_20261104/最新单模型候选包'


def prepare():
    if OUT.exists():raise FileExistsError('Keep all old results and partial runs')
    OUT.mkdir();protected={};views=[];cfg=read(ROOT/'task_workspaces/labelagent_point_ids_20261107/config.json')
    for p in [Path(cfg['reference_annotations']),Path(cfg['reference_description']),PACK/'model/model.safetensors',PACK/'inference_config.json',PACK/'calibrated_postprocess.py',PACK/'infer_single.py']+list(INC.glob('*/*/result.json')):protected[str(p)]=sha(p)
    assert protected[str(PACK/'model/model.safetensors')]=='d11c2e7f86cd5189605b857c4697e55a5caead0a6888b52b2f5ff814150f5291'
    for cam in (0,1):
        case=f'camera{cam}_frame76';source=INC/'inputs'/(case+'.png');protected[str(source)]=sha(source)
        for view in ['whole','q1','q2','q3','q4']:
            folder=OLD/case/view;meta=read(folder/'diagnosis.json');assert meta['input_sha256']==sha(source)
            for name in ['raw_queries.npz','diagnosis.json','panoptic.png']:protected[str(folder/name)]=sha(folder/name)
            a=np.load(folder/'raw_queries.npz',allow_pickle=False);assert a['class_logits'].shape==(100,8) and a['mask_logits'].shape[0]==100
            views.append({'case':case,'camera':cam,'view':view,'source':str(source),'folder':str(folder),'source_box':meta['crop'],'queries':100})
    save(OUT/'frozen_plan.json',{'at':datetime.datetime.now().astimezone().isoformat(),'driver_sha256':sha(__file__),'replay_sha256':sha(Path(__file__).with_name('m2f_query_replay.py')),'protected':protected,'views':views,'query_slots':1000,'new_model_forwards':0,'no_null_promotion':True,'native_probability_threshold':.5,'sparse_radius_for_diagnosis':3,'target_selection_after_all_old_inference_frozen':True,'not_candidate_generation':True,'original_final_radii_required':[0,3,6],'all_acceptance_gates_unchanged':True})
    print({'cached_views':len(views),'query_slots':1000,'new_forwards':0},flush=True)


def run():
    if (OUT/'completed.json').exists():raise FileExistsError('Completed audit is immutable')
    plan=read(OUT/'frozen_plan.json');assert sha(__file__)==plan['driver_sha256'] and sha(Path(__file__).with_name('m2f_query_replay.py'))==plan['replay_sha256']
    if psutil.virtual_memory().available<1500*2**20:raise RuntimeError('Insufficient available CPU memory; no run started')
    torch.set_num_threads(2);psutil.Process().nice(psutil.BELOW_NORMAL_PRIORITY_CLASS)
    cfg=read(ROOT/'task_workspaces/labelagent_point_ids_20261107/config.json');sys.path[:0]=[cfg['source_snapshot'],str(Path(cfg['source_snapshot'])/'tools'),cfg['dependency_dir']]
    import evaluate_projected_support as ev
    from source.agent.edit_agent import _rasterize
    assert sha(Path(cfg['source_snapshot'])/'tools/evaluate_projected_support.py')=='93c5152169d04d43b25dbf4a4139b12676daf78750a7f059837121656bb644c2'
    dataset=Path(cfg['dataset']);manifest=read(dataset/'import_manifest.json');data=frame_data(dataset,manifest,76,ev);cams=cameras(dataset,manifest,ev);parameters=read(PACK/'inference_config.json')['postprocessing'];kernel=cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(7,7));contexts={};basearea=0;originalarea=0
    for cam in (0,1):
        case=f'camera{cam}_frame76';sh=read(INC/'C_hybrid'/case/'result.json')['shapes'];old=[_rasterize(s,1920,1536) for s in sh if s['label']=='pallet'];union=np.maximum.reduce(old);basearea+=int(union.sum());originalarea+=int(np.maximum.reduce([_rasterize(s,1920,1536) for s in read(INC/'A_current'/case/'result.json')['shapes'] if s['label']=='pallet']).sum());people=cv2.dilate(np.maximum.reduce([_rasterize(s,1920,1536) for s in sh if s['label']=='person']),kernel);source,pixels=ev.project_visible(data['xyz'],cams[cam]);owned=data['owner'][source];known=(data['bits'][source]==1)&(owned>0);eligible=[i for i,b in enumerate(data['boxes']) if b['bit']==1 and np.count_nonzero(owned==i+1)>=5];contexts[case]={'old_union':union,'person_guard':people,'pixels':pixels,'owned':owned,'known':known,'eligible':eligible,'target_ids':[42,48] if cam==0 else [47,49]}
    assert basearea==122302 and originalarea==98009
    allrows=[];viewrows=[]
    for view in plan['views']:
        folder=Path(view['folder']);name=view['case']+'_'+view['view'];out=OUT/'views'/name;out.mkdir(parents=True,exist_ok=True);file=out/'audit.json'
        if file.exists():
            saved=read(file);allrows+=saved['queries'];viewrows.append(saved['view']);continue
        arrays=np.load(folder/'raw_queries.npz',allow_pickle=False);classes=arrays['class_logits'];logits=arrays['mask_logits'];x1,y1,x2,y2=view['source_box'];size=(y2-y1,x2-x1);ids,segs,replayed,winner=replay(classes,logits,size,parameters)
        encoded=np.asarray(Image.open(folder/'panoptic.png')).astype('uint32');oldids=encoded[:,:,0]+encoded[:,:,1]*256+encoded[:,:,2]*65536;assert np.array_equal(ids,oldids),'Legacy postprocess does not exactly reproduce cached panoptic'
        ctx=contexts[view['case']];rows=[]
        def score(local):
            full=np.zeros((1536,1920),np.uint8);full[y1:y2,x1:x2]=local;assert int(full.sum())==int(local.sum());hits=cv2.dilate(full,kernel)[ctx['pixels'][:,1],ctx['pixels'][:,0]]!=0;den=int(np.count_nonzero(hits&ctx['known']));ownership=[]
            for i in ctx['eligible']:
                n=int(np.count_nonzero(hits&(ctx['owned']==i+1)));total=int(np.count_nonzero(ctx['owned']==i+1));purity=n/den if den else 0
                if n:ownership.append({'object_id':data['boxes'][i]['obj_id'],'hits':n,'total':total,'coverage':n/total,'purity':purity,'qualified_support':n/total if purity>=.67 else 0})
            extra=int(np.count_nonzero(full&(ctx['old_union']==0)));border=bool(local[0].any() or local[-1].any() or local[:,0].any() or local[:,-1].any());return {'pixels':int(local.sum()),'extra_pixels':extra,'area_fits':basearea+extra<=1.25*originalarea,'person_conflict':bool((full&ctx['person_guard']).any()),'touches_source_crop_border':border,'ownership':ownership,'full_mask_sha256':hashlib.sha256(full.tobytes()).hexdigest()}
        for q in range(100):
            local=(probability(logits,q,size)>=.5).astype('uint8');native=score(local);legacy=None
            if replayed[q]['class_gate']:legacy=score(((winner==q)&(local!=0)).astype('uint8'))
            row={'key':name+f'_query{q:02d}','case':view['case'],'view':view['view'],'source_box':view['source_box'],**replayed[q],'native':native,'legacy_competition_selected':legacy,'native_mask_never_replaced_with_selected':True};rows.append(row)
        v={'case':view['case'],'view':view['view'],'slots':100,'legacy_panoptic_exact':True,'class_PAL_retained':sum(r['class_gate'] and r['top_class_id']==0 for r in rows),'postprocess_PAL_instances':sum(s['category_id']==1 for s in segs),'argmax_null_slots':sum(r['top_class_id']==7 for r in rows),'new_forwards':0};save(file,{'view':v,'queries':rows});allrows+=rows;viewrows.append(v);save(OUT/'progress.json',{'completed_views':len(viewrows),'completed_slots':len(allrows)});print(v,flush=True)
    assert len(allrows)==1000 and all(sha(p)==h for p,h in plan['protected'].items())
    targets=[]
    for case,ctx in contexts.items():
        for oid in ctx['target_ids']:
            def detail(row):return next((x for x in row['native']['ownership'] if x['object_id']==oid),{'hits':0,'coverage':0,'purity':0,'qualified_support':0})
            groups={}
            for group,condition in [('all_native_slots',lambda r:True),('PAL_argmax_score_kept',lambda r:r['class_gate'] and r['top_class_id']==0),('PAL_native_area_person_border',lambda r:r['class_gate'] and r['top_class_id']==0 and r['native']['area_fits'] and not r['native']['person_conflict'] and not r['native']['touches_source_crop_border'])]:
                options=[r for r in allrows if r['case']==case and condition(r)];best=sorted(options,key=lambda r:(-detail(r)['qualified_support'],-detail(r)['coverage'],-detail(r)['purity'],r['native']['extra_pixels'],-r['p_pallet'],r['key']))[:1];groups[group]=[{'key':r['key'],'top_class_id':r['top_class_id'],'p_pallet':r['p_pallet'],'p_null':r['p_null'],'native':r['native'],'target':detail(r)} for r in best]
            targets.append({'case':case,'object_id':oid,'groups':groups})
    result={'status':'complete_frozen_query_attribution','query_slots':1000,'old_inference_views':10,'new_model_forwards':0,'new_SAM_calls':0,'new_VLM_calls':0,'new_Agent_runs':0,'new_training':False,'legacy_panoptic_exact_views':len(viewrows),'views':viewrows,'targets':targets,'originals_and_old_six_unchanged':True,'retained_support':'13/17','retained_original_Agent':'6/6','remaining_misses':4,'adopted':False,'audit_radius':3,'not_admission_or_dense_truth':True};save(OUT/'all_query_metrics.json',allrows);save(OUT/'completed.json',result);print(json.dumps({'slots':1000,'cached_panoptic_exact':10,'targets':[{k:v for k,v in t.items() if k!='groups'} for t in targets]}),flush=True)


if __name__=='__main__':prepare() if sys.argv[1]=='prepare' else run()
