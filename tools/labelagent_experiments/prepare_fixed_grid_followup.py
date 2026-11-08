"""Replay both frozen CPU grid cohorts and prepare source-only consensus cues.

No reference annotations, projected points, poses, depth, or area budgets are
read. The two weight pairs keep their own original consensus control rule.
The 32 unique real-point review bound is shared across both pairs.
"""
import datetime,hashlib,importlib.util,json,shutil,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]
cfg=json.loads((ROOT/'task_workspaces/labelagent_point_ids_20261107/config.json').read_text('utf-8'))
sys.path.insert(0,cfg['dependency_dir'])
import cv2,numpy as np,torch
from PIL import Image
from run_rgb_layers import read,save
from deliver_context_graph import sha
from paired_source_cue import body
from m2f_query_replay import probability,replay

BASE=Path('D:/AI/Codex/Home/visualizations/2026/11/04/01a1a56c-0d6f-7d61-8326-d3bbd344d2df')
OUT=BASE/'labelagent_fixed_grid_followup_20261108'
COHORTS=[('original_pair',BASE/'labelagent_fixed_grid_teacher_20261108'),('registered_pair',BASE/'labelagent_registered_fixed_grid_20261108')]
PACK=ROOT/'outputs/mask2former_single_recovery_20261104/最新单模型候选包'

def caption_views(photo,point,folder):
    folder.mkdir(parents=True,exist_ok=True);ax,ay=point;bindings=[]
    for name,side in [('near',129),('context',385)]:
        half=side//2;box=[ax-half,ay-half,ax+half+1,ay+half+1];clip=[max(0,box[0]),max(0,box[1]),min(photo.width,box[2]),min(photo.height,box[3])]
        canvas=Image.new('RGB',(side,side),(127,)*3);canvas.paste(photo.crop(tuple(clip)),(clip[0]-box[0],clip[1]-box[1]));file=folder/(name+'.png');canvas.resize((512,512),Image.Resampling.LANCZOS).save(file)
        bindings.append({'image':str(file),'sha256':sha(file),'source_canvas_box':box,'clipped_box':clip,'side':side,'unmarked_original_photo':True})
    request=body(*[b['image'] for b in bindings]);key=hashlib.sha256(json.dumps(request,sort_keys=True).encode()).hexdigest();requestfile=OUT/'prepared_requests'/(key+'_PRIVATE.json');save(requestfile,request)
    return {'bindings':bindings,'request_file':str(requestfile),'request_sha256':key}

def main():
    if OUT.exists():raise FileExistsError('Preserve completed/interrupted source-only preparation')
    if shutil.disk_usage(BASE).free<192*2**20:raise RuntimeError('192MiB preparation reserve')
    torch.set_num_threads(2);torch.set_num_interop_threads(1)
    spec=importlib.util.spec_from_file_location('unaltered_original_calibrated',PACK/'calibrated_postprocess.py');stock=importlib.util.module_from_spec(spec);spec.loader.exec_module(stock)
    OUT.mkdir();protected={str(Path(__file__)):sha(__file__),str(PACK/'calibrated_postprocess.py'):sha(PACK/'calibrated_postprocess.py'),str(Path(__file__).with_name('m2f_query_replay.py')):sha(Path(__file__).with_name('m2f_query_replay.py')),str(Path(__file__).with_name('paired_source_cue.py')):sha(Path(__file__).with_name('paired_source_cue.py'))};checks=[];records={};options=[];rejections=[]
    for cohort,folder in COHORTS:
        p=read(folder/'frozen_plan.json');g=read(folder/'generation_completed.json');assert g['new_CPU_model_forwards']==24 and len(g['rows'])==24 and all(sha(f)==h for f,h in p['protected'].items());protected[str(folder/'frozen_plan.json')]=sha(folder/'frozen_plan.json');protected[str(folder/'generation_completed.json')]=sha(folder/'generation_completed.json');models=[m['name'] for m in p['models']];assert len(models)==2
        for r in g['rows']:
            f=Path(r['folder']);assert all(sha(f/n)==h for n,h in r['artifacts'].items());im=Image.open(r['input']).convert('RGB');size=(im.height,im.width)
            with np.load(f/'raw_queries.npz',allow_pickle=False) as a:
                classes=a['class_logits'];masks=a['mask_logits'];ids,segs,queries,winner=replay(classes,masks,size,p['postprocessing'])
                independent,original_segments=stock.postprocess({'class_queries_logits':torch.as_tensor(classes)[None],'masks_queries_logits':torch.as_tensor(masks)[None]},size,p['postprocessing']);np.testing.assert_array_equal(independent,ids);assert segs==original_segments==r['segments_info']
                rgb=np.asarray(Image.open(f/'panoptic.png')).astype('uint32');np.testing.assert_array_equal(ids,rgb[:,:,0]+256*rgb[:,:,1]+65536*rgb[:,:,2]);assert queries==read(f/'query_class_records.json')
                pal=[s for s in segs if s['category_id']==1];pal_mask=np.isin(ids,[s['id'] for s in pal]);records[(cohort,r['model'],r['teacher_key'])]=(r,pal_mask)
                checks.append({'cohort':cohort,'model':r['model'],'view':r['teacher_key'],'stock_PAL':len(pal),'synthetic_control':r['is_synthetic_control'],'original_postprocess_pixels_segments_query_records_exact':True})
                for name in r['artifacts']:protected[str(f/name)]=sha(f/name)
                protected[r['input']]=sha(r['input']);protected[r['source_image']]=sha(r['source_image'])
                if r['is_synthetic_control']:continue
                protected[r['original_source_image']]=sha(r['original_source_image'])
                # Rank first, then apply the unchanged geometry veto. Do not
                # replace a rejected top segment with a lower ranked segment.
                if not pal:continue
                s=sorted(pal,key=lambda s:(-s['score'],s['id']))[0];selected=ids==s['id'];q=np.unique(winner[selected]);assert len(q)==1;q=int(q[0]);native=probability(masks,q,size)>=.5
                border=lambda m:bool(m[0].any() or m[-1].any() or m[:,0].any() or m[:,-1].any())
                if border(native) or border(selected):
                    rejections.append({'cohort':cohort,'model':r['model'],'view':r['teacher_key'],'segment':s['id'],'reason':'highest_score_MF_native_or_stock_input_border','native_border':border(native),'stock_border':border(selected)});continue
                options.append({'cohort':cohort,'models':models,'row':r,'segment':s,'query':q,'selected':selected,'native':native})
    # Seal the complete output replay before any cue or diagnostic GT work.
    save(OUT/'all48_original_postprocess_readback.json',{'passed':True,'rows':checks,'total_saved_forwards_replayed_without_model_calls':48,'all_original_pixels_and_segment_metadata_exact':True,'new_model_calls':0,'reference_annotations_not_read':True,'stock_source_sha256':sha(PACK/'calibrated_postprocess.py')})
    candidates=[]
    for op in options:
        r=op['row'];selected=op['selected'];ys=np.where(selected)[0];ymin,ymax=int(ys.min()),int(ys.max());lower=ymin+2*(ymax-ymin+1)//3;distance=cv2.distanceTransform(np.pad(selected.astype('uint8'),1),cv2.DIST_L2,5)[1:-1,1:-1];x1,y1,x2,y2=r['observed_region_in_input'];observed=np.zeros(selected.shape,bool);observed[y1:y2,x1:x2]=True
        for policy in ['maximum_distance','lowest_third_maximum_distance']:
            scores=distance.copy();scores[~observed]=0
            if policy.startswith('lowest'):scores[:lower]=0
            if scores.max()<=0:continue
            y,x=map(int,np.unravel_index(scores.argmax(),scores.shape));evidence=[]
            for model in op['models']:
                other,other_pal=records[(op['cohort'],model,r['teacher_key'])];assert other['source_virtual_box']==r['source_virtual_box'] and other['artifacts'] and sha(other['input'])==sha(r['input']);evidence.append({'model':model,'panoptic':str(Path(other['folder'])/'panoptic.png'),'sha256':other['artifacts']['panoptic.png'],'PAL_at_point':bool(other_pal[y,x])})
            ax=x+r['source_virtual_box'][0];ay=y+r['source_virtual_box'][1];assert 0<=ax<1920 and 0<=ay<1536
            row={'cohort':op['cohort'],'model':r['model'],'teacher_key':r['teacher_key'],'case':r['case'],'source_view':r['source_view'],'source_image':r['original_source_image'],'padded_input':r['input'],'source_virtual_box':r['source_virtual_box'],'observed_region_in_input':r['observed_region_in_input'],'input_point':[x,y],'source_point':[ax,ay],'policy':policy,'stock_segment':op['segment'],'query':op['query'],'both_stock_panoptics_predict_PAL':all(e['PAL_at_point'] for e in evidence),'consensus_evidence':evidence}
            if not row['both_stock_panoptics_predict_PAL']:rejections.append(row|{'reason':'source_pair_stock_PAL_disagreement'});continue
            candidates.append(row)
    # Fair fixed-tile order, then two policies and each pair/model. Hash/pixels
    # of input determine dedup; there is no projected-instance preference.
    order={'whole':0,'q1':1,'q2':2,'q3':3,'q4':4};candidates.sort(key=lambda r:(order[r['source_view']],r['policy'],r['case'],r['cohort'],r['model']));rows=[];seen={}
    for r in candidates:
        key=(r['case'],*r['source_point'])
        if key in seen:rows[seen[key]]['logical_sources'].append(r);continue
        if len(rows)>=32:rejections.append(r|{'reason':'shared32_unique_real_point_bound'});continue
        index=len(rows)+1;photo=Image.open(r['source_image']).convert('RGB');views=caption_views(photo,r['source_point'],OUT/'inputs'/f'cue{index:02d}');row=r|views|{'index':index,'logical_sources':[r],'source_partition_index':index,'new_point_views':[views|{'new_point':{'x':r['input_point'][0],'y':r['input_point'][1],'label':1},'source_point':r['source_point']}],'not_human_truth':True};seen[key]=len(rows);rows.append(row)
    model_sha=read(ROOT/'outputs/labelagent_trained_point_review_20261108/frozen_plan.json')['model_config_sha256']
    # Two already-defined negative controls use the identical image pipeline.
    controls=[]
    for name in ['gray','white']:
        f=ROOT/'outputs/labelagent_semantic_recovery_20261105/neutral_controls'/(name+'.png');photo=Image.open(f).convert('RGB');protected[str(f)]=sha(f);views=caption_views(photo,[photo.width//2,photo.height//2],OUT/'controls'/name);controls.append(views|{'index':'control_'+name,'known_no_real_object':True,'name':name})
    for r in rows+controls:
        protected[r['request_file']]=sha(r['request_file'])
        for b in r['bindings']:protected[b['image']]=b['sha256']
    save(OUT/'frozen_plan.json',{'at':datetime.datetime.now().astimezone().isoformat(),'rows':rows,'controls':controls,'protected':protected,'model_config_sha256':model_sha,'postprocess_readback_sha256':sha(OUT/'all48_original_postprocess_readback.json'),'all_candidate_and_rejection_source_records':rejections,'source_selection_before_reference_audit':True,'source_pair_consensus_not_assumed':True,'max_unique_real_point_reviews':32,'max_unique_SAM_point_candidates':16,'max_paired_single_point_SAM_calls':32,'matched_plain_controls_additional_to_real_points':2,'SAM_point_only_no_MF_prior':True,'SAM_arms':['unmarked_original_full_source','same_fixed_grid_source_input_with_explicit_padding'],'SAM_mapping':'Save native whole output first; reject any foreground on synthetic padding; inverse map every foreground pixel with exact count, no cutting or clamping','all_final_gates_unchanged':True,'Agent_round_limit':3,'new_Agent_runs':0,'adopted':False,'new_model_calls_this_stage':0})
    save(OUT/'prepared.json',{'unique_real_source_points':len(rows),'logical_source_candidates':len(candidates),'controls':len(controls),'new_model_calls':0,'independent_saved_forward_replays':48,'source_only_before_reference_diagnosis':True});print(read(OUT/'prepared.json'),flush=True)

if __name__=='__main__':main()
