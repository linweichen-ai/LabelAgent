"""Frozen hard/soft teacher priors, same source pixels/points and one SAM call."""
import base64,datetime,hashlib,io,json,os,sys,time,urllib.request,urllib.error,subprocess
from pathlib import Path
import cv2,numpy as np
from PIL import Image
from run_rgb_layers import ROOT,INC,read,save,sha
from box_layer_policy import propose
from m2f_query_replay import probability
from soft_prior_capture import validate_prior

OUT=ROOT/'outputs/labelagent_soft_prior_20261108'
OLD=ROOT/'outputs/labelagent_hybrid_repair_20261104/query_diagnosis'


def png(im):
    b=io.BytesIO();im.save(b,'PNG');return b.getvalue()


def prepare():
    if OUT.exists():raise FileExistsError('Preserve immutable experiment')
    OUT.mkdir();subprocess.run(['compact.exe','/C','/Q',str(OUT)],check=True,capture_output=True)
    import torch
    torch.set_num_threads(2);cfg=read(ROOT/'task_workspaces/labelagent_point_ids_20261107/config.json');protected={str(p):sha(p) for p in [Path(cfg['reference_annotations']),Path(cfg['reference_description'])]+list(INC.glob('*/*/result.json'))};prepared=[];diagnostics=[]
    for cam in (0,1):
        case=f'camera{cam}_frame76';sourcefile=INC/'inputs'/(case+'.png');source=Image.open(sourcefile).convert('RGB');protected[str(sourcefile)]=sha(sourcefile)
        for view in ['whole','q1','q2','q3','q4']:
            folder=OLD/case/view;meta=read(folder/'diagnosis.json');assert meta['input_sha256']==sha(sourcefile);protected[str(folder/'raw_queries.npz')]=sha(folder/'raw_queries.npz');a=np.load(folder/'raw_queries.npz',allow_pickle=False);probs=torch.tensor(a['class_logits']).softmax(-1);scores,labels=probs.max(-1);vx1,vy1,vx2,vy2=meta['crop'];size=(vy2-vy1,vx2-vx1)
            for q in np.flatnonzero(((labels==0)&(scores>.2)).numpy()):
                # Same exact interpolation as original query postprocessing.
                import torch.nn.functional as F
                logits=F.interpolate(torch.tensor(a['mask_logits'][q])[None,None],size=size,mode='bilinear',align_corners=False)[0,0].numpy();binary=(probability(a['mask_logits'],int(q),size)>=.5).astype('uint8');key=f'{case}_{view}_query{q:02d}';diag={'teacher_key':key,'teacher_PAL_score':float(scores[q]),'teacher_argmax':0,'prior_pixels':int(binary.sum()),'used_reference_inputs':False}
                border=bool(binary[0].any() or binary[-1].any() or binary[:,0].any() or binary[:,-1].any())
                if not binary.any() or border or not np.array_equal(binary!=0,logits>=0):
                    diagnostics.append({**diag,'rejected':'empty, original source crop border or float sign disagreement','border':border});continue
                yy,xx=np.where(binary);localbox=[max(0,int(xx.min())-32),max(0,int(yy.min())-32),min(size[1],int(xx.max())+33),min(size[0],int(yy.max())+33)];x1,y1,x2,y2=localbox;box=[vx1+x1,vy1+y1,vx1+x2,vy1+y2];photo=source.crop(tuple(box));soft=logits[y1:y2,x1:x2].copy();mask=binary[y1:y2,x1:x2].copy();assert int(mask.sum())==int(binary.sum());rawphoto=png(photo);rawmask=png(Image.fromarray(mask*255));validate_prior(soft,rawphoto,rawmask);plans,geom=propose(np.asarray(photo),mask);dest=OUT/'teachers'/key;dest.mkdir(parents=True);(dest/'input.png').write_bytes(rawphoto);(dest/'prior.png').write_bytes(rawmask);np.save(dest/'teacher_source_logits.npy',soft);d={**diag,'source_box':box,'complete_teacher_mask_retained':True,'geometric_hypotheses':len(plans),'geometry':geom};diagnostics.append(d)
                for rank,p in enumerate(plans):
                    opposite=next((o for o in plans if o['side']!=p['side']),None)
                    if opposite is None:continue
                    pos=p['points'][0];neg=dict(opposite['points'][0],label=0)
                    if (pos['x']-neg['x'])**2+(pos['y']-neg['y'])**2<9:continue
                    points=[pos,neg];pair=key+'_'+p['side'];prepared.append({'pair':pair,'teacher_key':key,'teacher_score':float(scores[q]),'seed':len(prepared)+1,'case':case,'source_view':view,'query':int(q),'side':p['side'],'rank':rank,'box':box,'input':str(dest/'input.png'),'prior':str(dest/'prior.png'),'logits':str(dest/'teacher_source_logits.npy'),'points':points,'no_reference_inputs':True,'negative_inside_unconfirmed_parent':True,'kind':'real'})
    if not prepared:raise RuntimeError('No image-only teacher hypotheses; no inference')
    exemplar=prepared[0]
    for color,value in [('gray',127),('white',255)]:
        dest=OUT/'controls'/color;dest.mkdir(parents=True);Image.new('RGB',Image.open(exemplar['input']).size,(value,)*3).save(dest/'input.png');prepared.append({**exemplar,'pair':exemplar['pair']+'_'+color,'input':str(dest/'input.png'),'kind':color+'_control','reference_support_not_applicable':True})
    proposals=[]
    for p in prepared:
        for mode in ['hard','soft']:
            key=hashlib.sha256((sha(p['input'])+sha(p['prior'])+sha(p['logits'])+json.dumps(p['points'],sort_keys=True)+mode).encode()).hexdigest();proposals.append({**p,'mode':mode,'key':key,'input_sha256':sha(p['input']),'prior_sha256':sha(p['prior']),'source_logit_file_sha256':sha(p['logits'])})
    deployed=read(ROOT/'task_workspaces/labelagent_soft_prior_20261108/deployment/deployed.json');protected.update(deployed['protected_after'])
    save(OUT/'frozen_plan.json',{'at':datetime.datetime.now().astimezone().isoformat(),'driver_sha256':sha(__file__),'policy_sha256':sha(Path(__file__).with_name('box_layer_policy.py')),'protected':protected,'proposals':proposals,'teacher_diagnostics':diagnostics,'paired_only_change':'hard vs native soft mask_input values; identical source foreground, RGB, points, precision, single-mask path and one SAM call','no_teacher_class_threshold_change':True,'no_null_promotion':True,'teacher_checkpoint_sha256':'d11c2e7f86cd5189605b857c4697e55a5caead0a6888b52b2f5ff814150f5291','one_SAM_call_per_chain':True,'new_Agent_rounds':0,'Agent_round_limit':3,'all_final_gates_unchanged':True,'new_model_training':False,'max_requests':len(proposals),'mask_post_clipping':False});print({'real_pairs':len(prepared)-2,'control_pairs':2,'max_SAM_calls':len(proposals),'PAL_queries_seen':len(diagnostics)},flush=True)


def run():
    if (OUT/'generation_completed.json').exists():raise FileExistsError('Do not repeat successful inference')
    plan=read(OUT/'frozen_plan.json');assert sha(__file__)==plan['driver_sha256'];rows=[];attempts=0
    for p in plan['proposals']:
        assert sha(p['input'])==p['input_sha256'] and sha(p['prior'])==p['prior_sha256'] and sha(p['logits'])==p['source_logit_file_sha256'];folder=OUT/'captures'/p['key'];folder.mkdir(parents=True,exist_ok=True);file=folder/'response.json'
        if not file.exists():
            payload={'image':base64.b64encode(Path(p['input']).read_bytes()).decode(),'mask':base64.b64encode(Path(p['prior']).read_bytes()).decode(),'points':p['points'],'include_raw_capture':True,'diagnostic_prior_logits':base64.b64encode(Path(p['logits']).read_bytes()).decode(),'diagnostic_prior_mode':p['mode']};body=json.dumps(payload).encode()
            for _ in range(1500):
                if (OUT/'STOP').exists():raise RuntimeError('Explicit STOP')
                attempts+=1
                try:
                    with urllib.request.urlopen(urllib.request.Request('http://127.0.0.1:8766/segment_points',data=body,headers={'Content-Type':'application/json'}),timeout=600) as f:response=json.load(f)
                    break
                except urllib.error.HTTPError as e:
                    if e.code==503:time.sleep(.2);continue
                    response=json.loads(e.read());break
            else:raise RuntimeError('Shared manager busy; resume missing only')
            save(file,response)
        response=read(file);capture=response.get('raw_capture');row={**p,'folder':str(folder),'valid':False,'status':'capture_missing'}
        if capture:
            meta=capture['metadata'];save(folder/'metadata.json',meta);npz=base64.b64decode(capture['arrays_npz_base64']);(folder/'all_arrays.npz').write_bytes(npz);arrays=np.load(io.BytesIO(npz),allow_pickle=False);assert meta['diagnostic_prior_mode']==p['mode'] and meta['source_logit_file_sha256']==sha(p['logits']);row.update(status=meta['status'],predict_inst_calls=meta['predict_inst_calls'],source_error=meta.get('error'),all_native_low_res_logits_retained=meta['all_native_low_res_logits_retained'])
            for i,call in enumerate(meta['prior_calls']):
                original=arrays[f'{i}_original_hard_mask_input'];actual=arrays[f'{i}_actual_mask_input'];soft=np.load(p['logits'],allow_pickle=False);expected=original if p['mode']=='hard' else cv2.resize(soft,(original.shape[2],original.shape[1]),interpolation=cv2.INTER_LINEAR)[None];assert np.array_equal(actual,expected) and hashlib.sha256(actual.tobytes()).hexdigest()==call['actual_mask_input']['sha256'];assert call['multimask_output'] is False
            if meta.get('selected_index_observed') and meta['raw_capture_complete']:
                i=meta['selected_prediction_call'];j=meta['selected_index'];raw=arrays[f'{i}_masks'][j].squeeze();assert np.isin(raw,[0,1]).all();x1,y1,x2,y2=p['box'];assert raw.shape==(y2-y1,x2-x1);full=np.zeros((1536,1920),np.uint8);full[y1:y2,x1:x2]=(raw!=0);assert int(full.sum())==int((raw!=0).sum());np.save(folder/'complete_full_mask.npy',full);Image.fromarray((raw!=0).astype('uint8')*255).save(folder/'raw.png');border=bool(raw[0].any() or raw[-1].any() or raw[:,0].any() or raw[:,-1].any());agree=all(bool(raw[q['y'],q['x']])==bool(q['label']) for q in p['points']);row.update(raw_pixels=int(full.sum()),touches_border=border,point_pixels_agree=agree,mask_sha256=hashlib.sha256(full.tobytes()).hexdigest(),valid=meta['status']=='segment_returned' and not border and agree and not response.get('warnings'),raw_array_file=str(folder/'all_arrays.npz'),metadata_file=str(folder/'metadata.json'),prediction_call=i,native_index=j,source_selected=True)
        rows.append(row);save(OUT/'progress.json',{'rows':rows,'requests_completed':len(rows),'transport_attempts_this_execution':attempts});print({'pair':p['pair'],'mode':p['mode'],'status':row['status'],'valid':row['valid'],'pixels':row.get('raw_pixels')},flush=True)
    assert all(sha(p)==h for p,h in plan['protected'].items());save(OUT/'generation_completed.json',{'status':'complete','rows':rows,'unique_real_SAM_calls':sum(r.get('predict_inst_calls',0) for r in rows),'planned_requests':len(rows),'new_masks':sum('mask_sha256' in r for r in rows),'valid_complete_outputs':sum(r['valid'] for r in rows),'transport_attempts_this_execution':attempts,'new_Agent_runs':0,'adopted':False,'all_protected_exact':True})


if __name__=='__main__':prepare() if sys.argv[1]=='prepare' else run()
