"""Source RGB seams propose priors; only untouched native SAM is evaluated.

No reference data or acceptance-area budget is used to choose parents, seams,
points, priors or native alternatives. A paired original-prior arm isolates the
change in the SAM prior. This is an offline candidate trial, not an Agent run.
"""
import base64,hashlib,io,json,shutil,sys,time,urllib.request,urllib.error
from pathlib import Path
import cv2,numpy as np
from PIL import Image,ImageDraw
from run_rgb_layers import ROOT,read,save,sha
from native_prompt_guard import evaluate
BASE=Path('D:/AI/Codex/Home/visualizations/2026/11/04/01a1a56c-0d6f-7d61-8326-d3bbd344d2df')
OUT=BASE/'labelagent_curved_seam_prior_20261108'
SOURCE=Path('D:/Apps/LabelAgent/source/sam3_service/point_segment.py')
SOURCE_SHA='ab8861c57fbd275e8d16b5cc9bfbf734b3e4d7226d02b93d2b0ef9173fc49a81'
def body(row):return {'image':base64.b64encode(Path(row['input']).read_bytes()).decode(),'mask':base64.b64encode(Path(row['prior']).read_bytes()).decode(),'points':row['points'],'include_raw_capture':True}
def key(row):return hashlib.sha256(json.dumps(body(row),sort_keys=True).encode()).hexdigest()
def trace(edge,ylo,yhi,xlo,xhi,seed_y):
    # Bounded first-order dynamic program; the terminal row is not GT selected.
    h,w=edge.shape;ys=np.arange(ylo,yhi+1);cost=1-edge[ylo:yhi+1,xlo:xhi+1]+.12*np.abs(ys[:,None]-seed_y)/max(1,h)
    state=cost[:,0].copy();parents=np.zeros((len(ys),xhi-xlo+1),np.int16)
    for x in range(1,cost.shape[1]):
        choices=np.full((3,len(ys)),np.inf,np.float32);choices[1]=state;choices[0,1:]=state[:-1]+.06;choices[2,:-1]=state[1:]+.06
        move=choices.argmin(0);parents[:,x]=np.arange(len(ys))+move-1;state=cost[:,x]+choices[move,np.arange(len(ys))]
    row=int(state.argmin());path=np.zeros(cost.shape[1],np.int32)
    for x in range(cost.shape[1]-1,-1,-1):path[x]=row+ylo;row=int(parents[row,x])
    return path,float(np.mean(edge[path,np.arange(xlo,xhi+1)]))
def prepare():
    if OUT.exists():raise FileExistsError('Preserve frozen seam experiment')
    assert sha(SOURCE)==SOURCE_SHA
    if shutil.disk_usage(BASE).free<256*2**20:raise RuntimeError('D full evidence guard')
    nodes=[];protected={str(Path(__file__)):sha(__file__),str(SOURCE):SOURCE_SHA}
    for cohort in [ROOT/'outputs/labelagent_aspect_context_cues_20261108',BASE/'labelagent_highres_context_cues_20261108']:
        gp=cohort/'SAM/generation_completed.json';rp=cohort/'review_completed.json';protected[str(gp)]=sha(gp);protected[str(rp)]=sha(rp);reviews=read(rp)['rows']
        for r in read(gp)['rows']:
            if not r['valid'] or r.get('touches_border'):continue
            raw=np.load(r['raw_array_file'],allow_pickle=False)[f"{r['prediction_call']}_masks"][r['native_index']].squeeze()
            if not evaluate(raw,r['points'])['valid']:continue
            rr=next(x for x in reviews if x['partition']==r['index']);assert rr['PAL_subject_evidence']
            nodes.append({'row':r,'group':(r['case'],tuple(r['box'])),'uid':cohort.name+'/'+r['sam_request_sha256'],'neutral_review':rr})
    groups={}
    for n in nodes:groups.setdefault(n['group'],[]).append(n)
    selected=[max(g,key=lambda n:(n['row']['raw_pixels'],n['row']['stock_segment']['score'],n['uid'])) for _,g in sorted(groups.items())][:4]
    OUT.mkdir();rows=[];details=[]
    for number,node in enumerate(selected,1):
        r=node['row'];x1,y1,x2,y2=r['box'];photo=Image.open(r['input']).convert('RGB');full=np.load(Path(r['folder'])/'complete_full_mask.npy',allow_pickle=False);parent=full[y1:y2,x1:x2]!=0;sy,sx=r['points'][0]['y'],r['points'][0]['x'];assert len(r['points'])==1 and parent[sy,sx]
        for f in [r['input'],r['source_image'],str(Path(r['folder'])/'complete_full_mask.npy'),node['neutral_review']['response_file']]:protected[f]=sha(f)
        yy,xx=np.where(parent);left,right=int(xx.min()),int(xx.max());top,bottom=int(yy.min()),int(yy.max());reason='';prior=None;paths={}
        if sy-top>=5 and bottom-sy>=5 and right-left>=16:
            gray=cv2.cvtColor(np.asarray(photo),cv2.COLOR_RGB2GRAY).astype(np.float32);edge=np.abs(cv2.Sobel(gray,cv2.CV_32F,0,1,ksize=3));normalizer=max(1.,float(np.percentile(edge[parent],95)));edge=np.clip(edge/normalizer,0,1)
            upper,uquality=trace(edge,top,sy-4,left,right,sy-4);lower,lquality=trace(edge,sy+4,bottom,left,right,sy+4);band=np.zeros(parent.shape,bool)
            for x,u,l in zip(range(left,right+1),upper,lower):band[u+1:l,x]=True
            proposal=parent&band;n,cc=cv2.connectedComponents(proposal.astype('uint8'),8);prior=cc==cc[sy,sx] if proposal[sy,sx] else None
            if prior is None or not prior.any():reason='anchor_not_in_connected_image_seam_prior'
            else:paths={'left':left,'right':right,'upper':upper.tolist(),'lower':lower.tolist(),'upper_RGB_edge_mean':uquality,'lower_RGB_edge_mean':lquality,'normalizer_p95':normalizer}
        else:reason='insufficient_source_parent_context_above_below_anchor'
        folder=OUT/'parents'/f'{number:02d}';folder.mkdir(parents=True);Image.fromarray(parent.astype('uint8')*255).save(folder/'original_prior.png');debug=photo.copy();draw=ImageDraw.Draw(debug)
        if paths:
            draw.line(list(zip(range(left,right+1),paths['upper'])),fill='yellow',width=1);draw.line(list(zip(range(left,right+1),paths['lower'])),fill='cyan',width=1)
        draw.ellipse((sx-2,sy-2,sx+2,sy+2),outline='red');debug.resize((photo.width*3,photo.height*3)).save(folder/'source_RGB_seams.png')
        if prior is not None and prior.any() and not reason:Image.fromarray(prior.astype('uint8')*255).save(folder/'curved_prior.png')
        details.append({'parent':number,'source_uid':node['uid'],'case':r['case'],'box':r['box'],'source_positive':r['source_point'],'original_source_mask_pixels':int(parent.sum()),'curved_prior_pixels':int(prior.sum()) if prior is not None else None,'seams':paths,'not_a_returned_output_mask':True,'skip_reason':reason,'neutral_positive_response':node['neutral_review']['response_file'],'neutral_positive_response_sha256':node['neutral_review']['response_sha256']})
        if reason:continue
        for arm in ['original','curved']:
            row={k:r[k] for k in ['case','box','input','source_image','source_point','points','teacher_key']};row.update(index=len(rows)+1,parent=number,arm=arm,prior=str(folder/(arm+'_prior.png')),parent_source_uid=node['uid'],kind='real',positive_subject_previously_reviewed_not_whole_base_confirmation=True);row['key']=key(row);rows.append(row);protected[row['prior']]=sha(row['prior'])
    assert rows and len(rows)<=8 and sum(len(r['points']) for r in rows)<=32
    save(OUT/'frozen_plan.json',{'rows':rows,'parents':details,'protected':protected,'source_sha256':SOURCE_SHA,'source_nodes':len(nodes),'selection':'one largest original complete source SAM parent per case and identical source box, model score and UID ties; all groups up to4; no reference or acceptance area','paired_only_change':'original complete source prior versus source RGB dynamic-program seam prior; identical real RGB, original positive point, source and original native selection','seam_rule':'horizontal RGB Sobel evidence clipped by source-only95th percentile; unit-step dynamic programming with .06 slope cost and .12 normalized distance from seed; upper/lower constrained4px away; source parent connected component containing original positive','points':sum(len(r['points']) for r in rows),'max_requests':len(rows),'max_Agent_rounds':3,'new_Agent_runs':0,'all_native_outputs_saved_untrimmed':True,'prior_geometry_not_output_edit':True,'no_reference_or_acceptance_area_for_generation':True,'all_original_final_gates_required':True,'adopted':False});print({'parents':len(details),'planned_SAM':len(rows),'points':sum(len(r['points']) for r in rows),'new_model_calls':0},flush=True)
def run():
    if (OUT/'generation_completed.json').exists():raise FileExistsError('Preserve completed native results')
    p=read(OUT/'frozen_plan.json');assert all(sha(f)==h for f,h in p['protected'].items());rows=[];fresh=0
    for r in p['rows']:
        assert key(r)==r['key'];folder=OUT/'captures'/r['key'];folder.mkdir(parents=True,exist_ok=True);file=folder/'response.json'
        if not file.exists():
            for _ in range(600):
                if (OUT/'STOP').exists():raise RuntimeError('Explicit STOP')
                try:
                    req=urllib.request.Request('http://127.0.0.1:8766/segment_points',data=json.dumps(body(r)).encode(),headers={'Content-Type':'application/json'})
                    with urllib.request.urlopen(req,timeout=600) as f:d=json.load(f)
                    fresh+=1;break
                except urllib.error.HTTPError as e:
                    if e.code==503:time.sleep(.2);continue
                    d=json.loads(e.read());fresh+=1;break
            else:raise RuntimeError('Shared manager busy')
            save(file,d)
        d=read(file);capture=d['raw_capture'];meta=capture['metadata'];assert meta['source_sha256']==SOURCE_SHA and meta['image_sha256']==sha(r['input']) and meta['prior_mask_sha256']==sha(r['prior']) and meta['points']==r['points'];save(folder/'metadata.json',meta);raw=base64.b64decode(capture['arrays_npz_base64']);(folder/'all_arrays.npz').write_bytes(raw);a=np.load(io.BytesIO(raw),allow_pickle=False);row={**r,'folder':str(folder),'raw_array_file':str(folder/'all_arrays.npz'),'metadata_file':str(folder/'metadata.json'),'valid':False,'response_sha256':sha(file)}
        if meta.get('raw_capture_complete') and meta.get('selected_index_observed'):
            local=a[f"{meta['selected_prediction_call']}_masks"][meta['selected_index']].squeeze();assert np.isin(local,[0,1]).all();mask=(local!=0).astype('uint8');x1,y1,x2,y2=r['box'];assert mask.shape==(y2-y1,x2-x1);full=np.zeros((1536,1920),np.uint8);full[y1:y2,x1:x2]=mask;np.save(folder/'complete_full_mask.npy',full);Image.fromarray(mask*255).save(folder/'raw.png');guard=evaluate(mask,r['points']);border=bool(mask[0].any() or mask[-1].any() or mask[:,0].any() or mask[:,-1].any());row.update(valid=meta['status']=='segment_returned' and not border and guard['valid'] and not d.get('warnings'),raw_pixels=int(mask.sum()),touches_border=border,native_prompt_guard=guard,mask_sha256=hashlib.sha256(full.tobytes()).hexdigest(),native_index=meta['selected_index'],prediction_call=meta['selected_prediction_call'],source_selected=True,no_output_pixel_clipping=True)
        rows.append(row);save(OUT/'progress.json',{'rows':rows,'fresh_requests_this_execution':fresh});print({'parent':r['parent'],'arm':r['arm'],'pixels':row.get('raw_pixels'),'valid':row['valid']},flush=True)
    assert all(sha(f)==h for f,h in p['protected'].items());save(OUT/'generation_completed.json',{'status':'complete','rows':rows,'actual_new_SAM_calls':len(rows),'fresh_requests_this_execution':fresh,'all_full_native_masks_untouched':True,'new_Agent_runs':0,'adopted':False})
if __name__=='__main__':prepare() if sys.argv[1]=='prepare' else run()
