"""Cross-crop source-anchor brackets constrain real RGB boundary paths.

Candidates are hypotheses. No reference files, projected points or acceptance
area are read for generation. Native output bitmaps are never cut or edited.
"""
import base64,hashlib,io,json,shutil,sys,time,urllib.request,urllib.error
from pathlib import Path
import cv2,numpy as np
from PIL import Image,ImageDraw
from run_rgb_layers import ROOT,read,save,sha
from native_prompt_guard import evaluate
import run_curved_seam_prior as previous
BASE=previous.BASE;OUT=BASE/'labelagent_anchor_bracket_prior_20261108'
def orientation(photo,mask):
    gray=cv2.cvtColor(np.asarray(photo),cv2.COLOR_RGB2GRAY);edge=cv2.Canny(gray,50,150);support=cv2.dilate(mask.astype('uint8'),np.ones((7,7),np.uint8));edge[~support.astype(bool)]=0
    raw=cv2.HoughLinesP(edge,1,np.pi/180,15,minLineLength=max(16,photo.width//8),maxLineGap=4);lines=[]
    for x1,y1,x2,y2 in ([] if raw is None else raw[:,0].tolist()):
        if x2==x1:continue
        slope=(y2-y1)/(x2-x1);length=float(np.hypot(x2-x1,y2-y1))
        if abs(slope)<=np.tan(np.pi/6):lines.append({'points':[x1,y1,x2,y2],'slope':slope,'length':length})
    if not lines:return 0.,{'lines':[],'fallback_horizontal_no_visible_line':True}
    ordered=sorted(lines,key=lambda r:r['slope']);half=sum(r['length'] for r in ordered)/2;acc=0
    for row in ordered:
        acc+=row['length']
        if acc>=half:return row['slope'],{'lines':lines,'fallback_horizontal_no_visible_line':False}
def trace(edge,left,right,low,high,seed_line):
    h,w=edge.shape;ys=np.arange(h)[:,None];xs=np.arange(left,right+1);cost=1-edge[:,left:right+1]+.12*np.abs(ys-seed_line[None,:])/max(1,h)
    cost[(ys<low[None,:])|(ys>high[None,:])]=np.inf;state=cost[:,0].copy();parents=np.zeros(cost.shape,np.int16)
    for col in range(1,cost.shape[1]):
        choices=np.full((3,h),np.inf,np.float32);choices[1]=state;choices[0,1:]=state[:-1]+.06;choices[2,:-1]=state[1:]+.06;move=choices.argmin(0);parents[:,col]=np.arange(h)+move-1;state=cost[:,col]+choices[move,np.arange(h)]
    if not np.isfinite(state).any():return None
    pos=int(state.argmin());path=np.zeros(len(xs),np.int32)
    for col in range(len(xs)-1,-1,-1):path[col]=pos;pos=int(parents[pos,col])
    assert np.all(path>=low) and np.all(path<=high)
    return path
def prepare():
    if OUT.exists():raise FileExistsError('Preserve cross-crop bracket protocol')
    assert sha(previous.SOURCE)==previous.SOURCE_SHA
    if shutil.disk_usage(BASE).free<256*2**20:raise RuntimeError('Full output reserve')
    protected={str(Path(__file__)):sha(__file__),str(Path(previous.__file__)):sha(previous.__file__),str(previous.SOURCE):previous.SOURCE_SHA};nodes=[]
    for cohort in [ROOT/'outputs/labelagent_aspect_context_cues_20261108',BASE/'labelagent_highres_context_cues_20261108']:
        gf=cohort/'SAM/generation_completed.json';rf=cohort/'review_completed.json';protected[str(gf)]=sha(gf);protected[str(rf)]=sha(rf);review=read(rf)['rows']
        for r in read(gf)['rows']:
            if not r['valid'] or r.get('touches_border'):continue
            a=np.load(r['raw_array_file'],allow_pickle=False);local=a[f"{r['prediction_call']}_masks"][r['native_index']].squeeze()
            if not evaluate(local,r['points'])['valid']:continue
            rr=next(x for x in review if x['partition']==r['index']);assert rr['PAL_subject_evidence'] and len(r['points'])==1
            path=Path(r['folder'])/'complete_full_mask.npy';mask=np.load(path,allow_pickle=False)!=0;assert hashlib.sha256(mask.astype('uint8').tobytes()).hexdigest()==r['mask_sha256'];point=r['source_point'];assert mask[point[1],point[0]]
            nodes.append({'row':r,'mask':mask,'point':point,'uid':cohort.name+'/'+r['sam_request_sha256'],'review':rr,'source_image_sha256':sha(r['source_image'])})
            for f in [str(path),r['input'],r['source_image'],r['raw_array_file'],r['metadata_file'],rr['response_file']]:protected[f]=sha(f)
    candidates=[];graph=[]
    for node in nodes:
        r=node['row'];x1,y1,x2,y2=r['box'];photo=Image.open(r['input']).convert('RGB');parent=node['mask'][y1:y2,x1:x2];slope,lines=orientation(photo,parent);tx,ty=node['point'];target_v=ty-slope*tx;neighbors=[]
        for other in nodes:
            if node['uid']==other['uid'] or r['case']!=other['row']['case'] or node['source_image_sha256']!=other['source_image_sha256']:continue
            nx,ny=other['point'];v=ny-slope*nx;gap=v-target_v
            if abs(gap)<9 or abs(gap)>y2-y1 or other['mask'][max(0,ty-3):ty+4,max(0,tx-3):tx+4].any():continue
            if not other['mask'][max(0,ny-3):ny+4,max(0,nx-3):nx+4].all():continue
            overlap=int((node['mask']&other['mask']).sum())
            if overlap<9 or not x1<=nx<x2:continue
            neighbors.append({'uid':other['uid'],'point':[nx,ny],'rectified_v':v,'gap':gap,'side':'above' if gap<0 else 'below','overlap_pixels':overlap,'source_crop':other['row']['box'],'different_source_crop':other['row']['box']!=r['box']})
        chosen=[]
        for side in ['above','below']:
            group=[n for n in neighbors if n['side']==side]
            if group:chosen.append(min(group,key=lambda n:(abs(n['gap']),-n['overlap_pixels'],n['uid'])))
        entry={'target_uid':node['uid'],'target_point':[tx,ty],'estimated_slope':slope,'target_rectified_v':target_v,'all_neighbor_hypotheses':neighbors,'chosen_neighbors':chosen};graph.append(entry)
        if chosen:candidates.append({'node':node,'neighbors':chosen,'slope':slope,'line_diagnostic':lines,'graph':entry})
    selected=[];seen=set()
    for candidate in sorted(candidates,key=lambda c:(-len(c['neighbors']),-sum(n['different_source_crop'] for n in c['neighbors']),-c['node']['row']['stock_segment']['score'],c['node']['uid'])):
        node=candidate['node'];signature=(node['row']['case'],node['row']['mask_sha256'])
        if signature in seen:continue
        seen.add(signature);selected.append(candidate)
        if len(selected)==4:break
    OUT.mkdir();rows=[];details=[]
    for number,c in enumerate(selected,1):
        node=c['node'];r=node['row'];x1,y1,x2,y2=r['box'];photo=Image.open(r['input']).convert('RGB');mask=node['mask'][y1:y2,x1:x2];sy,sx=r['points'][0]['y'],r['points'][0]['x'];ys,xs=np.where(mask);left,right=int(xs.min()),int(xs.max());top,bottom=int(ys.min()),int(ys.max());xx=np.arange(left,right+1);slope=c['slope'];level=sy-slope*sx;upper_low=np.full(len(xx),top);upper_high=np.floor(level+slope*xx-4).astype(int);lower_low=np.ceil(level+slope*xx+4).astype(int);lower_high=np.full(len(xx),bottom)
        for neighbor in c['neighbors']:
            nx,ny=neighbor['point'];neighbor_level=ny-y1-slope*(nx-x1)
            if neighbor['side']=='above':upper_low=np.maximum(upper_low,np.ceil(neighbor_level+slope*xx+4).astype(int))
            else:lower_high=np.minimum(lower_high,np.floor(neighbor_level+slope*xx-4).astype(int))
        gray=cv2.cvtColor(np.asarray(photo),cv2.COLOR_RGB2GRAY).astype('float32');edge=np.abs(cv2.Sobel(gray,cv2.CV_32F,0,1,ksize=3));scale=max(1.,float(np.percentile(edge[mask],95)));edge=np.clip(edge/scale,0,1);upper=trace(edge,left,right,upper_low,upper_high,level+slope*xx-4);lower=trace(edge,left,right,lower_low,lower_high,level+slope*xx+4);folder=OUT/'parents'/f'{number:02d}';folder.mkdir(parents=True);reason='';proposal=None
        if upper is None or lower is None:reason='no_continuous_RGB_path_within_source_anchor_brackets'
        else:
            band=np.zeros(mask.shape,bool)
            for x,u,l in zip(range(left,right+1),upper,lower):band[u+1:l,x]=True
            area=band&mask
            if not area[sy,sx]:reason='anchor_not_inside_source_prior'
            else:
                _,cc=cv2.connectedComponents(area.astype('uint8'),8);proposal=cc==cc[sy,sx]
        Image.fromarray(mask.astype('uint8')*255).save(folder/'original_prior.png');debug=photo.copy();draw=ImageDraw.Draw(debug)
        if upper is not None:draw.line(list(zip(range(left,right+1),upper)),fill='yellow',width=1)
        if lower is not None:draw.line(list(zip(range(left,right+1),lower)),fill='cyan',width=1)
        for point,color in [(node['point'],'red')]+[(n['point'],'lime') for n in c['neighbors']]:
            x,y=point[0]-x1,point[1]-y1;draw.ellipse((x-3,y-3,x+3,y+3),outline=color,width=2)
        debug.resize((photo.width*3,photo.height*3)).save(folder/'source_RGB_anchor_paths.png')
        detail={'parent':number,'target_uid':node['uid'],'case':r['case'],'box':r['box'],'original_positive':node['point'],'source_review':node['review'],'neighbor_hypotheses':c['neighbors'],'estimated_slope':slope,'line_diagnostic':c['line_diagnostic'],'upper':None if upper is None else upper.tolist(),'lower':None if lower is None else lower.tolist(),'left':left,'right':right,'upper_limits':[upper_low.tolist(),upper_high.tolist()],'lower_limits':[lower_low.tolist(),lower_high.tolist()],'original_prior_pixels':int(mask.sum()),'bracket_prior_pixels':None if proposal is None else int(proposal.sum()),'skip_reason':reason,'neighbors_not_human_confirmed_instances':True};details.append(detail)
        if reason:continue
        Image.fromarray(proposal.astype('uint8')*255).save(folder/'bracket_prior.png')
        for arm in ['original','bracket']:
            row={k:r[k] for k in ['case','box','input','source_image','source_point','points','teacher_key']};row.update(index=len(rows)+1,parent=number,arm=arm,prior=str(folder/(arm+'_prior.png')),source_parent_uid=node['uid'],kind='real',no_generation_reference_or_area=True);row['key']=previous.key(row);row['exact_previous_capture']=None
            for old in read(previous.OUT/'generation_completed.json')['rows']:
                if previous.key(old)==row['key'] and sha(old['input'])==sha(row['input']) and sha(old['prior'])==sha(row['prior']):row['exact_previous_capture']=str(Path(old['folder'])/'response.json');protected[row['exact_previous_capture']]=sha(row['exact_previous_capture']);break
            rows.append(row);protected[row['prior']]=sha(row['prior'])
    assert rows and len(rows)<=8 and sum(len(r['points']) for r in rows)<=32
    save(OUT/'frozen_plan.json',{'rows':rows,'parents':details,'graph':graph,'source_nodes':len(nodes),'protected':protected,'source_sha256':previous.SOURCE_SHA,'selection':'source cross-crop overlap; neighbor excludes original target7x7 and contains own7x7; rectified anchor gap9..cropheight; nearest above/below, rank dual-sided and independent crops then stock score; deduplicate exact native bitmap','orientation':'length-weighted median original RGB Hough edge slope limited30deg, horizontal fallback explicit','prior_generation':'RGB dynamic paths constrained between rectified source anchor lines plus original4px margins; native parent connected component containing unchanged original positive','max_requests':len(rows),'max_new_requests':sum(r['exact_previous_capture'] is None for r in rows),'points':sum(len(r['points']) for r in rows),'same_RGB_points_and_original_source_selection_per_pair':True,'max_Agent_rounds':3,'new_Agent_runs':0,'no_reference_or_acceptance_area_for_generation':True,'all_final_gates_unchanged':True,'no_output_mask_clipping':True,'adopted':False});print({'source_nodes':len(nodes),'parent_hypotheses':len(selected),'prepared_pairs':len(rows)//2,'planned_fresh_SAM':sum(r['exact_previous_capture'] is None for r in rows),'exact_old_controls':sum(r['exact_previous_capture'] is not None for r in rows),'new_model_calls':0},flush=True)
def run():
    if (OUT/'generation_completed.json').exists():raise FileExistsError('Preserve completed prior trial')
    p=read(OUT/'frozen_plan.json');assert all(sha(f)==h for f,h in p['protected'].items());rows=[];fresh=0
    for r in p['rows']:
        assert previous.key(r)==r['key'];folder=OUT/'captures'/r['key'];folder.mkdir(parents=True,exist_ok=True);file=folder/'response.json'
        if not file.exists():
            if r['exact_previous_capture']:shutil.copy2(r['exact_previous_capture'],file)
            else:
                for _ in range(600):
                    if (OUT/'STOP').exists():raise RuntimeError('Explicit STOP')
                    try:
                        req=urllib.request.Request('http://127.0.0.1:8766/segment_points',data=json.dumps(previous.body(r)).encode(),headers={'Content-Type':'application/json'})
                        with urllib.request.urlopen(req,timeout=600) as f:d=json.load(f)
                        fresh+=1;break
                    except urllib.error.HTTPError as e:
                        if e.code==503:time.sleep(.2);continue
                        d=json.loads(e.read());fresh+=1;break
                else:raise RuntimeError('Shared manager busy')
                save(file,d)
        d=read(file);capture=d['raw_capture'];meta=capture['metadata'];assert meta['source_sha256']==previous.SOURCE_SHA and meta['image_sha256']==sha(r['input']) and meta['prior_mask_sha256']==sha(r['prior']) and meta['points']==r['points'];save(folder/'metadata.json',meta);raw=base64.b64decode(capture['arrays_npz_base64']);(folder/'all_arrays.npz').write_bytes(raw);a=np.load(io.BytesIO(raw),allow_pickle=False);row={**r,'folder':str(folder),'raw_array_file':str(folder/'all_arrays.npz'),'metadata_file':str(folder/'metadata.json'),'valid':False,'response_sha256':sha(file),'new_SAM_request':r['exact_previous_capture'] is None}
        if meta.get('raw_capture_complete') and meta.get('selected_index_observed'):
            mask=(a[f"{meta['selected_prediction_call']}_masks"][meta['selected_index']].squeeze()!=0).astype('uint8');x1,y1,x2,y2=r['box'];assert mask.shape==(y2-y1,x2-x1);full=np.zeros((1536,1920),np.uint8);full[y1:y2,x1:x2]=mask;np.save(folder/'complete_full_mask.npy',full);Image.fromarray(mask*255).save(folder/'raw.png');guard=evaluate(mask,r['points']);border=bool(mask[0].any() or mask[-1].any() or mask[:,0].any() or mask[:,-1].any());row.update(valid=meta['status']=='segment_returned' and not border and guard['valid'] and not d.get('warnings'),raw_pixels=int(mask.sum()),touches_border=border,native_prompt_guard=guard,mask_sha256=hashlib.sha256(full.tobytes()).hexdigest(),native_index=meta['selected_index'],prediction_call=meta['selected_prediction_call'],source_selected=True,no_output_pixel_clipping=True)
        rows.append(row);save(OUT/'progress.json',{'rows':rows,'fresh_requests_this_execution':fresh});print({'parent':r['parent'],'arm':r['arm'],'pixels':row.get('raw_pixels'),'valid':row['valid'],'cached':not row['new_SAM_request']},flush=True)
    assert all(sha(f)==h for f,h in p['protected'].items());save(OUT/'generation_completed.json',{'status':'complete','rows':rows,'actual_new_SAM_calls':sum(r['new_SAM_request'] for r in rows),'fresh_requests_this_execution':fresh,'exact_previous_requests_reused':sum(not r['new_SAM_request'] for r in rows),'all_full_native_masks_untouched':True,'new_VLM_calls':0,'new_Agent_runs':0,'adopted':False})
if __name__=='__main__':prepare() if sys.argv[1]=='prepare' else run()
