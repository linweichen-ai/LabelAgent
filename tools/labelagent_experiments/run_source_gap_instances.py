"""Discover source-parent gaps between disjoint child hypotheses, not GT seeds."""
import base64,hashlib,itertools,json,os,shutil,sys
from pathlib import Path
import cv2,numpy as np
from PIL import Image,ImageDraw
from run_rgb_layers import ROOT,read,save,sha
from native_prompt_guard import evaluate
from paired_source_cue import body
import run_anchor_bracket_prior as anchors
import run_curved_seam_prior as captured
BASE=anchors.BASE;OUT=BASE/'labelagent_source_gap_instances_20261108'
def prepare():
    if OUT.exists():raise FileExistsError('Preserve frozen gap-instance protocol')
    if shutil.disk_usage(BASE).free<256*2**20:raise RuntimeError('Full raw-output reserve')
    nodes=[];protected={str(Path(__file__)):sha(__file__),str(Path(anchors.__file__)):sha(anchors.__file__),str(Path(captured.__file__)):sha(captured.__file__)};model=None
    for folder in [ROOT/'outputs/labelagent_aspect_context_cues_20261108',BASE/'labelagent_highres_context_cues_20261108']:
        plan=read(folder/'frozen_plan.json');model=plan['model_config_sha256'];gf=folder/'SAM/generation_completed.json';rf=folder/'review_completed.json';protected[str(gf)]=sha(gf);protected[str(rf)]=sha(rf);rv=read(rf)['rows']
        for r in read(gf)['rows']:
            if not r['valid'] or r.get('touches_border'):continue
            a=np.load(r['raw_array_file'],allow_pickle=False);local=a[f"{r['prediction_call']}_masks"][r['native_index']].squeeze()
            if not evaluate(local,r['points'])['valid']:continue
            rr=next(x for x in rv if x['partition']==r['index']);assert rr['PAL_subject_evidence'];f=Path(r['folder'])/'complete_full_mask.npy';mask=np.load(f,allow_pickle=False)!=0;nodes.append({'row':r,'mask':mask,'uid':folder.name+'/'+r['sam_request_sha256'],'review':rr})
            for f in [str(f),r['input'],r['source_image'],r['raw_array_file'],r['metadata_file'],rr['response_file']]:protected[f]=sha(f)
    hypotheses=[]
    for a,b in itertools.combinations(nodes,2):
        ar,br=a['row'],b['row']
        if ar['case']!=br['case'] or sha(ar['source_image'])!=sha(br['source_image']):continue
        ax,ay=ar['source_point'];bx,by=br['source_point'];distance=float(np.hypot(ax-bx,ay-by));am,bm=a['mask'],b['mask']
        if not 20<=distance<=96 or abs(ay-by)<12 or am[by,bx] or bm[ay,ax]:continue
        iou=int((am&bm).sum())/max(1,int((am|bm).sum()))
        if iou>.25:continue
        mx,my=round((ax+bx)/2),round((ay+by)/2)
        if (am[my-3:my+4,mx-3:mx+4]|bm[my-3:my+4,mx-3:mx+4]).any():continue
        aa,bb=sorted([np.percentile(np.where(am)[0],[25,75]).tolist(),np.percentile(np.where(bm)[0],[25,75]).tolist()]);coregap=bb[0]-aa[1]
        if coregap<8:continue
        for parent in nodes:
            pr=parent['row'];pm=parent['mask']
            if pr['case']!=ar['case'] or parent['uid'] in [a['uid'],b['uid']] or not pm[my,mx]:continue
            ca=int((pm&am).sum())/int(am.sum());cb=int((pm&bm).sum())/int(bm.sum());ratio=int(pm.sum())/(int(am.sum())+int(bm.sum()))
            if min(ca,cb)<.8 or ratio<1.5:continue
            hypotheses.append({'parent':parent,'a':a,'b':b,'point':[mx,my],'source_core_vertical_gap':float(coregap),'anchor_distance':distance,'child_IoU':iou,'parent_child_coverage':[ca,cb],'parent_to_children_area_ratio':ratio})
    selected=[]
    for h in sorted(hypotheses,key=lambda h:(-h['source_core_vertical_gap'],h['anchor_distance'],h['parent']['uid'],h['a']['uid'],h['b']['uid'])):
        if any(h['parent']['row']['case']==old['parent']['row']['case'] and np.hypot(h['point'][0]-old['point'][0],h['point'][1]-old['point'][1])<24 for old in selected):continue
        selected.append(h)
        if len(selected)==2:break
    assert selected;OUT.mkdir();rows=[];details=[]
    for number,h in enumerate(selected,1):
        parent=h['parent'];r=parent['row'];x1,y1,x2,y2=r['box'];mx,my=h['point'];sx,sy=mx-x1,my-y1;photo=Image.open(r['input']).convert('RGB');mask=parent['mask'][y1:y2,x1:x2];slope,lines=anchors.orientation(photo,mask);level=sy-slope*sx;endpoints=sorted([h['a']['row']['source_point'],h['b']['row']['source_point']],key=lambda q:q[1]-slope*q[0]);above,below=endpoints;uv=above[1]-y1-slope*(above[0]-x1);lv=below[1]-y1-slope*(below[0]-x1);yy,xx=np.where(mask);left,right=int(xx.min()),int(xx.max());xs=np.arange(left,right+1);gray=cv2.cvtColor(np.asarray(photo),cv2.COLOR_RGB2GRAY).astype('float32');edge=np.abs(cv2.Sobel(gray,cv2.CV_32F,0,1,ksize=3));edge=np.clip(edge/max(1.,float(np.percentile(edge[mask],95))),0,1);upper=anchors.trace(edge,left,right,np.maximum(0,np.ceil(uv+slope*xs+4).astype(int)),np.floor(level+slope*xs-4).astype(int),level+slope*xs-4);lower=anchors.trace(edge,left,right,np.ceil(level+slope*xs+4).astype(int),np.minimum(photo.height-1,np.floor(lv+slope*xs-4).astype(int)),level+slope*xs+4)
        folder=OUT/'proposals'/f'{number:02d}';folder.mkdir(parents=True);prior=None
        if upper is not None and lower is not None:
            prior=np.zeros(mask.shape,bool)
            for x,u,l in zip(range(left,right+1),upper,lower):prior[u+1:l,x]=True
            if not prior[sy,sx]:prior=None
        priorfile=folder/'RGB_gap_prior.png'
        if prior is not None:Image.fromarray(prior.astype('uint8')*255).save(priorfile);protected[str(priorfile)]=sha(priorfile)
        source=Image.open(r['source_image']).convert('RGB');bindings=[]
        for name,size in [('near',129),('context',385)]:
            half=size//2;box=[mx-half,my-half,mx+half+1,my+half+1];clip=[max(0,box[0]),max(0,box[1]),min(source.width,box[2]),min(source.height,box[3])];canvas=Image.new('RGB',(size,size),(127,127,127));canvas.paste(source.crop(tuple(clip)),(clip[0]-box[0],clip[1]-box[1]));file=folder/(name+'.png');canvas.resize((512,512),Image.Resampling.LANCZOS).save(file);bindings.append({'image':str(file),'sha256':sha(file),'side':size,'source_canvas_box':box,'clipped_box':clip})
        request=body(*[b['image'] for b in bindings]);key=hashlib.sha256(json.dumps(request,sort_keys=True).encode()).hexdigest();requestfile=OUT/'prepared_requests'/(key+'_PRIVATE.json');save(requestfile,request);view={'new_point':{'x':sx,'y':sy,'label':1},'source_point':[mx,my],'bindings':bindings,'request_file':str(requestfile),'request_sha256':key};points=[{'x':sx,'y':sy,'label':1}]+[{'x':x-x1,'y':y-y1,'label':0} for x,y in endpoints];rows.append({'index':number,'source_partition_index':number,'case':r['case'],'box':r['box'],'input':r['input'],'source_image':r['source_image'],'source_point':[mx,my],'teacher_key':'source_gap_'+str(number),'points':points,'prior':None if prior is None else str(priorfile),'new_point_views':[view],'source_parent_uid':parent['uid'],'source_children_uids':[h['a']['uid'],h['b']['uid']],'no_generation_GT_or_area':True,'negative_endpoint_identities_unconfirmed':True});details.append({k:v for k,v in h.items() if k not in ['parent','a','b']}|{'proposal':number,'source_parent_uid':parent['uid'],'source_children_uids':[h['a']['uid'],h['b']['uid']],'source_endpoints':endpoints,'estimated_slope':slope,'RGB_lines':lines,'RGB_prior_available':prior is not None,'prior_pixels':None if prior is None else int(prior.sum()),'upper':None if upper is None else upper.tolist(),'lower':None if lower is None else lower.tolist(),'left':left,'right':right})
        debug=photo.copy();draw=ImageDraw.Draw(debug)
        for x,y,label in [(sx,sy,1)]+[(p['x'],p['y'],0) for p in points[1:]]:draw.ellipse((x-3,y-3,x+3,y+3),outline='red' if label else 'lime',width=2)
        if upper is not None:draw.line(list(zip(range(left,right+1),upper)),fill='yellow',width=1)
        if lower is not None:draw.line(list(zip(range(left,right+1),lower)),fill='cyan',width=1)
        debug.resize((photo.width*3,photo.height*3)).save(folder/'source_gap_anchors.png')
    save(OUT/'frozen_plan.json',{'rows':rows,'details':details,'source_nodes':len(nodes),'eligible_source_triplets':len(hypotheses),'protected':protected,'model_config_sha256':model,'source_sha256':captured.SOURCE_SHA,'new_instance_selection':'Parent contains>=.8 each of two child masks, size>=1.5 sum; child IoU<=.25, anchors20..96px and dy>=12; midpoint7x7 clear of both children but within parent; vertical core quartile gap>=8; rank core gap desc and distance asc;24px point suppression; at most2 source hypotheses','same_original_neutral_prompt_schema_and_tokens':True,'max_new_VLM_requests':len(rows),'planned_max_SAM_requests':sum(1+(r['prior'] is not None) for r in rows),'planned_point_annotations':sum(len(r['points'])*(1+(r['prior'] is not None)) for r in rows),'source_children_not_confirmed_physical_instances':True,'no_reference_or_area_inputs_for_generation':True,'new_Agent_runs':0,'Agent_round_limit':3,'all_original_gates_unchanged':True,'no_output_mask_clipping':True});print({'source_gap_hypotheses':len(hypotheses),'selected_new_points':[r['source_point'] for r in rows],'max_new_VLM':len(rows),'max_new_SAM':sum(1+(r['prior'] is not None) for r in rows),'new_model_calls':0},flush=True)
def review():
    os.environ.setdefault('LABELAGENT_PAIRED_CUE_OUTPUT',str(ROOT/'outputs/labelagent_paired_source_cue_20261108'));os.environ.setdefault('LABELAGENT_TRAINED_REVIEW_OUTPUT',str(ROOT/'outputs/labelagent_trained_point_review_20261108'))
    import run_base_material as reviewer
    catalogue=[];sources={}
    for base in [ROOT/'outputs',BASE]:
        for file in base.glob('labelagent_*/review_completed.json'):
            d=read(file)
            for row in d.get('rows',[]):
                if all(k in row for k in ['request_sha256','request_file','response_sha256']) and (row.get('response_file') or row.get('response')):catalogue.append(row)
            sources[str(file)]=sha(file)
    cache=OUT/'exact_cache_catalogue/completed.json'
    if not cache.exists():save(cache,{'rows':catalogue,'sources':sources})
    reviewer.OUT=OUT;reviewer.PAIR=cache.parent;reviewer.review()
def prepare_SAM():
    target=OUT/'SAM'
    if target.exists():raise FileExistsError('Preserve complete-input SAM protocol')
    plan=read(OUT/'frozen_plan.json');reviews=read(OUT/'review_completed.json')['rows'];rows=[];admissions=[]
    for r in plan['rows']:
        review=next(x for x in reviews if x['partition']==r['index']);admissions.append({'proposal':r['index'],'source_point':r['source_point'],'PAL_subject_evidence':review['PAL_subject_evidence'],'decision':review['decision']})
        if not review['PAL_subject_evidence']:continue
        for arm in ['point_only','RGB_gap_prior']:
            if arm=='RGB_gap_prior' and r['prior'] is None:continue
            row={k:r[k] for k in ['case','box','input','source_image','source_point','points','teacher_key']};row.update(index=len(rows)+1,parent=r['index'],arm=arm,kind='real',prior=None if arm=='point_only' else r['prior'],negative_endpoint_identities_unconfirmed=True,exact_previous_capture=None);row['key']=request_key(row);rows.append(row)
    assert sum(len(r['points']) for r in rows)<=32;save(target/'frozen_plan.json',{'rows':rows,'admissions':admissions,'source_sha256':captured.SOURCE_SHA,'protected':{str(OUT/'frozen_plan.json'):sha(OUT/'frozen_plan.json'),str(OUT/'review_completed.json'):sha(OUT/'review_completed.json'),str(Path(__file__)):sha(__file__)},'max_new_calls':len(rows),'points':sum(len(r['points']) for r in rows),'new_Agent_runs':0,'Agent_round_limit':3,'all_original_gates_unchanged':True,'no_mask_pixel_changes':True});print({'admitted_hypotheses':len(rows),'points':sum(len(r['points']) for r in rows),'new_model_calls':0},flush=True)
def request_body(row):
    b={'image':base64.b64encode(Path(row['input']).read_bytes()).decode(),'points':row['points'],'include_raw_capture':True}
    if row['prior'] is not None:b['mask']=base64.b64encode(Path(row['prior']).read_bytes()).decode()
    return b
def request_key(row):return hashlib.sha256(json.dumps(request_body(row),sort_keys=True).encode()).hexdigest()
def run():
    # Reuse the audited transport/capture engine, changing only the frozen body
    # serializer to support the declared no-prior comparison arm.
    oldout,oldbody,oldkey=anchors.OUT,captured.body,captured.key
    try:
        anchors.OUT=OUT/'SAM';captured.body=request_body;captured.key=request_key
        # Engine metadata requires expected prior digest; no-prior uses None.
        source=Path(anchors.__file__).read_text('utf-8');before="meta['prior_mask_sha256']==sha(r['prior'])";after="meta['prior_mask_sha256']==(None if r['prior'] is None else sha(r['prior']))";assert source.count(before)==1
        namespace={'__file__':anchors.__file__,'__name__':'source_gap_transport'};exec(compile(source.replace(before,after),anchors.__file__,'exec'),namespace);namespace['OUT']=OUT/'SAM';namespace['run']()
    finally:anchors.OUT=oldout;captured.body=oldbody;captured.key=oldkey
if __name__=='__main__':{'prepare':prepare,'review':review,'prepare_SAM':prepare_SAM,'run':run}[sys.argv[1]]()
