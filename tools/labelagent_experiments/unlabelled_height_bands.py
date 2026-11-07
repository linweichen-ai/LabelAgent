"""Sensor-only height hypotheses; never reads boxes, owners or labelled points."""
import hashlib,json
from pathlib import Path
import numpy as np,yaml
from PIL import Image
from run_rgb_layers import ROOT,read,save,sha
OUT=ROOT/'outputs/labelagent_unlabelled_height_bands_20261108'
PLAN=ROOT/'outputs/labelagent_zoom_teacher_20261108/frozen_plan.json'

def mixture(z,k):
    z=np.asarray(z,dtype=float);means=np.quantile(z,(np.arange(k)+.5)/k);variance=np.full(k,max(float(z.var()),1e-4));weight=np.full(k,1/k)
    for _ in range(60):
        log=-.5*((z[:,None]-means)**2/variance+np.log(2*np.pi*variance))+np.log(np.maximum(weight,1e-12));mx=log.max(1);prob=np.exp(log-mx[:,None]);prob/=prob.sum(1,keepdims=True);mass=prob.sum(0);means=(prob*z[:,None]).sum(0)/np.maximum(mass,1e-9);variance=np.maximum((prob*(z[:,None]-means)**2).sum(0)/np.maximum(mass,1e-9),1e-4);weight=mass/len(z)
    log=-.5*((z[:,None]-means)**2/variance+np.log(2*np.pi*variance))+np.log(np.maximum(weight,1e-12));mx=log.max(1);ll=float((mx+np.log(np.exp(log-mx[:,None]).sum(1))).sum());labels=log.argmax(1);counts=np.bincount(labels,minlength=k)
    return {'k':k,'means':means.tolist(),'variances':variance.tolist(),'weights':weight.tolist(),'counts':counts.tolist(),'BIC':float((3*k-1)*np.log(len(z))-2*ll),'labels':labels}

def project(xyz,camera):
    intr,trans,width,height=camera;cam=xyz@trans[:3,:3].T+trans[:3,3];valid=np.isfinite(cam).all(1)&(cam[:,2]>.2);source=np.flatnonzero(valid);cam=cam[valid];uv=cam[:,:2]/cam[:,2,None]*intr[:2]+intr[2:];inside=(uv[:,0]>=0)&(uv[:,0]<width-.5)&(uv[:,1]>=0)&(uv[:,1]<height-.5);source,cam,uv=source[inside],cam[inside],uv[inside];pix=np.floor(uv+.5).astype(np.int32);cols=(width+7)//8;cells=(pix[:,1]//8)*cols+pix[:,0]//8;depth=np.full(((height+7)//8)*cols,np.inf);np.minimum.at(depth,cells,cam[:,2]);keep=cam[:,2]<=depth[cells]+.15;return source[keep],pix[keep]

def prepare():
    if OUT.exists():raise FileExistsError('Preserve sensor-only hypothesis cohort')
    cfg=read(ROOT/'task_workspaces/labelagent_point_ids_20261107/config.json');dataset=Path(cfg['dataset']);manifest=read(dataset/'import_manifest.json');stamp=str(manifest['timestamps_ns'][76]);views=read(PLAN)['views'];allowed=[f'lidar/{stamp}.pcd','calib/lidar2camera.yaml','calib/camera0_calib.yaml','calib/camera1_calib.yaml'];protected={str(PLAN):sha(PLAN),str(Path(__file__)):sha(__file__)}
    for row in views:
        for key in ['input','source_image','prior']:protected[row[key]]=sha(row[key])
    OUT.mkdir();save(OUT/'frozen_plan.json',{'views':views,'dataset':str(dataset),'stamp':stamp,'allowed_sensor_files':{r:manifest['extracted_sha256'][r] for r in allowed},'protected':protected,'height_mixture_k':[1,2,3,4],'mixture_selection':'minimum BIC among fits with at least8 observed sensor points per component','maximum_components_followed_per_view':2,'component_ranking':'observed unlabelled sensor count then mean height','positive_rule':'observed projected point nearest source component median pixel, inside original MF native prior','negative_rule':'nearest other inferred height component projected point at least8px away; hypothesis only','output':'one positive and at most one hypothesized negative; no bitmap synthesis','no_annotation_or_reference_coordinate_access':True,'sensor_whitelist_loader_required':True,'not_same_input_as_RGB_only_baseline':True,'correlated_sensor_agreement_not_independent_ground_truth':True,'new_Agent_rounds':0,'Agent_round_limit':3,'no_output_mask_clipping':True,'original_quality_gates_unchanged':True});print({'frozen_views':len(views),'allowed_sensor_files':len(allowed),'new_model_calls':0},flush=True)

def run():
    if (OUT/'completed.json').exists():raise FileExistsError('Preserve completed sensor proposals')
    plan=read(OUT/'frozen_plan.json');assert all(sha(f)==h for f,h in plan['protected'].items());access=[];root=Path(plan['dataset'])/'source'
    def sensor(relative):
        assert relative in plan['allowed_sensor_files'];p=(root/relative).resolve();assert p.is_relative_to(root.resolve());raw=p.read_bytes();assert hashlib.sha256(raw).hexdigest()==plan['allowed_sensor_files'][relative];access.append({'relative':relative,'sha256':hashlib.sha256(raw).hexdigest(),'bytes':len(raw)});return raw
    raw=sensor(f"lidar/{plan['stamp']}.pcd");end=raw.index(b'DATA binary\n')+len(b'DATA binary\n');header={line.split()[0]:line.split()[1:] for line in raw[:end].decode('ascii').splitlines() if line.strip() and not line.startswith('#')};names=header['FIELDS'];counts=list(map(int,header.get('COUNT',['1']*len(names))));assert all(x==1 for x in counts);mapping={('F',4):'<f4',('F',8):'<f8',('U',1):'u1',('U',2):'<u2',('U',4):'<u4',('I',1):'i1',('I',2):'<i2',('I',4):'<i4'};dtype=np.dtype([(n,mapping[(t,int(s))]) for n,t,s in zip(names,header['TYPE'],header['SIZE'])]);data=np.frombuffer(raw[end:],dtype=dtype);assert len(data)==int(header['POINTS'][0]);xyz=np.column_stack([data[n] for n in ['x','y','z']]);finite=np.isfinite(xyz).all(1)&(np.linalg.norm(xyz,axis=1)>.001);raw_rows=np.flatnonzero(finite);xyz=xyz[finite];extr=yaml.safe_load(sensor('calib/lidar2camera.yaml'));projections={}
    for cam in [0,1]:
        text=sensor(f'calib/camera{cam}_calib.yaml').decode();intr=yaml.safe_load('\n'.join(l for l in text.splitlines() if not l.startswith('%YAML:')));e=extr[f'lidar_to_camera_{cam}'];assert intr['model_type']=='PINHOLE' and e['calibrated'] is True and all(float(v)==0 for v in intr.get('distortion_parameters',{}).values());params=np.array([intr['projection_parameters'][k] for k in ['fx','fy','cx','cy']],float);trans=np.array(e['transform:'],float).reshape(4,4);assert np.allclose(trans[3],[0,0,0,1]) and np.linalg.norm(trans[:3,:3].T@trans[:3,:3]-np.eye(3))<.01;camera=(params,trans,int(intr['image_width']),int(intr['image_height']));projections[cam]=project(xyz,camera)
    rows=[];diagnostics=[]
    for view in plan['views']:
        cam=int(view['case'][6]);source,pix=projections[cam];x1,y1,x2,y2=view['box'];prior=np.asarray(Image.open(view['prior']).convert('L'))>127;assert prior.shape==(y2-y1,x2-x1);inside=(pix[:,0]>=x1)&(pix[:,0]<x2)&(pix[:,1]>=y1)&(pix[:,1]<y2);ii=np.flatnonzero(inside);ii=ii[prior[pix[ii,1]-y1,pix[ii,0]-x1]];selected=source[ii];uv=pix[ii];z=xyz[selected,2];diag={'teacher_key':view['teacher_key'],'observed_sensor_points_in_MF_prior':len(ii),'fits':[]};diagnostics.append(diag)
        if len(ii)<16:diag['reason']='fewer_than16_unlabelled_sensor_returns';continue
        fits=[]
        for k in plan['height_mixture_k']:
            fit=mixture(z,k);diag['fits'].append({key:value for key,value in fit.items() if key!='labels'})
            if min(fit['counts'])>=8:fits.append(fit)
        fit=min(fits,key=lambda f:(f['BIC'],f['k']));labels=fit['labels'];diag['chosen_k']=fit['k'];diag['chosen_BIC']=fit['BIC'];np.savez_compressed(OUT/(view['teacher_key']+'_sensor_components.npz'),source_rows=raw_rows[selected],xyz=xyz[selected],pixels=uv,component=labels)
        if fit['k']==1:diag['reason']='BIC_does_not_support_separate_height_components';continue
        for component in sorted(range(fit['k']),key=lambda j:(-fit['counts'][j],fit['means'][j]))[:2]:
            own=np.flatnonzero(labels==component);center=np.median(uv[own],axis=0);p=int(own[np.argmin(((uv[own]-center)**2).sum(1))]);point=uv[p];other=np.flatnonzero((labels!=component)&(((uv-point)**2).sum(1)>=64));points=[{'x':int(point[0]-x1),'y':int(point[1]-y1),'label':1}];negative=None
            if len(other):
                n=int(other[np.argmin(((uv[other]-point)**2).sum(1))]);negative=uv[n].tolist();points.append({'x':int(uv[n,0]-x1),'y':int(uv[n,1]-y1),'label':0})
            rows.append({**view,'index':len(rows)+1,'source_partition_index':len(rows)+1,'component':component,'unlabelled_component_mean_z':fit['means'][component],'observed_component_count':fit['counts'][component],'points':points,'source_point':point.tolist(),'hypothesized_negative_source_point':negative,'not_reference_coordinates':True,'not_confirmed_independent_base':True})
    save(OUT/'completed.json',{'status':'complete_sensor_only_proposals_not_model_results','rows':rows,'diagnostics':diagnostics,'sensor_files_read':access,'reference_files_opened':0,'new_SAM_calls':0,'new_VLM_calls':0,'new_Agent_runs':0,'not_adopted':True});print({'sensor_files':len(access),'source_proposals':len(rows),'views':len(diagnostics),'chosen_height_components':[d.get('chosen_k') for d in diagnostics],'new_model_calls':0},flush=True)

if __name__=='__main__':import sys;{'prepare':prepare,'run':run}[sys.argv[1]]()
