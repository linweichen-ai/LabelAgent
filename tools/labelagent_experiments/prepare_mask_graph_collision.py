"""Source-only asymmetric mask conflicts propose neighbor negatives and anchors."""
import hashlib,json,shutil
from pathlib import Path
import cv2,numpy as np
from PIL import Image
from run_rgb_layers import ROOT,read,save,sha
from native_prompt_guard import evaluate
from paired_source_cue import body
OUT=Path('D:/AI/Codex/Home/visualizations/2026/11/04/01a1a56c-0d6f-7d61-8326-d3bbd344d2df/labelagent_mask_graph_collision_20261108')
LOW=ROOT/'outputs/labelagent_aspect_context_cues_20261108';HIGH=OUT.with_name('labelagent_highres_context_cues_20261108')

def main():
    if OUT.exists():raise FileExistsError('Preserve source mask-conflict protocol')
    if shutil.disk_usage(OUT.parent).free<128*2**20:raise RuntimeError('Source graph raw-output reserve')
    OUT.mkdir();nodes=[];protected={str(Path(__file__)):sha(__file__)}
    for cohort in [LOW,HIGH]:
        generation=read(cohort/'SAM/generation_completed.json');review=read(cohort/'review_completed.json');protected[str(cohort/'SAM/generation_completed.json')]=sha(cohort/'SAM/generation_completed.json');protected[str(cohort/'review_completed.json')]=sha(cohort/'review_completed.json')
        for r in generation['rows']:
            if not r['valid'] or r.get('touches_border'):continue
            rr=next(x for x in review['rows'] if x['partition']==r['index']);assert rr['PAL_subject_evidence'];a=np.load(r['raw_array_file'],allow_pickle=False);local=a[f"{r['prediction_call']}_masks"][r['native_index']].squeeze();guard=evaluate(local,r['points'])
            if not guard['valid']:continue
            assert len(r['points'])==1 and r['points'][0]['label']==1;maskfile=Path(r['folder'])/'complete_full_mask.npy';mask=np.load(maskfile,allow_pickle=False);assert hashlib.sha256(mask.tobytes()).hexdigest()==r['mask_sha256'];x1,y1,x2,y2=r['box'];np.testing.assert_array_equal(mask[y1:y2,x1:x2],local);point=[r['points'][0]['x']+x1,r['points'][0]['y']+y1];assert point==r['source_point'] and mask[point[1],point[0]];key=(r['case'],sha(r['input']),tuple(r['box']));uid=cohort.name+'/'+r['sam_request_sha256'];nodes.append({'row':r,'mask':mask,'point':point,'group':key,'uid':uid,'neutral_response':rr['response_file'],'neutral_response_sha256':rr['response_sha256']})
            for f in [str(maskfile),r['input'],r['source_image'],rr['response_file']]:protected[f]=sha(f)
    candidates=[];graph=[]
    for target in nodes:
        tx,ty=target['point'];possible=[]
        for neighbor in nodes:
            if target['uid']==neighbor['uid'] or target['group']!=neighbor['group']:continue
            if neighbor['mask'][max(0,ty-3):ty+4,max(0,tx-3):tx+4].any():continue
            x1,y1,x2,y2=target['row']['box'];overlap=(target['mask'][y1:y2,x1:x2]!=0)&(neighbor['mask'][y1:y2,x1:x2]!=0);yy,xx=np.indices(overlap.shape);overlap &= (xx+x1-tx)**2+(yy+y1-ty)**2>=64
            if int(overlap.sum())<9:continue
            depth=cv2.distanceTransform(np.pad(overlap.astype('uint8'),1),cv2.DIST_L2,5)[1:-1,1:-1];ly,lx=map(int,np.unravel_index(depth.argmax(),depth.shape));nx,ny=lx+x1,ly+y1;distance=(nx-tx)**2+(ny-ty)**2
            if ny==ty:continue
            possible.append({'node':neighbor,'negative_source_point':[nx,ny],'overlap_pixels':int(overlap.sum()),'distance':distance,'side':'above' if ny<ty else 'below'})
        chosen=[]
        for side in ['above','below']:
            side_rows=[r for r in possible if r['side']==side]
            if side_rows:chosen.append(min(side_rows,key=lambda r:(r['distance'],r['node']['uid'])))
        graph.append({'target':target['uid'],'possible_neighbors':[{'uid':r['node']['uid'],'negative_source_point':r['negative_source_point'],'neighbor_original_positive':r['node']['point'],'overlap_pixels':r['overlap_pixels'],'side':r['side'],'distance_squared':r['distance']} for r in possible],'selected_neighbor_uids':[r['node']['uid'] for r in chosen]})
        if chosen:candidates.append({'target':target,'neighbors':chosen,'conflicts':len(possible)})
    # Priority uses source conflicts and original model score, never reference support.
    selected=sorted(candidates,key=lambda c:(-c['conflicts'],-c['target']['row']['stock_segment']['score'],c['target']['uid']))[:4];rows=[];details=[]
    for rank,c in enumerate(selected,1):
        target=c['target'];r=target['row'];x1,y1,x2,y2=r['box'];positive=dict(r['points'][0]);negative=[{'x':int(n['negative_source_point'][0]-x1),'y':int(n['negative_source_point'][1]-y1),'label':0} for n in c['neighbors']];exclusive=target['mask'][y1:y2,x1:x2].copy().astype(bool)
        for n in c['neighbors']:exclusive &= ~n['node']['mask'][y1:y2,x1:x2].astype(bool)
        assert exclusive[positive['y'],positive['x']];_,cc=cv2.connectedComponents(exclusive.astype('uint8'),8);component=(cc==cc[positive['y'],positive['x']]);ys,xs=np.where(component);extra=[];folder=OUT/'source_sampling'/f'parent{rank:02d}';folder.mkdir(parents=True);Image.fromarray(component.astype('uint8')*255).save(folder/'positive_sampling_region_NOT_OUTPUT.png')
        if len(xs)>=16 and xs.max()-xs.min()>=8:
            for side in [0,1]:
                region=component.copy();mid=(int(xs.min())+int(xs.max()))//2
                if side==0:region[:,mid+1:]=0
                else:region[:,:mid+1]=0
                if not region.any():continue
                distance=cv2.distanceTransform(np.pad(region.astype('uint8'),1),cv2.DIST_L2,5)[1:-1,1:-1];yy,xx=map(int,np.unravel_index(distance.argmax(),distance.shape));q={'x':xx,'y':yy,'label':1}
                if all((q['x']-a['x'])**2+(q['y']-a['y'])**2>=64 for a in [positive]+extra):extra.append(q)
        detail={'parent':rank,'source_parent_uid':target['uid'],'source_positive':target['point'],'neighbor_hypotheses':[{'uid':n['node']['uid'],'negative_source_point':n['negative_source_point'],'neighbor_original_positive':n['node']['point'],'overlap_pixels':n['overlap_pixels'],'side':n['side'],'both_source_masks_contain_collision_point':True,'neighbor_clear_of_target7x7':True} for n in c['neighbors']],'sampling_region_pixels':int(component.sum()),'extra_positive_points':extra,'sampling_region_not_a_segmentation_candidate':True};details.append(detail)
        for arm,additions in [('neighbor_negatives',[]),('exclusive_surface_positives',extra)]:
            if arm=='exclusive_surface_positives' and len(additions)!=2:continue
            points=[positive]+additions+negative;index=len(rows)+1;views=[]
            for j,q in enumerate(additions,1):
                ax,ay=q['x']+x1,q['y']+y1;photo=Image.open(r['source_image']).convert('RGB');bindings=[]
                for name,size in [('near',129),('context',385)]:
                    half=size//2;box=[ax-half,ay-half,ax+half+1,ay+half+1];clip=[max(0,box[0]),max(0,box[1]),min(photo.width,box[2]),min(photo.height,box[3])];im=Image.new('RGB',(size,size),(127,127,127));im.paste(photo.crop(tuple(clip)),(clip[0]-box[0],clip[1]-box[1]));file=folder/f'{arm}_point{j}_{name}.png';im.resize((512,512),Image.Resampling.LANCZOS).save(file);bindings.append({'image':str(file),'sha256':sha(file),'side':size,'source_canvas_box':box,'clipped_box':clip})
                request=body(*[b['image'] for b in bindings]);key=hashlib.sha256(json.dumps(request,sort_keys=True).encode()).hexdigest();file=OUT/'prepared_requests'/(key+'_PRIVATE.json');save(file,request);views.append({'new_point':q,'source_point':[ax,ay],'bindings':bindings,'request_file':str(file),'request_sha256':key})
            rows.append({**r,'index':index,'source_partition_index':index,'source_parent_cue':r['index'],'source_parent_uid':target['uid'],'parent_rank':rank,'arm':arm,'points':points,'new_point_views':views,'new_positive_points':additions,'neighbor_negative_points':negative,'parent_source_neutral_PAL_bound':True,'negative_neighbors_are_hypotheses_not_human_truth':True,'negative_collision_pixels_not_claimed_neutrally_captioned':True,'sampling_mask_never_sent_as_prior_or_output':True})
    assert len(rows)<=8 and sum(len(r['points']) for r in rows)<=32
    save(OUT/'frozen_plan.json',{'rows':rows,'graph':graph,'details':details,'protected':protected,'model_config_sha256':read(LOW/'frozen_plan.json')['model_config_sha256'],'source_nodes':len(nodes),'max_parent_hypotheses':4,'max_unique_SAM_requests':8,'all_planned_point_annotations':sum(len(r['points']) for r in rows),'cumulative_point_budget':32,'max_new_VLM_requests':len({v['request_sha256'] for r in rows for v in r['new_point_views']}),'source_selection':'deepest overlapping source-mask pixel with reciprocal7x7 target-seed exclusion, one nearest collision above and below; rank by source conflict count then original stock score','extra_positive_selection':'connected source target component excluding neighbor source masks, deepest left and right points; sampling only, never a returned mask','no_reference_or_area_budget_for_generation':True,'all_original_masks_and_source_coordinates_preserved':True,'no_output_mask_clipping':True,'no_mask_prior':True,'new_Agent_rounds':0,'Agent_round_limit':3,'all_original_acceptance_gates_unchanged':True});save(OUT/'prepared.json',{'rows':rows,'source_only':True});print({'source_nodes':len(nodes),'selected_parents':len(selected),'planned_SAM_hypotheses':len(rows),'max_new_positive_reviews':len({v['request_sha256'] for r in rows for v in r['new_point_views']}),'planned_points':sum(len(r['points']) for r in rows),'new_model_calls':0},flush=True)

if __name__=='__main__':main()
