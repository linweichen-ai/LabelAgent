"""Freeze all source palette-part cues before neutral review and original scoring."""
import datetime,shutil,sys
from pathlib import Path
import prepare_fixed_grid_followup as views
from prepare_fixed_grid_followup import ROOT,BASE
import numpy as np,torch
from PIL import Image
from run_rgb_layers import read,save
from deliver_context_graph import sha
from m2f_query_replay import replay,probability
from palette_part_points import propose
OUT=BASE/'labelagent_palette_parts_20261108'

def main():
    if OUT.exists():raise FileExistsError('Preserve frozen RGB component study')
    if shutil.disk_usage(BASE).free<192*2**20:raise RuntimeError('Original192MiB source preparation reserve')
    torch.set_num_threads(2);OUT.mkdir();views.OUT=OUT;grid=BASE/'labelagent_registered_fixed_grid_20261108';p=read(grid/'frozen_plan.json');g=read(grid/'generation_completed.json');old=BASE/'labelagent_single_teacher_grid_20261108';assert all(sha(f)==h for f,h in p['protected'].items());protected={str(grid/'frozen_plan.json'):sha(grid/'frozen_plan.json'),str(grid/'generation_completed.json'):sha(grid/'generation_completed.json'),str(old/'frozen_plan.json'):sha(old/'frozen_plan.json'),str(old/'review_completed.json'):sha(old/'review_completed.json'),str(Path(__file__)):sha(__file__),str(Path(__file__).with_name('palette_part_points.py')):sha(Path(__file__).with_name('palette_part_points.py'))};candidates=[];diagnostics=[];vetoes=[]
    for r in g['rows']:
        if r['is_synthetic_control']:continue
        folder=Path(r['folder']);assert all(sha(folder/n)==h for n,h in r['artifacts'].items());im=Image.open(r['input']).convert('RGB');raw=np.load(folder/'raw_queries.npz',allow_pickle=False);ids,segs,_,winner=replay(raw['class_logits'],raw['mask_logits'],(im.height,im.width),p['postprocessing']);assert segs==r['segments_info'];pal=sorted([s for s in segs if s['category_id']==1],key=lambda s:(-s['score'],s['id']))[:3];protected[r['input']]=sha(r['input']);protected[r['original_source_image']]=sha(r['original_source_image']);protected[str(folder/'raw_queries.npz')]=r['artifacts']['raw_queries.npz'];x1,y1,x2,y2=r['observed_region_in_input'];observed=np.zeros(ids.shape,bool);observed[y1:y2,x1:x2]=True
        for rank,s in enumerate(pal,1):
            stock=ids==s['id'];q=np.unique(winner[stock]);assert len(q)==1;q=int(q[0]);native=probability(raw['mask_logits'],q,ids.shape)>=.5;border=lambda m:bool(m[0].any() or m[-1].any() or m[:,0].any() or m[:,-1].any())
            if border(native) or border(stock):vetoes.append({'model':r['model'],'teacher':r['teacher_key'],'source_rank':rank,'reason':'unchanged_MF_native_or_stock_input_border'});continue
            rows,diag=propose(np.asarray(im),stock,observed);diagnostics.append({'model':r['model'],'teacher':r['teacher_key'],'query':q,'source_rank':rank,'diagnostic':diag})
            for part_rank,part in enumerate(rows,1):
                px,py=part['point'];vx,vy,_,_=r['source_virtual_box'];ax,ay=px+vx,py+vy;assert 0<=ax<1920 and 0<=ay<1536 and stock[py,px] and observed[py,px];neg=[n|{'x':n['x']+vx,'y':n['y']+vy} for n in part['negative_hypotheses']];assert all(0<=n['x']<1920 and 0<=n['y']<1536 for n in neg);candidates.append({'cohort':'source_palette_parts','model':r['model'],'case':r['case'],'teacher_key':r['teacher_key'],'source_view':r['source_view'],'source_image':r['original_source_image'],'source_rank':rank,'part_rank':part_rank,'query':q,'stock_segment':s,'source_point':[ax,ay],'negative_hypotheses':neg,'palette_component':part,'teacher_stock_PAL_verified':True,'source_virtual_box':r['source_virtual_box'],'both_model_consensus_required_for_this_proposal_arm':False,'not_human_truth':True})
    order={'whole':0,'q1':1,'q2':2,'q3':3,'q4':4};candidates.sort(key=lambda r:(r['part_rank'],r['source_rank'],order[r['source_view']],r['case'],r['model']));rows=[];seen={}
    for r in candidates:
        k=(r['case'],*r['source_point'])
        if k in seen:rows[seen[k]]['logical_sources'].append(r);continue
        if len(rows)>=32:vetoes.append({'reason':'32_unique_source_point_limit','source':r});continue
        index=len(rows)+1;photo=Image.open(r['source_image']).convert('RGB');caption=views.caption_views(photo,r['source_point'],OUT/'inputs'/f'cue{index:02d}');seen[k]=len(rows);rows.append(r|caption|{'index':index,'logical_sources':[r],'source_partition_index':index})
    controls=[]
    for name in ['gray','white']:
        f=ROOT/'outputs/labelagent_semantic_recovery_20261105/neutral_controls'/(name+'.png');im=Image.open(f).convert('RGB');protected[str(f)]=sha(f);caption=views.caption_views(im,[im.width//2,im.height//2],OUT/'controls'/name);controls.append(caption|{'index':'control_'+name,'known_no_real_object':True,'name':name})
    for r in rows+controls:
        protected[r['request_file']]=sha(r['request_file'])
        for b in r['bindings']:protected[b['image']]=b['sha256']
    save(OUT/'frozen_plan.json',{'at':datetime.datetime.now().astimezone().isoformat(),'rows':rows,'controls':controls,'protected':protected,'model_config_sha256':read(old/'frozen_plan.json')['model_config_sha256'],'diagnostics':diagnostics,'vetoes':vetoes,'source_color_component_points_before_any_reference_evaluation':True,'point_control':'Previously completed exact single-teacher maximum-distance/lowest-third control remains unchanged, no repeats','SAM_arms':['same_original_full_source_one_positive','same_original_full_source_positive_and_source_component_negative_hypotheses'],'negative_physical_identities_unconfirmed':True,'paired_negative_arm_requires_at_least_one_existing_source_negative_no_invented_points':True,'max_unique_positive_reviews':32,'max_unique_SAM_point_candidates':16,'max_paired_SAM_calls':32,'new_MF_forwards':0,'new_training':False,'Agent_round_limit':3,'new_Agent_runs':0,'all_original_final_gates_unchanged':True,'final_SAM_input_border_explicitly_rechecked':True,'no_final_mask_cut_or_prior':True,'not_independent_validation':True,'adopted':False});save(OUT/'prepared.json',{'source_points':len(rows),'logical_source_parts':len(candidates),'plain_controls':2,'new_model_calls':0,'native_MF_border_vetoes':len([v for v in vetoes if v['reason'].startswith('unchanged')])});print(read(OUT/'prepared.json'),flush=True)

if __name__=='__main__':main()
