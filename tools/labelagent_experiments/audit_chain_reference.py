"""Post-freeze reference ownership for unchanged initial masks, never generation."""
import sys
from pathlib import Path
import numpy as np,cv2
from PIL import Image
from run_pallet_chain import ROOT,OUT,old
from projection_audit_core import frame_data,cameras


def main():
    assert (OUT/'semantic_readback.json').is_file() and old.read(OUT/'semantic_readback.json')['passed'];plan=old.read(OUT/'frozen_plan.json');cfg=old.read(ROOT/'task_workspaces/labelagent_point_ids_20261107/config.json');sys.path[:0]=[cfg['source_snapshot'],str(Path(cfg['source_snapshot'])/'tools'),cfg['dependency_dir']]
    import evaluate_projected_support as ev
    from source.agent.edit_agent import _rasterize
    assert old.digest(Path(cfg['source_snapshot'])/'tools/evaluate_projected_support.py')=='93c5152169d04d43b25dbf4a4139b12676daf78750a7f059837121656bb644c2';dataset=Path(cfg['dataset']);manifest=old.read(dataset/'import_manifest.json');data=frame_data(dataset,manifest,76,ev);cams=cameras(dataset,manifest,ev);rows=[];basearea=122302;cap=122511.25
    for camera in (0,1):
        incumbent=old.read(ROOT/f'outputs/labelagent_context_comparison_20261105/C_hybrid/camera{camera}_frame76/result.json');oldmasks=[_rasterize(s,1920,1536) for s in incumbent['shapes'] if s['label']=='pallet'];oldunion=np.maximum.reduce(oldmasks);people=[_rasterize(s,1920,1536) for s in incumbent['shapes'] if s['label']=='person'];person=cv2.dilate(np.maximum.reduce(people),cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(7,7)));source,pix=ev.project_visible(data['xyz'],cams[camera]);owned=data['owner'][source];known=(data['bits'][source]==1)&(owned>0)
        for seed in plan['seeds']:
            if int(seed['case'].split('_')[0][6:])!=camera:continue
            m=(np.asarray(Image.open(seed['mask_path']).convert('L'))!=0).astype('uint8');assert m.shape==(1536,1920);radii=[]
            for radius in (0,3,6):
                grown=cv2.dilate(m,cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(2*radius+1,2*radius+1))) if radius else m;hit=grown[pix[:,1],pix[:,0]]!=0;total=int((hit&known).sum());ownership=[]
                for i,box in enumerate(data['boxes']):
                    n=int((hit&(owned==i+1)).sum());den=int((owned==i+1).sum())
                    if box['bit']!=1 or den<5 or not n:continue
                    ownership.append({'object_id':box['obj_id'],'hit_points':n,'visible_points':den,'coverage':n/den,'purity':n/total if total else 0,'original_sparse_gate_pass':n/den>=.5 and n/total>=.67,'remaining_target':box['obj_id'] in ([42,48] if camera==0 else [47,49])})
                radii.append({'radius':radius,'known_PAL_hits':total,'ownership':ownership})
            extra=int(((m!=0)&(oldunion==0)).sum());rows.append({'seed':seed['index'],'case':seed['case'],'unchanged_original_seed_mask_sha256':old.digest(seed['mask_path']),'seed_pixels':int(m.sum()),'extra_pixels_if_added_unchanged':extra,'area_fits_if_added_alone':basearea+extra<=cap,'person_conflict':bool((m&person).any()),'radii':radii,'new_mask_generated':False})
    binding=old.read(ROOT/'outputs/labelagent_point_ids_20261107/incumbent_binding.json')['files'];assert all(old.digest(ROOT/p)==h for p,h in binding.items());old.save(OUT/'post_freeze_reference_audit.json',{'status':'complete_post_freeze_unchanged_seed_only','rows':rows,'base_union':basearea,'cap':cap,'generation_and_semantic_plans_already_frozen':True,'no_new_mask_or_adoption':True,'all_original_six_exact':True,'original_evaluator_sha256':'93c5152169d04d43b25dbf4a4139b12676daf78750a7f059837121656bb644c2'});print([{'seed':r['seed'],'extra':r['extra_pixels_if_added_unchanged'],'ownership':r['radii'][1]['ownership']} for r in rows],flush=True)


if __name__=='__main__':main()
