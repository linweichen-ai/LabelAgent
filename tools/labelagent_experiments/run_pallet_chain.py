"""Frozen direct-cache versus chained RGB correspondence on the same 8 seeds."""
import datetime,json,os,shutil,sys
from pathlib import Path
import audit_pallet_temporal_views as old

ROOT=old.ROOT;OLD=old.OUT;DATA=old.DATA
OUT=Path(os.environ.get('LABELAGENT_CHAIN_OUTPUT',str(ROOT/'outputs/labelagent_chain_rgb_20261108')))
PATHS=[[76,75,74],[76,77,78]]


def prepare():
    if OUT.exists():raise FileExistsError('Preserve chain study')
    if shutil.disk_usage(OUT.parent).free<256*2**20:raise RuntimeError('Output volume space guard')
    plan=old.read(OLD/'frozen_plan.json');assert plan['parameters']==old.PARAMETERS;manifest=old.read(DATA/'import_manifest.json');protected={str(OLD/'frozen_plan.json'):old.digest(OLD/'frozen_plan.json'),str(OLD/'completed.json'):old.digest(OLD/'completed.json'),str(Path(old.__file__)):old.digest(old.__file__),str(Path(__file__)):old.digest(__file__),str(Path(__file__).with_name('track_pallet_chain.py')):old.digest(Path(__file__).with_name('track_pallet_chain.py'))};sources=[];seeds=[];OUT.mkdir()
    for camera in (0,1):
        for frame in [74,75,76,77,78]:
            stamp=str(manifest['timestamps_ns'][frame]);relative=f'cameras/camera_{camera}/{stamp}.png';source=DATA/'source'/relative;expected=manifest['extracted_sha256'][relative];assert old.digest(source)==expected;protected[str(source)]=expected;sources.append({'camera':camera,'frame':frame,'path':str(source),'sha256':expected})
    protected[str(DATA/'import_manifest.json')]=old.digest(DATA/'import_manifest.json')
    for seed in plan['seeds']:
        file=OLD/f"seed{seed['index']:02d}/tracking.json";tracking=old.read(file);assert seed['key']==tracking['seed_key'];protected[str(file)]=old.digest(file);seeds.append({**seed,'initial_points':tracking['initial_points'],'old_tracking_file':str(file)})
    old.save(OUT/'frozen_plan.json',{'frozen_at':datetime.datetime.now().astimezone().isoformat(),'paths':PATHS,'seeds':seeds,'native_sources':sources,'protected':protected,'parameters':old.PARAMETERS,'whole_cycle_limit_same_as_original_edge_limit':1.5,'no_new_points_or_reanchoring':True,'baseline_reuses_exact_original_direct_results':True,'same_min_6_points_and_60_percent_diagnostic_gate':True,'extra_whole_cycle_gate_not_relaxed':True,'reference_labels_points_depth_pose_calibration_area_read':False,'model_requests':0,'annotation_writes':0,'Agent_round_limit':3,'new_Agent_runs':0,'all_final_acceptance_gates_unchanged':True});print({'frozen_seeds':len(seeds),'paired_endpoints':16,'native_RGB_sources':10,'model_requests':0},flush=True)


def run():
    if (OUT/'completed.json').exists():raise FileExistsError('Already complete')
    import cv2,numpy as np
    from PIL import Image
    from track_pallet_chain import chain
    plan=old.read(OUT/'frozen_plan.json');assert all(old.digest(p)==h for p,h in plan['protected'].items());assert plan['parameters']==old.PARAMETERS;cv2.setNumThreads(2);rows=[]
    for camera in (0,1):
        grays={}
        for s in plan['native_sources']:
            if s['camera']==camera:grays[s['frame']]=cv2.cvtColor(np.asarray(Image.open(s['path']).convert('RGB')),cv2.COLOR_RGB2GRAY)
        for seed in plan['seeds']:
            if int(seed['case'].split('_')[0][6:])!=camera:continue
            previous=old.read(seed['old_tracking_file'])
            for path in plan['paths']:
                result=chain(cv2,grays,path,seed['initial_points']);baseline=previous['tracking'][str(path[-1])];row={'seed':seed['index'],'case':seed['case'],'seed_key':seed['key'],'endpoint_frame':path[-1],'direct':baseline,'chain':result};rows.append(row);old.save(OUT/'progress.json',rows);print({'seed':seed['index'],'frame':path[-1],'old_valid':baseline['valid_points'],'chain_valid':result['valid_points'],'old_supported':baseline['diagnostic_image_correspondence_available'],'chain_supported':result['diagnostic_image_correspondence_available'],'chain_anchor':result['anchor_valid']},flush=True)
    assert all(old.digest(p)==h for p,h in plan['protected'].items());binding=old.read(ROOT/'outputs/labelagent_point_ids_20261107/incumbent_binding.json')['files'];assert all(old.digest(ROOT/p)==h for p,h in binding.items());old.save(OUT/'completed.json',{'status':'complete_RGB_correspondence_only','rows':rows,'pairs':len(rows),'direct_supported':sum(r['direct']['diagnostic_image_correspondence_available'] for r in rows),'chain_supported':sum(r['chain']['diagnostic_image_correspondence_available'] for r in rows),'direct_anchor_valid':sum(r['direct']['anchor_valid'] for r in rows),'chain_anchor_valid':sum(r['chain']['anchor_valid'] for r in rows),'all_originals_and_old_six_exact':True,'no_reference_inputs':True,'new_model_calls':0,'new_masks':0,'new_Agent_runs':0,'physical_identity_confirmations':0,'complete_base_confirmations':0,'adopted':False})


if __name__=='__main__':prepare() if sys.argv[1]=='prepare' else run()
