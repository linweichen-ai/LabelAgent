"""Fixed longer RGB views; never choose frames or points from reference hits."""
import datetime,os,shutil,sys
from pathlib import Path
from run_pallet_chain import ROOT,DATA,old

SHORT=Path(os.environ.get('LABELAGENT_SHORT_CHAIN_SOURCE',str(ROOT/'outputs/labelagent_chain_rgb_20261108')))
OUT=Path(os.environ.get('LABELAGENT_LONG_CHAIN_OUTPUT',str(ROOT/'outputs/labelagent_long_chain_rgb_20261108')))
ENDPOINTS=[70,72,80,82]


def prepare():
    if OUT.exists():raise FileExistsError('Preserve fixed longer-view study')
    if shutil.disk_usage(OUT.parent).free<256*2**20:raise RuntimeError('Output volume space guard')
    previous=old.read(SHORT/'frozen_plan.json');assert all(old.digest(p)==h for p,h in previous['protected'].items());manifest=old.read(DATA/'import_manifest.json');protected={str(SHORT/'frozen_plan.json'):old.digest(SHORT/'frozen_plan.json'),str(DATA/'import_manifest.json'):old.digest(DATA/'import_manifest.json'),str(Path(__file__)):old.digest(__file__),str(Path(old.__file__)):old.digest(old.__file__),str(Path(__file__).with_name('track_pallet_chain.py')):old.digest(Path(__file__).with_name('track_pallet_chain.py'))};sources=[]
    for camera in (0,1):
        for frame in range(70,83):
            relative=f"cameras/camera_{camera}/{manifest['timestamps_ns'][frame]}.png";p=DATA/'source'/relative;expected=manifest['extracted_sha256'][relative];assert old.digest(p)==expected;protected[str(p)]=expected;sources.append({'camera':camera,'frame':frame,'path':str(p),'sha256':expected})
    paths={str(f):list(range(76,f+(-1 if f<76 else 1),-1 if f<76 else 1)) for f in ENDPOINTS};OUT.mkdir();old.save(OUT/'frozen_plan.json',{'frozen_at':datetime.datetime.now().astimezone().isoformat(),'endpoints':ENDPOINTS,'paths':paths,'seeds':previous['seeds'],'native_sources':sources,'protected':protected,'parameters':old.PARAMETERS,'same_initial_seeds_points_and_anchors_as_short_chain':True,'all_8_seeds_all_4_fixed_endpoints':True,'no_reference_labels_points_poses_depth_calibration_or_budget_inputs':True,'fresh_direct_comparison_at_each_same_endpoint':True,'every_edge_and_full_return_cycle_limit_px':1.5,'diagnostic_min_points':6,'diagnostic_min_fraction':.6,'new_masks':0,'new_Agent_runs':0,'Agent_round_limit':3,'no_model_requests_in_this_stage':True,'all_final_annotation_acceptance_gates_unchanged':True});print({'seeds':8,'fixed_endpoints':ENDPOINTS,'paired_comparisons':32,'native_RGB_inputs':26},flush=True)


def run():
    if (OUT/'completed.json').exists():raise FileExistsError('Do not repeat completed image tracking')
    import cv2,numpy as np
    from PIL import Image
    from track_pallet_chain import chain
    plan=old.read(OUT/'frozen_plan.json');assert all(old.digest(p)==h for p,h in plan['protected'].items()) and plan['parameters']==old.PARAMETERS;cv2.setNumThreads(2);rows=[]
    for camera in (0,1):
        grays={s['frame']:cv2.cvtColor(np.asarray(Image.open(s['path']).convert('RGB')),cv2.COLOR_RGB2GRAY) for s in plan['native_sources'] if s['camera']==camera}
        for seed in plan['seeds']:
            if int(seed['case'].split('_')[0][6:])!=camera:continue
            for endpoint in plan['endpoints']:
                direct=old.track(cv2,np,grays[76],grays[endpoint],seed['initial_points']);linked=chain(cv2,grays,plan['paths'][str(endpoint)],seed['initial_points']);row={'seed':seed['index'],'case':seed['case'],'seed_key':seed['key'],'endpoint_frame':endpoint,'direct':direct,'chain':linked};rows.append(row);old.save(OUT/'progress.json',rows);print({'seed':seed['index'],'frame':endpoint,'direct_valid':direct['valid_points'],'chain_valid':linked['valid_points'],'direct_pass':direct['diagnostic_image_correspondence_available'],'chain_pass':linked['diagnostic_image_correspondence_available'],'chain_anchor':linked['anchor_valid']},flush=True)
    assert all(old.digest(p)==h for p,h in plan['protected'].items());binding=old.read(ROOT/'outputs/labelagent_point_ids_20261107/incumbent_binding.json')['files'];assert all(old.digest(ROOT/p)==h for p,h in binding.items());old.save(OUT/'completed.json',{'status':'complete_fixed_longer_RGB_correspondence','rows':rows,'pairs':len(rows),'direct_supported':sum(r['direct']['diagnostic_image_correspondence_available'] for r in rows),'chain_supported':sum(r['chain']['diagnostic_image_correspondence_available'] for r in rows),'direct_anchor_valid':sum(r['direct']['anchor_valid'] for r in rows),'chain_anchor_valid':sum(r['chain']['anchor_valid'] for r in rows),'chain_supported_with_anchor':sum(r['chain']['diagnostic_image_correspondence_available'] and r['chain']['anchor_valid'] for r in rows),'all_originals_and_old_six_exact':True,'new_model_calls':0,'new_masks':0,'new_Agent_runs':0,'physical_identity_confirmations':0,'complete_base_confirmations':0,'adopted':False})


if __name__=='__main__':prepare() if sys.argv[1]=='prepare' else run()
