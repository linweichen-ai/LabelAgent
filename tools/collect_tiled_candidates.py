"""Collect real fixed-grid SAM3 proposals into isolated, unconfirmed exports.

The predictor reads image records and a previously verified candidate export.
It never reads reference labels, calibration, point clouds or workbench APIs.
The frozen plan and per-crop responses allow interrupted work to be resumed.
Create <output>/STOP to stop between requests without stopping model services.
"""
import argparse
from copy import deepcopy
from datetime import datetime, timezone
import base64
import hashlib
import io
import json
import math
import os
from pathlib import Path
import signal
import sys
import time
import urllib.request

from PIL import Image

sys.path[:0] = [str(Path(__file__).resolve().parents[1]), str(Path(__file__).resolve().parent)]
from connect_local_vlm import loopback_url
from run_local_agent_acceptance import load_image, save
from tiled_candidate_utils import augment_candidates, remap_tile_result, tile_grid


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def method_grid(scores, budgets, nms_iou):
    scores = [float(value) for value in scores.split(',')]
    budgets = [float(value) for value in budgets.split(',')]
    if (not scores or not budgets or len(set(scores)) != len(scores) or len(set(budgets)) != len(budgets)
            or any(not math.isfinite(value) or not 0 <= value <= 1 or abs(value*100-round(value*100)) > 1e-8 for value in scores)
            or any(not math.isfinite(value) or not 0 < value <= 1 or abs(value*100-round(value*100)) > 1e-8 for value in budgets)
            or not 0 < nms_iou <= 1):
        raise ValueError('Grid values must be distinct, bounded and have at most two decimal places')
    return {f'tiles_s{round(score*100)}_b{round(budget*100)}':
        {'minimum_score': score, 'addition_ratio': budget, 'nms_iou': nms_iou}
        for score in scores for budget in budgets}


def check_frozen_selection(selection, methods, frames, metric_sha):
    if (selection.get('selected_using_tuning_only') is not True or selection.get('metric_version') != 2
            or selection.get('metric_sha256') != metric_sha or selection.get('winner') == 'pallet_baseline'
            or len(methods) != 1 or methods.get(selection.get('winner')) != selection.get('frozen_parameters')
            or frames != selection.get('heldout_frames')):
        raise ValueError('Validation must use exactly the frozen method, metric and held-out frames')
    winner = [row for row in selection.get('results', []) if row.get('method') == selection.get('winner')]
    if len(winner) != 1 or winner[0].get('eligible') is not True or not all(winner[0].get('checks', {}).values()):
        raise ValueError('Frozen winner did not pass the predeclared tuning checks')


def collect(args):
    dataset = Path(args.dataset).resolve()
    manifest_path = dataset/'import_manifest.json'
    manifest = json.loads(manifest_path.read_text(encoding='utf-8'))
    raw_baseline = Path(args.candidates_json).read_bytes()
    baseline = json.loads(raw_baseline)
    binding = json.loads(Path(args.prediction_receipt).read_text(encoding='utf-8'))
    if binding['prediction_json_sha256'] != sha(raw_baseline) or binding['archive_sha256'] != manifest['archive_sha256']:
        raise ValueError('Baseline receipt does not bind the requested export and dataset')
    selection_raw = Path(args.frozen_selection).read_bytes() if args.frozen_selection else None
    selection = json.loads(selection_raw) if selection_raw is not None else None
    if args.frames == 'heldout':
        if selection is None:
            raise ValueError('Held-out frame selection requires a frozen tuning result')
        frames = selection['heldout_frames']
    else:
        frames = (list(range(len(manifest['timestamps_ns']))) if args.frames == 'all'
            else [int(value) for value in args.frames.split(',')])
    cameras = [int(value) for value in args.cameras.split(',')]
    if (not frames or len(set(frames)) != len(frames) or any(value not in range(len(manifest['timestamps_ns'])) for value in frames)
            or not cameras or len(set(cameras)) != len(cameras) or any(value not in range(4) for value in cameras)):
        raise ValueError('Invalid frame or camera selection')
    topics = {f'/capture/camera_{camera}/image' for camera in cameras}
    records = sorted((row for row in manifest['records'] if row['topic'] in topics and row['frame'] in frames),
        key=lambda row: (row['frame'], row['topic']))
    if len(records) != len(frames)*len(cameras):
        raise ValueError('Incomplete source image selection')
    for row in records:
        if binding['image_sha256'].get(row['topic'], {}).get(str(row['frame'])) != row['sha256']:
            raise ValueError('Baseline is not bound to a selected source image')
        if str(row['frame']) not in baseline.get(row['topic'], {}):
            raise ValueError('Missing baseline candidates')
    sam = loopback_url(args.sam)
    methods = method_grid(args.minimum_scores, args.addition_ratios, args.nms_iou)
    if selection is not None:
        check_frozen_selection(selection, methods, frames, sha(Path(__file__).with_name('evaluate_projected_support.py').read_bytes()))
    requested_plan = {'version': 1, 'frames': frames, 'cameras': cameras, 'planned_images': len(records),
        'planned_SAM3_requests': 4*len(records), 'query': 'pallet', 'crop_fraction': .6, 'internal_edge_margin_pixels': 2,
        'methods': methods, 'prediction_uses_reference_labels': False,
        'ranking': 'confidence*(1-max_mask_iou_with_baseline); baseline preferred; NMS includes accepted additions',
        'addition_budget_rule': 'max(1,floor(number_of_baseline_pallets*addition_ratio))',
        'baseline_candidates_preserved': True, 'archive_sha256': manifest['archive_sha256'],
        'baseline_prediction_sha256': sha(raw_baseline), 'manifest_sha256': sha(manifest_path.read_bytes()),
        'collector_sha256': sha(Path(__file__).read_bytes()),
        'postprocessor_sha256': sha(Path(__file__).with_name('tiled_candidate_utils.py').read_bytes()),
        'frozen_selection_sha256': sha(selection_raw) if selection_raw is not None else None,
        'scope': 'frozen_method_validation' if selection is not None else 'tuning',
        'selection': {'primary': 'pallet one_to_one_object_support_fraction at 3px tolerance, metric V2',
            'minimum_absolute_gain': .02, 'pallet_projected_point_coverage_must_not_fall': True,
            'maximum_person_off_class_coverage_increase': .005, 'maximum_candidate_count_ratio': 1.5,
            'maximum_pallet_union_area_ratio': 1.25, 'select_using_tuning_only': True},
        'reference_is_independent_ground_truth': False}
    output = Path(args.output).resolve()
    if args.resume:
        plan = json.loads((output/'study_plan.json').read_text(encoding='utf-8'))
        if any(plan.get(key) != value for key, value in requested_plan.items()):
            raise ValueError('Resume parameters, input bytes or code differ from the frozen plan')
        prior = json.loads((output/'progress.json').read_text(encoding='utf-8'))
        if prior['status'] == 'complete':
            raise ValueError('This study already completed; do not restart it')
        state_before = prior.get('state_sha256_before')
    else:
        output.mkdir(parents=True, exist_ok=False)
        plan = {**requested_plan, 'created_utc': datetime.now(timezone.utc).isoformat()}
        save(output/'study_plan.json', plan)
        state_before = sha(Path(args.state_file).read_bytes()) if args.state_file else None
    plan_hash = sha((output/'study_plan.json').read_bytes())
    progress = {'status': 'running', 'pid': os.getpid(), 'started_utc': datetime.now(timezone.utc).isoformat(),
        'plan_sha256': plan_hash, 'planned_requests': plan['planned_SAM3_requests'], 'completed_requests': 0,
        'new_requests_this_process': 0, 'completed_images': 0, 'state_sha256_before': state_before,
        'workbench_modified': False, 'requests': [], 'postprocessing': []}
    save(output/'progress.json', progress)
    (output/'responses').mkdir(exist_ok=args.resume)
    stop_requested = False
    def request_stop(*_):
        nonlocal stop_requested
        stop_requested = True
    signal.signal(signal.SIGINT, request_stop)
    signal.signal(signal.SIGTERM, request_stop)
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    maps = {method: {} for method in ['pallet_baseline', *methods]}
    inputs = {}
    try:
        for row in records:
            if stop_requested or (output/'STOP').exists():
                progress['status'] = 'stopped'
                break
            topic, frame = row['topic'], str(row['frame'])
            raw = load_image(dataset, row)
            image = Image.open(io.BytesIO(raw)).convert('RGB')
            width, height = image.size
            proposals = []
            camera = topic.split('/')[2]
            for tile_index, bounds in enumerate(tile_grid(width, height, plan['crop_fraction'])):
                if stop_requested or (output/'STOP').exists():
                    progress['status'] = 'stopped'
                    break
                crop = image.crop(bounds)
                buffer = io.BytesIO()
                crop.save(buffer, format='PNG')
                crop_bytes = buffer.getvalue()
                response_file = output/'responses'/f'{camera}_{int(frame):04d}_tile_{tile_index}.json'
                if response_file.exists():
                    receipt = json.loads(response_file.read_text(encoding='utf-8'))
                    if (receipt['plan_sha256'] != plan_hash or receipt['source_image_sha256'] != row['sha256']
                            or receipt['crop_png_sha256'] != sha(crop_bytes) or receipt['bounds'] != list(bounds)):
                        raise ValueError('Cached crop response does not match the frozen inputs')
                else:
                    started = time.monotonic()
                    request = urllib.request.Request(sam+'/segment_text',
                        json.dumps({'image': base64.b64encode(crop_bytes).decode('ascii'), 'prompt': plan['query']}).encode(),
                        headers={'Content-Type': 'application/json'})
                    with opener.open(request, timeout=600) as response:
                        result = json.load(response)
                    if not isinstance(result.get('results'), list) or result.get('status', 'ok') not in ('ok', 'success'):
                        raise ValueError('SAM3 did not return valid results')
                    receipt = {'topic': topic, 'frame': row['frame'], 'tile_index': tile_index, 'bounds': list(bounds),
                        'source_image_sha256': row['sha256'], 'crop_png_sha256': sha(crop_bytes),
                        'query': plan['query'], 'plan_sha256': plan_hash, 'response': result,
                        'seconds': round(time.monotonic()-started, 3)}
                    save(response_file, receipt)
                    progress['new_requests_this_process'] += 1
                accepted = 0
                for index, result in enumerate(receipt['response']['results']):
                    mapped = remap_tile_result(result, bounds, width, height,
                        f'tile-{camera}-{frame}-{tile_index}-{index}', plan['internal_edge_margin_pixels'])
                    if mapped is not None:
                        proposals.append(mapped)
                        accepted += 1
                progress['completed_requests'] += 1
                progress['requests'].append({key: receipt[key] for key in ('topic','frame','tile_index','source_image_sha256','crop_png_sha256','seconds')})
                progress['requests'][-1].update(raw_candidates=len(receipt['response']['results']), nontruncated_candidates=accepted)
                save(output/'progress.json', progress)
            if progress['status'] == 'stopped':
                break
            seed = baseline[topic][frame]
            maps['pallet_baseline'].setdefault(topic, {})[frame] = deepcopy(seed)
            inputs.setdefault(topic, {})[frame] = row['sha256']
            for method, settings in methods.items():
                candidates, info = augment_candidates(seed, proposals, width, height, **settings)
                maps[method].setdefault(topic, {})[frame] = candidates
                progress['postprocessing'].append({'topic': topic, 'frame': row['frame'], 'method': method, **info})
            progress['completed_images'] += 1
            save(output/'progress.json', progress)
            print(json.dumps({'frame': row['frame'], 'camera': camera, 'requests': progress['completed_requests'],
                'images': progress['completed_images']}), flush=True)
        if progress['status'] == 'running':
            if progress['completed_requests'] != plan['planned_SAM3_requests']:
                raise ValueError('Incomplete request coverage')
            for method, predictions in maps.items():
                prediction_file = output/(method+'_candidates.json')
                save(prediction_file, predictions)
                save(output/(method+'_receipt.json'), {'created_utc': datetime.now(timezone.utc).isoformat(),
                    'archive_sha256': manifest['archive_sha256'], 'prediction_json_sha256': sha(prediction_file.read_bytes()),
                    'image_sha256': inputs, 'plan_sha256': plan_hash, 'method': method,
                    'source': 'real_SAM3_fixed_grid_or_verified_baseline', 'candidates_confirmed_as_truth': False})
            progress['status'] = 'complete'
    except Exception as error:
        progress.update(status='failed', error_type=type(error).__name__)
        (output/'local_error.txt').write_text(str(error), encoding='utf-8')
    finally:
        progress.update(finished_utc=datetime.now(timezone.utc).isoformat(),
            plan_unchanged=sha((output/'study_plan.json').read_bytes()) == plan_hash,
            baseline_cache_unchanged=sha(Path(args.candidates_json).read_bytes()) == binding['prediction_json_sha256'])
        progress['state_sha256_unchanged'] = (sha(Path(args.state_file).read_bytes()) == state_before) if args.state_file else None
        save(output/'progress.json', progress)
    print(json.dumps({key: value for key, value in progress.items() if key not in ('requests','postprocessing')}), flush=True)
    return 0 if progress['status'] == 'complete' else 2 if progress['status'] == 'stopped' else 1


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset', required=True)
    parser.add_argument('--candidates-json', required=True)
    parser.add_argument('--prediction-receipt', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--frames', default='15,45,76,105,135')
    parser.add_argument('--cameras', default='0,1,2,3')
    parser.add_argument('--sam', default='http://127.0.0.1:8766')
    parser.add_argument('--minimum-scores', default='.8,.9')
    parser.add_argument('--addition-ratios', default='.25,.5')
    parser.add_argument('--nms-iou', type=float, default=.8)
    parser.add_argument('--frozen-selection')
    parser.add_argument('--state-file')
    parser.add_argument('--resume', action='store_true')
    raise SystemExit(collect(parser.parse_args()))
