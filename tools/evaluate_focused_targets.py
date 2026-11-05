"""Reuse bound, verified localization and enforce region-only classification input."""
import argparse
import hashlib
import json
from pathlib import Path
import sys

REPO = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(REPO), str(REPO / 'tools')]
from source.agent.edit_agent import _rasterize
from tools.evaluate_sam_evidence_recheck import evaluate
from tools.grounded_target_views import location_ok, localization_payload, validated_location
from tools.run_local_agent_acceptance import save


def read(path):
    return json.loads(Path(path).read_bytes())


def verify_cached_locations(plan):
    old = read(plan['localization_plan']); f = read(old['fixture']); results = read(plan['localization_summary'])
    mapping = {m['case_id']: m for m in read(plan['localization_mapping'])}
    targets = {t['id']: t for t in read(plan['targets_fixture'])['targets']}
    assert results['status'] == 'complete' and results['localized_targets'] == 2
    verified = {}
    for result in results['target_results']:
        assert result['location_verified']
        case = next(c for c in f['cases'] if c['id'] == result['case_id']); target = targets[case['id']]
        mask = _rasterize(target['geometry'], *target['size']).astype(bool)
        row = result['locations'][-1]
        path = Path(plan['localization_summary']).parent / f"response_{row['number']:03d}.json"
        response = read(path); parsed = validated_location(response['raw_response'])
        fixture = {**f, 'localization_prompt': f['localization_prompt'] if row['view'] == 'whole' else f['localization_focus_prompt']}
        ref = case['localization_images'][row['view']]; data = (Path(f['asset_root']) / ref['path']).read_bytes()
        body = localization_payload(fixture, (data, 'image/jpeg'), {**case, 'localization_image': ref})
        digest = hashlib.sha256(json.dumps(body, sort_keys=True, separators=(',', ':')).encode()).hexdigest()
        assert digest == response['full_payload_sha256'] and parsed == row['parsed']
        if row['view'] == 'focus':
            x1, y1, x2, y2 = mapping[case['id']]['fallback_source_box']; local = mask[y1:y2, x1:x2]
        else: local = mask
        assert location_ok(parsed['cell'], local) and location_ok(result['global_cell'], mask)
        verified[case['id']] = mapping[case['id']]['branches'][result['global_cell']]['focus_box']
    return verified


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__); p.add_argument('--plan', required=True); p.add_argument('--plan-sha256', required=True); p.add_argument('--output', required=True)
    args = p.parse_args(); plan = read(args.plan)
    assert hashlib.sha256(Path(args.plan).read_bytes()).hexdigest() == args.plan_sha256
    boxes = verify_cached_locations(plan)
    mapping = read(plan['focused_mapping'])
    assert all(m['display_metadata']['verified_source_box'] == boxes[m['case_id']] for m in mapping)
    code = evaluate(args)
    path = Path(args.output) / 'summary.json'; result = read(path)
    result.update(cached_localization_verified_before_requests=True, classification_views_have_no_outside_region_pixels=True,
                  new_localization_calls=0, cached_verified_targets=2)
    save(path, result); raise SystemExit(code)
