"""Compare real description decoding using legacy regex and native string bounds.

Runs the production description prompt on identical image bytes. It never
configures the workbench or saves annotations. Full traces stay in a new local
output directory; model activation is measured separately from requests.
"""
import argparse
from copy import deepcopy
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parent))
from connect_local_vlm import loopback_url, request_json, require_real_service
from run_local_agent_acceptance import load_image, save


def legacy_description_schema(native):
    schema = deepcopy(native)
    schema['properties']['description'] = {'type': 'string', 'pattern': '[ -~]{1,700}'}
    schema['properties']['prompts']['items']['properties']['text'] = {
        'type': 'string', 'pattern': '[A-Za-z0-9 -]{1,40}'}
    return schema


def benchmark(args):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from source.agent import workflow, protocol_schema
    dataset = Path(args.dataset).resolve()
    manifest = json.loads((dataset/'import_manifest.json').read_text(encoding='utf-8'))
    matches = [row for row in manifest['records'] if row['topic'] == args.topic and row['frame'] == args.reference_frame]
    if len(matches) != 1:
        raise ValueError('Reference frame must uniquely match the import manifest')
    image = load_image(dataset, matches[0])
    reference_bytes = Path(args.reference_json).read_bytes()
    annotations = json.loads(reference_bytes)
    labels = [label.strip() for label in args.labels.split(',') if label.strip()]
    if not isinstance(annotations, list) or not annotations or not labels:
        raise ValueError('Provide reference annotations and labels')
    workflow.reference_labels(annotations, labels)
    base = loopback_url(args.base)
    order = args.variants.split(',')
    if not order or any(name not in {'legacy_regex', 'native_length'} for name in order):
        raise ValueError('Unknown benchmark variant')
    if not 1 <= args.repeats <= 10:
        raise ValueError('Repeats must be between 1 and 10')
    output = Path(args.output).resolve()
    output.mkdir(parents=True, exist_ok=False)
    report = {'status':'running', 'started_utc':datetime.now(timezone.utc).isoformat(),
              'reference_kind':args.reference_kind, 'reference_frame':args.reference_frame,
              'reference_json_sha256':hashlib.sha256(reference_bytes).hexdigest(),
              'reference_image_sha256':matches[0]['sha256'], 'results':[],
              'pixel_accuracy_evaluated':False, 'workbench_modified':False}
    save(output/'summary.json', report)
    original = protocol_schema.description_schema
    config = {'llm_url':base, 'llm_model':args.model, 'llm_key':'local', 'llm_prompt_profile':'compact'}
    marked = workflow.reference_overlay(image, annotations)
    try:
        activated = time.monotonic()
        require_real_service(base, args.model)
        report['initial_activation_seconds'] = round(time.monotonic()-activated, 3)
        for repeat in range(args.repeats):
            for name in order:
                protocol_schema.description_schema = (original if name == 'native_length' else
                    lambda labels: legacy_description_schema(original(labels)))
                identifier = f'{repeat:02d}_{name}'
                traces = {}
                def trace(event):
                    traces[event['id']] = event
                    save(output/(identifier+'_trace.json'), list(traces.values()))
                started = time.monotonic()
                row = {'variant':name, 'repeat':repeat}
                try:
                    with workflow.trace_calls(trace):
                        description = workflow.describe(config, image, annotations, labels, marked_image=marked)
                    save(output/(identifier+'_description.json'), description)
                    row.update(success=True, description_characters=len(description['description']),
                               description_at_limit=len(description['description']) == 700,
                               prompts=description['prompts'])
                except Exception as error:
                    (output/(identifier+'_error.txt')).write_text(str(error), encoding='utf-8')
                    row.update(success=False, error_type=type(error).__name__)
                row['seconds'] = round(time.monotonic()-started, 3)
                health = request_json(base[:-3]+'/health')
                row['inference_stats'] = health.get('last_inference_stats')
                report['results'].append(row)
                save(output/'summary.json', report)
                print(json.dumps(row), flush=True)
        report['status'] = 'complete' if all(row['success'] for row in report['results']) else 'incomplete'
    except Exception as error:
        report.update(status='failed', error_type=type(error).__name__)
        (output/'local_error.txt').write_text(str(error), encoding='utf-8')
    finally:
        protocol_schema.description_schema = original
        try:
            report['source_image_unchanged'] = image == load_image(dataset, matches[0])
        except Exception:
            report.update(source_image_unchanged=False, status='failed')
        report['finished_utc'] = datetime.now(timezone.utc).isoformat()
        save(output/'summary.json', report)
    return 0 if report['status'] == 'complete' else 1


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset', required=True)
    parser.add_argument('--reference-json', required=True)
    parser.add_argument('--reference-kind', required=True, choices=['human_reviewed','unconfirmed_functional_fixture'])
    parser.add_argument('--topic', default='/capture/camera_1/image')
    parser.add_argument('--reference-frame', type=int, default=0)
    parser.add_argument('--labels', default='pallet,person')
    parser.add_argument('--base', default='http://127.0.0.1:18868/v1')
    parser.add_argument('--model', default='labelagent-vlm')
    parser.add_argument('--variants', default='legacy_regex,native_length')
    parser.add_argument('--repeats', type=int, default=1)
    parser.add_argument('--output', required=True)
    raise SystemExit(benchmark(parser.parse_args()))
