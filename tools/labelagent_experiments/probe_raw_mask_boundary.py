"""Read-only worker identity evidence; does not attest in-memory Python source."""
import datetime
import hashlib
import json
import urllib.request
from pathlib import Path

import psutil

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / 'outputs/labelagent_raw_mask_audit_20261107'
SOURCE = Path('D:/Apps/LabelAgent')


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    target = OUT / 'runtime_boundary_probe.json'
    if target.exists():
        raise FileExistsError('A read-only snapshot is immutable; use a new timestamped snapshot')
    paths = [SOURCE / rel for rel in (
        'source/sam3_service/point_segment.py',
        'source/sam3_service/server.py', 'tools/local_gpu_manager.py')]
    receipt = {
        'checked_at': datetime.datetime.now().astimezone().isoformat(),
        'method': 'read-only current listeners, sanitized launch identity, health and disk source hashes',
        'model_inference_requests': 0, 'live_service_modified': False,
        'in_memory_module_digest_attested': False,
        'past_internal_masks_recoverable_from_wire_receipts': False,
        'source_files': {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in paths},
        'listeners': [], 'health': {},
        'limitation': 'Launch paths and current disk source are evidence of the configured route; the service does not expose loaded-module or internal-mask digests. They do not prove the exact historical in-memory source or any historical pixel loss.'}
    for connection in psutil.net_connections(kind='tcp'):
        if connection.status != 'LISTEN' or connection.laddr.port not in (8766, 17866, 18868, 18869):
            continue
        process = psutil.Process(connection.pid)
        argv = process.cmdline()
        modules = [argv[i + 1] for i, item in enumerate(argv[:-1]) if item == '-m']
        receipt['listeners'].append({
            'port': connection.laddr.port, 'pid': connection.pid,
            'exe': process.exe(), 'cwd': process.cwd(),
            'scripts': [item for item in argv[1:] if item.endswith('.py')],
            'modules': modules, 'process_create_time': process.create_time()})
    for name, url in (
        ('SAM', 'http://127.0.0.1:8766/health'),
        ('VLM', 'http://127.0.0.1:18868/health'),
        ('manager', 'http://127.0.0.1:8766/manager/status')):
        with urllib.request.urlopen(url, timeout=15) as response:
            value = json.load(response)
        allowed = ('status', 'model_loaded', 'on_demand', 'backend', 'active_worker',
                   'worker_ready', 'owned_worker_pid', 'live_owned_workers', 'workers_started', 'workers_stopped')
        receipt['health'][name] = {key: value[key] for key in allowed if key in value}
    receipt['source_files_unchanged_during_probe'] = all(
        hashlib.sha256(Path(path).read_bytes()).hexdigest() == digest
        for path, digest in receipt['source_files'].items())
    target.write_text(json.dumps(receipt, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    print(json.dumps({'file': str(target), 'source_unchanged': receipt['source_files_unchanged_during_probe'],
                      'listeners': len(receipt['listeners']), 'in_memory_attestation': False,
                      'inference_requests': 0}), flush=True)


if __name__ == '__main__':
    main()
