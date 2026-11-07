"""Opt-in diagnostics sharing the existing service predictor/cache."""
import base64
import hashlib
import io
import json
from pathlib import Path
import numpy as np
from . import point_segment
from .raw_capture_adapter import load_private_segment, capture_segment

def segment_with_capture(image_bytes, points, prior_mask=None):
    source = Path(point_segment.__file__)
    digest = hashlib.sha256(source.read_bytes()).hexdigest()
    private = load_private_segment(source, digest)
    # The service cache/lock remains the model owner. No second model loads.
    private._predictor = point_segment._predictor
    private._model_lock = point_segment._model_lock
    record = capture_segment(private, image_bytes, points, prior_mask)
    arrays = {}
    metadata = {k:v for k,v in record.items() if k not in ('predictions','response')}
    metadata['predictions'] = []
    for i,pred in enumerate(record['predictions']):
        info = {k:v for k,v in pred.items() if k not in ('raw_return','masks','scores','point_coords','point_labels')}
        for key in ('masks','scores','point_coords','point_labels'):
            if pred.get(key) is not None:
                arrays[f'{i}_{key}'] = pred[key]['array']
                info[key] = {k:v for k,v in pred[key].items() if k!='array'}
        metadata['predictions'].append(info)
    metadata['shared_service_predictor'] = True
    metadata['raw_capture_complete'] = bool(record['predictions']) and all(p.get('all_returned_candidates_retained') for p in record['predictions'])
    metadata['adapter_sha256'] = hashlib.sha256(Path(__file__).with_name('raw_capture_adapter.py').read_bytes()).hexdigest()
    buffer=io.BytesIO();np.savez_compressed(buffer,**arrays)
    payload = dict(record['response'] or {})
    payload['raw_capture']={'metadata':metadata,'arrays_npz_base64':base64.b64encode(buffer.getvalue()).decode('ascii')}
    if record['error']:payload['error']=record['error']['message']
    return payload, 200 if record['status']=='segment_returned' else 400
