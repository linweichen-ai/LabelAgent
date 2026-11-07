"""Explicit experimental box prompts with native-array capture, default unused."""
import base64,hashlib,io
from pathlib import Path
import numpy as np
from PIL import Image


def validate_box(box, width, height):
    if not isinstance(box, list) or len(box)!=4 or any(isinstance(x,bool) or not isinstance(x,(int,float)) for x in box):
        raise ValueError('diagnostic_box must contain four finite XYXY pixel coordinates')
    x1,y1,x2,y2=box
    if not np.isfinite(box).all() or not(0<=x1<x2<width and 0<=y1<y2<height):
        raise ValueError('diagnostic_box must lie inside the input image')
    return np.asarray(box,dtype=np.float32)


class BoxPredictor:
    def __init__(self,model,box):self.model=model;self.box=np.array(box,copy=True);self.calls=[]
    def __getattr__(self,key):return getattr(self.model,key)
    def predict_inst(self,*args,**kwargs):
        if 'box' in kwargs:raise ValueError('Duplicate box prompt')
        kwargs=dict(kwargs,box=self.box.copy())
        self.calls.append({'box':kwargs['box'].tolist(),'multimask_output':kwargs.get('multimask_output')})
        return self.model.predict_inst(*args,**kwargs)


def segment_with_box_capture(image_bytes,points,box):
    # Imports stay lazy: protocol tests never load SAM or a service worker.
    from . import point_segment
    from .raw_capture_adapter import load_private_segment,capture_segment
    image=Image.open(io.BytesIO(image_bytes));validated=validate_box(box,*image.size)
    source=Path(point_segment.__file__);private=load_private_segment(source,hashlib.sha256(source.read_bytes()).hexdigest())
    proxies=[]
    def factory():
        model,processor=point_segment._predictor();proxy=BoxPredictor(model,validated);proxies.append(proxy);return proxy,processor
    private._predictor=factory;private._model_lock=point_segment._model_lock
    record=capture_segment(private,image_bytes,points)
    metadata={k:v for k,v in record.items() if k not in ('predictions','response')};arrays={};metadata['predictions']=[]
    for i,pred in enumerate(record['predictions']):
        info={k:v for k,v in pred.items() if k not in ('raw_return','masks','scores','point_coords','point_labels')}
        for key in ('masks','scores','point_coords','point_labels'):
            if pred.get(key) is not None:
                arrays[f'{i}_{key}']=pred[key]['array'];info[key]={k:v for k,v in pred[key].items() if k!='array'}
        metadata['predictions'].append(info)
    metadata.update(shared_service_predictor=True,experimental_box_prompt=True,diagnostic_box=validated.tolist(),actual_box_calls=[c for p in proxies for c in p.calls],box_is_prompt_not_output_clip=True)
    metadata['raw_capture_complete']=bool(record['predictions']) and all(p.get('all_returned_candidates_retained') for p in record['predictions'])
    metadata['box_adapter_sha256']=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    buffer=io.BytesIO();np.savez_compressed(buffer,**arrays);payload=dict(record['response'] or {});payload['raw_capture']={'metadata':metadata,'arrays_npz_base64':base64.b64encode(buffer.getvalue()).decode()}
    if record['error']:payload['error']=record['error']['message']
    return payload,200 if record['status']=='segment_returned' else 400
