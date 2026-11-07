"""Explicit source-aligned teacher-logit prior diagnostics, never a default."""
import base64,hashlib,io
from pathlib import Path
import cv2,numpy as np
from PIL import Image


def validate_prior(logits,image_bytes,prior_bytes):
    image=Image.open(io.BytesIO(image_bytes));prior=Image.open(io.BytesIO(prior_bytes)).convert('L')
    if not isinstance(logits,np.ndarray) or logits.dtype!=np.float32 or logits.shape!=(image.height,image.width) or not np.isfinite(logits).all():
        raise ValueError('Teacher logits must be finite float32 source-aligned HxW values')
    if prior.size!=image.size or not np.array_equal(np.asarray(prior)>127,logits>=0):
        raise ValueError('Binary and soft teacher priors must have exactly the same source foreground')
    return logits


class PriorPredictor:
    def __init__(self,model,source_logits,mode):
        if mode not in ['hard','soft']:raise ValueError('Unknown prior mode')
        self.model=model;self.logits=np.array(source_logits,copy=True);self.mode=mode;self.calls=[]
    def __getattr__(self,key):return getattr(self.model,key)
    def predict_inst(self,*args,**kwargs):
        if 'mask_input' not in kwargs:raise ValueError('Original binary-prior path required')
        original=np.array(kwargs['mask_input'],copy=True)
        if original.ndim!=3 or original.shape[0]!=1:raise ValueError('Unexpected SAM mask-input shape')
        actual=original.copy() if self.mode=='hard' else cv2.resize(self.logits,(original.shape[2],original.shape[1]),interpolation=cv2.INTER_LINEAR)[None].astype('float32')
        kwargs=dict(kwargs,mask_input=actual);record={'original_hard_mask_input':original,'actual_mask_input':actual.copy(),'mode':self.mode,'multimask_output':kwargs.get('multimask_output')};self.calls.append(record)
        returned=self.model.predict_inst(*args,**kwargs)
        if isinstance(returned,(tuple,list)) and len(returned)>2 and returned[2] is not None:
            value=returned[2]
            if hasattr(value,'detach'):value=value.detach().cpu().numpy()
            record['native_low_res_logits']=np.array(value,copy=True)
        return returned


def segment_with_prior_capture(image_bytes,points,prior_bytes,logit_bytes,mode):
    from . import point_segment
    from .raw_capture_adapter import load_private_segment,capture_segment
    logits=np.load(io.BytesIO(logit_bytes),allow_pickle=False);validate_prior(logits,image_bytes,prior_bytes)
    source=Path(point_segment.__file__);private=load_private_segment(source,hashlib.sha256(source.read_bytes()).hexdigest());proxies=[]
    def factory():
        model,processor=point_segment._predictor();proxy=PriorPredictor(model,logits,mode);proxies.append(proxy);return proxy,processor
    private._predictor=factory;private._model_lock=point_segment._model_lock;record=capture_segment(private,image_bytes,points,prior_bytes);arrays={};metadata={k:v for k,v in record.items() if k not in ('predictions','response')};metadata['predictions']=[]
    for i,pred in enumerate(record['predictions']):
        info={k:v for k,v in pred.items() if k not in ('raw_return','masks','scores','point_coords','point_labels')}
        for key in ('masks','scores','point_coords','point_labels'):
            if pred.get(key) is not None:
                arrays[f'{i}_{key}']=pred[key]['array'];info[key]={k:v for k,v in pred[key].items() if k!='array'}
        metadata['predictions'].append(info)
    forwarded=[]
    for i,call in enumerate(c for proxy in proxies for c in proxy.calls):
        info={k:v for k,v in call.items() if not isinstance(v,np.ndarray)}
        for key,value in call.items():
            if isinstance(value,np.ndarray):
                arrays[f'{i}_{key}']=value;info[key]={'shape':list(value.shape),'dtype':str(value.dtype),'sha256':hashlib.sha256(value.tobytes()).hexdigest(),'min':float(value.min()),'max':float(value.max())}
        forwarded.append(info)
    metadata.update(shared_service_predictor=True,diagnostic_prior_mode=mode,source_logit_sha256=hashlib.sha256(logits.tobytes()).hexdigest(),source_logit_file_sha256=hashlib.sha256(logit_bytes).hexdigest(),prior_calls=forwarded,all_native_low_res_logits_retained=bool(forwarded) and all('native_low_res_logits' in c for c in forwarded),adapter_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest());metadata['raw_capture_complete']=bool(record['predictions']) and all(p.get('all_returned_candidates_retained') for p in record['predictions']);buffer=io.BytesIO();np.savez_compressed(buffer,**arrays);payload=dict(record['response'] or {});payload['raw_capture']={'metadata':metadata,'arrays_npz_base64':base64.b64encode(buffer.getvalue()).decode()}
    if record['error']:payload['error']=record['error']['message']
    return payload,200 if record['status']=='segment_returned' else 400
