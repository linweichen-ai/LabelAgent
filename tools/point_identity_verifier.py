"""Agreement with a real point-prompt mask, never semantic or pixel truth."""
from datetime import datetime, timezone
import base64
import hashlib
import json
from pathlib import Path
import urllib.request

import cv2
import numpy as np

from connect_local_vlm import loopback_url
from run_local_agent_acceptance import save
from source.agent.edit_agent import _rasterize,valid_shape


class PointVerificationStopped(RuntimeError):
    pass


def intersection_point(first,second,width,height):
    shared=_rasterize(first,width,height)&_rasterize(second,width,height)
    if not shared.any():raise ValueError('Masks have no shared positive-point region')
    distance=cv2.distanceTransform(shared,cv2.DIST_L2,5)
    _,radius,_,point=cv2.minMaxLoc(distance)
    return {'x':int(point[0]),'y':int(point[1]),'label':1},float(radius)


def mask_agreement(response,expected,point,width,height,label):
    shape={**response,'id':1,'label':label}
    if not valid_shape(shape,width,height):raise ValueError('Point service returned invalid geometry')
    actual=_rasterize(shape,width,height)
    union=_rasterize(expected,width,height)
    denominator=int((actual|union).sum())
    consistent=bool(actual[point['y'],point['x']]) and not bool(response.get('warnings'))
    return {'mask_iou':int((actual&union).sum())/denominator if denominator else 0.,
        'point_consistent':consistent,'point_mask_pixels':int(actual.sum()),'union_pixels':int(union.sum()),
        'model_score':response.get('score'),'semantic_class_verified':False}


class PointIdentityVerifier:
    """Cache by actual request inputs; compare against each proposed union anew."""
    def __init__(self,image,width,height,sam,cache,threshold,model_fingerprint,stats=None,stop_file=None,requester=None):
        if not .5<=threshold<=1:raise ValueError('Point agreement threshold must be between .5 and 1')
        self.image_sha256=hashlib.sha256(image).hexdigest()
        self.encoded=base64.b64encode(image).decode('ascii')
        self.width,self.height=width,height
        self.sam=loopback_url(sam)
        self.cache=Path(cache)
        self.cache.mkdir(parents=True,exist_ok=True)
        self.threshold,self.model_fingerprint=threshold,model_fingerprint
        self.stats=stats if stats is not None else {}
        for key in ('verifications','fresh_requests','cache_hits','allowed_merges'):
            self.stats.setdefault(key,0)
        self.stop_file=Path(stop_file) if stop_file else None
        self.requester=requester
        self.backend_mode='contract_test_double' if requester is not None else 'real_local_SAM3_request'

    def __call__(self,anchor,addition,candidate):
        point,radius=intersection_point(anchor,addition,self.width,self.height)
        return self.verify_point(candidate,point,anchor['label'],radius)

    def verify_point(self,candidate,point,label,interior_radius_pixels=0.):
        """One independent single-positive request; cached responses remain input-bound."""
        if self.stop_file and self.stop_file.exists():raise PointVerificationStopped('Study stop requested')
        if (set(point)!= {'x','y','label'} or type(point['x']) is not int or type(point['y']) is not int
                or type(point['label']) is not int or point['label']!=1
                or not 0<=point['x']<self.width or not 0<=point['y']<self.height):
            raise ValueError('Expected one positive integer point inside the source image')
        if not _rasterize(candidate,self.width,self.height)[point['y'],point['x']]:
            raise ValueError('Positive verification point must belong to the proposed union')
        request_inputs={'image_sha256':self.image_sha256,'points':[point],
            'model_fingerprint':self.model_fingerprint,'backend_mode':self.backend_mode}
        key=hashlib.sha256(json.dumps(request_inputs,sort_keys=True).encode()).hexdigest()
        path=self.cache/(key+'.json')
        cached=path.exists()
        if cached:
            receipt=json.loads(path.read_text(encoding='utf-8'))
            if receipt['request_inputs']!=request_inputs:raise ValueError('Cached point response does not match input bytes')
            response=receipt['response']
            self.stats['cache_hits']+=1
        else:
            payload={'image':self.encoded,'points':[point]}
            if self.requester is not None:
                response=self.requester(payload)
            else:
                request=urllib.request.Request(self.sam+'/segment_points',json.dumps(payload).encode(),
                    headers={'Content-Type':'application/json'})
                opener=urllib.request.build_opener(urllib.request.ProxyHandler({}))
                with opener.open(request,timeout=600) as result:response=json.load(result)
            self.stats['fresh_requests']+=1
            save(path,{'received_utc':datetime.now(timezone.utc).isoformat(),'request_inputs':request_inputs,'response':response})
        result=mask_agreement(response,candidate,point,self.width,self.height,label)
        result.update(allow_merge=bool(result['point_consistent'] and result['mask_iou']>=self.threshold),
            agreement_threshold=self.threshold,cache_key=key,cache_hit=cached,
            positive_point=point,interior_radius_pixels=interior_radius_pixels,backend_mode=self.backend_mode)
        self.stats['verifications']+=1
        self.stats['allowed_merges']+=int(result['allow_merge'])
        return result
