"""Exact neutral-caption lookup; model, image bytes, payload and tokens must bind."""
import hashlib,json
from pathlib import Path
from run_visual_budget import payload


def lookup(image_bytes,model_config_sha256,catalogue):
    key=hashlib.sha256(json.dumps(payload(image_bytes,256),sort_keys=True).encode()).hexdigest()
    for row in catalogue:
        if row.get('request_sha256')!=key:continue
        image_path=row.get('image',row.get('semantic_image'));response_path=row.get('response')
        if not image_path or not response_path:continue
        source=Path(image_path);file=Path(response_path)
        if not source.is_file() or not file.is_file():continue
        actual=source.read_bytes()
        if hashlib.sha256(json.dumps(payload(actual,256),sort_keys=True).encode()).hexdigest()!=key:continue
        reply_bytes=file.read_bytes();sha=hashlib.sha256(reply_bytes).hexdigest()
        if row.get('response_sha256') and sha!=row['response_sha256']:continue
        reply=json.loads(reply_bytes);model=reply.get('local_vlm',{})
        if model.get('model_config_sha256')!=model_config_sha256 or model.get('actual_image_token_counts')!=[256]:continue
        return {'request_sha256':key,'response':str(file),'response_sha256':sha,'reply':reply,'new_model_call':False,'exact_payload_model_image_tokens_verified':True}
    return None
