"""CPU-only tokenizer/prefix-filter reachability for an actual frozen request."""
import argparse,base64,hashlib,json
import sys
from io import BytesIO
from pathlib import Path
from PIL import Image
import torch
from transformers import AutoProcessor
from lmformatenforcer import JsonSchemaParser
from lmformatenforcer.integrations.transformers import build_token_enforcer_tokenizer_data,build_transformers_prefix_allowed_tokens_fn

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def audit(model_dir,request_path,output):
    model=Path(model_dir);request=Path(request_path);body=json.loads(request.read_bytes())['body'];schema=body['response_format']['json_schema']['schema']
    repo=model.parents[2];sys.path.insert(0,str(repo))
    from tools.local_vlm_server import decode_messages
    messages,images=decode_messages(body['messages'])
    processor=AutoProcessor.from_pretrained(str(model),local_files_only=True)
    processor.image_processor.size={'longest_edge':256*32*32,'shortest_edge':64*32*32}
    text=processor.apply_chat_template(messages,tokenize=False,add_generation_prompt=True)
    inputs=processor(text=[text],images=images,return_tensors='pt',padding=True);prefix=inputs['input_ids'][0].tolist()
    raw_path=request.with_name(request.name.replace('request_','response_').replace('_PRIVATE.json','.json'))
    raw=json.loads(raw_path.read_bytes())['raw_response'];stats=raw['local_vlm']['inference_stats']
    print(json.dumps(dict(rebuilt_prompt_tokens=len(prefix),expected_prompt_tokens=stats['prompt_tokens'],rebuilt_grid=inputs['image_grid_thw'].tolist(),expected_grid=stats['image_grid_thw'])),flush=True)
    assert len(prefix)==stats['prompt_tokens'] and inputs['image_grid_thw'].tolist()==stats['image_grid_thw']
    tokenizer=processor.tokenizer;data=build_token_enforcer_tokenizer_data(tokenizer);paths=[]
    for category in schema['properties']['category']['enum']:
        for order in ('category_first','reason_first'):
            obj={'category':category,'reason':'Source evidence.'} if order=='category_first' else {'reason':'Source evidence.','category':category}
            target=json.dumps(obj,separators=(',',':'))
            variants={'canonical':tokenizer.encode(target,add_special_tokens=False),'character_pieces':sum((tokenizer.encode(c,add_special_tokens=False) for c in target),[])}
            for kind,tokens in variants.items():
                assert tokenizer.decode(tokens,skip_special_tokens=False,clean_up_tokenization_spaces=False)==target
                fn=build_transformers_prefix_allowed_tokens_fn(data,JsonSchemaParser(schema));seq=list(prefix);fail=None
                for index,token in enumerate(tokens):
                    allowed=fn(0,torch.tensor(seq,dtype=torch.long))
                    if token not in allowed:
                        fail=dict(index=index,token_id=token,decoded_piece=tokenizer.decode([token]));break
                    seq.append(token)
                eos_allowed=False if fail else tokenizer.eos_token_id in fn(0,torch.tensor(seq,dtype=torch.long))
                paths.append(dict(category=category,field_order=order,tokenization=kind,token_count=len(tokens),all_tokens_allowed=fail is None,eos_allowed=eos_allowed,first_failure=fail))
    per_label={label:all(any(p['all_tokens_allowed'] and p['eos_allowed'] for p in paths if p['category']==label and p['field_order']==order) for order in ('category_first','reason_first')) for label in schema['properties']['category']['enum']}
    files={str(p):sha(p) for p in [request,raw_path,Path(__file__),repo/'tools/local_vlm_server.py',model/'config.json',model/'tokenizer.json',model/'tokenizer_config.json',model/'preprocessor_config.json',model/'processor_config.json',model/'chat_template.jinja'] if p.is_file()}
    result=dict(status='verified',prompt_tokens=len(prefix),image_grid_thw=inputs['image_grid_thw'].tolist(),backend_decode_used=True,prompt_encoding_matches_recorded_length_and_grid=True,
        paths=paths,per_label_reachable=per_label,all_labels_reachable=all(per_label.values()),source_bindings=files,
        model_weights_loaded=False,new_model_calls=0,GPU_calls=0,cuda_initialized=torch.cuda.is_initialized(),
        limits='This checks allowed paths with the installed tokenizer and prefix-filter implementation, not relative logits, learned preference, or visual correctness.')
    assert not result['cuda_initialized'];Path(output).write_text(json.dumps(result,indent=2),encoding='utf8')
    print(json.dumps({k:v for k,v in result.items() if k not in ('source_bindings','paths')}));return result
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--model-dir',required=True);p.add_argument('--request',required=True);p.add_argument('--output',required=True);a=p.parse_args();audit(a.model_dir,a.request,a.output)
