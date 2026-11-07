"""Compare base/adapter on reviewed cases. Never writes workbench annotations."""
import argparse
import json
from pathlib import Path
import time
from data_tools import PROMPT, check_records, expected, image_path, load_records, sha256


def decode_answer(text):
    try:
        value=json.loads(text.strip())
    except (ValueError, TypeError):
        return None
    if not isinstance(value,dict) or set(value)!={'decision','label'}:return None
    if value['decision'] not in {'keep','reject','relabel','uncertain'}:return None
    if value['label'] not in {'pallet','person','other','unknown'}:return None
    if value['decision']=='uncertain' and value['label']!='unknown':return None
    if value['decision']=='reject' and value['label']!='other':return None
    if value['decision'] in {'keep','relabel'} and value['label'] not in {'pallet','person'}:return None
    return value


def summarize(rows):
    if not rows: raise ValueError('Empty evaluation')
    groups={}
    for r in rows:groups.setdefault(r['expected']['decision'],[]).append(r)
    recalls={k:sum(r['correct'] for r in v)/len(v) for k,v in groups.items()}
    return {'cases':len(rows),'valid_json':sum(r['parsed'] is not None for r in rows),
        'exact_decision_and_label_correct':sum(r['correct'] for r in rows),
        'accuracy':sum(r['correct'] for r in rows)/len(rows),
        'balanced_accuracy_over_present_decisions':sum(recalls.values())/len(recalls),
        'per_decision_recall':recalls,
        'abstained_on_decidable':sum(r['expected']['decision']!='uncertain' and r['parsed'] is not None and r['parsed']['decision']=='uncertain' for r in rows),
        'destructive_action_on_keep':sum(r['expected']['decision']=='keep' and r['parsed'] is not None and r['parsed']['decision'] in {'reject','relabel'} for r in rows),
        'invalid_or_noop_relabel':sum(r['parsed'] is not None and (r['parsed']['decision']=='relabel' and r['parsed']['label']==r['proposed_label'] or r['parsed']['decision']=='keep' and r['parsed']['label']!=r['proposed_label']) for r in rows),
        'caveat':'Candidate review task only, not pixel accuracy or full Agent convergence'}


def main():
    p=argparse.ArgumentParser();p.add_argument('--model',required=True);p.add_argument('--adapter');p.add_argument('--records',required=True)
    p.add_argument('--data-root',required=True);p.add_argument('--out',required=True);p.add_argument('--max-pixels',type=int,default=451584)
    p.add_argument('--training-export');p.add_argument('--independent-test',action='store_true');a=p.parse_args()
    records=check_records(load_records(a.records),a.data_root,True)
    if not records:raise ValueError('No human-reviewed records')
    if a.independent_test:
        if not a.training_export:raise ValueError('Independent test needs training lineage to check')
        train=load_records(Path(a.training_export)/'train_reviewed.jsonl')
        dev=load_records(Path(a.training_export)/'dev_reviewed.jsonl')
        for key in ['capture_id','group_id','source_image_sha256']:
            if {r[key] for r in records}&{r[key] for r in train+dev}:raise ValueError('Not an independent test: '+key)
    out=Path(a.out)
    if out.exists():raise ValueError('Fresh output required')
    out.mkdir(parents=True)
    import torch
    from PIL import Image
    from transformers import AutoProcessor,Qwen3VLForConditionalGeneration
    model=Qwen3VLForConditionalGeneration.from_pretrained(a.model,dtype=torch.bfloat16,device_map='auto',attn_implementation='flash_attention_2')
    if a.adapter:
        from peft import PeftModel
        model=PeftModel.from_pretrained(model,a.adapter,is_trainable=False)
    model.eval()
    processor=AutoProcessor.from_pretrained(a.model,min_pixels=12544,max_pixels=a.max_pixels)
    rows=[]
    with (out/'responses.jsonl').open('x',encoding='utf-8') as f:
        for r in records:
            images=[Image.open(image_path(a.data_root,n)).convert('RGB') for n in r['images']]
            messages=[{'role':'user','content':[{'type':'image','image':im} for im in images]+[{'type':'text','text':PROMPT.replace('<image>\n','')}]}]
            inputs=processor.apply_chat_template(messages,tokenize=True,add_generation_prompt=True,return_dict=True,return_tensors='pt').to(model.device)
            start=time.monotonic()
            with torch.inference_mode():generated=model.generate(**inputs,max_new_tokens=192,do_sample=False)
            text=processor.batch_decode(generated[:,inputs['input_ids'].shape[-1]:],skip_special_tokens=True)[0]
            answer=decode_answer(text)
            valid_semantics=answer is not None and not (answer['decision']=='relabel' and answer['label']==r['proposed_label'] or answer['decision']=='keep' and answer['label']!=r['proposed_label'])
            row={'sample_id':r['sample_id'],'proposed_label':r['proposed_label'],'expected':expected(r),'text':text,'parsed':answer,
                 'correct':valid_semantics and answer==expected(r),'seconds':time.monotonic()-start,
                 'image_grid_thw':inputs['image_grid_thw'].tolist(),'input_tokens':inputs['input_ids'].shape[-1]}
            rows.append(row);f.write(json.dumps(row,ensure_ascii=False)+'\n');f.flush()
    result=summarize(rows);result.update({'model':a.model,'adapter':a.adapter,'records_sha256':sha256(a.records),
        'max_pixels':a.max_pixels,'independent_test':a.independent_test,'state_writes':0,
        'total_inference_seconds':sum(r['seconds'] for r in rows),'cuda_peak_bytes':[torch.cuda.max_memory_allocated(i) for i in range(torch.cuda.device_count())]})
    (out/'metrics.json').write_text(json.dumps(result,ensure_ascii=False,indent=2));print(json.dumps(result,ensure_ascii=False,indent=2))


if __name__=='__main__':main()
