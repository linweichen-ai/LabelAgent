"""Private-instrumentation helpers; no category postprocessing or model loading."""
import json,math,re
import numpy as np

def category_span_steps(tokenizer,ids):
    text=tokenizer.decode(ids,skip_special_tokens=True,clean_up_tokenization_spaces=False)
    match=re.search(r'"category"\s*:\s*"([^"\\]*)"',text)
    if match is None:raise ValueError('Generated category span missing')
    a,b=match.span(1);steps=[];previous=0
    for index in range(len(ids)):
        end=len(tokenizer.decode(ids[:index+1],skip_special_tokens=True,clean_up_tokenization_spaces=False))
        if end<previous:raise ValueError('Nonmonotonic decoded prefix')
        if previous<b and end>a:steps.append(index)
        previous=end
    if not steps:raise ValueError('No generated token overlaps category')
    return text,match.group(1),steps

def logsumexp(values):
    values=np.asarray(values,dtype=np.float64)
    if not len(values) or not np.isfinite(values).any():raise ValueError('No finite scoring option')
    maximum=float(np.max(values));return maximum+math.log(float(np.exp(values-maximum).sum()))

def sequence_statistics(logits,target_ids,allowed):
    values=np.asarray(logits,dtype=np.float32)
    if values.ndim!=2 or len(values)!=len(target_ids) or len(allowed)!=len(target_ids):raise ValueError('Scoring shape mismatch')
    raw=[];constrained=[]
    for row,token,permitted in zip(values,target_ids,allowed):
        if not 0<=token<len(row) or token not in permitted:raise ValueError('Target token is not schema reachable')
        raw.append(float(row[token])-logsumexp(row))
        constrained.append(float(row[token])-logsumexp(row[np.asarray(permitted,dtype=np.int64)]))
    return dict(token_count=len(target_ids),raw_token_logprobs=raw,raw_sum=float(sum(raw)),raw_mean=float(sum(raw)/len(raw)),
        schema_normalized_token_logprobs=constrained,schema_normalized_sum=float(sum(constrained)))

def select_sequence(scores,key='raw_sum'):
    maximum=max(float(v[key]) for v in scores.values());best=[k for k,v in scores.items() if float(v[key])==maximum]
    return best[0] if len(best)==1 else 'uncertain'

def top_tokens(tokenizer,values,count=20):
    x=np.asarray(values,dtype=np.float32);finite=np.flatnonzero(np.isfinite(x))
    ranked=finite[np.argsort(x[finite],kind='stable')[-count:][::-1]]
    return [dict(token_id=int(i),piece=tokenizer.decode([int(i)],clean_up_tokenization_spaces=False),score=float(x[i])) for i in ranked]

def trace_generation(model,tokenizer,sink):
    original=model.generate
    def wrapped(*args,**kwargs):
        if args:raise ValueError('Diagnostic requires explicit input_ids keyword')
        prefix=kwargs['input_ids'].shape[-1]
        result=original(**kwargs,return_dict_in_generate=True,output_scores=True,output_logits=True)
        ids=result.sequences[0,prefix:].detach().cpu().tolist()
        text,category,steps=category_span_steps(tokenizer,ids)
        raw=np.stack([result.logits[i][0].detach().float().cpu().numpy() for i in steps])
        processed=np.stack([result.scores[i][0].detach().float().cpu().numpy() for i in steps])
        chosen=[ids[i] for i in steps]
        assert all(int(np.argmax(row))==token for row,token in zip(processed,chosen))
        sink.update(category=category,generated_token_ids=ids,category_steps=steps,selected_token_ids=chosen,
            raw=raw,processed=processed,steps_detail=[dict(step=step,selected_token_id=token,
                selected_piece=tokenizer.decode([token],clean_up_tokenization_spaces=False),raw_selected_score=float(r[token]),
                processed_selected_score=float(p[token]),finite_processed_options=int(np.isfinite(p).sum()),
                raw_top=top_tokens(tokenizer,r),processed_top=top_tokens(tokenizer,p)) for step,token,r,p in zip(steps,chosen,raw,processed)])
        return result.sequences
    return original,wrapped

def allowed_label_path(tokenizer,original_ids,prefix_ids,target_ids,body_schema):
    import torch
    from lmformatenforcer import JsonSchemaParser
    from lmformatenforcer.integrations.transformers import build_transformers_prefix_allowed_tokens_fn,build_token_enforcer_tokenizer_data
    fn=build_transformers_prefix_allowed_tokens_fn(build_token_enforcer_tokenizer_data(tokenizer),JsonSchemaParser(body_schema))
    # Establish the original chat boundary before consuming forced JSON tokens.
    fn(0,torch.tensor(original_ids,dtype=torch.long))
    for count in range(1,len(prefix_ids)+1):fn(0,torch.tensor(original_ids+prefix_ids[:count],dtype=torch.long))
    conditioning=original_ids+prefix_ids
    return [fn(0,torch.tensor(conditioning+target_ids[:i],dtype=torch.long)) for i in range(len(target_ids))]

def score_label(backend,inputs,label,body_schema):
    import torch
    tokenizer=backend.processor.tokenizer
    # Three complete canonical quoted label paths, fixed before inference.
    prefix_text='{"category":"';suffix_text=label+'"'
    prefix_ids=tokenizer.encode(prefix_text,add_special_tokens=False);target_ids=tokenizer.encode(suffix_text,add_special_tokens=False)
    assert tokenizer.decode(prefix_ids+target_ids,clean_up_tokenization_spaces=False)==prefix_text+suffix_text
    original_ids=inputs['input_ids'][0].tolist();conditioning=original_ids+prefix_ids;sequence=conditioning+target_ids
    allowed=allowed_label_path(tokenizer,original_ids,prefix_ids,target_ids,body_schema)
    if not all(t in a for t,a in zip(target_ids,allowed)):raise ValueError('Canonical label path blocked')
    prepared=dict(inputs);device=inputs['input_ids'].device
    prepared['input_ids']=torch.tensor([sequence],dtype=torch.long,device=device)
    prepared['attention_mask']=torch.ones_like(prepared['input_ids'])
    positions=torch.arange(len(conditioning)-1,len(conditioning)+len(target_ids)-1,device=device,dtype=torch.long)
    with torch.inference_mode():
        output=backend.model(**prepared,use_cache=False,logits_to_keep=positions,return_dict=True)
    logits=output.logits[0].detach().float().cpu().numpy()
    statistics=sequence_statistics(logits,target_ids,allowed)
    return dict(label=label,prefix_token_ids=prefix_ids,prefix_text=prefix_text,target_token_ids=target_ids,target_text=suffix_text,
        conditioning_tokens=len(conditioning),allowed_token_ids=allowed,statistics=statistics),logits
