"""Use the common JSON prefix measured in actual generated token sequences."""
from tools.enum_score_diagnostic import allowed_label_path,sequence_statistics,trace_generation,select_sequence

MEASURED_PREFIX='{\n  "category": "'

def score_label(backend,inputs,label,body_schema):
    import torch
    tokenizer=backend.processor.tokenizer;prefix_text=MEASURED_PREFIX;suffix_text=label+'"'
    prefix_ids=tokenizer.encode(prefix_text,add_special_tokens=False);target_ids=tokenizer.encode(suffix_text,add_special_tokens=False)
    assert tokenizer.decode(prefix_ids+target_ids,clean_up_tokenization_spaces=False)==prefix_text+suffix_text
    original_ids=inputs['input_ids'][0].tolist();conditioning=original_ids+prefix_ids;sequence=conditioning+target_ids
    allowed=allowed_label_path(tokenizer,original_ids,prefix_ids,target_ids,body_schema)
    if not all(t in a for t,a in zip(target_ids,allowed)):raise ValueError('Measured-prefix label path blocked')
    prepared=dict(inputs);device=inputs['input_ids'].device
    prepared['input_ids']=torch.tensor([sequence],dtype=torch.long,device=device);prepared['attention_mask']=torch.ones_like(prepared['input_ids'])
    positions=torch.arange(len(conditioning)-1,len(conditioning)+len(target_ids)-1,device=device,dtype=torch.long)
    with torch.inference_mode():output=backend.model(**prepared,use_cache=False,logits_to_keep=positions,return_dict=True)
    logits=output.logits[0].detach().float().cpu().numpy()
    return dict(label=label,prefix_token_ids=prefix_ids,prefix_text=prefix_text,target_token_ids=target_ids,target_text=suffix_text,
        conditioning_tokens=len(conditioning),allowed_token_ids=allowed,statistics=sequence_statistics(logits,target_ids,allowed)),logits
