"""Register a verified dataset in the pinned official Qwen trainer; no source patch."""
import json
import os
from pathlib import Path
import subprocess
import sys
from data_tools import check_records, load_records, sha256


def main():
    root = Path(os.environ["LABELAGENT_PAI_ROOT"])
    export = Path(os.environ["LABELAGENT_EXPORT"])
    data_root = Path(os.environ["LABELAGENT_DATA_ROOT"])
    repo = root / "Qwen3-VL"
    pins = json.loads((Path(__file__).parent / "upstream_pin.json").read_text())
    head = subprocess.check_output(["git", "-C", str(repo), "rev-parse", "HEAD"], text=True).strip()
    if head != pins["qwen_git_commit"]:
        raise ValueError("Official Qwen training checkout differs from locked commit")
    if subprocess.check_output(["git", "-C", str(repo), "status", "--porcelain"], text=True).strip():
        raise ValueError("Official Qwen training checkout must remain clean")
    receipt = json.loads((export / "export_receipt.json").read_text())
    train_rows=check_records(load_records(export/'train_reviewed.jsonl'),data_root,True)
    dev_rows=check_records(load_records(export/'dev_reviewed.jsonl'),data_root,True)
    for key in ['group_id','source_image_sha256']:
        if {r[key] for r in train_rows} & {r[key] for r in dev_rows}:
            raise ValueError('Training/dev leakage')
    if {h for r in train_rows for h in r['image_sha256'].values()} & {h for r in dev_rows for h in r['image_sha256'].values()}:
        raise ValueError('Training/dev derived-image leakage')
    for name in ['train', 'dev']:
        if sha256(export/f'{name}.json')!=receipt[f'{name}_sha256']:
            raise ValueError('Reviewed export has changed')
    sys.path[:0] = [str(repo / 'qwen-vl-finetune'), str(repo / 'qwen-vl-finetune/qwenvl/train')]
    import qwenvl.data
    qwenvl.data.data_dict['labelagent_pilot'] = {
        'annotation_path':str((export/'train.json').resolve()),'data_path':str(data_root.resolve())}
    from qwenvl.train import train_qwen
    original = train_qwen.Trainer

    class AuditedTrainer(original):
        def __init__(self,*args,**kwargs):
            model=kwargs['model']
            names=[n for n,p in model.named_parameters() if p.requires_grad]
            if not names or any('lora_' not in n or '.visual.' in n for n in names):
                raise ValueError('Expected language-side LoRA only; trainable parameter audit failed')
            out=Path(kwargs['args'].output_dir);out.mkdir(parents=True,exist_ok=True)
            (out/f'trainable_rank{os.environ.get("RANK","0")}.json').write_text(json.dumps(names,indent=2))
            super().__init__(*args,**kwargs)

    train_qwen.Trainer=AuditedTrainer
    train_qwen.train(attn_implementation='flash_attention_2')


if __name__=='__main__':main()
