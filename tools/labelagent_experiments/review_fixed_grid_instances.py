"""Freeze ALL valid grid SAM outputs before unchanged full-instance semantics."""
import sys
from pathlib import Path
from prepare_fixed_grid_followup import ROOT,OUT
from run_rgb_layers import read,save
from deliver_context_graph import sha
import review_source_gap_instances as original
import review_box_layers as runner

def prepare():
    adapter=OUT/'source_selected_adapter';s=OUT/'semantic'
    if adapter.exists() or s.exists():raise FileExistsError('Preserve source semantic selection')
    source=OUT/'SAM/generation_completed.json';g=read(source);rows=[]
    for r in g['rows']:
        if r['valid']:
            rows.append(r|{'key':r['request_sha256'],'raw_pixels':r['native_pixels'],'mask_sha256':r['full_mask_sha256'],'kind':'real','source_selected':True})
    assert len({r['key'] for r in rows})==len(rows)
    save(adapter/'generation_completed.json',{'status':'complete','rows':rows,'original_generation_sha256':sha(source),'all_valid_source_selected_outputs_no_reference_or_area_filter':True,'new_model_calls':0})
    original.SOURCE=adapter;original.OUT=OUT;original.S=s;original.prepare()
    save(OUT/'semantic_source_selection.json',{'source_adapter_sha256':sha(adapter/'generation_completed.json'),'wrapper_sha256':sha(__file__),'original_semantic_preparer_sha256':sha(original.__file__),'selected_all_native_valid':len(rows),'selected_before_support_area_audit':not (OUT/'audits').exists(),'all_original_semantic_gates_unchanged':True})

if __name__=='__main__':
    if sys.argv[1]=='prepare':prepare()
    else:runner.OUT=OUT;runner.S=OUT/'semantic';runner.run()
