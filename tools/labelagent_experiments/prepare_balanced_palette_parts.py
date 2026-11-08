"""Bind the part-size proposal rule before any review or SAM requests."""
from pathlib import Path
import prepare_palette_parts as source
from balanced_palette_part_points import propose
from run_rgb_layers import read,save
from deliver_context_graph import sha
OUT=source.BASE/'labelagent_balanced_palette_parts_20261108'

if __name__=='__main__':
    source.OUT=OUT;source.propose=propose;source.main();p=read(OUT/'frozen_plan.json');save(OUT/'preparation_history/source_driver_plan_before_additional_bindings.json',p)
    for name in ['prepare_balanced_palette_parts.py','balanced_palette_part_points.py']:
        file=Path(__file__).with_name(name);p['protected'][str(file)]=sha(file)
    p.update(proposal_variant='Exclude any color component occupying over40 percent of original stock source pixels from part proposals; no output mask cutting or final threshold changes',base_palette_preparation_retained_with_zero_calls=True,original_distance_point_control_already_complete=True,all_additional_bindings_before_any_model_request=True);save(OUT/'frozen_plan.json',p);print({'bound_before_network':True,'source_points':len(p['rows']),'new_model_calls':0},flush=True)
