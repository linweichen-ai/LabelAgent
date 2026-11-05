from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];helpers=Path(__file__).parent
src=(helpers/'audit_recovery_stages.py').read_text(encoding='utf-8').replace('labelagent_recovery_20261105','labelagent_semantic_recovery_20261105').replace('labelagent_recovery_comparison_20261105','labelagent_semantic_comparison_20261105')
src=src.replace("cfg=read(work/'config.json')","cfg=read(work/'config_semantic.json')")
src=src.replace("refined=read(generation/'refinement_progress.json')['rows'];verified=read(generation/'verification_progress.json')['rows'];rows=[]","verified=read(generation/'verification_progress.json')['rows'];rows=[]")
start=src.index("  stages={'all_raw_grid_model_proposals'");end=src.index("  text=ev.checked_source",start)
src=src[:start]+"""  coarse=[s for p in sorted((ROOT/'outputs/labelagent_recovery_20261105/fixed_grid'/case).glob('*/prediction.json')) for s in read(p)['PAL_shapes']]
  stages={'old_raw_coarse_grid':coarse,'new_raw_fine_grid':raw,'combined_unfiltered_raw':coarse+raw,
   'neutral_two_view_evidence':[r['row']['shape'] for r in verified if r['row']['case']==case and r['passed']],
   'final_whole_mask_budget_selection':read(generation/case/'C_children.json')}
"""+src[end:]
(helpers/'audit_semantic_stages.py').write_text(src,encoding='utf-8')
print('post-freeze stage audit prepared')
