"""Reference-informed explanation only AFTER final prediction freeze."""
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];helpers=Path(__file__).parent
src=(helpers/'audit_semantic_stages.py').read_text(encoding='utf-8')
src=src.replace("out=ROOT/'outputs/labelagent_semantic_recovery_20261105/stage_audit'","out=ROOT/'outputs/labelagent_semantic_recovery_20261105/candidate_loss_audit'")
src=src.replace('import numpy as np,yaml','import numpy as np,yaml,hashlib\n from source.agent.edit_agent import _rasterize')
needle="  for stage,proposals in stages.items():"
insert="""  lookup={r['row']['key']:r for r in verified};admitted={r['key'] for r in read(generation/'C_selection.json')['selected'][case]}
  proposals=[]
  for file in sorted((generation.parent/'fixed_grid'/case).glob('*/prediction.json'))+sorted((ROOT/'outputs/labelagent_recovery_20261105/fixed_grid'/case).glob('*/prediction.json')):
   for shape in read(file)['PAL_shapes']:
    mask=_rasterize(shape,1920,1536);key=hashlib.sha256(case.encode()+mask.tobytes()).hexdigest();scored=ev.mask_support([shape],source,pixels,bits,owner,boxes,1920,1536,radii=(3,),minimum_points=5,minimum_purity=.67)
    matches=[o for o in scored['objects'] if o['label']=='pallet' and o['obj_id'] in ([42,48] if cam==0 else [47,48,49]) and o['best_candidate_supported_coverage']>=.5]
    if not matches:continue
    v=lookup.get(key);row={'case':case,'key':key,'source_prediction':str(file),'matches':matches,'admitted':key in admitted,'neutral_evidence_queried':v is not None,'neutral_evidence_passed':v['passed'] if v else None,'before_subject_guard':v.get('before_subject_guard') if v else None,'captions':[q['decision'] for q in v['views']] if v else [],'source_shape':shape}
    proposals.append(row)
  rows.extend(proposals)
  continue
  for stage,proposals in stages.items():"""
assert needle in src;src=src.replace(needle,insert)
exec(compile(src,__file__,'exec'),{'__name__':'__main__','__file__':__file__})
