"""First-round 2 counted context-gap fills and one unchanged full-frame SAM."""
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]
src=Path(__file__).with_name('probe_spatial_gaps.py').read_text(encoding='utf-8')
src=src.replace("out=ROOT/'outputs/labelagent_semantic_recovery_20261105/spatial_preflight'","out=ROOT/'outputs/labelagent_semantic_recovery_20261105/context_first_preflight'")
src=src.replace('spatial_sam,choose_early_hard_fills','spatial_sam,choose_context_first').replace('schedule=choose_early_hard_fills','schedule=choose_context_first')
src=src.replace("result=spatial_sam({'sam3_url':cfg['sam_url'],'sam_spatial_cache':str(out/'cache')},sam,image,prior,gaps,op['points']);raw=_rasterize(result,w,h)","""import base64,io
  buffer=io.BytesIO();Image.fromarray(prior*255).save(buffer,'PNG');file=out/(arm+'_response.json')
  result=read(file) if file.exists() else sam({'sam3_url':cfg['sam_url']},'/segment_points',image,points=op['points'],mask=base64.b64encode(buffer.getvalue()).decode())
  save(file,result);raw=_rasterize(result,w,h);Image.fromarray(raw*255).save(out/(arm+'_raw_full.png'))
  result['spatial_receipt']={'crop':None,'original_point_count':len(op['points']),'actual_point_count':len(op['points'])}""")
exec(compile(src,__file__,'exec'),{'__name__':'__main__','__file__':__file__})
