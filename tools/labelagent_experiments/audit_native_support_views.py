"""Correct diagnostic panels: no new candidate must display no new mask."""
from pathlib import Path
from native_resume_common import *
helpers=Path(__file__).parent
s=(helpers/'audit_target_purity.py').read_text(encoding='utf-8').replace('labelagent_target_identity_20261106','labelagent_native_resume_20261107').replace('labelagent_target_identity_comparison_20261106','labelagent_native_resume_comparison_20261107').replace("out/'seam_generation/old_and_new_hypotheses.json'","out/'seam_generation_all/old_and_new_hypotheses.json'").replace("dest=out/'instance_support_audit'","dest=out/'instance_support_audit_corrected'")
s=s.replace("('本轮新SAM最佳者（未通过最终验收）',[row['shape']])","('本轮完整新输出（无新输出则只显示原图）',[row['shape']] if new else [])")
s=s.replace('m=np.maximum.reduce(masks);rgb=','m=np.maximum.reduce(masks) if masks else np.zeros((1536,1920),np.uint8);rgb=')
s=s.replace('图示新候选：覆盖','新候选诊断（无新候选时数值仅为历史参考）：覆盖')
exec(compile(s,__file__,'exec'),{'__name__':'__main__','__file__':__file__})
assert read(OUT/'instance_support_audit/completed.json')==read(OUT/'instance_support_audit_corrected/completed.json')
save(OUT/'support_plot_correction.json',{'old_plot_drafts_preserved':True,'original_scoring_values_unchanged':True,'fix':'camera1 had no new SAM output: show unmodified source photo instead of a historic candidate under a new-output title','canonical_figures':'instance_support_audit_corrected','model_calls':0,'predictions_or_references_modified':False})
