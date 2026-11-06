"""Use the unchanged scorer only after final prediction freeze."""
from pathlib import Path
p=Path(__file__).with_name('audit_target_purity.py');s=p.read_text(encoding='utf-8').replace('labelagent_target_identity_20261106','labelagent_visible_base_20261107').replace('labelagent_target_identity_comparison_20261106','labelagent_visible_base_comparison_20261107').replace("out/'seam_generation/old_and_new_hypotheses.json'","out/'seam_generation_all/old_and_new_hypotheses.json'").replace("str(out/'seam_generation') in r['source_prediction']","str(out) in r['source_prediction']").replace('new_SAM','new_complete').replace('本轮新SAM最佳者（未通过最终验收）','本轮新模型候选（无新候选时显示原图）').replace('新SAM最高','新完整候选最高')
s=s.replace("('本轮新模型候选（无新候选时显示原图）',[row['shape']])","('本轮新模型候选（无新候选时显示原图）',[row['shape']] if new else [])")
s=s.replace("m=np.maximum.reduce(masks);rgb=", "m=np.maximum.reduce(masks) if masks else np.zeros((1536,1920),np.uint8);rgb=")
s=s.replace('图示新候选：覆盖', '图示候选（若本相机无新候选则为旧诊断参考）：覆盖')
exec(compile(s,__file__,'exec'),{'__name__':'__main__','__file__':__file__})
