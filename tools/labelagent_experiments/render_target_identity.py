"""Reuse verified media layout with all counts and provenance bound to this run."""
from pathlib import Path
p=Path(__file__).with_name('render_instance_separation.py');s=p.read_text(encoding='utf-8').replace('labelagent_instance_separation_20261105','labelagent_target_identity_20261106').replace('labelagent_instance_separation_comparison_20261105','labelagent_target_identity_comparison_20261106').replace('instance_generation_consistency','instance_generation')
s=s.replace('01_同次推理与未竞争查询','01_旧查询与目标身份核查').replace('02_可见接缝与无先验SAM','02_完整基座与无先验SAM')
s=s.replace('30个完整模型查询','复用30个旧完整模型查询').replace('15次真实SAM分离试跑','11次新真实SAM分离试跑').replace('16次模型配对推理：30个完整查询；15次真实无先验SAM：7个完整返回。','复用30个旧完整查询；11次新无先验SAM得到6个完整输出，无新Mask2Former前向。').replace('6个候选进入原剩余209.25像素预算，但均未通过原有中性语义复核。','10个候选进入原剩余209.25像素预算，但均未通过原中性语义门槛。')
s=s.replace('原始候选、失败v1与保守修正v2均保留。','旧失败响应、新完整候选、前景中心输入与窄区域身份诊断均保留。')
s=s.replace("'整摞误判", "'整摞误判")
marker=' # One readable final card, with a strict instance outcome foremost.'
insert=''' # The six exact diagnostic views are never presented as new detections.
 for name in ['original_neutral_0','original_neutral_1','near','context','extent_photo','extent_binary']:
  file=assets/('12of16_'+name+'.jpg');folder=out/'target_12of16_diagnostic';panel('12/16候选身份诊断 / '+name,[(name,Image.open(folder/(name+'.png')))],file,('原两张照片中心均在候选内；候选193×22、1400像素，宽视图占比仅1.7%。','重新居中仍被描述为容器；精确掩码结构观察为edge_fragment，仅模型诊断、不是人工真值。'));add(file,'12/16候选实际输入与身份诊断','01_旧查询与目标身份核查',3)
'''
s=s.replace(marker,insert+marker)
s=s.replace("'修复单实例复核矛盾：解释含“stack/多个”时，即使状态写single也不放行。'", "'新候选：相机1/49支持5/6、纯度5/6，但新增3132像素超预算，整项拒绝。','相机0/42新候选13/16、纯度1；仍未通过可见语义，不追加。'")
s=s.replace('正负功能对照是同一10次已保存响应重新解析，未增加Agent轮数或模型复核。','结构功能对照复用原10次输入绑定结果；新身份诊断3次、候选中性复核16次另行留账。')
exec(compile(s,__file__,'exec'),{'__name__':'__main__','__file__':__file__})
