"""Seal actual new and reused responses, rather than claiming they were rerun."""
from pathlib import Path
p=Path(__file__).with_name('package_instance_separation.py');s=p.read_text(encoding='utf-8').replace('labelagent_instance_separation_20261105','labelagent_target_identity_20261106').replace('遮挡粘连实例分离_20261105','候选身份与实例纯度复测_20261107').replace('instance_separation_ui_verification','target_identity_ui_verification').replace('instance_separation_regression','target_identity_regression').replace('instance_separation_bundle_verification','target_identity_bundle_verification').replace("['masks']==37","['masks']==43").replace('82项','86项').replace('37个','43个').replace('106张','96张')
s=s.replace('16次同权重模型前向、15次真实无先验SAM、2次明确标为合成的空白SAM诊断，30个原始查询和7个完整SAM候选全部保留。','无新Mask2Former前向，复用原30个查询和7个SAM；11次新真实无先验SAM返回6个完整候选，2次合成空白SAM单列诊断。身份诊断3次、语义新复核16次留账。原12/16窄候选仍无法确认身份，新5/6纯度候选因新增3132像素超预算拒绝。')
s=s.replace('v1失败和v2解析都保留；','旧失败输入及重新居中后的观察均保留；')
insert=""" reuse=home/'08_复用查询与旧SAM原响应';reuse.mkdir()
 for name in ['query_generation','seam_generation']:shutil.copytree(ROOT/'outputs/labelagent_instance_separation_20261105'/name,reuse/name,copy_function=frozen,ignore=shutil.ignore_patterns('__pycache__','*.pyc'))
"""
s=s.replace(" files=[{'path':p.relative_to(home).as_posix()",insert+" files=[{'path':p.relative_to(home).as_posix()")
exec(compile(s,__file__,'exec'),{'__name__':'__main__','__file__':__file__})
