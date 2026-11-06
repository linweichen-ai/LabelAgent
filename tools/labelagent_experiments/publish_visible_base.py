"""Publish auditable negative results; original images and weights remain local."""
import json,sys,shutil,hashlib,collections
from pathlib import Path
from plan_visible_bases import ROOT,work,out,read,save
helpers=Path(__file__).parent;repo=ROOT/'task_workspaces/labelagent_realtime_handoff_20261104';control=ROOT/'task_workspaces/labelagent_handoff_control_20261104'
NAMES=['prepare_visible_base.py','visible_base_policy.py','plan_visible_bases.py','generate_visible_bases.py','audit_visible_area.py','visible_base_native_policy.py','prepare_visible_native.py','plan_visible_native.py','generate_visible_native.py','generate_visible_text.py','prepare_visible_review.py','probe_visible_runtime.py','generate_visible_model.py','prepare_visible_text_inputs.py','review_visible_base.py','prepare_visible_comparison.py','run_visible_base_comparison.py','summarize_visible_base.py','audit_visible_purity.py','seal_visible_checks.py','render_visible_base.py','publish_visible_base.py','check_visible_regressions.py','package_visible_base.py','verify_visible_bundle.py','commit_visible_base.py','record_visible_delivery.py']
def main():
 for name in NAMES+['finalize_visible_owner.py']:
  if (helpers/name).exists():shutil.copy2(helpers/name,repo/'tools/labelagent_experiments'/name)
 shutil.copy2(helpers/'test_visible_base.py',repo/'tests/test_visible_base.py')
 if '--code-only' in sys.argv:return
 s=read(out/'public_summary.json');plans=read(out/'planning/progress.json');selected=read(out/'instance_generation/selection.json');qa=read(control/'visible_base_regression.json');audit=read(out/'instance_support_audit/completed.json');docs=repo/'docs/test-results';allrecords=read(out/'query_generation/all_hypotheses.json')+read(out/'seam_generation_all/old_and_new_hypotheses.json');unique=len({(r['case'],r['key']) for r in allrecords});rejections=dict(collections.Counter(r['reason'] for r in selected['rejected_whole_masks']));first=collections.Counter(r['rejection'] for r in plans);newrefs=[]
 for c in audit['cases']:
  for t in c['targets']:
   new=[r for r in t['best_candidates'] if r['new_complete']];best=max(new,key=lambda r:(r['raw_point_fraction'],r['purity'])) if new else None;newrefs.append({'case':c['case'],'object_id':t['object_id'],'visible_points':t['visible_reference_points'],'new_best_raw':t['new_complete_best_raw'],'new_best_support':t['new_complete_best_support'],'best_new_at_max_raw':best})
 save(docs/'visible-base-results.json',s);save(docs/'visible-base-checks.json',{'acceptance':read(out/'acceptance_checks.json'),'originals':read(out/'original_preservation.json'),'whole_masks':read(out/'whole_mask_preservation.json'),'initial_seeds':read(out/'initial_seed_reuse_proof.json'),'complete_generated_masks':{'passed':True,'records':len(allrecords),'unique_case_mask_keys':unique},'budget':read(out/'budget_audit.json'),'regression':qa});save(docs/'visible-base-failures.json',{'planner_declarations':[{'seed':r['index'],'cue':r['cue'],'decision':r.get('decision'),'reason':r['rejection']} for r in plans],'planner_rejections':dict(first),'planner_protocol_deviation':read(out/'planning/protocol_deviation.json'),'native_repair_not_executed':read(out/'planning_native/failure.json'),'whole_candidate_rejections':rejections,'post_freeze_new_support':newrefs})
 area=next(r for r in read(out/'area_diagnostic/completed.json')['rows'] if r['key'].startswith('2d854b6e'));save(docs/'visible-base-area.json',area)
 table='\n'.join(f"| {r['case']} / {r['object_id']} | {r['visible_points']} | {r['new_best_raw']:.4f} | {r['best_new_at_max_raw']['purity'] if r['best_new_at_max_raw'] else '无新候选'} | {r['new_best_support']:.4f} |" for r in newrefs)
 doc=f'''# 完整可见基座与不裁掩码的面积预算复测

2026-11-07。最终保持13/17托盘、490/579点、人员3/3与1349/1568点、人员误覆盖0，原6/6结果保持。4个漏检未恢复，未接纳新增候选，不替换默认。全部原27个托盘、3个人员掩码和原始标注不变。

## 5/6纯度候选的面积来自完整主体

逐像素回读上轮5/6支持与5/6纯度候选，完整掩码8289像素，仅有一个连通部分；全部3132个新增并集像素都在这个主体中。它不是可以删除几个离散小块就过关的情形。原并集122302、上限122511.25、余量209.25不变，保持该原始完整输出时至少超出2922.75像素。此结论只针对这份候选的几何，不是独立的人体/托盘真值，也不是对所有可能正确分割的面积下界。

面积红色图仅用于诊断，没有作为负点来源、剪裁模板或选择参考点的范围。本轮只能尝试让模型从实际可见结构重新生成更准确的完整实例，不能按预算扣像素、删除旧前景腾空间或提高面积上限。

## 窄条的完整物体识别与来源绑定

固定8个原图对象区域：上轮6个完整SAM输出、原窄条以及它的原图粗区域；未读取三维参考点位置。首版使用全貌和左右原始细节，开放命名实际物体并提出结构点，两种定位提示共16次真实VLM响应。部分回答自发识别为木托盘，也有容器、架子或仅边缘；但16组没有一组通过来源坐标检查，拒绝原因完整保留，未把“能说出托盘”当成完整实例识别成功。

原始点落在留白、重复/互相冲突，或模型自身只能确认edge_only，均不夹取、不吸附、不挪动来使它过关，故首版实际SAM点提示请求为0。该首版声明了512输出上限，但代码未启用compact配置，实际请求没有发送max_tokens；服务默认值未在该批逐请求绑定。因此这不是严格控制输出预算的实验，所有结果仅作失败诊断，不能据此宣称方法优劣或能力提升。

另冻结真实方形照片和显式0..1000坐标的输入修正。共享模型持续繁忙，客户端在有限等待后退出，没有取得修正版回答，也没有对应SAM推理。这部分仅是准备与失败请求记录，绝不计为16次新推理、验证成功或新样本结果。没有中断其他客户端或模型服务。运行时只读核对确认本地按需管理器和既有NF4模型配置摘要保持，服务激活前的简略health不能当作模型已更换证据。

## 真实新增推理与完整候选

独立冻结两项后续尝试：同一已训练Mask2Former权重CPU前向8次，保持原照片比例再填充到448×576；任何PAL查询只要在人工输入填充中有阳性像素就整项拒绝，不切掉阳性填充保留剩余掩码。9条可完整映射的查询记录保留，原权重摘要不变。此处改变输入方式，没有新训练、null提升、置信度阈值或后处理验收放宽。

另对8张真实方形照片用同一冻结文字提示请求SAM完整可见台面及支撑结构，全部在原置信度过滤下返回0个有效掩码，未降低0.5门槛。服务忙的传输拒绝与8次实际推理分开记账，没有用手绘轮廓替代空输出。没有有效点提示，所以本轮未产生新的合成空白SAM试跑。

连同旧30个模型查询与13个SAM输出，保留52条完整候选记录，按相机/掩码摘要去重为{unique}项；52条均与原源图坐标中的模型输出逐像素核对。超预算完整输出整项拒绝，原中性双照片、主体类别、单实例结构、整摞矛盾否决、可见范围和人员半径3保护均保留。旧同输入语义结果按输入字节/提示/别名绑定复用；本轮只为新增预算内输入发出2次真实语义请求，未通过原门槛。

| 相机/参考对象 | 原点数 | 本轮完整新查询最佳点覆盖 | 对应纯度 | 本轮最佳有效支持 |
| --- | --- | --- | --- | --- |
{table}

上表在全部候选和最终预测冻结后，才使用原稀疏投影评分器诊断。相机0/42的新完整查询虽最多覆盖13/16点，却同时覆盖实例43的50点；目标纯度13/63=0.206，原0.67门槛不通过。独立回读旧匹配确认43原本已获支持，因此补全区域在原投影指标下主要支持另一个已匹配物体，不能算目标42的完整恢复。该归属仍不是独立二维真值，未据此删点、改外参或更改参考。其余目标没有新的合格独立支持。不能将完整外形、并集点数或物体名称替代纯度、一对一支持与可见语义。原0/3/6像素检查、最低5点和0.5实例支持均不变，冲突参考没有删除。

## 结果复用、保护与交付

没有新实例接纳后，逐项对照六组全部7类初始掩码与原真实运行完全一致；六组原结果及原评分身份绑定复用，本轮新Agent运行0、Agent模型调用0。保留原6/6，不将它包装成一次新测量，也不再重跑相同地面初稿或追加轮数。三轮、每轮3操作、累计32点继续生效。

原参考、源图、模型权重及六组保留结果字节摘要不变，27托盘和3人员掩码逐像素一致。原标注写入0、训练更新0、默认安装0。{qa['tests']}项工程检查通过。交接含{s['pictures']}张浏览图片与{s['videos']}段1920×1080视频，全部{s['fully_decoded_video_frames']}帧解码核验；视频为实际保存阶段回放，原阶段明确标为复用，未执行的输入修正也明确标注。完整输入、空返回、错误坐标、共享服务繁忙记录和候选均保留。

这是两张已经暴露的同采集图的工程诊断，不是独立密集准确率、泛化或生产验收。公共Git仅发布代码和去标识化汇总，真实媒体、原标签和同权重模型包保留在本地交接。当前仍需获得可独立确认的完整可见基座轮廓和跨物体边界，才能判定更准确完整实例是否有机会在原209.25像素余量内成立；现有证据不足，不能宣布四个漏检已修复。
'''
 (docs/'visible-base.md').write_text(doc,encoding='utf-8');(out/'本轮结论.md').write_text(doc,encoding='utf-8');print(json.dumps({'records':len(allrecords),'unique':unique,'rejections':rejections,'point_plans':dict(first),'new_support':newrefs},ensure_ascii=False),flush=True)
if __name__=='__main__':main()
