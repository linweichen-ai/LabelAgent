"""Only code and deidentified metrics go to the two public repositories."""
import json,sys,shutil,hashlib,collections
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];helpers=Path(__file__).parent;out=ROOT/'outputs/labelagent_target_identity_20261106';control=ROOT/'task_workspaces/labelagent_handoff_control_20261104';repo=ROOT/'task_workspaces/labelagent_realtime_handoff_20261104'
NAMES=['prepare_target_identity.py','target_identity_policy.py','prepare_target_generation.py','generate_target_identity.py','audit_target_input_geometry.py','diagnose_target_identity.py','prepare_target_review.py','review_target_identity.py','prepare_target_comparison.py','run_target_identity_comparison.py','summarize_target_identity.py','audit_target_purity.py','render_target_identity.py','seal_target_checks.py','publish_target_identity.py','package_target_identity.py','record_target_delivery.py']
def read(p):return json.loads(p.read_text(encoding='utf-8'))
def save(p,v):p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(v,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
def main():
 for name in NAMES+['check_target_regressions.py','verify_target_bundle.py','commit_target_identity.py']:
  if (helpers/name).exists():shutil.copy2(helpers/name,repo/'tools/labelagent_experiments'/name)
 shutil.copy2(helpers/'test_target_identity.py',repo/'tests/test_target_identity.py')
 if '--code-only' in sys.argv:return
 s=read(out/'public_summary.json');a=read(out/'instance_support_audit/completed.json');g=read(out/'target_12of16_diagnostic/geometry.json');d=read(out/'target_12of16_diagnostic/completed.json');selected=read(out/'instance_generation/selection.json');qa=read(control/'target_identity_regression.json');docs=repo/'docs/test-results'
 counts=dict(collections.Counter(r['reason'] for r in selected['rejected_whole_masks']));targets=[{'case':c['case'],**t} for c in a['cases'] for t in c['targets']];best=[{'case':t['case'],'object_id':t['object_id'],**next(r for r in t['best_candidates'] if r['new_SAM'])} for t in targets]
 blank=[]
 for row in read(out/'seam_generation/blank_controls.json')['rows']:
  r=read(Path(row['folder'])/'synthetic_blank_response.json');blank.append({'nonempty_mask':bool(r.get('parts')),'score':r.get('score'),'synthetic_not_scene_data':True,'not_semantic_validation':True})
 table='\n'.join(f"| {t['case']} / {t['object_id']} | {t['visible_reference_points']} | {t['old_SAM_best_support']:.4f} | {t['new_SAM_best_raw']:.4f} | {t['new_SAM_best_support']:.4f} | {next(r for r in t['best_candidates'] if r['new_SAM'])['added_union_pixels']} |" for t in targets)
 save(docs/'target-identity-results.json',s);save(docs/'target-identity-purity.json',{'scope':'post-freeze sparse reference agreement, not dense accuracy','targets':targets,'whole_rejection_counts':counts,'new_blank_SAM_controls':blank});save(docs/'target-identity-checks.json',{'acceptance':read(out/'acceptance_checks.json'),'originals':read(out/'original_preservation.json'),'whole_masks':read(out/'whole_mask_preservation.json'),'generated_masks':{'passed':True,'count':43},'budget':read(out/'budget_audit.json'),'tests':qa})
 save(docs/'target-identity-diagnostic.json',{'old_center_in_candidate':[r['center_in_candidate'] for r in g['old_neutral_inputs']],'old_neutral_decisions':[r['original_cached_decision'] for r in g['old_neutral_inputs']],'old_crop_target_fractions':[r['candidate_fraction_of_source_box'] for r in g['old_neutral_inputs']],'mask_pixels':g['mask_pixels'],'bbox_width_height':g['bbox_width_height'],'new_observations':d,'not_human_truth':True})
 doc=f'''# 候选身份与粘连实例纯度复测

本轮于11月6日启动、11月7日交付。最终仍13/17托盘、490/579点、人员3/3与1349/1568点、误覆盖0，三方案6/6收敛，4个漏检未恢复。新候选未通过完整原验收，保留原27托盘、3人员及所有原标注，不替换默认。

## 12/16点候选为何语义不过

原候选只有193×22的外接范围，实际1400像素。逐字节重建旧两次中性照片并命中原输入摘要，确认两张照片的中心都在候选前景内，不能将失败简单归因于中心看偏。候选在紧照片占17.63%，在宽照片仅占1.70%；宽上下文还包含较大容器结构。

原两次回答分别为不确定的深蓝矩形面/箱体及灰色分段金属容器，既没有可靠识别托盘，也存在目标区域结构混淆。保持原不含托盘类别提示的中性问题、原JSON约束与256输出上限，把两张完整候选照片都几何居中到真实前景像素，仍回答“大型深灰分段容器”和“四个灰色塑料容器”。候选没有因为换取景而自动过关。

另一次有原照片定位及精确二值掩码的结构诊断回答edge_fragment，只能看到白色矩形物体的部分横边。该模型判断不是人工真值，不能据此断言真实类别；但与窄条几何、两种上下文反复无法确认物体的证据一致，当前不足以把12/16稀疏支持当成单独可见基座。新13/16完整候选仍未通过原中性语义门槛。

## 粘连分离与纯度

新策略从相同8组原预测区域和原图接缝产生最多两项提示，正点横跨台面与下部连接/脚，负点落在相邻层；最多3正点、4负点，无先验掩码，不读三维参考位置。原掩码和模型响应从不按预算裁剪。完成11次真实SAM请求，6个有效完整输出、5项失败/告警均保留；复用原30个Mask2Former完整查询及7个旧SAM输出，没有新增Mask2Former前向或训练。合计43项完整候选逐像素与原返回核验。

| 相机/参考对象 | 原点数 | 旧SAM最佳有效支持 | 新SAM最佳点覆盖 | 新SAM最佳有效支持 | 对应新最佳候选新增并集像素 |
| --- | --- | --- | --- | --- | --- |
{table}

相机1/49的新完整候选覆盖5/6点，其中它覆盖的6个已知托盘参考点中5个归属该实例、1个属于另一实例，纯度5/6，跨过原0.67纯度及0.5支持门槛。相比旧查询最多2/6覆盖且2/6纯度、旧SAM零支持，这是本批的局部分离改善。但完整候选新增3132像素，超过保留全部前景后仅剩209.25像素预算，必须整项拒绝；没有拿更纯的候选去裁小、删旧掩码腾空间或提高面积上限。语义未对这个面积已失败的候选额外发请求，不能宣称它已通过语义。

相机0/42由旧SAM12/16、纯度1变为新候选13/16、纯度1，新增17像素，仍未通过可见语义。相机0/48依旧能被宽混合查询覆盖27/27，但原0.67纯度不合格；新SAM未恢复。相机1/47的新策略未覆盖目标参考点，原保留4/11不退步。上表是冻结后原投影一致性诊断，候选生成、排序和复核没有读取这些目标位置；并非独立二维准确率或新候选采用结果。

原并集122302，上限122511.25、数量上限36、人员半径3保护均不变。最终43项候选中，32项因完整面积超限、1项因人员冲突拒绝；进入原预算的10项均未通过原中性双视图语义。结构复核保留原两种定位、整摞矛盾否决和正负功能控制，身份可见范围审计只能增加否决条件，不能覆盖失败的中性判断。没有为获得结果增加Agent轮数或放宽任何原验收。

## 控制与边界

本轮另外2次相同点提示、无先验的合成灰图SAM均返回非空掩码。它们是明确标注的提示依赖负对照，不是实景检出，说明有完整SAM轮廓或一个非零得分仍不能证明存在物体。真实前两项请求均未形成有效输出，故没有可解释的实景/空白掩码IoU；空白控制没有被说成通过语义的样本。

结构功能控制复用同图同提示的既有10次绑定响应，正负结论及整摞矛盾拦截保留，未称作新模型能力提升。新增VLM请求分开留账：身份诊断3次、候选语义16次；Agent请求另计。前景中心距离变换增加外部背景边框，避免全前景功能图把角点当最大距离；4项新检查含全前景控制、全部目标保留、输入区域外像素隔离及台面/脚提示保护。生成时实际代码版本独立冻结，未改写已运行版本。

最终A/B四组复用同输入同策略结果，C两组从原相同SAM地面初稿真实重新复核。3轮、每轮3操作、累计32点、原0/3/6像素评分、最低5点、0.67纯度、0.5一对一支持均保持。原参考、源图、同权重及旧6组结果字节摘要一致，27托盘与3人员掩码逐像素相同，原标注写入0、训练更新0。

本地交接包含{s['pictures']}张浏览图片、{s['videos']}段1080p视频，全部{s['fully_decoded_video_frames']}帧解码核验，视频为实际保存阶段回放、不是连续录屏；包含真实/合成的明确区分、全部失败请求、精确输入与原始输出。工程回归{qa['tests']}项通过，浏览器核验全部图片、视频播放/跳转和报告链接。公共Git仅放代码和去标识化汇总，真实媒体、标签、模型仅在本地交接。两张已暴露同采集图的开发诊断，不宣称泛化、密集准确率或上线验证。

后续还需要独立确认窄可见边缘所属的完整物体、遮挡下可见基座与参考实体的对应关系，以及可确认的粘连实例边界；当前不以投影点反推掩码，不删冲突参考、不写新真值。
'''
 (docs/'target-identity.md').write_text(doc,encoding='utf-8');(out/'本轮结论.md').write_text(doc,encoding='utf-8');save(out/'query_generation/reuse_receipt.json',{'source':'labelagent_instance_separation_20261105/query_generation','source_completed_sha256':hashlib.sha256((ROOT/'outputs/labelagent_instance_separation_20261105/query_generation/completed.json').read_bytes()).hexdigest(),'inherited_forwards':16,'new_forwards_this_iteration':0,'inherited_complete_queries':30});print(json.dumps({'public_results_ready':True,'whole_rejection_counts':counts,'blank_controls':blank,'best_new_candidates':best},ensure_ascii=False),flush=True)
if __name__=='__main__':main()
