"""Publish code and deidentified negative results; media stay in local handoff."""
import json,shutil,sys,subprocess,datetime,hashlib
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];helpers=Path(__file__).parent;repo=ROOT/'task_workspaces/labelagent_realtime_handoff_20261104';out=ROOT/'outputs/labelagent_instance_separation_20261105';control=ROOT/'task_workspaces/labelagent_handoff_control_20261104'
def read(p):return json.loads(p.read_text(encoding='utf-8'))
def save(p,v):p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(v,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
NAMES=['prepare_instance_separation.py','instance_separation_geometry.py','generate_instance_queries.py','generate_seam_instances.py','instance_evidence_views.py','review_instance_separation.py','instance_identity_consistency.py','prepare_instance_consistency_review.py','review_instance_consistency.py','prepare_instance_comparison.py','run_instance_comparison.py','summarize_instance_separation.py','audit_instance_separation.py','audit_instance_query_competition.py','render_instance_support_detail.py','render_instance_separation.py','publish_instance_separation.py','package_instance_separation.py','record_instance_delivery.py']
def main():
 for name in NAMES+['seal_instance_checks.py','check_instance_regressions.py']:
  if (helpers/name).exists():shutil.copy2(helpers/name,repo/'tools/labelagent_experiments'/name)
 shutil.copy2(helpers/'test_instance_separation.py',repo/'tests/test_instance_separation.py')
 if '--code-only' in sys.argv:return
 s=read(out/'public_summary.json');audit=read(out/'instance_support_audit/completed.json');stage=read(out/'instance_support_audit/same_forward_competition.json');qa=read(control/'instance_separation_regression.json');docs=repo/'docs/test-results'
 save(docs/'instance-separation-results.json',s);save(docs/'instance-separation-checks.json',{'acceptance':read(out/'acceptance_checks.json'),'preservation':read(out/'original_preservation.json'),'whole_masks':read(out/'whole_mask_preservation.json'),'budget':read(out/'budget_audit.json'),'functional_controls':read(out/'instance_generation_consistency/functional_controls/completed.json'),'regression':qa})
 save(docs/'instance-separation-stages.json',stage)
 targetrows=[]
 for c in audit['cases']:
  for t in c['targets']:
   targetrows.append({'case':c['case'],**{k:v for k,v in t.items() if k!='best_candidates'},'best_candidates':[{k:v for k,v in r.items() if k!='source_prediction'} for r in t['best_candidates']]})
 save(docs/'instance-separation-targets.json',{'post_freeze_only':True,'targets':targetrows,'whole_rejection_counts':audit['whole_rejection_counts'],'neutral_review':audit['neutral_review']})
 table='\n'.join(f"| {r['case']} / {r['object_id']} | {r['visible_reference_points']} | {r['incumbent_best_support']:.4f} | {r['raw_pool_best_raw']:.4f} | {r['raw_pool_best_qualified']:.4f} | {r['final_best_support']:.4f} |" for r in targetrows)
 doc=f'''# 遮挡、粘连与单实例支持的原标准复测

本轮仍为13/17托盘、490/579参考点、人员3/3与1349/1568点、人员误覆盖0，三方案6/6收敛。四个漏检没有恢复，保留原27个托盘与3个人员掩码，不追加候选、不替换默认。严格超过13个合格实例这一条件未通过，不能把候选数量或点覆盖当成改进。

使用同一已训练3960步Mask2Former权重，按原预测组件和原图材质选择8组区域，每组配对32/96像素上下文，真实CPU前向16次。保留同次原配置竞争后分割和竞争前PAL原始查询，得到30个可按原几何约束无损表达的查询候选。类PAL必须原argmax且原得分>0.2、掩码>=0.5；没有null提升、重训或修改默认后处理。未采用的越界/复杂度候选亦留档。

再用图像可见层间梯度峰定位相邻层，跨左右可见部分选择最多3个正点，在相邻层最多4个负点，不输入任何先验掩码。15次真实SAM请求获得7个完整有效输出，其余8次失败或告警保留；另外2次合成空白对照均被拒绝，合成输入不计真实场景检出。所有37个完整候选只按图像和原预测选择，不读取三维参考位置。

原保留PAL并集122302像素，上限122511.25，剩余209.25像素；数量上限36不变。在最终选择中，30项完整候选超出剩余面积、1项触碰原人员半径3保护区，均整项拒绝。6项能进入原预算，但原中性双视图照片复核均未通过；其中5项为SAM、1项为原始查询。没有裁掩码腾面积，也没有为了补实例而放行边缘碎片。

| 相机/参考对象 | 原点数 | 保留候选最佳有效支持 | 全部原候选最佳点覆盖 | 全部原候选最佳有效支持 | 最终有效支持 |
| --- | --- | --- | --- | --- | --- |
{table}

相机0/42的新SAM完整候选覆盖12/16点、纯度1.0，新增并集71像素，原实例指标合格，但中性照片分别被描述为不确定的深色矩形面/灰色分段容器，不能确认单个托盘基座，未采用。该例说明点支持合格仍不能替代可见语义；此前遮挡与投影冲突的证据边界保持，未据此修改参考。

相机0/48有候选能覆盖27/27点，但混入其他实例，未通过原0.67纯度；最纯的完整候选仅1/27。相机1/47的新候选最多1/11点，比原保留4/11还少；相机1/49最多2/6且纯度不足。完整保留竞争前查询后，这些阶段最佳有效支持与同次竞争后结果分别仍为0.5、1/27、0、0；SAM为0.75、0、1/11、0。因此本批不能把剩余漏检都归因于竞争后裁剪，也不能声称增加查询已解决遮挡粘连。实例混合、弱可见证据和语义确认不足仍在。

新增单实例结构复核的第一版存在明确失败：两种定位提示都把整摞蓝托盘写成single，解释却写stack；另因枚举样式pallet_base与普通名词的分隔符不一致，木托盘正对照被额外过滤。失败版本完整保留。修正版只重解析相同10次已绑定模型响应：解释含stack/多个/复数基座则否决single，并规范名词分隔符；原独立中性照片门槛完全不变。灰图、白图、机器人、整摞托盘均不能当单托盘通过，单个木托盘正对照通过，两种提示共10个功能视图均满足预期。该修正是保守的一致性拦截，不代表模型已学会正确计数；这些对照已暴露，不能作为独立泛化准确率。v2没有为重新解析增加任何模型请求。

最终A/B四组按相同输入与策略身份绑定复用，C两组从原相同SAM地面初稿重新真实复核，仍限定3轮、每轮3操作、累计32点，最终6/6。生成阶段的16次Mask2Former、15次实景SAM、2次合成SAM、10次结构视图和12次中性语义调用与Agent调用分开留账，没有通过加轮数改善结果。候选与全部预测冻结之后才读取原稀疏3D参考作上表诊断；原评分器、0/3/6像素检查、纯度0.67、一对一支持0.5和最低5点条件不变。前景逐像素、原参考文件和六组保留结果的字节摘要均核验，原标注写入0、训练更新0。

本地交接含{s['pictures']}张可浏览图片、{s['videos']}段1920×1080视频，全部{s['fully_decoded_video_frames']}帧解码通过，另有原始输入和失败输出。视频是实际保存阶段回放，非连续录屏。工程回归{qa['tests']}项通过；浏览器需验证所有图片、视频播放与跳转。公共Git仅包含代码与去标识化汇总，真实照片、视频、原标注和模型留在本地交接。仅两张已暴露同采集图的工程比较，不能声称密集像素准确率、泛化或生产验收。

下一步证据需求：对照0/42可见基座与投影是否指向同一实体，获得不以投影点代替可见轮廓的独立实例确认；对粘连托盘需要可确认的层间实例边界。当前不删除冲突点、不改原标注，不用本批评分结果倒推选择候选。
'''
 (docs/'instance-separation.md').write_text(doc,encoding='utf-8');(out/'本轮结论.md').write_text(doc,encoding='utf-8');print('Deidentified code and final results prepared',flush=True)
if __name__=='__main__':main()
