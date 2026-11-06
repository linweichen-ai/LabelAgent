"""Publish reproducible code and deidentified point-ID/temporal conclusions."""
import sys,shutil,json
from pathlib import Path
from point_ids_common import *
REPO=ROOT/'task_workspaces/labelagent_realtime_handoff_20261104';CONTROL=ROOT/'task_workspaces/labelagent_handoff_control_20261104'
NAMES=['point_ids_common.py','point_id_policy.py','plan_point_ids.py','prepare_point_ids_stages.py','generate_point_ids.py','repair_point_ids_zero_receipt.py','review_point_ids.py','run_point_ids_comparison.py','build_point_ids_sealer.py','seal_point_ids.py','audit_pallet_temporal_views.py','render_point_ids.py','package_point_ids.py','publish_point_ids.py','check_point_ids.py','commit_point_ids.py','record_point_ids_delivery.py']
TESTS=['test_point_id_policy.py','test_point_ids_zero_generation.py']
def main():
 for name in NAMES:
  if (Path(__file__).parent/name).exists():shutil.copy2(Path(__file__).parent/name,REPO/'tools/labelagent_experiments'/name)
 for name in TESTS:shutil.copy2(Path(__file__).parent/name,REPO/'tests'/name)
 if '--code-only' in sys.argv:return
 s=read(OUT/'final_summary.json');media=read(OUT/'media_verification.json');qa=read(CONTROL/'point_ids_regression.json');temporal=read(OUT/'temporal_audit/completed.json');docs=REPO/'docs/test-results'
 public={k:v for k,v in s.items() if k!='baseline_comparison_source'};public.update(pictures=media['pictures'],videos=len(media['videos']),fully_decoded_video_frames=media['fully_decoded_video_frames'],engineering_tests=qa['tests'])
 save(docs/'point-ids-results.json',public)
 save(docs/'point-ids-checks.json',{'acceptance':read(OUT/'acceptance_checks.json'),'all_radii':read(OUT/'all_radii_incumbent_guard.json'),'original_foreground':read(OUT/'whole_mask_preservation.json'),'complete_masks':{'passed':True,'reused_records':53,'new_records':s['new_complete_masks']},'seed_reuse':read(OUT/'initial_seed_reuse_proof.json'),'regression':qa,'zero_generation_receipt_repair':{'no_inference_added':True,'missing_empty_progress_record_fixed':True,'actual_no_network_integration_test':True}})
 rows=[]
 for r in temporal['rows']:
  for t in r['targets']:rows.append({'seed':r['index'],'case':r['case'],'frame':t['frame'],'initial_points':t['initial_points'],'valid_points':t['valid_points'],'valid_fraction':t['valid_fraction'],'anchor_valid':t['anchor_valid']})
 save(docs/'point-ids-temporal.json',{'RGB_only':True,'pairs':16,'diagnostic_supported_pairs':6,'identity_confirmations':0,'new_masks':0,'reference_points_or_calibration_read':False,'rows':rows})
 save(docs/'point-ids-contract.json',{'stacked_pallets_require_separate_instances':True,'cargo_body_excluded_from_base':True,'frame0_approval_does_not_confirm_frame76_targets':True,'boxes42_43_compatible_with_adjacent_thin_stack_layers':True,'two_dimensional_identity_confirmed':False,'steel_pallet_base_and_metal_skid_accepted_by_static_parser':True,'static_parser_examples_are_not_model_results':True,'no_label_or_threshold_change':True})
 table='\n'.join(f"| {r['seed']} / {r['case']} | {r['frame']} | {r['valid_points']}/{r['initial_points']} | {'通过' if r['anchor_valid'] else '失败'} |" for r in rows)
 text=f'''# 固定编号点与邻帧归属诊断

2026-11-07。本轮仍保留13/17托盘、490/579参考点、原有6/6 Agent收敛，剩余4个漏检未恢复，没有采用新候选。全27个旧托盘和3个人员掩码、原标注、原权重保持不变。三轮、每轮3操作、累计32点、原面积/数量/纯度/一对一支持及0/3/6像素各项保护不变。

## 编号点没有解决独立物体绑定

固定原8个图像种子，仅从完整旧候选几何产生最多23个编号点，原图中每个ID对应一个冻结整数像素。第一张为无标记同范围原照片，第二张为同照片上的原箭头/角标与点编号；没有读取3D参考点、原面积差图、预期类别答案。模型选择ID后只能精确查表，不能补点、替换、吸附或裁掩码。改动同时涉及表示和两图输入，不能当单因素因果实验。

16次真图回答均未形成有效提示：1组缺少足够正点，1组未绑定原seed，10组一面声称多个物体、一面填写点列表，另4组只给出不确定或空对象。重复冲突从前一方案7组降到0，但有效方案反而从1/16降到0/16，因此没有方法改善结论。原始回答、拒绝理由和所有图像目录完整留存，未把无效列表补成可执行SAM请求。

另有4次匹配灰白编号图诊断。托盘词声称为0/4，但4个回答都声称单物体，两个角标输入甚至形成几何有效点；这仍是物体存在性/定位幻觉，不能说空白问题已修好。原下游中性类别、主体、单基座、范围、人员及面积等门槛保留，并未用编号点的格式合法替代正确性。

本轮总计20次成功模型响应，236次共享模型忙碌503传输拒绝；实际请求参数512输出上限、温度0和两张图逐一核对。256次传输不是256次推理。无有效真图方案，因此新SAM请求0、新Mask2Former前向/训练0、新Agent运行0。所有53份历史完整输出只读复核，旧中性回答按原输入身份复用，不计新推理。六组全部7类初始种子相同后才复用原6/6，不将其包装成新测量。

## 相邻帧只能提供有限图像对应

固定8种子、同相机74/76/78三帧，先冻结金字塔LK、双向回读和有效比例条件，再运行纯RGB对应。原mask只限制起始图像角点中心；未读取标签、3D参考点、面积差、点云、外参或位姿。6/16组达到图像对应条件，0个身份确认、0个完整基座确认、0份新掩码。不能将这6/16与Agent原6/6混淆，也不能把图像warp当最终标注。

| 种子 / 相机 | 邻帧 | 有效图像点 | 原anchor |
| --- | --- | --- | --- |
{table}

关键5/6纯度候选seed05到两侧仅0/7、1/7，anchor均失败，不能据此补全。seed02/08两侧较稳定，seed03到74帧可作为下一阶段的结构观察输入；seed07到78帧总体通过但anchor失败，不能直接传播位置。56张源图与实际跟踪图逐一解码，24个无标记裁块与原图逐像素一致，16组保存坐标独立重算通过。

## 原类别与逐层实例规则仍保持

只读追溯确认“堆叠时每层分别标注”来自原用户要求；可见底部托盘、排除货箱主体来自候选包与训练负例口径。当前语义解析器要求明确托盘主体及后续结构证据，没有限制木质，也不是直接匹配container一词就拒绝；steel pallet base与metal skid的静态文本示例可通过，这只是解析行为，不是模型正确识别。

原42/43的标注几何与两个相邻叠放薄基座相容，尚不能独立确认具体二维对应。这一证据不支持把42改成前景货箱，也不能证明灰色四格物体就是托盘箱。原第0帧参考的JSON内容与已批准稿一致，旧来源字段和序列化差异另有核验；该批准不能延伸到76帧四个对象。没有修改类别、标注或验收门槛来换取通过。

下一步转向逐层结构与遮挡、描述对象偏移的诊断：需要从图像识别单层可见结构，再生成完整模型输出；继续禁止用参考点位置或剩余面积倒推候选。现有209.25像素全局余量仍是硬约束，不能通过删除旧前景或截输出绕过。

## 交付与修复

零有效规划路径曾遗漏空progress.json，但SAM、控制、输出和trace计数均为0。保留执行前源码与失败证据后，仅补齐可由零计数证明的空列表；修复生成器并通过真实执行且禁止任何网络/模型请求的集成测试，没有伪造调用。

{qa['tests']}项工程检查通过；本轮图库{media['pictures']}张图片、{len(media['videos'])}段1080p保存阶段回放，{media['fully_decoded_video_frames']}帧全量解码。它们不是连续录屏，也不是独立泛化或密集像素准确率。完整真实媒体、原资料、权重和语义审计只留本地交接；公共代码库只发布代码与去标识化汇总。
'''
 (docs/'point-ids.md').write_text(text,encoding='utf-8');(OUT/'本轮结论.md').write_text(text,encoding='utf-8');print(json.dumps({'pictures':media['pictures'],'videos':len(media['videos']),'tests':qa['tests']}),flush=True)
if __name__=='__main__':main()
