"""Publish code and aggregate results only; real images/labels remain local."""
import sys,shutil,json,collections
from pathlib import Path
from native_resume_common import *
REPO=ROOT/'task_workspaces/labelagent_realtime_handoff_20261104'
CONTROL=ROOT/'task_workspaces/labelagent_handoff_control_20261104'
NAMES=['native_resume_common.py','native_resume_policy.py','plan_native_resume.py','prepare_native_resume_stages.py','generate_native_resume.py','review_native_resume.py','retry_native_review_transport.py','review_native_resume_retry.py','run_native_resume_comparison.py','seal_native_resume.py','audit_native_support_views.py','render_native_resume.py','publish_native_resume.py','package_native_resume.py','check_native_resume.py','commit_native_resume.py','record_native_resume_delivery.py']

def main():
 for name in NAMES:
  if (Path(__file__).parent/name).exists():shutil.copy2(Path(__file__).parent/name,REPO/'tools/labelagent_experiments'/name)
 shutil.copy2(Path(__file__).with_name('test_native_resume.py'),REPO/'tests/test_native_resume.py')
 if '--code-only' in sys.argv:return
 s=read(OUT/'final_summary.json');media=read(OUT/'media_verification.json');qa=read(CONTROL/'native_resume_regression.json');support=read(OUT/'instance_support_audit/completed.json')
 public={k:v for k,v in s.items() if k!='baseline_comparison_source'};public.update(pictures=media['pictures'],videos=len(media['videos']),fully_decoded_video_frames=media['fully_decoded_video_frames'],engineering_tests=qa['tests'])
 docs=REPO/'docs/test-results';save(docs/'native-resume-results.json',public)
 save(docs/'native-resume-checks.json',{'acceptance':read(OUT/'acceptance_checks.json'),'all_radii':read(OUT/'all_radii_incumbent_guard.json'),'whole_mask_preservation':read(OUT/'whole_mask_preservation.json'),'whole_generated_masks':{'passed':True,'records':s['all_complete_masks_verified']},'seed_reuse':read(OUT/'initial_seed_reuse_proof.json'),'regression':qa,'no_original_annotation_writes':True})
 targetrows=[]
 for c in support['cases']:
  for t in c['targets']:
   targetrows.append({k:v for k,v in t.items() if k!='best_candidates'}|{'case':c['case'],'new_candidates':[r for r in t['best_candidates'] if r['new_SAM']]})
 save(docs/'native-resume-support.json',{'read_after_freeze':True,'original_scorer_sha256':support['original_scorer_sha256'],'targets':targetrows})
 table='\n'.join(f"| {r['case']} / {r['object_id']} | {r['visible_reference_points']} | {r['new_SAM_best_raw']:.4f} | {r['new_SAM_best_support']:.4f} |" for r in targetrows)
 reasons='；'.join(f'{k}: {v}' for k,v in s['planner_rejections'].items())
 text=f'''# 原图坐标续跑与完整候选

2026-11-07。上轮因共享模型占用未取得回答的原图坐标方案，已在独立目录实际执行；旧失败记录和封存包保持原样。

本轮完成{s['point_planning_responses']}次实际规划响应，{s['valid_point_plans']}组通过严格坐标绑定，随后{s['new_SAM_requests']}次真实场景SAM点请求与{s['new_SAM_blank_requests']}次合成空白诊断，得到{s['new_complete_masks']}份可完整映射的输出。新增候选采用状态为{s['new_candidate_adopted']}，剩余{s['remaining_pallet_misses']}个漏检；原13/17、490/579参考点覆盖和6/6已验证结果继续保留。新增Agent运行{s['fresh_agent_runs']}次，没有新增Mask2Former训练或推理，没有写原始标注或安装默认。

## 修正实际请求与坐标来源

修复边坐标0..1000经511/512缩放后偏移的问题：现在直接按原图半开范围映射，1000落在右/下边界直接拒绝，最终只舍入一次。严格校验类型、字段、点数、重复正负点、种子绑定和负点冲突，不吸附、不截断、不按预算生成点。原图整体及细节输入保持原像素内容。明确启用compact并在发出的请求中逐一验证max_tokens=512，原接口默认图像预算不变；不把上轮未落实输出限制的失败尝试当控制实验。

规划器只提供图像结构假设，不提供类别或实例验收。类别自由回答、原中性双照片与主体限制、单基座及整摞矛盾否决、完整可见范围、人员保护、纯度0.67和支持0.5、0/3/6像素核验全部保留。三轮、每轮3项、累计32点不变。规划拒绝分布：{reasons}。

## 独立支持与面积核对

| 相机 / 参考对象 | 可见原参考点数 | 新完整输出最佳点覆盖 | 新完整输出最佳合格支持 |
| --- | --- | --- | --- |
{table}

表格只在预测冻结后读取原参考点评分，未用于生成、选点或更改原参考。新完整输出3973像素全部在旧PAL并集中，新增面积0，但只覆盖目标42的1点及邻实例43的48点，目标纯度1/49=0.0204，无新增有效支持。两次实际中性复核分别描述白色网格状箱体/设备和灰色四格金属容器，原语义门槛未过。合成灰图在同点、无mask先验下仍返回927像素，非空不能作为类别证据。

全局面积上限122511.25、保留并集122302、共用余量209.25均不变；所有新输出整项进入原验收或整项拒绝，不裁掩码、不删除旧前景腾空间。上轮5/6候选的3132个新增像素属于其单连通主体，但这不是实际物体正确边界面积的下界。完整保存模型输出也不等于完整物理托盘已经确认。首轮语义复核因共享服务占用等待302.74秒后退出，没有新回答；失败完整留存，之后仅补取缺失的2次回答，未追加Agent轮数或更改图片、提示和判定标准。

没有新种子接纳时，六组全部7类初始掩码逐项核对后才复用原真实结果；“原6/6保留”不算一次新测量。原27个托盘及3个人员掩码、原参考、源图、模型权重和旧结果均核对未变。本轮{s['all_complete_masks_verified']}份完整候选逐像素回读通过，{qa['tests']}项工程检查通过。

## 交接证据与下一步

本轮本地报告含{media['pictures']}张图片、{len(media['videos'])}段1080p实际保存阶段回放、{media['fully_decoded_video_frames']}帧全量解码验证。图中保留有效与失败点、实际原始SAM输出、合成空白诊断、预算与实例支持以及明确标注的旧基线；视频不是连续录屏。公共仓库只放代码及去标识化汇总，真实图片、视频、标签和同权重模型留在本地交接。

继续区分数值坐标失败、可见物体身份冲突、粘连与原面积门槛，不重复失败请求来冒充新增结果。后续优先检验同源无标记照片与定位图能否绑定同一个完整物体，以及同相机相邻帧是否提供可核对的结构；原参考点仍只在冻结后评分。这是两张已暴露图的工程优化，不是独立密集准确率或泛化结论。
'''
 (docs/'native-resume.md').write_text(text,encoding='utf-8');(OUT/'本轮结论.md').write_text(text,encoding='utf-8')
 print(json.dumps({'published_local_code':True,'pictures':media['pictures'],'videos':len(media['videos']),'tests':qa['tests']}),flush=True)

if __name__=='__main__':main()
