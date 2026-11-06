"""Reuse original preservation checks, deriving only this run's new conclusions."""
from pathlib import Path
helpers=Path(__file__).parent
s=(helpers/'seal_native_resume.py').read_text(encoding='utf-8').replace('from native_resume_common import *','from point_ids_common import *')
s='\n'.join(line for line in s.splitlines() if "summary['conclusions'].append('新3973像素" not in line)+'\n'
extra=''' controls=read(OUT/'planner_controls/completed.json');temporal=read(OUT/'temporal_audit/completed.json');transport=read(OUT/'planning/request_receipts.json')
 summary.update(planner_synthetic_control_responses=controls['responses'],planner_blank_PAL_hallucinations=controls['PAL_hallucinations'],planner_blank_all_abstain=controls['all_abstain_with_empty_points'],point_ID_blank_PAL_guard_passed=controls['PAL_guard_passed'],planner_total_actual_responses=plans['point_plan_responses']+plans['synthetic_control_responses'],request_transmissions=len(transport),busy_503_transmissions=sum(r.get('http_status')==503 for r in transport),temporal_RGB_supported_pairs=temporal['diagnostic_supported_pairs'],temporal_RGB_pairs=temporal['pairs'],temporal_RGB_pictures=temporal['pictures'],temporal_identity_confirmations=temporal['object_identity_confirmations'],two_changed_factors_not_single_factor_ablation=True)
 summary['conclusions'].append(f"新增4次带同样标记的灰白规划对照：托盘声称{controls['PAL_hallucinations']}/4，但未全部弃权；不能宣布空白图物体幻觉已解决。")
 summary['conclusions'].append(f"同相机邻帧纯RGB诊断{temporal['diagnostic_supported_pairs']}/{temporal['pairs']}组达到冻结的图像对应条件；没有物体身份或完整基座确认，不能计作实例恢复。")
 summary['limitations'].append('本轮同时改变点表示和输入视图安排，不是单因素消融；20次真实模型响应和忙拒绝传输分开统计。')
 if (OUT/'semantic_contract_audit.json').exists():
  summary['semantic_contract_audit_sha256']=sha(OUT/'semantic_contract_audit.json')
  summary['conclusions'].append('原标注42/43为几乎同平面位置的两个薄Pallet框，高差约一层；与叠放基座相容，但二维归属未确认。用户原规则要求每层一个实例，不能把货箱改称托盘换取通过。')
 summary['limitations'].append('灰白编号图虽未称托盘，四个都声称单物体，两个角标输入甚至形成有效点；物体存在性与定位仍不可靠。')
 summary['zero_generation_receipt_repaired_without_inference']=(OUT/'engineering_failures/zero_generation_receipt/repair.json').exists()
'''
s=s.replace(" save(OUT/'final_summary.json',summary);",extra+" save(OUT/'final_summary.json',summary);")
(helpers/'seal_point_ids.py').write_text(s,encoding='utf-8')
