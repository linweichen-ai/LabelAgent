"""Publish code and aggregate functional evidence; no real labels or media."""
import hashlib
import json
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / 'outputs/labelagent_raw_mask_audit_20261107'
WORK = ROOT / 'task_workspaces/labelagent_raw_mask_audit_20261107'
REPO = ROOT / 'task_workspaces/labelagent_realtime_handoff_20261104'
CONTROL = ROOT / 'task_workspaces/labelagent_handoff_control_20261104'
HELPERS = Path(__file__).resolve().parent


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def save(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')


def main():
    tests = read(OUT / 'capture_tests.json')
    independent = read(OUT / 'independent_review.json')
    functional = read(OUT / 'function_diagnostic/completed.json')
    media = read(OUT / 'function_diagnostic/replay/media_verification.json')
    ui = read(CONTROL / 'raw_mask_audit_ui_verification.json')
    assert all(r['passed'] for r in (tests, independent, functional, media, ui))
    code = {name: HELPERS / name for name in (
        'sam_raw_capture.py', 'probe_raw_mask_boundary.py',
        'package_raw_mask_audit.py', 'publish_raw_mask_audit.py',
        'commit_raw_mask_audit.py', 'record_raw_mask_delivery.py')}
    for name in ('point_segment_source_copy.py', 'run_function_diagnostic.py', 'render_function_replay.py'):
        code['raw_mask_function_diagnostic/' + name] = WORK / 'function_diagnostic' / name
    published = []
    for name, source in code.items():
        target = REPO / 'tools/labelagent_experiments' / name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)
        published.append(target.relative_to(REPO).as_posix())
    test_target = REPO / 'tests/test_sam_raw_capture.py'
    shutil.copy2(HELPERS / 'test_sam_raw_capture.py', test_target)
    published.append(test_target.relative_to(REPO).as_posix())
    rows = [{key: row[key] for key in (
        'case', 'raw_pixels', 'wire_pixels', 'raw_to_wire_removed_pixels',
        'raw_to_wire_added_pixels', 'click_occupancy_preserved', 'warnings', 'wire_mapping_proof')}
        for row in functional['cases']]
    public = {
        'status': 'CPU_functional_boundary_verified_not_real_SAM_inference',
        'real_model_calls': 0, 'new_masks_for_acceptance': 0,
        'service_changes': 0, 'original_annotations_written': 0,
        'new_candidate_adopted': False, 'remaining_misses': 4,
        'retained_pallet_support': '13/17', 'retained_convergence': 'original 6/6',
        'old_wire_to_full_image_preservation_still_valid': True,
        'historical_raw_to_wire_loss_known': False,
        'in_memory_worker_source_attested': False,
        'cases': rows, 'functional_gallery_images': 4,
        'functional_videos': 1, 'fully_decoded_frames': 240,
        'capture_tests': tests['tests'],
        'capture_scope': 'private module only; all predictor mask candidates copied before source processing; errors retained; CPU synthetic tests only',
        'next_real_capture': 'prepared, not executed; coordinate the existing single-GPU manager before inference',
        'acceptance_limits_changed': False}
    docs = REPO / 'docs/test-results'
    save(docs / 'raw-mask-boundary.json', public)
    report = f'''# SAM内部掩码与返回轮廓的证据边界

2026-11-07。发现服务内部掩码到返回轮廓之间的组件筛选与简化，会改变像素；过去的完整性证明只覆盖**服务已返回轮廓到全图的映射**。本补记保留原记录，不把旧内部位图没有保存的部分推断成已知损失。

本轮直接执行当前磁盘服务函数，用明确的合成预测器替代模型。没有真实SAM/GPU/HTTP推理，没有修改服务、标注、模型权重或默认方案。以下均为CPU功能案例，不是实景分割精度。

| 功能案例 | 原二值面积 | 返回轮廓面积 | 删除像素 | 新增像素 |
| --- | ---: | ---: | ---: | ---: |
| 两大断开组件及小块，仅左部正点 | 21004 | 10000 | 11004 | 0 |
| 单组件非点击锯齿边界 | 64431 | 64600 | 0 | 169 |
| 同一断开示例加合成prior | 21004 | 21000 | 4 | 0 |

三例点击位置一致、无警告，已有wire映射检查仍全部0增删。原因是返回之前先按轮廓面积过滤，无prior时又保留含正点组件，之后简化轮廓；只有点击处变化才触发不简化回退。因此仅核验返回数据不够证明内部掩码保全。

不能由这些合成数字推断剩余四个真实托盘的损失；历史模型内部位图无法由旧轮廓恢复。本次监听和启动路径只证实管理器配置，采样时只有Qwen工作进程、SAM未加载，不能宣称历史SAM内存代码已被证明。

新增隔离捕获工具：按源码SHA加载私有模块，在predict_inst返回后复制所有mask候选和scores，保留原数组类型与内容，把同一返回对象交还原函数；捕获后处理异常时也留存已产生数组，不把失败当有效分割。它不能对现有live service模块直接打补丁。{tests['tests']}项CPU合成检查通过，真实GPU捕获尚未执行；不会因为脚本存在就计作新SAM结果。

下一次真实捕获需先与现有单GPU管理器协调，保存同一次请求的原输入、实际提示点、模型内部全部候选、选中索引、返回轮廓和差集。逐层候选继续只用RGB与既有图像提案，不能输入3D目标坐标、层高、参考点或剩余面积来反推掩码。原27个托盘和3个人员掩码保持，原面积/纯度/支持/人员及三轮限制全部不变。

目前仍是13/17、原有6/6，剩余4个漏检未恢复。前一轮编号点及邻帧交接包独立保留103图5视频，本补记另有4图1段24秒功能回放、240帧解码验证，不混算为实景新推理。

源码与合成用例可查tools/labelagent_experiments/sam_raw_capture.py、tests/test_sam_raw_capture.py和raw_mask_function_diagnostic子目录。历史功能运行脚本保留原机器路径和冻结快照依赖；迁机复现需重绑定这些路径。本地交接保留完整原始数组、源码副本、回读清单和失败路径，不向公共仓库发布真实数据或坐标。
'''
    (docs / 'raw-mask-boundary.md').write_text(report, encoding='utf-8')
    (OUT / '本轮补记.md').write_text(report, encoding='utf-8')
    published += ['docs/test-results/raw-mask-boundary.json', 'docs/test-results/raw-mask-boundary.md']
    save(CONTROL / 'raw_mask_publication_files.json', {
        'paths': published,
        'sha256': {name: hashlib.sha256((REPO / name).read_bytes()).hexdigest() for name in published},
        'private_scene_media_published': False})
    print(json.dumps({'files': len(published), 'capture_tests': tests['tests'], 'real_model_calls': 0}), flush=True)


if __name__ == '__main__':
    main()
