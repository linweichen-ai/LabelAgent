"""Seal a completed point-ID experiment and its full local lineage.

No model calls, remote writes, watcher changes or root handoff-index edits occur.
Original inputs and model weights are independent copies. Frozen binary media
may be hardlinked; all text, JSON, Python and configuration files are copied.
"""
import datetime
import hashlib
import html
import json
import os
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
WORK = ROOT / 'task_workspaces/labelagent_point_ids_20261107'
OUT = ROOT / 'outputs/labelagent_point_ids_20261107'
INC = ROOT / 'outputs/labelagent_context_comparison_20261105'
CONTROL = ROOT / 'task_workspaces/labelagent_handoff_control_20261104'
HOME = ROOT / 'deliverables/LabelAgent_实时交接/09_补充成果/固定编号点与邻帧归属_20261107'
MODEL_SHA = 'd11c2e7f86cd5189605b857c4697e55a5caead0a6888b52b2f5ff814150f5291'
IGNORED_NAMES = {'__pycache__', '.git'}
LINKABLE_BINARY = {'.png', '.jpg', '.jpeg', '.webp', '.bmp', '.gif', '.mp4', '.avi', '.mov',
                   '.npy', '.npz', '.safetensors', '.pt', '.pth', '.bin', '.pcd'}


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def save(path, value):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def files_under(folder):
    folder = Path(folder)
    for path in sorted(folder.rglob('*')):
        if any(part in IGNORED_NAMES for part in path.relative_to(folder).parts):
            continue
        if path.is_file() and path.suffix not in {'.pyc', '.pyo'}:
            yield path


def complete(path):
    record = read(path)
    if record.get('status') != 'complete':
        raise ValueError('Stage is not completed: ' + str(path))
    return record


def passed(path):
    record = read(path)
    if record.get('passed') is not True:
        raise ValueError('Required verification did not pass: ' + str(path))
    return record


def main():
    if (HOME / '交接SHA256.json').exists():
        raise FileExistsError('Sealed bundle is immutable; do not rerun packaging')
    cfg = read(WORK / 'config_comparison.json')
    comparison = Path(cfg['output'])
    if comparison.resolve() != (ROOT / 'outputs/labelagent_point_ids_comparison_20261107').resolve():
        raise ValueError('The current comparison path must bind this point-ID run')
    summary = read(OUT / 'final_summary.json')
    planning = complete(OUT / 'planning/completed.json')
    planner_controls = complete(OUT / 'planner_controls/completed.json')
    sam = complete(OUT / 'seam_generation/completed.json')
    generation = complete(OUT / 'instance_generation/completed.json')
    complete(comparison / 'completed.json'); complete(INC / 'completed.json')
    media = passed(OUT / 'media_verification.json')
    originals = passed(OUT / 'original_preservation.json')
    foreground = passed(OUT / 'whole_mask_preservation.json')
    whole_masks = passed(OUT / 'complete_generated_mask_proof.json')
    passed(OUT / 'all_radii_incumbent_guard.json')
    reuse = passed(OUT / 'initial_seed_reuse_proof.json')
    passed(OUT / 'budget_audit.json')
    acceptance = read(OUT / 'acceptance_checks.json')
    regression = passed(CONTROL / 'point_ids_regression.json')
    ui = passed(CONTROL / 'point_ids_ui_verification.json')
    if originals.get('original_annotations_written') != 0 or originals.get('training_updates') != 0:
        raise ValueError('Original-annotation or training preservation invariant failed')
    if foreground.get('budget_pixel_clipping') is not False or whole_masks.get('budget_clipping') is not False:
        raise ValueError('Whole-mask no-clipping proof is required')
    if ui.get('root') and Path(ui['root']).resolve() != OUT.resolve():
        raise ValueError('Browser verification belongs to a different artifact')
    video_count = len(media['videos'])
    if ui.get('images') != media['pictures'] or len(ui.get('videos', [])) != video_count:
        raise ValueError('Browser and media artifact counts do not match')
    if video_count < 3:
        raise ValueError('At least three verified actual-stage replays are required')
    if media.get('all_source_files_unchanged_during_render') is not True:
        raise ValueError('Media source preservation was not verified')
    if media.get('planning_input_images_per_request') != 2 or media.get('model_selects_IDs_not_coordinates') is not True:
        raise ValueError('Media must describe the two-photo fixed-ID experiment')
    if media.get('planner_synthetic_control_rows') != planner_controls.get('responses'):
        raise ValueError('Every actual synthetic planner-control answer must be accounted for')
    for source, expected in media.get('source_file_sha256', {}).items():
        if sha(source) != expected:
            raise ValueError('Media input changed after rendering; regenerate before sealing: ' + source)
    for item in media['videos']:
        path = (OUT / item['path']).resolve()
        if not path.is_relative_to(OUT.resolve()) or sha(path) != item['sha256']:
            raise ValueError('A video changed after decoding verification')
    for item in media.get('image_receipts', []):
        path = (OUT / item['path']).resolve()
        if not path.is_relative_to(OUT.resolve()) or sha(path) != item['sha256']:
            raise ValueError('An image changed after visual-artifact verification')
    if len(media.get('image_receipts', [])) != media['pictures']:
        raise ValueError('Every gallery image needs an individual receipt')

    old_instance = ROOT / 'outputs/labelagent_instance_separation_20261105'
    old_target = ROOT / 'outputs/labelagent_target_identity_20261106'
    old_visible = ROOT / 'outputs/labelagent_visible_base_20261107'
    old_native = ROOT / 'outputs/labelagent_native_resume_20261107'
    reuse_sources = [
        (old_instance / 'query_generation', '原30模型查询与原二值掩码'),
        (old_instance / 'seam_generation', '原7SAM与全部失败响应'),
        (old_target / 'seam_generation', '上轮6SAM与全部失败响应'),
        (old_visible / 'aspect_model_generation', '原比例输入9查询与完整掩码'),
        (old_native / 'seam_generation', '上轮原图坐标1SAM与全部失败响应'),
    ]
    reused_counts = []
    for folder, label in reuse_sources:
        complete(folder / 'completed.json')
        rows = read(folder / 'all_hypotheses.json')
        reused_counts.append({'folder': label, 'complete_candidate_records': len(rows),
                              'keys': [{'case': row['case'], 'key': row['key']} for row in rows]})
    reused_total = sum(row['complete_candidate_records'] for row in reused_counts)
    if reused_total != 53:
        raise ValueError('The historical 52+1 record lineage differs; investigate before sealing')
    generated_total = len(read(OUT / 'seam_generation/all_hypotheses.json'))
    if whole_masks['masks'] != reused_total + generated_total:
        raise ValueError('Complete-mask proof does not account for every reused and fresh record')

    required_files = [OUT / 'index.html', OUT / 'round_analysis.json',
                      OUT / 'instance_support_audit/completed.json', WORK / 'config.json',
                      CONTROL / 'point_ids_regression.log', OUT / 'planner_controls/progress.json',
                      OUT / 'temporal_audit/completed.json', OUT / 'temporal_audit/protected_inputs_readback.json',
                      OUT / 'temporal_audit/independent_readback.json']
    for path in required_files:
        if not path.is_file():
            raise FileNotFoundError('Completion receipt missing: ' + str(path))
    complete(OUT / 'instance_support_audit/completed.json')
    temporal = read(OUT / 'temporal_audit/completed.json')
    if temporal.get('status') != 'complete_RGB_diagnostic_only' or temporal.get('new_masks') != 0:
        raise ValueError('The completed same-camera RGB diagnostic must not create final masks')
    temporal_readback = passed(OUT / 'temporal_audit/independent_readback.json')
    if media.get('temporal_original_gallery_pictures') != temporal.get('pictures'):
        raise ValueError('Media must include every saved temporal diagnostic picture')
    roots = [OUT, comparison, INC, WORK / 'source_snapshot'] + [f for f, _ in reuse_sources]
    for folder in roots:
        pending = [p for p in files_under(folder) if '.pending' in p.name or p.name == 'STOP']
        if pending:
            raise ValueError('Unfinished/stopped files exist; do not seal: ' + str(pending[:3]))
    if shutil.disk_usage(ROOT).free <= 512 * 1024 ** 2:
        raise OSError('Not enough free disk space for independent model and native-input copies')

    source_digests = {}
    copy_rows = []

    def transfer(source, target, independent=False):
        source, target = Path(source), Path(target)
        if not target.resolve().is_relative_to(HOME.resolve()):
            raise ValueError('Package destination escaped the named bundle')
        digest = sha(source)
        source_digests[str(source.resolve())] = digest
        target.parent.mkdir(parents=True, exist_ok=True)
        link = not independent and source.suffix.lower() in LINKABLE_BINARY
        if target.exists():
            if sha(target) != digest:
                raise ValueError('A preserved partial-copy file differs: ' + str(target))
            if independent and os.path.samefile(source, target):
                raise ValueError('An original/weight copy must not be a hardlink')
        elif link:
            os.link(source, target)
        else:
            shutil.copy2(source, target)
        if sha(target) != digest:
            raise ValueError('Copy readback mismatch: ' + str(target))
        if independent and os.path.samefile(source, target):
            raise ValueError('Independent copy verification failed')
        copy_rows.append({'source': str(source), 'path': target.relative_to(HOME).as_posix(),
                          'sha256': digest, 'bytes': target.stat().st_size,
                          'hardlinked': os.path.samefile(source, target),
                          'independent_required': independent})
        return target

    def transfer_tree(source, target, independent=False):
        source, target = Path(source), Path(target)
        for path in files_under(source):
            transfer(path, target / path.relative_to(source), independent)

    HOME.mkdir(parents=True, exist_ok=True)
    for source, label in [(OUT, '01_主报告与全部试跑'),
                          (comparison, '02_原标准比较与结果来源'),
                          (INC, '03_保留13实例与6组结果')]:
        transfer_tree(source, HOME / label)

    native = HOME / '04_原始资料独立副本'
    data = Path(cfg['dataset']); manifest = read(data / 'import_manifest.json')
    selected = ['poses_aligned_motion.txt'] + [p for p in manifest['extracted_sha256'] if p.startswith('calib/')]
    for frame in (74, 75, 76, 77, 78):
        stamp = str(manifest['timestamps_ns'][frame])
        selected += [f'labels/{stamp}.json', f'lidar/{stamp}.pcd']
        selected += [f'cameras/camera_{cam}/{stamp}.png' for cam in range(4)]
    if len(selected) != len(set(selected)) or len(selected) != 36:
        raise ValueError('Expected the original 36 unique adjacent-frame/calibration/pose files')
    native_rows = []
    for relative in selected:
        source = (data / 'source' / relative).resolve()
        if not source.is_relative_to((data / 'source').resolve()):
            raise ValueError('Native source path escaped its dataset directory')
        expected = manifest['extracted_sha256'][relative]
        if sha(source) != expected:
            raise ValueError('Native source differs from the import manifest')
        target = transfer(source, native / 'source' / relative, independent=True)
        if sha(target) != expected:
            raise ValueError('Native source copy differs from the original manifest')
        native_rows.append({'path': relative, 'sha256': expected, 'bytes': target.stat().st_size,
                            'independent_copy': True})
    reference_rows = []
    for key in ('reference_annotations', 'reference_description'):
        target = transfer(cfg[key], native / (key + '.json'), independent=True)
        reference_rows.append({'path': target.name, 'sha256': sha(target), 'independent_copy': True})
    frame, cam = cfg['reference_case']
    record = next(r for r in manifest['records'] if r['frame'] == frame and r['topic'] == f'/capture/camera_{cam}/image')
    if sha(data / record['source_file']) != record['sha256']:
        raise ValueError('The original reference photograph changed')
    target = transfer(data / record['source_file'], native / '原参考图.png', independent=True)
    reference_rows.append({'path': target.name, 'sha256': sha(target), 'independent_copy': True})
    transfer(data / 'import_manifest.json', native / 'import_manifest_original.json', independent=True)
    save(native / '原始副本核验.json', {'passed': True, 'files': native_rows,
                                     'reference_files': reference_rows, 'originals_unchanged': True,
                                     'raw_MCAP_not_duplicated': True, 'independent_copies': True})

    repro = HOME / '05_冻结源码与配置'
    transfer_tree(WORK / 'source_snapshot', repro / 'source_snapshot')
    for path in sorted(Path(__file__).parent.glob('*.py')):
        transfer(path, repro / 'experiment_tools' / path.name, independent=True)
    for name in ('config.json', 'config_comparison.json'):
        transfer(WORK / name, repro / name, independent=True)
    qa = HOME / '06_核验'
    for name in ('point_ids_ui_verification.json', 'point_ids_regression.json', 'point_ids_regression.log'):
        transfer(CONTROL / name, qa / name, independent=True)
    for name in ('whole_mask_preservation.json', 'original_preservation.json', 'complete_generated_mask_proof.json',
                 'acceptance_checks.json', 'budget_audit.json', 'round_analysis.json', 'media_verification.json',
                 'all_radii_incumbent_guard.json', 'initial_seed_reuse_proof.json', 'final_summary.json'):
        transfer(OUT / name, qa / name, independent=True)
    for relative in ('planner_controls/completed.json', 'planner_controls/progress.json',
                     'temporal_audit/completed.json', 'temporal_audit/independent_readback.json',
                     'temporal_audit/image_readback.json', 'temporal_audit/protected_inputs_readback.json',
                     'temporal_audit/visual_observations.json'):
        transfer(OUT / relative, qa / relative, independent=True)

    package = Path(cfg['package']); model = HOME / '07_同权重Mask2Former独立模型包'
    if sha(package / 'model/model.safetensors') != MODEL_SHA:
        raise ValueError('The inherited trained model identity changed')
    transfer_tree(package / 'model', model / 'model', independent=True)
    for name in ('infer_single.py', 'calibrated_postprocess.py', 'inference_config.json', 'README.md'):
        transfer(package / name, model / name, independent=True)
    if sha(model / 'model/model.safetensors') != MODEL_SHA:
        raise ValueError('The independent model archive failed readback')
    save(qa / '模型独立复制核验.json', {'passed': True, 'sha256': MODEL_SHA, 'independent_copy': True,
                                      'training_updates': originals['training_updates']})

    reuse_root = HOME / '08_复用53候选的原响应与完整掩码'
    for source, label in reuse_sources:
        transfer_tree(source, reuse_root / label)
    save(reuse_root / '来源记录.json', {'complete_candidate_records': reused_total, 'sources': reused_counts,
                                      'new_candidates_are_in_section01': True,
                                      'absolute_source_paths_in_original_json_preserved': True,
                                      'restore_requires_rebinding_paths': True})
    save(qa / '复制来源与方式.json', {'files': copy_rows, 'text_files_are_normal_copies': True,
                                    'binary_hardlinks_are_readonly_archive_policy': True,
                                    'native_and_model_are_independent_copies': True})

    conclusions = summary.get('conclusions', [])
    limitations = summary.get('limitations', [])
    if isinstance(conclusions, str):
        conclusions = [conclusions]
    if isinstance(limitations, str):
        limitations = [limitations]
    status_lines = [
        '保留托盘支持：' + str(summary.get('retained_pallet_support', '未提供')),
        '收敛结果来源：' + str(summary.get('retained_convergence', '未提供')),
        '本轮新增 Agent 运行：' + str(summary.get('fresh_agent_runs', '未提供')),
        '新候选采用：' + str(summary.get('new_candidate_adopted', '未提供')),
        '剩余托盘漏检：' + str(summary.get('remaining_pallet_misses', '未提供')),
    ]
    conclusion_text = '\n\n'.join(status_lines + list(conclusions) + list(limitations)) + '\n'
    (HOME / '最终结论.md').write_text('# 固定编号点与邻帧归属\n\n' + conclusion_text, encoding='utf-8')
    frames = sum(v['frames'] for v in media['videos'])
    notes = f'''固定编号点与邻帧归属：已完成实验交接

打开 01_主报告与全部试跑/index.html：{media['pictures']} 张浏览图片、{video_count} 段 1920×1080 视频，全部 {frames} 帧完成解码核验。图片来自实际输入和保存返回，视频为保存阶段回放，不是连续录屏。旧基线和复用结果明确标记，不当作新增推理。

{conclusion_text}
02 保存本轮原标准比较与结果来源；03 保留原 13/17 版本及六组真实结果。是否重跑或复用以初始种子身份收据为准，不能把复用的 6/6 当新测量。原标准、三轮限制及完整掩码保护未因交接打包改变。

04 包含四相机、帧 74–78 的图像/点云/标签及原标定和位姿，共 {len(native_rows)} 份独立原始资料副本；另附原人工参考标注、说明和参考图。它们与原导入清单逐项回读一致，未改写源资料。
07 是同一已训练 Mask2Former 的独立模型包，权重 SHA256：{MODEL_SHA}。模型并非本轮新训练。
08 完整封存复用的 {reused_total} 条历史候选来源：原查询、两批旧 SAM 原响应、保持比例输入查询及上轮原图坐标新增 1 份 SAM 完整返回；原始二值掩码和失败请求也保留。本轮新完整候选记录 {generated_total} 条在 01 中，不能把历史候选数计为本轮推理数。

本轮模型规划实际接收两张同源图：无标记原图及带固定点编号的图；模型选编号，不直接编写像素坐标。{planner_controls['responses']} 条合成灰白规划对照为真实调用的功能诊断，明确不是实际场景或真实托盘结果。原始控制输入、失败答案及原返回均在 01 中。
邻帧 RGB 诊断保存 {temporal['pictures']} 张图，涉及 {temporal['seeds']} 个旧种子、{temporal['pairs']} 对同相机图像；独立回读已通过。图像关联、有效比例和锚点失败只供诊断；对象身份确认 {temporal['object_identity_confirmations']}、完整基座确认 {temporal['complete_base_confirmations']}、新增掩码 {temporal['new_masks']}。不把RGB关联数当作原Agent的6/6，也不显示任何warp为最终mask。

05 包含隔离源码、实验工具与配置。JSON 中保留真实来源路径，恢复后需按位置重绑；未重复复制完整 MCAP、SAM3/Qwen 权重或全部系统运行依赖，不宣称完整环境离线一键恢复。网页、图片和视频可离线浏览。
06 保存 {regression['tests']} 项工程检查、原保护证据、共 {whole_masks['masks']} 条完整候选逐像素证明及浏览器核验。工程检查不等于独立场景标注准确率。

文本、JSON、Python 和配置均为正常复制，避免归档文本被后续原地写入连带改动。已冻结图片/视频等二进制可使用同盘硬链接节省空间，必须视为只读；不要在归档内原地编辑，另存新版本。原始资料和模型均独立复制。备份时按内容正常复制；逐文件摘要见交接SHA256.json。
公共 Git 只同步代码及去标识化汇总，真实图像视频、标签和模型留在本地交接。此脚本没有修改根交接入口、同步进程或任何远端仓库。
'''
    (HOME / '交接说明.txt').write_text(notes, encoding='utf-8')
    (HOME / 'index.html').write_text(
        '<!doctype html><html lang="zh"><meta charset="utf-8"><title>固定编号点与邻帧归属</title>'
        '<style>body{max-width:1100px;margin:35px auto;padding:20px;font:18px/1.7 "Microsoft YaHei",sans-serif}pre{white-space:pre-wrap}</style>'
        f'<h1>固定编号点与邻帧归属</h1><p><a href="01_主报告与全部试跑/index.html">{media["pictures"]} 张图与 {video_count} 段视频</a></p>'
        '<p><a href="最终结论.md">本轮结论与证据边界</a></p><p><a href="交接SHA256.json">逐文件摘要清单</a></p><pre>'
        + html.escape(notes) + '</pre></html>', encoding='utf-8')

    changed_sources = [p for p, digest in source_digests.items() if sha(p) != digest]
    if changed_sources:
        raise ValueError('A source was still changing during packaging; do not seal: ' + str(changed_sources[:3]))
    manifest_rows = [{'path': p.relative_to(HOME).as_posix(), 'sha256': sha(p), 'bytes': p.stat().st_size}
                     for p in files_under(HOME) if p.name != '交接SHA256.json']
    save(HOME / '交接SHA256.json', {'files': manifest_rows})
    mismatches = [r['path'] for r in manifest_rows if sha(HOME / r['path']) != r['sha256']]
    if mismatches:
        raise ValueError('Sealed file readback mismatch: ' + str(mismatches[:3]))
    receipt = {
        'passed': True, 'checked_at': datetime.datetime.now().astimezone().isoformat(),
        'bundle': str(HOME), 'files': len(manifest_rows), 'mismatches': mismatches,
        'pictures': media['pictures'], 'videos': video_count, 'fully_decoded_frames': frames,
        'native_files_readback_verified': len(native_rows),
        'original_reference_files_readback_verified': len(reference_rows),
        'independent_native_copies_verified': True, 'independent_model_copy_verified': True,
        'model_sha256': MODEL_SHA, 'tests': regression['tests'],
        'incumbent_PAL_masks_preserved': sum(r['PAL_kept'] for r in foreground['incumbent']),
        'incumbent_person_masks_preserved': sum(r['person_kept'] for r in foreground['incumbent']),
        'complete_masks_verified': whole_masks['masks'], 'reused_candidate_records': reused_total,
        'new_complete_candidate_records': generated_total,
        'fresh_Agent_runs': reuse['fresh_Agent_runs'],
        'new_candidate_adopted': acceptance['adopt_new_candidate'],
        'annotation_writes': originals['original_annotations_written'],
        'training_updates': originals['training_updates'],
        'planning_responses': planning.get('point_plan_responses'),
        'valid_plans': planning.get('valid_prompts'),
        'synthetic_planner_control_responses': planner_controls.get('responses'),
        'temporal_pictures': temporal['pictures'], 'temporal_pairs': temporal['pairs'],
        'temporal_independent_readback_passed': temporal_readback['passed'],
        'temporal_new_masks': temporal['new_masks'],
        'actual_SAM_requests': sam.get('real_scene_SAM_requests'),
        'new_instances_proposed': generation.get('new_instances'),
        'source_files_unchanged_during_packaging': True,
        'text_files_are_normal_copies': True, 'root_index_or_remote_modified': False,
    }
    save(CONTROL / 'point_ids_bundle_verification.json', receipt)
    print(json.dumps(receipt, ensure_ascii=False), flush=True)


if __name__ == '__main__':
    main()
