"""Read back both remotes and bind the independent local addendum bundle."""
import json
import shutil
import subprocess
import maintain as m
from publish_raw_mask_audit import ROOT, OUT, REPO, CONTROL, read, save


def git(*args):
    return subprocess.check_output(['git', '-C', str(REPO), *args], text=True, encoding='utf-8').strip()


def main():
    cfg = read(CONTROL / 'sync_config.json')
    head = git('rev-parse', 'HEAD')
    branch = cfg['publish_branch']
    refs = {name: git('ls-remote', '--refs', name, 'refs/heads/' + branch).split()[0] for name in ('origin', 'cnb')}
    assert all(value == head for value in refs.values()) and not git('status', '--porcelain')
    bundle = read(CONTROL / 'raw_mask_audit_bundle_verification.json')
    ui = read(CONTROL / 'raw_mask_audit_bundle_ui_verification.json')
    assert bundle['passed'] and ui['passed'] and ui['images'] == 4 and len(ui['videos']) == 1
    receipt = {**bundle, 'bundle_checked_at': bundle.get('checked_at'),
               'checked_at': m.now(), 'commit': head, 'refs': refs, 'branch': branch,
               'both_verified': True, 'working_tree_clean': True,
               'retained_pallet_support': '13/17', 'retained_convergence': 'original 6/6',
               'remaining_pallet_misses': 4, 'new_candidate_adopted': False,
               'new_real_model_calls_in_raw_audit': 0, 'annotation_writes': 0,
               'live_service_modified': False, 'default_changed': False,
               'historical_internal_mask_loss_known': False}
    save(CONTROL / 'raw_mask_final_sync_verification.json', receipt)
    home = ROOT / 'deliverables/LabelAgent_实时交接'
    save(home / '06_核验/SAM内部掩码边界审计_最终同步核对_20261107.json', receipt)
    for name in ('raw_mask_audit_bundle_verification.json', 'raw_mask_audit_bundle_ui_verification.json', 'raw_mask_published_capture_tests.json'):
        shutil.copy2(CONTROL / name, home / '06_核验' / name)
    title = '确认SAM返回前会改变掩码，原13/17与6/6保留'
    if not any(entry.get('title') == title for entry in m.read_json(CONTROL / 'important_history.json', [])):
        m.record_important(cfg, {'title': title,
            'body': f'合成CPU示例复现断开部件筛除与轮廓面积变化；不能追算旧实景损失。4图1视频及隔离捕获代码已交接，未改服务，双端{head[:7]}。'})
    m.refresh_manifest(home)
    print(json.dumps(receipt, ensure_ascii=False), flush=True)


if __name__ == '__main__':
    main()
