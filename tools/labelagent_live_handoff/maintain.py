"""Incremental private handoff and fast-forward-only dual Git publication.

All machine paths and remote settings come from a local, unpublished config.
Never reads model inputs, modifies annotations, or starts inference.
"""
from __future__ import annotations
import argparse
import ctypes
import hashlib
import html
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import time
import zipfile
from datetime import datetime, timezone, timedelta
from urllib.parse import unquote
from html.parser import HTMLParser

TZ = timezone(timedelta(hours=8))
METHODS = ('pallet_baseline', 'four_crop_control', 'spatial_component_mass_control',
           'raw_certificate_priority', 'raw_certificate_components')


def now():
    return datetime.now(TZ).isoformat(timespec='seconds')


def shared_bytes(path):
    """Allow the producer to atomically replace a file while it is read."""
    path = Path(path)
    if os.name != 'nt':
        return path.read_bytes()
    import msvcrt
    k = ctypes.WinDLL('kernel32', use_last_error=True)
    k.CreateFileW.argtypes = [ctypes.c_wchar_p, ctypes.c_uint32, ctypes.c_uint32,
                             ctypes.c_void_p, ctypes.c_uint32, ctypes.c_uint32, ctypes.c_void_p]
    k.CreateFileW.restype = ctypes.c_void_p
    k.CloseHandle.argtypes = [ctypes.c_void_p]
    handle = k.CreateFileW(str(path.resolve()), 0x80000000, 7, None, 3, 0x80, None)
    if handle == ctypes.c_void_p(-1).value:
        raise ctypes.WinError(ctypes.get_last_error())
    try:
        fd = msvcrt.open_osfhandle(handle, os.O_RDONLY | os.O_BINARY)
    except Exception:
        k.CloseHandle(handle)
        raise
    with os.fdopen(fd, 'rb') as f:
        return f.read()


def read_json(path, default=None):
    for attempt in range(3):
        try:
            return json.loads(shared_bytes(path))
        except FileNotFoundError:
            return default
        except (PermissionError, json.JSONDecodeError):
            if attempt == 2:
                raise
            time.sleep(.1)


def atomic(path, content):
    path = Path(path)
    data = content.encode('utf-8') if isinstance(content, str) else content
    if path.exists() and shared_bytes(path) == data:
        return False
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + '.pending')
    temp.write_bytes(data)
    for attempt in range(10):
        try:
            os.replace(temp, path)
            return True
        except PermissionError:
            if attempt == 9:
                raise
            time.sleep(.1 * (attempt + 1))


def write_json(path, value):
    return atomic(path, json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n')


def digest(path):
    return hashlib.sha256(shared_bytes(path)).hexdigest()


def git(repo, *args, check=True, raw=False):
    result = subprocess.run(['git', '-C', str(repo), *args], capture_output=True, stdin=subprocess.DEVNULL,
                            creationflags=subprocess.CREATE_NO_WINDOW if os.name == 'nt' else 0,
                            encoding='utf-8', errors='replace', timeout=150,
                            env={**os.environ, 'GIT_TERMINAL_PROMPT': '0', 'GCM_INTERACTIVE': 'Never'})
    if check and result.returncode:
        detail = safe_message(result.stderr or result.stdout)
        detail = re.sub(r'https://[^/@\s]+:[^/@\s]+@', 'https://[redacted]@', detail)
        raise RuntimeError('Git operation failed: ' + ' '.join(args[:2]) +
                           ' (exit ' + str(result.returncode) + '): ' + detail[:700])
    return (result.stdout if raw else result.stdout.strip()), result.returncode


def process_matches(pid, created):
    if os.name != 'nt' or not pid or not created:
        return None
    from ctypes import wintypes as w
    k = ctypes.WinDLL('kernel32', use_last_error=True)
    k.OpenProcess.argtypes = [w.DWORD, w.BOOL, w.DWORD]
    k.OpenProcess.restype = w.HANDLE
    k.GetProcessTimes.argtypes = [w.HANDLE] + [ctypes.POINTER(w.FILETIME)] * 4
    k.GetExitCodeProcess.argtypes = [w.HANDLE, ctypes.POINTER(w.DWORD)]
    k.CloseHandle.argtypes = [w.HANDLE]
    h = k.OpenProcess(0x1000, False, int(pid))
    if not h:
        return False
    try:
        fields = [w.FILETIME() for _ in range(4)]
        code = w.DWORD()
        if not k.GetProcessTimes(h, *[ctypes.byref(f) for f in fields]):
            return None
        if not k.GetExitCodeProcess(h, ctypes.byref(code)):
            return None
        actual = ((fields[0].dwHighDateTime << 32) + fields[0].dwLowDateTime) / 1e7 - 11644473600
        return code.value == 259 and abs(actual - float(created)) < .05
    finally:
        k.CloseHandle(h)


def safe_message(text):
    # Entire credential-bearing messages are excluded, even from the private export.
    if re.search(r'password|密码|私钥|令牌|\btoken\b|Bearer\s|gh[pousr]_|github_pat_', text, re.I):
        return '[含账户或凭据信息的消息已省略]'
    text = re.sub(r'<oai-mem-citation>[\s\S]*?</oai-mem-citation>', '', text)
    text = re.sub(r'\b(?:\d{1,3}\.){3}\d{1,3}\b', '[本地或私有地址]', text)
    return text.strip()


def export_threads(config, state):
    results = []
    for item in config['threads']:
        path = Path(item['path'])
        st = path.stat()
        previous = state.get(item['id'], {})
        target = Path(config['handoff']) / '05_对话记录' / (item['id'] + '.txt')
        if (previous.get('size') == st.st_size and target.exists()
                and previous.get('summary', {}).get('notice_filter_version') == 2):
            results.append(previous['summary'])
            continue
        messages, lines, last, updates = 0, [], None, []
        with path.open(encoding='utf-8') as f:
            for line in f:
                try:
                    event = json.loads(line)
                except json.JSONDecodeError:
                    continue  # A producer may be appending its last line now.
                p = event.get('payload', {})
                if event.get('type') != 'response_item' or p.get('type') != 'message':
                    continue
                role = p.get('role')
                if role not in ('user', 'assistant'):
                    continue
                text = '\n'.join(v.get('text', '') for v in p.get('content', [])
                                 if v.get('type') in ('input_text', 'output_text'))
                if not text or text.startswith(('<codex_internal_context', '<environment_context',
                                                '<external_codex_apps_open_page')):
                    continue
                if role == 'user' and '## My request:' in text:
                    text = text.split('## My request:', 1)[1]
                elif role == 'user' and '<in-app-browser-context' in text:
                    continue
                text = safe_message(text)
                if not text:
                    continue
                messages += 1
                last = event.get('timestamp')
                lines.append(f"[{last}] {'用户' if role == 'user' else '助手'}\n{text}\n")
                retrospective = re.search(r'是的[，,].{0,10}问过|当时讨论了|您之前问过|您明确问过', text[:300])
                if role == 'assistant' and not retrospective and re.search(r'完成|通过|结果|失败|出错|无法|已建|重要|需要您', text):
                    updates.append({'time': last, 'text': text[:500]})
        atomic(target, '仅导出用户与助手可见文本；不含工具输出，凭据消息省略。\n\n' + '\n'.join(lines))
        summary = {'id': item['id'], 'messages_exported': messages, 'latest_message_utc': last,
                   'important_updates': updates[-5:], 'notice_filter_version': 2}
        state[item['id']] = {'size': st.st_size, 'summary': summary}
        results.append(summary)
    return results


class LocalMediaReferences(HTMLParser):
    def __init__(self):
        super().__init__()
        self.paths = []

    def handle_starttag(self, tag, attrs):
        if tag in ('img', 'video', 'source'):
            self.paths.extend(v for k, v in attrs if k in ('src', 'poster') and v)


def collect_delivered_artifacts(config):
    """Collect local outputs explicitly linked in the watched chats, never tool logs."""
    from handoff_artifact_policy import approved_artifact
    workspace = Path(config.get('media_workspace', Path(config['handoff']).parents[1])).resolve()
    home = Path(config['handoff'])
    control = Path(config['control'])
    old = read_json(control / 'delivered_artifacts.json', {})
    entries = dict(old)
    candidates = set()
    extensions = {'.png', '.jpg', '.jpeg', '.gif', '.mp4', '.webm', '.json', '.md', '.txt'}
    def approved(path):
        return approved_artifact(path, workspace, config.get('artifact_approved_roots', []))
    for transcript in (home / '05_对话记录').glob('*.txt'):
        text = shared_bytes(transcript).decode('utf-8')
        for match in re.finditer(r'\]\(([^\n]*?)\)', text):
            raw = unquote(match.group(1).strip().strip('<>'))
            if not re.match(r'^[A-Za-z]:[/\\]', raw):
                continue
            path = approved(Path(raw))
            if not path:
                continue
            if path.suffix.lower() == '.html':
                parser = LocalMediaReferences()
                parser.feed(shared_bytes(path).decode('utf-8', errors='replace'))
                for item in parser.paths:
                    if re.match(r'^(?:https?:|data:|//)', item):
                        continue
                    target = approved(path.parent / unquote(item))
                    if target and target.suffix.lower() in extensions:
                        candidates.add(target)
            elif path.suffix.lower() in extensions:
                candidates.add(path)
    for gallery_root in config.get('artifact_gallery_roots', []):
        for index in Path(gallery_root).glob('*/index.html'):
            path = approved(index)
            if not path:
                continue
            parser = LocalMediaReferences()
            parser.feed(shared_bytes(path).decode('utf-8', errors='replace'))
            for item in parser.paths:
                if re.match(r'^(?:https?:|data:|//)', item):
                    continue
                target = approved(path.parent / unquote(item))
                if target and target.suffix.lower() in extensions:
                    candidates.add(target)
    existing = {}
    for folder in ('03_图片', '04_视频', '09_新增结果', '09_补充成果'):
        for p in (home / folder).rglob('*'):
            if p.is_file():
                existing[digest(p)] = p.relative_to(home).as_posix()
    added = []
    for source in sorted(candidates):
        info = source.stat()
        key = str(source)
        previous = old.get(key, {})
        if previous.get('size') == info.st_size and previous.get('mtime_ns') == info.st_mtime_ns:
            external = previous.get('storage_mode') == 'external_reference_low_space'
            retry_copy = external and shutil.disk_usage(home).free >= info.st_size + 96 * 1024**2
            if not retry_copy and (home / previous['destination']).exists():
                continue
        if time.time() - info.st_mtime < config.get('artifact_stable_seconds', 10):
            continue
        if info.st_size > 512 * 1024 * 1024:
            continue
        data = shared_bytes(source)
        after = source.stat()
        if (after.st_size, after.st_mtime_ns) != (info.st_size, info.st_mtime_ns):
            continue
        suffix = source.suffix.lower()
        if suffix in ('.md', '.txt', '.json'):
            text = data.decode('utf-8', errors='replace')
            if safe_message(text) == '[含账户或凭据信息的消息已省略]':
                continue
        sha = hashlib.sha256(data).hexdigest()
        if sha in existing:
            destination = existing[sha]
        else:
            group = '04_视频/对话新增' if suffix in ('.mp4', '.webm') else (
                '03_图片/对话新增' if suffix in ('.png', '.jpg', '.jpeg', '.gif') else '09_新增结果')
            destination = group + '/' + source.stem[:75] + '_' + sha[:12] + suffix
            # Preserve the original 96 MiB reserve when media grows. An
            # explicit external index keeps D-drive evidence discoverable;
            # it does not pretend the bytes were copied to the C handoff.
            if shutil.disk_usage(home).free < len(data) + 96 * 1024**2:
                entries[key] = {'size': info.st_size, 'mtime_ns': info.st_mtime_ns,
                                'sha256': sha, 'destination': str(source.resolve()),
                                'storage_mode': 'external_reference_low_space',
                                'copied_into_handoff': False}
                continue
            atomic(home / destination, data)
            existing[sha] = destination
            added.append(destination)
        entries[key] = {'size': info.st_size, 'mtime_ns': info.st_mtime_ns,
                        'sha256': sha, 'destination': destination}
    write_json(control / 'delivered_artifacts.json', entries)
    write_json(home / '06_核验/新增资料来源.json', {'files': entries})
    external_paths = [v['destination'] for v in entries.values()
                      if v.get('storage_mode') == 'external_reference_low_space']
    atomic(home / '06_核验/空间保护_外部原始资料索引.txt',
           '以下资料保留在原位置，未复制到C盘交接；请连同这些文件或其完整交接目录一起备份。\n'
           '同步器保持96MiB媒体复制空间余量，空间恢复后会补齐复制。\n' + '\n'.join(external_paths))
    return added


def notify_desktop(config, title, body):
    script = config.get('notification_script')
    if not script:
        return {'status': 'not_configured'}
    payload = Path(config['control']) / 'notification_payload.json'
    write_json(payload, {'title': title, 'body': body[:220]})
    try:
        result = subprocess.run(['powershell.exe', '-NoProfile', '-ExecutionPolicy', 'Bypass',
                                 '-File', script, '-Payload', str(payload)],
                                capture_output=True, stdin=subprocess.DEVNULL,
                                creationflags=subprocess.CREATE_NO_WINDOW if os.name == 'nt' else 0,
                                timeout=20, encoding='utf-8', errors='replace')
        return {'status': 'submitted' if result.returncode == 0 else 'unavailable',
                'exit_code': result.returncode}
    except (OSError, subprocess.TimeoutExpired):
        return {'status': 'unavailable'}


def record_important(config, event):
    control = Path(config['control'])
    home = Path(config['handoff'])
    notices = read_json(control / 'important_history.json', [])
    event = {'time': now(), **event}
    event['desktop_notification'] = notify_desktop(config, 'LabelAgent · ' + event['title'], event['body'])
    notices.append(event)
    write_json(control / 'important_history.json', notices)
    write_json(home / '06_核验/重要进展历史.json', notices)
    atomic(home / '08_重要进展.txt', '\n\n'.join(
        f"[{v['time']}] {v['title']}\n{v['body']}" for v in reversed(notices)))
    return event


def report_changes(config, sync, observation, threads, added):
    path = Path(config['control']) / 'notice_state.json'
    prior = read_json(path, {})
    latest = {t['id']: t['latest_message_utc'] for t in threads}
    state = {'producer_commit': sync['producer_commit'], 'scores': observation['scores'],
             'primary_588_passed': observation['primary_588_passed'], 'threads': latest,
             'status': 'ok'}
    reasons, details = [], []
    if prior:
        if prior.get('producer_commit') != state['producer_commit']:
            reasons.append('项目更新已同步')
        if prior.get('scores') != state['scores'] or prior.get('primary_588_passed') != state['primary_588_passed']:
            reasons.append('验证结果有更新')
        if prior.get('status') == 'error':
            reasons.append('双平台同步已恢复')
        for thread in threads:
            seen = prior.get('threads', {}).get(thread['id'])
            for update in thread.get('important_updates', []):
                if seen and update.get('time') and update['time'] > seen:
                    text = re.sub(r'\[([^\]]*)\]\([^)]*\)', r'\1', update['text'])
                    rank = 3 if re.search(r'完整.{0,12}验证|未通过|评分|误覆盖', text) else 1
                    details.append((rank, re.sub(r'\s+', ' ', text)[:180]))
        if details:
            reasons.append('对话有重要进展')
    if added:
        reasons.append(f'新增{len(added)}份媒体或结果')
    if reasons:
        record_important(config, {'title': '；'.join(reasons),
                         'body': (max(details, key=lambda item: item[0])[1] if details else '两端提交已核对一致，新增资料已归入交接目录。'),
                         'commit': sync['commit'], 'new_artifacts': added})
    write_json(path, state)


def report_failure(config, result):
    path = Path(config['control']) / 'notice_state.json'
    state = read_json(path, {})
    key = result.get('error_type', '') + ':' + result.get('message', '')
    if state.get('error_key') != key or state.get('status') != 'error':
        record_important(config, {'title': '同步需要关注',
                         'body': '自动同步遇到问题，已保留现有文件并将重试。详情见交接目录后台同步状态。'})
    state.update(status='error', error_key=key)
    write_json(path, state)


def compact_score(value):
    if value.get('status') != 'complete' or value.get('evaluated_images') != 588:
        return None
    if value.get('metric_version') != 2 or value.get('missing_prediction_images') != 0:
        return None
    try:
        p, h = (value['groups']['all'][label] for label in ('pallet', 'person'))
        return dict(images=588, matched=p['radii']['3']['objects_matched_at_support_threshold'],
                    reference_objects=p['supported_reference_objects'],
                    covered_points=p['radii']['3']['covered_projected_points'],
                    reference_points=p['reference_projected_points'], candidates=p['candidates'],
                    person_cross_points=h['radii']['3']['points_covered_by_other_class'])
    except KeyError:
        return None


def public_observation(config):
    root = Path(config['evidence'])
    pipeline = read_json(root / 'raw_certificate_full_pipeline_v2.json', {})
    fields = ('status', 'full_images', 'primary_method', 'completed_requests', 'completed_images',
              'last_completed_stage', 'all_full_predictions_frozen_before_scoring')
    result = {'scope': 'same-capture sparse projected reference agreement; not pixel accuracy',
              'pipeline': {k: pipeline[k] for k in fields if k in pipeline},
              'pipeline_process_verified_live': process_matches(pipeline.get('pid'), pipeline.get('process_created')),
              'scores': {}, 'primary_588_passed': None, 'default_workbench_changed_by_sync': False,
              'annotation_writes_by_sync': False, 'model_requests_by_sync': 0}
    full = root / 'raw_certificate_full_predictions_v1'
    for method in METHODS:
        score = compact_score(read_json(full / ('evaluation_' + method) / 'summary.json', {}))
        if score:
            result['scores'][method] = score
    comparison = read_json(full / 'comparison.json', {})
    if comparison.get('status') == 'complete' and comparison.get('scored_images_per_method') == 588:
        if comparison.get('primary_method') == 'raw_certificate_priority' and len(result['scores']) == 5:
            passed = comparison.get('primary_passed')
            if type(passed) is bool:
                result['primary_588_passed'] = passed
                result['original_gate_comparison_complete'] = True
                primary = next((row for row in comparison.get('rows', [])
                                if row.get('method') == 'raw_certificate_priority'), {})
                allowed = ('strict_instance_gain_over_previous_best', 'point_coverage_not_below_previous_best',
                           'person_off_class_not_above_previous_best', 'original_minimum_two_percentage_point_gain',
                           'original_count_ratio_limit', 'original_union_area_limit',
                           'original_person_cross_class_limit', 'all_original_candidates_preserved')
                result['primary_failed_checks'] = [key for key in allowed if primary.get('checks', {}).get(key) is False]
                result['comparison_sha256'] = digest(full / 'comparison.json')
    video = read_json(Path(config['video']) / '_audit/verification_receipt.json', {})
    result['video_verified'] = video.get('status') == 'complete' and video.get('video_fully_decoded') is True
    return result


def copy_verified(source, target, expected=None):
    data = shared_bytes(source)
    actual = hashlib.sha256(data).hexdigest()
    if expected and expected != actual:
        raise ValueError('Media hash does not match the verified producer receipt')
    atomic(target, data)
    return actual


def collect_media(config):
    home = Path(config['handoff'])
    video = Path(config['video'])
    receipt = read_json(video / '交付信息.json', {})
    verify = read_json(video / '_audit/verification_receipt.json', {})
    if receipt.get('sha256') and verify.get('video_sha256') == receipt['sha256']:
        copy_verified(video / 'LabelAgent_操作演示_20261104.mp4', home / '04_视频/01_真实Agent运行.mp4', receipt['sha256'])
        for file in ('verification_receipt.json', 'encoding_receipt.json', 'recording_receipt.json'):
            # Local-only audit files; never stage them in the Git repository.
            copy_verified(video / '_audit' / file, home / '06_核验' / file)
        copy_verified(video / 'poster.png', home / '03_图片/视频封面.png')
    folders = [(Path(config['guide']), home / '03_图片/操作七步'),
               (Path(config['evidence']) / 'raw_certificate_tuning_visual_review_v1', home / '03_图片/调参诊断'),
               (video / '_audit/qa_frames_final', home / '03_图片/视频抽帧')]
    for src, dst in folders:
        for item in sorted(src.glob('*')):
            if item.suffix.lower() in ('.png', '.jpg', '.jpeg'):
                copy_verified(item, dst / item.name)
    for file in ('index.html',):
        path = folders[1][0] / file
        if path.exists():
            copy_verified(path, folders[1][1] / file)
    guide = Path(config['guide']) / 'LabelAgent_操作演示.html'
    if guide.exists():
        copy_verified(guide, home / '03_图片/操作七步/操作教程.html')


def public_document(observation, source, cnb):
    passed = observation['primary_588_passed']
    conclusion = '尚未取得完整验收结论' if passed is None else ('通过冻结的同集参考标准' if passed else '未通过冻结的同集参考标准')
    rows = ['| 方法 | 实例支持 | 覆盖参考点 | 人员交叉点 | 托盘候选 |',
            '| --- | --- | --- | --- | --- |']
    for method, score in observation['scores'].items():
        rows.append(f"| {method} | {score['matched']}/{score['reference_objects']} | "
                    f"{score['covered_points']}/{score['reference_points']} | {score['person_cross_points']} | {score['candidates']} |")
    table = '\n'.join(rows)
    reason = ('主方案人员交叉覆盖超过旧四分块，因此按原标准不采用；不放宽要求、不替换默认流程。'
              if 'person_off_class_not_above_previous_best' in observation.get('primary_failed_checks', []) else '')
    return f'''# 实时交接与双平台同步

更新于：{now()}（北京时间）

- 本交接分支同时发布到 GitHub 与 CNB，以提交摘要相同为同步完成条件。
- 上游工作代码提交：`{source}`；CNB 既有进度提交：`{cnb}`。
- 用户已补齐第三个独立链接，目前读取三段对话，涵盖研究优化、CNB同步和真实操作视频。
- 首轮冻结 588 图：{conclusion}。阶段 `{observation['pipeline'].get('status', 'unknown')}`，已完成评分组数 {len(observation['scores'])}/5。后续试验见最新进度文档。
- 当前数据为常熟0920：152 时刻、4 相机、608 图；参考采用 labels。20 图用于调参，588 图用于同采集复验。
- 32 图已有结果：实例 118→119/205，参考点覆盖 7406→7603，人员交叉点保持119。完整结论以588图原标准核验为准。
- 稀疏三维参考投影一致性不能解释为二维像素准确率或独立外部验证。未进行 Qwen 或 SAM3 权重微调。
- 本地已有 1080p 中文字幕真实运行视频（约3分22秒），另整理操作截图和调参诊断图，交给用户备份及飞书整理。
- 图片、视频、对话导出、原始数据、坐标、模型权重、授权信息只留本地交接目录，不随本分支发布。
- 持续任务仅同步已提交代码及白名单汇总，不启动推理、不确认标注。默认每60秒检查；机器休眠、退出或网络不可用时暂停，恢复后续传。STOP文件可停止。

机器可读阶段见 [live-handoff.json](live-handoff.json)。原始阶段记录见 [dataset-comparison.md](dataset-comparison.md)。

## 完整588图逐组结果

{table}

{reason}
'''


def sync_git(config, observation):
    repo = Path(config['repo'])
    if git(repo, 'status', '--porcelain')[0]:
        raise RuntimeError('Isolated sync checkout contains uncommitted changes; review before resuming')
    source_ref = 'refs/remotes/producer/' + config['producer_branch']
    cnb_ref = 'refs/remotes/cnb/' + config['cnb_progress_branch']
    git(repo, 'fetch', 'producer', 'refs/heads/' + config['producer_branch'] + ':' + source_ref)
    source = git(repo, 'rev-parse', source_ref)[0]
    git(repo, 'fetch', 'cnb', 'refs/heads/' + config['cnb_progress_branch'] + ':' + cnb_ref)
    cnb = git(repo, 'rev-parse', cnb_ref)[0]
    for commit in (source, cnb):
        if git(repo, 'merge-base', '--is-ancestor', commit, 'HEAD', check=False)[1]:
            # Refuse publishing newly added dataset/model/media/credential files.
            paths = git(repo, 'diff', '--name-only', '--diff-filter=A', 'HEAD...' + commit)[0].splitlines()
            forbidden = ('.png', '.jpg', '.jpeg', '.mp4', '.mcap', '.pcd', '.pt', '.pth', '.safetensors', '.npz', '.env')
            if any(p.lower().endswith(forbidden) or re.search(r'(^|/)(auth|credentials)\.', p, re.I) for p in paths):
                raise RuntimeError('Incoming commit adds private data/media/model or credential paths; review required')
            try:
                git(repo, 'merge', '--no-edit', commit)
            except Exception:
                conflicts = git(repo, 'diff', '--name-only', '--diff-filter=U')[0].splitlines()
                allowed = set(config.get('producer_authoritative_reports', []))
                if conflicts and set(conflicts) <= allowed:
                    # Only explicitly configured generated reports use the owning
                    # producer version. Both upstream histories remain intact.
                    for path in conflicts:
                        content = git(repo, 'show', source + ':' + path)[0]
                        atomic(repo / path, content + '\n')
                        git(repo, 'add', '--', path)
                    git(repo, 'commit', '--no-edit')
                else:
                    git(repo, 'merge', '--abort', check=False)
                    raise
    payload = {'schema': 2, 'producer_commit': source, 'cnb_progress_commit': cnb, **observation}
    dest = repo / 'docs/progress/live-handoff.json'
    prior = read_json(dest, {})
    if prior != payload:
        write_json(dest, payload)
        atomic(repo / 'docs/progress/live-handoff.md', public_document(observation, source, cnb))
        git(repo, 'add', '--', 'docs/progress/live-handoff.json', 'docs/progress/live-handoff.md')
        git(repo, 'commit', '-m', 'Update verified live handoff and dual-platform checkpoint')
    head = git(repo, 'rev-parse', 'HEAD')[0]
    refs = {}
    for remote in ('origin', 'cnb'):
        ref = 'refs/heads/' + config['publish_branch']
        current = git(repo, 'ls-remote', remote, ref)[0]
        remote_head = current.split()[0] if current else None
        if remote_head != head:
            git(repo, 'push', remote, head + ':' + ref)  # No force, ever.
        remote_head = git(repo, 'ls-remote', remote, ref)[0].split()[0]
        if remote_head != head:
            raise RuntimeError('Remote verification differs from the intended commit')
        refs[remote] = remote_head
    return {'checked_at': now(), 'commit': head, 'producer_commit': source,
            'cnb_progress_commit': cnb, 'refs': refs, 'both_verified': len(set(refs.values())) == 1,
            'branch': config['publish_branch']}


def refresh_manifest(home, overflow_directory=None):
    home = Path(home)
    manifest = [{'path': p.relative_to(home).as_posix(), 'bytes': p.stat().st_size, 'sha256': digest(p)}
                for p in sorted(home.rglob('*')) if p.is_file()
                and p.name not in ('文件SHA256清单.json', '当前文件清单位置.json')]
    payload = {'created_at': now(), 'files': manifest}
    required = len(json.dumps(payload, ensure_ascii=False, indent=2).encode('utf-8'))
    external = bool(overflow_directory and shutil.disk_usage(home).free < required + 48 * 1024**2)
    target = (Path(overflow_directory) if external else home / '06_核验') / '文件SHA256清单.json'
    write_json(target, payload)
    write_json(home / '06_核验/当前文件清单位置.json',
               {'path': str(target.resolve()), 'sha256': digest(target), 'members': len(manifest),
                'external_due_to_low_space': external,
                'old_local_manifest_retained_as_historical': external,
                'backup_requires_current_manifest_too': external})
    return manifest


def build_handoff(config, observation, sync, threads):
    home = Path(config['handoff'])
    copy = home / '02_代码' / ('LabelAgent_' + sync['commit'][:12] + '.zip')
    archive_source = copy
    if not copy.exists():
        bases = sorted((home / '02_代码').glob('LabelAgent_*.zip'), key=lambda p:p.stat().st_mtime, reverse=True)
        if bases and shutil.disk_usage(home).free < 512 * 1024**2:
            base = bases[0]
            base_commit = git(config['repo'], 'rev-parse', base.stem.removeprefix('LabelAgent_') + '^{commit}')[0]
            stem = 'LabelAgent_' + sync['commit'][:12]
            patch = home / '02_代码' / (stem + '.patch')
            atomic(patch, git(config['repo'], 'diff', '--binary', '--full-index', base_commit, sync['commit'], raw=True)[0])
            tree = git(config['repo'], '-c', 'core.quotepath=false', 'ls-tree', '-r', '-z', '--full-tree', sync['commit'])[0]
            files = {entry.split('\t',1)[1]:entry.split('\t',1)[0].split()[2] for entry in tree.split('\0') if entry}
            write_json(home / '02_代码' / (stem + '.json'), {'mode':'verified_full_base_plus_binary_diff',
                       'base_zip':base.name,'base_zip_sha256':digest(base),'base_commit':base_commit,
                       'target_commit':sync['commit'],'patch':patch.name,'patch_sha256':digest(patch),'git_blobs':files})
            shutil.copy2(Path(__file__).with_name('restore_code_snapshot.py'), home / '02_代码/restore_code_snapshot.py')
            atomic(home / '02_代码/空间不足时的代码备份说明.txt',
                   '旧完整ZIP全部保留。空间不足时，新提交以完整ZIP＋二进制差异＋逐文件Git内容摘要交接，没有丢弃代码更新。\n'
                   '还原：python restore_code_snapshot.py --receipt LabelAgent_目标提交.json --outdir 空目录\n'
                   '完整ZIP、对应JSON、patch和还原脚本需一起备份；脚本会核对还原后的每个目标文件。\n')
            archive_source = Path(config['control']) / 'current_docs.zip'
            git(config['repo'], 'archive', '--format=zip', '--output=' + str(archive_source.resolve()), sync['commit'], 'docs')
        else:
            git(config['repo'], 'archive', '--format=zip', '--output=' + str(copy.resolve()), sync['commit'])
    with zipfile.ZipFile(archive_source) as archive:
        for name in archive.namelist():
            parts = Path(name).parts
            if name.startswith('docs/') and not name.endswith('/') and '..' not in parts:
                atomic(home / '07_文档' / Path(*parts[1:]), archive.read(name))
    write_json(home / '06_核验/双平台提交核对.json', sync)
    write_json(home / '06_核验/当前进度.json', observation)
    write_json(home / '06_核验/对话读取范围.json', {'provided_links': 3, 'unique_threads': threads,
               'duplicate_first_second_link_corrected_by_user': True, 'third_distinct_link_pending': False})
    conclusion = observation['primary_588_passed']
    status = '尚未取得完整588图验收结论' if conclusion is None else ('通过冻结同集标准' if conclusion else '未通过冻结同集标准')
    pictures = sorted(p for p in (home / '03_图片').rglob('*') if p.suffix.lower() in ('.png', '.jpg', '.jpeg', '.gif'))
    videos = sorted(p for p in (home / '04_视频').rglob('*') if p.suffix.lower() in ('.mp4', '.webm'))
    scores = '\n'.join(f"{method}: 实例{v['matched']}/{v['reference_objects']}，点{v['covered_points']}/{v['reference_points']}，人员交叉{v['person_cross_points']}，候选{v['candidates']}"
                       for method, v in observation['scores'].items())
    failed = 'person_off_class_not_above_previous_best' in observation.get('primary_failed_checks', [])
    reason = '主方案人员交叉覆盖超过旧四分块，未通过；保持原标准，不采用为默认。' if failed else ''
    publication = ('云端同步待重试；本页包含最新本地资料。上次双端已核对提交：' + str(sync.get('last_verified_commit', '无'))
                   if sync.get('sync_pending') else '两端提交已回读一致。')
    text = f'''LabelAgent 交接说明（可直接整理到飞书）
更新时间：{now()}

当前代码：{sync['commit']}
工作源提交：{sync['producer_commit']}
GitHub/CNB交接分支：{sync['branch']}；{publication}

1. 数据和效果
当前主测常熟0920数据：152时刻、四相机、608图。labels全帧作为参考；Task3为历史另一套数据。
20图调参、588图同采集复验。已有32图实例118→119/205、点覆盖7406→7603，人员交叉点119不变。
首轮冻结588图：{status}；阶段 {observation['pipeline'].get('status')}；评分 {len(observation['scores'])}/5。
后续人员保护等试验的最新进展见08_重要进展.txt及07_文档，首轮结果作为历史对照保留。
{scores}
{reason}
当前指标是稀疏三维参考投影一致性，不是二维像素准确率；本轮常熟采集未做权重微调。外部CVAT已训练模型的迁移测试另见新增报告，尚无独立外部测试结论。

2. 图片和视频
本文件夹包含 {len(pictures)} 张图片、{len(videos)} 个视频。
01_真实Agent运行.mp4：真实第76帧操作，1080p、中文字幕、约3分22秒、无配音。
候选仍需人工确认；操作七步图片说明入口、选相机、参考帧、启动、SAM3和复查结果。
调参诊断包含疑似重复及叉车部件混淆样例，目视判断不作为真值。
补充视频若标为截图讲解，是图片顺序展示，不能当作另一轮连续模型录屏。

3. 备份与持续更新
整个目录可直接备份；先打开00_交接入口.html。02_代码保留按提交命名的源码ZIP；06_核验含SHA256清单和提交核对。
同步器每60秒检查已提交代码、对话文本、真实进度与新增素材；原工作台和实验进程由原任务维护。
同时收集对话中明确交付的本地图片、视频和结果摘要；新增资料按内容摘要保留版本，图片视频只留本地。
重要结果、代码更新、新资料及同步失败会发送Windows桌面通知，并记录在08_重要进展.txt；无变化不重复提醒。
机器休眠、进程退出或网络故障期间不能更新；恢复后续传。只有失败、完成或有实质变化时需要关注。
凭据消息和工具输出不进入对话导出；图片、视频和对话记录只保留本地，飞书上传由用户进行。
用户已补齐第三个独立链接，三段对话均已读取，脱敏文本在05_对话记录。

4. 操作、模型与部署资料
07_文档保留当前代码提交内的完整进度、部署与测试文档。
local-gpu.md：本机Qwen3-VL-2B NF4与SAM3单卡轮换。
local-vlm.md：视觉模型接口、独立验收与远端配置。
pai-dsw.md：PAI环境配置；准备完成不等于更大模型已部署。
workbench-walkthrough.md：参考帧、目标帧、自动标注和人工保存步骤。
progress/dataset-comparison.md：历史里程碑和失败实验；test-results记录各项对照。
本地工作台 http://127.0.0.1:8765/ ；逐步教程 http://127.0.0.1:8767/ 。离线图片与视频不依赖这些服务。
'''
    atomic(home / '01_进度与飞书说明.txt', text)
    blocks = ['<!doctype html><html lang="zh-CN"><meta charset="utf-8"><title>LabelAgent 实时交接</title>',
              '<style>body{font:16px/1.7 "Microsoft YaHei",sans-serif;background:#f1f5f9;color:#172033;max-width:1100px;margin:30px auto;padding:24px}section{background:white;padding:24px;border-radius:14px;margin:20px 0}img{max-width:100%;border-radius:8px}video{width:100%}pre{white-space:pre-wrap}a{color:#1260bd}.grid{display:grid;grid-template-columns:1fr 1fr;gap:18px}</style>',
              '<h1>LabelAgent 实时交接</h1>']
    local_reports = sorted((home / '09_补充成果').rglob('index.html'),
                           key=lambda p: p.stat().st_mtime, reverse=True)
    if local_reports:
        blocks.append('<section><h2>新增对照与诊断报告</h2>')
        for report in local_reports:
            rel = report.relative_to(home).as_posix()
            label = report.relative_to(home).parts[1] + ' / ' + report.parent.name
            blocks.append(f'<p><a href="{html.escape(rel)}">{html.escape(label)}</a></p>')
        blocks.append('</section>')
    blocks += ['<section><pre>' + html.escape(text) + '</pre></section>', '<section><h2>视频</h2>']
    for v in videos:
        rel = v.relative_to(home).as_posix()
        blocks.append(f'<h3>{html.escape(v.stem)}</h3><video controls preload="metadata" src="{html.escape(rel)}"></video>')
    blocks.append('</section><section><h2>图片证据</h2><div class="grid">')
    for p in pictures:
        rel = html.escape(p.relative_to(home).as_posix())
        blocks.append(f'<figure><a href="{rel}"><img loading="lazy" src="{rel}"></a><figcaption>{html.escape(p.relative_to(home).as_posix())}</figcaption></figure>')
    blocks.append('</div></section><section><h2>备份文件</h2>')
    for p in sorted(home.rglob('*')):
        if p.is_file() and p.suffix in ('.zip', '.txt', '.json', '.md', '.html') and p.name not in ('文件SHA256清单.json', '00_交接入口.html'):
            rel = html.escape(p.relative_to(home).as_posix())
            blocks.append(f'<p><a href="{rel}">{rel}</a></p>')
    blocks.append('</section></html>')
    atomic(home / '00_交接入口.html', '\n'.join(blocks))
    manifest = refresh_manifest(home, config.get('manifest_overflow_directory'))
    return {'pictures': len(pictures), 'videos': len(videos), 'files': len(manifest)}


def cycle(config):
    control = Path(config['control'])
    control.mkdir(parents=True, exist_ok=True)
    home = Path(config['handoff'])
    for folder in ('02_代码', '03_图片', '04_视频', '05_对话记录', '06_核验'):
        (home / folder).mkdir(parents=True, exist_ok=True)
    state = read_json(control / 'thread_offsets.json', {})
    threads = export_threads(config, state)
    write_json(control / 'thread_offsets.json', state)
    observation = public_observation(config)
    collect_media(config)
    added = collect_delivered_artifacts(config)
    try:
        sync = sync_git(config, observation)
    except Exception as error:
        # Local deliverables must remain available during a remote outage.
        prior = read_json(control / 'latest.json', {})
        pending = {**prior, 'commit': git(config['repo'], 'rev-parse', 'HEAD')[0],
                   'producer_commit': prior.get('producer_commit', '未核对'),
                   'branch': config['publish_branch'], 'both_verified': False,
                   'sync_pending': True, 'last_verified_commit': prior.get('commit'),
                   'error': safe_message(str(error))[:250]}
        counts = build_handoff(config, observation, pending, threads)
        write_json(control / 'local_handoff.json', {'status': 'local_ready_remote_pending',
                   'checked_at': now(), 'commit': pending['commit'], **counts})
        raise
    report_changes(config, sync, observation, threads, added)
    write_json(home / '06_核验/后台同步状态.json', {'status': 'ok', 'checked_at': now(),
               'pid': os.getpid(), 'interval_seconds': config.get('interval_seconds', 60),
               'stop_file': str(control / 'STOP'), 'last_verified_commit': sync['commit']})
    counts = build_handoff(config, observation, sync, threads)
    receipt = {'status': 'ok', 'checked_at': now(), 'pid': os.getpid(), **sync, **counts,
               'pipeline_stage': observation['pipeline'].get('status'),
               'primary_588_passed': observation['primary_588_passed']}
    write_json(control / 'latest.json', receipt)
    return receipt


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', required=True)
    parser.add_argument('--watch', action='store_true')
    args = parser.parse_args()
    config = read_json(args.config)
    control = Path(config['control'])
    control.mkdir(parents=True, exist_ok=True)
    def log_event(result):
        with (control / 'events.jsonl').open('a', encoding='utf-8') as log:
            log.write(json.dumps(result, ensure_ascii=False) + '\n')
        if __import__('sys').stdout is not None:
            print(json.dumps(result, ensure_ascii=False), flush=True)
    # Kernel-held lock releases on process death; stale metadata cannot block recovery.
    lock = open(control / 'watch.lock', 'a+b')
    if os.name == 'nt':
        import msvcrt
        lock.seek(0)
        if not lock.read(1):
            lock.write(b'0'); lock.flush()
        lock.seek(0)
        msvcrt.locking(lock.fileno(), msvcrt.LK_NBLCK, 1)
    write_json(control / 'watcher.json', {'pid': os.getpid(), 'started_at': now(),
               'interval_seconds': config.get('interval_seconds', 60), 'stop_file': str(control / 'STOP')})
    while True:
        if (control / 'STOP').exists():
            write_json(control / 'watcher.json', {'pid': os.getpid(), 'status': 'stopped', 'stopped_at': now()})
            return
        try:
            result = cycle(config)
            log_event(result)
        except Exception as exc:
            result = {'status': 'error', 'checked_at': now(), 'pid': os.getpid(),
                      'error_type': type(exc).__name__, 'message': str(exc)[:250]}
            write_json(control / 'last_error.json', result)
            write_json(Path(config['handoff']) / '06_核验/后台同步状态.json', result)
            try:
                report_failure(config, result)
            except Exception:
                pass  # Notification failure must not prevent the next sync attempt.
            refresh_manifest(config['handoff'], config.get('manifest_overflow_directory'))
            log_event(result)
            if not args.watch:
                raise
        if not args.watch:
            return
        time.sleep(config.get('interval_seconds', 60))


if __name__ == '__main__':
    main()
