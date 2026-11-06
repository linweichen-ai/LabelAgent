"""Copy and verify a small local SAM raw/wire audit supplement.

Requires completed independent, CPU capture-test and browser receipts. Does not
run models, access remote repositories, change services/watchers or rewrite an
older sealed package. All files are normal independent copies, never hardlinks.
"""
import datetime
import hashlib
import html
import json
import os
import shutil
from html.parser import HTMLParser
from pathlib import Path
from urllib.parse import unquote

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / 'outputs/labelagent_raw_mask_audit_20261107'
WORK = ROOT / 'task_workspaces/labelagent_raw_mask_audit_20261107'
CONTROL = ROOT / 'task_workspaces/labelagent_handoff_control_20261104'
HOME = ROOT / 'deliverables/LabelAgent_实时交接/09_补充成果/SAM内部掩码边界审计_20261107'
HELPERS = Path(__file__).parent


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for data in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(data)
    return digest.hexdigest()


def save(path, record):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(record, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')


def passed(path, accept_status=False):
    record = read(path)
    if record.get('passed') is not True and not (accept_status and record.get('status') == 'passed'):
        raise ValueError('Required completed verification has not passed: ' + str(path))
    return record


def allowed_files(root):
    for path in sorted(Path(root).rglob('*')):
        relative = path.relative_to(root)
        if any(part in {'__pycache__', '.git'} for part in relative.parts):
            continue
        if path.is_file() and path.suffix not in {'.pyc', '.pyo'} and '.pending' not in path.name:
            yield path


class MediaLinks(HTMLParser):
    def __init__(self):
        super().__init__(); self.images = []; self.videos = []
    def handle_starttag(self, tag, attributes):
        attrs = dict(attributes)
        if tag == 'img' and attrs.get('src'):
            self.images.append(attrs['src'])
        if tag == 'video' and attrs.get('src'):
            self.videos.append(attrs['src'])


def capture_source_bound(report, script):
    binding = report.get('source_sha256')
    expected = sha(script)
    if isinstance(binding, str):
        return binding.lower() == expected
    if isinstance(binding, dict):
        for name, value in binding.items():
            if Path(str(name)).name == script.name:
                digest = value.get('sha256') if isinstance(value, dict) else value
                if isinstance(digest, str) and digest.lower() == expected:
                    return True
    return False


def main():
    if (HOME / '交接SHA256.json').exists():
        raise FileExistsError('This supplement is already sealed; preserve it')
    function = passed(OUT / 'function_diagnostic/completed.json')
    media = passed(OUT / 'function_diagnostic/replay/media_verification.json')
    independent = passed(OUT / 'independent_review.json')
    capture = passed(OUT / 'capture_tests.json', accept_status=True)
    ui = passed(CONTROL / 'raw_mask_audit_ui_verification.json')
    if function.get('status') != 'complete_CPU_functional_not_model_inference':
        raise ValueError('This package is only for the declared CPU functional audit')
    if function.get('real_model_calls') != 0 or function.get('HTTP_requests') != 0 or function.get('GPU_used') is not False:
        raise ValueError('Do not mislabel model/HTTP/GPU activity as this CPU-only functional example')
    if media.get('functional_source_files_unchanged') is not True:
        raise ValueError('Functional sources were not preserved during media rendering')
    if capture.get('cpu_synthetic_only') is not True or capture.get('real_model_requests') != 0:
        raise ValueError('Capture tests must explicitly be synthetic CPU tests, not model inference')
    capture_script = HELPERS / 'sam_raw_capture.py'
    test_script = HELPERS / 'test_sam_raw_capture.py'
    probe_script = HELPERS / 'probe_raw_mask_boundary.py'
    delivery_scripts = tuple(HELPERS / name for name in (
        'publish_raw_mask_audit.py', 'commit_raw_mask_audit.py', 'record_raw_mask_delivery.py'))
    for path in (capture_script, test_script, probe_script, *delivery_scripts):
        if not path.is_file():
            raise FileNotFoundError('Required completed audit tool is missing: ' + str(path))
    if not capture_source_bound(capture, capture_script):
        raise ValueError('Capture tests are not bound to the current capture source hash')
    for path, digest in media.get('source_sha256', {}).items():
        if sha(path) != digest:
            raise ValueError('A functional source changed after media verification: ' + path)

    replay = OUT / 'function_diagnostic/replay'
    if ui.get('root') and Path(ui['root']).resolve() != replay.resolve():
        raise ValueError('Browser verification is for another artifact')
    parser = MediaLinks(); parser.feed((replay / 'index.html').read_text(encoding='utf-8'))
    image_files = [(replay / unquote(src)).resolve() for src in parser.images]
    video_files = [(replay / unquote(src)).resolve() for src in parser.videos]
    if any(not path.is_relative_to(OUT.resolve()) or not path.is_file() for path in image_files + video_files):
        raise ValueError('The actual gallery links escape this audit or are missing')
    if len(image_files) != media['pictures'] or len(video_files) != media['videos']:
        raise ValueError('Gallery counts differ from saved media verification')
    if ui.get('images') != len(image_files) or len(ui.get('videos', [])) != len(video_files):
        raise ValueError('Browser media counts differ from the gallery')
    if len(video_files) != 1 or sha(video_files[0]) != media['video_sha256']:
        raise ValueError('The single verified functional replay changed')
    if len(image_files) != 4 or media['fully_decoded_frames'] != 240:
        raise ValueError('Unexpected functional gallery/replay scope; inspect before sealing')

    next_review = ROOT / 'task_workspaces/labelagent_point_ids_20261107/audit/next_layer_review.json'
    next_record = read(next_review)
    if next_record.get('status') != 'complete_read_only_preflight_review':
        raise ValueError('The attached next-layer review is incomplete')
    required = [OUT / '复现路径与术语补记.md', OUT / 'runtime_boundary_probe.json',
                WORK / 'function_diagnostic/run_function_diagnostic.py',
                WORK / 'function_diagnostic/render_function_replay.py',
                WORK / 'runtime_audit/source_boundary.json']
    for path in required:
        if not path.is_file():
            raise FileNotFoundError('Required local evidence missing: ' + str(path))

    # Snapshot all source names and bytes before copying; never copy a growing
    # temporary file. A later source-name/byte change prevents a final seal.
    source_pairs = []
    roots = [(OUT, HOME / '01_功能证据与全部原始返回'),
             (WORK, HOME / '02_复现代码与运行边界材料')]
    inventories = {str(source): [str(p) for p in allowed_files(source)] for source, _ in roots}
    for source, target in roots:
        source_pairs.extend((path, target / path.relative_to(source)) for path in allowed_files(source))
    source_pairs += [(CONTROL / 'raw_mask_audit_ui_verification.json', HOME / '03_核验/raw_mask_audit_ui_verification.json'),
                     (next_review, HOME / '03_核验/next_layer_review.json')]
    source_pairs += [(path, HOME / '04_隔离捕获与核验工具' / path.name)
                     for path in (capture_script, test_script, probe_script, *delivery_scripts, Path(__file__))]
    before = {str(source.resolve()): sha(source) for source, _ in source_pairs}
    HOME.mkdir(parents=True, exist_ok=True); copied = []
    for source, target in source_pairs:
        if not target.resolve().is_relative_to(HOME.resolve()):
            raise ValueError('Destination escaped the named supplement')
        target.parent.mkdir(parents=True, exist_ok=True)
        if target.exists():
            if sha(target) != before[str(source.resolve())] or os.path.samefile(source, target):
                raise ValueError('A partial package differs or is not an independent copy')
        else:
            shutil.copy2(source, target)
        expected = before[str(source.resolve())]
        if sha(target) != expected or os.path.samefile(source, target):
            raise ValueError('Independent source/destination SHA readback failed')
        copied.append({'source': str(source), 'path': target.relative_to(HOME).as_posix(),
                       'sha256': expected, 'bytes': target.stat().st_size, 'independent_copy': True})
    if any([str(p) for p in allowed_files(Path(source))] != names for source, names in inventories.items()):
        raise ValueError('Source inventory changed while packaging; preserve partial results without sealing')
    changed = [path for path, digest in before.items() if sha(path) != digest]
    if changed:
        raise ValueError('Source bytes changed while packaging: ' + str(changed[:3]))

    png_count = sum(p.suffix.lower() == '.png' for p in allowed_files(OUT))
    low_level_mask_png = sum(p.suffix.lower() == '.png' and p.stem in {'raw', 'filter_only_exact', 'wire', 'removed', 'added'}
                            for p in allowed_files(OUT))
    tests_run = capture.get('tests_run', capture.get('tests'))
    if not isinstance(tests_run, int) or tests_run < 1:
        raise ValueError('Capture test receipt does not record the number of real CPU tests')
    save(HOME / '03_核验/来源与独立复制核验.json', {'passed': True, 'sources_before_after_unchanged': True,
                                                'all_files_are_independent_copies': True, 'files': copied})
    rows = function['cases']
    results = '\n'.join(f"{row['title']}：raw {row['raw_pixels']} → wire {row['wire_pixels']}；删 {row['raw_to_wire_removed_pixels']}、加 {row['raw_to_wire_added_pixels']} 像素。" for row in rows)
    notes = f'''SAM内部掩码边界审计：本地小补记

打开 01_功能证据与全部原始返回/function_diagnostic/replay/index.html：图库 4 张图、1 段 24 秒 1920×1080 回放，240 帧全部解码。图库数量不同于底层 PNG 数：本补记证据含 {png_count} 个 PNG，其中 {low_level_mask_png} 个是 raw/filtered/wire/差集等合成mask位图。

{results}

三例仅执行原服务函数，predictor替换为明确的合成CPU桩。真实模型调用0、HTTP请求0、GPU使用0；这是功能示例，不是实景效果数据，也不是过去真实请求的丢失像素统计。三例下游wire映射均0增删仍不能证明内部raw完整。

服务磁盘源码与冻结副本可按摘要绑定，但这不证明既有进程加载代码的内存身份。运行边界探测、捕获工具及{tests_run}项CPU测试均保留；捕获工具通过功能测试不等于已捕获真实历史模型tensor。具体状态按runtime_boundary_probe、source_boundary和capture_tests收据读取。

01保存完整功能材料、原始返回、独立审核、捕获测试和复现术语补记；02保存本次工作目录及源码副本；03保存浏览器核验和下一层只读审查；04包含sam_raw_capture、测试、边界探测、本打包脚本，以及publish_raw_mask_audit、commit_raw_mask_audit、record_raw_mask_delivery三份交付工具的普通副本。本打包过程不会执行这些发布、提交或同步工具。复现脚本实际位于02/function_diagnostic/run_function_diagnostic.py，原路径与重绑条件见01/复现路径与术语补记.md。

所有文件独立复制并逐项SHA回读，来源前后不变；没有修改原封存包、根交接入口/清单、同步服务或远端。最小补记未重复复制实景数据/模型，不宣称包含完整运行环境。next_layer_review中的预审意见不是新实验或新授权边界，不代表实例已恢复。
'''
    (HOME / '交接说明.txt').write_text(notes, encoding='utf-8')
    entry = ('<!doctype html><html lang="zh"><meta charset="utf-8"><title>SAM内部掩码边界审计</title>'
             '<style>body{max-width:1150px;margin:30px auto;padding:24px;font:18px/1.7 "Microsoft YaHei",sans-serif}pre{white-space:pre-wrap}</style>'
             '<h1>SAM内部掩码边界审计</h1><p><a href="01_功能证据与全部原始返回/function_diagnostic/replay/index.html">4张图与24秒功能回放</a></p>'
             '<p><a href="01_功能证据与全部原始返回/independent_review.json">独立审核</a> · '
             '<a href="01_功能证据与全部原始返回/capture_tests.json">捕获功能测试</a> · '
             '<a href="01_功能证据与全部原始返回/复现路径与术语补记.md">复现路径与术语</a></p><pre>'
             + html.escape(notes) + '</pre></html>')
    (HOME / '00_先读.html').write_text(entry, encoding='utf-8')
    manifest = [{'path': p.relative_to(HOME).as_posix(), 'sha256': sha(p), 'bytes': p.stat().st_size}
                for p in allowed_files(HOME) if p.name != '交接SHA256.json']
    save(HOME / '交接SHA256.json', {'files': manifest})
    mismatches = [row['path'] for row in manifest if sha(HOME / row['path']) != row['sha256']]
    if mismatches:
        raise ValueError('Final package readback differs')
    receipt = {'passed': True, 'checked_at': datetime.datetime.now().astimezone().isoformat(),
               'bundle': str(HOME), 'files': len(manifest), 'mismatches': mismatches,
               'gallery_pictures': len(image_files), 'videos': len(video_files),
               'fully_decoded_frames': media['fully_decoded_frames'], 'video_seconds': media['video_seconds'],
               'underlying_png_files': png_count, 'low_level_mask_png_files': low_level_mask_png,
               'capture_tests': tests_run, 'capture_source_sha256': sha(capture_script),
               'independent_review_passed': independent['passed'], 'browser_verification_passed': ui['passed'],
               'all_files_are_independent_copies': True, 'sources_before_after_unchanged': True,
               'real_model_calls': 0, 'original_annotations_written': 0,
               'runtime_loaded_code_proven_by_this_package': False,
               'root_manifest_remote_watcher_changes': False}
    save(CONTROL / 'raw_mask_audit_bundle_verification.json', receipt)
    print(json.dumps(receipt, ensure_ascii=False), flush=True)


if __name__ == '__main__':main()
