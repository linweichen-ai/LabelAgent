"""CPU-only functional demonstration of the unchanged SAM service serializer.

The predictor is an explicit synthetic stub. No model, checkpoint, GPU, HTTP
endpoint or real scene is used. This cannot establish real SAM accuracy.
"""
import copy
import hashlib
import io
import json
import sys
import types
from contextlib import nullcontext
from pathlib import Path

sys.dont_write_bytecode = True
ROOT = Path(__file__).resolve().parents[3]
WORK = Path(__file__).resolve().parent
OUT = ROOT / 'outputs/labelagent_raw_mask_audit_20261107/function_diagnostic'
SERVICE = Path('D:/Apps/LabelAgent/source/sam3_service/point_segment.py')
FROZEN_SERVICE = ROOT / 'task_workspaces/labelagent_point_ids_20261107/source_snapshot/source/sam3_service/point_segment.py'
HELPERS = ROOT / 'tools/labelagent_live_handoff'


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def save(path, value):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')


def main():
    import cv2
    import numpy as np
    from PIL import Image, ImageDraw, ImageFont, ImageOps

    if (OUT / 'completed.json').exists():
        raise FileExistsError('Functional example already complete; do not alter its evidence')
    OUT.mkdir(parents=True, exist_ok=True)
    source_bytes = SERVICE.read_bytes()
    source_sha = hashlib.sha256(source_bytes).hexdigest()
    frozen_sha = sha(FROZEN_SERVICE)
    (WORK / 'point_segment_source_copy.py').write_bytes(source_bytes)
    module = types.ModuleType('_synthetic_diagnostic_point_segment')
    module.__file__ = str(SERVICE)
    exec(compile(source_bytes, str(SERVICE), 'exec'), module.__dict__)

    # The unchanged function imports torch only for context managers here.
    # A context-only stub prevents a real torch/model/GPU path from being used.
    torch_stub = types.ModuleType('torch')
    torch_stub.cuda = types.SimpleNamespace(is_available=lambda: False)
    torch_stub.inference_mode = nullcontext
    torch_stub.autocast = lambda *args, **kwargs: nullcontext()
    torch_stub.bfloat16 = 'unused_cpu_stub'
    old_torch = sys.modules.get('torch')
    sys.modules['torch'] = torch_stub
    sys.path[:0] = [str(HELPERS), str(FROZEN_SERVICE.parents[2])]
    from wire_geometry_policy import map_complete_response
    from source.agent.edit_agent import _rasterize

    def png(array):
        output = io.BytesIO(); Image.fromarray(array).save(output, 'PNG'); return output.getvalue()

    height, width = 256, 384
    components = np.zeros((height, width), np.uint8)
    components[40:140, 30:130] = 1
    components[70:170, 220:330] = 1
    components[210:212, 170:172] = 1
    jagged = np.zeros((height, width), np.uint8)
    jagged[40:230, 20:360] = 1
    jagged[40, 21:359:2] = 0
    scenes = [
        ('disconnected_one_positive', '两个大断开部件＋小组件；仅左部一个正点', components, [{'x': 60, 'y': 90, 'label': 1}], False),
        ('jagged_boundary_one_positive', '单组件锯齿边界；正点远离边界', jagged, [{'x': 200, 'y': 100, 'label': 1}], False),
        ('disconnected_with_prior_control', '相同合成预测＋合成prior的分支功能对照', components, [{'x': 60, 'y': 90, 'label': 1}], True),
    ]
    rows = []
    fonts = {n: ImageFont.truetype('C:/Windows/Fonts/msyh.ttc', n) for n in (22, 27, 34)}

    def display(binary, color):
        rgb = np.full((height, width, 3), 225, dtype=np.uint8)
        rgb[binary != 0] = color
        return Image.fromarray(rgb)

    try:
        for name, title, raw, points, with_prior in scenes:
            folder = OUT / name; folder.mkdir()
            calls = []
            class FakeProcessor:
                def set_image(self, image):
                    return {'synthetic_image_size': image.size}
            class FakeModel:
                inst_interactive_predictor = types.SimpleNamespace(model=types.SimpleNamespace(
                    sam_prompt_encoder=types.SimpleNamespace(mask_input_size=(64, 64))))
                def predict_inst(self, state, **kwargs):
                    calls.append({'state': state, 'point_coords': kwargs['point_coords'].tolist(),
                                  'point_labels': kwargs['point_labels'].tolist(),
                                  'multimask_output': kwargs['multimask_output'],
                                  'mask_input_shape': list(kwargs['mask_input'].shape) if 'mask_input' in kwargs else None,
                                  'synthetic_predictor_no_model': True})
                    return raw[None].copy(), np.array([.9], dtype=np.float32), None
            module._predictor = lambda: (FakeModel(), FakeProcessor())
            captured = {}
            def observer(frame, event, arg):
                if frame.f_code is module.segment.__code__ and event == 'return':
                    captured.update(frame.f_locals)
            previous_profiler = sys.getprofile()
            source_image = np.full((height, width, 3), 127, np.uint8)
            source_png = png(source_image)
            try:
                sys.setprofile(observer)
                response = module.segment(source_png, points, prior_mask=png(raw * 255) if with_prior else None)
            finally:
                sys.setprofile(previous_profiler)
            if len(calls) != 1 or not np.array_equal(captured['mask'], raw):
                raise AssertionError('The injected synthetic prediction was not captured exactly')
            exact_parts = captured['extract_parts'](simplify=False)
            exact_filtered = captured['rasterize'](exact_parts)
            initially_simplified = captured['rasterize'](captured['extract_parts'](simplify=True))
            wire = _rasterize(response, width, height)
            if not np.array_equal(wire, captured['rendered']):
                raise AssertionError('Actual response raster differs from the function return raster')
            box = [24, 16, width + 24, height + 16]
            mapped, full_wire, wire_proof = map_complete_response(response, box, width + 48, height + 32)
            expected = np.zeros((height + 32, width + 48), np.uint8)
            expected[16:height + 16, 24:width + 24] = wire
            if not np.array_equal(full_wire, expected):
                raise AssertionError('Downstream mapping of actual wire geometry changed')

            removed = ((raw != 0) & (wire == 0)).astype('uint8')
            added = ((raw == 0) & (wire != 0)).astype('uint8')
            filter_removed = ((raw != 0) & (exact_filtered == 0)).astype('uint8')
            simplify_removed = ((exact_filtered != 0) & (wire == 0)).astype('uint8')
            simplify_added = ((exact_filtered == 0) & (wire != 0)).astype('uint8')
            point_agreement = all(bool(raw[p['y'], p['x']]) == bool(wire[p['y'], p['x']]) for p in points)
            row = {'case': name, 'title': title, 'synthetic_functional_example_not_model_result': True,
                   'input_size': [width, height], 'with_synthetic_prior': with_prior, 'points': points,
                   'fake_predictor_calls': len(calls), 'real_model_calls': 0,
                   'raw_pixels': int(raw.sum()), 'raw_connected_components': int(cv2.connectedComponents(raw, 8)[0] - 1),
                   'kept_outer_contours': len(captured['outer']), 'wire_parts': len(response['parts']),
                   'filter_only_exact_contour_pixels': int(exact_filtered.sum()),
                   'filter_only_removed_pixels': int(filter_removed.sum()),
                   'simplification_removed_pixels': int(simplify_removed.sum()),
                   'simplification_added_pixels': int(simplify_added.sum()),
                   'wire_pixels': int(wire.sum()), 'raw_to_wire_removed_pixels': int(removed.sum()),
                   'raw_to_wire_added_pixels': int(added.sum()), 'click_occupancy_preserved': point_agreement,
                   'wire_equals_initially_simplified': bool(np.array_equal(wire, initially_simplified)),
                   'warnings': response['warnings'], 'wire_mapping_proof': wire_proof,
                   'disk_service_sha256': source_sha, 'raw_binary_sha256': hashlib.sha256(raw.tobytes()).hexdigest(),
                   'wire_binary_sha256': hashlib.sha256(wire.tobytes()).hexdigest()}
            if name == 'disconnected_one_positive':
                assert raw[90, 250] and not wire[90, 250] and raw[210, 170] and not wire[210, 170]
                assert row['filter_only_removed_pixels'] == 11004 and row['raw_to_wire_added_pixels'] == 0
            elif name == 'jagged_boundary_one_positive':
                assert row['filter_only_removed_pixels'] == 0 and point_agreement
                assert row['simplification_added_pixels'] + row['simplification_removed_pixels'] > 0
            else:
                assert wire[90, 250] and not wire[210, 170]
                assert row['filter_only_removed_pixels'] == 4
            for label, array in [('raw', raw), ('filter_only_exact', exact_filtered), ('wire', wire),
                                 ('removed', removed), ('added', added)]:
                Image.fromarray(array * 255).save(folder / (label + '.png'))
            (folder / 'synthetic_input.png').write_bytes(source_png)
            save(folder / 'response.json', response); save(folder / 'exact_filtered_parts.json', exact_parts)
            save(folder / 'call_arguments.json', calls); save(folder / 'mapped_shape.json', mapped)
            save(folder / 'result.json', row)

            canvas = Image.new('RGB', (1920, 1080), '#142436'); draw = ImageDraw.Draw(canvas)
            draw.text((30, 24), '合成CPU功能示例（不是实际SAM推理）：' + title, font=fonts[34], fill='white')
            panels = [('注入的合成raw mask', display(raw, (50, 180, 220))),
                      ('服务选组件后，不简化轮廓', display(exact_filtered, (220, 185, 55))),
                      ('服务实际返回polygon再栅格化', display(wire, (240, 115, 65)))]
            for index, (label, image) in enumerate(panels):
                x = 30 + index * 630; draw.text((x, 110), label, font=fonts[22], fill='#a2e5cd')
                rendered = ImageOps.contain(image, (600, 400)); canvas.paste(rendered, (x, 160))
            difference = np.full((height, width, 3), 225, dtype=np.uint8)
            difference[removed != 0] = (235, 50, 70); difference[added != 0] = (60, 95, 245)
            diff_image = ImageOps.contain(Image.fromarray(difference), (600, 400)); canvas.paste(diff_image, (30, 620))
            draw.text((30, 570), '相对raw的差集：红=删去，蓝=新增', font=fonts[22], fill='#a2e5cd')
            lines = [f"raw {row['raw_pixels']}像素；wire {row['wire_pixels']}像素。",
                     f"组件筛选删去 {row['filter_only_removed_pixels']}像素。",
                     f"轮廓简化再删 {row['simplification_removed_pixels']} / 加 {row['simplification_added_pixels']}像素。",
                     f"点击点占用一致：{point_agreement}；警告数：{len(response['warnings'])}。",
                     '下游完整wire映射：0删去、0新增，仍不代表raw保全。',
                     '仅执行磁盘服务函数，predictor由显式合成CPU桩替代。',
                     '真实模型调用0；不证明历史真实请求发生相同数量丢失。']
            for index, line in enumerate(lines):
                draw.text((690, 630 + index * 48), line, font=fonts[27], fill='white')
            draw.text((30, 1035), '未修改服务、原图或封存结果；灰图和掩码均为明确合成的功能材料', font=fonts[22], fill='#b9cad8')
            canvas.save(folder / 'functional_comparison.jpg', quality=95)
            row['picture'] = (folder / 'functional_comparison.jpg').relative_to(OUT).as_posix()
            rows.append(row)
            print(json.dumps(row, ensure_ascii=False), flush=True)
    finally:
        if old_torch is None:
            sys.modules.pop('torch', None)
        else:
            sys.modules['torch'] = old_torch

    assert sha(SERVICE) == source_sha and sha(FROZEN_SERVICE) == frozen_sha
    assert all(r['wire_mapping_proof']['removed_pixels'] == r['wire_mapping_proof']['added_pixels'] == 0 for r in rows)
    result = {'status': 'complete_CPU_functional_not_model_inference', 'passed': True,
              'real_model_calls': 0, 'HTTP_requests': 0, 'GPU_used': False, 'cases': rows,
              'disk_service_sha256': source_sha, 'current_frozen_service_sha256': frozen_sha,
              'disk_and_current_frozen_service_equal': source_sha == frozen_sha,
              'disk_source_unchanged': True, 'live_runtime_code_identity_proven': False,
              'runtime_limit': 'The byte-identical disk function was executed in a fresh isolated module with a synthetic predictor. This does not prove what code a previously launched worker has loaded.',
              'live_service_changes': 0, 'annotation_writes': 0,
              'wire_mapping_policy_sha256': sha(HELPERS / 'wire_geometry_policy.py')}
    save(OUT / 'completed.json', result)
    text = f'''# SAM服务组件与轮廓序列化：合成CPU功能示例

这是功能材料，不是真实场景、真实SAM模型预测、标注效果或精度数据。直接执行磁盘 point_segment.py 的原 segment 函数，但将 predictor 明确替换为返回预先构造二值掩码的CPU桩。未加载模型/权重、未访问HTTP服务、未使用GPU、未改变服务文件或原封存结果。

磁盘服务源码 SHA256：{source_sha}。当前编号点实验冻结源码摘要相同：{source_sha == frozen_sha}。该对应只绑定这次CPU执行的磁盘函数；不能证明既有工作进程当前已加载的Python代码，真实历史模型tensor亦未由该示例取得。

第一例：两个大断开组件与一个小组件，仅左组件一个正点。raw={rows[0]['raw_pixels']}，实际wire={rows[0]['wire_pixels']}，删去{rows[0]['raw_to_wire_removed_pixels']}像素，无警告。无prior时，服务先过滤轮廓面积小于9者，再仅保留含正点的外部组件，因此另一个大组件也在HTTP返回之前消失。

第二例：单组件顶边有1像素锯齿，正点远离边界。组件筛选不减像素，但approxPolyDP导致删{rows[1]['simplification_removed_pixels']}、加{rows[1]['simplification_added_pixels']}。点击仍保持一致，故只检查点击处是否改变的回退条件不会发现所有非点击边界变化。

第三例：同一合成raw加合成prior仅验证另一代码分支，两个大组件保留，小组件仍被面积筛选去掉{rows[2]['filter_only_removed_pixels']}像素。它不是建议改变当前无prior实验，也不是加入先验后的真实效果对比。

三例下游 map_complete_response 都与已经返回的HTTP polygon栅格一致，0增删。但该证明的起点是wire，无法据此宣称服务内部原模型raw mask逐像素完整保留。这些功能差异是否在历史真实请求出现、实际损失多少，仍需要绑定真实内部tensor与同一请求的隔离捕获，不能从本合成示例推断。

复现：使用带numpy/Pillow/OpenCV的CPU解释器运行本目录来源脚本 run_function_diagnostic.py。输出目录已经完成时拒绝覆盖。call_arguments.json、response.json、raw/filter_only_exact/wire/removed/added.png和result.json保留每例来源。
'''
    (OUT / '功能示例说明.md').write_text(text, encoding='utf-8')
    file_rows = [{'path': p.relative_to(OUT).as_posix(), 'sha256': sha(p), 'bytes': p.stat().st_size}
                 for p in sorted(OUT.rglob('*')) if p.is_file()]
    save(OUT / 'files_sha256.json', {'files': file_rows})


if __name__ == '__main__':
    main()
