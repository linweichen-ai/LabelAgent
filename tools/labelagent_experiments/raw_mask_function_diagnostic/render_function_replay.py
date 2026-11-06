"""A small replay of saved CPU functional evidence, not a model recording."""
import hashlib
import html
import json
from fractions import Fraction
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
OUT = ROOT / 'outputs/labelagent_raw_mask_audit_20261107/function_diagnostic'


def main():
    import av
    import numpy as np
    from PIL import Image, ImageDraw, ImageFont, ImageOps
    target = OUT / 'replay'
    if (target / 'media_verification.json').exists():
        raise FileExistsError('Functional replay already complete')
    target.mkdir(exist_ok=True)
    source_files = [p for p in OUT.rglob('*') if p.is_file() and target not in p.parents]
    before = {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in source_files}
    result = json.loads((OUT / 'completed.json').read_text(encoding='utf-8'))
    assert result['passed'] and result['real_model_calls'] == 0
    font = ImageFont.truetype('C:/Windows/Fonts/msyh.ttc', 26)
    titlefont = ImageFont.truetype('C:/Windows/Fonts/msyh.ttc', 34)
    small = ImageFont.truetype('C:/Windows/Fonts/msyh.ttc', 21)
    card = Image.new('RGB', (1920, 1080), '#142436'); draw = ImageDraw.Draw(card)
    draw.text((30, 24), '合成CPU功能示例：正点所在部件与非点击边界细节', font=titlefont, fill='white')
    raw = np.array(Image.open(OUT / 'disconnected_one_positive/raw.png')) > 0
    rgb = np.full((*raw.shape, 3), 230, np.uint8); rgb[raw] = (50, 180, 220)
    panel = Image.fromarray(rgb); pd = ImageDraw.Draw(panel)
    pd.ellipse((56, 86, 64, 94), fill=(0, 230, 40), outline='black', width=1)
    panel = panel.resize((768, 512), Image.Resampling.NEAREST)
    card.paste(panel, (30, 150))
    draw.text((30, 104), '合成raw：绿色为唯一正点，其他组件无正点', font=font, fill='#a2e5cd')
    draw.text((30, 690), '右侧大部件和底部小块在wire返回之前消失。', font=font, fill='white')
    draw.text((30, 735), '不是按预算裁小；是原服务的组件选择分支。', font=font, fill='white')
    for index, name in enumerate(('raw', 'wire')):
        mask = np.array(Image.open(OUT / f'jagged_boundary_one_positive/{name}.png')) > 0
        crop = mask[35:47, 30:80]
        rgb = np.full((*crop.shape, 3), 230, np.uint8); rgb[crop] = (50, 180, 220) if name == 'raw' else (240, 115, 65)
        image = Image.fromarray(rgb).resize((900, 216), Image.Resampling.NEAREST)
        y = 165 + index * 325; card.paste(image, (950, y))
        draw.text((950, y - 42), ('原始raw顶边' if name == 'raw' else '实际wire顶边') + '：相同范围最近邻放大', font=font, fill='#a2e5cd')
    for index, line in enumerate(['锯齿示例的169个新增像素远离唯一正点。',
                                  '点击一致性检查通过，未触发回退；无warnings。',
                                  '本图只放大已保存的合成位图，不是新预测。']):
        draw.text((950, 760 + index * 44), line, font=font, fill='white')
    draw.text((30, 1008), '功能示例，不是真实场景或SAM效果数据；真实模型/HTTP/GPU调用均为0。', font=font, fill='#dceaf6')
    zoom = target / '合成正点与边缘放大.jpg'; card.save(zoom, quality=95)
    pictures = [OUT / row['picture'] for row in result['cases']] + [zoom]
    video_file = target / '合成CPU功能证据回放.mp4'
    container = av.open(str(video_file), 'w', format='mp4', options={'movflags': '+faststart'})
    stream = container.add_stream('libx264', rate=10); stream.width = 1920; stream.height = 1080
    stream.pix_fmt = 'yuv420p'; stream.codec_context.thread_count = 2
    stream.options = {'crf': '22', 'preset': 'veryfast'}
    count = 0
    for picture in pictures:
        image = ImageOps.contain(Image.open(picture).convert('RGB'), (1860, 1000))
        canvas = Image.new('RGB', (1920, 1080), '#142436')
        canvas.paste(image, ((1920 - image.width) // 2, 4))
        ImageDraw.Draw(canvas).text((30, 1035), '实际CPU保存证据回放 · 合成predictor · 不是连续录屏或真实模型推理', font=small, fill='white')
        frame = av.VideoFrame.from_image(canvas)
        for _ in range(60):
            frame.pts = count; frame.time_base = Fraction(1, 10); frame.pict_type = av.video.frame.PictureType.NONE
            for packet in stream.encode(frame): container.mux(packet)
            count += 1
    for packet in stream.encode(): container.mux(packet)
    container.close()
    with av.open(str(video_file)) as video:
        decoded = 0
        for frame in video.decode(video=0):
            assert (frame.width, frame.height) == (1920, 1080); decoded += 1
    assert decoded == count == 240
    assert all(hashlib.sha256(Path(path).read_bytes()).hexdigest() == digest for path, digest in before.items())
    page = ['<!doctype html><html lang="zh"><meta charset="utf-8"><title>合成CPU功能证据回放</title>',
            '<style>body{max-width:1400px;margin:auto;padding:24px;font:18px/1.7 "Microsoft YaHei",sans-serif}img,video{width:100%;height:auto}pre{white-space:pre-wrap}</style>',
            '<h1>合成CPU功能证据回放</h1><p>这是替换predictor后的真实CPU功能执行，不是SAM模型推理、实景或效果数据。没有修改服务、原标注或旧交接包。</p>',
            '<p>服务返回前已可能删除组件或简化边界；下游wire映射0增删不能证明内部raw完整。不能把本例数量外推为历史真实请求的损失。</p>',
            '<p><a href="../功能示例说明.md">方法与限制</a> · <a href="../completed.json">三例完整结果</a></p>',
            '<video controls preload="metadata" src="合成CPU功能证据回放.mp4"></video>']
    for picture in pictures:
        relative = picture.relative_to(OUT).as_posix()
        link = '../' + relative
        page.append('<p>' + html.escape(picture.parent.name + '/' + picture.name) + '</p><a href="' + link + '"><img src="' + link + '"></a>')
    page.append('</html>'); (target / 'index.html').write_text('\n'.join(page), encoding='utf-8')
    receipt = {'passed': True, 'pictures': 4, 'new_detail_pictures': 1, 'videos': 1,
               'fully_decoded_frames': count, 'video_seconds': count / 10, 'width': 1920, 'height': 1080,
               'video_sha256': hashlib.sha256(video_file.read_bytes()).hexdigest(),
               'functional_source_files_unchanged': True, 'source_sha256': before,
               'real_model_calls': 0, 'screen_recording': False, 'synthetic_functional_replay': True}
    (target / 'media_verification.json').write_text(json.dumps(receipt, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    print(json.dumps({k: v for k, v in receipt.items() if k != 'source_sha256'}, ensure_ascii=False), flush=True)


if __name__ == '__main__':main()
