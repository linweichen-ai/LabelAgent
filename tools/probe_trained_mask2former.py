"""CPU-only local Mask2Former preview; never writes annotation truth or trains.

Requires torch, transformers, numpy and Pillow in a separately prepared environment.
Example: python tools/probe_trained_mask2former.py --checkpoint <local-model>
    --image <local-image> --output-dir <new-private-directory>
"""
import argparse
import hashlib
import io
import json
import os
from pathlib import Path
import time

os.environ['CUDA_VISIBLE_DEVICES'] = ''
os.environ['HF_HUB_OFFLINE'] = '1'


def assign_panoptic(outputs, target_size, stuff_labels):
    """Class score>.5, absolute mask>=.5, overlap>=.8; unknown remains unassigned."""
    import torch
    import torch.nn.functional as F
    classes = outputs.class_queries_logits[0]
    scores, labels = classes.softmax(-1).max(-1)
    keep = (labels != classes.shape[-1] - 1) & (scores > .5)
    scores, labels = scores[keep], labels[keep]
    ids = torch.zeros(target_size, dtype=torch.int32)
    if not len(scores):
        return ids, []
    masks = F.interpolate(outputs.masks_queries_logits[:, keep], size=target_size,
                          mode='bilinear', align_corners=False)[0].sigmoid()
    winners = (masks * scores[:, None, None]).argmax(0)
    fused, metadata, counter = {}, [], 0
    for index, (score, label) in enumerate(zip(scores, labels)):
        label = int(label)
        assigned = winners == index
        original = masks[index] >= .5
        selected = assigned & original
        area = int(original.sum())
        if not area or not selected.any() or int(assigned.sum()) / area < .8:
            continue
        if label in stuff_labels and label in fused:
            ids[selected] = fused[label]
            continue
        counter += 1
        ids[selected] = counter
        metadata.append({'id': counter, 'label_id': label, 'score': float(score)})
        if label in stuff_labels:
            fused[label] = counter
    return ids, metadata


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--image', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--input-height', type=int, default=384)
    parser.add_argument('--input-width', type=int, default=512)
    args = parser.parse_args()
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        raise FileExistsError('Use a new private output directory')
    if min(args.input_height, args.input_width) < 32:
        raise ValueError('Input dimensions must be at least 32')
    import numpy as np
    import torch
    from PIL import Image
    from transformers import Mask2FormerForUniversalSegmentation, Mask2FormerImageProcessor
    torch.set_num_threads(2)
    torch.set_num_interop_threads(1)
    source = args.image.read_bytes()
    image = Image.open(io.BytesIO(source)).convert('RGB')
    weight = args.checkpoint / 'model.safetensors'
    weight_sha = hashlib.sha256(weight.read_bytes()).hexdigest()
    started = time.monotonic()
    model = Mask2FormerForUniversalSegmentation.from_pretrained(args.checkpoint, local_files_only=True).cpu().eval()
    processor = Mask2FormerImageProcessor.from_pretrained(args.checkpoint, local_files_only=True)
    names = {int(k): v for k, v in model.config.id2label.items()}
    stuff = {k for k, v in names.items() if v in ('floor', 'slope', 'cargo_floor')}
    inputs = processor(images=image.resize((args.input_width, args.input_height), Image.Resampling.BILINEAR),
                       do_resize=False, return_tensors='pt')
    with torch.inference_mode():
        ids, segments = assign_panoptic(model(**inputs), (image.height, image.width), stuff)
    ids = ids.numpy().astype(np.uint32)
    assert hashlib.sha256(weight.read_bytes()).hexdigest() == weight_sha
    args.output_dir.mkdir(parents=True, exist_ok=True)
    rgb = np.stack((ids % 256, ids // 256 % 256, ids // 65536 % 256), axis=-1).astype(np.uint8)
    Image.fromarray(rgb).save(args.output_dir / 'panoptic.png')
    original = np.asarray(image)
    rendered = original.copy()
    palette = {'pallet': (235, 85, 60), 'person': (255, 165, 0), 'floor': (40, 200, 190),
               'slope': (240, 220, 35), 'ego': (230, 85, 200), 'other_robot': (135, 90, 230),
               'cargo_floor': (45, 120, 230)}
    for segment in segments:
        region = ids == segment['id']
        name = names[segment['label_id']]
        segment.update(label=name, predicted_pixels=int(region.sum()))
        rendered[region] = (original[region] * .48 + np.array(palette.get(name, (100, 100, 220))) * .52).astype(np.uint8)
    Image.fromarray(rendered).save(args.output_dir / 'preview.jpg', quality=94)
    receipt = {'status': 'EXPERIMENTAL_PREDICTIONS_NOT_CONFIRMED_TRUTH', 'device': 'cpu',
               'real_model_forwards': 1, 'input_size': [args.input_height, args.input_width],
               'output_size': [image.height, image.width], 'image_sha256': hashlib.sha256(source).hexdigest(),
               'checkpoint_sha256': weight_sha, 'class_mapping': names, 'segments': segments,
               'unassigned_pixels': int((ids == 0).sum()), 'weights_unchanged': True,
               'annotation_writes': 0, 'training_updates': 0, 'pixel_accuracy_evaluated': False,
               'elapsed_seconds': round(time.monotonic() - started, 3)}
    (args.output_dir / 'prediction.json').write_text(json.dumps(receipt, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps({k: receipt[k] for k in ('status', 'real_model_forwards', 'elapsed_seconds')}))


if __name__ == '__main__':
    main()
