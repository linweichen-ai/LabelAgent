"""Five bounded, real-image requests on the owned Capture pilot; no label writes."""
import argparse
import base64
from datetime import datetime, timezone
import hashlib
import html
import io
import json
from pathlib import Path
import subprocess
import time

MODEL_REVISION = "89644892e4d85e24eaac8bacfd4f463576704203"
PROMPT = (
    "第一张是完整场景，第二张是同一候选的局部图，黄色轮廓标出待检查对象。"
    "仅复查这个候选，不要评价其他实例。pallet指独立货物托盘，person指人；"
    "叉车货叉、托盘车车体、反光及背景不能仅因外形相似就归为托盘。"
    "保留用keep，错误候选用reject，类别错误但轮廓可用用relabel，看不清用uncertain。"
    "如果图片完全空白，不得认定存在托盘或人员。"
    "只输出JSON：{\"decision\":\"keep|reject|relabel|uncertain\","
    "\"label\":\"pallet|person|other|unknown\"}。"
)


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def parse_answer(text):
    try:
        value = json.loads(text.strip())
    except (ValueError, TypeError):
        return None
    if not isinstance(value, dict) or set(value) != {"decision", "label"}:
        return None
    if value["decision"] not in {"keep", "reject", "relabel", "uncertain"}:
        return None
    allowed = {"keep": {"pallet", "person"}, "relabel": {"pallet", "person"},
               "reject": {"other"}, "uncertain": {"unknown"}}
    return value if value["label"] in allowed[value["decision"]] else None


def blank_passed(answer):
    return answer is not None and answer["decision"] in {"reject", "uncertain"}


def make_cases(root, Image):
    records = [json.loads(s) for s in (root / "review_queue.jsonl").read_text(encoding="utf-8-sig").splitlines() if s.strip()]
    ids = ["f015_camera_0_0", "f015_camera_0_1", "f076_camera_1_0"]
    cases = []
    for sid in ids:
        matched = [r for r in records if r["sample_id"] == sid]
        if len(matched) != 1:
            raise ValueError("Missing or duplicated pilot case: " + sid)
        r = matched[0]; images = []; hashes = []
        for name in r["images"]:
            p = (root / name).resolve()
            if not p.is_relative_to(root.resolve()) or digest(p) != r["image_sha256"][name]:
                raise ValueError("Input path/digest mismatch: " + sid)
            images.append(Image.open(p).convert("RGB")); hashes.append(digest(p))
        cases.append({"sample_id": sid, "kind": "natural_unconfirmed",
                      "proposed_label": r["proposed_label"], "images": images, "image_sha256": hashes})
    for color in ["white", "black"]:
        images = [Image.new("RGB", (768, 512), color=color), Image.new("RGB", (512, 512), color=color)]
        cases.append({"sample_id": "blank_" + color, "kind": "blank_control",
                      "proposed_label": "pallet", "images": images,
                      "image_sha256": [hashlib.sha256(im.tobytes()).hexdigest() for im in images]})
    return cases


def gpu_candidates(text, minimum_free_mib=12288, required_index=None):
    eligible = []
    for line in text.splitlines():
        if not line.strip():
            continue
        index, utilization, free = [x.strip() for x in line.split(",")]
        if int(utilization) <= 5 and int(free) >= minimum_free_mib and (required_index is None or int(index) == required_index):
            eligible.append((int(free), int(index)))
    if not eligible:
        raise ValueError("No idle GPU with >=12GiB free; leave existing jobs running and try later")
    return max(eligible)[1]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True); p.add_argument("--data-root", required=True)
    p.add_argument("--out", required=True); p.add_argument("--gpu", type=int, default=0)
    a = p.parse_args()
    out = Path(a.out)
    if out.exists():
        raise ValueError("Choose a fresh output directory")
    model_dir, root = Path(a.model), Path(a.data_root).resolve()
    if not (model_dir / "config.json").is_file():
        raise ValueError("Model download incomplete: config.json missing")
    import os
    if os.environ.get("CUDA_VISIBLE_DEVICES") not in {None, "", "0,1"}:
        raise ValueError("Unexpected GPU visibility mapping; inspect it before choosing a device")
    gpu_text = subprocess.check_output(["nvidia-smi", "--query-gpu=index,utilization.gpu,memory.free", "--format=csv,noheader,nounits"], text=True)
    device_index = gpu_candidates(gpu_text, required_index=a.gpu)
    import torch
    from PIL import Image
    from transformers import AutoProcessor, Qwen3VLForConditionalGeneration
    assert torch.cuda.is_available()
    device = f"cuda:{device_index}"
    queue_sha = digest(root / "review_queue.jsonl")
    cases = make_cases(root, Image)
    out.mkdir(parents=True)
    print("Loading local Qwen3-VL-2B-Instruct BF16/SDPA on " + device, flush=True)
    started = time.monotonic()
    processor = AutoProcessor.from_pretrained(model_dir, local_files_only=True, min_pixels=12544, max_pixels=451584)
    # Also set processor size explicitly, matching the pinned training data processor.
    ip = processor.image_processor
    if hasattr(ip, "min_pixels"):
        ip.min_pixels, ip.max_pixels = 12544, 451584
    if hasattr(ip, "size") and isinstance(ip.size, dict):
        ip.size["shortest_edge"], ip.size["longest_edge"] = 12544, 451584
    model = Qwen3VLForConditionalGeneration.from_pretrained(
        model_dir, dtype=torch.bfloat16, attn_implementation="sdpa", local_files_only=True).to(device).eval()
    model_seconds = time.monotonic() - started
    attention = {"model": model.config._attn_implementation,
                 "text": model.config.text_config._attn_implementation,
                 "vision": model.config.vision_config._attn_implementation}
    if any(v != "sdpa" for v in attention.values()):
        raise ValueError("Unexpected attention implementation: " + repr(attention))
    torch.cuda.reset_peak_memory_stats(device_index)
    results = []; cards = []
    with (out / "responses.jsonl").open("x", encoding="utf-8") as f:
        for c in cases:
            question = PROMPT + "本次候选原类别：" + c["proposed_label"] + "。"
            messages = [{"role": "user", "content": [{"type": "image", "image": im} for im in c["images"]] + [{"type": "text", "text": question}]}]
            inputs = processor.apply_chat_template(messages, tokenize=True, add_generation_prompt=True,
                                                    return_dict=True, return_tensors="pt")
            inputs.pop("token_type_ids", None)
            inputs = inputs.to(device)
            torch.cuda.synchronize(device_index); started = time.monotonic()
            with torch.inference_mode():
                output = model.generate(**inputs, max_new_tokens=192, do_sample=False)
            torch.cuda.synchronize(device_index)
            text = processor.batch_decode(output[:, inputs["input_ids"].shape[-1]:], skip_special_tokens=True)[0]
            answer = parse_answer(text)
            semantic_valid = answer is not None and not (
                answer["decision"] == "relabel" and answer["label"] == c["proposed_label"] or
                answer["decision"] == "keep" and answer["label"] != c["proposed_label"])
            row = {k: v for k, v in c.items() if k != "images"}
            row.update({"text": text, "parsed": answer, "semantic_valid": semantic_valid,
                        "seconds": time.monotonic() - started, "image_grid_thw": inputs["image_grid_thw"].tolist(),
                        "input_tokens": inputs["input_ids"].shape[-1],
                        "output_tokens": output.shape[-1] - inputs["input_ids"].shape[-1],
                        "blank_not_hallucinated": blank_passed(answer) if c["kind"] == "blank_control" else None})
            results.append(row); f.write(json.dumps(row, ensure_ascii=False) + "\n"); f.flush()
            print(json.dumps(row, ensure_ascii=False), flush=True)
            tags = []
            for im in c["images"]:
                b = io.BytesIO(); im.save(b, format="JPEG", quality=88)
                tags.append('<img style="width:45%;max-height:400px;object-fit:contain" src="data:image/jpeg;base64,' + base64.b64encode(b.getvalue()).decode() + '">')
            cards.append("<article><h2>" + html.escape(c["sample_id"]) + "</h2>" + "".join(tags) + "<pre>" + html.escape(text) + "</pre></article>")
    assert digest(root / "review_queue.jsonl") == queue_sha, "Review records changed during run"
    summary = {"status": "five_requests_completed_NOT_accuracy_validation", "observed_utc": datetime.now(timezone.utc).isoformat(),
               "model": "Qwen3-VL-2B-Instruct", "expected_download_revision": MODEL_REVISION,
               "model_revision_cryptographically_verified_by_this_probe": False,
               "model_directory": str(model_dir), "model_config_sha256": digest(model_dir / "config.json"),
               "dtype": "bfloat16", "attention": attention, "gpu": device_index,
               "gpu_name": torch.cuda.get_device_name(device_index), "model_load_seconds": model_seconds,
               "cases": len(results), "strict_json": sum(x["parsed"] is not None for x in results),
               "semantic_valid": sum(x["semantic_valid"] for x in results),
               "blank_controls_passed": sum(x["blank_not_hallucinated"] is True for x in results), "blank_controls": 2,
               "peak_allocated_bytes": torch.cuda.max_memory_allocated(device_index), "source_queue_sha256": queue_sha,
               "human_truth_available_for_natural_cases": False, "pixel_accuracy_evaluated": False,
               "model_weights_trained": False, "source_annotation_writes": 0, "full_Agent_verified": False,
               "caveat": "Protocol/input diagnostic on three unconfirmed candidates and two blank controls; do not call these results segmentation accuracy"}
    (out / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    page = '<!doctype html><html lang="zh-CN"><meta charset="utf-8"><title>Qwen2B SDPA真实图片测试</title><style>body{font:16px/1.6 sans-serif;max-width:1200px;margin:auto;padding:24px}article{border:1px solid #ddd;padding:16px;margin:20px 0}pre{white-space:pre-wrap}</style><h1>Qwen2B SDPA真实图片测试</h1><p>仅为功能与候选复查诊断，无二维准确率或完整Agent结论。</p>' + ''.join(cards) + '<pre>' + html.escape(json.dumps(summary, ensure_ascii=False, indent=2)) + '</pre></html>'
    (out / "report.html").write_text(page, encoding="utf-8")
    print("RESULT_DIRECTORY=" + str(out.resolve()), flush=True)
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
