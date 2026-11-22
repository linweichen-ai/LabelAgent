#!/usr/bin/env python3
"""Build a local, candidate-only pallet boundary comparison pack via GETs.

The live workbench remains read-only.  Current candidates and source images are
read through public GET endpoints, while the replacement is produced locally
by the existing deterministic pallet geometry cleanup function.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import io
import json
import shutil
import sys
import tempfile
from pathlib import Path
from urllib.parse import urlencode, urlsplit
from urllib.request import Request, urlopen

import cv2
import numpy as np
from PIL import Image, ImageDraw

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from source.agent.candidate_class_audit import candidate_fingerprint
from source.agent.candidate_comparison import read_packet
from source.agent.edit_agent import _rasterize
from source.agent.workflow import refine_pallet_candidate_geometry


SCHEMA_VERSION = "labelagent.candidate-comparison-pack.v1"


def _canonical_bytes(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True,
                      separators=(",", ":"), allow_nan=False).encode("utf-8")


def _sha256(payload):
    return hashlib.sha256(payload).hexdigest()


def _safe_base_url(base_url):
    value = str(base_url).rstrip("/")
    parsed = urlsplit(value)
    if (parsed.scheme not in {"http", "https"} or not parsed.hostname
            or parsed.username is not None or parsed.password is not None
            or parsed.query or parsed.fragment):
        raise ValueError("base_url 必须是无凭证、无查询参数的 HTTP(S) 地址")
    return value


def _read_response(opener, url):
    request = Request(url, method="GET")
    with opener(request, timeout=20) as response:
        return response.read()


def _get_json(opener, base_url, path, query=None):
    url = f"{base_url}{path}"
    if query:
        url += "?" + urlencode(query)
    try:
        value = json.loads(_read_response(opener, url).decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError(f"GET {path} 未返回有效 JSON") from error
    if not isinstance(value, dict):
        raise ValueError(f"GET {path} 未返回 JSON 对象")
    return value


def _get_frame(opener, base_url, frame, scope):
    url = f"{base_url}/api/frame/{frame}?{urlencode(scope)}"
    payload = _read_response(opener, url)
    try:
        with Image.open(io.BytesIO(payload)) as decoded:
            image = np.asarray(decoded.convert("RGB"))
    except OSError as error:
        raise ValueError(f"frame {frame} 图像无法解码") from error
    if image.ndim != 3 or image.shape[2] != 3:
        raise ValueError(f"frame {frame} 图像通道无效")
    return payload, image


def _frame_values(mapping, frame):
    if not isinstance(mapping, dict):
        return []
    value = mapping.get(str(frame), mapping.get(frame, []))
    return value if isinstance(value, list) else []


def _png_bytes(image):
    target = io.BytesIO()
    Image.fromarray(image).save(target, format="PNG", optimize=False,
                                compress_level=9)
    return target.getvalue()


def _mask_png(mask):
    return _png_bytes((mask.astype(np.uint8) * 255))


def _preview_png(image_rgb, mask, title, color):
    output = image_rgb.copy()
    selected = mask.astype(bool)
    tint = np.zeros_like(output)
    tint[...] = np.asarray(color, dtype=np.uint8)
    output[selected] = np.clip(
        output[selected].astype(np.float32) * .58
        + tint[selected].astype(np.float32) * .42, 0, 255).astype(np.uint8)
    contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_CCOMP,
                                   cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(output, contours, -1, tuple(int(v) for v in color), 1)
    result = Image.fromarray(output)
    draw = ImageDraw.Draw(result)
    draw.rectangle((0, 0, min(result.width, 520), 30), fill=(10, 18, 26))
    draw.text((8, 8), title, fill=(255, 255, 255))
    target = io.BytesIO()
    result.save(target, format="PNG", optimize=False, compress_level=9)
    return target.getvalue()


def _comparison_roi(before, after, padding=24):
    selected = (before != 0) | (after != 0)
    ys, xs = np.where(selected)
    if not len(xs):
        raise ValueError("候选 mask 为空，无法生成对比预览")
    height, width = selected.shape
    return (max(0, int(xs.min()) - padding),
            max(0, int(ys.min()) - padding),
            min(width, int(xs.max()) + padding + 1),
            min(height, int(ys.max()) + padding + 1))


def _candidate_binding(candidates):
    return _sha256(_canonical_bytes(candidates))


def _validate_bootstrap(value):
    path = value.get("path")
    topic = value.get("topic")
    generation = value.get("file_activation_generation")
    catalog = value.get("catalog")
    if (value.get("loaded") is not True or not isinstance(path, str) or not path
            or not Path(path).is_absolute() or not isinstance(topic, str) or not topic
            or isinstance(generation, bool) or not isinstance(generation, int)
            or generation < 1 or generation > (1 << 53) - 1
            or not isinstance(catalog, dict)
            or not isinstance(catalog.get(topic), dict)
            or isinstance(catalog[topic].get("count"), bool)
            or not isinstance(catalog[topic].get("count"), int)
            or catalog[topic]["count"] < 0):
        raise ValueError("工作台 bootstrap 未提供完整的当前数据作用域")
    return path, topic, generation, catalog[topic]["count"]


def _validate_frame_state(value, *, frame, scope):
    if (value.get("path") != scope["path"] or value.get("topic") != scope["topic"]
            or value.get("file_activation_generation") !=
            scope["file_activation_generation"] or value.get("frame") != frame):
        raise ValueError(f"frame {frame} 状态读回与请求作用域不一致")
    candidates = _frame_values(value.get("candidates"), frame)
    binding = (value.get("candidate_review_bindings") or {}).get(str(frame))
    if not isinstance(binding, dict):
        raise ValueError(f"frame {frame} 缺少服务端候选绑定")
    artifact_sha = binding.get("candidate_artifact_sha256")
    if artifact_sha != _candidate_binding(candidates):
        raise ValueError(f"frame {frame} 候选集合 SHA 读回不一致")
    entries = binding.get("candidates")
    if not isinstance(entries, list) or len(entries) != len(candidates):
        raise ValueError(f"frame {frame} 候选指纹清单不完整")
    return candidates, binding


def _entry_for(binding, candidate):
    matches = [item for item in binding["candidates"]
               if type(item.get("candidate_id")) is type(candidate.get("id"))
               and item.get("candidate_id") == candidate.get("id")]
    fingerprint = candidate_fingerprint(candidate)
    if (len(matches) != 1 or matches[0].get("candidate_fingerprint") != fingerprint):
        raise ValueError("服务端候选指纹缺失、重复或不一致")
    return fingerprint


def _write_asset(folder, name, payload):
    path = folder / name
    if path.exists():
        raise ValueError(f"输出资源重复：{name}")
    path.write_bytes(payload)
    return _sha256(payload)


def _tree_sha256(folder):
    rows = [{"path": path.relative_to(folder).as_posix(),
             "sha256": _sha256(path.read_bytes())}
            for path in sorted(folder.rglob("*")) if path.is_file()]
    return _sha256(_canonical_bytes(rows)), rows


def build_pack(base_url, frames, output, *, opener=urlopen):
    """Build one deterministic comparison pack without any server mutation."""
    base_url = _safe_base_url(base_url)
    output = Path(output).resolve()
    if output.exists():
        raise FileExistsError(f"输出目录已存在：{output}")
    normalized_frames = []
    for value in frames:
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            raise ValueError("frames 必须是非负整数")
        if value in normalized_frames:
            raise ValueError("frames 不得重复")
        normalized_frames.append(value)
    if not normalized_frames:
        raise ValueError("frames 不能为空")

    bootstrap = _get_json(opener, base_url, "/api/bootstrap")
    source_path, topic, generation, frame_count = _validate_bootstrap(bootstrap)
    if any(frame >= frame_count for frame in normalized_frames):
        raise ValueError(f"frames 超出当前 topic 的 {frame_count} 帧范围")
    scope = {"path": source_path, "topic": topic,
             "file_activation_generation": generation}

    output.parent.mkdir(parents=True, exist_ok=True)
    temp_parent = (output.parent.parent
                   if output.parent.name == "candidate_comparison_packs"
                   else output.parent)
    wrapper = Path(tempfile.mkdtemp(prefix=".cc-", dir=temp_parent))
    try:
        staging = wrapper / "candidate_comparison_packs" / output.name
        staging.mkdir(parents=True)
        rows = []
        for frame in normalized_frames:
            state_before = _get_json(
                opener, base_url, f"/api/frame_state/{frame}", scope)
            candidates, binding = _validate_frame_state(
                state_before, frame=frame, scope=scope)
            encoded_frame, image_rgb = _get_frame(
                opener, base_url, frame, scope)
            height, width = image_rgb.shape[:2]
            for candidate in candidates:
                if str(candidate.get("label", "")).casefold() != "pallet":
                    continue
                identifier = candidate.get("id")
                if (isinstance(identifier, bool)
                        or not isinstance(identifier, int) or identifier < 0):
                    raise ValueError(
                        f"frame {frame} pallet 候选 ID 必须是非负整数")
                fingerprint = _entry_for(binding, candidate)
                before = _rasterize(candidate, width, height).astype(np.uint8)
                refined, audit = refine_pallet_candidate_geometry(
                    copy.deepcopy(candidate), width, height,
                    image_rgb=image_rgb)
                after = _rasterize(refined, width, height).astype(np.uint8)
                changed = int(np.count_nonzero(before != after))
                if audit is None or changed == 0:
                    continue
                prefix = f"f{frame:06d}-c{identifier}"
                current_mask_name = f"{prefix}-current-mask.png"
                refined_mask_name = f"{prefix}-refined-mask.png"
                current_preview_name = f"{prefix}-current-preview.png"
                refined_preview_name = f"{prefix}-refined-preview.png"
                x1, y1, x2, y2 = _comparison_roi(before, after)
                preview_image = image_rgb[y1:y2, x1:x2]
                current_mask_sha = _write_asset(
                    staging, current_mask_name, _mask_png(before))
                refined_mask_sha = _write_asset(
                    staging, refined_mask_name, _mask_png(after))
                current_preview_sha = _write_asset(
                    staging, current_preview_name,
                    _preview_png(preview_image, before[y1:y2, x1:x2],
                                 f"frame {frame} candidate {identifier} current",
                                 (25, 165, 255)))
                refined_preview_sha = _write_asset(
                    staging, refined_preview_name,
                    _preview_png(preview_image, after[y1:y2, x1:x2],
                                 f"frame {frame} candidate {identifier} refined",
                                 (25, 210, 120)))
                before_area = int(before.sum())
                after_area = int(after.sum())
                rows.append({
                    "comparison_id": f"frame-{frame}-candidate-{identifier}-pallet-boundary-cleanup",
                    "frame": frame,
                    "source_frame_sha256": _sha256(encoded_frame),
                    "current_candidate_id": identifier,
                    "current_candidate_artifact_sha256": binding["candidate_artifact_sha256"],
                    "current_candidate_fingerprint": fingerprint,
                    "current": {"image": current_preview_name,
                                "image_sha256": current_preview_sha,
                                "mask": current_mask_name,
                                "mask_sha256": current_mask_sha},
                    "current_pixel_count": before_area,
                    "current_metrics": {
                        "component_count": int(audit.get("component_count_before", 0)),
                        "preview_x1": x1, "preview_y1": y1,
                        "preview_x2": x2, "preview_y2": y2,
                    },
                    "variants": [{
                        "alternative_id": "pallet-boundary-cleanup-v6",
                        "kind": "alternative",
                        "label": "pallet boundary cleanup",
                        "policy": "refine_pallet_candidate_geometry",
                        "mask": refined_mask_name,
                        "mask_sha256": refined_mask_sha,
                        "pixel_count": after_area,
                        "preview": {"image": refined_preview_name,
                                    "image_sha256": refined_preview_sha},
                        "metrics": {
                            "changed_pixels": changed,
                            "change_ratio": round(changed / max(before_area, 1), 6),
                            "pixel_delta": after_area - before_area,
                            "filled_hole_pixels": int(audit.get("filled_hole_pixels", 0)),
                            "removed_island_pixels": int(audit.get("removed_island_pixels", 0)),
                            "component_count_before": int(audit.get("component_count_before", 0)),
                            "component_count_after": int(audit.get("component_count_after", 0)),
                            "geometry_cleanup_version": int(audit.get("geometry_cleanup_version", 0)),
                        },
                    }],
                })
            state_after = _get_json(
                opener, base_url, f"/api/frame_state/{frame}", scope)
            candidates_after, binding_after = _validate_frame_state(
                state_after, frame=frame, scope=scope)
            if (_canonical_bytes(candidates_after) != _canonical_bytes(candidates)
                    or _canonical_bytes(binding_after) != _canonical_bytes(binding)):
                raise ValueError(f"frame {frame} 候选在离线生成期间发生变化")

        if not rows:
            raise ValueError("所选帧没有几何确实变化的 pallet 候选")
        bootstrap_after = _get_json(opener, base_url, "/api/bootstrap")
        if _validate_bootstrap(bootstrap_after)[:3] != (
                source_path, topic, generation):
            raise ValueError("生成期间工作台数据作用域已变化")
        rows.sort(key=lambda item: (item["frame"], item["current_candidate_id"]))
        manifest = {
            "schema_version": SCHEMA_VERSION,
            "candidate_only": True,
            "human_truth": False,
            "approved_for_training": False,
            "training_authorized": False,
            "automatic_approval": False,
            "assistant_decision": None,
            "human_decision": None,
            "source": {"path": source_path, "topic": topic,
                       "file_activation_generation": generation},
            "rows": rows,
        }
        (staging / "manifest.json").write_text(
            json.dumps(manifest, ensure_ascii=False, sort_keys=True, indent=2,
                       allow_nan=False) + "\n", encoding="utf-8")
        verified = read_packet(wrapper, output.name)
        if verified["comparison_count"] != len(rows):
            raise ValueError("生成后 schema 读回数量不一致")
        staging.replace(output)
        tree_sha, files = _tree_sha256(output)
        return {
            "schema_version": "labelagent.pallet-boundary-comparison-build.v1",
            "output": str(output),
            "manifest_sha256": _sha256((output / "manifest.json").read_bytes()),
            "tree_sha256": tree_sha,
            "file_count": len(files),
            "requested_frames": normalized_frames,
            "comparison_count": len(rows),
            "candidate_only": True,
            "human_truth": False,
            "approved_for_training": False,
            "training_authorized": False,
            "automatic_approval": False,
            "server_methods": ["GET"],
        }
    finally:
        shutil.rmtree(wrapper, ignore_errors=True)


def _parse_frames(value):
    try:
        return [int(item.strip()) for item in value.split(",") if item.strip()]
    except ValueError as error:
        raise argparse.ArgumentTypeError("frames 必须是逗号分隔整数") from error


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-url", required=True)
    parser.add_argument("--frames", required=True, type=_parse_frames)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    result = build_pack(args.base_url, args.frames, args.output)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
