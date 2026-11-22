"""Fail-closed reader for local, read-only candidate comparison packs."""
from __future__ import annotations

import copy
import hashlib
import io
import json
import math
import re
from pathlib import Path

from PIL import Image


SCHEMA_VERSION = "labelagent.candidate-comparison-pack.v1"
INDEX_SCHEMA_VERSION = "labelagent.candidate-comparison-pack-index.v1"
_SAFE_ID = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]{0,199}")
_IMAGE_SUFFIXES = {".png": "image/png", ".jpg": "image/jpeg", ".jpeg": "image/jpeg"}
_MAX_MANIFEST_BYTES = 4 * 1024 * 1024
_MAX_IMAGE_BYTES = 8 * 1024 * 1024
_MAX_PACK_IMAGE_BYTES = 128 * 1024 * 1024


def _valid_sha256(value):
    return (isinstance(value, str) and
            re.fullmatch(r"[0-9a-f]{64}", value) is not None)


def _safe_id(value, label):
    if not isinstance(value, str) or _SAFE_ID.fullmatch(value) is None:
        raise ValueError(f"{label} 无效")
    return value


def _scalar(value, label):
    if value is None or isinstance(value, (str, int, float)) and not isinstance(value, bool):
        if isinstance(value, float) and not math.isfinite(value):
            raise ValueError(f"{label} 必须是有限数值")
        if isinstance(value, str) and len(value) > 200:
            raise ValueError(f"{label} 过长")
        return value
    raise ValueError(f"{label} 必须是标量")


def _metrics(value):
    if value is None:
        return {}
    if not isinstance(value, dict) or len(value) > 30:
        raise ValueError("候选对比指标无效或过多")
    result = {}
    for key, item in value.items():
        if not isinstance(key, str) or not key or len(key) > 80:
            raise ValueError("候选对比指标名称无效")
        result[key] = _scalar(item, f"指标 {key}")
    return result


def _asset(value, *, label, manifest_dir, allowed_images, total_bytes):
    if not isinstance(value, dict):
        raise ValueError(f"{label} 图片绑定无效")
    name = value.get("image")
    digest = value.get("image_sha256")
    if (not isinstance(name, str) or Path(name).name != name
            or _SAFE_ID.fullmatch(name) is None
            or Path(name).suffix.lower() not in _IMAGE_SUFFIXES
            or not _valid_sha256(digest)):
        raise ValueError(f"{label} 图片绑定无效")
    if name in allowed_images:
        # Reuse would make a URL ambiguous when two manifest rows claim
        # different evidence identities, so v1 intentionally rejects it.
        raise ValueError("候选对比包重复使用图片")
    path = (manifest_dir / name).resolve()
    if path.parent != manifest_dir or not path.is_file():
        raise ValueError(f"{label} 图片不存在或路径越界")
    payload = path.read_bytes()
    if not payload or len(payload) > _MAX_IMAGE_BYTES:
        raise ValueError(f"{label} 图片为空或过大")
    if total_bytes[0] + len(payload) > _MAX_PACK_IMAGE_BYTES:
        raise ValueError("候选对比包图片总量过大")
    if hashlib.sha256(payload).hexdigest() != digest:
        raise ValueError(f"{label} 图片哈希不一致")
    try:
        with Image.open(io.BytesIO(payload)) as decoded:
            decoded.verify()
        with Image.open(io.BytesIO(payload)) as decoded:
            if (decoded.format not in {"PNG", "JPEG"}
                    or decoded.width <= 0 or decoded.height <= 0
                    or decoded.width > 4096 or decoded.height > 4096
                    or decoded.width * decoded.height > 16_000_000):
                raise ValueError(f"{label} 图片格式或尺寸无效")
    except (OSError, Image.DecompressionBombError) as error:
        raise ValueError(f"{label} 图片无法安全解码") from error
    total_bytes[0] += len(payload)
    allowed_images[name] = {
        "bytes": payload,
        "mimetype": _IMAGE_SUFFIXES[Path(name).suffix.lower()],
    }
    return {"image": name, "image_sha256": digest}


def _manifest_path(work_root, packet_id):
    packet_id = _safe_id(packet_id, "候选对比包标识")
    root = (Path(work_root) / "candidate_comparison_packs").resolve()
    path = (root / packet_id / "manifest.json").resolve()
    if path.parent.parent != root:
        raise ValueError("候选对比包路径越界")
    if not path.is_file():
        raise ValueError("候选对比包不存在")
    return path


def manifest_sha256(work_root, packet_id):
    """Read only the bounded manifest identity for a short live-state check."""
    path = _manifest_path(work_root, packet_id)
    raw = path.read_bytes()
    if not raw or len(raw) > _MAX_MANIFEST_BYTES:
        raise ValueError("候选对比包清单为空或过大")
    return hashlib.sha256(raw).hexdigest()


def read_packet(work_root, packet_id):
    """Load and verify one comparison pack without exposing its source path."""
    path = _manifest_path(work_root, packet_id)
    raw = path.read_bytes()
    if not raw or len(raw) > _MAX_MANIFEST_BYTES:
        raise ValueError("候选对比包清单为空或过大")
    try:
        value = json.loads(raw.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError("候选对比包清单不是有效 UTF-8 JSON") from error
    if not isinstance(value, dict) or value.get("schema_version") != SCHEMA_VERSION:
        raise ValueError("候选对比包版本不受支持")
    if (value.get("candidate_only") is not True
            or value.get("human_truth") is not False
            or value.get("approved_for_training") is not False
            or value.get("training_authorized") is not False
            or value.get("automatic_approval") is not False
            or value.get("human_decision") is not None
            or value.get("assistant_decision") is not None):
        raise ValueError("候选对比包越过候选、真值或训练隔离门禁")
    source = value.get("source")
    if not isinstance(source, dict):
        raise ValueError("候选对比包缺少源作用域")
    source_path = source.get("path")
    source_topic = source.get("topic")
    if (not isinstance(source_path, str) or not Path(source_path).is_absolute()
            or not isinstance(source_topic, str) or not source_topic.strip()
            or len(source_topic) > 500):
        raise ValueError("候选对比包源作用域无效")
    rows = value.get("rows")
    if not isinstance(rows, list) or not 1 <= len(rows) <= 200:
        raise ValueError("候选对比包记录为空或过多")

    allowed_images = {}
    total_bytes = [0]
    normalized = []
    seen_ids = set()
    seen_identity = set()
    for row in rows:
        if not isinstance(row, dict):
            raise ValueError("候选对比记录无效")
        comparison_id = _safe_id(row.get("comparison_id"), "候选对比记录标识")
        if comparison_id in seen_ids:
            raise ValueError("候选对比包包含重复记录标识")
        frame = row.get("frame")
        candidate_id = row.get("current_candidate_id")
        if isinstance(frame, bool) or not isinstance(frame, int) or frame < 0:
            raise ValueError("候选对比记录帧号无效")
        # Live workbench candidates use integer IDs and their canonical
        # fingerprint rejects any other type.  Accepting a string here would
        # create a pack that can never pass the live scope check.
        if isinstance(candidate_id, bool) or not isinstance(candidate_id, int):
            raise ValueError("候选对比记录候选 ID 无效")
        identity = (frame, json.dumps(candidate_id, ensure_ascii=False, sort_keys=True))
        if identity in seen_identity:
            raise ValueError("候选对比包重复绑定同帧候选")
        for field in ("source_frame_sha256", "current_candidate_artifact_sha256",
                      "current_candidate_fingerprint"):
            if not _valid_sha256(row.get(field)):
                raise ValueError(f"候选对比记录 {field} 无效")
        current_pixel_count = row.get("current_pixel_count")
        if (current_pixel_count is not None
                and (isinstance(current_pixel_count, bool)
                     or not isinstance(current_pixel_count, int)
                     or current_pixel_count < 0)):
            raise ValueError("当前候选像素数无效")
        if (row.get("candidate_only") not in {None, True}
                or row.get("human_truth") not in {None, False}
                or row.get("approved_for_training") not in {None, False}
                or row.get("training_authorized") not in {None, False}
                or row.get("automatic_approval") not in {None, False}
                or row.get("human_decision") is not None
                or row.get("assistant_decision") is not None):
            raise ValueError("候选对比记录越过候选、真值或训练隔离门禁")
        current = _asset(row.get("current"), label="当前候选",
                         manifest_dir=path.parent, allowed_images=allowed_images,
                         total_bytes=total_bytes)
        variants = row.get("variants")
        if not isinstance(variants, list) or not 1 <= len(variants) <= 8:
            raise ValueError("候选对比记录替代候选为空或过多")
        normalized_variants = []
        seen_variants = set()
        for variant in variants:
            if not isinstance(variant, dict):
                raise ValueError("替代候选记录无效")
            variant_id = _safe_id(variant.get("alternative_id"), "替代候选标识")
            if variant_id in seen_variants:
                raise ValueError("候选对比记录包含重复替代候选")
            kind = variant.get("kind")
            if kind not in {"alternative", "consensus"}:
                raise ValueError("替代候选类型无效")
            mask_sha = variant.get("mask_sha256")
            pixels = variant.get("pixel_count")
            if (not _valid_sha256(mask_sha) or isinstance(pixels, bool)
                    or not isinstance(pixels, int) or pixels < 0):
                raise ValueError("替代候选 mask 绑定无效")
            asset = _asset(variant.get("preview"), label="替代候选",
                           manifest_dir=path.parent, allowed_images=allowed_images,
                           total_bytes=total_bytes)
            normalized_variants.append({
                "alternative_id": variant_id,
                "kind": kind,
                "label": _scalar(variant.get("label"), "替代候选标签"),
                "policy": _scalar(variant.get("policy"), "替代候选策略"),
                "mask_sha256": mask_sha,
                "pixel_count": pixels,
                "preview": asset,
                "metrics": _metrics(variant.get("metrics")),
            })
            seen_variants.add(variant_id)
        normalized.append({
            "comparison_id": comparison_id,
            "frame": frame,
            "current_candidate_id": candidate_id,
            "source_frame_sha256": row["source_frame_sha256"],
            "current_candidate_artifact_sha256": row["current_candidate_artifact_sha256"],
            "current_candidate_fingerprint": row["current_candidate_fingerprint"],
            "current": dict(current, pixel_count=current_pixel_count,
                            metrics=_metrics(row.get("current_metrics"))),
            "variants": normalized_variants,
            "candidate_only": True,
            "human_truth": False,
            "approved_for_training": False,
            "training_authorized": False,
            "automatic_approval": False,
            "assistant_decision": None,
            "human_decision": None,
        })
        seen_ids.add(comparison_id)
        seen_identity.add(identity)

    return {
        "schema_version": SCHEMA_VERSION,
        "packet_id": _safe_id(packet_id, "候选对比包标识"),
        "manifest_sha256": hashlib.sha256(raw).hexdigest(),
        "source_topic": source_topic,
        "source_path_internal": source_path,
        "comparison_count": len(normalized),
        "rows": normalized,
        "read_only_packet": True,
        "session_preferences_only": True,
        "human_truth_writes": False,
        "training_data_writes": False,
        "automatic_approval": False,
        "allowed_images": allowed_images,
    }


def public_packet(packet):
    result = copy.deepcopy({key: value for key, value in packet.items()
                            if key not in {"source_path_internal", "allowed_images"}})
    packet_id = result["packet_id"]
    manifest_sha = result["manifest_sha256"]
    for row in result["rows"]:
        row["current"]["image_url"] = (
            f"/api/candidate_comparison_packs/{packet_id}/images/"
            f"{row['current']['image']}?manifest_sha256={manifest_sha}"
            f"&image_sha256={row['current']['image_sha256']}")
        for variant in row["variants"]:
            variant["preview"]["image_url"] = (
                f"/api/candidate_comparison_packs/{packet_id}/images/"
                f"{variant['preview']['image']}?manifest_sha256={manifest_sha}"
                f"&image_sha256={variant['preview']['image_sha256']}")
    return result


def packet_summary(packet):
    return copy.deepcopy({key: packet[key] for key in (
        "schema_version", "packet_id", "manifest_sha256", "source_topic",
        "comparison_count", "read_only_packet", "session_preferences_only",
        "human_truth_writes", "training_data_writes", "automatic_approval")})
