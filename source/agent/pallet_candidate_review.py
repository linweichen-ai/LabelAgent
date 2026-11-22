"""Fail-closed reader for the optional pallet candidate review page."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import threading


SCHEMA = "labelagent.pallet_visibility_candidate_bundle.v1"
ENV_NAME = "LABELAGENT_PALLET_CANDIDATE_REVIEW_DIR"
_CACHE_LOCK = threading.Lock()
_CACHE_TOKEN = None
_CACHE_PAGE = None


class CandidateReviewUnavailable(ValueError):
    """Raised when the optional review page is disabled or fails validation."""


def configured_root() -> Path:
    raw = os.environ.get(ENV_NAME, "").strip()
    if not raw:
        raise CandidateReviewUnavailable("候选复核页未启用")
    root = Path(raw).expanduser().resolve()
    if not root.is_dir():
        raise CandidateReviewUnavailable("候选复核目录不可用")
    return root


def _safe_relative_path(value: object) -> str:
    if not isinstance(value, str) or not value:
        raise CandidateReviewUnavailable("候选资源路径无效")
    path = PurePosixPath(value)
    if path.is_absolute() or ".." in path.parts or "." in path.parts:
        raise CandidateReviewUnavailable("候选资源路径越界")
    return path.as_posix()


def _verified_asset(root: Path, relative: str, expected_sha256: str) -> bytes:
    normalized = _safe_relative_path(relative)
    path = (root / Path(*PurePosixPath(normalized).parts)).resolve()
    if root not in path.parents or not path.is_file():
        raise CandidateReviewUnavailable("候选资源不存在")
    data = path.read_bytes()
    if hashlib.sha256(data).hexdigest() != expected_sha256:
        raise CandidateReviewUnavailable("候选资源哈希不一致")
    if not data.startswith(b"\x89PNG\r\n\x1a\n"):
        raise CandidateReviewUnavailable("候选资源不是 PNG")
    return data


def _copy_cached_page(page: dict) -> dict:
    copied = dict(page)
    copied["assets"] = dict(page["assets"])
    return copied


def load_review_page() -> dict:
    """Return verified page bytes and its exact allow-listed image assets."""
    global _CACHE_TOKEN, _CACHE_PAGE
    root = configured_root()
    try:
        bundle_bytes = (root / "bundle.json").read_bytes()
        page_bytes = (root / "index.html").read_bytes()
        bundle = json.loads(bundle_bytes.decode("utf-8"))
        page_text = page_bytes.decode("utf-8")
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise CandidateReviewUnavailable("候选清单或复核页不可读") from error
    token = (str(root), hashlib.sha256(bundle_bytes).hexdigest(),
             hashlib.sha256(page_bytes).hexdigest())
    with _CACHE_LOCK:
        if token == _CACHE_TOKEN and _CACHE_PAGE is not None:
            return _copy_cached_page(_CACHE_PAGE)
    if (bundle.get("schema_version") != SCHEMA
            or bundle.get("production_deployed") is not False
            or bundle.get("human_truth_writes") != 0
            or bundle.get("training_data_writes") != 0
            or bundle.get("workbench_writes") != 0):
        raise CandidateReviewUnavailable("候选清单安全标记不完整")
    rows = bundle.get("rows")
    previews = bundle.get("previews")
    if (not isinstance(rows, list) or not isinstance(previews, list)
            or bundle.get("candidate_count") != len(rows)
            or bundle.get("preview_count") != len(previews)):
        raise CandidateReviewUnavailable("候选清单计数不一致")

    allowed_assets = {}
    for item in previews:
        relative = _safe_relative_path(item.get("path"))
        allowed_assets[relative] = _verified_asset(
            root, relative, item.get("sha256", ""))
    for row in rows:
        relative = row.get("mask_path")
        if relative is None:
            if row.get("mask_pixels") != 0:
                raise CandidateReviewUnavailable("抑制项仍包含掩码像素")
            continue
        relative = _safe_relative_path(relative)
        allowed_assets[relative] = _verified_asset(
            root, relative, row.get("mask_sha256", ""))

    forbidden = ("<script", "<form", "<button", " href=", "http://", "https://")
    if ("离线候选 · 未部署 · 不是真值" not in page_text
            or any(token in page_text.lower() for token in forbidden)):
        raise CandidateReviewUnavailable("候选复核页不是只读页面")
    page = {
        "page_bytes": page_bytes,
        "assets": allowed_assets,
        "candidate_count": len(rows),
        "preview_count": len(previews),
        "mask_count": sum(row.get("mask_path") is not None for row in rows),
        "suppressed_count": sum(row.get("mask_path") is None for row in rows),
    }
    with _CACHE_LOCK:
        _CACHE_TOKEN = token
        _CACHE_PAGE = page
    return _copy_cached_page(page)

