"""Loopback-only FastAPI SAM3 service used by the factory-4090 deployment.

Besides feature reuse within one multi-prompt request, this version keeps a
bounded CPU JSON cache for exact repeated requests.  The speculative endpoint
never queues behind unrelated model work, so UI prewarming cannot delay a
foreground request.
"""

from __future__ import annotations

import argparse
import base64
import hashlib
import io
import json
import os
import threading
from contextlib import nullcontext
from time import perf_counter

import torch
from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
from PIL import Image

try:  # Package import for tests and repository use.
    from .point_segment import (
        _model_lock, _predictor, model_loaded, segment, segment_many,
        segment_prompt_groups)
    from .response_cache import ExactResponseCoordinator
    from .sam_multi_helpers import evaluate_prompt_states, validate_prompt_map
except ImportError:  # Direct script execution in the remote deployment folder.
    from point_segment import (
        _model_lock, _predictor, model_loaded, segment, segment_many,
        segment_prompt_groups)
    from response_cache import ExactResponseCoordinator
    from sam_multi_helpers import evaluate_prompt_states, validate_prompt_map


SERVICE_VERSION = "labelagent.remote-sam3.v3-point-groups"
app = FastAPI(title="LabelAgent remote SAM3", version="2.0")
request_lock = threading.Lock()
response_cache = ExactResponseCoordinator(
    max_entries=int(os.environ.get("SAM3_RESPONSE_CACHE_ENTRIES", "24")),
    max_bytes=int(os.environ.get("SAM3_RESPONSE_CACHE_BYTES", str(128 * 1024 * 1024))),
)


def decode_image(value):
    if not isinstance(value, str):
        raise ValueError("image must be base64 text")
    raw = base64.b64decode(value, validate=True)
    if len(raw) > 64 * 1024 * 1024:
        raise ValueError("image is too large")
    return raw


def predict_text_prompts(image, prompts):
    """Encode one image once, then evaluate several text prompts sequentially."""
    _, processor = _predictor()
    precision = (torch.autocast("cuda", dtype=torch.bfloat16)
                 if torch.cuda.is_available() else nullcontext())
    started = perf_counter()
    with torch.inference_mode(), precision:
        encode_started = perf_counter()
        state = processor.set_image(image)
        image_encoding_seconds = perf_counter() - encode_started
        _, outputs, prompt_seconds = evaluate_prompt_states(
            processor, state, prompts)
    return outputs, {
        "image_encoded_once": True,
        "image_encoding_seconds": image_encoding_seconds,
        "prompt_seconds": prompt_seconds,
        "total_seconds": perf_counter() - started,
    }


def exact_multi_key(image_bytes, prompts):
    """Bind cache reuse to exact bytes, ordered labels, prompts and implementation."""
    digest = hashlib.sha256()
    digest.update(SERVICE_VERSION.encode("utf-8"))
    digest.update(b"\0multi\0")
    digest.update(len(image_bytes).to_bytes(8, "big"))
    digest.update(image_bytes)
    digest.update(json.dumps(list(prompts.items()), ensure_ascii=False,
                             separators=(",", ":")).encode("utf-8"))
    return digest.hexdigest()


def exact_text_key(image_bytes, prompt):
    digest = hashlib.sha256()
    digest.update(SERVICE_VERSION.encode("utf-8"))
    digest.update(b"\0text\0")
    digest.update(len(image_bytes).to_bytes(8, "big"))
    digest.update(image_bytes)
    digest.update(prompt.encode("utf-8"))
    return digest.hexdigest()


def _prepare_multi(body):
    image_bytes = decode_image(body.get("image"))
    prompts = validate_prompt_map(body.get("prompts"))
    key = exact_multi_key(image_bytes, prompts)
    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    return image, prompts, key


def _prepare_text(body):
    image_bytes = decode_image(body.get("image"))
    prompt = body.get("prompt")
    if not isinstance(prompt, str) or not prompt.strip():
        raise ValueError("prompt must be non-empty text")
    prompt = prompt.strip()
    key = exact_text_key(image_bytes, prompt)
    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    return image, prompt, key


def _cache_response(payload, key, *, hit, coalesced=False, wait_seconds=0.0):
    result = dict(payload)
    diagnostics = dict(result.get("diagnostics") or {})
    diagnostics["response_cache"] = {
        "hit": bool(hit),
        "coalesced": bool(coalesced),
        "coalesced_wait_seconds": round(float(wait_seconds), 6),
        "key_sha256": key,
        **response_cache.stats(),
    }
    result["diagnostics"] = diagnostics
    return result


def _wait_for_claim(claim, key):
    payload, waited = response_cache.wait(claim.flight, timeout=1200)
    return _cache_response(payload, key, hit=True, coalesced=True,
                           wait_seconds=waited)


def _compute_owned_multi(image, prompts, key, claim, lock_already_held=False):
    try:
        if lock_already_held:
            with _model_lock:
                outputs, timing = predict_text_prompts(image, prompts)
        else:
            with request_lock, _model_lock:
                outputs, timing = predict_text_prompts(image, prompts)
        payload = {"prompts": outputs, "diagnostics": timing}
        response_cache.complete(key, claim.flight, payload)
        return _cache_response(payload, key, hit=False)
    except BaseException as error:
        response_cache.fail(key, claim.flight, error)
        raise


def _compute_owned_text(image, prompt, key, claim, lock_already_held=False):
    try:
        if lock_already_held:
            with _model_lock:
                outputs, _ = predict_text_prompts(image, {"result": prompt})
        else:
            with request_lock, _model_lock:
                outputs, _ = predict_text_prompts(image, {"result": prompt})
        payload = outputs["result"]
        response_cache.complete(key, claim.flight, payload)
        return _cache_response(payload, key, hit=False)
    except BaseException as error:
        response_cache.fail(key, claim.flight, error)
        raise


@app.get("/health")
def health():
    return {
        "status": "ok",
        "model_loaded": model_loaded(),
        "backend": "remote_4090_sam3",
        "service_version": SERVICE_VERSION,
        "response_cache": response_cache.stats(),
    }


@app.post("/segment_points")
def segment_points(body: dict):
    try:
        image = decode_image(body.get("image"))
        points = body.get("points")
        if not isinstance(points, list):
            raise ValueError("points must be a list")
        mask = body.get("mask")
        prior = base64.b64decode(mask, validate=True) if isinstance(mask, str) else None
        with request_lock:
            return segment(image, points, prior_mask=prior)
    except (ValueError, TypeError) as error:
        raise HTTPException(status_code=400, detail=str(error)) from error


@app.post("/segment_points_batch")
def segment_points_batch(body: dict):
    """Generate independent point candidates with one image feature pass."""
    try:
        image = decode_image(body.get("image"))
        points = body.get("points")
        if not isinstance(points, list):
            raise ValueError("points must be a list")
        with request_lock:
            return segment_many(image, points)
    except (ValueError, TypeError) as error:
        raise HTTPException(status_code=400, detail=str(error)) from error


@app.post("/segment_point_groups_batch")
def segment_point_groups_batch(body: dict):
    """Generate independent positive/negative groups with one feature pass."""
    try:
        image = decode_image(body.get("image"))
        prompt_groups = body.get("prompt_groups")
        if not isinstance(prompt_groups, list):
            raise ValueError("prompt_groups must be a list")
        with request_lock:
            return segment_prompt_groups(image, prompt_groups)
    except (ValueError, TypeError) as error:
        raise HTTPException(status_code=400, detail=str(error)) from error


@app.post("/segment_text")
def segment_text(body: dict):
    try:
        image, prompt, key = _prepare_text(body)
        claim = response_cache.claim(key)
        if claim.role == "hit":
            return _cache_response(claim.payload, key, hit=True)
        if claim.role == "wait":
            return _wait_for_claim(claim, key)
        return _compute_owned_text(image, prompt, key, claim)
    except (ValueError, TypeError) as error:
        raise HTTPException(status_code=400, detail=str(error)) from error


@app.post("/prefetch_text")
def prefetch_text(body: dict):
    """Warm one exact text prompt without queuing behind unrelated work."""
    try:
        image, prompt, key = _prepare_text(body)
        observed = response_cache.peek(key)
        if observed.role == "hit":
            return _cache_response(observed.payload, key, hit=True)
        if observed.role == "wait":
            return _wait_for_claim(observed, key)
        if not request_lock.acquire(blocking=False):
            return JSONResponse(status_code=202, content={
                "prefetched": False,
                "code": "sam_busy",
                "key_sha256": key,
                "response_cache": response_cache.stats(),
            })
        lock_held = True
        try:
            claim = response_cache.claim(key)
            if claim.role == "hit":
                return _cache_response(claim.payload, key, hit=True)
            if claim.role == "wait":
                request_lock.release()
                lock_held = False
                return _wait_for_claim(claim, key)
            result = _compute_owned_text(
                image, prompt, key, claim, lock_already_held=True)
            result["prefetched"] = True
            return result
        finally:
            if lock_held:
                request_lock.release()
    except (ValueError, TypeError) as error:
        raise HTTPException(status_code=400, detail=str(error)) from error


@app.post("/segment_text_multi")
def segment_text_multi(body: dict):
    """Serve foreground work, coalescing and caching exact repeats."""
    try:
        image, prompts, key = _prepare_multi(body)
        claim = response_cache.claim(key)
        if claim.role == "hit":
            return _cache_response(claim.payload, key, hit=True)
        if claim.role == "wait":
            return _wait_for_claim(claim, key)
        return _compute_owned_multi(image, prompts, key, claim)
    except (ValueError, TypeError) as error:
        raise HTTPException(status_code=400, detail=str(error)) from error


@app.post("/prefetch_text_multi")
def prefetch_text_multi(body: dict):
    """Warm an exact response only when unrelated model work is not active."""
    try:
        image, prompts, key = _prepare_multi(body)
        observed = response_cache.peek(key)
        if observed.role == "hit":
            return _cache_response(observed.payload, key, hit=True)
        if observed.role == "wait":
            return _wait_for_claim(observed, key)
        if not request_lock.acquire(blocking=False):
            return JSONResponse(status_code=202, content={
                "prefetched": False,
                "code": "sam_busy",
                "key_sha256": key,
                "response_cache": response_cache.stats(),
            })
        lock_held = True
        try:
            # A foreground request may have claimed this key between peek and
            # lock acquisition. Recheck so only one exact computation survives.
            claim = response_cache.claim(key)
            if claim.role == "hit":
                return _cache_response(claim.payload, key, hit=True)
            if claim.role == "wait":
                # The existing owner may itself be waiting for request_lock.
                # Release before joining it or the two requests could deadlock.
                request_lock.release()
                lock_held = False
                return _wait_for_claim(claim, key)
            result = _compute_owned_multi(
                image, prompts, key, claim, lock_already_held=True)
            result["prefetched"] = True
            return result
        finally:
            if lock_held:
                request_lock.release()
    except (ValueError, TypeError) as error:
        raise HTTPException(status_code=400, detail=str(error)) from error


if __name__ == "__main__":
    import uvicorn

    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=18766)
    args = parser.parse_args()
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise RuntimeError("exactly one CUDA GPU must be visible")
    print({"service": "sam3", "visible": os.environ.get("CUDA_VISIBLE_DEVICES"),
           "gpu": torch.cuda.get_device_name(0), "port": args.port,
           "version": SERVICE_VERSION}, flush=True)
    uvicorn.run(app, host=args.host, port=args.port, workers=1)
