#!/usr/bin/env python3
"""Collect isolated candidate-centered SAM pallet proposals.

This bounded diagnostic derives every crop from the current candidate itself.
It writes only to a new output directory, never to the workbench, truth, or a
training set.  Each request and remapped response is bound to source hashes so
the output can be reviewed independently before any product decision.
"""

from __future__ import annotations

import argparse
import base64
import hashlib
import io
import json
import math
import os
import time
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from PIL import Image

from source.agent.edit_agent import _rasterize
from tools.run_local_agent_acceptance import save
from tools.tiled_candidate_utils import remap_tile_result


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _mapping(payload: dict) -> dict:
    nested = payload.get("candidates_by_topic")
    return nested if isinstance(nested, dict) else payload


def centered_crop_bounds(shape: dict, width: int, height: int, factor: float, *,
                         minimum_side: int = 256,
                         maximum_side: int = 1024) -> tuple[int, int, int, int]:
    if (not math.isfinite(factor) or factor <= 1 or minimum_side <= 0
            or maximum_side < minimum_side or width <= 0 or height <= 0):
        raise ValueError("invalid candidate-centered crop settings")
    mask = _rasterize(shape, width, height).astype(bool)
    rows, columns = np.nonzero(mask)
    if not len(rows):
        raise ValueError(f"empty target shape: {shape.get('id')}")
    x0, x1 = int(columns.min()), int(columns.max()) + 1
    y0, y1 = int(rows.min()), int(rows.max()) + 1
    side = math.ceil(max(x1 - x0, y1 - y0) * factor)
    side = min(maximum_side, max(minimum_side, side), width, height)
    cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
    left = max(0, min(width - side, round(cx - side / 2)))
    top = max(0, min(height - side, round(cy - side / 2)))
    return left, top, left + side, top + side


def _read_json(url: str, timeout: int = 10) -> dict:
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    with opener.open(url, timeout=timeout) as response:
        payload = json.load(response)
    if not isinstance(payload, dict):
        raise ValueError(f"expected JSON object: {url}")
    return payload


def collect(args) -> int:
    baseline_path = Path(args.baseline).resolve()
    audit_path = Path(args.audit).resolve()
    image_root = Path(args.image_root).resolve()
    output = Path(args.output).resolve()
    state_file = Path(args.state_file).resolve() if args.state_file else None
    if output.exists():
        raise FileExistsError(f"output already exists: {output}")
    factors = [float(value) for value in args.factors.split(",")]
    if (not factors or len(set(factors)) != len(factors)
            or any(not math.isfinite(value) or value <= 1 for value in factors)):
        raise ValueError("factors must be distinct numbers greater than one")
    source_hashes = {
        str(baseline_path): _sha256(baseline_path),
        str(audit_path): _sha256(audit_path),
    }
    state_before = _sha256(state_file) if state_file else None
    baseline = _mapping(json.loads(baseline_path.read_text(encoding="utf-8")))
    audit = json.loads(audit_path.read_text(encoding="utf-8"))
    rows = sorted(
        (row for row in audit.get("results", [])
         if row.get("size_band") in {"small", "tiny_skipped"}
         and float(row.get("change_ratio") or 0) > 0),
        key=lambda row: (-float(row.get("change_ratio") or 0),
                         str(row.get("topic")), int(row.get("frame", -1)),
                         str(row.get("shape_id"))))[:args.limit]
    if not rows:
        raise ValueError("no distant pallet targets selected")

    image_inventory = {
        camera: sorted((image_root / camera).glob("*.png"))
        for camera in {str(row["topic"]).split("/")[2] for row in rows}
    }
    requests = []
    for row in rows:
        topic, frame = str(row["topic"]), str(row["frame"])
        shapes = baseline.get(topic, {}).get(frame)
        if shapes is None:
            raise ValueError(f"missing baseline frame: {(topic, frame)}")
        targets = [shape for shape in shapes
                   if str(shape.get("id")) == str(row.get("shape_id"))]
        if len(targets) != 1:
            raise ValueError(f"target shape is not unique: {(topic, frame, row.get('shape_id'))}")
        seen = set()
        for factor in factors:
            bounds = centered_crop_bounds(
                targets[0], args.width, args.height, factor,
                minimum_side=args.minimum_side, maximum_side=args.maximum_side)
            if bounds in seen:
                continue
            seen.add(bounds)
            requests.append({
                "topic": topic,
                "frame": frame,
                "shape_id": row.get("shape_id"),
                "factor": factor,
                "bounds": list(bounds),
                "target": targets[0],
            })

    sam = args.sam.rstrip("/")
    health = _read_json(sam + "/health")
    if health.get("status") != "ok" or health.get("model_loaded") is not True:
        raise RuntimeError("SAM service is not loaded and ready")
    output.mkdir(parents=True, exist_ok=False)
    (output / "responses").mkdir()
    plan = {
        "schema_version": "labelagent.candidate_centered_pallet_plan.v1",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "baseline_sha256": source_hashes[str(baseline_path)],
        "audit_sha256": source_hashes[str(audit_path)],
        "collector_sha256": _sha256(Path(__file__)),
        "selection": "highest_geometry_change_small_candidates",
        "selected_target_count": len(rows),
        "planned_request_count": len(requests),
        "factors": factors,
        "minimum_side": args.minimum_side,
        "maximum_side": args.maximum_side,
        "prompt": args.prompt,
        "candidate_only": True,
        "human_truth": False,
        "approved_for_training": False,
        "workbench_writes": 0,
    }
    save(output / "study_plan.json", plan)
    plan_sha = _sha256(output / "study_plan.json")
    progress = {
        "status": "running",
        "pid": os.getpid(),
        "planned_requests": len(requests),
        "completed_requests": 0,
        "successful_requests": 0,
        "raw_proposals": 0,
        "edge_rejected": 0,
        "mapped_proposals": 0,
        "workbench_writes": 0,
    }
    save(output / "progress.json", progress)
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    appended: dict[tuple[str, str], list[dict]] = {}
    image_hashes: dict[str, dict[str, str]] = {}
    try:
        for index, item in enumerate(requests):
            topic, frame = item["topic"], item["frame"]
            camera = topic.split("/")[2]
            image_path = image_inventory[camera][int(frame)]
            raw = image_path.read_bytes()
            image_hash = hashlib.sha256(raw).hexdigest()
            image_hashes.setdefault(topic, {})[frame] = image_hash
            image = Image.open(io.BytesIO(raw)).convert("RGB")
            if image.size != (args.width, args.height):
                raise ValueError(f"unexpected image dimensions: {image_path}")
            bounds = tuple(item["bounds"])
            crop = image.crop(bounds)
            buffer = io.BytesIO()
            crop.save(buffer, format="PNG")
            crop_bytes = buffer.getvalue()
            body = json.dumps({
                "image": base64.b64encode(crop_bytes).decode("ascii"),
                "prompt": args.prompt,
            }).encode("utf-8")
            request = urllib.request.Request(
                sam + "/segment_text", body,
                headers={"Content-Type": "application/json"})
            started = time.monotonic()
            with opener.open(request, timeout=args.timeout) as response:
                result = json.load(response)
            if not isinstance(result.get("results"), list):
                raise ValueError("SAM response has no results list")
            receipt = {
                "request_index": index,
                "topic": topic,
                "frame": frame,
                "shape_id": item["shape_id"],
                "factor": item["factor"],
                "bounds": item["bounds"],
                "source_image_sha256": image_hash,
                "crop_png_sha256": hashlib.sha256(crop_bytes).hexdigest(),
                "prompt": args.prompt,
                "plan_sha256": plan_sha,
                "seconds": round(time.monotonic() - started, 3),
                "response": result,
            }
            save(output / "responses" / f"request-{index:03d}.json", receipt)
            accepted = []
            for position, proposal in enumerate(result["results"]):
                mapped = remap_tile_result(
                    proposal, bounds, args.width, args.height,
                    f"centered-{camera}-{frame}-{index}-{position}",
                    args.edge_margin)
                if mapped is None:
                    progress["edge_rejected"] += 1
                    continue
                mapped["source"] = "real_SAM3_candidate_centered_unconfirmed"
                mapped["target_shape_id"] = item["shape_id"]
                mapped["crop_factor"] = item["factor"]
                accepted.append(mapped)
            appended.setdefault((topic, frame), []).extend(accepted)
            progress["completed_requests"] += 1
            progress["successful_requests"] += 1
            progress["raw_proposals"] += len(result["results"])
            progress["mapped_proposals"] += len(accepted)
            save(output / "progress.json", progress)
            print(json.dumps({
                "completed": progress["completed_requests"],
                "total": progress["planned_requests"],
                "topic": topic,
                "frame": frame,
                "mapped": len(accepted),
                "seconds": receipt["seconds"],
            }), flush=True)

        augmented = {}
        for topic, frame in sorted(appended):
            augmented.setdefault(topic, {})[frame] = [
                *baseline[topic][frame], *appended[topic, frame]]
        candidate_path = output / "candidate_centered_augmented.json"
        save(candidate_path, augmented)
        receipt_path = output / "candidate_centered_receipt.json"
        save(receipt_path, {
            "schema_version": "labelagent.candidate_centered_pallet_receipt.v1",
            "created_utc": datetime.now(timezone.utc).isoformat(),
            "baseline_sha256": source_hashes[str(baseline_path)],
            "audit_sha256": source_hashes[str(audit_path)],
            "prediction_json_sha256": _sha256(candidate_path),
            "image_sha256": image_hashes,
            "plan_sha256": plan_sha,
            "new_model_requests": progress["completed_requests"],
            "workbench_writes": 0,
            "candidate_only": True,
            "human_truth": False,
            "approved_for_training": False,
        })
        progress["status"] = "complete"
        progress["candidate_file"] = str(candidate_path)
        progress["candidate_file_sha256"] = _sha256(candidate_path)
        progress["receipt_sha256"] = _sha256(receipt_path)
    except Exception as error:
        progress["status"] = "failed"
        progress["error_type"] = type(error).__name__
        (output / "local_error.txt").write_text(str(error), encoding="utf-8")
    finally:
        progress["finished_utc"] = datetime.now(timezone.utc).isoformat()
        progress["inputs_unchanged"] = all(
            _sha256(Path(path)) == digest for path, digest in source_hashes.items())
        progress["state_sha256_before"] = state_before
        progress["state_sha256_after"] = _sha256(state_file) if state_file else None
        progress["state_unchanged"] = (
            progress["state_sha256_after"] == state_before if state_file else None)
        progress["plan_unchanged"] = _sha256(output / "study_plan.json") == plan_sha
        save(output / "progress.json", progress)
    print(json.dumps(progress, ensure_ascii=False), flush=True)
    return 0 if progress["status"] == "complete" else 1


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", required=True)
    parser.add_argument("--audit", required=True)
    parser.add_argument("--image-root", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--sam", required=True)
    parser.add_argument("--state-file")
    parser.add_argument("--limit", type=int, default=16)
    parser.add_argument("--factors", default="3,6")
    parser.add_argument("--minimum-side", type=int, default=256)
    parser.add_argument("--maximum-side", type=int, default=1024)
    parser.add_argument("--width", type=int, default=1920)
    parser.add_argument("--height", type=int, default=1536)
    parser.add_argument("--prompt", default="pallet", choices=(
        "pallet", "wooden pallet", "plastic pallet", "pallet base"))
    parser.add_argument("--edge-margin", type=int, default=2)
    parser.add_argument("--timeout", type=int, default=600)
    return collect(parser.parse_args())


if __name__ == "__main__":
    raise SystemExit(main())
