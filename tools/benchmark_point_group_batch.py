"""Compare sequential point prompts with the shared-encoding group endpoint."""
from __future__ import annotations

import argparse
import base64
import hashlib
import json
import statistics
import time
import urllib.error
import urllib.request
from datetime import datetime, timezone
from pathlib import Path


def _post(base_url: str, endpoint: str, image: bytes, **payload):
    body = {"image": base64.b64encode(image).decode("ascii"), **payload}
    request = urllib.request.Request(
        base_url.rstrip("/") + endpoint,
        data=json.dumps(body).encode("utf-8"),
        headers={"Content-Type": "application/json"}, method="POST")
    try:
        with urllib.request.urlopen(request, timeout=30) as response:
            return json.loads(response.read())
    except urllib.error.HTTPError as error:
        try:
            detail = json.loads(error.read()).get("detail", str(error))
        except (ValueError, AttributeError):
            detail = str(error)
        return {"error": detail}


def _percentile(values: list[float], fraction: float) -> float:
    ordered = sorted(values)
    return ordered[min(len(ordered) - 1, int((len(ordered) - 1) * fraction))]


def benchmark(image_directory: Path, manifest_path: Path, evidence_path: Path,
              baseline_url: str, candidate_url: str) -> dict:
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    evidence = json.loads(evidence_path.read_text(encoding="utf-8"))
    selected = evidence["selected_frames"]
    contract = evidence["contract"]
    negatives = [
        {"x": x, "y": y, "label": 0}
        for x, y in contract["negative_points"]
    ]
    groups = [
        [{"x": x, "y": y, "label": 1}, *negatives]
        for x, y in contract["positive_points"]
    ]

    # Warm the isolated service outside measured rows.  The reference frame is
    # never included in the evaluation set or quality metrics.
    warm = image_directory / manifest["frames"][evidence["reference_frame"]]["file"]
    _post(candidate_url, "/segment_point_groups_batch", warm.read_bytes(),
          prompt_groups=groups)

    rows = []
    for frame in selected:
        meta = manifest["frames"][frame]
        image = (image_directory / meta["file"]).read_bytes()
        if hashlib.sha256(image).hexdigest() != meta["sha256"]:
            raise ValueError(f"source image hash mismatch for frame {frame}")

        started = time.perf_counter()
        sequential = [
            _post(baseline_url, "/segment_points", image, points=group)
            for group in groups
        ]
        sequential_seconds = time.perf_counter() - started

        started = time.perf_counter()
        batch = _post(candidate_url, "/segment_point_groups_batch", image,
                      prompt_groups=groups)
        batch_seconds = time.perf_counter() - started
        items = batch.get("items", []) if isinstance(batch, dict) else []
        normalized_batch = [
            item.get("result", {"error": item.get("error")}) for item in items
        ]
        rows.append({
            "frame": frame,
            "source_image_sha256": meta["sha256"],
            "sequential_seconds": round(sequential_seconds, 6),
            "batch_seconds": round(batch_seconds, 6),
            "results_exactly_equal": sequential == normalized_batch,
            "sequential_success_count": sum("error" not in item for item in sequential),
            "batch_success_count": sum("result" in item for item in items),
            "candidate_diagnostics": batch.get("diagnostics", {}),
        })
    sequential_times = [row["sequential_seconds"] for row in rows]
    batch_times = [row["batch_seconds"] for row in rows]
    sequential_median = statistics.median(sequential_times)
    batch_median = statistics.median(batch_times)
    return {
        "schema_version": "labelagent.point_group_real_ab.v1",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "reference_frame_excluded_from_measurement": True,
        "selected_frames": selected,
        "sample_count": len(rows),
        "group_count_per_frame": len(groups),
        "results_exactly_equal_count": sum(
            row["results_exactly_equal"] for row in rows),
        "success_counts_equal_count": sum(
            row["sequential_success_count"] == row["batch_success_count"]
            for row in rows),
        "sequential_median_seconds": round(sequential_median, 6),
        "batch_median_seconds": round(batch_median, 6),
        "median_speedup": round(sequential_median / batch_median, 4),
        "sequential_p95_seconds": round(_percentile(sequential_times, .95), 6),
        "batch_p95_seconds": round(_percentile(batch_times, .95), 6),
        "rows": rows,
        "real_images_uploaded": False,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image-directory", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--round402-evidence", type=Path, required=True)
    parser.add_argument("--baseline-url", default="http://127.0.0.1:28766")
    parser.add_argument("--candidate-url", default="http://127.0.0.1:28767")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    for url in (args.baseline_url, args.candidate_url):
        if not url.startswith("http://127.0.0.1:"):
            raise ValueError("SAM A/B endpoints must be loopback")
    if args.output.exists():
        raise FileExistsError(args.output)
    report = benchmark(
        args.image_directory, args.manifest, args.round402_evidence,
        args.baseline_url, args.candidate_url)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({key: report[key] for key in (
        "sample_count", "results_exactly_equal_count",
        "success_counts_equal_count", "sequential_median_seconds",
        "batch_median_seconds", "median_speedup", "sequential_p95_seconds",
        "batch_p95_seconds")}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
