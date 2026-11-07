"""Run real local SAM3 text segmentation on every imported camera frame.

This checks inference coverage and exports candidates, not pixel accuracy.
"""
import argparse
import base64
from collections import Counter
import csv
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import time
import urllib.request
import numpy as np
from PIL import Image


def save(path, value):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, separators=(",", ":")), encoding="utf-8")
    temporary.replace(path)


def baseline(dataset, endpoint):
    dataset = Path(dataset)
    manifest = json.loads((dataset / "import_manifest.json").read_text(encoding="utf-8"))
    output = dataset / "sam3_baseline"
    responses = output / "responses"
    responses.mkdir(parents=True, exist_ok=True)
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    records = sorted((row for row in manifest["records"] if row["topic"].endswith("/image")),
                     key=lambda row: (row["frame"], row["topic"]))
    prompts = [("pallet", "pallet"), ("person", "person")]
    candidates, requests = {}, []
    started = datetime.now(timezone.utc).isoformat()

    def progress(complete=False):
        successful = [row for row in requests if row["status"] == "ok"]
        value = {"started_utc": started, "updated_utc": datetime.now(timezone.utc).isoformat(),
                 "status": ("complete" if len(successful) == len(requests) else "complete_with_errors") if complete else "running",
                 "archive_sha256": manifest["archive_sha256"], "planned_images": len(records),
                 "planned_requests": len(records) * len(prompts), "completed_requests": len(requests),
                 "successful_requests": len(successful), "failed_requests": len(requests) - len(successful),
                 "candidate_shapes": sum(row["masks"] for row in successful),
                 "pixel_accuracy_evaluated": False, "external_vlm_calls": 0,
                 "full_agent_verified": False, "candidates_confirmed_as_truth": False}
        if successful:
            value["request_seconds_median"] = float(np.median([row["seconds"] for row in successful]))
            value["request_seconds_p95"] = float(np.percentile([row["seconds"] for row in successful], 95))
        save(output / "public_progress.json", value)
        save(output / "candidates_by_topic.json", candidates)
        return value

    for image_index, row in enumerate(records):
        path = dataset / row["source_file"]
        raw = path.read_bytes()
        if hashlib.sha256(raw).hexdigest() != row["sha256"]:
            raise ValueError("Source image changed since import")
        with Image.open(path) as image:
            width, height = image.size
        camera = row["topic"].split("/")[2]
        target = candidates.setdefault(row["topic"], {}).setdefault(str(row["frame"]), [])
        encoded = base64.b64encode(raw).decode("ascii")
        for label, prompt in prompts:
            record_path = responses / f"{camera}_{row['frame']:04d}_{label}.json"
            if record_path.exists():
                result = json.loads(record_path.read_text(encoding="utf-8"))
                if result["input_sha256"] != row["sha256"] or result["prompt"] != prompt:
                    raise ValueError("Cached request is for different input or prompt")
            else:
                result = {"frame": row["frame"], "camera": camera, "label": label, "prompt": prompt,
                          "input_sha256": row["sha256"], "status": "error", "masks": 0}
                start = time.monotonic()
                try:
                    request = urllib.request.Request(endpoint.rstrip("/") + "/segment_text",
                        data=json.dumps({"image": encoded, "prompt": prompt}).encode(),
                        headers={"Content-Type": "application/json"})
                    with opener.open(request, timeout=180) as response:
                        payload = json.load(response)
                    for item in payload["results"]:
                        if not item.get("parts"):
                            raise ValueError("Segmentation result has no parts")
                        for part in item["parts"]:
                            for ring in [part["points"], *part.get("holes", [])]:
                                points = np.asarray(ring, dtype=float)
                                if points.ndim != 2 or points.shape[0] < 3 or points.shape[1] != 2 or not np.isfinite(points).all():
                                    raise ValueError("Invalid segmentation polygon")
                                if (points[:, 0] < 0).any() or (points[:, 0] >= width).any() or (points[:, 1] < 0).any() or (points[:, 1] >= height).any():
                                    raise ValueError("Segmentation polygon outside input image")
                    result.update(status="ok", masks=len(payload["results"]), response=payload)
                except Exception as exc:
                    result["error"] = f"{type(exc).__name__}: {exc}"
                result["seconds"] = round(time.monotonic() - start, 4)
                save(record_path, result)
            requests.append({key: result[key] for key in ("frame", "camera", "label", "status", "masks", "seconds")})
            if result["status"] == "ok":
                for index, item in enumerate(result["response"]["results"]):
                    part = item["parts"][0]
                    target.append({"id": f"sam3-{camera}-{row['frame']}-{label}-{index}", "label": label,
                                   "points": part["points"], "holes": part.get("holes", []), "parts": item["parts"],
                                   "score": item.get("score"), "source": "sam3_text_unconfirmed"})
        if image_index % 8 == 0 or image_index == len(records) - 1:
            current = progress()
            print(json.dumps({key: current[key] for key in ("completed_requests", "planned_requests", "failed_requests")}), flush=True)
    summary = progress(complete=True)
    with (output / "requests.csv").open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=["frame", "camera", "label", "status", "masks", "seconds"])
        writer.writeheader()
        writer.writerows(requests)
    summary["by_camera"] = {camera: {"requests": sum(row["camera"] == camera for row in requests),
        "successful": sum(row["camera"] == camera and row["status"] == "ok" for row in requests)}
        for camera in sorted({row["camera"] for row in requests})}
    save(output / "summary.json", summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--endpoint", default="http://127.0.0.1:8766")
    args = parser.parse_args()
    baseline(args.dataset, args.endpoint)
