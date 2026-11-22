"""Read-only perceptual near-duplicate screen for frozen VLM splits.

Pixels are decoded on the data host.  The command never writes remotely and
returns only bounded JSON fingerprints and cross-split pair diagnostics.
"""

from __future__ import annotations

import argparse
import base64
from datetime import datetime, timezone
import json
from pathlib import Path, PurePosixPath
import shutil
import subprocess

ALLOWED_REMOTE_ROOT = PurePosixPath(
    "/home/clw/datasets/sfl-cpd15y/20260920_cs-heli_Dev_000001")
THRESHOLDS = {"phash_hamming_max": 6, "dhash_hamming_max": 6,
              "thumbnail_mad_max": 0.08}


def read_jsonl(path: Path) -> list[dict]:
    rows = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        value = json.loads(line)
        if not isinstance(value, dict):
            raise ValueError(f"JSONL object required at {path}:{line_number}")
        rows.append(value)
    return rows

REMOTE_CODE = r'''
import hashlib, json, math, os, sys
from pathlib import Path
from PIL import Image

ROOT = Path("/home/clw/datasets/sfl-cpd15y/20260920_cs-heli_Dev_000001").resolve()
VALUES = json.load(sys.stdin)
COS = [[math.cos((2*x+1)*u*math.pi/64) for x in range(32)] for u in range(8)]
try:
    RESAMPLE = Image.Resampling.LANCZOS
except AttributeError:
    RESAMPLE = Image.LANCZOS

def bits_to_hex(bits):
    value = 0
    for bit in bits:
        value = (value << 1) | int(bool(bit))
    return f"{value:016x}"

def fingerprint(entry):
    path = Path(entry["image_path"])
    resolved = path.resolve(strict=True)
    if ROOT not in resolved.parents or not resolved.is_file():
        raise ValueError("image path is outside the frozen dataset root")
    payload = resolved.read_bytes()
    actual = hashlib.sha256(payload).hexdigest()
    if actual != entry["image_sha256"]:
        raise ValueError(f"image hash mismatch: {entry['sample_id']}")
    with Image.open(resolved) as image:
        gray = image.convert("L")
        thumb = list(gray.resize((32, 32), RESAMPLE).getdata())
        dh = list(gray.resize((9, 8), RESAMPLE).getdata())
    dhash = bits_to_hex(dh[y*9+x] > dh[y*9+x+1] for y in range(8) for x in range(8))
    coeff = []
    for v in range(8):
        for u in range(8):
            total = 0.0
            for y in range(32):
                cy = COS[v][y]
                row = y * 32
                total += cy * sum(thumb[row+x] * COS[u][x] for x in range(32))
            coeff.append(total)
    median_source = sorted(coeff[1:])[len(coeff[1:]) // 2]
    phash = bits_to_hex(value > median_source for value in coeff)
    return dict(entry, actual_sha256=actual, phash=phash, dhash=dhash,
                thumbnail=thumb)

def hamming(left, right):
    return (int(left, 16) ^ int(right, 16)).bit_count()

items = [fingerprint(value) for value in VALUES]
pairs = []
for left_index, left in enumerate(items):
    for right in items[left_index + 1:]:
        if left["split"] == right["split"]:
            continue
        ph = hamming(left["phash"], right["phash"])
        dh = hamming(left["dhash"], right["dhash"])
        mad = sum(abs(a-b) for a, b in zip(left["thumbnail"], right["thumbnail"])) / (1024 * 255)
        pairs.append({
            "left": left["sample_id"], "left_split": left["split"],
            "right": right["sample_id"], "right_split": right["split"],
            "phash_hamming": ph, "dhash_hamming": dh,
            "thumbnail_mad": round(mad, 6),
            "threshold_hit": ph <= 6 and dh <= 6 and mad <= 0.08,
        })
pairs.sort(key=lambda row: (row["phash_hamming"], row["dhash_hamming"],
                            row["thumbnail_mad"], row["left"], row["right"]))
print(json.dumps({
    "images": len(items), "cross_split_pairs": len(pairs),
    "threshold_hits": [row for row in pairs if row["threshold_hit"]],
    "nearest_pairs": pairs[:20],
    "fingerprints": [{key: value for key, value in item.items()
                       if key != "thumbnail"} for item in items],
}, separators=(",", ":")))
'''


def manifest_entries(paths: dict[str, Path]) -> list[dict]:
    entries = []
    seen = set()
    for split, path in paths.items():
        for row in read_jsonl(path):
            image_path = row.get("image_path")
            image_sha = str(row.get("image_sha256") or "").lower()
            sample_id = row.get("sample_id")
            if not all(isinstance(value, str) and value for value in
                       (image_path, image_sha, sample_id)):
                raise ValueError(f"{split}: incomplete image binding")
            pure_path = PurePosixPath(image_path)
            if (not pure_path.is_absolute()
                    or ALLOWED_REMOTE_ROOT not in pure_path.parents):
                raise ValueError(f"{split}: image path outside frozen dataset root")
            identity = (split, sample_id)
            if identity in seen:
                raise ValueError(f"{split}: duplicate sample id")
            seen.add(identity)
            entries.append({"split": split, "sample_id": sample_id,
                            "image_path": image_path,
                            "image_sha256": image_sha})
    return entries


def screen_remote(entries: list[dict], host: str, ssh: str) -> dict:
    encoded_code = base64.b64encode(REMOTE_CODE.encode("utf-8")).decode("ascii")
    remote_command = (
        "python3 -c \"import base64;exec(base64.b64decode('"
        + encoded_code + "'))\"")
    completed = subprocess.run(
        [ssh, "-o", "BatchMode=yes", "-o", "ConnectTimeout=10", host,
         remote_command],
        input=json.dumps(entries, separators=(",", ":")), text=True,
        capture_output=True, timeout=180, check=False)
    if completed.returncode:
        detail = completed.stderr.strip()[-1000:]
        raise RuntimeError(f"remote read-only screen failed: {detail}")
    value = json.loads(completed.stdout)
    if not isinstance(value, dict):
        raise ValueError("remote result is not an object")
    return value


def report(entries: list[dict], remote: dict, host: str) -> dict:
    expected_pairs = sum(
        1 for index, left in enumerate(entries)
        for right in entries[index + 1:] if left["split"] != right["split"])
    if remote.get("images") != len(entries):
        raise ValueError("remote image count differs")
    if remote.get("cross_split_pairs") != expected_pairs:
        raise ValueError("remote pair count differs")
    hits = remote.get("threshold_hits")
    nearest = remote.get("nearest_pairs")
    if not isinstance(hits, list) or not isinstance(nearest, list) or len(nearest) > 20:
        raise ValueError("remote bounded result shape differs")
    return {
        "schema_version": "labelagent.vlm-near-duplicate-screen.v1",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "status": "review_required" if hits else "passed_heuristic_screen",
        "scope": "read-only cross-split perceptual-hash screen",
        "host": host,
        "images": len(entries),
        "cross_split_pairs": expected_pairs,
        "thresholds": THRESHOLDS,
        "threshold_hit_count": len(hits),
        "threshold_hits": hits,
        "nearest_pairs": nearest,
        "remote_writes": False,
        "pixels_downloaded": False,
        "human_truth": False,
        "production_training_approved": False,
        "limitations": [
            "Perceptual hashes and 32x32 grayscale MAD are a screening heuristic, not proof of semantic independence.",
            "No human scene-identity review was performed.",
            "A threshold miss cannot rule out related routes, objects, or backgrounds.",
        ],
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train", required=True, type=Path)
    parser.add_argument("--dev", required=True, type=Path)
    parser.add_argument("--test24", required=True, type=Path)
    parser.add_argument("--host", default="factory-4090")
    parser.add_argument("--ssh", default=shutil.which("ssh") or "ssh")
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    entries = manifest_entries({"train": args.train.resolve(),
                                "dev": args.dev.resolve(),
                                "test24": args.test24.resolve()})
    value = report(entries, screen_remote(entries, args.host, args.ssh), args.host)
    output = args.output.resolve()
    if output.exists():
        raise ValueError("choose a fresh output path")
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n",
                         encoding="utf-8")
    temporary.replace(output)
    print(json.dumps(value, ensure_ascii=False), flush=True)
    return 0 if value["status"] == "passed_heuristic_screen" else 2


if __name__ == "__main__":
    raise SystemExit(main())
