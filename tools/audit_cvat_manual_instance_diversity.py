"""Screen manual-instance candidates for redundant near-duplicate images."""

from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
import math
from pathlib import Path
from typing import Any

from PIL import Image


THRESHOLDS = {"phash_hamming_max": 4, "dhash_hamming_max": 4,
              "thumbnail_mad_max": 0.035}


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _read(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"JSON object required: {path}")
    return value


def _bits_hex(bits: list[bool]) -> str:
    value = 0
    for bit in bits:
        value = (value << 1) | int(bit)
    return f"{value:016x}"


def _fingerprint(path: Path) -> dict[str, Any]:
    with Image.open(path) as opened:
        gray = opened.convert("L")
        thumb_image = gray.resize((32, 32), Image.Resampling.LANCZOS)
        dh_image = gray.resize((9, 8), Image.Resampling.LANCZOS)
        thumb = list(thumb_image.get_flattened_data() if hasattr(thumb_image, "get_flattened_data") else thumb_image.getdata())
        dh = list(dh_image.get_flattened_data() if hasattr(dh_image, "get_flattened_data") else dh_image.getdata())
    dhash = _bits_hex([dh[y * 9 + x] > dh[y * 9 + x + 1]
                       for y in range(8) for x in range(8)])
    coeff: list[float] = []
    cosine = [[math.cos((2 * x + 1) * u * math.pi / 64)
               for x in range(32)] for u in range(8)]
    for v in range(8):
        for u in range(8):
            total = 0.0
            for y in range(32):
                row = y * 32
                total += cosine[v][y] * sum(
                    thumb[row + x] * cosine[u][x] for x in range(32))
            coeff.append(total)
    median = sorted(coeff[1:])[len(coeff[1:]) // 2]
    phash = _bits_hex([value > median for value in coeff])
    return {"phash": phash, "dhash": dhash, "thumbnail": thumb}


def _hamming(left: str, right: str) -> int:
    return (int(left, 16) ^ int(right, 16)).bit_count()


def audit(candidate_manifest: Path, candidate_receipt: Path,
          image_root: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    candidate_manifest = candidate_manifest.resolve()
    candidate_receipt = candidate_receipt.resolve()
    image_root = image_root.resolve()
    value, receipt = _read(candidate_manifest), _read(candidate_receipt)
    digest = sha256(candidate_manifest)
    if receipt.get("private_manifest_sha256") != digest:
        raise ValueError("candidate manifest hash differs from receipt")
    if value.get("status") != "prepared_not_training_authorized":
        raise ValueError("candidate manifest status differs")
    if value.get("errors") != [] or receipt.get("errors") != []:
        raise ValueError("candidate source contains errors")
    candidates = value.get("candidates")
    summary = receipt.get("summary")
    if not isinstance(candidates, list) or not isinstance(summary, dict):
        raise ValueError("candidate inventory is absent")
    if summary.get("candidate_count") != len(candidates):
        raise ValueError("candidate count differs from receipt")

    items: list[dict[str, Any]] = []
    seen_ids: set[str] = set()
    seen_hashes: set[str] = set()
    for candidate in candidates:
        if not isinstance(candidate, dict):
            raise ValueError("candidate is not an object")
        candidate_id = candidate.get("candidate_id")
        image = candidate.get("image")
        shape = candidate.get("shape")
        if not isinstance(candidate_id, str) or candidate_id in seen_ids:
            raise ValueError("candidate id is invalid or repeated")
        if not isinstance(image, dict) or not isinstance(shape, dict):
            raise ValueError(f"candidate {candidate_id}: binding is incomplete")
        if (shape.get("label") != "pallet"
                or (shape.get("cvat_metadata") or {}).get("source") != "manual"):
            raise ValueError(f"candidate {candidate_id}: provenance differs")
        filename, expected = image.get("file"), image.get("sha256")
        if not isinstance(filename, str) or Path(filename).name != filename:
            raise ValueError(f"candidate {candidate_id}: unsafe image filename")
        path = image_root / filename
        if not path.is_file() or sha256(path) != expected:
            raise ValueError(f"candidate {candidate_id}: image hash differs")
        if expected in seen_hashes:
            raise ValueError("candidate image hash is repeated")
        seen_ids.add(candidate_id)
        seen_hashes.add(expected)
        items.append({"candidate_id": candidate_id, "image_sha256": expected,
                      **_fingerprint(path)})

    pairs: list[dict[str, Any]] = []
    parent = list(range(len(items)))

    def root(index: int) -> int:
        while parent[index] != index:
            parent[index] = parent[parent[index]]
            index = parent[index]
        return index

    def union(left: int, right: int) -> None:
        a, b = root(left), root(right)
        if a != b:
            parent[max(a, b)] = min(a, b)

    for left_index, left in enumerate(items):
        for right_index in range(left_index + 1, len(items)):
            right = items[right_index]
            ph = _hamming(left["phash"], right["phash"])
            dh = _hamming(left["dhash"], right["dhash"])
            mad = sum(abs(a - b) for a, b in zip(left["thumbnail"], right["thumbnail"])) / (1024 * 255)
            hit = (ph <= THRESHOLDS["phash_hamming_max"]
                   and dh <= THRESHOLDS["dhash_hamming_max"]
                   and mad <= THRESHOLDS["thumbnail_mad_max"])
            row = {"left": left["candidate_id"], "right": right["candidate_id"],
                   "phash_hamming": ph, "dhash_hamming": dh,
                   "thumbnail_mad": round(mad, 6), "threshold_hit": hit}
            pairs.append(row)
            if hit:
                union(left_index, right_index)
    groups: dict[int, list[int]] = {}
    for index in range(len(items)):
        groups.setdefault(root(index), []).append(index)
    clusters = []
    representatives = []
    for indices in sorted(groups.values(), key=lambda group: min(group)):
        ids = [items[index]["candidate_id"] for index in indices]
        representative = ids[0]
        clusters.append({"representative": representative, "candidate_ids": ids,
                         "size": len(ids)})
        representatives.append(representative)
    hits = [row for row in pairs if row["threshold_hit"]]
    pairs.sort(key=lambda row: (row["phash_hamming"], row["dhash_hamming"],
                                row["thumbnail_mad"], row["left"], row["right"]))
    private = {
        "schema_version": "labelagent.cvat-manual-instance-diversity-audit.v1",
        "status": "assistant_triage_only",
        "candidate_manifest_sha256": digest,
        "candidate_receipt_sha256": sha256(candidate_receipt),
        "thresholds": THRESHOLDS,
        "candidate_count": len(items),
        "pair_count": len(pairs),
        "threshold_hit_count": len(hits),
        "cluster_count": len(clusters),
        "representative_count": len(representatives),
        "cluster_size_histogram": dict(sorted(Counter(cluster["size"] for cluster in clusters).items())),
        "clusters": clusters,
        "representative_candidate_ids": representatives,
        "nearest_pairs": pairs[:100],
        "candidate_only": True,
        "human_truth": False,
        "samples_removed": 0,
        "training_authorized": False,
        "automatic_approval": False,
        "limitations": [
            "Perceptual similarity is a review-efficiency heuristic, not a semantic duplicate proof.",
            "Cluster representatives are suggestions only; no candidate is deleted or approved.",
        ],
    }
    public = {key: private[key] for key in (
        "schema_version", "status", "candidate_manifest_sha256",
        "candidate_receipt_sha256", "thresholds", "candidate_count", "pair_count",
        "threshold_hit_count", "cluster_count", "representative_count",
        "cluster_size_histogram", "candidate_only", "human_truth", "samples_removed",
        "training_authorized", "automatic_approval", "limitations")}
    public["private_cluster_ids_published"] = False
    public["raw_images_published"] = False
    return private, public


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidates", required=True, type=Path)
    parser.add_argument("--candidate-receipt", required=True, type=Path)
    parser.add_argument("--image-root", required=True, type=Path)
    parser.add_argument("--private-output", required=True, type=Path)
    parser.add_argument("--receipt-output", required=True, type=Path)
    args = parser.parse_args()
    if args.private_output.exists() or args.receipt_output.exists():
        raise ValueError("choose fresh output paths")
    private, public = audit(args.candidates, args.candidate_receipt, args.image_root)
    args.private_output.parent.mkdir(parents=True, exist_ok=True)
    args.private_output.write_text(json.dumps(private, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    public["private_audit_sha256"] = sha256(args.private_output)
    args.receipt_output.parent.mkdir(parents=True, exist_ok=True)
    args.receipt_output.write_text(json.dumps(public, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(public, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
