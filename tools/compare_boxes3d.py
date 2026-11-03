"""Compare archived 3-D cuboid sources against the selected labels reference.

Uses exact convex-polyhedron intersection with Rz(yaw) Ry(pitch) Rx(roll),
then maximum-cardinality, maximum-IoU same-class matching at each threshold.
Missing prediction frames are coverage gaps, not implicit empty predictions.
"""
import argparse
from collections import defaultdict
from datetime import datetime, timezone
import hashlib
from itertools import combinations, product
import json
from pathlib import Path
import time
import numpy as np
from scipy.optimize import linear_sum_assignment
from scipy.spatial import ConvexHull, QhullError

TRIPLES = np.array(list(combinations(range(12), 3)))
CORNERS = np.array(list(product((-0.5, 0.5), repeat=3)))


class Box:
    def __init__(self, obj):
        psr = obj["obj_psr"]
        self.label = obj["obj_type"]
        self.center = np.array([psr["position"][key] for key in ("x", "y", "z")], dtype=float)
        self.size = np.array([psr["scale"][key] for key in ("x", "y", "z")], dtype=float)
        angles = np.array([psr["rotation"][key] for key in ("roll", "pitch", "yaw")], dtype=float)
        if not np.isfinite(np.concatenate((self.center, self.size, angles))).all() or (self.size <= 0).any():
            raise ValueError("Invalid cuboid geometry")
        sr, sp, sy = np.sin(angles)
        cr, cp, cy = np.cos(angles)
        self.rotation = np.array([[cy*cp, cy*sp*sr-sy*cr, cy*sp*cr+sy*sr],
                                  [sy*cp, sy*sp*sr+cy*cr, sy*sp*cr-cy*sr],
                                  [-sp, cp*sr, cp*cr]])
        self.corners = (CORNERS * self.size) @ self.rotation.T + self.center
        self.normals = np.concatenate((self.rotation.T, -self.rotation.T))
        self.offsets = self.normals @ self.center + np.tile(self.size / 2, 2)
        self.volume = float(np.prod(self.size))
        self.low, self.high = self.corners.min(axis=0), self.corners.max(axis=0)


def iou3d(a, b):
    if np.any(np.minimum(a.high, b.high) - np.maximum(a.low, b.low) <= 0):
        return 0.0
    normals = np.concatenate((a.normals, b.normals))
    offsets = np.concatenate((a.offsets, b.offsets))
    matrices = normals[TRIPLES]
    invertible = np.abs(np.linalg.det(matrices)) > 1e-10
    intersections = np.linalg.solve(matrices[invertible], offsets[TRIPLES][invertible, :, None])[..., 0]
    candidates = np.concatenate((a.corners, b.corners, intersections))
    vertices = candidates[(candidates @ normals.T <= offsets + 1e-7).all(axis=1)]
    vertices = np.unique(np.round(vertices, 11), axis=0)
    if len(vertices) < 4 or np.linalg.matrix_rank(vertices - vertices[0], tol=1e-9) < 3:
        return 0.0
    try:
        intersection = float(ConvexHull(vertices).volume)
    except QhullError:
        return 0.0
    intersection = min(intersection, a.volume, b.volume)
    return max(0.0, min(1.0, intersection / (a.volume + b.volume - intersection)))


def match(matrix, threshold):
    if not matrix.size:
        return []
    reward = (matrix >= threshold) * (min(matrix.shape) + 1) + matrix
    rows, columns = linear_sum_assignment(-reward)
    return [(int(row), int(column), float(matrix[row, column]))
            for row, column in zip(rows, columns) if matrix[row, column] >= threshold]


def scores(tp, predictions, references, overlaps):
    denominator = predictions + references
    return {"matched": tp, "predictions": predictions, "references": references,
            "unmatched_predictions": predictions - tp, "unmatched_references": references - tp,
            "precision_against_reference": tp / predictions if predictions else None,
            "recall_against_reference": tp / references if references else None,
            "f1_against_reference": 2 * tp / denominator if denominator else None,
            "matched_iou_mean": float(np.mean(overlaps)) if overlaps else None}


def compare(dataset, output):
    dataset, output = Path(dataset), Path(output)
    manifest = json.loads((dataset / "import_manifest.json").read_text(encoding="utf-8"))
    output.mkdir(parents=True, exist_ok=True)
    frame_indices = {str(timestamp): index for index, timestamp in enumerate(manifest["timestamps_ns"])}

    def read_source(name):
        values, ignored = {}, 0
        for path in sorted((dataset / "source" / name).glob("*.json")):
            if not path.stem.isdigit():
                continue
            raw = path.read_bytes()
            if hashlib.sha256(raw).hexdigest() != manifest["extracted_sha256"][f"{name}/{path.name}"]:
                raise ValueError("Annotation source changed after import")
            record = json.loads(raw)
            objects = record["objects"]
            ignored += sum(bool(obj.get("ignore")) for obj in objects)
            values[path.stem] = [Box(obj) for obj in objects if not obj.get("ignore")]
        return values, ignored

    reference, reference_ignored = read_source("labels")
    report = {"created_utc": datetime.now(timezone.utc).isoformat(),
              "archive_sha256": manifest["archive_sha256"], "reference_source": "labels",
              "reference_frames": len(reference), "reference_objects_after_ignore_filter": sum(map(len, reference.values())),
              "reference_ignored_objects": reference_ignored,
              "reference_is_independent_ground_truth": False, "predictions_reused_from_archive": True,
              "fresh_3d_model_inference_performed": False, "rotation_convention": "Rz(yaw) Ry(pitch) Rx(roll), radians; assumed from field names",
              "matching": "same class; maximum cardinality above threshold, then maximum summed IoU",
              "tracking_identity_evaluated": False, "pixel_accuracy_evaluated": False, "methods": {}}
    detailed = {}
    for method in ("ali_label", "wilddet3d-labels"):
        start = time.monotonic()
        predicted, ignored = read_source(method)
        common = sorted(set(reference) & set(predicted), key=int)
        accumulators = defaultdict(lambda: {"tp": 0, "predictions": 0, "references": 0, "overlaps": []})
        frames = []
        for number, stamp in enumerate(common):
            refs, preds = reference[stamp], predicted[stamp]
            row = {"frame_index": frame_indices[stamp], "classes": {}}
            for label in sorted({box.label for box in refs + preds}):
                ref_class = [box for box in refs if box.label == label]
                pred_class = [box for box in preds if box.label == label]
                matrix = np.zeros((len(ref_class), len(pred_class)))
                for i, a in enumerate(ref_class):
                    for j, b in enumerate(pred_class):
                        matrix[i, j] = iou3d(a, b)
                row["classes"][label] = {}
                for threshold in (0.25, 0.5):
                    matches = match(matrix, threshold)
                    overlaps = [item[2] for item in matches]
                    row["classes"][label][str(threshold)] = scores(len(matches), len(pred_class), len(ref_class), overlaps)
                    for group in (label, "all"):
                        stats = accumulators[group, threshold]
                        stats["tp"] += len(matches)
                        stats["predictions"] += len(pred_class)
                        stats["references"] += len(ref_class)
                        stats["overlaps"].extend(overlaps)
            frames.append(row)
            if number % 10 == 0:
                print(f"{method}: {number + 1}/{len(common)} common frames", flush=True)
        metrics = {}
        for (label, threshold), stats in sorted(accumulators.items()):
            metrics.setdefault(label, {})[str(threshold)] = scores(stats["tp"], stats["predictions"], stats["references"], stats["overlaps"])
        report["methods"][method] = {"available_prediction_frames": len(predicted), "evaluated_common_frames": len(common),
            "missing_prediction_frames": len(set(reference) - set(predicted)),
            "orphan_prediction_frames": len(set(predicted) - set(reference)),
            "frame_coverage": len(common) / len(reference) if reference else None,
            "ignored_prediction_objects": ignored, "metrics": metrics, "seconds": round(time.monotonic() - start, 3)}
        detailed[method] = frames
    (output / "public_summary.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    (output / "per_frame.json").write_text(json.dumps(detailed, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    compare(args.dataset, args.output)
