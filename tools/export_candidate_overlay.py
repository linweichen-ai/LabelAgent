"""Read one workbench candidate and export a non-mutating, hash-audited overlay."""
import argparse
import hashlib
import io
import json
from pathlib import Path
import sys
import urllib.parse
import urllib.request

from PIL import Image


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from source.agent.closed_loop import canonical_json
from source.agent.downstream_mapping import mapping_contract
from source.agent.edit_agent import _rasterize
from source.agent.workflow import candidate_overlay


def loopback_url(value):
    parsed = urllib.parse.urlparse(value)
    if parsed.scheme != "http" or parsed.hostname not in {"127.0.0.1", "localhost", "::1"}:
        raise ValueError("workbench must be a loopback HTTP URL")
    if parsed.username or parsed.password or parsed.query or parsed.fragment:
        raise ValueError("workbench URL cannot contain credentials, query, or fragment")
    return value.rstrip("/")


def request(url):
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    with opener.open(url, timeout=60) as response:
        return response.read()


def write_once(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    existing = path.read_bytes() if path.exists() else None
    if existing is not None and existing != data:
        raise ValueError(f"refusing to overwrite different artifact: {path}")
    if existing is None:
        path.write_bytes(data)


def export(args):
    base = loopback_url(args.workbench)
    state = json.loads(request(base + "/api/state"))
    if not state.get("loaded") or args.topic not in state.get("topics", {}):
        raise ValueError("requested workbench dataset/topic is not loaded")
    shapes = state.get("candidates", {}).get(str(args.frame), [])
    shapes = [shape for shape in shapes if shape.get("label") == args.label]
    if not shapes:
        raise ValueError("requested candidate label is absent on this frame")
    frame_url = base + f"/api/frame/{args.frame}?" + urllib.parse.urlencode({"topic": args.topic})
    frame = request(frame_url)
    frame_hash = hashlib.sha256(frame).hexdigest()
    if args.expected_frame_sha256 and frame_hash != args.expected_frame_sha256:
        raise ValueError("frame SHA-256 does not match --expected-frame-sha256")
    with Image.open(io.BytesIO(frame)) as image:
        width, height = image.size
        image_format = image.format
    overlay = candidate_overlay(frame, shapes)
    overlay_hash = hashlib.sha256(overlay).hexdigest()
    candidate_hash = hashlib.sha256(canonical_json(shapes)).hexdigest()
    masks = [_rasterize(shape, width, height) for shape in shapes]
    union = masks[0].copy()
    for mask in masks[1:]:
        union |= mask
    ys, xs = union.nonzero()
    if not len(xs):
        raise ValueError("candidate rasterization is empty")
    geometry = {"outer_vertex_count": sum(len(shape.get("points", [])) for shape in shapes),
                "mask_pixels": int(union.sum()),
                "image_fraction": round(float(union.mean()), 8),
                "bbox_xyxy": [int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())]}
    truth = state.get("truth", {}).get(str(args.frame), [])
    references = state.get("reference_frames", {}).get(args.topic, [])
    evidence = {
        "evidence_type": "slope_candidate_overlay",
        "dataset_id": args.dataset_id,
        "source_batch": args.source_batch,
        "source_uri": state.get("path"),
        "topic": args.topic,
        "frame_index": args.frame,
        "captured_at": args.captured_at,
        "hard_case_stratum": "slope",
        "semantic_label": "slope_visible",
        "business_mapping": "traversable_area",
        "downstream_mapping": mapping_contract(),
        "traversability_status": "passable",
        "policy_default": "passable",
        "policy_basis": "user_defined_project_semantics",
        "policy_override": {"status": "none", "rule_id": None, "reason": None,
                            "decided_by": None, "evidence_artifact_ids": []},
        "frame": {"index": args.frame, "width": width, "height": height,
                  "format": image_format, "bytes": len(frame), "sha256": frame_hash},
        "candidate": {"label": args.label, "shape_count": len(shapes),
                      "sha256": candidate_hash,
                      "scores": [shape.get("score") for shape in shapes],
                      "sources": [shape.get("source") for shape in shapes],
                      "geometry": geometry},
        "overlay": {"path": str(Path(args.output).resolve()), "bytes": len(overlay),
                    "sha256": overlay_hash},
        "human_truth_shape_count": len(truth),
        "is_reference_frame": args.frame in references,
        "quality_metrics_available": False,
        "quality_claim": False,
        "human_review_status": "pending",
    }
    encoded = (json.dumps(evidence, ensure_ascii=False, indent=2, sort_keys=True) + "\n").encode("utf-8")
    write_once(args.output, overlay)
    write_once(args.evidence, encoded)
    return evidence


def parser():
    value = argparse.ArgumentParser(description=__doc__)
    value.add_argument("--workbench", default="http://127.0.0.1:8767")
    value.add_argument("--dataset-id", required=True)
    value.add_argument("--source-batch", required=True)
    value.add_argument("--topic", required=True)
    value.add_argument("--frame", type=int, required=True)
    value.add_argument("--captured-at", required=True)
    value.add_argument("--label", default="slope")
    value.add_argument("--expected-frame-sha256")
    value.add_argument("--output", required=True)
    value.add_argument("--evidence", required=True)
    return value


def main():
    evidence = export(parser().parse_args())
    print(json.dumps(evidence, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
