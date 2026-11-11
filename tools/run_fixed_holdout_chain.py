"""Run the bounded factory-holdout candidate review chain on existing services.

The tool never writes human truth or training state.  Phase one renders the
existing SAM candidate and asks the VLM to review the original plus overlay.
Phase two records an explicit Agent action and performs an independent VLM
re-review of the resulting candidate.
"""
from __future__ import annotations

import argparse
import base64
import hashlib
import io
import json
import time
import urllib.request
from pathlib import Path

from PIL import Image, ImageDraw


INPUT_ID = "bc706b57eedaea59e3deda1e25eb5a20002e90b61089083e84d2f98130c5600b"
FRAME_ID = "1789897506782450197"
IMAGE_SHA256 = "34e5094d24db487b3fcd5c458ac810cde5d58293f89fe0b92a2933c72a146812"
SAM_SHA256 = "0be59d677c0a910fbf79437484f25cdfdd8256aed4e347a656447bbe9f642593"
ALLOWED_ACTIONS = {"split", "merge", "resegment", "reclassify", "drop", "no_op"}


def sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def canonical(value) -> bytes:
    return json.dumps(value, ensure_ascii=False, sort_keys=True,
                      separators=(",", ":")).encode("utf-8")


def write_once(path: Path, data: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        if path.read_bytes() != data:
            raise FileExistsError(f"immutable artifact conflict: {path}")
        return
    path.write_bytes(data)


def write_json_once(path: Path, value) -> None:
    write_once(path, json.dumps(value, ensure_ascii=False, indent=2,
                                sort_keys=True).encode("utf-8") + b"\n")


def image_data_url(path: Path) -> str:
    media = "image/png" if path.suffix.lower() == ".png" else "image/jpeg"
    return f"data:{media};base64," + base64.b64encode(path.read_bytes()).decode("ascii")


def render_overlay(image_path: Path, candidate: dict, output: Path) -> str:
    image = Image.open(image_path).convert("RGBA")
    layer = Image.new("RGBA", image.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(layer)
    for part in candidate.get("parts", []):
        points = [tuple(item) for item in part.get("points", [])]
        if len(points) >= 3:
            draw.polygon(points, fill=(255, 66, 66, 88), outline=(255, 0, 0, 255), width=3)
    draw.ellipse((384, 214, 396, 226), fill=(0, 255, 255, 255),
                 outline=(0, 0, 0, 255), width=2)
    combined = Image.alpha_composite(image, layer).convert("RGB")
    buffer = io.BytesIO()
    combined.save(buffer, format="PNG", optimize=False, compress_level=9)
    write_once(output, buffer.getvalue())
    return sha256(buffer.getvalue())


def review_schema(second: bool = False) -> dict:
    properties = {
        "candidate_class": {"enum": ["pallet", "person", "floor", "slope", "ego",
                                       "other_robot", "cargo_floor", "unknown"]},
        "known_class": {"type": "boolean"},
        "unknown_candidate": {"type": "boolean"},
        "new_label_suggestion": {"type": "string", "maxLength": 80},
        "mask_missing": {"enum": ["none", "minor", "major", "uncertain"]},
        "mask_excess": {"enum": ["none", "minor", "major", "uncertain"]},
        "mask_adhesion": {"enum": ["none", "minor", "major", "uncertain"]},
        "candidate_supported": {"type": "boolean"},
        "confidence": {"type": "number", "minimum": 0, "maximum": 1},
        "recommended_action": {"enum": sorted(ALLOWED_ACTIONS)},
        "reason": {"type": "string", "maxLength": 320},
    }
    if second:
        properties.update({
            "consistent_with_first_review": {"type": "boolean"},
            "final_route": {"enum": ["candidate_final", "human_review_required"]},
        })
    return {"type": "object", "additionalProperties": False,
            "required": list(properties), "properties": properties}


def vlm_payload(original: Path, overlay: Path, prompt: str, *, second: bool,
                model: str) -> dict:
    schema = review_schema(second)
    return {"model": model, "temperature": 0, "max_tokens": 384,
            "messages": [{"role": "user", "content": [
                {"type": "text", "text": prompt},
                {"type": "image_url", "image_url": {"url": image_data_url(original)}},
                {"type": "image_url", "image_url": {"url": image_data_url(overlay)}},
            ]}],
            "response_format": {"type": "json_schema", "json_schema": {
                "name": "fixed_holdout_second_review" if second else "fixed_holdout_first_review",
                "schema": schema, "strict": True}}}


def call_vlm(url: str, payload: dict) -> tuple[bytes, float]:
    request_data = canonical(payload)
    request = urllib.request.Request(url.rstrip("/") + "/v1/chat/completions",
        data=request_data, headers={"Content-Type": "application/json"}, method="POST")
    started = time.perf_counter()
    with urllib.request.urlopen(request, timeout=300) as response:
        raw = response.read()
        if response.status != 200:
            raise RuntimeError(f"VLM HTTP {response.status}: {raw[:500]!r}")
    return raw, round(time.perf_counter() - started, 3)


def structured_response(raw: bytes) -> tuple[dict, dict]:
    envelope = json.loads(raw)
    content = envelope["choices"][0]["message"]["content"]
    return envelope, json.loads(content)


def agent_output(action: str, sam_path: Path, sam: dict,
                 corrected_candidate: str | None) -> tuple[dict, bytes]:
    """Return the Agent candidate and artifact bytes; no_op is byte-preserving."""
    if action == "drop":
        candidate = {"holes": [], "parts": [], "points": [], "score": 0,
                     "warnings": ["agent_drop"]}
        return candidate, canonical(candidate)
    if action == "no_op":
        return sam, sam_path.read_bytes()
    if not corrected_candidate:
        raise ValueError(f"{action} requires --corrected-candidate")
    corrected = Path(corrected_candidate)
    return json.loads(corrected.read_text(encoding="utf-8")), corrected.read_bytes()


def workflow_route(candidate_state: str, *, human_pixel_truth_available: bool) -> str:
    if candidate_state not in {"candidate_final", "human_review_required"}:
        raise ValueError("candidate state is not routable")
    if not human_pixel_truth_available:
        return "human_review_required"
    return candidate_state


def phase_first(args, root: Path, image: Path, sam_path: Path, sam: dict) -> None:
    overlay = root / "sam-candidate-overlay.png"
    overlay_sha = render_overlay(image, sam, overlay)
    prompt = (
        "You are reviewing a machine segmentation candidate on a fixed real-factory 640x512 "
        "holdout image. The first image is the original; the second is a red candidate-mask "
        "overlay. The cyan point at (390,220) was selected by a Codex/assistant operator seed, "
        "not by a human annotator and not as truth. Known labels are pallet, person, floor, "
        "slope, ego, other_robot, cargo_floor. Inspect whether the candidate is slope or another "
        "known class, whether it suggests an unknown/new label, and whether the mask misses, "
        "over-covers, or sticks to adjacent objects. Recommend exactly one allowed Agent action. "
        "Slope is a distinct annotation/evaluation class and maps with floor to downstream "
        "traversable_area, but this candidate must not be treated as human truth."
    )
    payload = vlm_payload(image, overlay, prompt, second=False, model=args.vlm_model)
    request_bytes = canonical(payload)
    write_once(root / "vlm-first-request.json", request_bytes)
    raw, seconds = call_vlm(args.vlm_url, payload)
    write_once(root / "vlm-first-response.json", raw)
    envelope, decision = structured_response(raw)
    write_json_once(root / "vlm-first-decision.json", decision)
    receipt = {
        "phase": "first_vlm_review", "input_id": INPUT_ID, "frame_id": FRAME_ID,
        "camera": "camera_0", "holdout_excluded_from_training": True,
        "image_sha256": IMAGE_SHA256, "sam_candidate_sha256": SAM_SHA256,
        "operator_seed": {"x": 390, "y": 220, "label": 1,
                          "prompt_provenance": "assistant_selected_prompt",
                          "prompt_role": "operator_seed", "human_selected": False,
                          "human_truth": False},
        "overlay_sha256": overlay_sha,
        "request_sha256": sha256(request_bytes), "response_sha256": sha256(raw),
        "seconds": seconds, "usage": envelope.get("usage"),
        "local_vlm": envelope.get("local_vlm"), "decision": decision,
        "allowed_next_states": ["candidate_final", "human_review_required"],
        "forbidden_states": ["human_accepted", "truth_committed", "training_ready"],
    }
    write_json_once(root / "phase-first-receipt.json", receipt)
    print(json.dumps(receipt, ensure_ascii=False, indent=2))


def phase_second(args, root: Path, image: Path, sam_path: Path, sam: dict) -> None:
    if args.action not in ALLOWED_ACTIONS:
        raise ValueError(f"--action must be one of {sorted(ALLOWED_ACTIONS)}")
    first_path = (Path(args.first_receipt) if args.first_receipt
                  else root / "phase-first-receipt.json")
    first = json.loads(first_path.read_text(encoding="utf-8"))
    output_candidate, candidate_bytes = agent_output(
        args.action, sam_path, sam, args.corrected_candidate)
    write_once(root / "agent-output-candidate.json", candidate_bytes)
    action_record = {
        "action": args.action, "reason": args.reason,
        "input_artifact_sha256": sha256(sam_path.read_bytes()),
        "output_artifact_sha256": sha256(candidate_bytes),
        "first_vlm_response_sha256": first["response_sha256"],
        "first_vlm_recommended_action": first["decision"]["recommended_action"],
        "human_truth_used": False,
    }
    if args.prior_failed_response:
        prior = Path(args.prior_failed_response)
        action_record["prior_failed_second_response_sha256"] = sha256(prior.read_bytes())
        action_record["prior_failure"] = "structured JSON truncated at max_tokens; not accepted"
    write_json_once(root / "agent-action.json", action_record)
    overlay = root / "agent-output-overlay.png"
    overlay_sha = render_overlay(image, output_candidate, overlay)
    prompt = (
        "Independently re-review the fixed real-factory holdout. The first image is the original; "
        "the second is the Agent-output mask overlay after action '" + args.action + "'. "
        "Do not assume the first VLM was correct. Re-evaluate class, known/unknown status, mask "
        "missing/excess/adhesion, and whether the candidate is supported. candidate_final means "
        "machine candidate only, never human truth or training data. Route to human_review_required "
        "when any material defect, uncertainty, class conflict, or unknown suggestion remains."
    )
    payload = vlm_payload(image, overlay, prompt, second=True, model=args.vlm_model)
    request_bytes = canonical(payload)
    write_once(root / "vlm-second-request.json", request_bytes)
    raw, seconds = call_vlm(args.vlm_url, payload)
    write_once(root / "vlm-second-response.json", raw)
    envelope, decision = structured_response(raw)
    if args.action == "drop":
        decision["final_route"] = "human_review_required"
    write_json_once(root / "vlm-second-decision.json", decision)
    candidate_state = decision["final_route"]
    # A model may freeze a machine candidate, but missing human pixel truth
    # always routes this bounded holdout to the human gate next.
    final_route = workflow_route(candidate_state, human_pixel_truth_available=False)
    receipt = {
        "schema_version": "labelagent.fixed-holdout-chain.v1",
        "input_id": INPUT_ID, "dataset_id": "sfl-cpd15y",
        "source_batch": "20260920_cs-heli_Dev_000001/000",
        "frame_id": FRAME_ID, "camera": "camera_0", "source_kind": "real",
        "image_sha256": IMAGE_SHA256, "holdout_excluded_from_training": True,
        "label_type": "slope", "semantic_label": "slope_visible",
        "traversability_status": "passable", "policy_default": "passable",
        "policy_basis": "user_defined_project_semantics",
        "prompt_provenance": "assistant_selected_prompt", "human_truth": False,
        "sam_candidate_sha256": SAM_SHA256,
        "first_vlm_request_sha256": first["request_sha256"],
        "first_vlm_response_sha256": first["response_sha256"],
        "first_vlm_decision": first["decision"],
        "agent_action": action_record,
        "agent_output_overlay_sha256": overlay_sha,
        "second_vlm_request_sha256": sha256(request_bytes),
        "second_vlm_response_sha256": sha256(raw),
        "second_vlm_seconds": seconds, "second_vlm_usage": envelope.get("usage"),
        "second_local_vlm": envelope.get("local_vlm"),
        "second_vlm_decision": decision, "candidate_state": candidate_state,
        "final_state": final_route,
        "human_gate_passed": False, "truth_committed": False,
        "human_pixel_truth_available": False, "approved_for_training": False,
        "training_ready": False, "supervised_finetuning_started": False,
        "downstream_mapping": {"traversable_area": "floor UNION slope",
                               "training_labels_preserved": ["floor", "slope"]},
    }
    unsigned = canonical(receipt)
    receipt["receipt_sha256"] = sha256(unsigned)
    write_json_once(root / "final-chain-receipt.json", receipt)
    print(json.dumps(receipt, ensure_ascii=False, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--phase", choices=["first", "second"], required=True)
    parser.add_argument("--source-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--vlm-url", default="http://127.0.0.1:18878")
    parser.add_argument("--vlm-model", default="labelagent-vlm8b")
    parser.add_argument("--action", choices=sorted(ALLOWED_ACTIONS))
    parser.add_argument("--reason", default="")
    parser.add_argument("--corrected-candidate")
    parser.add_argument("--first-receipt")
    parser.add_argument("--prior-failed-response")
    args = parser.parse_args()
    source, root = Path(args.source_dir), Path(args.output_dir)
    image, sam_path = source / "input.png", source / "sam-result.json"
    if sha256(image.read_bytes()) != IMAGE_SHA256:
        raise ValueError("fixed holdout image hash mismatch")
    if sha256(sam_path.read_bytes()) != SAM_SHA256:
        raise ValueError("existing SAM candidate hash mismatch")
    sam = json.loads(sam_path.read_text(encoding="utf-8"))
    if args.phase == "first":
        phase_first(args, root, image, sam_path, sam)
    else:
        phase_second(args, root, image, sam_path, sam)


if __name__ == "__main__":
    main()
