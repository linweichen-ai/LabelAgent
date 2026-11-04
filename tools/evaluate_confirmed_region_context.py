"""Paired local VLM diagnostic on category-confirmed, model-anchored regions.

The single cutout is shared. Only the source/outlined context crop changes.
Expected categories and human approval records never enter model requests.
"""
import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import io
import json
import math
import os
from pathlib import Path
import sys
import time
import traceback
import urllib.request

import cv2
import numpy as np
import psutil
from PIL import Image, ImageDraw

REPO = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(REPO), str(REPO / "tools")]
from source.agent.edit_agent import _rasterize
from tools.connect_local_vlm import loopback_url, request_json, require_real_service
from tools.evaluate_vlm_grounding import build_payload, validated_response
from tools.run_local_agent_acceptance import save


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def expanded_context_views(image_bytes, shape, minimum_side=160, expansion=1.5):
    """Native source pixels, larger class-blind context and selection outline."""
    if type(minimum_side) is not int or minimum_side < 1 or not 1 <= expansion <= 3:
        raise ValueError("Invalid fixed context geometry")
    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    mask = _rasterize(shape, *image.size)
    yy, xx = np.nonzero(mask)
    if not len(xx):
        raise ValueError("Selection has no pixels")
    side = max(minimum_side, math.ceil(expansion * max(int(xx.max() - xx.min() + 1), int(yy.max() - yy.min() + 1))))
    cx, cy = (int(xx.min()) + int(xx.max()) + 1) / 2, (int(yy.min()) + int(yy.max()) + 1) / 2
    left = max(0, min(math.floor(cx - side / 2), image.width - side))
    top = max(0, min(math.floor(cy - side / 2), image.height - side))
    right, bottom = min(image.width, left + side), min(image.height, top + side)
    raw = image.crop((left, top, right, bottom))
    cropped = mask[top:bottom, left:right]
    outlined = raw.copy()
    draw = ImageDraw.Draw(outlined)
    contours, _ = cv2.findContours(cropped, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    for contour in contours:
        points = [tuple(p) for p in contour.reshape(-1, 2).tolist()]
        if len(points) >= 2:
            draw.line(points + points[:1], fill=(255, 0, 255), width=2)
    assert int(cropped.sum()) == int(mask.sum())
    return [raw, outlined], {"crop_box": [left, top, right, bottom], "selected_pixels": int(mask.sum()),
                            "image_resized": False, "class_or_ID_text_rendered": False}


def aggregate(cases, rows):
    by_case = {(r["case_id"], r["view"]): r for r in rows}
    result = {}
    for policy, view in (("tight_context", "context"), ("expanded_context", "expanded")):
        outcomes = []
        for case in cases:
            a, b = by_case.get((case["id"], "single")), by_case.get((case["id"], view))
            valid = bool(a and b and a.get("parsed") and b.get("parsed"))
            consensus = a["parsed"]["category"] if valid and a["parsed"]["category"] == b["parsed"]["category"] else "uncertain"
            outcomes.append({"case_id": case["id"], "expected_category": case["expected_category"],
                             "valid_pair": valid, "consensus": consensus,
                             "correct_consensus": valid and consensus == case["expected_category"]})
        result[policy] = {"cases": len(cases), "valid_pairs": sum(o["valid_pair"] for o in outcomes),
                          "correct_consensus": sum(o["correct_consensus"] for o in outcomes),
                          "consensus_counts": dict(Counter(o["consensus"] for o in outcomes)),
                          "by_confirmed_category": {}, "outcomes": outcomes}
        for label in ("pallet", "person"):
            group = [o for o in outcomes if o["expected_category"] == label]
            view_rows = [r for r in rows if r["view"] in ("single", view) and any(c["id"] == r["case_id"] and c["expected_category"] == label for c in cases)]
            result[policy]["by_confirmed_category"][label] = {
                "cases": len(group), "correct_consensus": sum(o["correct_consensus"] for o in group),
                "wrong_opposite_class_views": sum(bool(r.get("parsed")) and r["parsed"]["category"] not in (label, "uncertain") for r in view_rows),
                "uncertain_consensus": sum(o["consensus"] == "uncertain" for o in group)}
    return result


def check_idle():
    state = request_json("http://127.0.0.1:8765/api/run/status")
    if state.get("status") not in ("idle", "completed"):
        raise InterruptedError("Workbench active: no further requests started")
    clients = [c for c in psutil.net_connections(kind="tcp") if c.status == "ESTABLISHED" and c.raddr and c.raddr.port in (8766, 18868) and c.pid != os.getpid()]
    if clients:
        raise InterruptedError("Another model client is connected: defer instead of switching it")


def evaluate(args):
    fixture_path = Path(args.fixture_json).resolve()
    fixture = json.loads(fixture_path.read_bytes())
    asset_root = Path(fixture["asset_root"]).resolve()
    if fixture["categories"] != ["pallet", "person", "uncertain"] or fixture["temperature"] != 0 or fixture["max_tokens"] != 192:
        raise ValueError("Frozen vocabulary and decoding required")
    cases = fixture["cases"]
    if len({c["id"] for c in cases}) != len(cases) or not cases:
        raise ValueError("Distinct nonempty confirmed cases required")
    assets, bindings = {}, {str(fixture_path): sha(fixture_path)}
    for case in cases:
        if case["expected_category"] not in ("pallet", "person") or set(case["views"]) != {"single", "context", "expanded"}:
            raise ValueError("Unsupported category-confirmed case")
        if case["views"]["context"][0] != case["views"]["expanded"][0] or case["views"]["single"][0] != case["views"]["context"][0]:
            raise ValueError("Cutout must be byte-identical across policies")
        for kind, refs in case["views"].items():
            if len(refs) != (1 if kind == "single" else 3):
                raise ValueError("One cutout or three images required")
            for ref in refs:
                p = (asset_root / ref["path"]).resolve()
                if not p.is_relative_to(asset_root) or sha(p) != ref["sha256"]:
                    raise ValueError("Unbound image asset")
                with Image.open(p) as image:
                    if image.format != "JPEG":
                        raise ValueError("Fixture images must be frozen JPEGs")
                    image.verify()
                assets[ref["path"]] = (p.read_bytes(), "image/jpeg")
                bindings[str(p)] = ref["sha256"]
    for p, digest in fixture["approval_bindings"].items():
        if sha(p) != digest:
            raise ValueError("Human confirmation binding changed")
        bindings[p] = digest
    for path in (Path(__file__), REPO / "tools/evaluate_vlm_grounding.py", REPO / "tools/connect_local_vlm.py", REPO / "tools/local_vlm_server.py"):
        bindings[str(path.resolve())] = sha(path)
    output = Path(args.output).resolve()
    if output == asset_root or output.is_relative_to(asset_root):
        raise ValueError("Keep output separate from frozen fixture")
    check_idle()
    output.mkdir(parents=True, exist_ok=False)
    state_path = REPO / "work/state.json"
    state_before = sha(state_path)
    plan = {"created_utc": datetime.now(timezone.utc).isoformat(), "planned_requests": 3 * len(cases),
            "case_count": len(cases), "factor": "native context crop geometry only; shared cutout and unchanged prompts/schema/model",
            "image_token_budget": 256, "expected_model_config_sha256": fixture["model_config_sha256"],
            "input_code_approval_bindings": bindings, "case_order": [c["id"] for c in cases],
            "request_order": "single first; tight/expanded context order alternates by case",
            "old_functional_controls_remain_failed": True,
            "adoption_possible_from_this_diagnostic_alone": False,
            "predeclared_checks": ["all requests valid real CUDA", "strict correct-consensus gain", "no previously correct region lost", "no opposite-class view increase", "no new opposite-class context error at any region"],
            "independent_pixel_accuracy": False, "workbench_writes": False, "new_SAM_calls": 0}
    save(output / "study_plan.json", plan)
    plan_sha = sha(output / "study_plan.json")
    summary = {"status": "loading_model", "pid": os.getpid(), "process_create_time": psutil.Process().create_time(),
               "attempted_requests": 0, "responses_received": 0, "valid_real_CUDA_responses": 0,
               "invalid_model_outputs": 0, "rows": [], "model_calls_repeated": 0,
               "workbench_writes": False, "adopted": False, "default_enabled": False}
    save(output / "summary.json", summary)
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    try:
        check_idle()
        health = require_real_service(loopback_url(args.base), fixture["model"])
        if health.get("model_config_sha256") != fixture["model_config_sha256"] or health.get("image_token_budget") != 256 or health.get("quantization") != "nf4":
            raise ValueError("Unexpected model identity/settings")
        save(output / "backend_before.json", health)
        summary["status"] = "running"
        save(output / "summary.json", summary)
        for index, case in enumerate(cases):
            order = ("single", "context", "expanded") if index % 2 == 0 else ("single", "expanded", "context")
            for kind in order:
                if (output / "STOP").exists():
                    raise InterruptedError("Stop requested")
                check_idle()
                view_case = {"views": {"single": case["views"][kind]}}
                local_fixture = {**fixture, "prompts": {"single": fixture["prompts"]["single" if kind == "single" else "context"]}}
                body, fingerprint = build_payload(local_fixture, assets, view_case, "single", fixture["model"])
                assert set(body) == {"model", "temperature", "max_tokens", "messages", "response_format"}
                number = summary["attempted_requests"] + 1
                pending = {"number": number, "case_id": case["id"], "view": kind, "payload_sha256": hashlib.sha256(json.dumps(body, sort_keys=True).encode()).hexdigest()}
                save(output / "request_inflight.json", pending)
                summary["attempted_requests"] = number
                save(output / "summary.json", summary)
                started = time.monotonic()
                req = urllib.request.Request(args.base.rstrip("/") + "/chat/completions", json.dumps(body).encode(), headers={"Content-Type": "application/json"})
                with opener.open(req, timeout=300) as response:
                    raw = json.load(response)
                row = {"case_id": case["id"], "view": kind, "seconds": round(time.monotonic() - started, 3),
                       "input_fingerprint": fingerprint, "payload_sha256": pending["payload_sha256"]}
                # Persist every accepted response before parsing; never repeat invalid output.
                save(output / f"response_{number:03d}.json", {**row, "raw_response": raw})
                (output / "request_inflight.json").unlink()
                summary["responses_received"] += 1
                try:
                    row["parsed"] = validated_response(raw, fixture["categories"], len(case["views"][kind]))
                    meta = raw["local_vlm"]
                    if meta.get("model_config_sha256") != fixture["model_config_sha256"] or meta.get("quantization") != "nf4" or meta.get("effective_image_token_budget") != 256:
                        raise ValueError("Response model identity differs")
                    tokens = meta.get("actual_image_token_counts")
                    if not isinstance(tokens, list) or len(tokens) != len(case["views"][kind]) or not all(type(v) is int and v > 0 for v in tokens):
                        raise ValueError("Actual image encoding evidence missing")
                    row["actual_image_token_counts"] = tokens
                    summary["valid_real_CUDA_responses"] += 1
                except (ValueError, KeyError, TypeError):
                    row.pop("parsed", None)
                    row["invalid_output"] = True
                    summary["invalid_model_outputs"] += 1
                summary["rows"].append(row)
                save(output / "summary.json", summary)
                print(json.dumps({"completed": summary["responses_received"], "planned": plan["planned_requests"],
                                  "case_id": case["id"], "view": kind, "category": row.get("parsed", {}).get("category"), "invalid": row.get("invalid_output", False)}), flush=True)
        summary["status"] = "complete"
    except InterruptedError:
        summary["status"] = "stopped"
    except Exception as error:
        summary.update(status="failed", error_type=type(error).__name__)
        (output / "local_error.txt").write_text(traceback.format_exc(), encoding="utf-8")
    finally:
        summary["policy_results"] = aggregate(cases, summary["rows"])
        a, b = (summary["policy_results"][p] for p in ("tight_context", "expanded_context"))
        complete = summary["status"] == "complete" and summary["valid_real_CUDA_responses"] == plan["planned_requests"]
        indexed = {(r["case_id"], r["view"]): r for r in summary["rows"]}
        new_wrong_context = []
        if complete:
            for case in cases:
                tight = indexed[case["id"], "context"]["parsed"]["category"]
                expanded = indexed[case["id"], "expanded"]["parsed"]["category"]
                allowed = (case["expected_category"], "uncertain")
                if expanded not in allowed and tight in allowed:
                    new_wrong_context.append(case["id"])
        summary["new_opposite_class_context_error_cases"] = new_wrong_context
        summary["checks"] = {"all_requests_valid_real_CUDA": complete,
                             "strict_correct_consensus_gain": bool(complete and b["correct_consensus"] > a["correct_consensus"]),
                             "no_previously_correct_region_lost": bool(complete and all(not x["correct_consensus"] or y["correct_consensus"] for x, y in zip(a["outcomes"], b["outcomes"]))),
                             "no_opposite_class_view_increase": bool(complete and all(b["by_confirmed_category"][c]["wrong_opposite_class_views"] <= a["by_confirmed_category"][c]["wrong_opposite_class_views"] for c in ("pallet", "person"))),
                             "no_new_opposite_class_context_error_at_any_region": bool(complete and not new_wrong_context)}
        summary["state_unchanged"] = sha(state_path) == state_before
        summary["input_code_approval_bindings_unchanged"] = all(sha(p) == h for p, h in bindings.items())
        summary["plan_unchanged"] = sha(output / "study_plan.json") == plan_sha
        summary["all_predeclared_checks_passed"] = all(summary["checks"].values()) and summary["input_code_approval_bindings_unchanged"] and summary["plan_unchanged"] and summary["state_unchanged"]
        summary["finished_utc"] = datetime.now(timezone.utc).isoformat()
        save(output / "summary.json", summary)
    print(json.dumps({"status": summary["status"], "valid_responses": summary["valid_real_CUDA_responses"],
                      "tight_correct": a["correct_consensus"], "expanded_correct": b["correct_consensus"], "checks": summary["checks"]}), flush=True)
    return 0 if summary["status"] == "complete" else 1


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fixture-json", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--base", default="http://127.0.0.1:18868/v1")
    raise SystemExit(evaluate(parser.parse_args()))
