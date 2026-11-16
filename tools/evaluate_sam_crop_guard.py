"""Collect paired SAM crop evidence, or replay its frozen complete class panel."""
import argparse
import base64
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import sys
import time
import traceback
import urllib.request

import psutil

REPO = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(REPO), str(REPO / "tools")]
from source.agent.edit_agent import _rasterize
from tools.connect_local_vlm import require_real_service
from tools.evaluate_confirmed_region_context import check_idle
from tools.evaluate_source_availability import score
from tools.evaluate_vlm_grounding import validated_response
from tools.run_local_agent_acceptance import save
from tools.sam_class_evidence import cached_class_masks, guard_proposal, mask_support
from tools.source_image_evidence import guard_category, inspect_source


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read(path):
    return json.loads(Path(path).read_bytes())


def checked_plan(path, digest):
    if sha(path) != digest:
        raise ValueError("Frozen plan differs")
    plan = read(path)
    if plan["method"] != "paired-same-source-crop-SAM-plus-fullframe-v1" or not all(sha(p) == h for p, h in plan["source_bindings"].items()):
        raise ValueError("Method/input/code/state binding differs")
    return plan


def collect(args):
    plan = checked_plan(args.plan, args.plan_sha256)
    check_idle()
    output = Path(args.output).resolve(); output.mkdir(parents=True, exist_ok=False)
    summary = {"status": "running", "pid": os.getpid(), "create_time": psutil.Process().create_time(),
               "attempted_requests": 0, "successful_requests": 0, "planned_requests": 54,
               "rows": [], "workbench_writes": False, "new_VLM_calls": 0}
    save(output / "progress.json", summary)
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    try:
        for index, case in enumerate(plan["crop_cases"]):
            raw = Path(case["unmarked_source"]["path"]).read_bytes()
            assert hashlib.sha256(raw).hexdigest() == case["unmarked_source"]["sha256"]
            for category in (("pallet", "person") if index % 2 == 0 else ("person", "pallet")):
                if (output / "STOP").exists():
                    raise InterruptedError("Stop requested")
                check_idle()
                body = {"image": base64.b64encode(raw).decode("ascii"), "prompt": category}
                number = summary["attempted_requests"] + 1
                pending = {"number": number, "case_id": case["id"], "category": category,
                           "request_body_sha256": hashlib.sha256(json.dumps(body).encode()).hexdigest()}
                save(output / "request_inflight.json", pending)
                summary["attempted_requests"] = number; save(output / "progress.json", summary)
                started = time.monotonic()
                request = urllib.request.Request("http://127.0.0.1:8766/segment_text", data=json.dumps(body).encode(), headers={"Content-Type": "application/json"})
                with opener.open(request, timeout=180) as response:
                    payload = json.load(response)
                receipt = {**pending, "input_sha256": case["unmarked_source"]["sha256"],
                           "frame": case["frame"], "camera": case["camera"], "label": category, "prompt": category,
                           "source_image_sha256": case["image_sha256"], "crop_box": case["crop_box"],
                           "status": "ok", "masks": len(payload["results"]), "response": payload,
                           "seconds": round(time.monotonic() - started, 4), "observed_utc": datetime.now(timezone.utc).isoformat()}
                cached_class_masks(receipt, image_sha256=case["unmarked_source"]["sha256"], frame=case["frame"],
                                   camera=case["camera"], size=case["size"], category=category)
                save(output / f"response_{number:03d}.json", receipt)
                (output / "request_inflight.json").unlink()
                summary["rows"].append({**pending, "seconds": receipt["seconds"], "masks": receipt["masks"]})
                summary["successful_requests"] += 1; save(output / "progress.json", summary)
                print(json.dumps({"completed": summary["successful_requests"], "planned": 54, "case_id": case["id"], "category": category, "masks": receipt["masks"]}), flush=True)
        summary["status"] = "complete"
    except InterruptedError:
        summary["status"] = "stopped"
    except Exception as error:
        summary.update(status="failed", error_type=type(error).__name__)
        (output / "local_error.txt").write_text(traceback.format_exc(), encoding="utf-8")
    finally:
        summary["finished_utc"] = datetime.now(timezone.utc).isoformat()
        summary["source_bindings_unchanged"] = all(sha(p) == h for p, h in plan["source_bindings"].items())
        save(output / "progress.json", summary)
    if summary["status"] == "complete":
        check_idle()
        health = require_real_service("http://127.0.0.1:18868/v1", "labelagent-vlm")
        assert health["image_token_budget"] == 256
        save(output / "restored_VLM_health_PRIVATE.json", health)
    return 0 if summary["status"] == "complete" and summary["source_bindings_unchanged"] else 1


def evaluate(args):
    plan = checked_plan(args.plan, args.plan_sha256)
    collection = Path(args.collection).resolve()
    progress = read(collection / "progress.json")
    if progress["status"] != "complete" or progress["successful_requests"] != 54 or progress["attempted_requests"] != 54:
        raise ValueError("Require complete54 paired SAM crop calls")
    responses = {(r["case_id"], r["category"]): collection / f"response_{r['number']:03d}.json" for r in progress["rows"]}
    if len(responses) != 54:
        raise ValueError("Distinct paired cache records required")
    output = Path(args.output).resolve(); output.mkdir(parents=True, exist_ok=False)
    full_plan = read(plan["full_guard_plan"])
    base = read(full_plan["source_guard_plan"])
    fixture = read(base["fixture_path"])
    targets = {r["id"]: r for r in read(full_plan["target_fixture"])["targets"]}
    crops = {r["id"]: r for r in plan["crop_cases"]}
    rows, details = [], []
    bindings = {str(collection / "progress.json"): sha(collection / "progress.json")}
    bindings.update({str(p): sha(p) for p in responses.values()})
    for case in base["cases"]:
        source = inspect_source(Path(case["unmarked_source"]["path"]).read_bytes())
        support = None
        if source.available:
            target, crop = targets[case["id"]], crops[case["id"]]
            x1, y1, x2, y2 = crop["crop_box"]
            full_target = _rasterize(target["geometry"], *target["size"]).astype(bool)
            target_mask = full_target[y1:y2, x1:x2]
            assert target_mask.sum() == full_target.sum()
            masks = {}
            for category in ("pallet", "person"):
                full = cached_class_masks(read(target["caches"][category]["path"]), image_sha256=target["image_sha256"], frame=target["frame"], camera=target["camera"], size=target["size"], category=category)
                local = cached_class_masks(read(responses[case["id"], category]), image_sha256=crop["unmarked_source"]["sha256"], frame=crop["frame"], camera=crop["camera"], size=crop["size"], category=category)
                masks[category] = [m[y1:y2, x1:x2] for m in full] + local
            support = mask_support(target_mask, masks)
        record = {"case_id": case["id"], "views": {}}
        for view in ("overview", "linked"):
            raw = read(case["responses"][view]["path"])["raw_response"]
            parsed = validated_response(raw, fixture["categories"], 1)
            first = guard_category(parsed["category"], source)
            decision = guard_proposal(first["category"], support) if support else {"category": "uncertain", "support": None}
            rows.append({"case_id": case["id"], "view": view, "parsed": {**parsed, "category": decision["category"]}})
            record["views"][view] = {"raw_category": parsed["category"], **decision}
        details.append(record)
    scores = score(fixture, rows)
    full = read(plan["full_guard_summary"])
    lost = sorted(set(full["guarded_scores"]["human_correct_ids"]) - set(scores["human_correct_ids"]))
    quality = {**scores["original_quality_checks"], "all14_card_correct_consensus_retained": not lost,
               "zero_wrong_human_consensus": not scores["human_wrong_consensus_ids"]}
    integrity = {"all29_cases_retained": len(details) == 29, "all54_new_calls_bound": len(bindings) == 55,
                 "sources_and_code_unchanged": all(sha(p) == h for p, h in plan["source_bindings"].items()),
                 "response_files_unchanged": all(sha(p) == h for p, h in bindings.items()),
                 "plan_unchanged": sha(args.plan) == args.plan_sha256}
    result = {"status": "complete", "method": plan["method"], "finished_utc": datetime.now(timezone.utc).isoformat(),
               "plan_sha256": args.plan_sha256, "cases": details, "scores": scores, "lost_correct": lost,
               "quality_checks": quality, "all_quality_checks_passed": all(quality.values()),
               "integrity_checks": integrity, "all_integrity_checks_passed": all(integrity.values()),
               "new_SAM_calls": 54, "new_VLM_calls": 0, "cached_VLM_responses_reused": 58,
               "workbench_writes": False, "default_enabled": False, "adopted": False,
               "posthoc_development_study": True, "independent_validation": False}
    save(output / "response_bindings_PRIVATE.json", bindings)
    save(output / "summary.json", result)
    print(json.dumps({"human_correct": scores["human_correct_consensus"], "human_wrong": scores["human_wrong_consensus_ids"], "lost_correct": lost, "legacy_checks": scores["legacy_checks"], "quality_checks": quality, "integrity": integrity}))
    return 0 if all(integrity.values()) else 1


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("collect", "score"))
    parser.add_argument("--plan", required=True)
    parser.add_argument("--plan-sha256", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--collection")
    args = parser.parse_args()
    raise SystemExit(collect(args) if args.mode == "collect" else evaluate(args))
