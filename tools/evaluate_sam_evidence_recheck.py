"""Fresh bounded local VLM rechecks, retaining the complete original panel."""
import argparse
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
from tools.connect_local_vlm import require_real_service
from tools.evaluate_confirmed_region_context import check_idle
from tools.evaluate_source_availability import score
from tools.evaluate_target_cards import card_payload
from tools.evaluate_vlm_grounding import validated_response
from tools.run_local_agent_acceptance import save
from tools.sam_evidence_recheck import needs_recheck, rechecked_pair


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read(path):
    return json.loads(Path(path).read_bytes())


def evaluate(args):
    plan_path = Path(args.plan).resolve()
    if sha(plan_path) != args.plan_sha256:
        raise ValueError("Frozen plan differs")
    plan = read(plan_path)
    if plan["method"] != "unique-SAM-conflict-fresh-visual-recheck-v1" or not all(sha(p) == h for p, h in plan["source_bindings"].items()):
        raise ValueError("Method, source, code or state changed")
    fixture = read(plan["fixture"])
    previous = read(plan["previous_summary"])
    base_fixture = read(plan["original_fixture"])
    original_cases = {c["id"]: c for c in base_fixture["cases"]}
    previous_cases = {c["case_id"]: c for c in previous["cases"]}
    derived = []
    for c in previous["cases"]:
        a, b = (c["views"][v] for v in ("overview", "linked"))
        supported = (a.get("support") or {}).get("supported_classes", [])
        if needs_recheck([a["raw_category"], b["raw_category"]], supported):
            derived.append(c["case_id"])
    if len(base_fixture["cases"]) != 29 or derived != [c["id"] for c in fixture["cases"]] or len(derived) != 2:
        raise ValueError("Entire original panel and exact label-blind trigger set required")
    assets = {}
    for ref in [r for c in fixture["cases"] for refs in c["views"].values() for r in refs] + [fixture["guard_image"]]:
        path = Path(fixture["asset_root"]) / ref["path"]
        if sha(path) != ref["sha256"]:
            raise ValueError("Frozen evidence asset changed")
        assets[ref["path"]] = (path.read_bytes(), "image/jpeg")
    check_idle()
    output = Path(args.output).resolve(); output.mkdir(parents=True, exist_ok=False)
    summary = {"status": "starting", "pid": os.getpid(), "create_time": psutil.Process().create_time(),
        "attempted_requests": 0, "valid_real_CUDA_responses": 0, "rows": [], "guard": None,
        "new_SAM_calls": 0, "adopted": False, "default_enabled": False, "workbench_writes": False}
    save(output / "progress.json", summary)
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    def call(case, view, guard=False):
        check_idle()
        if (output / "STOP").exists():
            raise InterruptedError("Stop requested")
        body, digest = card_payload(fixture, assets, case, view, guard)
        number = summary["attempted_requests"] + 1
        row = {"number": number, "case_id": case["id"], "view": view, "guard": guard, "full_payload_sha256": digest}
        save(output / "request_inflight.json", row)
        summary["attempted_requests"] = number; save(output / "progress.json", summary)
        started = time.monotonic()
        request = urllib.request.Request("http://127.0.0.1:18868/v1/chat/completions", data=json.dumps(body).encode(), headers={"Content-Type": "application/json"})
        with opener.open(request, timeout=300) as response:
            raw = json.load(response)
        row["seconds"] = round(time.monotonic() - started, 3)
        save(output / f"response_{number:03d}.json", {**row, "raw_response": raw})
        (output / "request_inflight.json").unlink()
        parsed = validated_response(raw, fixture["categories"], 1)
        meta = raw["local_vlm"]
        assert meta["model_config_sha256"] == fixture["model_config_sha256"] and meta["quantization"] == "nf4"
        assert meta["effective_image_token_budget"] == (256 if guard else fixture["image_token_budgets"][view])
        assert meta["actual_image_token_counts"] == (fixture["guard_expected_actual_tokens"] if guard else [fixture["image_token_budgets"][view]])
        row.update(parsed=parsed, actual_image_token_counts=meta["actual_image_token_counts"], image_grid_thw=meta["image_grid_thw"])
        summary["valid_real_CUDA_responses"] += 1
        if guard:
            summary["guard"] = row
        else:
            summary["rows"].append(row)
        save(output / "progress.json", summary)
        print(json.dumps({"completed": number, "planned": 5, "case_id": case["id"], "view": view,
                          "category": parsed["category"], "guard": guard}), flush=True)
    try:
        health = require_real_service("http://127.0.0.1:18868/v1", fixture["model"])
        assert health["model_config_sha256"] == fixture["model_config_sha256"] and health["image_token_budget"] == 256
        assert health["supports_single_image_budget_override"]
        save(output / "backend_before_PRIVATE.json", health)
        summary["status"] = "running"
        for case in fixture["cases"]:
            for view in ("overview", "linked"):
                call(case, view)
        call({"id": "default_budget_guard"}, "guard", guard=True)
        summary["status"] = "complete"
    except InterruptedError:
        summary["status"] = "stopped"
    except Exception as error:
        summary.update(status="failed", error_type=type(error).__name__)
        (output / "local_error.txt").write_text(traceback.format_exc(), encoding="utf-8")
    finally:
        save(output / "progress.json", summary)
        indexed = {(r["case_id"], r["view"]): r["parsed"] for r in summary["rows"]}
        final_rows, outcomes, fresh_opposite = [], [], []
        for old in previous["cases"]:
            pair = [old["views"][v]["category"] for v in ("overview", "linked")]
            if old["case_id"] in derived:
                fresh = [indexed.get((old["case_id"], v)) for v in ("overview", "linked")]
                supported = old["views"]["overview"]["support"]["supported_classes"]
                decision = rechecked_pair(pair, supported, [r["category"] for r in fresh]) if all(fresh) else {
                    "categories": pair, "fresh_pair_matched_unique_SAM_support": False}
                outcomes.append({"case_id": old["case_id"], "before": pair, "fresh": fresh, **decision})
                pair = decision["categories"]
                truth = original_cases[old["case_id"]]
                expected = truth.get("expected_category") if truth["group"] == "human" else (
                    "pallet" if truth["role"] == "pallet_fixture" else "person" if truth["role"] == "person_fixture" else None)
                for view, r in zip(("overview", "linked"), fresh):
                    if r and expected and r["category"] not in (expected, "uncertain"):
                        fresh_opposite.append({"case_id": old["case_id"], "view": view, "category": r["category"]})
            for view, category in zip(("overview", "linked"), pair):
                final_rows.append({"case_id": old["case_id"], "view": view,
                                   "parsed": {"category": category, "reason": "guarded prior or fresh paired evidence"}})
        scores = score(base_fixture, final_rows)
        old_correct = previous["scores"]["human_correct_ids"]
        legacy_correct = sum(r["role"] == "pallet_fixture" and r["consensus"] == "pallet" for r in scores["legacy_pairs"])
        complete = summary["status"] == "complete" and summary["valid_real_CUDA_responses"] == 5
        checks = {**scores["original_quality_checks"], "all5_new_responses_valid_real_CUDA": complete,
            "all14_previous_human_correct_retained": not (set(old_correct) - set(scores["human_correct_ids"])),
            "zero_wrong_human_consensus": not scores["human_wrong_consensus_ids"],
            "no_fresh_explicit_opposite_class_view": not fresh_opposite,
            "strict_categorical_gain_vs_previous_guard": scores["human_correct_consensus"] > 14 or legacy_correct > 2,
            "sensitive_default_budget_guard_passed": bool(summary["guard"] and summary["guard"]["actual_image_token_counts"] == fixture["guard_expected_actual_tokens"]),
            "all_source_code_config_state_bindings_unchanged": all(sha(p) == h for p, h in plan["source_bindings"].items()),
            "plan_unchanged": sha(plan_path) == args.plan_sha256}
        result = {**summary, "finished_utc": datetime.now(timezone.utc).isoformat(), "plan_sha256": args.plan_sha256,
            "all_original_cases": 29, "outcomes": outcomes, "scores": scores,
            "legacy_pallet_correct_consensus": legacy_correct, "fresh_opposite_class_views": fresh_opposite,
            "checks": checks, "all_checks_passed": all(checks.values()), "posthoc_development_study": True,
            "independent_validation": False, "pixel_accuracy_measured": False}
        save(output / "summary.json", result)
    print(json.dumps({"status": result["status"], "human_correct": scores["human_correct_consensus"],
        "human_wrong": scores["human_wrong_consensus_ids"], "legacy_pallet_correct": legacy_correct,
        "fresh_opposite_views": fresh_opposite, "checks": checks}), flush=True)
    return 0 if result["status"] == "complete" else 1


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", required=True)
    parser.add_argument("--plan-sha256", required=True)
    parser.add_argument("--output", required=True)
    raise SystemExit(evaluate(parser.parse_args()))
