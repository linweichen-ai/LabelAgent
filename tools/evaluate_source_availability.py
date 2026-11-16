"""Replay a frozen source-availability plan against immutable cached responses.

No network requests, annotation changes or automatic class corrections occur.
This is a posthoc development replay, not a new independent VLM experiment.
"""
import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys

REPO = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(REPO), str(REPO / "tools")]
from tools.evaluate_instance_focus import focus_checks
from tools.evaluate_target_cards import pair_rows
from tools.evaluate_vlm_grounding import validated_response
from tools.source_image_evidence import POLICY, guard_category, inspect_source


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read(path):
    return json.loads(Path(path).read_bytes())


def score(fixture, rows):
    pairs = pair_rows(fixture["cases"], rows)
    cases = {c["id"]: c for c in fixture["cases"]}
    human = [r for r in pairs if r["group"] == "human"]
    correct = [r["id"] for r in human if r["valid_pair"] and r["consensus"] == cases[r["id"]]["expected_category"]]
    wrong = [r["id"] for r in human if r["consensus"] not in ("uncertain", cases[r["id"]]["expected_category"])]
    legacy = [{**r, "id": cases[r["id"]]["source_id"], "role": cases[r["id"]]["role"]}
              for r in pairs if r["group"] == "legacy"]
    valid = sum(bool(r["views"][v]["parsed"]) for r in legacy for v in ("single", "context"))
    new_opposite = []
    for row in rows:
        case = cases[row["case_id"]]
        if case["group"] == "human":
            allowed = (case["expected_category"], "uncertain")
            previous = fixture["baseline_human_view_categories"][case["id"]][row["view"]]
            if row["parsed"]["category"] not in allowed and previous in allowed:
                new_opposite.append({"case_id": case["id"], "view": row["view"]})
    checks = focus_checks(fixture["legacy_requirements"], legacy, valid)
    all_people = all(r["id"] in correct for r in human if cases[r["id"]]["expected_category"] == "person")
    original_quality_checks = {"human_consensus_strictly_above11": len(correct) > 11,
                               "no_previous_correct_human_region_lost": not (set(fixture["baseline_human_correct_ids"]) - set(correct)),
                               "all5_confirmed_people_correct": all_people,
                               "no_new_opposite_class_view_error": not new_opposite,
                               "all_original_legacy_checks_passed": all(checks.values())}
    return {"human_correct_consensus": len(correct), "human_correct_ids": correct,
            "human_wrong_consensus_ids": wrong, "human_pairs": human,
            "new_opposite_class_view_errors": new_opposite,
            "legacy_pairs": legacy, "legacy_checks": checks,
            "original_quality_checks": original_quality_checks,
            "all_original_quality_checks_passed": all(original_quality_checks.values())}


def evaluate(plan_path, expected_sha, output):
    plan_path, output = Path(plan_path).resolve(), Path(output).resolve()
    if sha(plan_path) != expected_sha:
        raise ValueError("Frozen plan digest differs")
    plan = read(plan_path)
    if plan["policy"] != POLICY or plan["posthoc_development_replay"] is not True:
        raise ValueError("Unexpected replay policy")
    if not all(sha(p) == h for p, h in plan["source_bindings"].items()):
        raise ValueError("Frozen input/code/config binding differs")
    fixture = read(plan["fixture_path"])
    cases = plan["cases"]
    if (len(cases) != 29 or len({c["id"] for c in cases}) != 29
            or [c["id"] for c in cases] != [c["id"] for c in fixture["cases"]]
            or Counter(c["group"] for c in fixture["cases"]) != {"human": 19, "legacy": 10}):
        raise ValueError("All original29 cases must remain")
    output.mkdir(parents=True, exist_ok=False)
    raw_rows, guarded_rows, details, response_ids = [], [], [], []
    for case in cases:
        source = case["unmarked_source"]
        if sha(source["path"]) != source["sha256"]:
            raise ValueError("Source crop digest differs")
        evidence = inspect_source(Path(source["path"]).read_bytes())
        record = {"case_id": case["id"], "views": {}}
        for view in ("overview", "linked"):
            ref = case["responses"][view]
            if sha(ref["path"]) != ref["sha256"]:
                raise ValueError("Cached response digest differs")
            saved = read(ref["path"])
            if saved["case_id"] != case["id"] or saved["view"] != view or saved["guard"]:
                raise ValueError("Response case/view binding differs")
            raw = saved["raw_response"]
            parsed = validated_response(raw, fixture["categories"], 1)
            meta = raw["local_vlm"]
            if meta["model_config_sha256"] != fixture["model_config_sha256"] or meta["quantization"] != "nf4":
                raise ValueError("Cached model identity differs")
            response_ids.append(raw["id"])
            decision = guard_category(parsed["category"], evidence)
            raw_rows.append({"case_id": case["id"], "view": view, "parsed": parsed})
            guarded_rows.append({"case_id": case["id"], "view": view,
                                 "parsed": {"category": decision["category"], "reason": evidence.reason}})
            record["views"][view] = decision
        details.append(record)
    if len(set(response_ids)) != 58:
        raise ValueError("Responses not unique")
    raw_score, guarded_score = score(fixture, raw_rows), score(fixture, guarded_rows)
    original_summary = read(plan["original_summary_path"])
    if (raw_score["human_correct_ids"] != original_summary["human_correct_ids"]
            or raw_score["legacy_checks"] != original_summary["legacy_checks"]
            or raw_score["new_opposite_class_view_errors"] != original_summary["new_opposite_class_view_errors"]):
        raise ValueError("Raw replay does not reproduce original results")
    checks = {"all29_cases_and58_quality_responses_replayed": len(details) == 29 and len(raw_rows) == 58,
              "raw_results_reproduced": True,
              "all_available_source_predictions_unchanged": all(
                  not d["changed"] for r in details for d in r["views"].values() if d["source_evidence"]["available"]),
              "no_new_confident_class_generated": all(
                  not d["changed"] or d["category"] == "uncertain" for r in details for d in r["views"].values()),
              "no_correct_human_region_lost_vs_raw_cards": not (set(raw_score["human_correct_ids"]) - set(guarded_score["human_correct_ids"])),
              "input_code_config_annotation_bindings_unchanged": all(sha(p) == h for p, h in plan["source_bindings"].items()),
              "plan_unchanged": sha(plan_path) == expected_sha}
    result = {"status": "complete", "finished_utc": datetime.now(timezone.utc).isoformat(),
              "policy": POLICY, "plan_sha256": expected_sha, "posthoc_development_replay": True,
              "new_model_calls": 0, "new_SAM_calls": 0, "raw_cached_quality_responses_reused": 58,
              "original_guard_response_preserved_not_quality_scored": True,
              "cases": details, "raw_scores": raw_score, "guarded_scores": guarded_score,
              "availability_checks": checks, "all_availability_checks_passed": all(checks.values()),
              "adopted": False, "default_enabled": False, "workbench_writes": False,
              "independent_validation": False, "pixel_accuracy_measured": False}
    (output / "summary.json").write_bytes((json.dumps(result, indent=2) + "\n").encode())
    print(json.dumps({"status": "complete", "availability_checks": checks,
                      "human_correct": guarded_score["human_correct_consensus"],
                      "legacy_checks": guarded_score["legacy_checks"],
                      "all_original_quality_checks_passed": guarded_score["all_original_quality_checks_passed"]}))
    return 0 if all(checks.values()) else 1


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", required=True)
    parser.add_argument("--plan-sha256", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    raise SystemExit(evaluate(args.plan, args.plan_sha256, args.output))
