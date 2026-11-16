"""Offline complete-panel replay of source and bound cached SAM class guards."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys

REPO = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(REPO), str(REPO / "tools")]
from source.agent.edit_agent import _rasterize
from tools.evaluate_source_availability import score
from tools.evaluate_vlm_grounding import validated_response
from tools.sam_class_evidence import POLICY, cached_class_masks, guard_proposal, mask_support
from tools.source_image_evidence import guard_category, inspect_source


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read(path):
    return json.loads(Path(path).read_bytes())


def evaluate(plan_path, digest, output):
    plan_path, output = Path(plan_path).resolve(), Path(output).resolve()
    if sha(plan_path) != digest:
        raise ValueError("Frozen plan digest differs")
    plan = read(plan_path)
    if plan["policy"] != POLICY or not plan["posthoc_development_replay"]:
        raise ValueError("Unexpected policy")
    if not all(sha(p) == h for p, h in plan["source_bindings"].items()):
        raise ValueError("Frozen source or code changed")
    base_plan = read(plan["source_guard_plan"])
    fixture = read(base_plan["fixture_path"])
    targets = {r["id"]: r for r in read(plan["target_fixture"])["targets"]}
    if len(base_plan["cases"]) != 29 or len(targets) != 27:
        raise ValueError("All29 cases and27 nonsynthetic geometry bindings required")
    output.mkdir(parents=True, exist_ok=False)
    raw_rows, source_rows, guarded_rows, details, ids = [], [], [], [], []
    masks_by_path = {}
    for case in base_plan["cases"]:
        source = inspect_source(Path(case["unmarked_source"]["path"]).read_bytes())
        support = None
        if source.available:
            target = targets[case["id"]]
            class_masks = {}
            for category in ("pallet", "person"):
                ref = target["caches"][category]
                key = (ref["path"], target["image_sha256"], target["frame"], target["camera"], tuple(target["size"]), category)
                if key not in masks_by_path:
                    masks_by_path[key] = cached_class_masks(read(ref["path"]), image_sha256=target["image_sha256"],
                        frame=target["frame"], camera=target["camera"], size=target["size"], category=category)
                class_masks[category] = masks_by_path[key]
            support = mask_support(_rasterize(target["geometry"], *target["size"]).astype(bool), class_masks)
        elif case["id"] in targets:
            raise ValueError("Unexpected unavailable real source")
        item = {"case_id": case["id"], "views": {}}
        for view in ("overview", "linked"):
            saved = read(case["responses"][view]["path"])
            raw = saved["raw_response"]
            parsed = validated_response(raw, fixture["categories"], 1)
            if saved["case_id"] != case["id"] or saved["view"] != view or saved["guard"]:
                raise ValueError("Cached response mapping differs")
            ids.append(raw["id"])
            first = guard_category(parsed["category"], source)
            second = guard_proposal(first["category"], support) if support else {
                "category": "uncertain", "reason": "source_unavailable_no_SAM_lookup", "support": None}
            raw_rows.append({"case_id": case["id"], "view": view, "parsed": parsed})
            source_rows.append({"case_id": case["id"], "view": view, "parsed": {**parsed, "category": first["category"]}})
            guarded_rows.append({"case_id": case["id"], "view": view, "parsed": {**parsed, "category": second["category"]}})
            item["views"][view] = {"source_guard": first, "SAM_guard": second, "raw_category": parsed["category"], "category": second["category"]}
        details.append(item)
    if len(ids) != 58 or len(set(ids)) != 58:
        raise ValueError("All58 unique cached quality responses required")
    raw_scores, source_scores, guarded_scores = (score(fixture, rows) for rows in (raw_rows, source_rows, guarded_rows))
    previous = read(plan["source_guard_summary"])
    if (source_scores["human_correct_ids"] != previous["guarded_scores"]["human_correct_ids"]
            or source_scores["legacy_checks"] != previous["guarded_scores"]["legacy_checks"]):
        raise ValueError("Prior source-only scores not reproduced")
    lost = sorted(set(source_scores["human_correct_ids"]) - set(guarded_scores["human_correct_ids"]))
    quality_checks = {**guarded_scores["original_quality_checks"],
                      "all14_card_correct_consensus_retained": not lost,
                      "wrong_human_consensus_strictly_below1": len(guarded_scores["human_wrong_consensus_ids"]) < 1}
    integrity = {"all29_original_cases_retained": len(details) == 29,
                 "all27_real_targets_source_bound": len(targets) == 27,
                 "all58_cached_quality_responses_unique": len(set(ids)) == 58,
                 "no_new_confident_class_generated": all(d["category"] in (d["raw_category"], "uncertain") for r in details for d in r["views"].values()),
                 "frozen_source_code_config_state_unchanged": all(sha(p) == h for p, h in plan["source_bindings"].items()),
                 "plan_unchanged": sha(plan_path) == digest}
    result = {"status": "complete", "finished_utc": datetime.now(timezone.utc).isoformat(),
              "policy": POLICY, "posthoc_development_replay": True, "plan_sha256": digest,
              "cases": details, "raw_scores": raw_scores, "source_only_scores": source_scores,
              "guarded_scores": guarded_scores, "lost_correct_vs_source_only": lost,
              "quality_checks": quality_checks, "all_quality_checks_passed": all(quality_checks.values()),
              "integrity_checks": integrity, "all_integrity_checks_passed": all(integrity.values()),
              "cached_quality_responses_reused": 58, "unique_SAM_cache_files_used": len(masks_by_path),
              "new_VLM_calls": 0, "new_SAM_calls": 0, "new_training": False,
              "workbench_writes": False, "default_enabled": False, "adopted": False,
              "independent_pixel_accuracy_measured": False, "independent_validation": False}
    (output / "summary.json").write_bytes((json.dumps(result, indent=2) + "\n").encode())
    print(json.dumps({"status": "complete", "human_correct": guarded_scores["human_correct_consensus"],
        "human_wrong": guarded_scores["human_wrong_consensus_ids"], "lost_correct": lost,
        "legacy_checks": guarded_scores["legacy_checks"], "quality_checks": quality_checks, "integrity": integrity}))
    return 0 if all(integrity.values()) else 1


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", required=True)
    parser.add_argument("--plan-sha256", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    raise SystemExit(evaluate(args.plan, args.plan_sha256, args.output))
