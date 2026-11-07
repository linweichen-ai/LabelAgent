"""Local target overview and linked-card experiment; never writes annotations."""
import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import io
import json
import os
from pathlib import Path
import sys
import time
import traceback
import urllib.request

import psutil
from PIL import Image, ImageDraw, ImageFont

REPO = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(REPO), str(REPO / "tools")]
from tools.connect_local_vlm import require_real_service
from tools.evaluate_confirmed_region_context import check_idle
from tools.evaluate_instance_focus import focus_checks
from tools.evaluate_vlm_grounding import build_payload, validated_response
from tools.run_local_agent_acceptance import save


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def fit_panel(image, width, height):
    """Fit the entire source with aspect ratio preserved, including thin objects."""
    image = image.convert("RGB")
    scale = min(width / image.width, height / image.height)
    size = (max(1, round(image.width * scale)), max(1, round(image.height * scale)))
    fitted = image.resize(size, Image.Resampling.LANCZOS)
    canvas = Image.new("RGB", (width, height), (127, 127, 127))
    offset = ((width - size[0]) // 2, (height - size[1]) // 2)
    canvas.paste(fitted, offset)
    return canvas, {"source_size": list(image.size), "display_size": list(size), "offset": list(offset), "scale": scale}


def compose_target_views(selected, scene, outlined):
    """No category, ID, score, expected answer or fixture role is accepted here."""
    overview, overview_meta = fit_panel(outlined, 512, 512)
    linked = Image.new("RGB", (1536, 512), (127, 127, 127))
    draw = ImageDraw.Draw(linked)
    font = ImageFont.load_default(size=14)
    panels = []
    for index, (title, image) in enumerate(zip(("SELECTED", "SOURCE", "OUTLINE"), (selected, scene, outlined))):
        panel, info = fit_panel(image, 480, 480)
        left = 24 + index * 504
        linked.paste(panel, (left, 16))
        draw.text((left + 4, 0), title, font=font, fill=(240, 240, 240))
        panels.append(info)
    return {"overview": overview, "linked": linked}, {"overview": overview_meta, "linked_panels": panels,
            "foreground_pixels_resized_for_display": True, "class_or_ID_text_rendered": False,
            "neutral_panel_titles": ["SELECTED", "SOURCE", "OUTLINE"]}


def card_payload(fixture, assets, case, view, guard=False):
    local = {**fixture, "prompts": {"single": fixture["guard_prompt"] if guard else fixture["prompts"][view]}}
    refs = [fixture["guard_image"]] if guard else case["views"][view]
    body, _ = build_payload(local, assets, {"views": {"single": refs}}, "single", fixture["model"])
    if not guard:
        body["image_token_budget"] = fixture["image_token_budgets"][view]
    digest = hashlib.sha256(json.dumps(body, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    return body, digest


def pair_rows(cases, rows):
    indexed = {(r["case_id"], r["view"]): r for r in rows}
    pairs = []
    for case in cases:
        a, b = (indexed.get((case["id"], view)) for view in ("overview", "linked"))
        valid = bool(a and b and a.get("parsed") and b.get("parsed"))
        consensus = a["parsed"]["category"] if valid and a["parsed"]["category"] == b["parsed"]["category"] else "uncertain"
        pairs.append({"id": case["id"], "group": case["group"], "valid_pair": valid, "consensus": consensus,
                      "views": {"single": {"parsed": a.get("parsed") if a else None},
                                "context": {"parsed": b.get("parsed") if b else None}}})
    return pairs


def evaluate(args):
    fixture_path = Path(args.fixture_json).resolve()
    fixture = json.loads(fixture_path.read_bytes())
    cases = fixture["cases"]
    if len(cases) != 29 or len({c["id"] for c in cases}) != 29 or Counter(c["group"] for c in cases) != {"human": 19, "legacy": 10}:
        raise ValueError("All19 confirmed regions and all10 old controls required")
    if fixture["image_token_budgets"] != {"overview": 256, "linked": 768} or fixture["temperature"] != 0 or fixture["max_tokens"] != 192:
        raise ValueError("Unexpected frozen method settings")
    assets = {}
    asset_root = Path(fixture["asset_root"]).resolve()
    bindings = {str(fixture_path): sha(fixture_path), **fixture["source_bindings"]}
    refs = [ref for c in cases for group in c["views"].values() for ref in group] + [fixture["guard_image"]]
    for ref in refs:
        path = (asset_root / ref["path"]).resolve()
        if not path.is_relative_to(asset_root) or sha(path) != ref["sha256"]:
            raise ValueError("Unbound card asset")
        with Image.open(path) as image:
            if image.format != "JPEG":
                raise ValueError("Frozen JPEG required")
            image.verify()
        assets[ref["path"]] = (path.read_bytes(), "image/jpeg")
        bindings[str(path)] = ref["sha256"]
    for path in (Path(__file__), REPO / "tools/evaluate_vlm_grounding.py", REPO / "tools/evaluate_instance_focus.py",
                 REPO / "tools/evaluate_confirmed_region_context.py", REPO / "tools/local_vlm_server.py"):
        bindings[str(path.resolve())] = sha(path)
    assert all(sha(p) == digest for p, digest in bindings.items())
    check_idle()
    output = Path(args.output).resolve()
    output.mkdir(parents=True, exist_ok=False)
    state_path = REPO / "work/state.json"
    state_before = sha(state_path)
    plan = {"created_utc": datetime.now(timezone.utc).isoformat(), "planned_quality_requests": 58,
            "planned_default_budget_guard_requests": 1, "planned_requests": 59,
            "intervention": "Context-retaining target overview and linked three-panel card, corresponding prompts and measured image budgets; combined intervention, not single-factor causality",
            "checks": fixture["predeclared_checks"], "source_bindings": bindings,
            "model_config_sha256": fixture["model_config_sha256"],
            "case_order": [c["id"] for c in cases], "image_token_budgets": fixture["image_token_budgets"],
            "request_order": "Alternate overview/linked order; last quality request uses768, followed by default-budget guard",
            "state_sha256_before": state_before, "new_SAM_calls": 0, "workbench_writes": False,
            "independent_pixel_accuracy": False, "default_enabled": False}
    save(output / "study_plan.json", plan)
    plan_sha = sha(output / "study_plan.json")
    summary = {"status": "preflight", "pid": os.getpid(), "create_time": psutil.Process().create_time(),
               "attempted_requests": 0, "responses_received": 0, "valid_real_CUDA_responses": 0,
               "invalid_outputs": 0, "rows": [], "guard": None, "adopted": False, "workbench_writes": False}
    save(output / "summary.json", summary)
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))

    def call(case, view, guard=False):
        if (output / "STOP").exists():
            raise InterruptedError("Stop requested")
        check_idle()
        body, digest = card_payload(fixture, assets, case, view, guard)
        number = summary["attempted_requests"] + 1
        pending = {"number": number, "case_id": case["id"], "view": view, "guard": guard, "full_payload_sha256": digest}
        save(output / "request_inflight.json", pending)
        summary["attempted_requests"] = number
        save(output / "summary.json", summary)
        started = time.monotonic()
        request = urllib.request.Request(args.base.rstrip("/") + "/chat/completions", json.dumps(body).encode(), headers={"Content-Type": "application/json"})
        with opener.open(request, timeout=300) as response:
            raw = json.load(response)
        row = {**pending, "seconds": round(time.monotonic() - started, 3)}
        save(output / f"response_{number:03d}.json", {**row, "raw_response": raw})
        (output / "request_inflight.json").unlink()
        summary["responses_received"] += 1
        try:
            parsed = validated_response(raw, fixture["categories"], 1)
            meta = raw["local_vlm"]
            budget = 256 if guard else fixture["image_token_budgets"][view]
            assert meta["model_config_sha256"] == fixture["model_config_sha256"] and meta["quantization"] == "nf4"
            assert meta["effective_image_token_budget"] == budget and meta["actual_image_token_counts"]
            row.update(parsed=parsed, actual_image_token_counts=meta["actual_image_token_counts"], image_grid_thw=meta["image_grid_thw"])
            summary["valid_real_CUDA_responses"] += 1
        except (ValueError, KeyError, TypeError, AssertionError):
            row["invalid_output"] = True
            summary["invalid_outputs"] += 1
        if guard:
            summary["guard"] = row
        else:
            summary["rows"].append(row)
        save(output / "summary.json", summary)
        print(json.dumps({"completed": summary["responses_received"], "planned": 59, "case_id": case["id"],
                          "view": view, "category": row.get("parsed", {}).get("category"),
                          "actual_tokens": row.get("actual_image_token_counts"), "guard": guard}), flush=True)

    try:
        health = require_real_service(args.base, fixture["model"])
        assert health["model_config_sha256"] == fixture["model_config_sha256"] and health["quantization"] == "nf4"
        assert health["supports_single_image_budget_override"] and health["image_token_budget"] == 256
        save(output / "backend_before.json", health)
        summary["status"] = "running"
        save(output / "summary.json", summary)
        for index, case in enumerate(cases):
            for view in (("overview", "linked") if index % 2 == 0 else ("linked", "overview")):
                call(case, view)
        assert summary["rows"][-1]["view"] == "linked"
        call({"id": "default_budget_guard"}, "guard", guard=True)
        summary["status"] = "complete"
    except InterruptedError:
        summary["status"] = "stopped"
    except Exception as error:
        summary.update(status="failed", error_type=type(error).__name__)
        (output / "local_error.txt").write_text(traceback.format_exc(), encoding="utf-8")
    finally:
        pairs = pair_rows(cases, summary["rows"])
        human = [r for r in pairs if r["group"] == "human"]
        expected = {c["id"]: c["expected_category"] for c in cases if c["group"] == "human"}
        correct = [r["id"] for r in human if r["valid_pair"] and r["consensus"] == expected[r["id"]]]
        legacy = [{**r, "id": next(c["source_id"] for c in cases if c["id"] == r["id"]),
                   "role": next(c["role"] for c in cases if c["id"] == r["id"])} for r in pairs if r["group"] == "legacy"]
        valid_legacy = sum(bool(r["views"][v]["parsed"]) for r in legacy for v in ("single", "context"))
        legacy_checks = focus_checks(fixture["legacy_requirements"], legacy, valid_legacy) if valid_legacy == 20 else {"all_cases_have_valid_real_CUDA_responses": False}
        baseline = fixture["baseline_human_correct_ids"]
        lost = sorted(set(baseline) - set(correct))
        new_opposite = []
        for row in summary["rows"]:
            if row["case_id"] in expected and row.get("parsed"):
                previous = fixture["baseline_human_view_categories"][row["case_id"]][row["view"]]
                allowed = (expected[row["case_id"]], "uncertain")
                if row["parsed"]["category"] not in allowed and previous in allowed:
                    new_opposite.append({"case_id": row["case_id"], "view": row["view"]})
        complete = summary["status"] == "complete" and summary["valid_real_CUDA_responses"] == 59
        guard = summary["guard"] or {}
        checks = {"all59_new_responses_valid_real_CUDA": complete,
                  "human_consensus_strictly_above11": bool(complete and len(correct) > 11),
                  "no_previous_correct_human_region_lost": bool(complete and not lost),
                  "all5_confirmed_people_correct": bool(complete and sum(r["id"] in correct for r in human if expected[r["id"]] == "person") == 5),
                  "no_new_opposite_class_view_error": bool(complete and not new_opposite),
                  "all_original_legacy_checks_passed": bool(complete and all(legacy_checks.values())),
                  "default_budget_restored_measured_guard": bool(complete and guard.get("actual_image_token_counts") == fixture["guard_expected_actual_tokens"]),
                  "state_unchanged": sha(state_path) == state_before,
                  "input_code_source_bindings_unchanged": all(sha(p) == h for p, h in bindings.items()),
                  "plan_unchanged": sha(output / "study_plan.json") == plan_sha}
        summary.update(human_correct_consensus=len(correct), human_correct_ids=correct, human_pairs=human,
                       lost_human_regions=lost, new_opposite_class_view_errors=new_opposite,
                       legacy_pairs=legacy, legacy_checks=legacy_checks, checks=checks,
                       all_predeclared_checks_passed=all(checks.values()), finished_utc=datetime.now(timezone.utc).isoformat())
        save(output / "summary.json", summary)
    print(json.dumps({"status": summary["status"], "valid_responses": summary["valid_real_CUDA_responses"],
                      "human_correct": len(correct), "checks": checks}), flush=True)
    return 0 if summary["status"] == "complete" else 1


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fixture-json", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--base", default="http://127.0.0.1:18868/v1")
    raise SystemExit(evaluate(parser.parse_args()))
