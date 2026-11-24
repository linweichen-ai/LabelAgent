import hashlib
import json
import subprocess
import sys
from pathlib import Path

import pytest

from tools.run_combined_expert_shadow import (
    canary_guard, run_combined, stratified_canary_order,
    validate_execution_credential,
)


ROOT = Path(__file__).resolve().parents[1]
TARGET = {"provider": "openai_chat",
          "url": "https://expert.example/v1/chat/completions",
          "model": "openai/gpt-6-astra"}


def loader(*args, **kwargs):
    name = kwargs.get("authorization_run_name")
    count = 34 if name == "all34_source_640x512" else 17
    return ([{"sample_id": f"sample-{index}", "cohort": "hard",
              "source_image_sha256": f"{index + 1:064x}"}
             for index in range(count)], {
        "authorization_schema": "labelagent.external-upload-authorization.v4",
        "authorization_sha256": "a" * 64,
        "authorized_upload_count": 51,
        "authorized_expert_target": TARGET,
    })


def runner(statuses, calls):
    completed_by_path = {}
    def run(*, samples, bindings, output_path, config, max_new_attempts=None,
            claim_attempt_fn=None):
        calls.append(max_new_attempts)
        status = statuses.pop(0)
        key = str(output_path)
        cumulative = completed_by_path.get(key, 0)
        if max_new_attempts is None:
            target = len(samples)
        elif max_new_attempts > 0:
            target = min(len(samples), cumulative + max_new_attempts)
        else:
            target = cumulative
        for sample in samples[cumulative:target]:
            upload_sha = hashlib.sha256(
                (key + sample["sample_id"]).encode("utf-8")).hexdigest()
            if claim_attempt_fn is not None:
                claim_attempt_fn(sample, {"upload_image_sha256": upload_sha})
        cumulative = target
        completed_by_path[key] = cumulative
        payload = {
            "status": status,
            "external_attempts": cumulative,
            "samples_passed": cumulative,
            "latency_median_ms": 5000,
            "external_usage": {
                "complete": True, "samples_with_usage": cumulative,
                "total_tokens": cumulative * 4000,
            },
            "rows": [],
        }
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(payload), encoding="utf-8")
        return payload
    return run


def arguments(tmp_path):
    return dict(all_queue="all", hybrid_queue="hybrid", manifest="manifest",
                receipt="receipt", image_root="images", authorization="auth",
                output_dir=tmp_path, ledger_dir=tmp_path / "ledger",
                credential_preflight_fn=lambda config: {
                    "credential_available": True, "expert_target": TARGET},
                config={"llm_expert_retry_count": 0,
                    "llm_expert_provider": TARGET["provider"],
                    "llm_expert_url": TARGET["url"],
                    "llm_expert_model": TARGET["model"]})


def test_runs_two_prevalidated_stages_with_exact_51_attempts(tmp_path):
    calls = []
    result = run_combined(**arguments(tmp_path), load_fn=loader,
                          run_fn=runner(
                              ["in_progress", "completed", "completed"], calls))
    assert calls == [8, None, None]
    assert result["status"] == "completed"
    assert result["external_attempts"] == 51
    assert result["new_external_attempts"] == 51
    assert result["automatic_retry_enabled"] is False
    assert result["candidate_only"] is True
    assert result["human_truth"] is False
    assert result["authorization_ledger"]["claimed_attempts"] == 51
    assert len(result["authorization_ledger"]["session_sha256"]) == 64
    assert result["credential_preflight"]["credential_available"] is True


def test_stops_before_resized_stage_when_source_is_incomplete(tmp_path):
    calls = []
    result = run_combined(**arguments(tmp_path), load_fn=loader,
                          run_fn=runner(["partial"], calls))
    assert calls == [8]
    assert result["status"] == "source_canary_failed_resized_not_started"
    assert result["resized_status"] == "not_started"
    assert result["external_attempts"] == 8


def test_validates_both_stages_before_any_network_call(tmp_path):
    calls = []
    count = 0
    def bad_loader(*args, **kwargs):
        nonlocal count
        count += 1
        if count == 2:
            raise ValueError("resized hash mismatch")
        return loader(*args, **kwargs)
    with pytest.raises(ValueError, match="resized hash mismatch"):
        run_combined(**arguments(tmp_path), load_fn=bad_loader,
                     run_fn=runner(["in_progress"], calls))
    assert calls == []


def test_legacy_v3_combined_authorization_is_rejected_before_network(tmp_path):
    calls = []
    def legacy_loader(*args, **kwargs):
        samples, bindings = loader(*args, **kwargs)
        return samples, {**bindings,
                         "authorization_schema":
                         "labelagent.external-upload-authorization.v3"}
    with pytest.raises(ValueError, match="combined authorization binding differs"):
        run_combined(**arguments(tmp_path), load_fn=legacy_loader,
                     run_fn=runner(["in_progress"], calls))
    assert calls == []


def test_rejects_retry_configuration_before_network(tmp_path):
    calls = []
    args = arguments(tmp_path)
    args["config"] = {**args["config"], "llm_expert_retry_count": 1}
    with pytest.raises(ValueError, match="forbids automatic API retries"):
        run_combined(**args, load_fn=loader,
                     run_fn=runner(["in_progress"], calls))
    assert calls == []


def test_rejects_model_or_endpoint_drift_before_network(tmp_path):
    for index, changed in enumerate((
            {"llm_expert_model": "different-model"},
            {"llm_expert_url": "https://different.example/v1/chat/completions"})):
        calls = []
        args = arguments(tmp_path / str(index))
        args["config"].update(changed)
        with pytest.raises(ValueError, match="authorization target differs"):
            run_combined(**args, load_fn=loader,
                         run_fn=runner(["in_progress"], calls))
        assert calls == []


def test_missing_credential_is_rejected_before_ledger_or_network(
        tmp_path, monkeypatch):
    monkeypatch.delenv("LABELAGENT_MISSING_TEST_KEY", raising=False)
    calls = []
    args = arguments(tmp_path)
    args.pop("credential_preflight_fn")
    args["config"].update({
        "llm_expert_enabled": True,
        "llm_expert_key_env": "LABELAGENT_MISSING_TEST_KEY",
    })
    with pytest.raises(ValueError, match="密钥环境变量未配置"):
        run_combined(**args, load_fn=loader,
                     run_fn=runner(["in_progress"], calls))
    assert calls == []
    assert not (tmp_path / "ledger").exists()


def test_credential_preflight_returns_only_safe_target_metadata(monkeypatch):
    monkeypatch.setenv("LABELAGENT_TEST_EXPERT_KEY", "fixture-secret-value")
    result = validate_execution_credential({
        "llm_expert_enabled": True,
        "llm_expert_provider": TARGET["provider"],
        "llm_expert_url": TARGET["url"],
        "llm_expert_model": TARGET["model"],
        "llm_expert_key_env": "LABELAGENT_TEST_EXPERT_KEY",
        "llm_expert_retry_count": 0,
    })
    assert result == {"credential_available": True, "expert_target": TARGET}
    assert "fixture-secret-value" not in json.dumps(result)


def test_authorization_cannot_be_reused_with_another_output_directory(tmp_path):
    shared_ledger = tmp_path / "shared-ledger"
    first = arguments(tmp_path / "first")
    first["ledger_dir"] = shared_ledger
    run_combined(**first, load_fn=loader,
                 run_fn=runner(["in_progress", "completed", "completed"], []))
    second_calls = []
    second = arguments(tmp_path / "second")
    second["ledger_dir"] = shared_ledger
    with pytest.raises(ValueError, match="already claimed by another output"):
        run_combined(**second, load_fn=loader,
                     run_fn=runner(["in_progress"], second_calls))
    assert second_calls == []


def test_deleted_result_cannot_repeat_an_already_claimed_paid_call(tmp_path):
    args = arguments(tmp_path)
    first_calls = []
    result = run_combined(**args, load_fn=loader,
                          run_fn=runner(["partial"], first_calls))
    assert result["external_attempts"] == 8
    (tmp_path / "all34-source-results.json").unlink()
    second_calls = []
    with pytest.raises(ValueError, match="already claimed"):
        run_combined(**args, load_fn=loader,
                     run_fn=runner(["in_progress"], second_calls))
    assert second_calls == [8]
    assert len(list((tmp_path / "ledger" / ("a" * 64) /
                     "attempts").glob("*.json"))) == 8


def test_cli_entrypoint_loads_from_project_root():
    result = subprocess.run(
        [sys.executable, str(ROOT / "tools" / "run_combined_expert_shadow.py"),
         "--help"], capture_output=True, text=True, check=False)
    assert result.returncode == 0
    assert "--all-queue" in result.stdout


def test_canary_order_covers_each_cohort_before_filling():
    samples = [
        {"sample_id": "a1", "cohort": "a"},
        {"sample_id": "a2", "cohort": "a"},
        {"sample_id": "b1", "cohort": "b"},
        {"sample_id": "c1", "cohort": "c"},
        {"sample_id": "d1", "cohort": "d"},
    ]
    ordered = stratified_canary_order(samples, 4)
    assert [row["sample_id"] for row in ordered[:4]] == ["a1", "b1", "c1", "d1"]
    assert {row["sample_id"] for row in ordered} == {
        "a1", "a2", "b1", "c1", "d1"}


def test_canary_guard_rejects_latency_or_token_anomaly():
    base = {"status": "in_progress", "external_attempts": 8,
            "samples_passed": 8, "latency_median_ms": 5000,
            "external_usage": {"complete": True, "samples_with_usage": 8,
                               "total_tokens": 32000}}
    assert canary_guard(base, expected_attempts=8,
                        max_median_latency_ms=60000,
                        max_average_tokens=30000)["passed"] is True
    slow = {**base, "latency_median_ms": 61000}
    assert "canary_latency_exceeds_guard" in canary_guard(
        slow, expected_attempts=8, max_median_latency_ms=60000,
        max_average_tokens=30000)["reasons"]
    costly = {**base, "external_usage": {
        "complete": True, "samples_with_usage": 8, "total_tokens": 240008}}
    assert "canary_tokens_exceed_guard" in canary_guard(
        costly, expected_attempts=8, max_median_latency_ms=60000,
        max_average_tokens=30000)["reasons"]
