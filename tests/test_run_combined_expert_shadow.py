import json
import subprocess
import sys
from pathlib import Path

import pytest

from tools.run_combined_expert_shadow import run_combined


ROOT = Path(__file__).resolve().parents[1]


def loader(*args, **kwargs):
    name = kwargs.get("authorization_run_name")
    count = 34 if name == "all34_source_640x512" else 17
    return ([{"sample_id": f"sample-{index}"} for index in range(count)], {
        "authorization_schema": "labelagent.external-upload-authorization.v3",
        "authorization_sha256": "a" * 64,
        "authorized_upload_count": 51,
    })


def runner(statuses, calls):
    def run(*, samples, bindings, output_path, config):
        calls.append(len(samples))
        status = statuses.pop(0)
        payload = {
            "status": status,
            "external_attempts": len(samples),
            "rows": [],
        }
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(payload), encoding="utf-8")
        return payload
    return run


def arguments(tmp_path):
    return dict(all_queue="all", hybrid_queue="hybrid", manifest="manifest",
                receipt="receipt", image_root="images", authorization="auth",
                output_dir=tmp_path, config={"llm_expert_retry_count": 0})


def test_runs_two_prevalidated_stages_with_exact_51_attempts(tmp_path):
    calls = []
    result = run_combined(**arguments(tmp_path), load_fn=loader,
                          run_fn=runner(["completed", "completed"], calls))
    assert calls == [34, 17]
    assert result["status"] == "completed"
    assert result["external_attempts"] == 51
    assert result["new_external_attempts"] == 51
    assert result["automatic_retry_enabled"] is False
    assert result["candidate_only"] is True
    assert result["human_truth"] is False


def test_stops_before_resized_stage_when_source_is_incomplete(tmp_path):
    calls = []
    result = run_combined(**arguments(tmp_path), load_fn=loader,
                          run_fn=runner(["partial"], calls))
    assert calls == [34]
    assert result["status"] == "source_incomplete_resized_not_started"
    assert result["resized_status"] == "not_started"
    assert result["external_attempts"] == 34


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
                     run_fn=runner(["completed"], calls))
    assert calls == []


def test_rejects_retry_configuration_before_network(tmp_path):
    calls = []
    args = arguments(tmp_path)
    args["config"] = {"llm_expert_retry_count": 1}
    with pytest.raises(ValueError, match="forbids automatic API retries"):
        run_combined(**args, load_fn=loader,
                     run_fn=runner(["completed"], calls))
    assert calls == []


def test_cli_entrypoint_loads_from_project_root():
    result = subprocess.run(
        [sys.executable, str(ROOT / "tools" / "run_combined_expert_shadow.py"),
         "--help"], capture_output=True, text=True, check=False)
    assert result.returncode == 0
    assert "--all-queue" in result.stdout
