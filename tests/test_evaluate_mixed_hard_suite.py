import subprocess
import sys
import json
from pathlib import Path

from tools import evaluate_mixed_hard_suite as suite
from tools.evaluate_mixed_hard_suite import _quality_summary


ROOT = Path(__file__).resolve().parents[1]


def arm(*, f1, recall, fp, mae=1, quality=0.5):
    return {"macro_f1": f1, "slope_positive_recall": recall,
            "slope_false_positives": fp, "pallet_count_mae": mae,
            "pallet_quality_accuracy": quality}


def test_summary_requires_strict_slope_gain_and_reports_unassessed_gates():
    three = {"arms": {
        "local_only": arm(f1=0.7, recall=0.5, fp=2),
        "hybrid": arm(f1=0.72, recall=0.75, fp=2, mae=0.8, quality=0.6),
    }}
    resolution = {"delta_resize_minus_source": {
        "macro_f1": 0.0, "slope_positive_recall": 0.0,
        "slope_false_positives": 0,
    }}
    cascade = {"external_calls": 20, "escalated_sample_count": 3}
    summary = _quality_summary(three, resolution, cascade)
    assert summary["measured_checks_all_passed"] is True
    assert summary["cascade_external_calls"] == 20
    assert summary["unassessed_required_gates"]
    assert summary["deployment_gate_passed"] is False
    three["arms"]["hybrid"]["slope_positive_recall"] = 0.5
    summary = _quality_summary(three, resolution, cascade)
    assert summary["measured_checks"][
        "hybrid_slope_recall_strict_improvement"] is False
    assert summary["measured_checks_all_passed"] is False


def test_cli_entrypoint_loads_project_modules():
    result = subprocess.run(
        [sys.executable, str(ROOT / "tools" / "evaluate_mixed_hard_suite.py"),
         "--help"], capture_output=True, text=True, check=False)
    assert result.returncode == 0
    assert "--source-results" in result.stdout
    assert "--resized-results" in result.stdout


def test_suite_runs_all_three_offline_evaluators_on_same_inputs(
        tmp_path, monkeypatch):
    paths = {}
    for name in ("adjudication", "local", "hybrid", "all", "source", "resized"):
        path = tmp_path / f"{name}.json"
        path.write_text(json.dumps({"name": name}), encoding="utf-8")
        paths[name] = path
    monkeypatch.setattr(suite, "validate_packet", lambda _: ({}, "f" * 64))
    calls = []
    three = {"arms": {
        "local_only": arm(f1=0.7, recall=0.5, fp=2),
        "hybrid": arm(f1=0.72, recall=0.75, fp=2, mae=0.8, quality=0.6),
    }}
    resolution = {"delta_resize_minus_source": {
        "macro_f1": 0, "slope_positive_recall": 0,
        "slope_false_positives": 0}}
    cascade = {"external_calls": 20, "escalated_sample_count": 3}
    monkeypatch.setattr(suite, "evaluate_three_arm", lambda *a, **k: (
        calls.append("three") or three))
    monkeypatch.setattr(suite, "evaluate_resolution_ab", lambda *a, **k: (
        calls.append("resolution") or resolution))
    monkeypatch.setattr(suite, "evaluate_cascade", lambda *a, **k: (
        calls.append("cascade") or cascade))
    result = suite.evaluate_suite(
        packet_dir=tmp_path, adjudication_path=paths["adjudication"],
        local_path=paths["local"], hybrid_queue_path=paths["hybrid"],
        all_queue_path=paths["all"], source_results_path=paths["source"],
        resized_results_path=paths["resized"])
    assert calls == ["three", "resolution", "cascade"]
    assert result["model_calls_performed"] == 0
    assert result["candidate_only"] is True
    assert result["human_truth"] is False
    assert result["production_deployed"] is False
    assert len(result["source_results_sha256"]) == 64
