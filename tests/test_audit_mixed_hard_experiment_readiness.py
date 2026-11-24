import json

from tools import audit_mixed_hard_experiment_readiness as audit
from tools.build_mixed_hard_adjudication_workspace import LABELS


def _write(path, payload):
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def _prediction():
    return {
        "assessments": {label: "absent" for label in LABELS},
        "pallet_instance_count": 0, "pallet_quality": "no_pallet",
    }


def _packet(ids, *, external):
    key = "expert_result" if external else "suggestion"
    return {
        "schema_version": ("labelagent.mixed-hard-external-shadow.v1" if external
                           else "labelagent.local-qwen-mixed-hard-suggestions.v1"),
        "status": "completed", "samples_passed": len(ids),
        "samples_failed": 0, "candidate_only": True, "human_truth": False,
        "approved_for_training": False, "test24_used": False,
        "rows": [{
            "sample_id": sample_id, "source_image_sha256": f"{index + 1:064x}",
            "latency_ms": 1, key: _prediction(), "candidate_only": True,
            "human_truth": False, "approved_for_training": False,
            "test24_used": False,
        } for index, sample_id in enumerate(ids)],
    }


def _queue(ids):
    return {
        "schema_version": "labelagent.mixed-hard-expert-shadow-queue.v1",
        "sample_count": len(ids), "samples": [{"sample_id": item} for item in ids],
        "candidate_only": True, "human_truth": False,
        "approved_for_training": False, "test24_used": False,
    }


def _authorization(authorized):
    return {
        "schema_version": "labelagent.external-upload-authorization.v2",
        "authorized": authorized,
        "authorized_by": "user" if authorized else "",
        "authorized_sample_count": 17, "authorized_upload_count": 34,
        "max_external_calls": 34, "candidate_only": True,
        "human_truth": False, "approved_for_training": False,
        "test24_used": False,
    }


def _files(tmp_path, *, authorized=False, external=False):
    ids34 = [f"sample-{index:02d}" for index in range(34)]
    ids17 = ids34[:17]
    paths = {
        "adjudication": _write(tmp_path / "adjudication.json", {}),
        "local": _write(tmp_path / "local.json", _packet(ids34, external=False)),
        "hybrid": _write(tmp_path / "hybrid.json", _queue(ids17)),
        "all_queue": _write(tmp_path / "all-queue.json", _queue(ids34)),
        "authorization": _write(tmp_path / "authorization.json",
                                _authorization(authorized)),
    }
    if external:
        paths.update(
            all_external=_write(tmp_path / "all-external.json",
                                _packet(ids34, external=True)),
            source=_write(tmp_path / "source.json", _packet(ids17, external=True)),
            resized=_write(tmp_path / "resized.json", _packet(ids17, external=True)),
        )
    return paths, ids34


def test_audit_reports_exact_missing_components_without_running_models(tmp_path, monkeypatch):
    paths, ids = _files(tmp_path)
    monkeypatch.setattr(audit, "validate_packet", lambda _: (
        {"rows": [{"sample_id": item} for item in ids]}, "f" * 64))
    monkeypatch.setattr(audit, "validate_submission", lambda *args: {
        "benchmark_reference_ready": False})
    result = audit.audit_readiness(
        packet_dir=tmp_path, adjudication_path=paths["adjudication"],
        local_path=paths["local"], hybrid_queue_path=paths["hybrid"],
        all_queue_path=paths["all_queue"],
        resolution_authorization_path=paths["authorization"])
    assert result["components"]["local34"]["ready"] is True
    assert result["components"]["hybrid17_queue"]["ready"] is True
    assert result["components"]["resolution_ab_authorization"]["ready"] is False
    assert result["gates"]["three_arm_quality_evaluation_ready"] is False
    assert "obtain explicit resolution A/B authorization" in result["next_actions"]
    assert result["external_call_performed"] is False


def test_audit_marks_quality_inputs_ready_but_never_production(tmp_path, monkeypatch):
    paths, ids = _files(tmp_path, authorized=True, external=True)
    monkeypatch.setattr(audit, "validate_packet", lambda _: (
        {"rows": [{"sample_id": item} for item in ids]}, "f" * 64))
    monkeypatch.setattr(audit, "validate_submission", lambda *args: {
        "benchmark_reference_ready": True})
    result = audit.audit_readiness(
        packet_dir=tmp_path, adjudication_path=paths["adjudication"],
        local_path=paths["local"], hybrid_queue_path=paths["hybrid"],
        all_queue_path=paths["all_queue"],
        resolution_authorization_path=paths["authorization"],
        all_external_path=paths["all_external"], source_ab_path=paths["source"],
        resized_ab_path=paths["resized"])
    assert result["gates"]["three_arm_quality_evaluation_ready"] is True
    assert result["gates"]["resolution_ab_quality_evaluation_ready"] is True
    assert result["gates"]["production_deployment_ready"] is False
    assert result["next_actions"] == []
    assert result["candidate_only"] is True

