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


def _packet(ids, *, external, manifest_sha="f" * 64):
    key = "expert_result" if external else "suggestion"
    return {
        "schema_version": ("labelagent.mixed-hard-external-shadow.v1" if external
                           else "labelagent.local-qwen-mixed-hard-suggestions.v1"),
        "manifest_sha256": manifest_sha,
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


def _queue(ids, source_sha):
    return {
        "schema_version": "labelagent.mixed-hard-expert-shadow-queue.v1",
        "source_suggestions_sha256": source_sha,
        "sample_count": len(ids), "samples": [{"sample_id": item} for item in ids],
        "candidate_only": True, "human_truth": False,
        "approved_for_training": False, "test24_used": False,
    }


def _authorization(authorized, *, manifest_sha, receipt_sha,
                   all_queue_sha, hybrid_queue_sha, schema="v4"):
    return {
        "schema_version": f"labelagent.external-upload-authorization.{schema}",
        "authorized": authorized,
        "authorized_by": "user" if authorized else "",
        "authorized_unique_source_samples": 34,
        "authorized_upload_count": 51, "max_external_calls": 51,
        "authorized_runs": [
            {"name": "all34_source_640x512",
             "queue_sha256": all_queue_sha},
            {"name": "hybrid17_resize_512x410",
             "queue_sha256": hybrid_queue_sha},
        ],
        "review_manifest_sha256": manifest_sha,
        "review_receipt_sha256": receipt_sha,
        "expert_target": {
            "provider": "openai_chat",
            "url": "https://expert.example/v1",
            "model": "openai/gpt-6-astra",
        },
        "candidate_only": True,
        "human_truth": False, "approved_for_training": False,
        "test24_used": False,
    }


def _files(tmp_path, *, authorized=False, external=False):
    ids34 = [f"sample-{index:02d}" for index in range(34)]
    ids17 = ids34[:17]
    manifest = _write(tmp_path / "manifest.json", {"rows": []})
    receipt = _write(tmp_path / "receipt.json", {"files": {}})
    manifest_sha = audit.sha256_file(manifest)
    local = _write(tmp_path / "local.json", _packet(
        ids34, external=False, manifest_sha=manifest_sha))
    source_sha = audit.sha256_file(local)
    hybrid = _write(tmp_path / "hybrid.json", _queue(ids17, source_sha))
    all_queue = _write(tmp_path / "all-queue.json", _queue(ids34, source_sha))
    paths = {
        "manifest": manifest,
        "receipt": receipt,
        "adjudication": _write(tmp_path / "adjudication.json", {}),
        "local": local,
        "hybrid": hybrid,
        "all_queue": all_queue,
        "authorization": _write(tmp_path / "authorization.json",
                                _authorization(
                                    authorized,
                                    manifest_sha=manifest_sha,
                                    receipt_sha=audit.sha256_file(receipt),
                                    all_queue_sha=audit.sha256_file(all_queue),
                                    hybrid_queue_sha=audit.sha256_file(hybrid))),
    }
    if external:
        paths.update(
            all_external=_write(tmp_path / "all-external.json",
                                _packet(ids34, external=True)),
            source=_write(tmp_path / "source.json", _packet(ids34, external=True)),
            resized=_write(tmp_path / "resized.json", _packet(ids17, external=True)),
        )
    return paths, ids34


def _validated_packet(paths, ids):
    return ({"rows": [{"sample_id": item} for item in ids]},
            audit.sha256_file(paths["manifest"]))


def test_audit_reports_exact_missing_components_without_running_models(tmp_path, monkeypatch):
    paths, ids = _files(tmp_path)
    monkeypatch.setattr(audit, "validate_packet",
                        lambda _: _validated_packet(paths, ids))
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
    assert "run authorized combined all34 source and hybrid17 resized shadows" in result[
        "next_actions"]
    assert "run complete all-34 external shadow" not in result["next_actions"]
    assert result["external_call_performed"] is False


def test_audit_describes_unfilled_review_template_as_progress_not_schema_error(
        tmp_path, monkeypatch):
    paths, ids = _files(tmp_path)
    paths["adjudication"] = _write(tmp_path / "adjudication-template.json", {
        "schema_version": "labelagent.mixed-hard-adjudication-template.v2",
        "rows": [{
            "sample_id": sample_id,
            "candidate_decision": None,
            "slope_presence": None,
            "class_conflict": None,
            "pallet_quality": None,
            "class_presence": {label: None for label in LABELS},
            "pallet_instance_count": None,
        } for sample_id in ids],
    })
    monkeypatch.setattr(audit, "validate_packet",
                        lambda _: _validated_packet(paths, ids))
    result = audit.audit_readiness(
        packet_dir=tmp_path, adjudication_path=paths["adjudication"],
        local_path=paths["local"], hybrid_queue_path=paths["hybrid"],
        all_queue_path=paths["all_queue"],
        resolution_authorization_path=paths["authorization"])
    review = result["components"]["declared_review"]
    assert review["state"] == "review_template_not_submission"
    assert review["completed_rows"] == 0
    assert review["remaining_rows"] == 34
    assert review["expected_export_filename"] == "mixed-hard-adjudication.json"
    assert "unsupported adjudication schema" not in review["error"]
    assert result["next_actions"][0].startswith("complete 34-row review")


def test_audit_marks_quality_inputs_ready_but_never_production(tmp_path, monkeypatch):
    paths, ids = _files(tmp_path, authorized=True, external=True)
    monkeypatch.setattr(audit, "validate_packet",
                        lambda _: _validated_packet(paths, ids))
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


def test_audit_fails_closed_when_packet_and_local_manifest_differ(
        tmp_path, monkeypatch):
    paths, ids = _files(tmp_path)
    payload = json.loads(paths["local"].read_text(encoding="utf-8"))
    payload["manifest_sha256"] = "0" * 64
    paths["local"] = _write(paths["local"], payload)
    monkeypatch.setattr(audit, "validate_packet",
                        lambda _: _validated_packet(paths, ids))
    result = audit.audit_readiness(
        packet_dir=tmp_path, adjudication_path=paths["adjudication"],
        local_path=paths["local"], hybrid_queue_path=paths["hybrid"],
        all_queue_path=paths["all_queue"],
        resolution_authorization_path=paths["authorization"])
    assert result["components"]["local34"]["ready"] is False
    assert result["components"]["local34"]["error"] == (
        "review manifest binding differs")
    assert result["gates"]["three_arm_quality_evaluation_ready"] is False


def test_audit_fails_closed_when_queue_uses_different_local_suggestions(
        tmp_path, monkeypatch):
    paths, ids = _files(tmp_path)
    queue = json.loads(paths["hybrid"].read_text(encoding="utf-8"))
    queue["source_suggestions_sha256"] = "0" * 64
    paths["hybrid"] = _write(paths["hybrid"], queue)
    monkeypatch.setattr(audit, "validate_packet",
                        lambda _: _validated_packet(paths, ids))
    result = audit.audit_readiness(
        packet_dir=tmp_path, adjudication_path=paths["adjudication"],
        local_path=paths["local"], hybrid_queue_path=paths["hybrid"],
        all_queue_path=paths["all_queue"],
        resolution_authorization_path=paths["authorization"])
    assert result["components"]["local34"]["ready"] is True
    assert result["components"]["hybrid17_queue"]["ready"] is False
    assert result["components"]["hybrid17_queue"]["error"] == (
        "queue source suggestions binding differs")


def test_audit_rejects_legacy_v3_combined_authorization(tmp_path, monkeypatch):
    paths, ids = _files(tmp_path, authorized=True)
    payload = json.loads(paths["authorization"].read_text(encoding="utf-8"))
    payload["schema_version"] = "labelagent.external-upload-authorization.v3"
    paths["authorization"] = _write(paths["authorization"], payload)
    monkeypatch.setattr(audit, "validate_packet",
                        lambda _: _validated_packet(paths, ids))
    result = audit.audit_readiness(
        packet_dir=tmp_path, adjudication_path=paths["adjudication"],
        local_path=paths["local"], hybrid_queue_path=paths["hybrid"],
        all_queue_path=paths["all_queue"],
        resolution_authorization_path=paths["authorization"])
    authorization = result["components"]["resolution_ab_authorization"]
    assert authorization["ready"] is False
    assert authorization["safe_template"] is False
    assert authorization["error"] == (
        "legacy combined authorization is not executable")


def test_audit_rejects_v4_receipt_binding_drift(tmp_path, monkeypatch):
    paths, ids = _files(tmp_path, authorized=True)
    payload = json.loads(paths["authorization"].read_text(encoding="utf-8"))
    payload["review_receipt_sha256"] = "0" * 64
    paths["authorization"] = _write(paths["authorization"], payload)
    monkeypatch.setattr(audit, "validate_packet",
                        lambda _: _validated_packet(paths, ids))
    result = audit.audit_readiness(
        packet_dir=tmp_path, adjudication_path=paths["adjudication"],
        local_path=paths["local"], hybrid_queue_path=paths["hybrid"],
        all_queue_path=paths["all_queue"],
        resolution_authorization_path=paths["authorization"])
    authorization = result["components"]["resolution_ab_authorization"]
    assert authorization["ready"] is False
    assert authorization["safe_template"] is False
    assert authorization["error"] == (
        "authorization identity or safety binding differs")

