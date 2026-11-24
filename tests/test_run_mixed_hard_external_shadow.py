import hashlib
import io
import json

import pytest
from PIL import Image

from tools.run_local_qwen_mixed_hard_suggestions import LABELS
from tools.run_mixed_hard_external_shadow import (
    load_authorized_inputs, prepare_upload_image, run_shadow, sha256_file,
)


def _expert_result():
    return {
        "assessments": {label: "absent" for label in LABELS},
        "overall_confidence": 0.9, "pallet_instance_count": 0,
        "pallet_quality": "no_pallet", "slope_geometry": "no_visible_slope",
        "class_conflicts": ["none"], "difficulty_flags": ["none"],
        "unknown_candidates": [], "review_note": "fixture",
    }


def _fixture(tmp_path, authorize=True):
    root = tmp_path / "workspace"; (root / "raw").mkdir(parents=True)
    image = root / "raw" / "sample.jpg"; image.write_bytes(b"image")
    image_sha = hashlib.sha256(b"image").hexdigest()
    manifest = tmp_path / "manifest.json"
    manifest.write_text(json.dumps({
        "summary": {"test24_intersection_count": 0},
        "rows": [{"sample_id": "sample", "raw_image": "raw/sample.jpg"}],
    }), encoding="utf-8")
    receipt = tmp_path / "receipt.json"
    receipt.write_text(json.dumps({
        "files": {"raw/sample.jpg": image_sha}, "candidate_only": True,
        "human_truth": False, "approved_for_training": False,
    }), encoding="utf-8")
    queue = tmp_path / "queue.json"
    queue.write_text(json.dumps({
        "schema_version": "labelagent.mixed-hard-expert-shadow-queue.v1",
        "sample_count": 1, "images_copied": False,
        "image_paths_included": False, "candidate_only": True,
        "human_truth": False, "approved_for_training": False,
        "test24_used": False,
        "samples": [{"sample_id": "sample", "cohort": "hard",
                     "source_image_sha256": image_sha}],
    }), encoding="utf-8")
    auth = tmp_path / "authorization.json"
    auth.write_text(json.dumps({
        "schema_version": "labelagent.external-upload-authorization.v1",
        "queue_sha256": sha256_file(queue), "authorized": authorize,
        "authorized_sample_count": 1, "scope": "mixed_hard_external_shadow",
        "authorized_by": "declared-user", "candidate_only": True,
        "human_truth": False, "approved_for_training": False,
        "test24_used": False,
    }), encoding="utf-8")
    return queue, manifest, receipt, root, auth


def _v2_fixture(tmp_path):
    queue, manifest, receipt, root, auth = _fixture(tmp_path)
    buffer = io.BytesIO()
    Image.new("RGB", (640, 512), (120, 130, 140)).save(buffer, "JPEG")
    image_bytes = buffer.getvalue()
    image_sha = hashlib.sha256(image_bytes).hexdigest()
    (root / "raw" / "sample.jpg").write_bytes(image_bytes)
    queue_payload = json.loads(queue.read_text(encoding="utf-8"))
    queue_payload["samples"][0]["source_image_sha256"] = image_sha
    queue.write_text(json.dumps(queue_payload), encoding="utf-8")
    receipt_payload = json.loads(receipt.read_text(encoding="utf-8"))
    receipt_payload["files"]["raw/sample.jpg"] = image_sha
    receipt.write_text(json.dumps(receipt_payload), encoding="utf-8")
    auth.write_text(json.dumps({
        "schema_version": "labelagent.external-upload-authorization.v2",
        "queue_sha256": sha256_file(queue), "authorized": True,
        "authorized_sample_count": 1,
        "scope": "mixed_hard_external_resolution_ab_shadow",
        "authorized_by": "declared-user",
        "authorized_image_variants": [
            {"name": "source_640x512", "mode": "source",
             "expected_width": 640, "expected_height": 512},
            {"name": "resize_512x410", "mode": "resize_lanczos_jpeg",
             "width": 512, "height": 410, "jpeg_quality": 90},
        ],
        "authorized_upload_count": 2, "max_external_calls": 2,
        "candidate_only": True, "human_truth": False,
        "approved_for_training": False, "test24_used": False,
    }), encoding="utf-8")
    return queue, manifest, receipt, root, auth


def _v3_fixture(tmp_path):
    root = tmp_path / "combined-workspace"
    (root / "raw").mkdir(parents=True)
    buffer = io.BytesIO()
    Image.new("RGB", (640, 512), (80, 90, 100)).save(buffer, "JPEG")
    image_bytes = buffer.getvalue()
    image_sha = hashlib.sha256(image_bytes).hexdigest()
    ids = [f"sample-{index:02d}" for index in range(34)]
    rows, files = [], {}
    for sample_id in ids:
        relative = f"raw/{sample_id}.jpg"
        (root / relative).write_bytes(image_bytes)
        rows.append({"sample_id": sample_id, "raw_image": relative})
        files[relative] = image_sha
    manifest = _write_json(tmp_path / "combined-manifest.json", {
        "summary": {"test24_intersection_count": 0}, "rows": rows})
    receipt = _write_json(tmp_path / "combined-receipt.json", {
        "files": files, "candidate_only": True, "human_truth": False,
        "approved_for_training": False})
    def queue(path, selected):
        return _write_json(path, {
            "schema_version": "labelagent.mixed-hard-expert-shadow-queue.v1",
            "sample_count": len(selected), "images_copied": False,
            "image_paths_included": False, "candidate_only": True,
            "human_truth": False, "approved_for_training": False,
            "test24_used": False,
            "samples": [{"sample_id": sample_id, "cohort": "hard",
                         "source_image_sha256": image_sha}
                        for sample_id in selected],
        })
    all_queue = queue(tmp_path / "all34.json", ids)
    hybrid_queue = queue(tmp_path / "hybrid17.json", ids[:17])
    auth = _write_json(tmp_path / "combined-authorization.json", {
        "schema_version": "labelagent.external-upload-authorization.v3",
        "scope": "mixed_hard_combined_external_shadow",
        "authorized": True, "authorized_by": "declared-user",
        "authorized_unique_source_samples": 34,
        "authorized_upload_count": 51, "max_external_calls": 51,
        "authorized_runs": [
            {"name": "all34_source_640x512",
             "queue_sha256": sha256_file(all_queue), "sample_count": 34,
             "max_external_calls": 34,
             "image_variant": {"name": "source_640x512", "mode": "source",
                               "expected_width": 640, "expected_height": 512}},
            {"name": "hybrid17_resize_512x410",
             "queue_sha256": sha256_file(hybrid_queue), "sample_count": 17,
             "max_external_calls": 17,
             "image_variant": {"name": "resize_512x410",
                               "mode": "resize_lanczos_jpeg", "width": 512,
                               "height": 410, "jpeg_quality": 90}},
        ],
        "candidate_only": True, "human_truth": False,
        "approved_for_training": False, "test24_used": False,
    })
    return all_queue, hybrid_queue, manifest, receipt, root, auth


def _write_json(path, payload):
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def test_refuses_execution_without_matching_explicit_authorization(tmp_path):
    queue, manifest, receipt, root, auth = _fixture(tmp_path, authorize=False)
    unauthorized = auth.read_text(encoding="utf-8")
    auth.unlink()
    with pytest.raises(ValueError, match="explicit external-upload authorization"):
        load_authorized_inputs(queue, manifest, receipt, root, auth)
    auth.write_text(unauthorized, encoding="utf-8")
    with pytest.raises(ValueError, match="explicit external-upload authorization"):
        load_authorized_inputs(queue, manifest, receipt, root, auth)
    payload = json.loads(auth.read_text(encoding="utf-8"))
    payload.update(authorized=True, queue_sha256="0" * 64)
    auth.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="explicit external-upload authorization"):
        load_authorized_inputs(queue, manifest, receipt, root, auth)


def test_authorized_fixture_runs_candidate_only_without_truth_write(tmp_path):
    queue, manifest, receipt, root, auth = _fixture(tmp_path)
    samples, bindings = load_authorized_inputs(queue, manifest, receipt, root, auth)
    calls = []
    def respond(*args, **kwargs):
        kwargs["metadata_out"].update({"external_vlm": {
            "usage": {"prompt_tokens": 20, "completion_tokens": 5,
                      "total_tokens": 25}, "usage_available": True}})
        calls.append(args)
        return _expert_result()
    result = run_shadow(
        samples=samples, bindings=bindings, output_path=tmp_path / "result.json",
        config={"llm_expert_model": "fixture"},
        chat_fn=respond)
    assert result["status"] == "completed"
    assert result["samples_passed"] == 1
    assert result["automatic_truth_write"] is False
    assert result["candidate_only"] is True
    assert result["human_truth"] is False
    assert result["approved_for_training"] is False
    assert result["test24_used"] is False
    assert result["rows"][0]["expert_metadata"]["external_vlm"]["usage"][
        "total_tokens"] == 25
    assert result["external_attempts"] == 1
    assert result["automatic_retry_enabled"] is False
    assert len(calls) == 1


def test_failed_authorized_attempt_is_not_retried_by_resume(tmp_path):
    queue, manifest, receipt, root, auth = _fixture(tmp_path)
    samples, bindings = load_authorized_inputs(queue, manifest, receipt, root, auth)
    output = tmp_path / "failed-result.json"
    calls = []
    def fail(*args, **kwargs):
        calls.append(1)
        raise TimeoutError("fixture timeout")
    first = run_shadow(samples=samples, bindings=bindings, output_path=output,
                       config={"llm_expert_model": "fixture"}, chat_fn=fail)
    second = run_shadow(samples=samples, bindings=bindings, output_path=output,
                        config={"llm_expert_model": "fixture"}, chat_fn=fail)
    assert first["status"] == "partial"
    assert second["status"] == "partial"
    assert first["external_attempts"] == 1
    assert second["external_attempts"] == 1
    assert len(calls) == 1
    assert second["rows"][0]["status"] == "failed"


def test_authorized_shadow_rejects_internal_retry_configuration(tmp_path):
    queue, manifest, receipt, root, auth = _fixture(tmp_path)
    samples, bindings = load_authorized_inputs(queue, manifest, receipt, root, auth)
    with pytest.raises(ValueError, match="forbids automatic API retries"):
        run_shadow(samples=samples, bindings=bindings,
                   output_path=tmp_path / "result.json",
                   config={"llm_expert_model": "fixture",
                           "llm_expert_retry_count": 1},
                   chat_fn=lambda *args, **kwargs: _expert_result())


def test_v1_authorization_cannot_be_reused_for_resize(tmp_path):
    queue, manifest, receipt, root, auth = _fixture(tmp_path)
    with pytest.raises(ValueError, match="explicit external-upload authorization"):
        load_authorized_inputs(
            queue, manifest, receipt, root, auth, "resize_512x410")


def test_v2_authorization_binds_and_records_resized_upload(tmp_path):
    queue, manifest, receipt, root, auth = _v2_fixture(tmp_path)
    samples, bindings = load_authorized_inputs(
        queue, manifest, receipt, root, auth, "resize_512x410")
    observed = {}
    def respond(config, instruction, image, response_schema, metadata_out):
        with Image.open(io.BytesIO(image)) as uploaded:
            observed["size"] = uploaded.size
        metadata_out.update({"external_vlm": {"usage": {
            "prompt_tokens": 20, "completion_tokens": 5,
            "total_tokens": 25}}})
        return _expert_result()
    result = run_shadow(
        samples=samples, bindings=bindings,
        output_path=tmp_path / "resized-result.json",
        config={"llm_expert_model": "fixture"}, chat_fn=respond)
    row = result["rows"][0]
    assert observed["size"] == (512, 410)
    assert row["upload_image"]["name"] == "resize_512x410"
    assert row["upload_image"]["source_width"] == 640
    assert row["upload_image"]["source_height"] == 512
    assert len(row["upload_image"]["upload_image_sha256"]) == 64
    assert result["image_variant"]["width"] == 512
    assert result["authorized_upload_count"] == 2


def test_source_variant_rejects_dimension_drift():
    buffer = io.BytesIO()
    Image.new("RGB", (320, 256), (1, 2, 3)).save(buffer, "JPEG")
    with pytest.raises(ValueError, match="dimensions differ"):
        prepare_upload_image(buffer.getvalue(), {
            "name": "source_640x512", "mode": "source",
            "expected_width": 640, "expected_height": 512})


def test_v3_combined_authorization_binds_two_nonduplicated_runs(tmp_path):
    all_queue, hybrid_queue, manifest, receipt, root, auth = _v3_fixture(tmp_path)
    all_samples, all_bindings = load_authorized_inputs(
        all_queue, manifest, receipt, root, auth,
        authorization_run_name="all34_source_640x512")
    hybrid_samples, hybrid_bindings = load_authorized_inputs(
        hybrid_queue, manifest, receipt, root, auth,
        authorization_run_name="hybrid17_resize_512x410")
    assert len(all_samples) == 34
    assert len(hybrid_samples) == 17
    assert all_bindings["image_variant"]["expected_width"] == 640
    assert hybrid_bindings["image_variant"]["width"] == 512
    assert all_bindings["authorized_upload_count"] == 51
    assert hybrid_bindings["authorized_upload_count"] == 51
    with pytest.raises(ValueError, match="explicit external-upload authorization"):
        load_authorized_inputs(
            hybrid_queue, manifest, receipt, root, auth,
            authorization_run_name="all34_source_640x512")


def test_rejects_image_hash_drift_before_model_call(tmp_path):
    queue, manifest, receipt, root, auth = _fixture(tmp_path)
    (root / "raw" / "sample.jpg").write_bytes(b"changed")
    with pytest.raises(ValueError, match="hash mismatch"):
        load_authorized_inputs(queue, manifest, receipt, root, auth)
