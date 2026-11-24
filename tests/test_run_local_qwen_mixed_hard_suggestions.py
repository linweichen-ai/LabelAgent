import hashlib
import json

import pytest

from tools.run_local_qwen_mixed_hard_suggestions import (
    LABELS, load_inputs, run_suggestions,
)


def _result():
    return {
        "assessments": {label: "absent" for label in LABELS},
        "overall_confidence": 0.9,
        "pallet_instance_count": 0,
        "pallet_quality": "no_pallet",
        "slope_geometry": "no_visible_slope",
        "class_conflicts": ["none"],
        "difficulty_flags": ["none"],
        "unknown_candidates": [],
        "review_note": "fixture only",
    }


def _fixture(tmp_path):
    image_root = tmp_path / "workspace"
    raw = image_root / "raw"
    raw.mkdir(parents=True)
    rows, files = [], {}
    for index in range(34):
        sample = f"sample-{index:02d}"
        relative = f"raw/{sample}.jpg"
        data = f"image-{index}".encode()
        (image_root / relative).write_bytes(data)
        files[relative] = hashlib.sha256(data).hexdigest()
        rows.append({
            "sample_id": sample, "cohort": "fixture",
            "raw_image": relative, "candidate_only": True,
            "human_truth": False, "approved_for_training": False,
        })
    manifest = tmp_path / "manifest.json"
    manifest.write_text(json.dumps({
        "schema_version": "labelagent.mixed-hard-review-pack.v1",
        "summary": {"test24_intersection_count": 0}, "rows": rows,
    }), encoding="utf-8")
    receipt = tmp_path / "receipt.json"
    receipt.write_text(json.dumps({
        "files": files, "candidate_only": True, "human_truth": False,
        "approved_for_training": False,
    }), encoding="utf-8")
    return manifest, receipt, image_root


def test_generates_separate_candidate_only_resume_packet(tmp_path):
    manifest, receipt, image_root = _fixture(tmp_path)
    rows, bindings = load_inputs(manifest, receipt, image_root)
    calls = []
    output = tmp_path / "suggestions.json"

    def fake_chat(config, instruction, image, response_schema=None):
        calls.append(image)
        return _result()

    first = run_suggestions(
        rows=rows, bindings=bindings, output_path=output,
        config={"llm_model": "fixture"}, limit=2, chat_fn=fake_chat)
    assert first["status"] == "partial"
    assert first["samples_passed"] == 2
    assert first["automatic_truth_prefill"] is False
    second = run_suggestions(
        rows=rows, bindings=bindings, output_path=output,
        config={"llm_model": "fixture"}, limit=1, chat_fn=fake_chat)
    assert second["samples_passed"] == 3
    assert len(calls) == 3
    assert all(row["candidate_only"] is True for row in second["rows"])
    assert all(row["human_truth"] is False for row in second["rows"])
    assert all(row["expert_shadow_route"]["dispatch_performed"] is False
               for row in second["rows"])


def test_rejects_test24_or_changed_image(tmp_path):
    manifest, receipt, image_root = _fixture(tmp_path)
    payload = json.loads(manifest.read_text(encoding="utf-8"))
    payload["summary"]["test24_intersection_count"] = 1
    manifest.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="test24"):
        load_inputs(manifest, receipt, image_root)
    payload["summary"]["test24_intersection_count"] = 0
    manifest.write_text(json.dumps(payload), encoding="utf-8")
    (image_root / "raw" / "sample-00.jpg").write_bytes(b"changed")
    with pytest.raises(ValueError, match="hash mismatch"):
        load_inputs(manifest, receipt, image_root)


def test_rejects_foreign_resume_packet(tmp_path):
    manifest, receipt, image_root = _fixture(tmp_path)
    rows, bindings = load_inputs(manifest, receipt, image_root)
    output = tmp_path / "suggestions.json"
    output.write_text(json.dumps({
        "manifest_sha256": "0" * 64, "receipt_sha256": "1" * 64,
        "rows": [],
    }), encoding="utf-8")
    with pytest.raises(ValueError, match="different review pack"):
        run_suggestions(rows=rows, bindings=bindings, output_path=output,
                        config={}, chat_fn=lambda *args, **kwargs: _result())
