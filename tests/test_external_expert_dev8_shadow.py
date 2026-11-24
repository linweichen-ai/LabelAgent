import json
from pathlib import Path

from tools.run_external_expert_dev8_shadow import (
    LABELS, compare_with_machine_teacher, load_dev8, run_shadow,
)


def _result(evidence="absent"):
    return {
        "assessments": {
            label: {"evidence": evidence, "confidence": 0.9, "reason": "visible test pixels"}
            for label in LABELS
        },
        "unknown_candidates": [],
    }


def test_compare_is_explicitly_machine_teacher_agreement():
    expert = _result("absent")
    teacher = _result("absent")
    teacher["assessments"]["slope"]["evidence"] = "present"
    result = compare_with_machine_teacher(expert, teacher)
    assert result["matched_labels"] == 7
    assert result["slope_match"] is False


def test_load_dev8_rejects_test24(tmp_path):
    image_dir = tmp_path / "images"
    image_dir.mkdir()
    contract = tmp_path / "contract.json"
    contract.write_text(json.dumps({"labels": list(LABELS), "response_schema": {}}))
    dataset = tmp_path / "dev.jsonl"
    dataset.write_text("\n".join(json.dumps({
        "sample_id": f"s{i}", "split": "dev", "track": "test24",
        "instruction": "x", "target": json.dumps(_result()), "image_sha256": "x",
    }) for i in range(8)))
    try:
        load_dev8(dataset, image_dir, contract)
    except ValueError as exc:
        assert "test24" in str(exc)
    else:
        raise AssertionError("test24 must be rejected")


def test_run_shadow_writes_candidate_only_receipt(tmp_path):
    image = tmp_path / "image.png"
    image.write_bytes(b"image")
    row = {
        "sample_id": "dev0", "image_sha256": "hash", "instruction": "blind",
        "target": json.dumps(_result()), "_image_path": image,
    }
    contract = {"response_schema": {"type": "object"}}
    seen = {}

    def fake_chat(config, instruction, image_bytes, response_schema=None):
        seen.update(instruction=instruction, image_bytes=image_bytes,
                    response_schema=response_schema)
        return _result()

    summary = run_shadow(rows=[row], contract=contract, output_dir=tmp_path / "out",
                         config={}, chat_fn=fake_chat)
    assert summary["status"] == "completed"
    assert summary["test24_used"] is False
    assert seen["instruction"] == "blind"
    record = json.loads((tmp_path / "out" / "results.jsonl").read_text())
    assert record["candidate_only"] is True
    assert record["human_truth"] is False
    assert record["approved_for_training"] is False
