import hashlib
import json
import threading
import urllib.error
import urllib.request

import pytest

from tools import serve_mixed_hard_review as review_server


def _sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def workspace(tmp_path):
    root = tmp_path / "review"
    root.mkdir(parents=True)
    (root / "reviewer.html").write_text(
        "<!doctype html><title>human review</title>", encoding="utf-8")
    rows = [{"sample_id": f"sample-{index:02d}"} for index in range(34)]
    (root / "adjudication-template.json").write_text(json.dumps({
        "schema_version": review_server.TEMPLATE_SCHEMA,
        "sample_count": 34,
        "rows": rows,
        "candidate_only": True,
        "human_truth": False,
        "approved_for_training": False,
        "training_authorized": False,
        "automatic_approval": False,
    }), encoding="utf-8")
    files = {name: _sha(root / name) for name in (
        "reviewer.html", "adjudication-template.json")}
    (root / "receipt.json").write_text(json.dumps({
        "schema_version": review_server.RECEIPT_SCHEMA,
        "review_manifest_sha256": "a" * 64,
        "files": files,
        "candidate_only": True,
        "human_truth": False,
        "approved_for_training": False,
        "automatic_approval": False,
    }), encoding="utf-8")
    return root


def test_validates_hash_bound_candidate_only_workspace(tmp_path):
    root = workspace(tmp_path)
    before = {path.relative_to(root): path.read_bytes()
              for path in root.rglob("*") if path.is_file()}
    result = review_server.validate_workspace(root)
    after = {path.relative_to(root): path.read_bytes()
             for path in root.rglob("*") if path.is_file()}
    assert result["status"] == "ready"
    assert result["sample_count"] == 34
    assert result["verified_file_count"] == 2
    assert result["external_call_performed"] is False
    assert before == after


def test_refuses_hash_drift_and_unsafe_flags(tmp_path):
    root = workspace(tmp_path)
    (root / "reviewer.html").write_text("changed", encoding="utf-8")
    with pytest.raises(ValueError, match="hash mismatch"):
        review_server.validate_workspace(root)
    root = workspace(tmp_path / "second")
    receipt = json.loads((root / "receipt.json").read_text(encoding="utf-8"))
    receipt["human_truth"] = True
    (root / "receipt.json").write_text(json.dumps(receipt), encoding="utf-8")
    with pytest.raises(ValueError, match="safety flags"):
        review_server.validate_workspace(root)


def test_refuses_path_escape_and_non_loopback_binding(tmp_path):
    root = workspace(tmp_path)
    receipt = json.loads((root / "receipt.json").read_text(encoding="utf-8"))
    receipt["files"]["../secret.txt"] = "b" * 64
    (root / "receipt.json").write_text(json.dumps(receipt), encoding="utf-8")
    with pytest.raises(ValueError, match="unsafe file path"):
        review_server.validate_workspace(root)
    clean = workspace(tmp_path / "clean")
    with pytest.raises(ValueError, match="loopback"):
        review_server.build_server(clean, host="0.0.0.0")
    with pytest.raises(ValueError, match="loopback"):
        review_server.build_server(clean, host="::1")


def test_serves_only_receipted_files_with_no_store_headers(tmp_path):
    root = workspace(tmp_path)
    (root / "private.txt").write_text("not receipted", encoding="utf-8")
    server, summary = review_server.build_server(root)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        url = f"http://127.0.0.1:{server.server_port}"
        with urllib.request.urlopen(url + "/", timeout=3) as response:
            assert b"human review" in response.read()
            assert response.headers["Cache-Control"] == "no-store"
            assert response.headers["X-Content-Type-Options"] == "nosniff"
        with pytest.raises(urllib.error.HTTPError) as error:
            urllib.request.urlopen(url + "/private.txt", timeout=3)
        assert error.value.code == 404
        assert summary["human_truth"] is False
    finally:
        server.shutdown()
        server.server_close()
        thread.join(3)


def test_cli_check_only_and_structured_refusal(tmp_path, capsys):
    root = workspace(tmp_path)
    assert review_server.main([str(root), "--check-only"]) == 0
    ready = json.loads(capsys.readouterr().out)
    assert ready["status"] == "ready"
    assert "allowed_paths" not in ready
    (root / "reviewer.html").write_text("changed", encoding="utf-8")
    assert review_server.main([str(root), "--check-only"]) == 2
    refused = json.loads(capsys.readouterr().out)
    assert refused["status"] == "refused"
    assert refused["human_truth"] is False
