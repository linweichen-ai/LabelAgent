import json
import threading

import pytest

from tools.external_authorization_ledger import (
    AuthorizationLedger, inspect_ledger_session,
)


AUTH = "a" * 64
SAMPLE = {"sample_id": "sample-01", "source_image_sha256": "b" * 64}
UPLOAD = {"upload_image_sha256": "c" * 64}


def test_same_session_can_resume_but_different_output_is_rejected(tmp_path):
    root = tmp_path / "ledger"
    inspection = inspect_ledger_session(
        root, authorization_sha256=AUTH, output_dir=tmp_path / "output")
    assert inspection["state"] == "unclaimed"
    assert not root.exists()
    first = AuthorizationLedger(root, authorization_sha256=AUTH,
                                output_dir=tmp_path / "output")
    resumed = AuthorizationLedger(root, authorization_sha256=AUTH,
                                  output_dir=tmp_path / "output")
    assert first.session_path == resumed.session_path
    assert inspect_ledger_session(
        root, authorization_sha256=AUTH,
        output_dir=tmp_path / "output")["state"] == "resume"
    with pytest.raises(ValueError, match="another output directory"):
        AuthorizationLedger(root, authorization_sha256=AUTH,
                            output_dir=tmp_path / "different")


def test_attempt_claim_is_durable_and_contains_no_image_or_secret(tmp_path):
    ledger = AuthorizationLedger(tmp_path / "ledger",
                                 authorization_sha256=AUTH,
                                 output_dir=tmp_path / "output")
    result = ledger.claim_attempt("source", SAMPLE, UPLOAD)
    payload = json.loads(ledger.directory.joinpath(
        "attempts", result["claim_id"] + ".json").read_text(encoding="utf-8"))
    assert payload["source_image_sha256"] == "b" * 64
    assert payload["upload_image_sha256"] == "c" * 64
    assert set(payload) == {
        "schema_version", "authorization_sha256", "run_name", "sample_id",
        "source_image_sha256", "upload_image_sha256"}
    with pytest.raises(ValueError, match="duplicate paid call"):
        ledger.claim_attempt("source", SAMPLE, UPLOAD)
    snapshot = ledger.snapshot()
    assert snapshot["claimed_attempts"] == 1
    assert len(snapshot["session_sha256"]) == 64


def test_concurrent_duplicate_claim_allows_exactly_one_owner(tmp_path):
    ledger = AuthorizationLedger(tmp_path / "ledger",
                                 authorization_sha256=AUTH,
                                 output_dir=tmp_path / "output")
    barrier = threading.Barrier(3)
    outcomes = []
    lock = threading.Lock()

    def claim():
        barrier.wait()
        try:
            ledger.claim_attempt("source", SAMPLE, UPLOAD)
            outcome = "claimed"
        except ValueError as error:
            outcome = str(error)
        with lock:
            outcomes.append(outcome)

    workers = [threading.Thread(target=claim) for _ in range(2)]
    for worker in workers:
        worker.start()
    barrier.wait()
    for worker in workers:
        worker.join(2)
    assert outcomes.count("claimed") == 1
    assert sum("duplicate paid call" in item for item in outcomes) == 1


def test_corrupt_existing_session_fails_closed(tmp_path):
    root = tmp_path / "ledger"
    path = root / AUTH / "session.json"
    path.parent.mkdir(parents=True)
    path.write_text("not-json", encoding="utf-8")
    with pytest.raises(ValueError, match="unreadable"):
        AuthorizationLedger(root, authorization_sha256=AUTH,
                            output_dir=tmp_path / "output")
