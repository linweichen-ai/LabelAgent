"""Durable, secret-free claim ledger for authorized external image calls."""
import hashlib
import json
import os
from pathlib import Path


SCHEMA_VERSION = "labelagent.external-authorization-ledger.v1"


def default_ledger_dir():
    configured = os.environ.get("LABELAGENT_EXTERNAL_AUTH_LEDGER_DIR")
    if configured:
        return Path(configured).expanduser()
    return Path.home() / ".labelagent" / "external-authorization-ledger"


def _exclusive_json(path, payload):
    encoded = (json.dumps(payload, ensure_ascii=False, sort_keys=True,
                          separators=(",", ":")) + "\n").encode("utf-8")
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(encoded)
            stream.flush()
            os.fsync(stream.fileno())
    except BaseException:
        try:
            path.unlink()
        except OSError:
            pass
        raise


def _read_json(path):
    try:
        return json.loads(Path(path).read_text(encoding="utf-8"))
    except Exception as error:
        raise ValueError("external authorization ledger is unreadable") from error


class AuthorizationLedger:
    """Bind one authorization hash to one output and claim every paid call once."""

    def __init__(self, root, *, authorization_sha256, output_dir):
        if (not isinstance(authorization_sha256, str)
                or len(authorization_sha256) != 64
                or any(character not in "0123456789abcdef"
                       for character in authorization_sha256.lower())):
            raise ValueError("authorization ledger hash is invalid")
        self.authorization_sha256 = authorization_sha256.lower()
        self.output_dir = str(Path(output_dir).resolve())
        self.directory = Path(root).expanduser().resolve() / self.authorization_sha256
        self.session_path = self.directory / "session.json"
        session = {
            "schema_version": SCHEMA_VERSION,
            "authorization_sha256": self.authorization_sha256,
            "output_dir": self.output_dir,
        }
        try:
            _exclusive_json(self.session_path, session)
        except FileExistsError:
            if _read_json(self.session_path) != session:
                raise ValueError(
                    "external authorization was already claimed by another output directory")

    def claim_attempt(self, run_name, sample, upload_image):
        sample_id = sample.get("sample_id") if isinstance(sample, dict) else None
        source_sha = (sample.get("source_image_sha256")
                      if isinstance(sample, dict) else None)
        upload_sha = (upload_image.get("upload_image_sha256")
                      if isinstance(upload_image, dict) else None)
        if (not isinstance(run_name, str) or not run_name
                or not isinstance(sample_id, str) or not sample_id
                or not isinstance(source_sha, str) or len(source_sha) != 64
                or not isinstance(upload_sha, str) or len(upload_sha) != 64):
            raise ValueError("external attempt claim identity is invalid")
        identity = "\0".join((run_name, sample_id, source_sha, upload_sha))
        claim_id = hashlib.sha256(identity.encode("utf-8")).hexdigest()
        claim = {
            "schema_version": SCHEMA_VERSION,
            "authorization_sha256": self.authorization_sha256,
            "run_name": run_name,
            "sample_id": sample_id,
            "source_image_sha256": source_sha,
            "upload_image_sha256": upload_sha,
        }
        path = self.directory / "attempts" / f"{claim_id}.json"
        try:
            _exclusive_json(path, claim)
        except FileExistsError as error:
            existing = _read_json(path)
            if existing != claim:
                raise ValueError("external attempt ledger identity differs") from error
            raise ValueError(
                "external attempt was already claimed; refusing a duplicate paid call") from error
        return {"claim_id": claim_id, "path": str(path)}

    def snapshot(self):
        attempts = self.directory / "attempts"
        count = len(list(attempts.glob("*.json"))) if attempts.is_dir() else 0
        session_sha = hashlib.sha256(self.session_path.read_bytes()).hexdigest()
        return {
            "schema_version": SCHEMA_VERSION,
            "authorization_sha256": self.authorization_sha256,
            "session_sha256": session_sha,
            "claimed_attempts": count,
        }
