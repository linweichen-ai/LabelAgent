"""Hash-bound archival reader for immutable historical evidence.

This reader deliberately returns metadata only.  Historical bytes may be
audited, but they cannot be normalized into or emitted as current events.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
from pathlib import Path
import re


HEX64 = re.compile(r"^[0-9a-f]{64}$")


class LegacyEvidenceError(ValueError):
    """Raised when immutable historical evidence fails its archival gate."""


@dataclass(frozen=True)
class LegacyEvidenceSnapshot:
    artifact_path: str
    sha256: str
    bytes: int
    prior_event_id: str
    archival_only: bool = True
    current_ingest_allowed: bool = False

    def as_current_event(self):
        raise LegacyEvidenceError(
            "archival evidence cannot be normalized or emitted as a current event"
        )


def read_legacy_evidence(
    path: str | Path,
    *,
    expected_sha256: str,
    prior_event_id: str,
) -> LegacyEvidenceSnapshot:
    if not isinstance(expected_sha256, str) or not HEX64.fullmatch(expected_sha256):
        raise LegacyEvidenceError("expected_sha256 must be lowercase SHA-256")
    if not isinstance(prior_event_id, str) or not prior_event_id.strip():
        raise LegacyEvidenceError("prior_event_id must be a non-empty string")
    artifact_path = Path(path)
    raw = artifact_path.read_bytes()
    actual = hashlib.sha256(raw).hexdigest()
    if actual != expected_sha256:
        raise LegacyEvidenceError(
            f"legacy evidence SHA-256 mismatch: expected {expected_sha256}, got {actual}"
        )
    return LegacyEvidenceSnapshot(
        artifact_path=str(artifact_path),
        sha256=actual,
        bytes=len(raw),
        prior_event_id=prior_event_id.strip(),
    )
