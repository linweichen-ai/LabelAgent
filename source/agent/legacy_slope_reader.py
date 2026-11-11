"""Read immutable legacy slope evidence without promoting it to current events.

Legacy slope artifacts contain policy fields that the current writer deliberately
rejects.  This module verifies their bytes and exposes only archival metadata; it
never normalizes, migrates, or emits a current closed-loop event.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import re
from typing import Any


HEX64 = re.compile(r"^[0-9a-f]{64}$")
LEGACY_TRAVERSABILITY = {"unknown", "review_required"}
LEGACY_GEOMETRY_FIELDS = {
    "geometry_evidence_sufficient",
    "geometry_evidence_artifact_ids",
}


class LegacySlopeArtifactError(ValueError):
    """Raised when a legacy artifact cannot be verified or is misused."""


@dataclass(frozen=True)
class LegacySlopeSnapshot:
    artifact_path: str
    sha256: str
    old_event_id: str | None
    legacy_identifier: str
    legacy_fields: tuple[str, ...]
    archival_only: bool = True
    current_ingest_allowed: bool = False

    def as_current_event(self) -> dict[str, Any]:
        raise LegacySlopeArtifactError(
            "legacy slope artifacts cannot be emitted as current events"
        )


def read_legacy_slope_artifact(
    path: str | Path,
    *,
    expected_sha256: str,
    legacy_identifier: str | None = None,
) -> LegacySlopeSnapshot:
    """Verify one historical artifact and return archival metadata only."""
    if not isinstance(expected_sha256, str) or not HEX64.fullmatch(expected_sha256):
        raise LegacySlopeArtifactError("expected_sha256 must be lowercase SHA-256")
    artifact_path = Path(path)
    raw = artifact_path.read_bytes()
    actual_sha256 = hashlib.sha256(raw).hexdigest()
    if actual_sha256 != expected_sha256:
        raise LegacySlopeArtifactError(
            f"legacy artifact SHA-256 mismatch: expected {expected_sha256}, got {actual_sha256}"
        )
    try:
        payload = json.loads(raw)
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise LegacySlopeArtifactError("legacy artifact is not valid UTF-8 JSON") from error
    if not isinstance(payload, dict):
        raise LegacySlopeArtifactError("legacy artifact root must be an object")
    slope = payload.get("slope_context")
    if not isinstance(slope, dict):
        decision = payload.get("decision")
        if (isinstance(decision, dict)
                and decision.get("classification") == "slope"):
            slope = decision
        else:
            raise LegacySlopeArtifactError(
                "legacy artifact is missing slope_context or a slope decision"
            )

    legacy_fields = sorted(LEGACY_GEOMETRY_FIELDS.intersection(slope))
    if slope.get("traversability_status") in LEGACY_TRAVERSABILITY:
        legacy_fields.append("traversability_status")
    legacy_fields = sorted(set(legacy_fields))
    if not legacy_fields:
        raise LegacySlopeArtifactError(
            "artifact does not contain recognized legacy slope policy fields"
        )

    old_event_id = payload.get("event_id")
    if old_event_id is not None and not isinstance(old_event_id, str):
        raise LegacySlopeArtifactError("legacy event_id must be a string or null")
    identifier = legacy_identifier or old_event_id or payload.get("evidence_type")
    if not isinstance(identifier, str) or not identifier.strip():
        raise LegacySlopeArtifactError("legacy_identifier is required")
    return LegacySlopeSnapshot(
        artifact_path=str(artifact_path),
        sha256=actual_sha256,
        old_event_id=old_event_id,
        legacy_identifier=identifier.strip(),
        legacy_fields=tuple(legacy_fields),
    )
