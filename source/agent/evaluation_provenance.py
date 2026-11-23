"""Fail-closed provenance classification for saved evaluation targets."""


def classify_target_provenance(session, frames, target_labels):
    """Classify saved targets without trusting their historical storage key.

    ``truth`` is only a file-format name. Human truth requires every selected
    frame to be bound to the current contract and exact annotation hash.
    """
    frame_keys = [str(frame) for frame in frames]
    target_labels = set(target_labels)
    truth_by_frame = session.get("truth") or {}
    review_status = session.get("truth_review_status") or {}
    source_counts = {"manual": 0, "auto": 0, "unknown": 0}
    boundary_counts = {
        "explicit_human_truth_true": 0,
        "explicit_human_truth_false": 0,
        "candidate_only_true": 0,
        "approved_for_training_true": 0,
    }
    selected_shape_count = 0
    for frame_key in frame_keys:
        for shape in truth_by_frame.get(frame_key, []):
            if shape.get("label") not in target_labels:
                continue
            selected_shape_count += 1
            metadata = shape.get("cvat_metadata")
            source = metadata.get("source") if isinstance(metadata, dict) else None
            source_counts[source if source in ("manual", "auto") else "unknown"] += 1
            boundary_counts["explicit_human_truth_true"] += shape.get("human_truth") is True
            boundary_counts["explicit_human_truth_false"] += shape.get("human_truth") is False
            boundary_counts["candidate_only_true"] += shape.get("candidate_only") is True
            boundary_counts["approved_for_training_true"] += (
                shape.get("approved_for_training") is True)

    verified_frames = []
    unverified_frames = []
    review_reasons = {}
    for frame_key in frame_keys:
        status = review_status.get(frame_key) if isinstance(review_status, dict) else None
        checks = status.get("checks") if isinstance(status, dict) else None
        verified = bool(
            isinstance(status, dict)
            and status.get("current_human_review_verified") is True
            and status.get("reusable_as_reference") is True
            and status.get("reason_code") == "verified_current_human_truth"
            and isinstance(checks, dict)
            and checks.get("nonempty_truth") is True
            and checks.get("current_contract") is True
            and checks.get("truth_sha256_matches") is True
            and checks.get("transaction_committed") is True
        )
        (verified_frames if verified else unverified_frames).append(int(frame_key))
        review_reasons[frame_key] = (status.get("reason_code")
                                     if isinstance(status, dict)
                                     else "missing_current_human_review_receipt")

    all_verified = bool(frame_keys) and not unverified_frames
    has_machine_boundary = bool(
        boundary_counts["explicit_human_truth_false"]
        or boundary_counts["candidate_only_true"])
    if all_verified:
        kind = "human_reviewed"
        note = ("Every evaluated frame has a current committed human-review receipt "
                "bound to the exact saved annotation hash.")
    elif has_machine_boundary:
        kind = "machine_candidate"
        note = ("At least one saved target is explicitly machine-only; scores measure "
                "candidate agreement and are not human-truth accuracy.")
    elif selected_shape_count and source_counts["auto"] == selected_shape_count:
        kind = "legacy_auto_preannotation"
        note = ("All selected targets have legacy CVAT source=auto and lack current "
                "human-review receipts; scores measure auto-preannotation consistency, "
                "not human-truth accuracy or generalization.")
    else:
        kind = "unverified_saved_annotations"
        note = ("Saved targets lack current human-review receipts; their accuracy "
                "provenance is unverified and scores must not be presented as human truth.")
    return {
        "kind": kind,
        "human_truth_verified": all_verified,
        "frame_count": len(frame_keys),
        "selected_shape_count": selected_shape_count,
        "source_counts": source_counts,
        "boundary_counts": boundary_counts,
        "verified_frames": verified_frames,
        "unverified_frames": unverified_frames,
        "review_reason_by_frame": review_reasons,
        "evidence_note": note,
    }
