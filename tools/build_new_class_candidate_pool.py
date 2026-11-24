"""Build a cross-frame, candidate-only new-class review pool.

The tool never mutates the label registry.  It deduplicates repeated machine
proposals, collapses only explicitly configured aliases, and separates known
label aliases from genuinely open-set names before human ontology review.
"""
import argparse
import hashlib
import json
import re
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path


INPUT_SCHEMA = "labelagent.new-class-observations.v1"
CONFIG_SCHEMA = "labelagent.new-class-policy.v1"
POOL_SCHEMA = "labelagent.new-class-candidate-pool.v1"
REVIEW_SCHEMA = "labelagent.new-class-review-template.v1"


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def file_sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def normalized_name(value):
    value = str(value or "").strip().casefold().replace("-", "_")
    value = re.sub(r"[\s/]+", "_", value)
    value = re.sub(r"[^\w\u4e00-\u9fff]+", "", value, flags=re.UNICODE)
    return re.sub(r"_+", "_", value).strip("_")


def _safe_boundaries(payload, label):
    if payload.get("candidate_only") is not True:
        raise ValueError(f"{label} must remain candidate_only")
    for key in ("human_truth", "approved_for_training", "test24_used"):
        if payload.get(key) is not False:
            raise ValueError(f"unsafe {label} boundary: {key}")


def _alias_index(groups, label):
    index = {}
    for canonical, aliases in groups.items():
        canonical_name = normalized_name(canonical)
        if not canonical_name:
            raise ValueError(f"empty canonical name in {label}")
        for alias in [canonical, *(aliases or [])]:
            key = normalized_name(alias)
            if not key:
                raise ValueError(f"empty alias in {label}")
            owner = index.setdefault(key, canonical_name)
            if owner != canonical_name:
                raise ValueError(f"ambiguous alias in {label}: {alias}")
    return index


def _observation_key(row):
    image_hash = row.get("image_sha256")
    if isinstance(image_hash, str) and len(image_hash) == 64:
        return image_hash
    sample_id = str(row.get("sample_id") or "").strip()
    if sample_id:
        return sample_id
    raise ValueError("observation requires sample_id or image_sha256")


def build_pool(observations_path, policy_path):
    payload, policy = read_json(observations_path), read_json(policy_path)
    if payload.get("schema_version") != INPUT_SCHEMA:
        raise ValueError("unsupported observation schema")
    if policy.get("schema_version") != CONFIG_SCHEMA:
        raise ValueError("unsupported new-class policy schema")
    _safe_boundaries(payload, "observation manifest")
    if policy.get("automatic_registry_mutation") is not False:
        raise ValueError("policy must disable automatic registry mutation")
    known = _alias_index(policy.get("known_labels") or {}, "known labels")
    candidate_aliases = _alias_index(
        policy.get("candidate_synonym_groups") or {}, "candidate synonym groups")
    for alias, canonical in candidate_aliases.items():
        if alias in known or canonical in known:
            raise ValueError(f"candidate alias overlaps known registry: {alias}")
    min_frames = int(policy.get("minimum_distinct_frames", 2))
    min_sequences = int(policy.get("minimum_distinct_sequences", 1))
    if min_frames < 2 or min_sequences < 1:
        raise ValueError("cross-frame thresholds are unsafe")

    observations = payload.get("observations")
    if not isinstance(observations, list):
        raise ValueError("observations must be a list")
    grouped, known_conflicts, ignored = defaultdict(dict), defaultdict(dict), []
    seen_observation_ids = set()
    for row in observations:
        if row.get("test24") is True or row.get("split") == "test24":
            raise ValueError("test24 observation is forbidden")
        observation_id = str(row.get("observation_id") or "").strip()
        if not observation_id or observation_id in seen_observation_ids:
            raise ValueError("missing or duplicate observation_id")
        seen_observation_ids.add(observation_id)
        frame_key = _observation_key(row)
        sequence = str(row.get("sequence") or "unknown")
        candidates = row.get("unknown_candidates")
        if not isinstance(candidates, list):
            raise ValueError(f"unknown_candidates must be a list: {observation_id}")
        within_frame = {}
        for candidate in candidates:
            raw_name = str(candidate.get("candidate_name") or "").strip()
            name = normalized_name(raw_name)
            evidence = str(candidate.get("evidence") or "uncertain").casefold()
            confidence = float(candidate.get("confidence", 0.0))
            if not name or not 0.0 <= confidence <= 1.0:
                raise ValueError(f"invalid new-class candidate: {observation_id}")
            if evidence == "absent":
                ignored.append({"observation_id": observation_id, "candidate_name": raw_name,
                                "reason": "absent_evidence"})
                continue
            if evidence not in {"present", "uncertain"}:
                raise ValueError(f"invalid evidence value: {observation_id}")
            canonical = known.get(name)
            destination = known_conflicts if canonical else grouped
            canonical = canonical or candidate_aliases.get(name, name)
            item = {
                "observation_id": observation_id,
                "sample_id": str(row.get("sample_id") or ""),
                "frame_key": frame_key,
                "sequence": sequence,
                "camera": row.get("camera"),
                "frame_id": row.get("frame_id"),
                "raw_candidate_name": raw_name,
                "normalized_candidate_name": name,
                "evidence": evidence,
                "confidence": confidence,
                "reason": str(candidate.get("reason") or ""),
            }
            prior = within_frame.get(("known" if destination is known_conflicts else "new",
                                      canonical))
            if prior is None or item["confidence"] > prior["confidence"]:
                within_frame[("known" if destination is known_conflicts else "new",
                              canonical)] = item
        for (kind, canonical), item in within_frame.items():
            destination = known_conflicts if kind == "known" else grouped
            destination[canonical][frame_key] = item

    def summarize(canonical, frame_map, status):
        items = sorted(frame_map.values(), key=lambda item: (
            item["sequence"], item["frame_key"], item["observation_id"]))
        names = sorted({item["raw_candidate_name"] for item in items},
                       key=lambda value: value.casefold())
        sequences = sorted({item["sequence"] for item in items})
        present_frames = sum(item["evidence"] == "present" for item in items)
        confidences = [item["confidence"] for item in items]
        return {
            "canonical_candidate_name": canonical,
            "observed_names": names,
            "status": status,
            "distinct_frames": len(items),
            "distinct_sequences": len(sequences),
            "sequences": sequences,
            "present_frames": present_frames,
            "uncertain_frames": len(items) - present_frames,
            "mean_confidence": round(sum(confidences) / len(confidences), 6),
            "max_confidence": max(confidences),
            "observations": items,
            "human_review_required": True,
            "downstream_value_confirmed": False,
            "ontology_mutation_allowed": False,
            "registry_mutated": False,
            "approved_for_training": False,
        }

    candidates = []
    for canonical, frame_map in sorted(grouped.items()):
        frames = len(frame_map)
        sequences = len({item["sequence"] for item in frame_map.values()})
        status = ("human_downstream_value_review_required"
                  if frames >= min_frames and sequences >= min_sequences
                  else "insufficient_cross_frame_support")
        candidates.append(summarize(canonical, frame_map, status))
    conflicts = [summarize(canonical, frame_map, "known_label_alias_not_new_class")
                 for canonical, frame_map in sorted(known_conflicts.items())]
    eligible = sum(item["status"] == "human_downstream_value_review_required"
                   for item in candidates)
    return {
        "schema_version": POOL_SCHEMA,
        "created_at": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "status": "candidate_only_human_ontology_review_required",
        "source_receipts": {
            "observations_sha256": file_sha256(observations_path),
            "policy_sha256": file_sha256(policy_path),
        },
        "thresholds": {"minimum_distinct_frames": min_frames,
                       "minimum_distinct_sequences": min_sequences},
        "summary": {
            "observations": len(observations),
            "new_name_groups": len(candidates),
            "human_review_queue": eligible,
            "insufficient_support_groups": len(candidates) - eligible,
            "known_label_alias_groups": len(conflicts),
            "ignored_absent_mentions": len(ignored),
            "automatic_registry_mutations": 0,
            "training_approved_candidates": 0,
        },
        "candidates": candidates,
        "known_label_conflicts": conflicts,
        "ignored_mentions": ignored,
        "candidate_only": True,
        "human_truth": False,
        "approved_for_training": False,
        "test24_used": False,
        "automatic_registry_mutation": False,
        "registry_mutated": False,
        "production_deployed": False,
    }


def review_template(pool, pool_sha256):
    rows = []
    for item in pool["candidates"]:
        if item["status"] != "human_downstream_value_review_required":
            continue
        rows.append({
            "canonical_candidate_name": item["canonical_candidate_name"],
            "observed_names": item["observed_names"],
            "distinct_frames": item["distinct_frames"],
            "distinct_sequences": item["distinct_sequences"],
            "ontology_decision": None,
            "downstream_value": None,
            "merge_target": None,
            "review_note": "",
        })
    return {
        "schema_version": REVIEW_SCHEMA,
        "candidate_pool_sha256": pool_sha256,
        "reviewer_declared_id": None,
        "rows": rows,
        "candidate_only": True,
        "human_truth": False,
        "approved_for_training": False,
        "test24_used": False,
        "automatic_registry_mutation": False,
        "registry_mutated": False,
    }


def validate_review_submission(pool_path, submission_path):
    pool_payload, submission = read_json(pool_path), read_json(submission_path)
    if pool_payload.get("schema_version") != POOL_SCHEMA:
        raise ValueError("unsupported candidate pool schema")
    if submission.get("schema_version") != REVIEW_SCHEMA:
        raise ValueError("unsupported ontology review schema")
    _safe_boundaries(submission, "ontology review")
    for key in ("automatic_registry_mutation", "registry_mutated"):
        if submission.get(key) is not False:
            raise ValueError(f"unsafe ontology review boundary: {key}")
    if submission.get("candidate_pool_sha256") != file_sha256(pool_path):
        raise ValueError("ontology review is bound to another candidate pool")
    reviewer = str(submission.get("reviewer_declared_id") or "").strip()
    if not reviewer:
        raise ValueError("reviewer_declared_id is required")
    expected = {
        item["canonical_candidate_name"]: item
        for item in pool_payload.get("candidates") or []
        if item.get("status") == "human_downstream_value_review_required"
    }
    rows = submission.get("rows")
    if not isinstance(rows, list) or len(rows) != len(expected):
        raise ValueError("ontology review row count differs")
    decisions = {"accept_candidate", "reject_candidate", "merge_known_label", "uncertain"}
    values = {"yes", "no", "uncertain"}
    seen, manual_proposals = set(), []
    for row in rows:
        name = row.get("canonical_candidate_name")
        if name in seen or name not in expected:
            raise ValueError("unknown or duplicate ontology candidate")
        seen.add(name)
        decision, downstream = row.get("ontology_decision"), row.get("downstream_value")
        if decision not in decisions or downstream not in values:
            raise ValueError(f"incomplete ontology review: {name}")
        merge_target = str(row.get("merge_target") or "").strip()
        if decision == "merge_known_label" and not merge_target:
            raise ValueError(f"merge_target is required: {name}")
        if decision != "merge_known_label" and merge_target:
            raise ValueError(f"merge_target only applies to merge decision: {name}")
        if decision == "accept_candidate" and downstream == "yes":
            manual_proposals.append(name)
    return {
        "status": "valid_human_ontology_review_record",
        "reviewer_declared_id": reviewer,
        "reviewed_candidates": len(rows),
        "manual_registry_proposals": manual_proposals,
        "manual_registry_change_required": bool(manual_proposals),
        "automatic_registry_mutations": 0,
        "registry_mutated": False,
        "candidate_only": True,
        "human_truth": False,
        "approved_for_training": False,
    }


def write_outputs(observations_path, policy_path, output):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    pool = build_pool(observations_path, policy_path)
    pool_path = output / "candidate-pool.json"
    pool_path.write_text(json.dumps(pool, ensure_ascii=False, indent=2) + "\n",
                         encoding="utf-8")
    template = review_template(pool, file_sha256(pool_path))
    (output / "human-ontology-review-template.json").write_text(
        json.dumps(template, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    receipt = {
        "schema_version": "labelagent.new-class-candidate-pool-receipt.v1",
        "candidate_pool_sha256": file_sha256(pool_path),
        "review_template_sha256": file_sha256(
            output / "human-ontology-review-template.json"),
        "candidate_only": True,
        "human_truth": False,
        "approved_for_training": False,
        "test24_used": False,
        "automatic_registry_mutation": False,
    }
    (output / "receipt.json").write_text(
        json.dumps(receipt, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return pool


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--observations")
    parser.add_argument("--policy")
    parser.add_argument("--output")
    parser.add_argument("--candidate-pool")
    parser.add_argument("--validate-review")
    args = parser.parse_args()
    if args.validate_review:
        if not args.candidate_pool:
            parser.error("--candidate-pool is required with --validate-review")
        print(json.dumps(validate_review_submission(
            args.candidate_pool, args.validate_review), ensure_ascii=False))
        return
    if not args.observations or not args.policy or not args.output:
        parser.error("--observations, --policy and --output are required to build a pool")
    pool = write_outputs(args.observations, args.policy, args.output)
    response = {"output": str(Path(args.output).resolve()), "summary": pool["summary"]}
    print(json.dumps(response, ensure_ascii=False))


if __name__ == "__main__":
    main()
