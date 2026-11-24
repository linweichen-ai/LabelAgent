"""Strict human-referenced three-arm LabelAgent benchmark."""
import math
import statistics


INPUT_SCHEMA = "labelagent.three-arm-eval-input.v1"
OUTPUT_SCHEMA = "labelagent.three-arm-eval-report.v1"
ARMS = ("local_only", "all_external", "hybrid")
LABELS = ("车辆", "pallet", "person", "floor", "slope", "ego",
          "other_robot", "cargo_floor")


def _ratio(numerator, denominator):
    return round(numerator / denominator, 6) if denominator else None


def _f1_counts(tp, fp, fn):
    denominator = 2 * tp + fp + fn
    return round(2 * tp / denominator, 6) if denominator else None


def _percentile(values, percentile):
    if not values:
        return None
    ordered = sorted(values)
    return round(ordered[max(0, math.ceil(percentile * len(ordered)) - 1)], 3)


def _nonnegative(value, field, integer=False):
    allowed = (int,) if integer else (int, float)
    if type(value) not in allowed or value < 0:
        raise ValueError(f"{field} must be non-negative")
    return int(value) if integer else float(value)


def _validate_reference(payload):
    if payload.get("schema_version") != INPUT_SCHEMA:
        raise ValueError("three-arm input schema is invalid")
    partition = payload.get("partition")
    if (not isinstance(partition, str) or not partition.strip()
            or "test24" in partition.casefold() or payload.get("test24") is not False):
        raise ValueError("test24 is evaluation-only and cannot enter three-arm development A/B")
    synthetic = payload.get("synthetic_fixture")
    if type(synthetic) is not bool:
        raise ValueError("synthetic_fixture must be explicit")
    if synthetic:
        if payload.get("human_reference") is not False or payload.get("reviewer_confirmed") is not False:
            raise ValueError("synthetic fixture cannot claim human reference")
    elif payload.get("human_reference") is not True or payload.get("reviewer_confirmed") is not True:
        raise ValueError("real quality metrics require reviewer-confirmed human reference")
    samples = payload.get("samples")
    if not isinstance(samples, list) or not 1 <= len(samples) <= 500:
        raise ValueError("three-arm A/B requires 1-500 samples")
    references = {}
    for sample in samples:
        if not isinstance(sample, dict):
            raise ValueError("reference sample must be an object")
        sample_id = sample.get("sample_id")
        presence = sample.get("class_presence")
        if (not isinstance(sample_id, str) or not sample_id.strip()
                or sample_id in references or not isinstance(presence, dict)
                or set(presence) != set(LABELS)
                or any(value not in {"present", "absent"} for value in presence.values())):
            raise ValueError("reference sample identity or class_presence is invalid")
        pallet_count = _nonnegative(sample.get("pallet_instance_count"),
                                    "pallet_instance_count", integer=True)
        if (presence["pallet"] == "absent") != (pallet_count == 0):
            raise ValueError("pallet presence and instance count disagree")
        references[sample_id] = {"class_presence": presence,
                                 "pallet_instance_count": pallet_count}
    return partition.strip(), references, synthetic


def _validate_prediction(prediction):
    if not isinstance(prediction, dict):
        raise ValueError("passed run requires a prediction")
    labels = prediction.get("present_labels")
    if (not isinstance(labels, list) or any(label not in LABELS for label in labels)
            or len(set(labels)) != len(labels)):
        raise ValueError("present_labels must be unique known labels")
    values = {key: _nonnegative(prediction.get(key), key, integer=True) for key in (
        "pallet_instance_count", "pallet_hole_errors", "pallet_merge_errors",
        "vehicle_robot_conflicts")}
    if ("pallet" not in labels) != (values["pallet_instance_count"] == 0):
        raise ValueError("predicted pallet presence and instance count disagree")
    return {"present_labels": set(labels), **values}


def _validate_run(run, references):
    required = {"arm", "sample_id", "status", "external_called",
                "fallback_occurred", "timeout", "latency_ms", "estimated_cost_usd",
                "sam_artifact_sha256", "candidate_only", "human_truth",
                "approved_for_training"}
    if not isinstance(run, dict) or not required.issubset(run):
        raise ValueError("three-arm run fields are incomplete")
    arm, sample_id, status = run["arm"], run["sample_id"], run["status"]
    if arm not in ARMS or sample_id not in references or status not in {"passed", "failed"}:
        raise ValueError("three-arm run identity or status is invalid")
    if (run["candidate_only"] is not True or run["human_truth"] is not False
            or run["approved_for_training"] is not False):
        raise ValueError("three-arm run violates candidate-only boundaries")
    for key in ("external_called", "fallback_occurred", "timeout"):
        if type(run[key]) is not bool:
            raise ValueError(f"{key} must be boolean")
    if arm == "local_only" and (run["external_called"] or run["fallback_occurred"]):
        raise ValueError("local_only cannot call or fall back from external model")
    if arm == "all_external" and not run["external_called"]:
        raise ValueError("all_external must call the external model")
    if run["fallback_occurred"] and not run["external_called"]:
        raise ValueError("fallback requires an external call attempt")
    if run["timeout"] and not run["external_called"]:
        raise ValueError("external timeout requires an external call")
    latency = _nonnegative(run["latency_ms"], "latency_ms")
    cost = _nonnegative(run["estimated_cost_usd"], "estimated_cost_usd")
    sam_hash = run["sam_artifact_sha256"]
    if (not isinstance(sam_hash, str) or len(sam_hash) != 64
            or any(character not in "0123456789abcdefABCDEF" for character in sam_hash)):
        raise ValueError("sam_artifact_sha256 is invalid")
    if status == "failed":
        if "prediction" in run:
            raise ValueError("failed run cannot contain a prediction")
        prediction = None
    else:
        prediction = _validate_prediction(run.get("prediction"))
    return {"arm": arm, "sample_id": sample_id, "status": status,
            "external_called": run["external_called"],
            "fallback_occurred": run["fallback_occurred"], "timeout": run["timeout"],
            "latency_ms": latency, "estimated_cost_usd": cost,
            "sam_artifact_sha256": sam_hash.casefold(), "prediction": prediction}


def build_three_arm_report(payload):
    if not isinstance(payload, dict):
        raise ValueError("three-arm input must be an object")
    partition, references, synthetic = _validate_reference(payload)
    runs = payload.get("runs")
    if not isinstance(runs, list) or len(runs) != len(references) * len(ARMS):
        raise ValueError("three-arm A/B requires exactly one run per arm and sample")
    validated, seen = [], set()
    for run in runs:
        item = _validate_run(run, references)
        identity = (item["arm"], item["sample_id"])
        if identity in seen:
            raise ValueError("duplicate arm/sample run")
        seen.add(identity)
        validated.append(item)
    expected = {(arm, sample_id) for arm in ARMS for sample_id in references}
    if seen != expected:
        raise ValueError("three-arm coverage is incomplete")
    for sample_id in references:
        hashes = {item["sam_artifact_sha256"] for item in validated
                  if item["sample_id"] == sample_id}
        if len(hashes) != 1:
            raise ValueError(f"SAM artifact was not reused across arms: {sample_id}")

    reports = []
    for arm in ARMS:
        arm_runs = [item for item in validated if item["arm"] == arm]
        counts = {label: {"tp": 0, "fp": 0, "fn": 0} for label in LABELS}
        reference_pallets = recovered_pallets = missed_pallets = extra_pallets = 0
        holes = merges = conflicts = slope_fp_on_floor = 0
        for item in arm_runs:
            reference = references[item["sample_id"]]
            predicted = item["prediction"]
            predicted_labels = predicted["present_labels"] if predicted else set()
            if (reference["class_presence"]["slope"] == "absent"
                    and reference["class_presence"]["floor"] == "present"
                    and "slope" in predicted_labels):
                slope_fp_on_floor += 1
            for label in LABELS:
                expected_present = reference["class_presence"][label] == "present"
                predicted_present = label in predicted_labels
                if expected_present and predicted_present:
                    counts[label]["tp"] += 1
                elif predicted_present:
                    counts[label]["fp"] += 1
                elif expected_present:
                    counts[label]["fn"] += 1
            reference_count = reference["pallet_instance_count"]
            predicted_count = predicted["pallet_instance_count"] if predicted else 0
            reference_pallets += reference_count
            recovered_pallets += min(reference_count, predicted_count)
            missed_pallets += max(0, reference_count - predicted_count)
            extra_pallets += max(0, predicted_count - reference_count)
            if predicted:
                holes += predicted["pallet_hole_errors"]
                merges += predicted["pallet_merge_errors"]
                conflicts += predicted["vehicle_robot_conflicts"]
        per_class, total = {}, {"tp": 0, "fp": 0, "fn": 0}
        for label in LABELS:
            item = counts[label]
            precision = _ratio(item["tp"], item["tp"] + item["fp"])
            recall = _ratio(item["tp"], item["tp"] + item["fn"])
            per_class[label] = {**item, "precision": precision, "recall": recall,
                                "f1": _f1_counts(item["tp"], item["fp"], item["fn"])}
            for key in total:
                total[key] += item[key]
        micro_precision = _ratio(total["tp"], total["tp"] + total["fp"])
        micro_recall = _ratio(total["tp"], total["tp"] + total["fn"])
        slope_negative_floor_frames = sum(
            ref["class_presence"]["slope"] == "absent"
            and ref["class_presence"]["floor"] == "present" for ref in references.values())
        reports.append({
            "arm": arm,
            "attempted_samples": len(arm_runs),
            "successful_samples": sum(item["status"] == "passed" for item in arm_runs),
            "failed_samples": sum(item["status"] == "failed" for item in arm_runs),
            "timeout_count": sum(item["timeout"] for item in arm_runs),
            "fallback_count": sum(item["fallback_occurred"] for item in arm_runs),
            "external_calls": sum(item["external_called"] for item in arm_runs),
            "external_call_ratio": _ratio(sum(item["external_called"] for item in arm_runs),
                                           len(arm_runs)),
            "mean_latency_ms": round(statistics.mean(item["latency_ms"] for item in arm_runs), 3),
            "p95_latency_ms": _percentile([item["latency_ms"] for item in arm_runs], 0.95),
            "estimated_cost_usd_total": round(sum(
                item["estimated_cost_usd"] for item in arm_runs), 6),
            "class_metrics": per_class,
            "micro_precision": micro_precision,
            "micro_recall": micro_recall,
            "micro_f1": _f1_counts(total["tp"], total["fp"], total["fn"]),
            "slope_positive_recall": per_class["slope"]["recall"],
            "slope_false_positives_on_floor": slope_fp_on_floor,
            "slope_negative_floor_frames": slope_negative_floor_frames,
            "slope_false_positive_rate_on_floor": _ratio(
                slope_fp_on_floor, slope_negative_floor_frames),
            "pallet_reference_instances": reference_pallets,
            "pallet_recovered_instances": recovered_pallets,
            "pallet_instance_support_rate": _ratio(recovered_pallets, reference_pallets),
            "pallet_missed_instances": missed_pallets,
            "pallet_extra_instances": extra_pallets,
            "pallet_hole_errors": holes,
            "pallet_merge_errors": merges,
            "vehicle_robot_class_conflicts": conflicts,
            "sam_artifacts_reused_across_arms": True,
        })
    return {
        "schema_version": OUTPUT_SCHEMA,
        "partition": partition,
        "reference_mode": "synthetic_fixture" if synthetic else "reviewer_confirmed_human",
        "synthetic_fixture": synthetic,
        "human_reference": not synthetic,
        "reviewer_confirmed": not synthetic,
        "real_world_quality_evidence": not synthetic,
        "test24_used": False,
        "sample_count": len(references),
        "arms": reports,
        "slope_and_floor_kept_separate_for_analysis": True,
        "downstream_traversable_union_allowed": True,
        "automatic_winner_selected": False,
        "deployment_decision": "not_evaluated",
        "candidate_only": True,
        "human_truth_written": False,
        "approved_for_training": False,
    }
