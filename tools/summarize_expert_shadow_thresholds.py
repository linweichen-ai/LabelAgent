"""Summarize bounded expert-routing thresholds from an upload-free audit."""
import argparse
import json
from pathlib import Path


DEFAULT_HARD_CODES = (
    "edge_slope_candidate",
    "class_audit_conflict",
    "agent_nonconvergence",
    "open_set_candidate",
    "registry_conflict",
    "protocol_failure",
    "manual_priority",
)


def summarize(audit, *, thresholds, hard_codes=DEFAULT_HARD_CODES):
    if audit.get("schema_version") != "labelagent.expert-shadow-routing-audit.v2":
        raise ValueError("unsupported expert shadow audit schema")
    if audit.get("dispatch_performed") is not False:
        raise ValueError("threshold calibration requires an upload-free audit")
    if audit.get("external_upload_performed") is not False:
        raise ValueError("threshold calibration requires zero external uploads")
    if audit.get("candidate_only") is not True:
        raise ValueError("threshold calibration requires candidate-only rows")
    frames = audit.get("frames")
    if not isinstance(frames, list) or not frames:
        raise ValueError("expert shadow audit contains no frames")
    normalized_thresholds = sorted(set(thresholds))
    if (not normalized_thresholds
            or any(type(value) is not int or not 1 <= value <= 100
                   for value in normalized_thresholds)):
        raise ValueError("thresholds must be integers from 1 to 100")
    hard_codes = tuple(dict.fromkeys(hard_codes))
    rows = []
    for threshold in normalized_thresholds:
        selected = []
        for frame in frames:
            score = frame.get("difficulty_score")
            reasons = frame.get("reason_codes")
            if type(score) is not int or not isinstance(reasons, list):
                raise ValueError("invalid frame routing evidence")
            hard_matches = sorted(set(reasons).intersection(hard_codes))
            if score >= threshold or hard_matches:
                selected.append({
                    "frame": frame.get("frame"),
                    "difficulty_score": score,
                    "selected_by_score": score >= threshold,
                    "hard_signal_codes": hard_matches,
                })
        reason_coverage = {
            code: {
                "available": sum(code in frame["reason_codes"] for frame in frames),
                "selected": sum(
                    code in frame["reason_codes"]
                    and any(item["frame"] == frame.get("frame")
                            for item in selected)
                    for frame in frames),
            }
            for code in hard_codes
        }
        rows.append({
            "threshold": threshold,
            "selected_count": len(selected),
            "external_call_ratio": round(len(selected) / len(frames), 4),
            "reason_coverage": reason_coverage,
            "selected_frames": selected,
        })
    return {
        "schema_version": "labelagent.expert-shadow-threshold-summary.v1",
        "frames_audited": len(frames),
        "hard_signal_codes": list(hard_codes),
        "thresholds": rows,
        "dispatch_performed": False,
        "external_upload_performed": False,
        "candidate_only": True,
        "human_truth": False,
        "approved_for_training": False,
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--audit", type=Path, required=True)
    parser.add_argument("--threshold", type=int, action="append", required=True)
    parser.add_argument("--output", type=Path, required=True)
    arguments = parser.parse_args(argv)
    audit = json.loads(arguments.audit.read_text(encoding="utf-8"))
    result = summarize(audit, thresholds=arguments.threshold)
    arguments.output.parent.mkdir(parents=True, exist_ok=True)
    arguments.output.write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8")
    print(json.dumps({key: value for key, value in result.items()
                      if key != "thresholds"}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
