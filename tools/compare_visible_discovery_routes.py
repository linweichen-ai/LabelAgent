"""Compare paired visible-label discovery routes without changing state.

The input files are produced by ``benchmark_visible_discovery.py``.  This
tool intentionally treats the primary route as a comparison baseline, not as
human truth.  It is a deployment guard: a faster route must not silently lose
known visible labels, open-set proposals, or consistency warnings.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from tools.benchmark_visible_discovery import bounded_image_name, median


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare paired primary and fast discovery benchmarks.")
    parser.add_argument("--primary", nargs="+", required=True, type=Path)
    parser.add_argument("--fast", nargs="+", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    return parser.parse_args()


def canonical(value) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True,
                      separators=(",", ":"))


def read_runs(paths: list[Path]) -> list[dict]:
    rows = []
    seen = set()
    for path in paths:
        payload = json.loads(path.expanduser().resolve().read_text(encoding="utf-8"))
        if payload.get("schema_version") != "labelagent.visible_discovery_benchmark.v1":
            raise ValueError(f"unsupported benchmark schema: {path}")
        if payload.get("read_only") is not True:
            raise ValueError(f"benchmark is not marked read-only: {path}")
        for row in payload.get("runs") or []:
            digest = row.get("image_sha256")
            if not isinstance(digest, str) or not digest:
                raise ValueError(f"benchmark row lacks image_sha256: {path}")
            if digest in seen:
                raise ValueError(f"duplicate image_sha256 across inputs: {digest}")
            seen.add(digest)
            rows.append(row)
    return rows


def assessment_map(row: dict) -> dict[str, dict]:
    result = {}
    for item in row.get("assessments") or []:
        if not isinstance(item, dict) or not isinstance(item.get("label"), str):
            raise ValueError("invalid assessment row")
        if item["label"] in result:
            raise ValueError(f"duplicate assessment label: {item['label']}")
        result[item["label"]] = item
    return result


def compare_routes(primary_runs: list[dict], fast_runs: list[dict]) -> dict:
    primary = {row.get("image_sha256"): row for row in primary_runs}
    fast = {row.get("image_sha256"): row for row in fast_runs}
    if len(primary) != len(primary_runs) or len(fast) != len(fast_runs):
        raise ValueError("duplicate image_sha256 in paired runs")
    if set(primary) != set(fast):
        raise ValueError("primary and fast image sets do not match")

    labels: dict[str, dict] = {}
    rows = []
    for digest in sorted(primary):
        base, candidate = primary[digest], fast[digest]
        base_assessments = assessment_map(base)
        fast_assessments = assessment_map(candidate)
        if set(base_assessments) != set(fast_assessments):
            raise ValueError(f"assessment label sets do not match: {digest}")
        evidence_differences = []
        for label in sorted(base_assessments):
            base_evidence = base_assessments[label].get("evidence")
            fast_evidence = fast_assessments[label].get("evidence")
            stats = labels.setdefault(label, {
                "total": 0, "evidence_exact": 0,
                "primary_present_fast_not": 0,
                "primary_not_fast_present": 0,
            })
            stats["total"] += 1
            if base_evidence == fast_evidence:
                stats["evidence_exact"] += 1
            else:
                evidence_differences.append({
                    "label": label, "primary": base_evidence,
                    "fast": fast_evidence,
                })
                if base_evidence == "present" and fast_evidence != "present":
                    stats["primary_present_fast_not"] += 1
                if base_evidence != "present" and fast_evidence == "present":
                    stats["primary_not_fast_present"] += 1

        base_visible = sorted(base.get("visible_labels") or [])
        fast_visible = sorted(candidate.get("visible_labels") or [])
        base_unknown = base.get("unknown_candidates") or []
        fast_unknown = candidate.get("unknown_candidates") or []
        base_flags = base.get("consistency_flags") or []
        fast_flags = candidate.get("consistency_flags") or []
        image_value = base.get("image") or "unknown"
        rows.append({
            "image": bounded_image_name(Path(image_value)),
            "image_sha256": digest,
            "evidence_exact": not evidence_differences,
            "visible_exact": base_visible == fast_visible,
            "unknown_exact": canonical(base_unknown) == canonical(fast_unknown),
            "consistency_flags_exact": canonical(base_flags) == canonical(fast_flags),
            "open_set_complete_exact": (
                base.get("open_set_scan_complete")
                == candidate.get("open_set_scan_complete")),
            "evidence_differences": evidence_differences,
            "primary_visible": base_visible,
            "fast_visible": fast_visible,
        })

    primary_latency = [float(row["elapsed_ms"]) for row in primary_runs
                       if row.get("failure") is None and row.get("elapsed_ms") is not None]
    fast_latency = [float(row["elapsed_ms"]) for row in fast_runs
                    if row.get("failure") is None and row.get("elapsed_ms") is not None]
    primary_median = median(primary_latency)
    fast_median = median(fast_latency)
    speedup = (round((1 - fast_median / primary_median) * 100, 2)
               if primary_median and fast_median is not None else None)
    all_semantics_exact = all(
        row["visible_exact"] and row["unknown_exact"]
        and row["consistency_flags_exact"] and row["open_set_complete_exact"]
        for row in rows)
    no_present_drift = all(
        item["primary_present_fast_not"] == 0
        and item["primary_not_fast_present"] == 0
        for item in labels.values())
    accepted = bool(rows and all_semantics_exact and no_present_drift
                    and speedup is not None and speedup > 0)
    return {
        "schema_version": "labelagent.visible_discovery_route_comparison.v1",
        "read_only": True,
        "candidate_written": False,
        "human_truth_modified": False,
        "training_data_modified": False,
        "primary_is_human_truth": False,
        "sample_count": len(rows),
        "primary_median_ms": primary_median,
        "fast_median_ms": fast_median,
        "median_speedup_pct": speedup,
        "visible_exact_count": sum(row["visible_exact"] for row in rows),
        "unknown_exact_count": sum(row["unknown_exact"] for row in rows),
        "evidence_exact_count": sum(row["evidence_exact"] for row in rows),
        "label_stats": labels,
        "authoritative_replacement_accepted": accepted,
        "decision": ("accepted_for_bounded_authoritative_route" if accepted
                     else "rejected_for_authoritative_route"),
        "mismatches": [row for row in rows if not (
            row["evidence_exact"] and row["visible_exact"]
            and row["unknown_exact"] and row["consistency_flags_exact"]
            and row["open_set_complete_exact"])],
    }


def main() -> int:
    args = parse_args()
    primary = read_runs(args.primary)
    fast = read_runs(args.fast)
    result = compare_routes(primary, fast)
    output = args.output.expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, ensure_ascii=False, indent=2),
                      encoding="utf-8")
    print(json.dumps({
        "output": str(output),
        "sample_count": result["sample_count"],
        "median_speedup_pct": result["median_speedup_pct"],
        "decision": result["decision"],
    }, ensure_ascii=False))
    return 0 if result["authoritative_replacement_accepted"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
