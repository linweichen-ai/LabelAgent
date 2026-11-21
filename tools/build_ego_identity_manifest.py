#!/usr/bin/env python3
"""Build a portable, hash-bound ego sample identity manifest."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import re

try:
    from tools.build_ego_component_union_review import (
        _verified_bytes, load_identity_manifest)
except ModuleNotFoundError:  # Direct ``python tools/...py`` execution.
    from build_ego_component_union_review import (  # type: ignore
        _verified_bytes, load_identity_manifest)


def _parts(sample_id: str) -> tuple[str, str, str]:
    match = re.fullmatch(r"(\d{3})-(camera[13])-(?:[A-Za-z0-9_]+-)?(\d+)",
                         sample_id)
    if match is None:
        raise ValueError(f"invalid sample identity: {sample_id}")
    return match.group(1), match.group(2), match.group(3)


def build_manifest(existing_manifest: Path | None,
                   diagnostic_sources: list[tuple[Path, Path]],
                   output: Path, dataset: str) -> dict:
    if output.exists():
        raise FileExistsError(output)
    samples = (load_identity_manifest(existing_manifest)
               if existing_manifest is not None else {})
    samples = {name: dict(item) for name, item in samples.items()}
    added = 0
    for diagnostic_root, allowed_root in diagnostic_sources:
        allowed = allowed_root.resolve(strict=True)
        for sample_dir in sorted(diagnostic_root.iterdir()):
            report_path = sample_dir / "results.json"
            if not sample_dir.is_dir() or not report_path.is_file():
                continue
            sample_id = sample_dir.name
            package, camera, frame_id = _parts(sample_id)
            report = json.loads(report_path.read_text(encoding="utf-8"))
            if (report.get("candidate_only") is not True
                    or report.get("human_truth") is not False
                    or report.get("approved_for_training") is not False
                    or report.get("production_deployed") is not False
                    or report.get("source_state_modified") is not False):
                raise ValueError(f"diagnostic safety flags are inconsistent: {sample_id}")
            image_path = Path(report.get("image", ""))
            _verified_bytes(image_path, report.get("image_sha256"), "source image")
            image_resolved = image_path.resolve(strict=True)
            try:
                relative = image_resolved.relative_to(allowed)
            except ValueError as error:
                raise ValueError(
                    f"source image is outside allowed root: {sample_id}") from error
            entry = {
                "package": package,
                "camera": camera,
                "frame_id": frame_id,
                "image_sha256": report["image_sha256"],
                "relative_path": relative.as_posix(),
            }
            previous = samples.get(sample_id)
            if previous is not None and previous != entry:
                raise ValueError(f"conflicting duplicate sample identity: {sample_id}")
            if previous is None:
                samples[sample_id] = entry
                added += 1
    if not added:
        raise ValueError("no new diagnostic samples were added")
    payload = {
        "schema_version": "labelagent.ego_sample_identity_manifest.v2",
        "dataset": dataset,
        "candidate_only": True,
        "human_truth": False,
        "approved_for_training": False,
        "production_deployed": False,
        "sample_count": len(samples),
        "samples": {name: samples[name] for name in sorted(samples)},
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
                      encoding="utf-8")
    return payload


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--existing-manifest", type=Path)
    parser.add_argument("--diagnostic-root", action="append", type=Path, required=True)
    parser.add_argument("--allowed-source-root", action="append", type=Path,
                        required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--dataset", required=True)
    arguments = parser.parse_args()
    if len(arguments.diagnostic_root) != len(arguments.allowed_source_root):
        parser.error("each --diagnostic-root needs one --allowed-source-root")
    payload = build_manifest(
        arguments.existing_manifest,
        list(zip(arguments.diagnostic_root, arguments.allowed_source_root)),
        arguments.output, arguments.dataset)
    print(json.dumps({
        "output": str(arguments.output.resolve()),
        "sample_count": payload["sample_count"],
    }, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
