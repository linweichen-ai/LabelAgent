#!/usr/bin/env python3
"""Hash-gated, explicit deployment of a frozen formal overlay candidate."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import sys
import tempfile
import time
from urllib.request import Request, urlopen


SCHEMA = "labelagent.formal-minimal-overlay-candidate.v1"
RECEIPT_SCHEMA = "labelagent.formal-minimal-overlay-deploy-receipt.v1"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def contained(root: Path, relative: str) -> Path:
    if not relative or Path(relative).is_absolute():
        raise ValueError(f"unsafe relative path: {relative!r}")
    target = (root / Path(relative.replace("/", os.sep))).resolve()
    try:
        target.relative_to(root.resolve())
    except ValueError as exc:
        raise ValueError(f"path escapes declared root: {relative}") from exc
    return target


def read_json(path: Path) -> dict:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def probe_idle(url: str, timeout: float) -> dict:
    request = Request(url, headers={"Accept": "application/json"})
    with urlopen(request, timeout=timeout) as response:
        payload = json.loads(response.read().decode("utf-8"))
    if not isinstance(payload, dict) or payload.get("status") != "idle":
        raise RuntimeError("formal service is not confirmed idle")
    return {"status": "idle"}


def load_plan(candidate_root: Path, target_root: Path) -> tuple[dict, list[dict]]:
    manifest_path = candidate_root / "manifest.json"
    manifest = read_json(manifest_path)
    if manifest.get("schema_version") != SCHEMA:
        raise ValueError("unsupported candidate manifest schema")
    if manifest.get("candidate_validated") is not True:
        raise ValueError("candidate is not validated")
    if manifest.get("deployment_performed") is not False:
        raise ValueError("candidate manifest already claims deployment")
    if manifest.get("formal_files_written") != 0:
        raise ValueError("candidate manifest claims formal writes")
    rows = manifest.get("files")
    if not isinstance(rows, list) or not rows:
        raise ValueError("candidate manifest has no files")

    plan = []
    seen = set()
    for row in rows:
        if not isinstance(row, dict):
            raise ValueError("invalid candidate file row")
        relative = row.get("relative")
        if relative in seen:
            raise ValueError(f"duplicate candidate file: {relative}")
        seen.add(relative)
        source = contained(candidate_root / "files", relative)
        target = contained(target_root, relative)
        if not source.is_file():
            raise FileNotFoundError(f"missing candidate file: {relative}")
        new_hash = sha256(source)
        if new_hash != row.get("candidate_sha256"):
            raise RuntimeError(f"candidate hash mismatch: {relative}")
        exists = target.is_file()
        old_hash = sha256(target) if exists else None
        expected = row.get("formal_sha256")
        if old_hash != expected and old_hash != new_hash:
            raise RuntimeError(f"formal target hash mismatch: {relative}")
        plan.append({
            "relative": relative,
            "source": source,
            "target": target,
            "old_sha256": old_hash,
            "new_sha256": new_hash,
            "existed_before": exists,
            "needs_update": old_hash != new_hash,
        })
    return manifest, plan


def deploy(plan: list[dict], target_root: Path, backup_root: Path) -> list[dict]:
    backup_root.mkdir(parents=True, exist_ok=False)
    prepared: list[tuple[dict, Path, Path | None]] = []
    applied: list[tuple[dict, Path | None]] = []
    try:
        for row in plan:
            if not row["needs_update"]:
                continue
            target = row["target"]
            if not target.parent.is_dir():
                raise FileNotFoundError(f"target directory missing: {target.parent}")
            fd, temporary_name = tempfile.mkstemp(
                prefix=f".{target.name}.", suffix=".tmp", dir=target.parent
            )
            os.close(fd)
            temporary = Path(temporary_name)
            shutil.copyfile(row["source"], temporary)
            if sha256(temporary) != row["new_sha256"]:
                raise RuntimeError(f"prepared hash mismatch: {row['relative']}")
            backup = contained(backup_root, row["relative"]) if row["existed_before"] else None
            if backup is not None:
                backup.parent.mkdir(parents=True, exist_ok=True)
                if backup.exists():
                    raise FileExistsError(f"backup already exists: {backup}")
                shutil.copyfile(target, backup)
                if sha256(backup) != row["old_sha256"]:
                    raise RuntimeError(f"backup hash mismatch: {row['relative']}")
            prepared.append((row, temporary, backup))

        for row, temporary, backup in prepared:
            os.replace(temporary, row["target"])
            if sha256(row["target"]) != row["new_sha256"]:
                raise RuntimeError(f"deployed hash mismatch: {row['relative']}")
            applied.append((row, backup))
    except Exception:
        for row, backup in reversed(applied):
            if backup is None:
                row["target"].unlink(missing_ok=True)
            else:
                restore = row["target"].with_name(f".{row['target'].name}.restore.tmp")
                shutil.copyfile(backup, restore)
                os.replace(restore, row["target"])
            if (sha256(row["target"]) if row["target"].is_file() else None) != row["old_sha256"]:
                raise RuntimeError(f"rollback readback failed: {row['relative']}")
        raise
    finally:
        for _, temporary, _ in prepared:
            temporary.unlink(missing_ok=True)

    return [{
        "relative": row["relative"],
        "old_sha256": row["old_sha256"],
        "new_sha256": row["new_sha256"],
        "backup": str(backup) if backup else None,
    } for row, backup in applied]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--candidate-root", type=Path, required=True)
    parser.add_argument("--target-root", type=Path, required=True)
    parser.add_argument("--status-url", default="")
    parser.add_argument("--status-timeout", type=float, default=5.0)
    parser.add_argument("--backup-root", type=Path)
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()

    candidate_root = args.candidate_root.resolve(strict=True)
    target_root = args.target_root.resolve(strict=True)
    manifest, plan = load_plan(candidate_root, target_root)
    pending = [row for row in plan if row["needs_update"]]
    idle = None
    if args.execute:
        if not args.status_url:
            raise ValueError("--status-url is required with --execute")
        idle = probe_idle(args.status_url, args.status_timeout)

    stamp = time.strftime("%Y%m%d-%H%M%S")
    backup_root = (args.backup_root or target_root / ".local" / "deploy-backups" /
                   f"formal-minimal-overlay-{stamp}").resolve()
    receipt = {
        "schema_version": RECEIPT_SCHEMA,
        "status": "dry_run" if not args.execute else "already_current" if not pending else "deployed",
        "candidate_schema": manifest["schema_version"],
        "created_from_formal_head": manifest.get("created_from_formal_head"),
        "explicit_execute": bool(args.execute),
        "idle_probe": idle,
        "pending_files": [row["relative"] for row in pending],
        "deployed_files": [],
        "backup_root": str(backup_root) if pending else None,
        "candidate_only": True,
        "human_truth": False,
        "approved_for_training": False,
    }
    if args.execute and pending:
        receipt["deployed_files"] = deploy(plan, target_root, backup_root)
        (backup_root / "deploy-receipt.json").write_text(
            json.dumps(receipt, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
    print(json.dumps(receipt, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        raise SystemExit(1)
