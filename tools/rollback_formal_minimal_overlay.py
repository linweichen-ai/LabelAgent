#!/usr/bin/env python3
"""Explicit, hash-gated rollback for a formal minimal overlay deployment."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import shutil
import sys
import tempfile
import time

try:
    from tools.deploy_formal_minimal_overlay import contained, probe_idle, sha256
except ModuleNotFoundError:
    from deploy_formal_minimal_overlay import contained, probe_idle, sha256


DEPLOY_SCHEMA = "labelagent.formal-minimal-overlay-deploy-receipt.v1"
ROLLBACK_SCHEMA = "labelagent.formal-minimal-overlay-rollback-receipt.v1"


def load_plan(receipt_path: Path, target_root: Path) -> tuple[dict, list[dict]]:
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    if not isinstance(receipt, dict) or receipt.get("schema_version") != DEPLOY_SCHEMA:
        raise ValueError("unsupported deployment receipt")
    if receipt.get("status") != "deployed" or receipt.get("explicit_execute") is not True:
        raise ValueError("receipt does not describe an executed deployment")
    backup_root_value = receipt.get("backup_root")
    if not isinstance(backup_root_value, str) or not backup_root_value:
        raise ValueError("deployment receipt has no backup root")
    backup_root = Path(backup_root_value).resolve(strict=True)
    if receipt_path.resolve().parent != backup_root:
        raise ValueError("deployment receipt is not located in its backup root")
    rows = receipt.get("deployed_files")
    if not isinstance(rows, list) or not rows:
        raise ValueError("deployment receipt has no deployed files")

    plan = []
    seen = set()
    for row in rows:
        if not isinstance(row, dict):
            raise ValueError("invalid deployed file row")
        relative = row.get("relative")
        if relative in seen:
            raise ValueError(f"duplicate deployed file: {relative}")
        seen.add(relative)
        target = contained(target_root, relative)
        current = sha256(target) if target.is_file() else None
        new_hash = row.get("new_sha256")
        old_hash = row.get("old_sha256")
        if current not in (new_hash, old_hash):
            raise RuntimeError(f"formal target drifted after deployment: {relative}")
        backup_value = row.get("backup")
        backup = None
        if old_hash is not None:
            if not isinstance(backup_value, str) or not backup_value:
                raise ValueError(f"missing rollback backup: {relative}")
            backup = Path(backup_value).resolve(strict=True)
            try:
                backup.relative_to(backup_root)
            except ValueError as exc:
                raise ValueError(f"backup escapes receipt root: {relative}") from exc
            if sha256(backup) != old_hash:
                raise RuntimeError(f"rollback backup hash mismatch: {relative}")
        elif backup_value is not None:
            raise ValueError(f"unexpected backup for newly created file: {relative}")
        plan.append({"relative": relative, "target": target, "backup": backup,
                     "old_sha256": old_hash, "new_sha256": new_hash,
                     "needs_rollback": current == new_hash})
    return receipt, plan


def rollback(plan: list[dict], backup_root: Path) -> list[dict]:
    undo_root = backup_root / f"rollback-undo-{time.strftime('%Y%m%d-%H%M%S')}"
    undo_root.mkdir(parents=False, exist_ok=False)
    prepared = []
    applied = []
    try:
        for row in plan:
            if not row["needs_rollback"]:
                continue
            undo = contained(undo_root, row["relative"])
            undo.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(row["target"], undo)
            if sha256(undo) != row["new_sha256"]:
                raise RuntimeError(f"rollback undo hash mismatch: {row['relative']}")
            restore = None
            if row["backup"] is not None:
                fd, name = tempfile.mkstemp(prefix=f".{row['target'].name}.",
                                            suffix=".rollback.tmp", dir=row["target"].parent)
                os.close(fd)
                restore = Path(name)
                shutil.copyfile(row["backup"], restore)
                if sha256(restore) != row["old_sha256"]:
                    raise RuntimeError(f"prepared rollback hash mismatch: {row['relative']}")
            prepared.append((row, restore, undo))

        for row, restore, undo in prepared:
            if restore is None:
                row["target"].unlink()
            else:
                os.replace(restore, row["target"])
            applied.append((row, undo))
            observed = sha256(row["target"]) if row["target"].is_file() else None
            if observed != row["old_sha256"]:
                raise RuntimeError(f"rollback readback mismatch: {row['relative']}")
    except Exception:
        for row, undo in reversed(applied):
            restore_new = row["target"].with_name(f".{row['target'].name}.reapply.tmp")
            shutil.copyfile(undo, restore_new)
            os.replace(restore_new, row["target"])
            if sha256(row["target"]) != row["new_sha256"]:
                raise RuntimeError(f"rollback failure recovery failed: {row['relative']}")
        raise
    finally:
        for _, restore, _ in prepared:
            if restore is not None:
                restore.unlink(missing_ok=True)
    return [{"relative": row["relative"], "restored_sha256": row["old_sha256"]}
            for row, _, _ in prepared]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--receipt", type=Path, required=True)
    parser.add_argument("--target-root", type=Path, required=True)
    parser.add_argument("--status-url", default="")
    parser.add_argument("--status-timeout", type=float, default=5.0)
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()
    receipt_path = args.receipt.resolve(strict=True)
    target_root = args.target_root.resolve(strict=True)
    deployment, plan = load_plan(receipt_path, target_root)
    pending = [row["relative"] for row in plan if row["needs_rollback"]]
    idle = None
    if args.execute:
        if not args.status_url:
            raise ValueError("--status-url is required with --execute")
        idle = probe_idle(args.status_url, args.status_timeout)
    result = {
        "schema_version": ROLLBACK_SCHEMA,
        "status": "dry_run" if not args.execute else "already_rolled_back" if not pending else "rolled_back",
        "explicit_execute": bool(args.execute), "idle_probe": idle,
        "pending_files": pending, "restored_files": [],
        "candidate_only": True, "human_truth": False, "approved_for_training": False,
    }
    if args.execute and pending:
        backup_root = Path(deployment["backup_root"]).resolve(strict=True)
        result["restored_files"] = rollback(plan, backup_root)
        (backup_root / "rollback-receipt.json").write_text(
            json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        raise SystemExit(1)
