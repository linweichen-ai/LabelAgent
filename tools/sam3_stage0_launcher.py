"""Two-stage SAM3 launcher. Compile never creates a process; launch is gated."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
import sys

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from source.agent.stage0_training_contract import compile_dry_run, rejection_receipt


ROOT = Path(__file__).resolve().parents[1]


def load(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def compile_only(package_path, *, compiled_at):
    package_path = Path(package_path).resolve()
    plan = compile_dry_run(load(package_path), root=ROOT,
                           package_path=package_path, compiled_at=compiled_at)
    return {"plan": plan,
            "rejection": rejection_receipt(plan) if plan["status"] == "rejected_fail_closed" else None}


def explicit_launch(compiled, *, explicit_launch=False):
    """Second stage. Current checked-in package always fails before Popen."""
    plan = compiled["plan"]
    if not explicit_launch:
        raise PermissionError("second-stage launch requires --explicit-launch")
    if plan["status"] != "launch_authorized" or plan["blockers"]:
        raise PermissionError("Stage-0 launch rejected by compiled fail-closed plan")
    return subprocess.Popen(plan["command_preview"], cwd=ROOT)


def main():
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command", required=True)
    compile_parser = sub.add_parser("compile")
    compile_parser.add_argument("--package", type=Path, default=ROOT / "configs/sam3_sfl_cpd15y_stage0_training_package.json")
    compile_parser.add_argument("--compiled-at", required=True)
    launch_parser = sub.add_parser("launch")
    launch_parser.add_argument("--compiled-plan", type=Path, required=True)
    launch_parser.add_argument("--explicit-launch", action="store_true")
    args = parser.parse_args()
    if args.command == "compile":
        value = compile_only(args.package, compiled_at=args.compiled_at)
        json.dump(value, sys.stdout, ensure_ascii=False, indent=2, sort_keys=True)
        sys.stdout.write("\n")
        return
    compiled = load(args.compiled_plan)
    process = explicit_launch(compiled, explicit_launch=args.explicit_launch)
    print(json.dumps({"pid": process.pid, "status": "started_by_explicit_second_stage"}))


if __name__ == "__main__":
    main()
