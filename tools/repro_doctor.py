"""Read-only reproducibility checks for a fresh LabelAgent checkout."""

from __future__ import annotations

import argparse
import importlib
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import socket
import sys
import urllib.error
import urllib.request


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_WORK = ROOT / "work"
DEFAULT_WEIGHT_DIR = ROOT / "source" / "sam3" / "weights"
CORE_IMPORTS = {
    "flask": "Flask",
    "av": "av",
    "mcap": "mcap",
    "mcap_protobuf": "mcap-protobuf-support",
    "numpy": "numpy",
    "cv2": "opencv-python-headless",
    "PIL": "Pillow",
}


def item(name: str, ok: bool, detail: str, *, required: bool = True) -> dict:
    return {"name": name, "ok": bool(ok), "required": required, "detail": detail}


def import_check(module: str, package: str) -> dict:
    try:
        importlib.import_module(module)
        try:
            version = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            version = "installed"
        return item(f"dependency:{package}", True, str(version))
    except Exception as error:  # Import failures often include useful DLL details.
        return item(f"dependency:{package}", False, f"{type(error).__name__}: {error}")


def url_json(url: str, timeout: float = 3.0) -> tuple[bool, str]:
    try:
        with urllib.request.urlopen(url, timeout=timeout) as response:
            payload = json.loads(response.read().decode("utf-8"))
        return True, json.dumps(payload, ensure_ascii=False, sort_keys=True)
    except (OSError, ValueError, urllib.error.URLError) as error:
        return False, f"{type(error).__name__}: {error}"


def port_available(port: int) -> bool:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as probe:
        probe.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        try:
            probe.bind(("127.0.0.1", port))
        except OSError:
            return False
    return True


def build_report(args: argparse.Namespace) -> dict:
    checks = []
    version_ok = sys.version_info[:2] == (3, 12)
    checks.append(item("python", version_ok,
                       f"{platform.python_version()} ({sys.executable})"))

    for relative in ("requirements.txt", "source/agent/server.py",
                     "source/sam3_service/server.py", "source/web/index.html"):
        path = ROOT / relative
        checks.append(item(f"file:{relative}", path.is_file(), str(path)))

    checks.extend(import_check(module, package)
                  for module, package in CORE_IMPORTS.items())

    try:
        torch = importlib.import_module("torch")
        cuda = bool(torch.cuda.is_available())
        detail = f"torch={torch.__version__}; cuda={cuda}"
        if cuda:
            detail += f"; device={torch.cuda.get_device_name(0)}"
        checks.append(item("pytorch-cuda", cuda, detail,
                           required=args.require_cuda))
    except Exception as error:
        checks.append(item("pytorch-cuda", False,
                           f"{type(error).__name__}: {error}",
                           required=args.require_cuda))

    weights = sorted(DEFAULT_WEIGHT_DIR.glob("*.pt"))
    weight_detail = ", ".join(f"{path.name} ({path.stat().st_size} bytes)"
                              for path in weights) or str(DEFAULT_WEIGHT_DIR)
    checks.append(item("sam3-weight", bool(weights), weight_detail,
                       required=args.require_weight))

    work = Path(os.environ.get("LABELAGENT_WORK_DIR") or args.work_dir).resolve()
    try:
        work.mkdir(parents=True, exist_ok=True)
        marker = work / ".labelagent-write-check"
        marker.write_text("ok", encoding="utf-8")
        marker.unlink()
        checks.append(item("work-directory", True, str(work)))
    except OSError as error:
        checks.append(item("work-directory", False, f"{work}: {error}"))

    if args.require_services:
        sam_ok, sam_detail = url_json(args.sam_url.rstrip("/") + "/health")
        web_ok, web_detail = url_json(args.web_url.rstrip("/") + "/api/run/status")
        checks.append(item("service:sam3", sam_ok, sam_detail))
        checks.append(item("service:workbench", web_ok, web_detail))
    else:
        web_port_available = port_available(8765)
        sam_port_available = port_available(8766)
        checks.append(item("port:8765", web_port_available,
                           "available" if web_port_available else "already in use",
                           required=False))
        checks.append(item("port:8766", sam_port_available,
                           "available" if sam_port_available else "already in use",
                           required=False))

    failed = [check for check in checks if check["required"] and not check["ok"]]
    return {
        "schema_version": 1,
        "root": str(ROOT),
        "platform": platform.platform(),
        "ready": not failed,
        "failed_required": [check["name"] for check in failed],
        "checks": checks,
    }


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Check whether this LabelAgent checkout is reproducible and runnable")
    parser.add_argument("--work-dir", default=str(DEFAULT_WORK))
    parser.add_argument("--sam-url", default="http://127.0.0.1:8766")
    parser.add_argument("--web-url", default="http://127.0.0.1:8765")
    parser.add_argument("--require-cuda", action="store_true")
    parser.add_argument("--require-weight", action="store_true")
    parser.add_argument("--require-services", action="store_true")
    parser.add_argument("--json", action="store_true", dest="as_json")
    args = parser.parse_args()
    report = build_report(args)
    if args.as_json:
        print(json.dumps(report, ensure_ascii=False, indent=2))
    else:
        for check in report["checks"]:
            state = "PASS" if check["ok"] else ("FAIL" if check["required"] else "INFO")
            print(f"[{state}] {check['name']}: {check['detail']}")
        print("READY" if report["ready"] else
              "NOT READY: " + ", ".join(report["failed_required"]))
    return 0 if report["ready"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
