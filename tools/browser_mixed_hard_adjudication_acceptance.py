#!/usr/bin/env python3
"""Real-browser acceptance for a receipt-bound mixed-hard review workspace."""

from __future__ import annotations

import argparse
import hashlib
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import mimetypes
from pathlib import Path, PurePosixPath
import tempfile
from threading import Thread
import time
from typing import Any
from urllib.parse import unquote, urlparse


RECEIPT_SCHEMA = "labelagent.mixed-hard-adjudication-workspace-receipt.v2"
LABELS = ("车辆", "pallet", "person", "floor", "slope", "ego",
          "other_robot", "cargo_floor")


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def validate_workspace(workspace: Path | str) -> dict[str, Any]:
    root = Path(workspace).resolve()
    receipt_path = root / "receipt.json"
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    if receipt.get("schema_version") != RECEIPT_SCHEMA:
        raise ValueError("unsupported review workspace receipt")
    for key, expected in {
        "candidate_only": True,
        "human_truth": False,
        "approved_for_training": False,
        "automatic_approval": False,
        "assistant_hint_blind_gate": True,
    }.items():
        if receipt.get(key) is not expected:
            raise ValueError(f"unsafe review workspace boundary: {key}")
    files = receipt.get("files")
    if not isinstance(files, dict) or "reviewer.html" not in files:
        raise ValueError("review workspace files are missing")
    verified: dict[str, str] = {}
    for relative, expected in files.items():
        pure = PurePosixPath(relative)
        if (not relative or relative.startswith("/") or "\\" in relative
                or any(part in {"", ".", ".."} for part in pure.parts)
                or not isinstance(expected, str) or len(expected) != 64):
            raise ValueError("unsafe or invalid receipt-bound path")
        path = (root / Path(*pure.parts)).resolve()
        if root not in path.parents or not path.is_file():
            raise ValueError(f"receipt-bound file is unavailable: {relative}")
        actual = sha256(path)
        if actual != expected:
            raise ValueError(f"receipt-bound file hash differs: {relative}")
        verified[pure.as_posix()] = actual
    return {"root": root, "receipt": receipt, "files": verified,
            "receipt_sha256": sha256(receipt_path)}


class _WorkspaceServer:
    def __init__(self, validated: dict[str, Any]):
        self.validated = validated
        owner = self

        class Handler(BaseHTTPRequestHandler):
            def do_GET(self):  # noqa: N802
                parsed = urlparse(self.path)
                relative = "reviewer.html" if parsed.path in {"/", "/index.html"} else unquote(parsed.path.lstrip("/"))
                expected = owner.validated["files"].get(relative)
                if self.command != "GET" or expected is None:
                    self.send_error(404)
                    return
                path = owner.validated["root"] / Path(*PurePosixPath(relative).parts)
                data = path.read_bytes()
                if hashlib.sha256(data).hexdigest() != expected:
                    self.send_error(404)
                    return
                self.send_response(200)
                self.send_header("Content-Type", mimetypes.guess_type(path.name)[0] or "application/octet-stream")
                self.send_header("Cache-Control", "private, no-store")
                self.send_header("X-Content-Type-Options", "nosniff")
                if relative == "reviewer.html":
                    self.send_header("Content-Security-Policy",
                                     "default-src 'none'; img-src 'self'; style-src 'unsafe-inline'; script-src 'unsafe-inline'; connect-src 'none'; object-src 'none'; base-uri 'none'; form-action 'none'")
                self.send_header("Content-Length", str(len(data)))
                self.end_headers()
                self.wfile.write(data)

            def log_message(self, *_args):
                return

        self.server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self.thread = Thread(target=self.server.serve_forever, daemon=True)

    def __enter__(self):
        self.thread.start()
        host, port = self.server.server_address
        if host != "127.0.0.1":
            raise RuntimeError("review acceptance server is not loopback-only")
        return f"http://127.0.0.1:{port}/"

    def __exit__(self, *_args):
        self.server.shutdown()
        self.server.server_close()
        self.thread.join(timeout=5)


def _values(page) -> dict[str, Any]:
    return page.evaluate("""() => ({
      candidate: document.querySelector('#candidate').value,
      slope: document.querySelector('#slope').value,
      conflict: document.querySelector('#conflict').value,
      pallet: document.querySelector('#pallet').value,
      palletCount: document.querySelector('#palletCount').value,
      presence: [...document.querySelectorAll('#presence select')].map(x => x.value)
    })""")


def run_acceptance(workspace: Path | str, output: Path | str,
                   screenshot: bool = False) -> dict[str, Any]:
    validated = validate_workspace(workspace)
    output = Path(output).resolve()
    if output.exists():
        raise FileExistsError(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.mkdir()
    from playwright.sync_api import sync_playwright

    blocked_requests: list[str] = []
    page_errors: list[str] = []
    dialogs: list[str] = []
    started = time.perf_counter()
    with tempfile.TemporaryDirectory(prefix="labelagent-hard-review-") as temporary:
        draft_path = Path(temporary) / "synthetic-draft.json"
        with _WorkspaceServer(validated) as url, sync_playwright() as runtime:
            browser = runtime.chromium.launch(channel="msedge", headless=True)
            try:
                page = browser.new_page(viewport={"width": 1440, "height": 1000},
                                        accept_downloads=True)
                page.on("pageerror", lambda error: page_errors.append(str(error)))
                page.on("console", lambda message: page_errors.append(message.text)
                        if message.type == "error" else None)

                def route_request(route):
                    parsed = urlparse(route.request.url)
                    expected_port = urlparse(url).port
                    if parsed.scheme in {"data", "blob"} or (
                            parsed.scheme == "http" and parsed.hostname == "127.0.0.1"
                            and parsed.port == expected_port):
                        route.continue_()
                    else:
                        blocked_requests.append(
                            f"{route.request.method} {parsed.scheme}://{parsed.netloc}{parsed.path}")
                        route.abort("blockedbyclient")

                page.route("**/*", route_request)
                page.goto(url, wait_until="domcontentloaded", timeout=30_000)
                page.locator("#title").wait_for(state="visible")
                first_sample = page.locator("#title").text_content()
                if page.locator("#priorityBadge").get_attribute("class") != "visible":
                    raise AssertionError("first review item is not priority-bound")
                initial = _values(page)
                if any((initial["candidate"], initial["slope"], initial["conflict"], initial["pallet"])):
                    raise AssertionError("review answers were preselected")
                if any(initial["presence"]):
                    raise AssertionError("class presence was preselected")
                overlap_count = page.evaluate("""() => {
                  const action = document.querySelector('.actions').getBoundingClientRect();
                  return [...document.querySelectorAll('select,input,textarea,details')]
                    .filter(element => {
                      const target = element.getBoundingClientRect();
                      const width = Math.max(0, Math.min(action.right, target.right) - Math.max(action.left, target.left));
                      const height = Math.max(0, Math.min(action.bottom, target.bottom) - Math.max(action.top, target.top));
                      return width * height > 0;
                    }).length;
                }""")
                if overlap_count:
                    raise AssertionError(
                        f"action bar overlaps {overlap_count} form controls")

                page.once("dialog", lambda dialog: (
                    dialogs.append(dialog.message), dialog.dismiss()))
                page.locator("#assistantPanel summary").click()
                page.wait_for_timeout(100)
                if page.locator("#assistantPanel").get_attribute("open") is not None:
                    raise AssertionError("assistant hint opened before blind review")
                if not dialogs or "请先完整填写本帧独立判断" not in dialogs[-1]:
                    raise AssertionError("blind-review refusal was not explained")

                page.select_option("#candidate", "accept_candidate")
                page.select_option("#slope", "present")
                page.select_option("#conflict", "none")
                page.select_option("#pallet", "complete")
                for index, label in enumerate(LABELS):
                    page.select_option(f"#presence-{index}",
                                       "present" if label in {"pallet", "floor", "slope"} else "absent")
                page.fill("#palletCount", "1")
                page.fill("#reviewer", "synthetic-browser-acceptance")
                page.fill("#note", "合成浏览器验收，不是人工真值")
                completed = _values(page)
                page.locator("#assistantPanel summary").click()
                page.wait_for_function("document.querySelector('#assistantPanel').open === true")
                hint = page.locator("#assistantHint").text_content()
                if not hint or "斜坡形态" not in hint:
                    raise AssertionError("assistant hint did not render after blind review")
                if _values(page) != completed:
                    raise AssertionError("assistant hint changed review answers")

                page.reload(wait_until="domcontentloaded")
                page.locator("#title").wait_for(state="visible")
                if page.locator("#title").text_content() != first_sample:
                    raise AssertionError("refresh changed current review item")
                if _values(page) != completed:
                    raise AssertionError("refresh did not preserve local draft")
                if page.locator("#assistantPanel").get_attribute("open") is not None:
                    raise AssertionError("assistant hint reopened after refresh")

                with page.expect_download() as download_info:
                    page.locator("#exportDraft").click()
                download_info.value.save_as(draft_path)
                draft = json.loads(draft_path.read_text(encoding="utf-8"))
                if (draft.get("candidate_only") is not True
                        or draft.get("human_truth") is not False
                        or draft.get("approved_for_training") is not False
                        or draft.get("training_authorized") is not False
                        or draft.get("automatic_approval") is not False):
                    raise AssertionError("exported draft changed safety boundaries")
                if len(draft.get("rows", [])) != 34:
                    raise AssertionError("exported draft row count differs")
                first_row = next(row for row in draft["rows"]
                                 if row.get("sample_id") == first_sample)
                if first_row.get("slope_presence") != "present":
                    raise AssertionError("exported draft lost blind review decision")

                page.once("dialog", lambda dialog: (
                    dialogs.append(dialog.message), dialog.dismiss()))
                page.locator("#export").click()
                page.wait_for_timeout(100)
                if not dialogs or "请填写复核人" not in dialogs[-1]:
                    raise AssertionError("incomplete official export was not rejected")
                if screenshot:
                    page.evaluate("window.scrollTo(0, 0)")
                    page.wait_for_timeout(100)
                    page.screenshot(path=output / "blind-review-browser.png",
                                    full_page=True)
            finally:
                browser.close()

        receipt = {
            "schema_version": "labelagent.mixed-hard-adjudication-browser-acceptance.v1",
            "passed": True,
            "workspace_receipt_sha256": validated["receipt_sha256"],
            "reviewer_html_sha256": validated["files"]["reviewer.html"],
            "first_priority_sample": first_sample,
            "blank_answers_initially": True,
            "form_control_overlap_count": overlap_count,
            "hint_blocked_before_complete_review": True,
            "hint_available_after_complete_review": True,
            "hint_changed_answers": False,
            "draft_persisted_after_refresh": True,
            "hint_closed_after_refresh": True,
            "partial_draft_exported": True,
            "official_export_blocked_while_incomplete": True,
            "candidate_only": True,
            "human_truth": False,
            "approved_for_training": False,
            "training_authorized": False,
            "automatic_approval": False,
            "external_requests": len(blocked_requests),
            "blocked_requests": blocked_requests,
            "page_errors": page_errors,
            "screenshot_captured": screenshot,
            "synthetic_answers_only": True,
            "elapsed_ms": round((time.perf_counter() - started) * 1000, 3),
        }
        if blocked_requests or page_errors:
            raise AssertionError("browser acceptance observed external requests or page errors")
        (output / "receipt.json").write_text(
            json.dumps(receipt, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8")
        return receipt


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workspace", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--screenshot", action="store_true")
    args = parser.parse_args()
    try:
        receipt = run_acceptance(args.workspace, args.output, args.screenshot)
    except Exception as error:
        failure = {"schema_version":
                   "labelagent.mixed-hard-adjudication-browser-failure.v1",
                   "passed": False, "error_type": type(error).__name__,
                   "error": str(error)}
        if args.output.exists() and args.output.is_dir() and not any(args.output.iterdir()):
            (args.output / "failure.json").write_text(
                json.dumps(failure, ensure_ascii=False, indent=2) + "\n",
                encoding="utf-8")
        print(json.dumps(failure, ensure_ascii=False))
        return 2
    print(json.dumps(receipt, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
