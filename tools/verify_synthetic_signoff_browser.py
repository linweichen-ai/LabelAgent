"""Run a public, synthetic real-browser check of the offline signoff workflow."""

from __future__ import annotations

import argparse
import base64
import hashlib
import json
import sys
import tempfile
import threading
import time
from contextlib import suppress
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlparse

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools.build_cvat_human_signoff_reviewer import _page_html

EXPECTED_COUNT = 81
SYNTHETIC_REVIEWER = "synthetic-browser-reviewer"
SYNTHETIC_TIME = "2026-11-22T12:00:00+08:00"
ONE_PIXEL_PNG = "data:image/png;base64," + base64.b64encode(
    base64.b64decode(
        "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mNk+A8AAQUBAScY42YAAAAASUVORK5CYII="
    )
).decode("ascii")


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_file(path: Path) -> str:
    return sha256_bytes(path.read_bytes())


def synthetic_rows() -> list[dict]:
    rows = []
    for index in range(EXPECTED_COUNT):
        rows.append(
            {
                "candidate_id": f"synthetic-{index:03d}",
                "frame_id": index,
                "source_image_sha256": sha256_bytes(f"synthetic-source-{index}".encode()),
                "canonical_geometry_sha256": sha256_bytes(f"synthetic-geometry-{index}".encode()),
                "focused_image_sha256": sha256_bytes(f"synthetic-focused-{index}".encode()),
                "decision": None,
                "reviewer": None,
                "reviewed_at": None,
                "note": None,
            }
        )
    return rows


class _PageServer:
    def __init__(self, html: bytes):
        self.html = html
        owner = self

        class Handler(BaseHTTPRequestHandler):
            def do_GET(self):  # noqa: N802 - standard-library callback name
                parsed = urlparse(self.path)
                if parsed.path not in {"/", "/index.html"}:
                    self.send_error(404)
                    return
                self.send_response(200)
                self.send_header("Content-Type", "text/html; charset=utf-8")
                self.send_header("Cache-Control", "no-store")
                self.send_header("Content-Length", str(len(owner.html)))
                self.end_headers()
                self.wfile.write(owner.html)

            def log_message(self, _format, *_args):
                return

        self.server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)

    @property
    def url(self) -> str:
        host, port = self.server.server_address
        if host != "127.0.0.1":
            raise RuntimeError("synthetic reviewer server did not bind to loopback")
        return f"http://127.0.0.1:{port}"

    def __enter__(self):
        self.thread.start()
        return self

    def __exit__(self, *_exc):
        self.server.shutdown()
        self.server.server_close()
        self.thread.join(timeout=5)
        if self.thread.is_alive():
            raise RuntimeError("synthetic reviewer server did not stop")


def _download_json(page, selector: str, path: Path) -> dict:
    with page.expect_download(timeout=10_000) as pending:
        page.locator(selector).click()
    download = pending.value
    download.save_as(path)
    if download.failure():
        raise RuntimeError("browser download failed")
    return json.loads(path.read_text(encoding="utf-8"))


def run(output: Path, *, screenshot: bool = False) -> dict:
    try:
        from playwright.sync_api import sync_playwright
    except ImportError as exc:
        raise RuntimeError(
            "Playwright is required; install tools/requirements-browser-qa.txt in the project venv"
        ) from exc

    output = output.resolve()
    if output.exists():
        raise FileExistsError("output directory already exists")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.mkdir()
    started = time.monotonic()
    rows = synthetic_rows()
    manifest_sha = sha256_bytes(b"labelagent-public-synthetic-review-manifest-v1")
    template_sha = sha256_bytes(b"labelagent-public-synthetic-signoff-template-v1")
    html = _page_html(rows, [ONE_PIXEL_PNG] * EXPECTED_COUNT, manifest_sha, template_sha).encode("utf-8")
    blocked_requests: list[str] = []
    page_errors: list[str] = []

    with tempfile.TemporaryDirectory(prefix="labelagent-signoff-synthetic-") as temp_name:
        temp = Path(temp_name)
        draft_path = temp / "draft.json"
        bad_path = temp / "bad.json"
        full_path = temp / "full.json"
        official_path = temp / "official.json"
        with _PageServer(html) as server, sync_playwright() as runtime:
            browser = runtime.chromium.launch(channel="msedge", headless=True)
            try:
                page = browser.new_page(viewport={"width": 1440, "height": 960}, accept_downloads=True)
                page.on("pageerror", lambda error: page_errors.append(str(error)))
                page.on("console", lambda message: page_errors.append(message.text) if message.type == "error" else None)
                allowed = urlparse(server.url)

                def guard(route):
                    parsed = urlparse(route.request.url)
                    if parsed.scheme in {"data", "blob"} or (
                        parsed.scheme == "http"
                        and parsed.hostname == "127.0.0.1"
                        and parsed.port == allowed.port
                    ):
                        route.continue_()
                    else:
                        blocked_requests.append(f"{route.request.method} {parsed.scheme}://{parsed.netloc}{parsed.path}")
                        route.abort("blockedbyclient")

                page.route("**/*", guard)
                page.goto(server.url, wait_until="domcontentloaded", timeout=30_000)
                page.locator("#position").wait_for(state="visible")
                if page.locator("#position").text_content() != "1 / 81" or not page.locator("#export").is_disabled():
                    raise AssertionError("initial reviewer gate differs")

                page.locator('input[name="decision"][value="uncertain"]').check()
                page.locator("#reviewer").fill(SYNTHETIC_REVIEWER)
                page.locator("#reviewedAt").fill(SYNTHETIC_TIME)
                page.locator("#note").fill("public synthetic browser roundtrip")
                draft = _download_json(page, "#exportDraft", draft_path)
                if draft["rows"][0]["decision"] != "uncertain" or "training_authorized" in draft:
                    raise AssertionError("partial draft content differs")

                page.reload(wait_until="domcontentloaded")
                if page.locator("#reviewer").input_value():
                    raise AssertionError("refresh unexpectedly retained browser state")
                page.locator("#importDraft").set_input_files(draft_path)
                page.wait_for_function("document.querySelector('#status')?.textContent === '草稿已导入'")
                if page.locator("#reviewer").input_value() != SYNTHETIC_REVIEWER:
                    raise AssertionError("partial draft did not restore")

                bad = json.loads(json.dumps(draft))
                bad["rows"][0]["focused_image_sha256"] = "0" * 64
                bad_path.write_text(json.dumps(bad, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
                page.locator("#importDraft").set_input_files(bad_path)
                page.wait_for_function("document.querySelector('#status')?.textContent === '草稿导入失败'")
                if page.locator("#reviewer").input_value() != SYNTHETIC_REVIEWER:
                    raise AssertionError("failed import mutated restored state")
                if not page.locator('input[name="decision"][value="uncertain"]').is_checked():
                    raise AssertionError("failed import mutated decision")

                complete = json.loads(json.dumps(draft))
                decisions = ("pass", "reject", "uncertain")
                for index, row in enumerate(complete["rows"]):
                    row.update(
                        decision=decisions[index % len(decisions)],
                        reviewer=SYNTHETIC_REVIEWER,
                        reviewed_at=SYNTHETIC_TIME,
                        note=f"public-synthetic-{index}",
                    )
                full_path.write_text(json.dumps(complete, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
                page.locator("#importDraft").set_input_files(full_path)
                page.wait_for_function("document.querySelector('#status')?.textContent === '草稿已导入'")
                if page.locator("#counter").text_content() != "未完成 0 / 81" or not page.locator("#export").is_enabled():
                    raise AssertionError("complete synthetic draft did not unlock export")
                official = _download_json(page, "#export", official_path)
                for key in ("training_authorized", "automatic_approval", "approved_for_training"):
                    if official.get(key) is not False:
                        raise AssertionError(f"authority gate changed: {key}")
                if official.get("human_pass_is_training_authorization") is not False:
                    raise AssertionError("human pass became training authorization")
                if any(row.get("approved_for_training") is not False for row in official["rows"]):
                    raise AssertionError("row-level training authority changed")
                if blocked_requests or page_errors:
                    raise AssertionError("browser self-test observed external requests or page errors")
                if screenshot:
                    page.screenshot(path=output / "synthetic-reviewer.png", full_page=True)
            finally:
                browser.close()

        receipt = {
            "schema_version": "labelagent.synthetic-signoff-browser-self-test.v1",
            "passed": True,
            "synthetic_only": True,
            "candidate_count": EXPECTED_COUNT,
            "reviewer_page_sha256": sha256_bytes(html),
            "partial_draft_roundtrip": True,
            "refresh_cleared_memory_before_import": True,
            "invalid_binding_rejected_without_state_change": True,
            "official_export_blocked_before_completion": True,
            "official_export_enabled_after_complete_synthetic_draft": True,
            "downloaded_draft_sha256": sha256_file(draft_path),
            "downloaded_official_sha256": sha256_file(official_path),
            "training_authorized": False,
            "automatic_approval": False,
            "approved_for_training": False,
            "external_requests": len(blocked_requests),
            "page_errors": page_errors,
            "screenshot_captured": screenshot,
            "real_human_decisions_written": 0,
            "elapsed_ms": round((time.monotonic() - started) * 1000),
        }
    receipt_path = output / "receipt.json"
    with receipt_path.open("x", encoding="utf-8", newline="\n") as handle:
        json.dump(receipt, handle, ensure_ascii=False, indent=2)
        handle.write("\n")
    return receipt


def cli() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True, help="new output directory")
    parser.add_argument("--screenshot", action="store_true", help="save a synthetic-only screenshot")
    args = parser.parse_args()
    try:
        receipt = run(args.output, screenshot=args.screenshot)
    except Exception as exc:
        with suppress(Exception):
            if args.output.exists() and not any(args.output.iterdir()):
                args.output.rmdir()
        parser.error(str(exc))
    print(json.dumps(receipt, ensure_ascii=False, separators=(",", ":")))
    return 0


if __name__ == "__main__":
    raise SystemExit(cli())
