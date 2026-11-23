#!/usr/bin/env python3
"""Exercise the same-file reload warning on an isolated workbench.

The check intentionally reloads the active file through the server API.  Its
path and topic therefore stay unchanged while its activation generation moves
forward, exactly as after a service restart or another window reload.  It
refuses the production port and any non-loopback address.  It never writes
truth, candidates, model configuration, or training state.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from urllib.parse import urlparse

from playwright.sync_api import sync_playwright


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", required=True)
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args()

    base_url = args.base_url.rstrip("/")
    parsed = urlparse(base_url)
    if parsed.scheme != "http" or parsed.hostname not in {"127.0.0.1", "localhost"}:
        raise ValueError("acceptance is restricted to a loopback workbench")
    if parsed.port in {None, 8765}:
        raise ValueError("cross-window acceptance refuses the production workbench")

    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    truth_writes: list[str] = []
    errors: list[str] = []
    started = time.monotonic()

    with sync_playwright() as runtime:
        browser = runtime.chromium.launch(channel="msedge", headless=True)
        try:
            page = browser.new_page(
                viewport={"width": 1760, "height": 1120}, device_scale_factor=1
            )
            page.on("pageerror", lambda error: errors.append(str(error)))

            def on_request(request):
                path = urlparse(request.url).path
                if request.method != "GET" and path in {"/api/truth", "/api/human_review"}:
                    truth_writes.append(f"{request.method} {request.url}")

            page.on("request", on_request)
            response = page.goto(base_url, wait_until="domcontentloaded", timeout=60_000)
            assert response is not None and response.status == 200
            page.evaluate("() => window.labelAgentWorkspaceReady")

            initial_scope = page.evaluate("""async () => {
                const response = await fetch('/api/bootstrap', {cache: 'no-store'});
                const data = await response.json();
                return {path: data.path, topic: data.topic,
                    file_activation_generation: data.file_activation_generation};
            }""")
            assert 0 < initial_scope["file_activation_generation"] <= (1 << 53) - 1
            target = page.evaluate("""async () => {
                const response = await fetch('/api/files', {cache: 'no-store'});
                const data = await response.json();
                const item = (data.files || []).find(file => file.available && file.loaded);
                if (!item) throw new Error('isolated workbench has no loaded file');
                return {id: item.id, path: item.path, name: item.name};
            }""")
            switched = page.evaluate("""async target => {
                const response = await fetch('/api/files/' + encodeURIComponent(target.id)
                    + '/load', {method: 'POST', headers: {'Content-Type': 'application/json'},
                    body: '{}'});
                const data = await response.json();
                if (!response.ok) throw new Error(data.error || 'same-file reload failed');
                return {path: data.path, topic: data.topic,
                    file_activation_generation: data.file_activation_generation};
            }""", target)
            assert switched["path"] == initial_scope["path"]
            assert switched["topic"] == initial_scope["topic"]
            assert switched["file_activation_generation"] > initial_scope[
                "file_activation_generation"
            ]

            status_scope = page.evaluate("""async () => {
                const response = await fetch('/api/run/status', {cache: 'no-store'});
                const data = await response.json();
                if (!response.ok) throw new Error(data.error || 'run status failed');
                return {path: data.selected_file, topic: data.selected_topic,
                    file_activation_generation:
                        data.selected_file_activation_generation};
            }""")
            assert status_scope == switched

            page.evaluate("() => refreshAgentProgress()")
            page.wait_for_function(
                "() => document.querySelector('#runButton')?.textContent.includes('切换数据')"
            )
            result = page.evaluate("""() => ({
                button_text: document.querySelector('#runButton')?.textContent,
                button_disabled: document.querySelector('#runButton')?.disabled,
                button_title: document.querySelector('#runButton')?.title,
                toast: document.querySelector('#toast')?.textContent,
                toast_visible: getComputedStyle(document.querySelector('#toast')).display !== 'none',
            })""")
            assert result["button_disabled"] is True
            assert "切换数据" in result["button_text"]
            assert "已安全暂停" in result["toast"]
            assert result["toast_visible"] is True
            assert truth_writes == []
            assert errors == []

            screenshot = output / "cross-window-scope-warning.png"
            page.screenshot(path=str(screenshot), full_page=True)
            receipt = {
                "schema_version": "labelagent.cross-window-scope-acceptance.v2",
                "passed": True,
                "elapsed_ms": round((time.monotonic() - started) * 1000, 1),
                "base_url": base_url,
                "initial_scope": initial_scope,
                "target": target,
                "server_scope": switched,
                "run_status_scope": status_scope,
                "ui": result,
                "truth_write_requests": truth_writes,
                "page_errors": errors,
                "screenshot": str(screenshot),
                "isolated_only": True,
            }
            (output / "result.json").write_text(
                json.dumps(receipt, ensure_ascii=False, indent=2) + "\n",
                encoding="utf-8",
            )
            print(json.dumps(receipt, ensure_ascii=False, separators=(",", ":")))
        finally:
            browser.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
