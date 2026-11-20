#!/usr/bin/env python3
"""Browser acceptance for live background progress in the reference review guide.

The check is intentionally read-only: it exercises the UI with a synthetic job
status and rejects any truth or human-review write request.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from urllib.parse import urlparse

from playwright.sync_api import sync_playwright


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", default="http://127.0.0.1:8898")
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args()

    parsed = urlparse(args.base_url)
    if parsed.scheme != "http" or parsed.hostname not in {"127.0.0.1", "localhost"}:
        raise ValueError("acceptance is restricted to a loopback workbench")
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    screenshot = output / "reference-background-progress.png"
    receipt = output / "result.json"
    page_errors: list[str] = []
    failed_requests: list[str] = []
    http_errors: list[str] = []
    truth_writes: list[str] = []

    with sync_playwright() as runtime:
        browser = runtime.chromium.launch(channel="msedge", headless=True)
        try:
            page = browser.new_page(viewport={"width": 1760, "height": 1120})
            page.on("pageerror", lambda error: page_errors.append(str(error)))
            page.on(
                "console",
                lambda message: page_errors.append(message.text)
                if message.type == "error" else None,
            )
            page.on(
                "requestfailed",
                lambda request: failed_requests.append(
                    f"{request.method} {request.url} {request.failure or ''}"
                ),
            )

            def record_response(response):
                if response.status >= 400:
                    http_errors.append(
                        f"{response.request.method} {response.status} {response.url}"
                    )

            def record_request(request):
                path = urlparse(request.url).path
                if request.method != "GET" and path in {"/api/truth", "/api/human_review"}:
                    truth_writes.append(f"{request.method} {request.url}")

            page.on("response", record_response)
            page.on("request", record_request)
            response = page.goto(args.base_url, wait_until="domcontentloaded", timeout=60_000)
            assert response is not None and response.status == 200
            page.evaluate("() => window.labelAgentBootstrapReady")
            if page.evaluate("() => document.body.dataset.workspace") != "manual":
                page.locator("#manualModeButton").click()
                page.wait_for_function(
                    "() => document.body.dataset.workspace === 'manual'"
                )
            scope = page.evaluate("""() => ({
                path: loadedDataPath, topic, frame,
                uiVersion: document.querySelector('script[src^="/app.js"]')?.src,
            })""")
            assert "app.js?v=203" in scope["uiVersion"]
            page.evaluate("""() => {
                referencePreparationIntent={source:'machine_candidate',path:loadedDataPath,topic,frame};
                const guide=document.querySelector('#referenceFinalizeGuide');
                guide.hidden=false;
                const status=document.querySelector('#referenceFinalizeStatus');
                status.dataset.baseStatus='轮廓草稿待人工确认';
                status.textContent=status.dataset.baseStatus;
                renderAgentProgress({status:'running',phase:'agent_refining',frame,
                    phase_elapsed_seconds:7,total:1,completed:0,shape_count:9,
                    max_rounds:6,events:[]});
            }""")
            refining = page.locator("#referenceFinalizeStatus").inner_text()
            assert "初始候选已显示" in refining
            assert "可先逐个检查" in refining
            assert "不会写入人工真值" in refining
            page.evaluate("""() => renderAgentProgress({
                status:'running',phase:'candidate_class_audit',frame,
                phase_elapsed_seconds:13,total:1,completed:0,shape_count:9,
                audit_completed:2,audit_total:5,max_rounds:6,events:[]})""")
            auditing = page.locator("#referenceFinalizeStatus").inner_text()
            assert "后台复核 2/5" in auditing
            assert "不会写入人工真值" in auditing
            page.locator("#referenceFinalizeGuide").scroll_into_view_if_needed()
            page.screenshot(path=str(screenshot), full_page=True)
            page.evaluate("""() => renderAgentProgress({
                status:'completed',phase:'completed',frame,total:1,completed:1,
                shape_count:9,max_rounds:6,events:[]})""")
            completed = page.locator("#referenceFinalizeStatus").inner_text()
            assert completed == "轮廓草稿待人工确认"
            assert not truth_writes
            assert not page_errors
            assert not failed_requests
            assert not http_errors
            result = {
                "schema_version": "labelagent.reference-background-progress-acceptance.v1",
                "passed": True,
                "scope": scope,
                "refining_text": refining,
                "auditing_text": auditing,
                "completed_text": completed,
                "truth_write_requests": truth_writes,
                "page_errors": page_errors,
                "failed_requests": failed_requests,
                "http_errors": http_errors,
                "screenshot": str(screenshot),
            }
            receipt.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
            print(json.dumps(result, ensure_ascii=False))
        finally:
            browser.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
