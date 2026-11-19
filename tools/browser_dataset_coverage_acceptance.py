#!/usr/bin/env python3
"""Read-only browser acceptance for dataset identity and class coverage UI.

This Python entry point replaces the historical Node script's dependency on a
Codex-runtime absolute path.  It uses the project virtual environment's pinned
Playwright package and never starts inference or writes truth.
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
    parser.add_argument("--base-url", default="http://127.0.0.1:8765")
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args()

    base_url = args.base_url.rstrip("/")
    parsed = urlparse(base_url)
    if parsed.scheme != "http" or parsed.hostname not in {"127.0.0.1", "localhost"}:
        raise ValueError("acceptance is restricted to a loopback workbench")
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    manual_screenshot = output / "manual-inference-shortcut.png"
    coverage_screenshot = output / "dataset-class-coverage.png"
    receipt_path = output / "result.json"

    errors: list[str] = []
    failed_requests: list[str] = []
    http_errors: list[str] = []
    truth_writes: list[str] = []
    started = time.monotonic()

    with sync_playwright() as runtime:
        browser = runtime.chromium.launch(channel="msedge", headless=True)
        try:
            page = browser.new_page(
                viewport={"width": 1760, "height": 1120}, device_scale_factor=1
            )
            page.on("pageerror", lambda error: errors.append(str(error)))

            def on_console(message):
                if message.type == "error":
                    errors.append(message.text)

            def on_request_failed(request):
                failure = request.failure or ""
                failed_requests.append(
                    f"{request.method} {request.url} {failure}"
                )

            def on_response(response):
                if response.status >= 400:
                    http_errors.append(
                        f"{response.request.method} {response.status} {response.url}"
                    )

            def on_request(request):
                path = urlparse(request.url).path
                if request.method != "GET" and path in {"/api/truth", "/api/human_review"}:
                    truth_writes.append(f"{request.method} {request.url}")

            page.on("console", on_console)
            page.on("requestfailed", on_request_failed)
            page.on("response", on_response)
            page.on("request", on_request)

            response = page.goto(base_url, wait_until="domcontentloaded", timeout=60_000)
            assert response is not None and response.status == 200
            page.evaluate("() => window.labelAgentBootstrapReady")
            page.wait_for_function(
                "() => { const item=document.querySelector('#datasetIdentity');"
                " return item && !item.classList.contains('unloaded')"
                " && !item.classList.contains('restoring'); }"
            )
            identity = page.evaluate("""() => ({
                kind: document.querySelector('#datasetIdentityKind')?.textContent,
                name: document.querySelector('#datasetIdentityName')?.textContent,
                detail: document.querySelector('#datasetIdentityDetail')?.textContent,
                title: document.querySelector('#datasetIdentity')?.title,
            })""")
            assert identity["kind"] in {"新数据集", "旧验证集", "已加载数据"}
            assert (
                "/capture/camera_" in identity["detail"]
                or "/CVAT/Task3/images" in identity["detail"]
            )

            page.locator("#datasetIdentitySwitch").click()
            page.locator("#fileManager[open]").wait_for(state="visible")
            current_row = page.locator(".fileRow", has_text="当前已加载")
            current_row.wait_for(state="visible")
            file_row = current_row.inner_text()
            assert "当前已加载" in file_row
            page.locator("#closeFilesDialog").click()

            page.locator("#manualModeButton").click()
            page.wait_for_function("() => document.body.dataset.workspace === 'manual'")
            shortcut = page.locator("#goToAutoInference")
            shortcut.wait_for(state="visible")
            assert "设置 N 帧" in shortcut.inner_text()
            page.screenshot(path=str(manual_screenshot), full_page=True)
            shortcut.click()
            page.wait_for_function("() => document.body.dataset.workspace === 'auto'")
            page.wait_for_function("() => document.activeElement?.id === 'taskCount'")
            assert page.locator("#taskCount").input_value() == "1"
            assert shortcut.is_hidden()
            page.wait_for_function(
                "() => document.querySelectorAll('#autoLabelOptions input:checked').length === 8"
            )
            page.wait_for_function(
                "() => document.querySelectorAll('#candidateSummary button[data-coverage-label]').length === 8"
            )
            coverage = page.evaluate("""() => ({
                checked: [...document.querySelectorAll('#autoLabelOptions input:checked')]
                    .map(item => item.value),
                summary: document.querySelector('#candidateSummary')?.innerText,
                hint: document.querySelector('#autoLabelHint')?.textContent,
                chips: [...document.querySelectorAll(
                    '#candidateSummary button[data-coverage-label]')].map(item => ({
                        label: item.dataset.coverageLabel,
                        text: item.textContent,
                        title: item.title,
                        tag: item.tagName,
                    })),
            })""")
            assert len(coverage["checked"]) == 8
            assert len(coverage["chips"]) == 8
            assert all(item["tag"] == "BUTTON" for item in coverage["chips"])
            assert all("点击查看判定依据" in item["title"] for item in coverage["chips"])
            assert "已选 8/8 类" in coverage["summary"]
            assert "不等于每帧必有该类" in coverage["hint"]

            first_label = coverage["chips"][0]["label"]
            page.locator(
                f"#candidateSummary button[data-coverage-label='{first_label}']"
            ).click()
            page.wait_for_function(
                "label => { const detail=document.querySelector('#candidateCoverageDetail');"
                " return detail && !detail.hidden && detail.dataset.coverageLabel===label; }",
                arg=first_label,
            )
            coverage_detail = page.locator("#candidateCoverageDetail").inner_text()
            assert coverage_detail.strip()

            page.evaluate("() => window.openReferencePreparation()")
            page.locator("#referencePrepDialog[open]").wait_for(state="visible")
            option = page.locator(
                "#referenceAutoLabel option", has_text="逐类尝试全部已选推理类别"
            )
            assert option.count() == 1
            page.locator("#referencePrepClose").click()
            page.screenshot(path=str(coverage_screenshot), full_page=True)

            assert truth_writes == []
            assert errors == []
            assert failed_requests == []
            assert http_errors == []
            receipt = {
                "schema_version": "labelagent.dataset-coverage-browser-acceptance.v2",
                "passed": True,
                "elapsed_ms": round((time.monotonic() - started) * 1000, 3),
                "base_url": base_url,
                "identity": identity,
                "file_row": file_row,
                "coverage": coverage,
                "coverage_detail_checked": {
                    "label": first_label,
                    "text": coverage_detail,
                    "interactive": True,
                },
                "inference_shortcut": {
                    "switched_to_auto": True,
                    "task_count_focused": True,
                    "default_count": 1,
                    "manual_screenshot": str(manual_screenshot),
                },
                "forced_all_selected_option": True,
                "truth_write_requests": truth_writes,
                "page_errors": errors,
                "failed_requests": failed_requests,
                "http_errors": http_errors,
                "coverage_screenshot": str(coverage_screenshot),
                "candidate_only": True,
                "human_truth": False,
                "approved_for_training": False,
            }
            receipt_path.write_text(
                json.dumps(receipt, ensure_ascii=False, indent=2) + "\n",
                encoding="utf-8",
            )
            print(json.dumps(receipt, ensure_ascii=False, separators=(",", ":")))
        finally:
            browser.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
