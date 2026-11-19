#!/usr/bin/env python3
"""Run a candidate-only browser acceptance for the guarded ego fallback.

The target must be an isolated LabelAgent workbench.  The script uploads one
image, runs the normal reference-draft UI, verifies the candidate/truth
boundary and the visible ego safety warning, then writes a screenshot and a
small JSON receipt.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from collections import Counter
from pathlib import Path
from urllib.parse import urlparse

from playwright.sync_api import sync_playwright


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-url", required=True)
    parser.add_argument("--image", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--screenshot", type=Path, required=True)
    args = parser.parse_args()

    image = args.image.resolve(strict=True)
    output_json = args.output_json.resolve()
    screenshot = args.screenshot.resolve()
    output_json.parent.mkdir(parents=True, exist_ok=True)
    screenshot.parent.mkdir(parents=True, exist_ok=True)
    base_url = args.base_url.rstrip("/")
    page_errors: list[str] = []
    started = time.monotonic()

    with sync_playwright() as runtime:
        browser = runtime.chromium.launch(channel="msedge", headless=True)
        try:
            page = browser.new_page(viewport={"width": 1600, "height": 1050})
            page.on("pageerror", lambda error: page_errors.append(str(error)))
            response = page.goto(base_url, wait_until="networkidle")
            assert response and response.status == 200
            page.wait_for_function("() => window.labelAgentBootstrapReady")

            page.locator("#openFilesDialog").click()
            page.locator("#fileManager").wait_for(state="visible")
            upload_started = time.monotonic()
            page.locator("#mcapFile").set_input_files(str(image))
            page.wait_for_function(
                "() => document.getElementById('fileManagerStatus')?.textContent.includes('已添加')")
            upload_ms = round((time.monotonic() - upload_started) * 1000, 3)

            row = page.locator(".fileRow", has_text=image.name)
            row.wait_for(state="visible")
            load_started = time.monotonic()
            row.get_by_role("button", name="加载", exact=True).click()
            page.wait_for_function(
                "() => document.getElementById('fileManagerStatus')?.textContent.includes('兼容检查通过')")
            load_ms = round((time.monotonic() - load_started) * 1000, 3)
            load_status = page.locator("#fileManagerStatus").inner_text()
            page.locator("#closeFilesDialog").click()

            page.evaluate("() => window.openReferencePreparation()")
            page.locator("#referencePrepDialog").wait_for(state="visible")
            page.wait_for_function(
                "() => { const b=document.getElementById('runReferenceAutoDraft'); return b && !b.disabled; }")
            initial = page.evaluate("() => window.currentWorkbenchFrameState()")
            assert initial.get("truth") == {}
            assert initial.get("candidates") == {}

            draft_started = time.monotonic()
            with page.expect_response(
                    lambda item: urlparse(item.url).path == "/api/reference_draft",
                    timeout=90_000) as response_info:
                page.locator("#runReferenceAutoDraft").click()
            draft_response = response_info.value
            draft_payload = draft_response.json()
            assert draft_response.status == 200
            assert draft_payload.get("candidate_only") is True
            assert draft_payload.get("human_truth") is False
            assert draft_payload.get("approved_for_training") is False

            page.wait_for_function(
                "() => Array.isArray(window.currentWorkbenchFrameState().candidates?.['0'])"
                " && window.currentWorkbenchFrameState().candidates['0'].length > 0",
                timeout=90_000)
            candidate_visible_ms = round((time.monotonic() - draft_started) * 1000, 3)
            page.wait_for_function(
                "() => { const r=window.currentWorkbenchFrameState().agent_runs?.['0'];"
                " return r?.agent_review_status==='complete'"
                " && r?.candidate_class_audit_status==='complete'; }",
                timeout=90_000)

            final_state = page.evaluate("() => window.currentWorkbenchFrameState()")
            candidates = final_state["candidates"]["0"]
            run = final_state["agent_runs"]["0"]
            fallback = run.get("ego_bottom_point_fallback") or {}
            assert fallback.get("status") == "candidate_generated"
            assert any(item.get("label") == "ego" for item in candidates)
            assert "0" not in final_state.get("truth", {})
            assert 0 not in final_state.get("reference_frames", {}).get(final_state["topic"], [])

            alert = page.locator(".egoFallbackCandidateAlert")
            alert.wait_for(state="visible")
            alert_text = alert.inner_text()
            assert "文本分割漏检" in alert_text
            assert "人工真值" in alert_text
            guide_settle_started = time.monotonic()
            page.wait_for_function(
                "() => { const t=document.getElementById('guideTitle')?.textContent || '';"
                " return !t.includes('正在处理') && !t.includes('请等待结果'); }",
                timeout=10_000)
            guide_settle_ms = round((time.monotonic() - guide_settle_started) * 1000, 3)
            guide_title = page.locator("#guideTitle").inner_text()
            assert page_errors == []
            page.screenshot(path=str(screenshot), full_page=True)

            receipt = {
                "schema_version": "labelagent.browser-ego-fallback-acceptance.v1",
                "base_url": base_url,
                "input": str(image),
                "input_bytes": image.stat().st_size,
                "input_sha256": sha256(image),
                "page_status": response.status,
                "upload_ms": upload_ms,
                "load_ms": load_ms,
                "load_status": load_status,
                "candidate_visible_ms": candidate_visible_ms,
                "background_complete_ms": round((time.monotonic() - draft_started) * 1000, 3),
                "candidate_count": len(candidates),
                "candidate_counts_by_label": dict(sorted(Counter(
                    item.get("label") for item in candidates).items())),
                "ego_bottom_point_fallback": fallback,
                "ego_alert_text": alert_text,
                "guide_title_after_background": guide_title,
                "guide_settle_ms": guide_settle_ms,
                "candidate_only": draft_payload.get("candidate_only"),
                "human_truth": draft_payload.get("human_truth"),
                "approved_for_training": draft_payload.get("approved_for_training"),
                "truth_frame_created": "0" in final_state.get("truth", {}),
                "reference_frame_created": 0 in final_state.get(
                    "reference_frames", {}).get(final_state["topic"], []),
                "page_errors": page_errors,
                "screenshot": str(screenshot),
                "total_ms": round((time.monotonic() - started) * 1000, 3),
            }
            output_json.write_text(
                json.dumps(receipt, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
            print(json.dumps(receipt, ensure_ascii=False))
        finally:
            browser.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
