#!/usr/bin/env python3
"""Run one candidate-only inference frame on an isolated real new dataset."""

from __future__ import annotations

import argparse
import json
import time
from collections import Counter
from pathlib import Path
from urllib.parse import urlparse

from playwright.sync_api import sync_playwright


CAPTURE_TOPICS = tuple(f"/capture/camera_{index}/image" for index in range(4))


def expected_status_poll_abort(method: str, url: str, failure: str) -> bool:
    """Separate the UI's deliberate stale-poll cancellation from outages."""
    return (
        method == "GET"
        and urlparse(url).path == "/api/run/status"
        and "ERR_ABORTED" in failure
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", required=True)
    parser.add_argument("--output-json", required=True, type=Path)
    parser.add_argument("--screenshot", required=True, type=Path)
    parser.add_argument(
        "--expected-topic", choices=CAPTURE_TOPICS, default=CAPTURE_TOPICS[0]
    )
    parser.add_argument("--frame", default=0, type=int)
    parser.add_argument("--coverage-probe", action="store_true")
    parser.add_argument("--timeout-seconds", default=180, type=int)
    args = parser.parse_args()
    if args.frame < 0:
        raise ValueError("frame must be non-negative")

    base_url = args.base_url.rstrip("/")
    parsed = urlparse(base_url)
    if parsed.scheme != "http" or parsed.hostname not in {"127.0.0.1", "localhost"}:
        raise ValueError("acceptance is restricted to a loopback isolated workbench")
    output_json = args.output_json.resolve()
    screenshot = args.screenshot.resolve()
    output_json.parent.mkdir(parents=True, exist_ok=True)
    screenshot.parent.mkdir(parents=True, exist_ok=True)
    timeout_ms = max(30, min(900, args.timeout_seconds)) * 1000
    page_errors: list[str] = []
    failed_requests: list[str] = []
    expected_aborted_requests: list[str] = []
    http_errors: list[str] = []
    truth_writes: list[str] = []
    started = time.monotonic()

    with sync_playwright() as runtime:
        browser = runtime.chromium.launch(channel="msedge", headless=True)
        try:
            page = browser.new_page(viewport={"width": 1760, "height": 1120})
            page.on("pageerror", lambda error: page_errors.append(str(error)))

            def on_console(message):
                if message.type == "error":
                    page_errors.append(message.text)

            def on_failed(request):
                failure = request.failure or ""
                record = f"{request.method} {request.url} {failure}"
                if expected_status_poll_abort(request.method, request.url, failure):
                    expected_aborted_requests.append(record)
                else:
                    failed_requests.append(record)

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
            page.on("requestfailed", on_failed)
            page.on("response", on_response)
            page.on("request", on_request)

            response = page.goto(base_url, wait_until="domcontentloaded", timeout=60_000)
            assert response is not None and response.status == 200
            page.evaluate("() => window.labelAgentBootstrapReady")
            page.wait_for_function(
                "() => { const item=document.querySelector('#datasetIdentity');"
                " return item && !item.classList.contains('unloaded')"
                " && !item.classList.contains('restoring'); }",
                timeout=60_000,
            )
            identity = page.evaluate("""() => ({
                kind: document.querySelector('#datasetIdentityKind')?.textContent,
                name: document.querySelector('#datasetIdentityName')?.textContent,
                detail: document.querySelector('#datasetIdentityDetail')?.textContent,
            })""")
            # Capture_20261103 has a dedicated identity badge.  Keeping the
            # older generic values here would let this acceptance silently
            # pass after the UI regressed back to an ambiguous dataset name.
            assert identity["kind"] == "新实采数据"
            assert "Capture_20261103" in (identity["name"] or "")
            assert args.expected_topic in identity["detail"]
            assert "4 路图像" in identity["detail"]

            page.locator("#frameNumber").fill(str(args.frame))
            page.locator("#frameNumber").dispatch_event("change")
            page.wait_for_function(
                """expected => {
                    const input=document.getElementById('frameNumber');
                    const image=document.getElementById('frameImage');
                    if(Number(input?.value)!==expected || !image?.complete || !image.src)
                        return false;
                    return new URL(image.src).pathname===`/api/frame/${expected}`;
                }""",
                arg=args.frame,
                timeout=60_000,
            )
            page.wait_for_timeout(300)

            initial = page.evaluate("() => window.currentWorkbenchFrameState()")
            assert initial["topic"] == args.expected_topic
            assert len(initial["labels"]) == 8
            assert len(initial["auto_labels"]) == 8
            frame_key = str(args.frame)
            assert frame_key not in initial.get("truth", {})
            assert frame_key not in initial.get("candidates", {})

            page.evaluate("() => window.openReferencePreparation()")
            dialog = page.locator("#referencePrepDialog[open]")
            dialog.wait_for(state="visible")
            if args.coverage_probe:
                page.locator("#referenceAutoLabel").select_option(
                    "__auto_visible_with_coverage__"
                )
            page.wait_for_function(
                "() => { const button=document.getElementById('runReferenceAutoDraft');"
                " return button && !button.disabled; }",
                timeout=60_000,
            )
            draft_started = time.monotonic()
            with page.expect_response(
                lambda item: urlparse(item.url).path == "/api/reference_draft",
                timeout=timeout_ms,
            ) as response_info:
                page.locator("#runReferenceAutoDraft").click()
            draft_response = response_info.value
            draft_body = draft_response.text()
            if draft_response.status != 200:
                raise RuntimeError(
                    "reference draft request failed: "
                    f"HTTP {draft_response.status}; body={draft_body[:1200]!r}"
                )
            try:
                draft_payload = json.loads(draft_body)
            except json.JSONDecodeError as exc:
                raise RuntimeError(
                    "reference draft returned a non-JSON success response: "
                    f"content_type={draft_response.headers.get('content-type')!r}; "
                    f"body={draft_body[:1200]!r}"
                ) from exc
            assert draft_payload.get("candidate_only") is True
            assert draft_payload.get("human_truth") is False
            assert draft_payload.get("approved_for_training") is False
            first_response_ms = round((time.monotonic() - draft_started) * 1000, 3)

            page.wait_for_function(
                "expected => Array.isArray(window.currentWorkbenchFrameState().candidates?.[String(expected)])"
                " && window.currentWorkbenchFrameState().candidates[String(expected)].length > 0",
                arg=args.frame,
                timeout=timeout_ms,
            )
            candidate_visible_ms = round((time.monotonic() - draft_started) * 1000, 3)
            page.wait_for_function(
                "expected => { const run=window.currentWorkbenchFrameState().agent_runs?.[String(expected)];"
                " return run?.agent_review_status==='complete'"
                " && run?.candidate_class_audit_status==='complete'; }",
                arg=args.frame,
                timeout=timeout_ms,
            )
            background_complete_ms = round((time.monotonic() - draft_started) * 1000, 3)

            final_state = page.evaluate("() => window.currentWorkbenchFrameState()")
            candidates = final_state["candidates"][frame_key]
            run = final_state["agent_runs"][frame_key]
            assert frame_key not in final_state.get("truth", {})
            assert args.frame not in final_state.get("reference_frames", {}).get(
                final_state["topic"], []
            )
            assert all(item.get("label") in final_state["auto_labels"] for item in candidates)

            page.wait_for_function(
                "() => document.querySelectorAll('#candidateSummary button[data-coverage-label]').length === 8"
            )
            coverage = page.evaluate("""() => ({
                summary: document.querySelector('#candidateSummary')?.innerText,
                chips: [...document.querySelectorAll(
                    '#candidateSummary button[data-coverage-label]')].map(item => ({
                        label: item.dataset.coverageLabel,
                        text: item.textContent,
                    })),
            })""")
            assert len(coverage["chips"]) == 8
            assert "已选 8/8 类" in coverage["summary"]
            assert any("已生成" in item["text"] for item in coverage["chips"])

            canvas = page.evaluate("""() => {
                const item=document.querySelector('#canvas');
                return {hidden:item?.hidden, width:item?.width, height:item?.height,
                        rect:item?.getBoundingClientRect().toJSON()};
            }""")
            assert canvas["hidden"] is False
            assert canvas["width"] > 0 and canvas["height"] > 0
            assert canvas["rect"]["width"] > 0 and canvas["rect"]["height"] > 0

            inspection_trace = run.get("inspection_trace") or []
            assert inspection_trace
            page.screenshot(path=str(screenshot), full_page=True)

            receipt = {
                "schema_version": "labelagent.new-dataset-inference-browser-acceptance.v1",
                "passed": not any((page_errors, failed_requests, http_errors, truth_writes)),
                "base_url": base_url,
                "identity": identity,
                "topic": final_state["topic"],
                "frame": args.frame,
                "reference_draft_mode": (
                    "auto_visible_with_coverage"
                    if args.coverage_probe
                    else "auto_visible_fast"
                ),
                "selected_labels": final_state["auto_labels"],
                "first_response_ms": first_response_ms,
                "candidate_visible_ms": candidate_visible_ms,
                "background_complete_ms": background_complete_ms,
                "candidate_count": len(candidates),
                "candidate_counts_by_label": dict(sorted(Counter(
                    item.get("label") for item in candidates
                ).items())),
                "coverage": coverage,
                "canvas": canvas,
                "agent_review_status": run.get("agent_review_status"),
                "candidate_class_audit_status": run.get("candidate_class_audit_status"),
                "inspection_trace_count": len(inspection_trace),
                "inspection_trace_services": sorted({
                    str(item.get("service")) for item in inspection_trace if item.get("service")
                }),
                "candidate_only": draft_payload.get("candidate_only"),
                "human_truth": draft_payload.get("human_truth"),
                "approved_for_training": draft_payload.get("approved_for_training"),
                "truth_frame_created": frame_key in final_state.get("truth", {}),
                "reference_frame_created": args.frame in final_state.get(
                    "reference_frames", {}
                ).get(final_state["topic"], []),
                "truth_write_requests": truth_writes,
                "page_errors": page_errors,
                "failed_requests": failed_requests,
                "expected_aborted_requests": expected_aborted_requests,
                "http_errors": http_errors,
                "screenshot": str(screenshot),
                "total_ms": round((time.monotonic() - started) * 1000, 3),
            }
            output_json.write_text(
                json.dumps(receipt, ensure_ascii=False, indent=2) + "\n",
                encoding="utf-8",
            )
            print(json.dumps(receipt, ensure_ascii=False, separators=(",", ":")))
            assert page_errors == []
            assert failed_requests == []
            assert http_errors == []
            assert truth_writes == []
        finally:
            browser.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
