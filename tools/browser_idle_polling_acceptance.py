#!/usr/bin/env python3
"""Read-only real-browser acceptance for idle workbench polling.

The check only opens a loopback LabelAgent page, waits while it is idle, and
counts status requests.  It never starts inference or writes review/truth data.
"""

from __future__ import annotations

import argparse
from collections import Counter
import json
import math
from pathlib import Path
import time
from urllib.parse import urlparse

from playwright.sync_api import sync_playwright


WRITE_GUARD_PATHS = {
    "/api/run",
    "/api/truth",
    "/api/human_review",
    "/api/candidates",
}


def validate_loopback_url(value: str) -> str:
    parsed = urlparse(value.rstrip("/"))
    if parsed.scheme != "http" or parsed.hostname not in {"127.0.0.1", "localhost"}:
        raise ValueError("acceptance is restricted to a loopback workbench")
    if parsed.port is None:
        raise ValueError("an explicit loopback port is required")
    return value.rstrip("/")


def request_budget(duration_seconds: float) -> dict[str, int]:
    if duration_seconds < 15:
        raise ValueError("duration must be at least 15 seconds")
    return {
        "run_status": math.floor(duration_seconds / 15) + 2,
        "state_summary": math.floor(duration_seconds / 30) + 3,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", default="http://127.0.0.1:8765")
    parser.add_argument("--duration-seconds", type=float, default=32)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    base_url = validate_loopback_url(args.base_url)
    if not 15 <= args.duration_seconds <= 120:
        raise ValueError("duration must be between 15 and 120 seconds")
    output = args.output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)

    counts: Counter[str] = Counter()
    request_times: dict[str, list[float]] = {
        "/api/run/status": [],
        "/api/state/summary": [],
    }
    external_requests: list[str] = []
    guarded_writes: list[str] = []
    page_errors: list[str] = []
    console_errors: list[str] = []
    http_errors: list[str] = []
    started = time.monotonic()
    base_origin = urlparse(base_url)

    with sync_playwright() as runtime:
        browser = runtime.chromium.launch(channel="msedge", headless=True)
        try:
            page = browser.new_page(viewport={"width": 1600, "height": 1000})

            def on_request(request):
                parsed = urlparse(request.url)
                if parsed.scheme in {"data", "blob"}:
                    return
                if (parsed.hostname, parsed.port) != (base_origin.hostname, base_origin.port):
                    external_requests.append(request.url)
                    return
                counts[parsed.path] += 1
                if parsed.path in request_times:
                    request_times[parsed.path].append(time.monotonic() - started)
                if request.method != "GET" and parsed.path in WRITE_GUARD_PATHS:
                    guarded_writes.append(f"{request.method} {parsed.path}")

            page.on("request", on_request)
            page.on("pageerror", lambda error: page_errors.append(str(error)))
            page.on(
                "console",
                lambda message: console_errors.append(message.text)
                if message.type == "error" else None,
            )
            page.on(
                "response",
                lambda response: http_errors.append(
                    f"{response.status} {urlparse(response.url).path}"
                ) if response.status >= 400 else None,
            )
            response = page.goto(base_url, wait_until="domcontentloaded", timeout=60_000)
            if response is None or response.status != 200:
                raise RuntimeError("workbench root did not return HTTP 200")
            page.evaluate("() => window.labelAgentWorkspaceReady")
            page.wait_for_timeout(round(args.duration_seconds * 1000))
            ui_version = page.locator('meta[name="labelagent-ui-version"]').get_attribute("content")
        finally:
            browser.close()

    budget = request_budget(args.duration_seconds)
    result = {
        "schema_version": "labelagent.idle-polling-browser-acceptance.v1",
        "passed": False,
        "base_url": base_url,
        "duration_seconds": args.duration_seconds,
        "ui_version": ui_version,
        "run_status_requests": counts["/api/run/status"],
        "state_summary_requests": counts["/api/state/summary"],
        "request_budget": budget,
        "request_times_seconds": {
            name: [round(value, 3) for value in values]
            for name, values in request_times.items()
        },
        "external_requests": external_requests,
        "guarded_writes": guarded_writes,
        "page_errors": page_errors,
        "console_errors": console_errors,
        "http_errors": http_errors,
    }
    result["passed"] = bool(
        counts["/api/run/status"] <= budget["run_status"]
        and counts["/api/state/summary"] <= budget["state_summary"]
        and not external_requests
        and not guarded_writes
        and not page_errors
        and not console_errors
        and not http_errors
    )
    output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False))
    return 0 if result["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
