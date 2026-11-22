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
import re
import time
from urllib.parse import urlparse

from playwright.sync_api import sync_playwright


ALLOWED_READONLY_POSTS = {"/api/topic"}


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
        "run_status": math.floor(duration_seconds / 15) + 1,
        "state_summary": math.floor(duration_seconds / 30) + 1,
    }


def sanitize_initiators(items: list[dict[str, object]], base_url: str) -> list[dict[str, object]]:
    """Keep only loopback paths and JavaScript function frames."""
    sanitized: list[dict[str, object]] = []
    base_origin = urlparse(base_url)
    for item in items:
        target = urlparse(str(item.get("target", "")))
        if target.scheme and (target.hostname, target.port) != (
            base_origin.hostname, base_origin.port
        ):
            continue
        path = target.path or str(item.get("target", "")).split("?", 1)[0]
        frames: list[str] = []
        for raw_line in str(item.get("stack", "")).splitlines()[1:]:
            line = raw_line.strip()
            if "<anonymous>" in line:
                frames.append("<anonymous>")
                continue
            match = re.search(r"at\s+([^\s(]+).*?(https?://[^\s)]+)", line)
            if not match:
                continue
            frame_url = urlparse(match.group(2))
            if (frame_url.hostname, frame_url.port) != (
                base_origin.hostname, base_origin.port
            ):
                continue
            frames.append(f"{match.group(1)}@{frame_url.path}")
        sanitized.append({
            "path": path,
            "at_ms": round(float(item.get("at", 0.0)), 3),
            "frames": frames,
        })
    return sanitized


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", default="http://127.0.0.1:8765")
    parser.add_argument("--duration-seconds", type=float, default=32)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument(
        "--capture-initiators",
        action="store_true",
        help="record local JavaScript fetch call stacks for status requests",
    )
    args = parser.parse_args()
    base_url = validate_loopback_url(args.base_url)
    if not 15 <= args.duration_seconds <= 120:
        raise ValueError("duration must be between 15 and 120 seconds")
    output = args.output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)

    counts: Counter[str] = Counter()
    steady_counts: Counter[str] = Counter()
    request_times: dict[str, list[float]] = {
        "/api/run/status": [],
        "/api/state/summary": [],
    }
    external_requests: list[str] = []
    guarded_writes: list[str] = []
    readonly_post_requests: list[str] = []
    page_errors: list[str] = []
    console_errors: list[str] = []
    http_errors: list[str] = []
    request_initiators: list[dict[str, object]] = []
    started = time.monotonic()
    steady_started: float | None = None
    base_origin = urlparse(base_url)

    with sync_playwright() as runtime:
        browser = runtime.chromium.launch(channel="msedge", headless=True)
        try:
            page = browser.new_page(viewport={"width": 1600, "height": 1000})
            if args.capture_initiators:
                page.add_init_script(
                    """
                    (() => {
                      const originalFetch = window.fetch.bind(window);
                      window.__labelAgentFetchInitiators = [];
                      window.fetch = (...args) => {
                        const target = String(args[0]?.url || args[0] || '');
                        if (target.includes('/api/run/status') || target.includes('/api/state/summary')) {
                          window.__labelAgentFetchInitiators.push({
                            target,
                            at: performance.now(),
                            stack: String(new Error('fetch initiator').stack || ''),
                          });
                        }
                        return originalFetch(...args);
                      };
                    })();
                    """
                )

            def on_request(request):
                parsed = urlparse(request.url)
                if parsed.scheme in {"data", "blob"}:
                    return
                if (parsed.hostname, parsed.port) != (base_origin.hostname, base_origin.port):
                    external_requests.append(request.url)
                    return
                counts[parsed.path] += 1
                if steady_started is not None:
                    steady_counts[parsed.path] += 1
                if parsed.path in request_times:
                    request_times[parsed.path].append(time.monotonic() - started)
                if request.method != "GET" and parsed.path.startswith("/api/"):
                    entry = f"{request.method} {parsed.path}"
                    if request.method == "POST" and parsed.path in ALLOWED_READONLY_POSTS:
                        readonly_post_requests.append(entry)
                    else:
                        guarded_writes.append(entry)

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
            workspace_ready_seconds = time.monotonic() - started
            startup_counts = counts.copy()
            steady_started = time.monotonic()
            page.wait_for_timeout(round(args.duration_seconds * 1000))
            ui_version = page.locator('meta[name="labelagent-ui-version"]').get_attribute("content")
            if args.capture_initiators:
                request_initiators = sanitize_initiators(
                    page.evaluate("() => window.__labelAgentFetchInitiators || []"),
                    base_url,
                )
        finally:
            browser.close()

    budget = request_budget(args.duration_seconds)
    result = {
        "schema_version": "labelagent.idle-polling-browser-acceptance.v2",
        "passed": False,
        "base_url": base_url,
        "duration_seconds": args.duration_seconds,
        "workspace_ready_seconds": round(workspace_ready_seconds, 3),
        "ui_version": ui_version,
        "run_status_requests": counts["/api/run/status"],
        "state_summary_requests": counts["/api/state/summary"],
        "request_budget": budget,
        "startup_request_counts": {
            "run_status": startup_counts["/api/run/status"],
            "state_summary": startup_counts["/api/state/summary"],
        },
        "steady_state_request_counts": {
            "run_status": steady_counts["/api/run/status"],
            "state_summary": steady_counts["/api/state/summary"],
        },
        "request_times_seconds": {
            name: [round(value, 3) for value in values]
            for name, values in request_times.items()
        },
        "external_requests": external_requests,
        "guarded_writes": guarded_writes,
        "allowed_readonly_posts": readonly_post_requests,
        "page_errors": page_errors,
        "console_errors": console_errors,
        "http_errors": http_errors,
        "request_initiators": request_initiators,
    }
    result["passed"] = bool(
        startup_counts["/api/run/status"] <= 2
        and startup_counts["/api/state/summary"] <= 2
        and steady_counts["/api/run/status"] <= budget["run_status"]
        and steady_counts["/api/state/summary"] <= budget["state_summary"]
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
