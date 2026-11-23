"""Read-only real-browser acceptance for hash-bound candidate comparisons."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from urllib.parse import urlsplit

from playwright.sync_api import sync_playwright


def installed_browser() -> str:
    candidates = (
        Path(r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe"),
        Path(r"C:\Program Files\Microsoft\Edge\Application\msedge.exe"),
        Path(r"C:\Program Files\Google\Chrome\Application\chrome.exe"),
    )
    match = next((path for path in candidates if path.is_file()), None)
    if match is None:
        raise RuntimeError("Microsoft Edge or Google Chrome is required")
    return str(match)


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def validate_url(value: str) -> str:
    parsed = urlsplit(value)
    if (parsed.scheme != "http" or parsed.hostname != "127.0.0.1"
            or parsed.port is None or parsed.username or parsed.password
            or parsed.path not in {"", "/"}
            or parsed.query or parsed.fragment):
        raise ValueError("acceptance URL must be an isolated 127.0.0.1 HTTP server")
    return value.rstrip("/") + "/"


def open_dialog(page) -> None:
    page.evaluate("window.setWorkspaceMode?.('auto', false)")
    page.wait_for_selector("#candidateComparisonOpen", state="attached")
    page.evaluate("document.querySelector('#candidateComparisonOpen').click()")
    page.wait_for_selector("#candidateComparisonDialog[open]", state="visible")
    page.wait_for_function(
        "document.querySelector('#candidateComparisonStatus')?.classList.contains('ready')")
    page.wait_for_function("""() => {
      const images = [...document.querySelectorAll('#candidateComparisonContent img')];
      return images.length === 2 && images.every(image => image.complete && image.naturalWidth > 0);
    }""")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", required=True)
    parser.add_argument("--output", required=True, type=Path)
    arguments = parser.parse_args()
    url = validate_url(arguments.url)
    output = arguments.output.resolve()
    if output.exists():
        raise FileExistsError(f"output already exists: {output}")
    output.mkdir(parents=True)
    report = {
        "schema_version": "labelagent.candidate-comparison-browser-acceptance.v1",
        "url": url,
        "checks": {},
        "page_errors": [],
        "console_errors": [],
        "http_errors": [],
        "failed_requests": [],
        "candidate_api_non_get": [],
        "protected_writes": [],
    }

    with sync_playwright() as runtime:
        browser = runtime.chromium.launch(
            headless=True, executable_path=installed_browser())
        context = browser.new_context(
            viewport={"width": 1600, "height": 1000}, device_scale_factor=1)
        page = context.new_page()
        page.set_default_timeout(30_000)
        page.on("pageerror", lambda error: report["page_errors"].append(str(error)))
        page.on("console", lambda message: report["console_errors"].append(message.text)
                if message.type == "error" else None)
        page.on("response", lambda response: report["http_errors"].append(
            {"status": response.status, "method": response.request.method,
             "url": response.url})
            if response.status >= 400 else None)
        page.on("requestfailed", lambda request: report["failed_requests"].append(
            {"method": request.method, "url": request.url,
             "failure": request.failure}))

        def record_request(request) -> None:
            if "/api/candidate_comparison_packs" in request.url and request.method != "GET":
                report["candidate_api_non_get"].append(
                    {"method": request.method, "url": request.url})
            if request.method != "GET" and any(
                    token in request.url for token in
                    ("/api/truth", "/api/human_review", "/api/reference_frames")):
                report["protected_writes"].append(
                    {"method": request.method, "url": request.url})

        page.on("request", record_request)
        response = page.goto(url, wait_until="domcontentloaded")
        page.wait_for_function("window.labelAgentBootstrapReady !== undefined")
        open_dialog(page)
        initial = page.evaluate("""() => ({
          badge: document.querySelector('#candidateComparisonBadge')?.textContent,
          packs: document.querySelector('#candidateComparisonPack')?.options.length,
          rows: document.querySelector('#candidateComparisonRow')?.options.length,
          cards: document.querySelectorAll('#candidateComparisonContent .candidateComparisonCard').length,
          status: document.querySelector('#candidateComparisonStatus')?.textContent,
          boundary: document.querySelector('.candidateComparisonBoundary')?.textContent,
        })""")
        preference = page.locator(
            "#candidateComparisonContent .candidateComparisonCard button").first
        preference.click()
        page.wait_for_function("""() => {
          const button = document.querySelector('#candidateComparisonContent .candidateComparisonCard button');
          return button?.classList.contains('selected');
        }""")
        stored = page.evaluate(
            "sessionStorage.getItem('labelagentCandidateComparisonPreferencesV1')")
        screenshot = output / "candidate-comparison.png"
        page.locator("#candidateComparisonDialog").screenshot(path=str(screenshot))

        page.reload(wait_until="domcontentloaded")
        page.wait_for_function("window.labelAgentBootstrapReady !== undefined")
        open_dialog(page)
        retained = page.locator(
            "#candidateComparisonContent .candidateComparisonCard button").first
        retained_selected = "selected" in (retained.get_attribute("class") or "").split()

        # Simulate a catalog that disappeared after refresh.  This is a browser-
        # local GET override and does not mutate the isolated workbench.
        page.route("**/api/candidate_comparison_packs", lambda route: route.fulfill(
            status=200, content_type="application/json", body=json.dumps({
                "schema_version": "labelagent.candidate-comparison-pack-index.v1",
                "packets": [], "rejected_packets": 0, "read_only": True,
                "session_preferences_only": True, "human_truth_writes": False,
                "training_data_writes": False, "automatic_approval": False,
            })))
        page.locator("#candidateComparisonRefresh").click()
        page.wait_for_function("""() =>
          document.querySelector('#candidateComparisonStatus')?.textContent.includes('没有发现')""")
        empty_refresh = page.evaluate("""() => ({
          packs: document.querySelector('#candidateComparisonPack')?.options.length,
          rows: document.querySelector('#candidateComparisonRow')?.options.length,
          cards: document.querySelectorAll('#candidateComparisonContent .candidateComparisonCard').length,
          content: document.querySelector('#candidateComparisonContent')?.textContent,
        })""")

        clean_context = browser.new_context(
            viewport={"width": 1400, "height": 900}, device_scale_factor=1)
        clean = clean_context.new_page()
        clean.set_default_timeout(30_000)
        clean.on("response", lambda response: report["http_errors"].append(
            {"status": response.status, "method": response.request.method,
             "url": response.url})
            if response.status >= 400 else None)
        clean.on("requestfailed", lambda request: report["failed_requests"].append(
            {"method": request.method, "url": request.url,
             "failure": request.failure}))
        clean.goto(url, wait_until="domcontentloaded")
        clean.wait_for_function("window.labelAgentBootstrapReady !== undefined")
        open_dialog(clean)
        clean_selected = clean.locator(
            "#candidateComparisonContent .candidateComparisonCard button").first
        new_session_selected = "selected" in (
            clean_selected.get_attribute("class") or "").split()

        report["checks"] = {
            "http_status": response.status if response else None,
            "initial": initial,
            "session_preference_written": bool(stored),
            "same_session_reload_retained": retained_selected,
            "new_session_isolated": not new_session_selected,
            "empty_refresh_failed_closed": empty_refresh,
            "screenshot": str(screenshot),
            "screenshot_sha256": sha256(screenshot),
        }
        expected_optional_404 = [item for item in report["http_errors"]
                                 if item["status"] == 404
                                 and item["method"] == "GET"
                                 and item["url"] == url.rstrip("/")
                                 + "/api/pallet-candidate-review"]
        unexpected_http = [item for item in report["http_errors"]
                           if item not in expected_optional_404]
        unexpected_console = [message for message in report["console_errors"]
                              if not (expected_optional_404 and message.startswith(
                                  "Failed to load resource:"))]
        report["checks"]["expected_optional_404_count"] = len(expected_optional_404)
        report["checks"]["unexpected_http_errors"] = unexpected_http
        report["checks"]["unexpected_console_errors"] = unexpected_console
        report["passed"] = bool(
            report["checks"]["http_status"] == 200
            and initial["badge"] == "1" and initial["packs"] == 1
            and initial["rows"] == 33 and initial["cards"] == 2
            and "不会修改 mask" in initial["boundary"]
            and stored and retained_selected and not new_session_selected
            and empty_refresh == {"packs": 0, "rows": 0, "cards": 0, "content": ""}
            and not report["candidate_api_non_get"]
            and not report["protected_writes"]
            and not report["page_errors"] and not unexpected_console
            and not unexpected_http and not report["failed_requests"])
        clean_context.close()
        context.close()
        browser.close()

    (output / "acceptance.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    if not report["passed"]:
        raise SystemExit("candidate comparison browser acceptance failed")


if __name__ == "__main__":
    main()
