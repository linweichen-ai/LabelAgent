"""Exercise the read-only focused component review workflow in a real browser."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from playwright.sync_api import sync_playwright


def installed_browser() -> str | None:
    candidates = (
        Path(r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe"),
        Path(r"C:\Program Files\Microsoft\Edge\Application\msedge.exe"),
        Path(r"C:\Program Files\Google\Chrome\Application\chrome.exe"),
    )
    return str(next((path for path in candidates if path.is_file()), "")) or None


def open_dialog(page) -> None:
    page.wait_for_selector("#componentReviewOpen", state="attached")
    button = page.locator("#componentReviewOpen")
    if button.is_visible():
        button.click()
    else:
        # An isolated server without a loaded dataset keeps the auto workbench
        # collapsed.  Dispatch the same entry action without bypassing its JS.
        page.evaluate("document.querySelector('#componentReviewOpen').click()")
    page.wait_for_selector("#componentReviewDialog[open]", state="visible")
    page.wait_for_function(
        "document.querySelector('#componentReviewStatus')?.classList.contains('ready')")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", required=True)
    parser.add_argument("--output", required=True, type=Path)
    arguments = parser.parse_args()
    output = arguments.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    report = {"checks": {}, "page_errors": [], "console_errors": [],
              "component_api_posts": []}

    with sync_playwright() as runtime:
        browser = runtime.chromium.launch(
            headless=True, executable_path=installed_browser())
        first_context = browser.new_context(
            viewport={"width": 1600, "height": 1050}, device_scale_factor=1)
        page = first_context.new_page()
        page.set_default_timeout(30_000)
        page.on("pageerror", lambda error: report["page_errors"].append(str(error)))
        page.on("console", lambda message: report["console_errors"].append(message.text)
                if message.type == "error" else None)
        page.on("request", lambda request: report["component_api_posts"].append(
            {"method": request.method, "url": request.url})
            if "/api/component_review_packs" in request.url
            and request.method != "GET" else None)
        response = page.goto(arguments.url, wait_until="domcontentloaded")
        page.wait_for_function("window.labelAgentBootstrapReady !== undefined")
        page.evaluate("window.setWorkspaceMode?.('auto', false)")
        page.wait_for_function(
            "document.querySelector('#componentReviewBadge')?.textContent === '19'")
        open_dialog(page)
        page.wait_for_function(
            "document.querySelectorAll('#componentReviewGrid .componentReviewCard').length === 4")
        page.wait_for_function("""() => {
          const images = [...document.querySelectorAll('#componentReviewGrid img')];
          return images.length === 4 && images.every(image => image.complete && image.naturalWidth > 0);
        }""")
        initial = page.evaluate("""() => ({
          badge: document.querySelector('#componentReviewBadge')?.textContent,
          progress: document.querySelector('#componentReviewProgressTitle')?.textContent,
          cards: document.querySelectorAll('#componentReviewGrid .componentReviewCard').length,
          boundary: document.querySelector('.componentReviewBoundary')?.textContent,
          status: document.querySelector('#componentReviewStatus')?.textContent,
        })""")
        page.locator('[data-component-filter="priority"]').click()
        page.wait_for_function(
            "document.querySelectorAll('#componentReviewGrid .componentReviewCard').length === 1")
        priority_id = page.locator(
            "#componentReviewGrid .componentReviewCardHead strong").inner_text()
        page.locator("#componentReviewGrid .componentReviewActions button",
                     has_text="不确定").click()
        page.wait_for_function(
            "document.querySelector('#componentReviewProgressTitle')?.textContent.startsWith('1 / 19')")
        stored = page.evaluate(
            "sessionStorage.getItem('labelagentComponentReviewDraftsV1')")
        page.reload(wait_until="domcontentloaded")
        page.evaluate("window.setWorkspaceMode?.('auto', false)")
        open_dialog(page)
        page.wait_for_function(
            "document.querySelector('#componentReviewProgressTitle')?.textContent.startsWith('1 / 19')")
        page.wait_for_function("""() => {
          const images = [...document.querySelectorAll('#componentReviewGrid img')];
          return images.length === 4 && images.every(image => image.complete && image.naturalWidth > 0);
        }""")
        retained = page.locator("#componentReviewProgressTitle").inner_text()
        screenshot = output / "component-review-session-draft.png"
        page.locator("#componentReviewDialog").screenshot(path=str(screenshot))

        second_context = browser.new_context(
            viewport={"width": 1400, "height": 900}, device_scale_factor=1)
        clean = second_context.new_page()
        clean.set_default_timeout(30_000)
        clean.goto(arguments.url, wait_until="domcontentloaded")
        clean.evaluate("window.setWorkspaceMode?.('auto', false)")
        open_dialog(clean)
        clean_progress = clean.locator("#componentReviewProgressTitle").inner_text()

        report["checks"] = {
            "http_status": response.status if response else None,
            "initial": initial,
            "priority_component_id": priority_id,
            "session_storage_written": bool(stored),
            "same_session_reload_progress": retained,
            "new_session_progress": clean_progress,
            "screenshot": str(screenshot),
        }
        report["passed"] = bool(
            report["checks"]["http_status"] == 200
            and initial["badge"] == "19"
            and initial["progress"].startswith("0 / 19")
            and initial["cards"] == 4
            and "不会修改 mask" in initial["boundary"]
            and priority_id == "004-camera1-idx5-1793760663542839987-c002"
            and stored
            and retained.startswith("1 / 19")
            and clean_progress.startswith("0 / 19")
            and not report["component_api_posts"]
            and not report["page_errors"]
            and not report["console_errors"])
        second_context.close()
        first_context.close()
        browser.close()

    (output / "acceptance.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    if not report["passed"]:
        raise SystemExit("component review browser acceptance failed")


if __name__ == "__main__":
    main()
