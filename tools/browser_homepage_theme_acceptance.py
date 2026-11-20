"""Capture a read-only visual acceptance packet for the LabelAgent theme."""

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
        Path(r"C:\Program Files (x86)\Google\Chrome\Application\chrome.exe"),
    )
    return str(next((path for path in candidates if path.is_file()), "")) or None


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", required=True)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--width", type=int, default=1600)
    parser.add_argument("--height", type=int, default=1000)
    args = parser.parse_args()

    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    console_errors: list[str] = []
    page_errors: list[str] = []

    with sync_playwright() as runtime:
        executable = installed_browser()
        browser = runtime.chromium.launch(headless=True, executable_path=executable)
        page = browser.new_page(
            viewport={"width": args.width, "height": args.height}, device_scale_factor=1
        )
        page.on("console", lambda message: console_errors.append(message.text) if message.type == "error" else None)
        page.on("pageerror", lambda error: page_errors.append(str(error)))
        response = page.goto(args.url, wait_until="domcontentloaded", timeout=30_000)
        page.wait_for_selector(".windowToolbar", state="visible", timeout=15_000)
        page.wait_for_timeout(2_000)

        manual_path = output / "manual-workbench.png"
        page.screenshot(path=str(manual_path), full_page=False)

        metrics = page.evaluate(
            """() => {
              const style = getComputedStyle(document.documentElement);
              const toolbar = document.querySelector('.windowToolbar')?.getBoundingClientRect();
              const guide = document.querySelector('#workflowGuide')?.getBoundingClientRect();
              const grid = document.querySelector('#windowGrid')?.getBoundingClientRect();
              const workbench = document.querySelector('#workbench')?.getBoundingClientRect();
              const stage = document.querySelector('#stage')?.getBoundingClientRect();
              const image = document.querySelector('#frameImage')?.getBoundingClientRect();
              const viewerStyle = getComputedStyle(document.querySelector('.viewer'));
              const batchPanel = document.querySelector('#batchReviewPanel');
              const candidatePanel = document.querySelector('#candidateReviewPanel');
              const inspectionPanel = document.querySelector('#inspectionPanel');
              const theme = [...document.styleSheets].some(sheet => {
                try { return new URL(sheet.href).pathname.endsWith('/homepage-theme.css'); }
                catch (_) { return false; }
              });
              return {
                ui_version: document.querySelector('meta[name="labelagent-ui-version"]')?.content || null,
                theme_loaded: theme,
                variables: {
                  paper: style.getPropertyValue('--la-paper').trim(),
                  heading: style.getPropertyValue('--la-heading').trim(),
                  link: style.getPropertyValue('--la-link').trim(),
                  line: style.getPropertyValue('--la-line').trim(),
                  font: style.getPropertyValue('--la-sans').trim(),
                },
                geometry: {
                  toolbar_height: toolbar?.height || 0,
                  guide_height: guide?.height || 0,
                  grid_width: grid?.width || 0,
                  grid_height: grid?.height || 0,
                  workbench_height: workbench?.height || 0,
                  stage_width: stage?.width || 0,
                  stage_height: stage?.height || 0,
                  image_width: image?.width || 0,
                  image_height: image?.height || 0,
                  horizontal_letterbox: Math.max(0, (stage?.width || 0) - (image?.width || 0)),
                  vertical_letterbox: Math.max(0, (stage?.height || 0) - (image?.height || 0)),
                },
                responsive_layout: {
                  viewer_display: viewerStyle.display,
                  viewer_columns: viewerStyle.gridTemplateColumns,
                  batch_panel_height: batchPanel?.getBoundingClientRect().height || 0,
                  batch_compact: Boolean(batchPanel?.classList.contains('compact')),
                  candidate_panel_height: candidatePanel?.getBoundingClientRect().height || 0,
                  candidate_unavailable: Boolean(candidatePanel?.classList.contains('unavailable')),
                  inspection_client_height: inspectionPanel?.clientHeight || 0,
                  inspection_scroll_height: inspectionPanel?.scrollHeight || 0,
                  inspection_overflow_y: inspectionPanel ? getComputedStyle(inspectionPanel).overflowY : '',
                },
                controls: {
                  manual_mode: Boolean(document.querySelector('#manualModeButton')),
                  auto_mode: Boolean(document.querySelector('#autoModeButton')),
                  guide_action: Boolean(document.querySelector('#guideAction')),
                  play_button: Boolean(document.querySelector('#playButton')),
                  frame_number: Boolean(document.querySelector('#frameNumber')),
                  workbench: Boolean(document.querySelector('#workbench')),
                },
              };
            }"""
        )

        page.locator("#autoModeButton").click()
        page.wait_for_timeout(800)
        auto_path = output / "auto-workbench.png"
        page.screenshot(path=str(auto_path), full_page=False)

        metrics["auto_workspace"] = page.locator("body").get_attribute("data-workspace")
        metrics["console_errors"] = console_errors
        metrics["page_errors"] = page_errors
        metrics["http_status"] = response.status if response else None
        metrics["screenshots"] = {
            "manual": str(manual_path),
            "auto": str(auto_path),
        }
        metrics["passed"] = bool(
            metrics["http_status"] == 200
            and metrics["theme_loaded"]
            and metrics["ui_version"] == "159"
            and all(metrics["controls"].values())
            and metrics["auto_workspace"] == "auto"
            and not console_errors
            and not page_errors
        )
        (output / "acceptance.json").write_text(
            json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        browser.close()

    if not metrics["passed"]:
        raise SystemExit("visual theme acceptance failed")


if __name__ == "__main__":
    main()
