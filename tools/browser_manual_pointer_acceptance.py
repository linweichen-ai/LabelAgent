"""Exercise manual canvas tools in a browser without loading factory data.

The acceptance uses a generated blank image and an in-memory draft polygon.  It
does not call SAM/VLM, save truth, or mutate the workbench state on disk.
"""

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


def canvas_point(box: dict[str, float], x: float, y: float) -> tuple[float, float]:
    return (
        box["x"] + x / 640 * box["width"],
        box["y"] + y / 480 * box["height"],
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", required=True)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()

    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    console_errors: list[str] = []
    page_errors: list[str] = []
    write_requests: list[dict[str, str]] = []

    with sync_playwright() as runtime:
        browser = runtime.chromium.launch(
            headless=True, executable_path=installed_browser()
        )
        page = browser.new_page(viewport={"width": 1440, "height": 1000})
        page.on(
            "console",
            lambda message: console_errors.append(message.text)
            if message.type == "error"
            else None,
        )
        page.on("pageerror", lambda error: page_errors.append(str(error)))
        page.goto(args.url, wait_until="domcontentloaded", timeout=30_000)
        page.wait_for_selector("#canvas", state="attached", timeout=15_000)
        page.wait_for_function(
            "() => { try { modelHealthState; return true; } catch (_) { return false; } }"
        )
        page.evaluate("window.labelAgentWorkspaceReady")
        page.evaluate(
            """async () => {
              frame = 0;
              topic = 'synthetic-manual';
              loadedDataPath = 'synthetic://manual-pointer-acceptance';
              labels = ['pallet'];
              shapes = {'0': [{label: 'pallet', points: [[100,100],[300,100],[300,300],[100,300]]}]};
              truthReviewStatus = {};
              playing = null;
              loading = true;
              selectedShapeIndex = 0;
              const select = document.querySelector('#labelSelect');
              select.innerHTML = '<option value="pallet">pallet</option>';
              const source = document.createElement('canvas');
              source.width = 640;
              source.height = 480;
              const context = source.getContext('2d');
              context.fillStyle = '#d9dde2';
              context.fillRect(0, 0, 640, 480);
              const image = document.querySelector('#frameImage');
              image.src = source.toDataURL('image/png');
              await image.decode();
              loading = false;
              document.querySelector('#primaryWindow').hidden = false;
              setAnnotationMode(true);
              updateGrid();
              draw();
            }"""
        )
        page.wait_for_function(
            "document.querySelector('#frameImage').naturalWidth === 640"
        )
        page.evaluate("refreshShapeList(); refreshPointTools(); draw()")
        page.wait_for_timeout(100)
        page.on(
            "request",
            lambda request: write_requests.append(
                {"method": request.method, "url": request.url}
            )
            if request.method not in {"GET", "HEAD", "OPTIONS"}
            else None,
        )

        page.evaluate("document.querySelector('#vertexEditButton').click()")
        box = page.locator("#canvas").bounding_box()
        if not box:
            raise RuntimeError("canvas has no visible bounds")
        start = canvas_point(box, 100, 100)
        moved = canvas_point(box, 135, 125)
        page.mouse.move(*start)
        page.mouse.down()
        page.mouse.move(*moved, steps=6)
        page.mouse.up()
        moved_vertex = page.evaluate(
            "JSON.parse(JSON.stringify(selectedShapes()[0].points[0]))"
        )

        page.keyboard.press("Control+Z")
        restored_vertex = page.evaluate(
            "JSON.parse(JSON.stringify(selectedShapes()[0].points[0]))"
        )

        page.evaluate("document.querySelector('#freehandButton').click()")
        freehand_points = [(360, 120), (480, 120), (500, 220), (430, 280), (350, 210)]
        first = canvas_point(box, *freehand_points[0])
        page.mouse.move(*first)
        page.mouse.down()
        for point in freehand_points[1:]:
            page.mouse.move(*canvas_point(box, *point), steps=5)
        page.mouse.up()
        freehand_vertex_count = page.evaluate("selectedShapes().at(-1).points.length")

        page.evaluate("document.querySelector('#ellipseButton').click()")
        ellipse_start = canvas_point(box, 360, 320)
        ellipse_end = canvas_point(box, 520, 430)
        page.mouse.move(*ellipse_start)
        page.mouse.down()
        page.mouse.move(*ellipse_end, steps=6)
        page.mouse.up()
        ellipse_vertex_count = page.evaluate("selectedShapes().at(-1).points.length")
        shape_count = page.evaluate("selectedShapes().length")
        active_drags = page.evaluate(
            "({freehand: Boolean(freehandDrag), rectangle: Boolean(rectangleDrag), vertex: Boolean(vertexDrag)})"
        )

        screenshot = output / "manual-pointer-tools.png"
        page.screenshot(path=str(screenshot), full_page=False)
        browser.close()

    receipt = {
        "schema_version": "labelagent.manual-pointer-acceptance.v1",
        "source": "synthetic_blank_image_only",
        "factory_data_used": False,
        "truth_writes": 0,
        "model_calls": 0,
        "moved_vertex": moved_vertex,
        "restored_vertex": restored_vertex,
        "freehand_vertex_count": freehand_vertex_count,
        "ellipse_vertex_count": ellipse_vertex_count,
        "shape_count": shape_count,
        "active_drags_after_completion": active_drags,
        "write_requests": write_requests,
        "console_errors": console_errors,
        "page_errors": page_errors,
        "screenshot": screenshot.name,
    }
    receipt["passed"] = bool(
        moved_vertex != [100, 100]
        and restored_vertex == [100, 100]
        and freehand_vertex_count >= 3
        and ellipse_vertex_count == 40
        and shape_count == 3
        and not any(active_drags.values())
        and not write_requests
        and not console_errors
        and not page_errors
    )
    (output / "acceptance.json").write_text(
        json.dumps(receipt, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    if not receipt["passed"]:
        raise SystemExit("manual pointer acceptance failed")


if __name__ == "__main__":
    main()
