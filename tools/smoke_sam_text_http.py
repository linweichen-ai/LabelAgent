"""Send one audited text-prompt request to SAM and reject empty or invalid masks."""
import argparse
import base64
import hashlib
import json
from pathlib import Path
import time
import urllib.request

from PIL import Image


def validate_results(payload, width, height):
    results = payload.get("results") if isinstance(payload, dict) else None
    if not isinstance(results, list) or not results:
        raise ValueError("SAM text response contains no results")
    part_count = 0
    vertex_count = 0
    hole_count = 0
    scores = []
    for result_index, result in enumerate(results):
        parts = result.get("parts") if isinstance(result, dict) else None
        if not isinstance(parts, list) or not parts:
            raise ValueError(f"result {result_index} contains no parts")
        score = result.get("score")
        if isinstance(score, bool) or not isinstance(score, (int, float)):
            raise ValueError(f"result {result_index} has no numeric score")
        scores.append(float(score))
        for part_index, part in enumerate(parts):
            points = part.get("points") if isinstance(part, dict) else None
            if not isinstance(points, list) or len(points) < 3:
                raise ValueError(f"result {result_index} part {part_index} is empty")
            for point in points:
                if (not isinstance(point, list) or len(point) != 2 or
                        any(isinstance(value, bool) or not isinstance(value, (int, float))
                            for value in point)):
                    raise ValueError("mask vertex is not an [x, y] numeric pair")
                x, y = point
                if not 0 <= x < width or not 0 <= y < height:
                    raise ValueError(f"mask vertex is outside {width}x{height}: {point}")
            holes = part.get("holes", [])
            if not isinstance(holes, list):
                raise ValueError("part holes must be a list")
            for hole in holes:
                if not isinstance(hole, list) or len(hole) < 3:
                    raise ValueError("mask hole is empty")
                for point in hole:
                    if (not isinstance(point, list) or len(point) != 2 or
                            any(isinstance(value, bool) or not isinstance(value, (int, float))
                                for value in point)):
                        raise ValueError("hole vertex is not an [x, y] numeric pair")
                    x, y = point
                    if not 0 <= x < width or not 0 <= y < height:
                        raise ValueError(f"hole vertex is outside {width}x{height}: {point}")
            part_count += 1
            vertex_count += len(points)
            hole_count += len(holes)
    return {"result_count": len(results), "part_count": part_count,
            "outer_vertex_count": vertex_count, "hole_count": hole_count,
            "scores": scores}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image", required=True)
    parser.add_argument("--prompt", required=True)
    parser.add_argument("--url", default="http://127.0.0.1:18766")
    parser.add_argument("--output-json", required=True,
                        help="Path for the exact HTTP response bytes")
    args = parser.parse_args()

    image_path = Path(args.image).resolve()
    image_bytes = image_path.read_bytes()
    with Image.open(image_path) as image:
        width, height = image.size
        media_format = image.format
    body = json.dumps({"image": base64.b64encode(image_bytes).decode("ascii"),
                       "prompt": args.prompt}).encode("utf-8")
    request = urllib.request.Request(args.url.rstrip("/") + "/segment_text",
        data=body, headers={"Content-Type": "application/json"})
    started = time.monotonic()
    with urllib.request.urlopen(request, timeout=600) as response:
        response_bytes = response.read()
        status = response.status
    seconds = time.monotonic() - started
    output = Path(args.output_json).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_bytes(response_bytes)
    payload = json.loads(response_bytes)
    validation = validate_results(payload, width, height)
    summary = {
        "acceptance": "pass",
        "http_status": status,
        "seconds": round(seconds, 3),
        "endpoint": args.url.rstrip("/") + "/segment_text",
        "prompt": args.prompt,
        "image": str(image_path),
        "image_bytes": len(image_bytes),
        "image_width": width,
        "image_height": height,
        "image_format": media_format,
        "image_sha256": hashlib.sha256(image_bytes).hexdigest(),
        "response_bytes": len(response_bytes),
        "response_sha256": hashlib.sha256(response_bytes).hexdigest(),
        "output_json": str(output),
        **validation,
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
