"""Send one auditable point-prompt request to a running SAM3 HTTP service."""
import argparse
import base64
import hashlib
import json
from pathlib import Path
import time
import urllib.request


def point(value):
    fields = value.split(",")
    if len(fields) != 3:
        raise argparse.ArgumentTypeError("point must be x,y,label")
    x, y, label = (int(field) for field in fields)
    if label not in (0, 1):
        raise argparse.ArgumentTypeError("point label must be 0 or 1")
    return {"x": x, "y": y, "label": label}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image", required=True)
    parser.add_argument("--url", default="http://127.0.0.1:18766")
    parser.add_argument("--point", action="append", type=point, required=True)
    parser.add_argument("--output-json")
    args = parser.parse_args()

    image_path = Path(args.image).resolve()
    image = image_path.read_bytes()
    payload = json.dumps({"image": base64.b64encode(image).decode("ascii"),
                          "points": args.point}).encode()
    request = urllib.request.Request(args.url.rstrip("/") + "/segment_points",
        data=payload, headers={"Content-Type": "application/json"})
    started = time.monotonic()
    with urllib.request.urlopen(request, timeout=600) as response:
        result = json.load(response)
        status = response.status
    elapsed = time.monotonic() - started
    if args.output_json:
        Path(args.output_json).write_text(json.dumps(result, ensure_ascii=False,
                                                     indent=2), encoding="utf-8")
    parts = result.get("parts", [])
    summary = {
        "http_status": status,
        "seconds": round(elapsed, 3),
        "image": str(image_path),
        "image_bytes": len(image),
        "image_sha256": hashlib.sha256(image).hexdigest(),
        "points": args.point,
        "score": result.get("score"),
        "part_count": len(parts),
        "outer_vertex_count": sum(len(part.get("points", [])) for part in parts),
        "hole_count": sum(len(part.get("holes", [])) for part in parts),
        "warnings": result.get("warnings", []),
        "output_json": str(Path(args.output_json).resolve()) if args.output_json else None,
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
