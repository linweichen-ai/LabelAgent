"""Probe a real loopback VLM, then optionally configure LabelAgent."""
import argparse
import base64
import io
import json
from pathlib import Path
import urllib.parse
import urllib.request
from PIL import Image


def loopback_url(value):
    parsed = urllib.parse.urlparse(value)
    if parsed.scheme != "http" or parsed.hostname not in {"127.0.0.1", "localhost", "::1"} or parsed.username or parsed.password or parsed.query or parsed.fragment:
        raise ValueError("Use an HTTP loopback URL backed by the private SSH tunnel")
    return value.rstrip("/")


def request_json(url, body=None):
    data = None if body is None else json.dumps(body).encode("utf-8")
    request = urllib.request.Request(url, data, headers={"Content-Type": "application/json"})
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    with opener.open(request, timeout=300) as response:
        return json.load(response)


def require_real_service(base, model):
    base = loopback_url(base)
    if not base.endswith("/v1"):
        raise ValueError("VLM base URL must end in /v1")
    health = request_json(base[:-3] + "/health")
    if not health.get("model_loaded") or health.get("backend") != "transformers_local_cuda" or health.get("model") != model:
        raise ValueError("A loaded real CUDA backend with the requested model is required")
    models = request_json(base + "/models")
    if model not in {item["id"] for item in models.get("data", [])}:
        raise ValueError("Requested model is not served")
    return health


def probe(base, model, image_paths):
    base = loopback_url(base)
    require_real_service(base, model)
    parts = [{"type": "text", "text": '观察所附图像，返回严格 JSON：{"visible_objects":["图中可见的物体类别"],"image_count":图片数}。不要输出其他文字。'}]
    if not 1 <= len(image_paths) <= 8:
        raise ValueError("Provide one to eight images")
    for path in image_paths:
        with Image.open(path) as image:
            output = io.BytesIO()
            image.convert("RGB").save(output, "JPEG", quality=90)
        parts.append({"type": "image_url", "image_url": {"url": "data:image/jpeg;base64," + base64.b64encode(output.getvalue()).decode()}})
    result = request_json(base + "/chat/completions", {"model": model, "temperature": 0, "max_tokens": 512,
                          "messages": [{"role": "user", "content": parts}]})
    text = result["choices"][0]["message"]["content"].strip().removeprefix("```json").removeprefix("```").removesuffix("```").strip()
    decision = json.loads(text)
    if not isinstance(decision, dict) or not isinstance(decision.get("visible_objects"), list):
        raise ValueError("Real vision probe did not return the required object")
    if result.get("local_vlm", {}).get("image_count") != len(image_paths):
        raise ValueError("Backend did not confirm receiving all images")
    return {"real_vlm_probe_passed": True, "input_images": len(image_paths),
            "usage": result.get("usage"), "full_agent_verified": False}


def apply_configuration(base, model, workbench):
    base, workbench = loopback_url(base), loopback_url(workbench)
    require_real_service(base, model)
    if request_json(workbench + "/api/run/status")["status"] != "idle":
        raise ValueError("Workbench has an active task; configuration was not changed")
    # Explicit nonsecret value replaces any previously configured cloud key.
    return request_json(workbench + "/api/config", {"llm_url": base, "llm_model": model, "llm_key": "local"})


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base", default="http://127.0.0.1:18868/v1")
    parser.add_argument("--model", default="labelagent-vlm")
    parser.add_argument("--image", action="append", required=True)
    parser.add_argument("--workbench", default="http://127.0.0.1:8765")
    parser.add_argument("--apply", action="store_true")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    result = probe(args.base, args.model, args.image)
    if args.apply:
        result["configuration"] = apply_configuration(args.base, args.model, args.workbench)
    Path(args.output).write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))
