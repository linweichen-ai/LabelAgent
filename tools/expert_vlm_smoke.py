"""Safe external-VLM connectivity smoke using only a generated synthetic image."""
import argparse
import hashlib
import io
import json
import os
import re
import sys
import time
from pathlib import Path
from urllib.parse import urlsplit

from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from source.agent.vlm_providers import (SUPPORTED_PROVIDERS,
                                        validate_external_credential)
from source.agent.workflow import chat_external_expert


SCHEMA_VERSION = "labelagent.external-vlm-synthetic-smoke.v1"
KEY_ENV_PATTERN = re.compile(r"^[A-Z][A-Z0-9_]{0,127}$")
RESPONSE_SCHEMA = {
    "type": "object",
    "properties": {
        "status": {"type": "string", "enum": ["synthetic_image_received"]},
    },
    "required": ["status"],
    "additionalProperties": False,
}


def synthetic_factory_image():
    """Create a deterministic, data-free image that cannot contain factory IP."""
    image = Image.new("RGB", (640, 512), (225, 228, 232))
    draw = ImageDraw.Draw(image)
    draw.rectangle((0, 320, 639, 511), fill=(135, 138, 142))
    draw.rectangle((65, 360, 285, 420), fill=(150, 92, 45))
    draw.rectangle((85, 335, 265, 360), fill=(170, 110, 55))
    draw.rectangle((430, 255, 565, 345), fill=(195, 55, 45))
    draw.rectangle((470, 205, 525, 255), fill=(75, 78, 82))
    draw.polygon([(350, 320), (610, 320), (639, 460), (400, 460)],
                 fill=(190, 180, 145))
    output = io.BytesIO()
    image.save(output, "JPEG", quality=90, optimize=False, progressive=False)
    return output.getvalue()


def validate_config(provider, url, model, key_env):
    if provider not in SUPPORTED_PROVIDERS:
        raise ValueError(f"unsupported provider: {provider}")
    parsed = urlsplit(url)
    if (parsed.scheme not in {"http", "https"} or not parsed.hostname
            or parsed.username is not None or parsed.password is not None):
        raise ValueError("expert URL must be HTTP(S) without embedded credentials")
    if not model.strip():
        raise ValueError("expert model is required")
    if KEY_ENV_PATTERN.fullmatch(key_env) is None:
        raise ValueError("expert key environment variable name is invalid")
    return parsed.hostname


def run_smoke(*, provider, url, model, key_env, execute=False,
              acknowledge_synthetic_upload=False, environ=None,
              chat_fn=chat_external_expert):
    environ = os.environ if environ is None else environ
    host = validate_config(provider, url, model, key_env)
    image = synthetic_factory_image()
    receipt = {
        "schema_version": SCHEMA_VERSION,
        "mode": "execute" if execute else "dry_run",
        "provider": provider,
        "model": model,
        "endpoint_host": host,
        "synthetic_image_sha256": hashlib.sha256(image).hexdigest(),
        "synthetic_image_bytes": len(image),
        "factory_data_uploaded": False,
        "candidate_only": True,
        "human_truth": False,
        "approved_for_training": False,
        "connection_tested": False,
        "external_upload_performed": False,
    }
    if not execute:
        receipt["ready_to_execute"] = bool(environ.get(key_env))
        return receipt
    if acknowledge_synthetic_upload is not True:
        raise ValueError("execute requires explicit synthetic-upload acknowledgement")
    credential = environ.get(key_env)
    if not credential:
        raise ValueError(f"expert credential environment variable is missing: {key_env}")
    validate_external_credential(credential)
    config = {
        "llm_expert_enabled": True,
        "llm_expert_provider": provider,
        "llm_expert_url": url,
        "llm_expert_model": model,
        "llm_expert_key_env": key_env,
        "llm_expert_timeout_seconds": 120,
    }
    started = time.perf_counter()
    result = chat_fn(
        config,
        "This is a generated synthetic test image. Return only the requested JSON status.",
        image,
        response_schema=RESPONSE_SCHEMA,
    )
    latency_ms = round((time.perf_counter() - started) * 1000, 3)
    if result != {"status": "synthetic_image_received"}:
        raise ValueError("external expert returned an unexpected smoke response")
    receipt.update(
        connection_tested=True,
        external_upload_performed=True,
        uploaded_content="generated_synthetic_image_only",
        latency_ms=latency_ms,
        response_sha256=hashlib.sha256(
            json.dumps(result, sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest(),
        status="passed",
    )
    return receipt


def main(argv=None):
    parser = argparse.ArgumentParser(
        description="Validate or execute a synthetic-only external VLM smoke test")
    parser.add_argument("--provider", required=True, choices=sorted(SUPPORTED_PROVIDERS))
    parser.add_argument("--url", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--key-env", required=True)
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--acknowledge-synthetic-upload", action="store_true")
    args = parser.parse_args(argv)
    receipt = run_smoke(
        provider=args.provider, url=args.url, model=args.model,
        key_env=args.key_env, execute=args.execute,
        acknowledge_synthetic_upload=args.acknowledge_synthetic_upload)
    print(json.dumps(receipt, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
