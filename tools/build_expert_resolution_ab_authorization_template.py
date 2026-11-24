"""Build a non-authorizing, hash-bound template for expert resolution A/B."""
import argparse
import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools.run_mixed_hard_external_shadow import QUEUE_SCHEMA, sha256_file


SCHEMA_VERSION = "labelagent.external-upload-authorization.v2"
VARIANTS = [
    {"name": "source_640x512", "mode": "source",
     "expected_width": 640, "expected_height": 512},
    {"name": "resize_512x410", "mode": "resize_lanczos_jpeg",
     "width": 512, "height": 410, "jpeg_quality": 90},
]


def build_template(queue_path):
    queue_path = Path(queue_path)
    queue = json.loads(queue_path.read_text(encoding="utf-8"))
    samples = queue.get("samples") if isinstance(queue, dict) else None
    if (queue.get("schema_version") != QUEUE_SCHEMA
            or not isinstance(samples, list) or not samples
            or queue.get("sample_count") != len(samples)
            or queue.get("candidate_only") is not True
            or queue.get("human_truth") is not False
            or queue.get("approved_for_training") is not False
            or queue.get("test24_used") is not False
            or queue.get("images_copied") is not False
            or queue.get("image_paths_included") is not False):
        raise ValueError("expert queue identity or safety boundary differs")
    upload_count = len(samples) * len(VARIANTS)
    return {
        "schema_version": SCHEMA_VERSION,
        "status": "awaiting_explicit_user_authorization",
        "scope": "mixed_hard_external_resolution_ab_shadow",
        "queue_sha256": sha256_file(queue_path),
        "authorized": False,
        "authorized_by": "",
        "authorized_sample_count": len(samples),
        "authorized_image_variants": VARIANTS,
        "authorized_upload_count": upload_count,
        "max_external_calls": upload_count,
        "purpose": "candidate-only 640x512 versus 512x410 expert quality and efficiency A/B",
        "candidate_only": True,
        "human_truth": False,
        "approved_for_training": False,
        "test24_used": False,
        "automatic_truth_write": False,
        "production_deployed": False,
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--queue", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    template = build_template(args.queue)
    encoded = json.dumps(template, ensure_ascii=False, indent=2) + "\n"
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(args.output.suffix + ".tmp")
    temporary.write_text(encoded, encoding="utf-8")
    temporary.replace(args.output)
    print(encoded, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

