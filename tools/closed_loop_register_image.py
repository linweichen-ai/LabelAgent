"""Register one immutable image as a closed-loop input without inventing model events."""
import argparse
import hashlib
import json
import mimetypes
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from source.agent.closed_loop import ClosedLoopLedger, SCHEMA_VERSION, canonical_json


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_once(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    encoded = json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    if path.exists():
        if path.read_text(encoding="utf-8") != encoded:
            raise ValueError("output bundle already exists with different content")
        return False
    path.write_text(encoded, encoding="utf-8", newline="\n")
    return True


def register_image(args):
    image_path = Path(args.image).resolve()
    if not image_path.is_file():
        raise ValueError("image does not exist or is not a regular file")
    content_hash = sha256_file(image_path)
    if args.expected_sha256 and content_hash != args.expected_sha256:
        raise ValueError("image SHA-256 does not match --expected-sha256")
    media_type = mimetypes.guess_type(image_path.name)[0] or "application/octet-stream"
    if not media_type.startswith("image/"):
        raise ValueError("input file does not have an image media type")

    envelope = {
        "dataset_id": args.dataset_id,
        "source_batch": args.source_batch,
        "frame_id": args.frame_id,
        "sensor": args.camera,
        "captured_at": args.captured_at,
        "source_kind": "real",
        "content_sha256": content_hash,
        "source_uri": args.source_uri,
        "timestamp_source": args.timestamp_source,
    }
    ledger = ClosedLoopLedger(args.ledger_root, envelope)
    registration = ledger.register()
    receipt = ledger.receipt()
    bundle = {
        "schema_version": SCHEMA_VERSION,
        "input": envelope,
        "local_readback": {
            "path": str(image_path),
            "bytes": image_path.stat().st_size,
            "media_type": media_type,
        },
        "registration": registration,
        "events": ledger.events(),
        "receipt": receipt,
        "evidence_boundary": {
            "model_events_recorded": False,
            "human_gate_recorded": False,
            "minimum_closed_loop_complete": False,
            "statement": "Immutable real-image input registration only; no SAM, VLM, Agent, or human result is claimed.",
        },
    }
    bundle["bundle_sha256"] = hashlib.sha256(canonical_json(bundle)).hexdigest()
    write_once(args.output, bundle)
    return bundle


def parser():
    value = argparse.ArgumentParser(description=__doc__)
    value.add_argument("--image", required=True)
    value.add_argument("--dataset-id", required=True)
    value.add_argument("--source-batch", required=True)
    value.add_argument("--frame-id", required=True)
    value.add_argument("--camera", required=True)
    value.add_argument("--captured-at", required=True)
    value.add_argument("--source-uri", required=True)
    value.add_argument("--timestamp-source", default="frame_id_unix_ns")
    value.add_argument("--expected-sha256")
    value.add_argument("--ledger-root", required=True)
    value.add_argument("--output", required=True)
    return value


def main():
    bundle = register_image(parser().parse_args())
    print(json.dumps(bundle, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
