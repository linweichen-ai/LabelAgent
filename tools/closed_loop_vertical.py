"""Register one real workbench frame and emit an idempotent closed-loop receipt."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys
import urllib.parse
import urllib.request

from mcap.reader import make_reader
from mcap_protobuf.decoder import DecoderFactory

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from source.agent.closed_loop import ClosedLoopLedger, SCHEMA_VERSION, canonical_json
from source.agent.sensors import message_time


def request(url):
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    with opener.open(url, timeout=60) as response:
        return response.read(), response.headers.get_content_type()


def request_json(url):
    data, _ = request(url)
    return json.loads(data)


def captured_time(path, topic, frame):
    factory = DecoderFactory()
    with Path(path).open("rb") as stream:
        messages = make_reader(stream).iter_messages(topics=[topic])
        for index, (schema, channel, message) in enumerate(messages):
            if index != frame:
                continue
            decoder = factory.decoder_for(channel.message_encoding, schema)
            decoded = decoder(message.data) if decoder else message.data
            timestamp = message_time(decoded, message)
            source = "capture" if timestamp else "log_time"
            timestamp = timestamp or message.log_time
            value = datetime.fromtimestamp(timestamp / 1_000_000_000,
                                           timezone.utc).isoformat().replace("+00:00", "Z")
            return value, source
    raise ValueError("frame was not found in the selected MCAP topic")


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


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workbench", default="http://127.0.0.1:8767")
    parser.add_argument("--dataset-id", required=True)
    parser.add_argument("--source-batch", required=True)
    parser.add_argument("--frame", type=int, required=True)
    parser.add_argument("--topic", required=True)
    parser.add_argument("--source-kind", choices=["real", "sim"], required=True)
    parser.add_argument("--ledger-root", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    base = args.workbench.rstrip("/")
    parsed = urllib.parse.urlparse(base)
    if parsed.scheme != "http" or parsed.hostname not in {"127.0.0.1", "localhost", "::1"}:
        raise ValueError("workbench must be a loopback HTTP URL")
    state = request_json(base + "/api/state")
    if not state.get("loaded") or args.topic not in state.get("topics", {}):
        raise ValueError("the requested dataset/topic is not loaded")
    topic_count = state["topics"][args.topic]["count"]
    if not 0 <= args.frame < topic_count:
        raise ValueError("frame is outside the selected topic")
    image, media_type = request(base + f"/api/frame/{args.frame}?" +
                                urllib.parse.urlencode({"topic": args.topic}))
    image_hash = hashlib.sha256(image).hexdigest()
    timestamp, timestamp_source = captured_time(state["path"], args.topic, args.frame)
    envelope = {"dataset_id": args.dataset_id, "source_batch": args.source_batch,
                "frame_id": str(args.frame), "sensor": args.topic,
                "captured_at": timestamp, "source_kind": args.source_kind,
                "content_sha256": image_hash, "source_uri": state["path"],
                "timestamp_source": timestamp_source}
    ledger = ClosedLoopLedger(args.ledger_root, envelope)
    registration = ledger.register()
    history = ledger.events()
    truth = state.get("truth", {}).get(str(args.frame), [])
    references = state.get("reference_frames", {}).get(args.topic, [])
    if history[-1]["state"] == "received" and (not truth or args.frame not in references):
        reasons = []
        if not truth:
            reasons.append("missing_human_truth")
        if args.frame not in references:
            reasons.append("not_a_reference_frame")
        ledger.append_event("human_review_required",
            actor={"type": "system", "id": "vertical-frame-gate"},
            action="route_missing_reference_to_human", label_layer="preannotation",
            confidence=None,
            payload={"reasons": reasons, "routing": {"novel": False,
                "low_confidence": False, "model_conflict": False,
                "quality_failure": True}, "production_truth_write": False},
            artifacts=[{"kind": "input_frame", "sha256": image_hash,
                        "bytes": len(image), "media_type": media_type}],
            metrics={"end_to_end_latency_ms": None, "human_minutes": None,
                     "single_frame_cost": None})
    receipt = ledger.receipt()
    bundle = {"schema_version": SCHEMA_VERSION, "input": envelope,
              "registration": registration, "events": ledger.events(),
              "receipt": receipt,
              "workbench_readback": {"truth_shape_count": len(truth),
                  "is_reference_frame": args.frame in references,
                  "candidate_shape_count": len(state.get("candidates", {}).get(str(args.frame), []))}}
    bundle["bundle_sha256"] = hashlib.sha256(canonical_json(bundle)).hexdigest()
    write_once(args.output, bundle)
    print(json.dumps(bundle, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
