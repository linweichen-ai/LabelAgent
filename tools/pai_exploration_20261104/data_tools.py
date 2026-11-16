"""Human-review ingestion and grouped export for the isolated LabelAgent pilot."""
import argparse
import hashlib
import json
from pathlib import Path

PROMPT = (
    "<image>\n<image>\n第一张是完整场景，第二张是同一候选的局部图，黄色轮廓标出待检查对象。"
    "仅复查这个候选，不要评价其他实例。pallet指独立货物托盘，person指人；"
    "叉车货叉、托盘车车体、反光及背景不能仅因外形相似就归为托盘。"
    "保留用keep，错误候选用reject，类别错误但轮廓可用用relabel，看不清用uncertain。"
    "只输出JSON：{\"decision\":\"keep|reject|relabel|uncertain\","
    "\"label\":\"pallet|person|other|unknown\"}。"
)
DECISIONS = {"keep", "reject", "relabel", "uncertain"}


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load_records(path):
    return [json.loads(line) for line in Path(path).read_text(encoding="utf-8-sig").splitlines() if line.strip()]


def image_path(root, name):
    root = Path(root).resolve()
    path = (root / name).resolve()
    if not path.is_relative_to(root) or not path.is_file():
        raise ValueError(f"Missing or outside data root: {name}")
    return path


def check_records(records, root, require_reviewed=False):
    seen = set()
    for r in records:
        sid = r["sample_id"]
        if sid in seen:
            raise ValueError(f"Duplicate sample_id: {sid}")
        seen.add(sid)
        if r.get("source_kind") != "model_candidate_unconfirmed":
            raise ValueError(f"Unsupported lineage for this pilot: {sid}")
        if not r.get("capture_id") or not r.get("group_id") or not r.get("source_image_sha256"):
            raise ValueError(f"Missing group/lineage: {sid}")
        if len(r["images"]) != 2:
            raise ValueError(f"Expected scene plus crop: {sid}")
        for name in r["images"]:
            p = image_path(root, name)
            if sha256(p) != r["image_sha256"][name]:
                raise ValueError(f"Image digest mismatch: {sid}")
        reviewed = r.get("review_status") == "human_reviewed"
        if require_reviewed and not reviewed:
            raise ValueError(f"Pending human review: {sid}")
        if reviewed:
            if not str(r.get("reviewer", "")).strip() or not r.get("reviewed_at"):
                raise ValueError(f"Missing human-review receipt: {sid}")
            decision, label = r.get("decision"), r.get("correct_label")
            if decision not in DECISIONS:
                raise ValueError(f"Invalid decision: {sid}")
            allowed = {"reject": {"other"}, "uncertain": {"unknown"},
                       "keep": {r["proposed_label"]}, "relabel": {"pallet", "person"}}[decision]
            if label not in allowed or (decision == "relabel" and label == r["proposed_label"]):
                raise ValueError(f"Invalid/no-op relabel: {sid}")
    return records


def expected(r):
    return {"decision": r["decision"], "label": r["correct_label"]}


def export_reviewed(reviewed, root, out, dev_group, min_train, min_dev):
    records = check_records(load_records(reviewed), root, require_reviewed=True)
    train = [r for r in records if r["group_id"] != dev_group]
    dev = [r for r in records if r["group_id"] == dev_group]
    if len(train) < min_train or len(dev) < min_dev:
        raise ValueError(f"Too few reviewed samples: train={len(train)}, dev={len(dev)}")
    for key in ["group_id", "source_image_sha256"]:
        if {r[key] for r in train} & {r[key] for r in dev}:
            raise ValueError(f"Train/dev leakage: {key}")
    if {h for r in train for h in r['image_sha256'].values()} & {h for r in dev for h in r['image_sha256'].values()}:
        raise ValueError('Train/dev leakage: derived image bytes')
    out = Path(out)
    if out.exists():
        raise ValueError("Choose a fresh export directory; existing data was not overwritten")
    out.mkdir(parents=True)
    for name, rows in [("train", train), ("dev", dev)]:
        converted = [{"image": r["images"], "conversations": [
            {"from": "human", "value": PROMPT},
            {"from": "gpt", "value": json.dumps(expected(r), ensure_ascii=False, separators=(",", ":"))}
        ]} for r in rows]
        (out / f"{name}.json").write_text(json.dumps(converted, ensure_ascii=False, indent=2), encoding="utf-8")
        (out / f"{name}_reviewed.jsonl").write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows), encoding="utf-8")
    receipt = {"status": "human_reviewed_grouped_pilot_only", "train": len(train), "dev": len(dev),
               "train_groups": sorted({r["group_id"] for r in train}), "dev_groups": [dev_group],
               "input_sha256": sha256(reviewed), "train_sha256": sha256(out / "train.json"),
               "dev_sha256": sha256(out / "dev.json"), "external_test_available": False,
               "data_root": str(Path(root).resolve()), "prompt": PROMPT,
               "caveat": "Same-capture development pilot; not independent external accuracy"}
    (out / "export_receipt.json").write_text(json.dumps(receipt, ensure_ascii=False, indent=2), encoding="utf-8")
    return receipt


def main():
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command", required=True)
    v = sub.add_parser("validate")
    v.add_argument("--records", required=True); v.add_argument("--data-root", required=True)
    v.add_argument("--require-reviewed", action="store_true")
    e = sub.add_parser("export")
    e.add_argument("--records", required=True); e.add_argument("--data-root", required=True)
    e.add_argument("--out", required=True); e.add_argument("--dev-group", required=True)
    e.add_argument("--min-train", type=int, default=24); e.add_argument("--min-dev", type=int, default=6)
    a = parser.parse_args()
    if a.command == "validate":
        rows = check_records(load_records(a.records), a.data_root, a.require_reviewed)
        print(json.dumps({"valid_records": len(rows), "human_reviewed": sum(r.get("review_status") == "human_reviewed" for r in rows)}))
    else:
        print(json.dumps(export_reviewed(a.records, a.data_root, a.out, a.dev_group, a.min_train, a.min_dev), ensure_ascii=False))


if __name__ == "__main__":
    main()
