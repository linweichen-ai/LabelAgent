"""Validate private assistant triage without promoting it to human truth."""
from __future__ import annotations
import argparse, hashlib, json
from collections import Counter
from pathlib import Path
from typing import Any
DECISIONS = {"likely_ok", "needs_human_review", "obvious_issue"}
def sha256(path: Path) -> str: return hashlib.sha256(path.read_bytes()).hexdigest()
def read(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict): raise ValueError(f"JSON object required: {path}")
    return value
def validate(pack_manifest: Path, triage_input: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    pack, submitted = read(pack_manifest), read(triage_input); pack_sha = sha256(pack_manifest)
    if pack.get("schema_version") != "labelagent.cvat-geometry-priority-review-pack.v1" or pack.get("status") != "assistant_triage_pending": raise ValueError("review pack identity differs")
    if submitted.get("review_pack_manifest_sha256") != pack_sha: raise ValueError("triage input binds a different review pack")
    decisions = submitted.get("decisions")
    if not isinstance(decisions, list): raise ValueError("triage decision list is absent")
    high_ids = [cid for page in pack.get("pages") or [] if isinstance(page, dict) and page.get("priority") == "high" for cid in (page.get("candidate_ids") or [])]
    submitted_ids = [row.get("candidate_id") for row in decisions if isinstance(row, dict)]
    if len(high_ids) != len(set(high_ids)) or len(submitted_ids) != len(set(submitted_ids)): raise ValueError("candidate ids are repeated")
    if set(high_ids) != set(submitted_ids): raise ValueError("triage decisions do not exactly cover high-priority candidates")
    counts: Counter[str] = Counter(); normalized = []
    for row in decisions:
        if not isinstance(row, dict): raise ValueError("triage row is not an object")
        decision, reason = row.get("assistant_triage"), row.get("reason")
        if decision not in DECISIONS: raise ValueError(f"unknown assistant triage decision: {decision}")
        if not isinstance(reason, str) or not reason.strip() or len(reason) > 240: raise ValueError("triage reason is absent or too long")
        counts[decision] += 1; normalized.append({"candidate_id": row["candidate_id"], "assistant_triage": decision, "reason": reason.strip()})
    normalized.sort(key=lambda row: row["candidate_id"])
    private = {"schema_version":"labelagent.cvat-assistant-triage.v1","status":"assistant_triage_complete_human_review_pending","review_pack_manifest_sha256":pack_sha,"reviewer_kind":"independent_assistant_visual_review","decision_counts":{key:counts.get(key,0) for key in sorted(DECISIONS)},"decisions":normalized,"human_truth":False,"human_signoff_count":0,"training_authorized":False,"automatic_approval":False,"next_gate":"human_review_of_needs_human_review_and_all_training_candidates"}
    public = {key: private[key] for key in ("schema_version","status","review_pack_manifest_sha256","reviewer_kind","decision_counts","human_truth","human_signoff_count","training_authorized","automatic_approval","next_gate")}
    public.update({"private_candidate_ids_published":False,"private_reasons_published":False})
    return private, public
def main() -> int:
    parser=argparse.ArgumentParser(description=__doc__); parser.add_argument("--review-pack",required=True,type=Path); parser.add_argument("--triage-input",required=True,type=Path); parser.add_argument("--private-output",required=True,type=Path); parser.add_argument("--receipt-output",required=True,type=Path); args=parser.parse_args()
    if args.private_output.exists() or args.receipt_output.exists(): raise ValueError("choose fresh output paths")
    private,public=validate(args.review_pack,args.triage_input); args.private_output.parent.mkdir(parents=True,exist_ok=True); args.private_output.write_text(json.dumps(private,ensure_ascii=False,indent=2)+"\n",encoding="utf-8"); public["private_triage_sha256"]=sha256(args.private_output); args.receipt_output.parent.mkdir(parents=True,exist_ok=True); args.receipt_output.write_text(json.dumps(public,ensure_ascii=False,indent=2)+"\n",encoding="utf-8"); print(json.dumps(public,ensure_ascii=False)); return 0
if __name__ == "__main__": raise SystemExit(main())
