"""Validate zoom-review assistant suggestions without human-truth promotion."""
from __future__ import annotations
import argparse,hashlib,json
from collections import Counter
from pathlib import Path
ALLOWED={"likely_ok_after_zoom","uncertain_needs_human","obvious_issue"}
def sha(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def read(path):
    v=json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(v,dict): raise ValueError("JSON object required")
    return v
def validate(pack_path,input_path):
    pack,submitted=read(pack_path),read(input_path); digest=sha(pack_path)
    if pack.get("schema_version")!="labelagent.cvat-triage-zoom-pack.v1" or pack.get("status")!="human_review_pending": raise ValueError("zoom pack identity differs")
    if submitted.get("zoom_pack_manifest_sha256")!=digest: raise ValueError("zoom triage binds a different pack")
    expected=[row.get("candidate_id") for row in pack.get("items") or [] if isinstance(row,dict)]; rows=submitted.get("decisions")
    if not isinstance(rows,list): raise ValueError("zoom decisions are absent")
    actual=[row.get("candidate_id") for row in rows if isinstance(row,dict)]
    if len(expected)!=len(set(expected)) or len(actual)!=len(set(actual)) or set(expected)!=set(actual): raise ValueError("zoom decisions do not exactly cover pack candidates")
    counts=Counter(); normalized=[]
    for row in rows:
        decision,reason=row.get("zoom_triage"),row.get("reason")
        if decision not in ALLOWED: raise ValueError("unknown zoom triage decision")
        if not isinstance(reason,str) or not reason.strip() or len(reason)>300: raise ValueError("invalid zoom triage reason")
        counts[decision]+=1; normalized.append({"candidate_id":row["candidate_id"],"zoom_triage":decision,"reason":reason.strip()})
    normalized.sort(key=lambda x:x["candidate_id"]); private={"schema_version":"labelagent.cvat-zoom-assistant-triage.v1","status":"zoom_triage_complete_human_signoff_pending","zoom_pack_manifest_sha256":digest,"decision_counts":{k:counts.get(k,0) for k in sorted(ALLOWED)},"decisions":normalized,"human_truth":False,"human_signoff_count":0,"training_authorized":False,"automatic_approval":False}; public={k:private[k] for k in ("schema_version","status","zoom_pack_manifest_sha256","decision_counts","human_truth","human_signoff_count","training_authorized","automatic_approval")}; public.update({"private_candidate_ids_published":False,"private_reasons_published":False}); return private,public
def main():
    p=argparse.ArgumentParser(); p.add_argument("--zoom-pack",required=True,type=Path); p.add_argument("--triage-input",required=True,type=Path); p.add_argument("--private-output",required=True,type=Path); p.add_argument("--receipt-output",required=True,type=Path); a=p.parse_args()
    if a.private_output.exists() or a.receipt_output.exists(): raise ValueError("choose fresh outputs")
    private,public=validate(a.zoom_pack,a.triage_input); a.private_output.parent.mkdir(parents=True,exist_ok=True); a.private_output.write_text(json.dumps(private,ensure_ascii=False,indent=2)+"\n",encoding="utf-8"); public["private_triage_sha256"]=sha(a.private_output); a.receipt_output.parent.mkdir(parents=True,exist_ok=True); a.receipt_output.write_text(json.dumps(public,ensure_ascii=False,indent=2)+"\n",encoding="utf-8"); print(json.dumps(public,ensure_ascii=False)); return 0
if __name__=="__main__": raise SystemExit(main())
