"""Validate assistant temporal triage without promoting it to human truth."""
from __future__ import annotations
import argparse,hashlib,json
from collections import Counter
from pathlib import Path
ALLOWED={"likely_ok_with_temporal","still_uncertain","obvious_issue"}
def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def read(p):
    v=json.loads(Path(p).read_text(encoding="utf-8"))
    if not isinstance(v,dict): raise ValueError("JSON object required")
    return v
def validate(pack_path,input_path):
    pack,submitted=read(pack_path),read(input_path); digest=sha(pack_path)
    if pack.get("schema_version")!="labelagent.cvat-uncertain-temporal-context.v1" or pack.get("status")!="assistant_review_pending": raise ValueError("temporal pack identity differs")
    if submitted.get("temporal_pack_manifest_sha256")!=digest: raise ValueError("triage binds a different temporal pack")
    expected=[r.get("candidate_id") for r in pack.get("items") or [] if isinstance(r,dict)]; rows=submitted.get("decisions")
    if not isinstance(rows,list): raise ValueError("temporal decisions absent")
    actual=[r.get("candidate_id") for r in rows if isinstance(r,dict)]
    if len(expected)!=len(set(expected)) or len(actual)!=len(set(actual)) or set(expected)!=set(actual): raise ValueError("temporal decisions do not exactly cover pack candidates")
    counts=Counter(); normalized=[]
    for row in rows:
        decision,reason=row.get("temporal_triage"),row.get("reason")
        if decision not in ALLOWED: raise ValueError("unknown temporal decision")
        if not isinstance(reason,str) or not reason.strip() or len(reason)>320: raise ValueError("invalid temporal reason")
        counts[decision]+=1; normalized.append({"candidate_id":row["candidate_id"],"temporal_triage":decision,"reason":reason.strip()})
    normalized.sort(key=lambda r:r["candidate_id"]); private={"schema_version":"labelagent.cvat-temporal-assistant-triage.v1","status":"temporal_triage_complete_human_signoff_pending","temporal_pack_manifest_sha256":digest,"decision_counts":{k:counts.get(k,0) for k in sorted(ALLOWED)},"decisions":normalized,"human_truth":False,"human_signoff_count":0,"training_authorized":False,"automatic_approval":False,"scene_transition_excluded":True}; public={k:private[k] for k in ("schema_version","status","temporal_pack_manifest_sha256","decision_counts","human_truth","human_signoff_count","training_authorized","automatic_approval","scene_transition_excluded")}; public.update({"private_candidate_ids_published":False,"private_reasons_published":False}); return private,public
def main():
    p=argparse.ArgumentParser(); p.add_argument("--temporal-pack",required=True,type=Path); p.add_argument("--triage-input",required=True,type=Path); p.add_argument("--private-output",required=True,type=Path); p.add_argument("--receipt-output",required=True,type=Path); a=p.parse_args()
    if a.private_output.exists() or a.receipt_output.exists(): raise ValueError("choose fresh outputs")
    private,public=validate(a.temporal_pack,a.triage_input); a.private_output.parent.mkdir(parents=True,exist_ok=True); a.private_output.write_text(json.dumps(private,ensure_ascii=False,indent=2)+"\n",encoding="utf-8"); public["private_triage_sha256"]=sha(a.private_output); a.receipt_output.parent.mkdir(parents=True,exist_ok=True); a.receipt_output.write_text(json.dumps(public,ensure_ascii=False,indent=2)+"\n",encoding="utf-8"); print(json.dumps(public,ensure_ascii=False)); return 0
if __name__=="__main__": raise SystemExit(main())
