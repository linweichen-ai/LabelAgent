"""Prepare a hash-bound human signoff template for all manual CVAT instances."""
from __future__ import annotations
import argparse,hashlib,json
from pathlib import Path
def sha(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def read(path):
    v=json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(v,dict): raise ValueError("JSON object required")
    return v
def prepare(review_manifest_path,output_path):
    review=read(review_manifest_path)
    if review.get("schema_version")!="labelagent.cvat-manual-instance-review-pack.v1" or review.get("status")!="pending_independent_review": raise ValueError("review manifest identity differs")
    if review.get("test24_intersection_count")!=0 or review.get("training_authorized") is not False: raise ValueError("review manifest isolation boundary differs")
    items=review.get("review_items")
    if not isinstance(items,list) or len(items)!=81: raise ValueError("expected 81 review items")
    ids=[]; rows=[]
    for item in items:
        cid=item.get("candidate_id")
        if not isinstance(cid,str) or cid in ids: raise ValueError("candidate id is invalid or repeated")
        ids.append(cid)
        for key in ("source_image_sha256","canonical_geometry_sha256","focused_image_sha256"):
            if not isinstance(item.get(key),str) or len(item[key])!=64: raise ValueError(f"candidate binding absent: {key}")
        rows.append({"candidate_id":cid,"frame_id":item.get("frame_id"),"source_image_sha256":item["source_image_sha256"],"canonical_geometry_sha256":item["canonical_geometry_sha256"],"focused_image_sha256":item["focused_image_sha256"],"decision":None,"reviewer":None,"reviewed_at":None,"note":None})
    value={"schema_version":"labelagent.cvat-human-instance-signoff.v1","status":"pending_human_review","review_manifest_sha256":sha(review_manifest_path),"candidate_count":81,"allowed_decisions":["pass","reject","uncertain"],"rows":rows,"human_signoff_count":0,"training_authorized":False,"automatic_approval":False,"instructions":["Every row requires an explicit decision, reviewer identity, and reviewed_at timestamp.","Only pass may enter a later training-authorization receipt.","Assistant triage must never populate reviewer or decision fields."]}
    output_path=Path(output_path)
    if output_path.exists(): raise ValueError("choose a fresh output path")
    output_path.parent.mkdir(parents=True,exist_ok=True); output_path.write_text(json.dumps(value,ensure_ascii=False,indent=2)+"\n",encoding="utf-8")
    return value,{"schema_version":"labelagent.cvat-human-instance-signoff-template-receipt.v1","status":"pending_human_review","review_manifest_sha256":value["review_manifest_sha256"],"candidate_count":81,"pending_count":81,"human_signoff_count":0,"training_authorized":False,"automatic_approval":False,"private_template_sha256":sha(output_path),"private_candidate_ids_published":False}
def main():
    p=argparse.ArgumentParser(); p.add_argument("--review-manifest",required=True,type=Path); p.add_argument("--private-template",required=True,type=Path); p.add_argument("--receipt-output",required=True,type=Path); a=p.parse_args()
    if a.receipt_output.exists(): raise ValueError("choose fresh receipt")
    _,receipt=prepare(a.review_manifest,a.private_template); a.receipt_output.parent.mkdir(parents=True,exist_ok=True); a.receipt_output.write_text(json.dumps(receipt,ensure_ascii=False,indent=2)+"\n",encoding="utf-8"); print(json.dumps(receipt,ensure_ascii=False)); return 0
if __name__=="__main__": raise SystemExit(main())
