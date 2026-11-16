"""Build real, pending-review examples from the existing tuning cohort only."""
import argparse
import hashlib
import json
import math
from pathlib import Path
from PIL import Image, ImageDraw
from data_tools import sha256


def all_points(c):
    return [p for part in c.get("parts", [c]) for p in part.get("points", [])]


def bounds(c):
    p = all_points(c)
    return min(x for x,y in p), min(y for x,y in p), max(x for x,y in p), max(y for x,y in p)


def area(c):
    x1,y1,x2,y2 = bounds(c)
    return (x2-x1)*(y2-y1)


def paint(image, candidate, offset=(0, 0)):
    image = image.copy(); draw = ImageDraw.Draw(image)
    for part in candidate.get("parts", [candidate]):
        p = [(x-offset[0], y-offset[1]) for x,y in part.get("points", [])]
        if len(p) >= 3:
            draw.line(p + [p[0]], fill="#ffda00", width=4)
    return image


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--dataset", required=True); ap.add_argument("--out", required=True)
    a = ap.parse_args(); source = Path(a.dataset); out = Path(a.out)
    if out.exists(): raise ValueError("Fresh output required")
    manifest = json.loads((source / "import_manifest.json").read_text(encoding="utf-8-sig"))
    baseline_path = source / "sam3_baseline/candidates_by_topic.json"
    candidates = json.loads(baseline_path.read_text(encoding="utf-8-sig"))
    rows=[]; image_receipts=[]; items=[]
    for record in manifest["records"]:
        frame = record["frame"]
        if frame not in [15,45,76,105,135] or not record['topic'].endswith('/image'): continue
        topic = record["topic"]; cam=topic.split("/")[-2]
        path = source / Path(record["source_file"].replace("\\", "/"))
        assert sha256(path) == record["sha256"], "Source image changed from import receipt"
        image=Image.open(path).convert("RGB")
        options=[c for c in candidates[topic][str(frame)] if len(all_points(c)) >= 3]
        pallets=sorted([c for c in options if c["label"]=="pallet"], key=area)
        people=[c for c in options if c["label"]=="person"]
        proposed = pallets[:1] + people[:1] + sorted(pallets, key=lambda c:c.get("score",0), reverse=True)[:1] + list(reversed(pallets)) + options
        selected=[]; seen=set()
        for c in proposed:
            if c["id"] not in seen:
                seen.add(c["id"]); selected.append(c)
        items.append((record,path,selected))
    assert len(items)==20 and sum(len(x[2]) for x in items)>=60
    budgets=[0]*len(items)
    while sum(budgets)<60:
        for i,item in enumerate(items):
            if budgets[i]<len(item[2]):budgets[i]+=1
            if sum(budgets)==60:break
    (out / "images").mkdir(parents=True)
    for (record,path,selected),budget in zip(items,budgets):
        frame=record['frame'];topic=record['topic'];cam=topic.split('/')[-2]
        image=Image.open(path).convert('RGB')
        selected=selected[:budget]
        for i,c in enumerate(selected):
            sid=f"f{frame:03d}_{cam}_{i}"
            marked=paint(image,c); marked.thumbnail((1440,1152))
            x1,y1,x2,y2=bounds(c); pad=max(48,int(max(x2-x1,y2-y1)*.45))
            rect=(max(0,int(x1-pad)),max(0,int(y1-pad)),min(image.width,int(x2+pad+1)),min(image.height,int(y2+pad+1)))
            crop=paint(image.crop(rect),c,rect[:2]); crop.thumbnail((896,896))
            names=[f"images/{sid}_scene.jpg",f"images/{sid}_crop.jpg"]
            marked.save(out/names[0],quality=94);crop.save(out/names[1],quality=94)
            rows.append({"sample_id":sid,"capture_id":"Capture_20261103","group_id":f"Capture_20261103:block_{frame//30:03d}",
                "frame":frame,"camera":cam,"timestamp_ns":record["timestamp_ns"],"source_image_sha256":record["sha256"],
                "source_kind":"model_candidate_unconfirmed","candidate_id":c["id"],"proposed_label":c["label"],
                "proposal_score":c.get("score"),"images":names,"image_sha256":{n:sha256(out/n) for n in names},
                "review_status":"pending","reviewer":"","reviewed_at":None,"decision":None,"correct_label":None,
                "notes":"Review candidate category only; its polygon is NOT human-approved segmentation truth."})
        image_receipts.append({"frame":frame,"camera":cam,"sha256":record["sha256"]})
    assert len(rows)==60 and len(image_receipts)==20
    (out/'review_queue.jsonl').write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in rows),encoding='utf-8')
    receipt={"status":"pending_human_review_not_training_labels","images":20,"cases":60,"groups":5,"human_reviewed":0,
        "cohort":"original 20 tuning images only; no new validation cases accessed", "source_baseline_sha256":sha256(baseline_path),
        "source_image_receipts":image_receipts,"case_selection":"round-robin 60 unique candidates across 20 images; small pallet, person if present, high-score pallet, unique fallback",
        "independent_test_available":False,"reference_3d_files_opened":0,"new_model_requests":0,"raw_data_modified":False}
    (out/'data_receipt.json').write_text(json.dumps(receipt,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps({k:v for k,v in receipt.items() if k!='source_image_receipts'},ensure_ascii=False))


if __name__=='__main__':main()
