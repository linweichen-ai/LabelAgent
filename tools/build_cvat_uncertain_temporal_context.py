"""Build same-parity temporal context for uncertain CVAT pallet instances."""
from __future__ import annotations
import argparse,hashlib,json,math
from pathlib import Path
from typing import Any
from PIL import Image,ImageDraw,ImageFont
OFFSETS=(-4,-2,0,2,4)
def sha(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def read(path):
    v=json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(v,dict): raise ValueError("JSON object required")
    return v
def shape_points(shape):
    parts=shape.get("parts") if isinstance(shape.get("parts"),list) and shape.get("parts") else [shape]
    return [p for part in parts for p in (part.get("points") or [])]
def crop_box(shape,size):
    pts=shape_points(shape); xs=[float(p[0]) for p in pts]; ys=[float(p[1]) for p in pts]; span=max(max(xs)-min(xs),max(ys)-min(ys)); pad=max(36,round(span*.25)); w,h=size
    return max(0,math.floor(min(xs))-pad),max(0,math.floor(min(ys))-pad),min(w,math.ceil(max(xs))+pad+1),min(h,math.ceil(max(ys))+pad+1)
def fit(image,size=(360,300)):
    copy=image.copy(); copy.thumbnail(size,Image.Resampling.LANCZOS); canvas=Image.new("RGB",size,(18,27,34)); canvas.paste(copy,((size[0]-copy.width)//2,(size[1]-copy.height)//2)); return canvas
def outline(image,shape):
    result=image.copy(); draw=ImageDraw.Draw(result); parts=shape.get("parts") if isinstance(shape.get("parts"),list) and shape.get("parts") else [shape]
    for part in parts:
        pts=[tuple(map(float,p)) for p in (part.get("points") or [])]
        if len(pts)>=3: draw.line(pts+[pts[0]],fill=(255,145,38),width=2)
    return result
def build(candidates_path,zoom_triage_path,zoom_receipt_path,import_manifest_path,image_root,output_dir):
    candidates,triage,receipt,imports=map(read,(candidates_path,zoom_triage_path,zoom_receipt_path,import_manifest_path)); output_dir=Path(output_dir).resolve(); image_root=Path(image_root).resolve()
    if output_dir.exists(): raise ValueError("choose a fresh output directory")
    if receipt.get("private_triage_sha256")!=sha(zoom_triage_path): raise ValueError("zoom triage hash differs from receipt")
    selected=[r for r in triage.get("decisions") or [] if isinstance(r,dict) and r.get("zoom_triage")=="uncertain_needs_human"]
    if len(selected)!=3: raise ValueError("expected exactly three uncertain candidates")
    by_id={r.get("candidate_id"):r for r in candidates.get("candidates") or [] if isinstance(r,dict)}; frame_rows={r.get("frame"):r for r in imports.get("frames") or [] if isinstance(r,dict)}
    output_dir.mkdir(parents=True); items=[]
    for selected_index,row in enumerate(selected,1):
        cid=row["candidate_id"]; candidate=by_id.get(cid)
        if candidate is None: raise ValueError(f"unknown candidate: {cid}")
        frame=candidate.get("frame_id"); shape=candidate.get("shape")
        if not isinstance(frame,int) or not isinstance(shape,dict): raise ValueError("candidate binding is incomplete")
        target_row=frame_rows.get(frame)
        if target_row is None: raise ValueError("target frame absent from import manifest")
        target_path=image_root/target_row["file"]
        if not target_path.is_file() or sha(target_path)!=target_row.get("sha256"): raise ValueError("target image hash differs")
        with Image.open(target_path) as opened: target=opened.convert("RGB")
        box=crop_box(shape,target.size); panels=[]; evidence=[]
        for offset in OFFSETS:
            neighbor=frame+offset; manifest_row=frame_rows.get(neighbor)
            if manifest_row is None: raise ValueError(f"neighbor frame absent: {neighbor}")
            path=image_root/manifest_row["file"]
            if not path.is_file() or sha(path)!=manifest_row.get("sha256"): raise ValueError(f"neighbor image hash differs: {neighbor}")
            with Image.open(path) as opened: image=opened.convert("RGB")
            if image.size!=target.size: raise ValueError("neighbor image size differs")
            rendered=outline(image,shape) if offset==0 else image; panel=fit(rendered.crop(box)); canvas=Image.new("RGB",(360,326),"white"); canvas.paste(panel,(0,26)); ImageDraw.Draw(canvas).text((8,8),f"offset={offset:+d} frame={neighbor}"+(" TARGET" if offset==0 else ""),fill="black",font=ImageFont.load_default()); panels.append(canvas); evidence.append({"offset":offset,"frame":neighbor,"image_sha256":manifest_row["sha256"]})
        sheet=Image.new("RGB",(360*5,326),(235,240,242))
        for i,panel in enumerate(panels): sheet.paste(panel,(i*360,0))
        path=output_dir/f"temporal_{selected_index:02d}.jpg"; sheet.save(path,quality=93,optimize=True); items.append({"candidate_id":cid,"target_frame":frame,"evidence":evidence,"sheet_file":path.name,"sheet_sha256":sha(path)})
    manifest={"schema_version":"labelagent.cvat-uncertain-temporal-context.v1","status":"assistant_review_pending","candidate_manifest_sha256":sha(candidates_path),"zoom_triage_sha256":sha(zoom_triage_path),"import_manifest_sha256":sha(import_manifest_path),"offsets":list(OFFSETS),"same_parity_only":True,"item_count":3,"items":items,"human_truth":False,"human_signoff_count":0,"training_authorized":False,"automatic_approval":False}; mp=output_dir/"manifest.json"; mp.write_text(json.dumps(manifest,ensure_ascii=False,indent=2)+"\n",encoding="utf-8"); public={k:manifest[k] for k in ("schema_version","status","candidate_manifest_sha256","zoom_triage_sha256","import_manifest_sha256","offsets","same_parity_only","item_count","human_truth","human_signoff_count","training_authorized","automatic_approval")}; public.update({"private_manifest_sha256":sha(mp),"private_candidate_ids_published":False,"private_images_published":False}); return manifest,public
def main():
    p=argparse.ArgumentParser(); [p.add_argument(n,required=True,type=Path) for n in ("--candidates","--zoom-triage","--zoom-receipt","--import-manifest","--image-root","--output-dir","--receipt-output")]; a=p.parse_args()
    if a.receipt_output.exists(): raise ValueError("choose fresh receipt")
    _,receipt=build(a.candidates,a.zoom_triage,a.zoom_receipt,a.import_manifest,a.image_root,a.output_dir); a.receipt_output.parent.mkdir(parents=True,exist_ok=True); a.receipt_output.write_text(json.dumps(receipt,ensure_ascii=False,indent=2)+"\n",encoding="utf-8"); print(json.dumps(receipt,ensure_ascii=False)); return 0
if __name__=="__main__": raise SystemExit(main())
