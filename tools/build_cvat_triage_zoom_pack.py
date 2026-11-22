"""Render raw / mask-only / overlay zooms for assistant-triaged CVAT instances."""
from __future__ import annotations
import argparse, hashlib, json, math
from pathlib import Path
from typing import Any
from PIL import Image, ImageDraw, ImageFont
def sha256(path: Path)->str: return hashlib.sha256(path.read_bytes()).hexdigest()
def read(path: Path)->dict[str,Any]:
    value=json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value,dict): raise ValueError(f"JSON object required: {path}")
    return value
def parts(shape:dict)->list[dict]:
    value=shape.get("parts"); return value if isinstance(value,list) and value else [{"points":shape.get("points") or [],"holes":shape.get("holes") or []}]
def geometry_sha(shape:dict)->str:
    payload={key:shape.get(key,[]) for key in ("points","holes","parts")}; return hashlib.sha256(json.dumps(payload,sort_keys=True,separators=(",",":"),ensure_ascii=False).encode()).hexdigest()
def mask_for(shape:dict,size:tuple[int,int])->Image.Image:
    mask=Image.new("L",size,0); draw=ImageDraw.Draw(mask)
    for part in parts(shape):
        outer=[tuple(map(float,p)) for p in (part.get("points") or [])]
        if len(outer)>=3: draw.polygon(outer,fill=255)
        for hole in part.get("holes") or []:
            ring=[tuple(map(float,p)) for p in hole]
            if len(ring)>=3: draw.polygon(ring,fill=0)
    return mask
def crop_box(shape:dict,size:tuple[int,int])->tuple[int,int,int,int]:
    pts=[p for part in parts(shape) for p in (part.get("points") or [])]
    if len(pts)<3: raise ValueError("shape has too few points")
    xs=[float(p[0]) for p in pts]; ys=[float(p[1]) for p in pts]; span=max(max(xs)-min(xs),max(ys)-min(ys)); pad=max(32,round(span*.22)); w,h=size
    return max(0,math.floor(min(xs))-pad),max(0,math.floor(min(ys))-pad),min(w,math.ceil(max(xs))+pad+1),min(h,math.ceil(max(ys))+pad+1)
def fit(image:Image.Image,size=(420,420))->Image.Image:
    copy=image.copy(); copy.thumbnail(size,Image.Resampling.LANCZOS); canvas=Image.new("RGB",size,(18,27,34)); canvas.paste(copy,((size[0]-copy.width)//2,(size[1]-copy.height)//2)); return canvas
def overlay(image:Image.Image,mask:Image.Image,shape:dict)->Image.Image:
    base=image.convert("RGBA"); green=Image.new("RGBA",base.size,(27,210,139,0)); green.putalpha(mask.point(lambda p:70 if p else 0)); result=Image.alpha_composite(base,green); draw=ImageDraw.Draw(result)
    for part in parts(shape):
        outer=[tuple(map(float,p)) for p in (part.get("points") or [])]
        if len(outer)>=3: draw.line(outer+[outer[0]],fill=(255,145,38,255),width=2)
        for hole in part.get("holes") or []:
            ring=[tuple(map(float,p)) for p in hole]
            if len(ring)>=3: draw.line(ring+[ring[0]],fill=(255,72,72,255),width=1)
    return result.convert("RGB")
def build(candidates_path:Path,review_pack_path:Path,triage_path:Path,triage_receipt_path:Path,image_root:Path,output_dir:Path)->tuple[dict,dict]:
    candidates,review,triage,receipt=map(read,(candidates_path,review_pack_path,triage_path,triage_receipt_path)); output_dir=output_dir.resolve(); image_root=image_root.resolve()
    if output_dir.exists(): raise ValueError("choose a fresh output directory")
    if receipt.get("private_triage_sha256")!=sha256(triage_path): raise ValueError("triage hash differs from receipt")
    if triage.get("review_pack_manifest_sha256")!=sha256(review_pack_path): raise ValueError("triage binds a different review pack")
    if triage.get("human_truth") is not False or triage.get("training_authorized") is not False: raise ValueError("triage crossed the truth boundary")
    by_id={row.get("candidate_id"):row for row in candidates.get("candidates") or [] if isinstance(row,dict)}
    selected=[row for row in triage.get("decisions") or [] if isinstance(row,dict) and row.get("assistant_triage")=="needs_human_review"]
    if len(selected)!=8: raise ValueError("expected exactly eight human-review candidates")
    output_dir.mkdir(parents=True); item_rows=[]; cards=[]
    for index,row in enumerate(selected,1):
        cid=row["candidate_id"]; candidate=by_id.get(cid)
        if candidate is None: raise ValueError(f"unknown candidate: {cid}")
        shape=candidate.get("shape"); meta=candidate.get("image")
        if not isinstance(shape,dict) or not isinstance(meta,dict) or shape.get("label")!="pallet" or (shape.get("cvat_metadata") or {}).get("source")!="manual": raise ValueError(f"candidate provenance differs: {cid}")
        filename=meta.get("file")
        if not isinstance(filename,str) or Path(filename).name!=filename: raise ValueError("unsafe image filename")
        image_path=image_root/filename
        if not image_path.is_file() or sha256(image_path)!=meta.get("sha256"): raise ValueError(f"image hash differs: {cid}")
        with Image.open(image_path) as opened: image=opened.convert("RGB")
        mask=mask_for(shape,image.size); box=crop_box(shape,image.size); raw=fit(image.crop(box)); mask_rgb=fit(Image.merge("RGB",(mask,mask,mask)).crop(box)); over=fit(overlay(image,mask,shape).crop(box)); card=Image.new("RGB",(1260,462),"white"); card.paste(raw,(0,42)); card.paste(mask_rgb,(420,42)); card.paste(over,(840,42)); draw=ImageDraw.Draw(card); draw.text((8,8),f"{cid} | RAW",fill="black",font=ImageFont.load_default()); draw.text((428,8),"MASK ONLY",fill="black",font=ImageFont.load_default()); draw.text((848,8),"OVERLAY | assistant triage only",fill="black",font=ImageFont.load_default()); draw.text((8,25),row.get("reason") or "",fill="black",font=ImageFont.load_default()); path=output_dir/f"zoom_{index:02d}.jpg"; card.save(path,quality=94,optimize=True); cards.append((cid,path,card)); item_rows.append({"candidate_id":cid,"source_image_sha256":meta["sha256"],"canonical_geometry_sha256":geometry_sha(shape),"zoom_file":path.name,"zoom_sha256":sha256(path)})
    pages=[]
    for offset in range(0,len(cards),2):
        batch=cards[offset:offset+2]; page=Image.new("RGB",(1260,462*len(batch)),(235,240,242))
        for i,(_,_,card) in enumerate(batch): page.paste(card,(0,i*462))
        path=output_dir/f"zoom_page_{offset//2+1:02d}.jpg"; page.save(path,quality=92,optimize=True); pages.append({"file":path.name,"sha256":sha256(path),"candidate_ids":[x[0] for x in batch]})
    manifest={"schema_version":"labelagent.cvat-triage-zoom-pack.v1","status":"human_review_pending","candidate_manifest_sha256":sha256(candidates_path),"review_pack_manifest_sha256":sha256(review_pack_path),"assistant_triage_sha256":sha256(triage_path),"item_count":8,"page_count":len(pages),"items":item_rows,"pages":pages,"human_truth":False,"human_signoff_count":0,"training_authorized":False,"automatic_approval":False}; mp=output_dir/"manifest.json"; mp.write_text(json.dumps(manifest,ensure_ascii=False,indent=2)+"\n",encoding="utf-8")
    public={key:manifest[key] for key in ("schema_version","status","candidate_manifest_sha256","review_pack_manifest_sha256","assistant_triage_sha256","item_count","page_count","human_truth","human_signoff_count","training_authorized","automatic_approval")}; public.update({"private_manifest_sha256":sha256(mp),"private_candidate_ids_published":False,"private_review_images_published":False}); return manifest,public
def main()->int:
    p=argparse.ArgumentParser(description=__doc__); [p.add_argument(name,required=True,type=Path) for name in ("--candidates","--review-pack","--triage","--triage-receipt","--image-root","--output-dir","--receipt-output")]; a=p.parse_args()
    if a.receipt_output.exists(): raise ValueError("choose a fresh receipt output")
    _,receipt=build(a.candidates,a.review_pack,a.triage,a.triage_receipt,a.image_root,a.output_dir); a.receipt_output.parent.mkdir(parents=True,exist_ok=True); a.receipt_output.write_text(json.dumps(receipt,ensure_ascii=False,indent=2)+"\n",encoding="utf-8"); print(json.dumps(receipt,ensure_ascii=False)); return 0
if __name__=="__main__": raise SystemExit(main())
