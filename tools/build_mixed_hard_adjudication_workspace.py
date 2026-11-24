"""Build and validate an offline, hash-bound review UI for the mixed hard set."""
import argparse
import hashlib
import html
import json
import shutil
from pathlib import Path


MANIFEST_SCHEMA = "labelagent.mixed-hard-review-pack.v1"
TEMPLATE_SCHEMA = "labelagent.mixed-hard-adjudication-template.v2"
SUBMISSION_SCHEMA = "labelagent.mixed-hard-adjudication.v2"
LEGACY_SUBMISSION_SCHEMA = "labelagent.mixed-hard-adjudication.v1"
LABELS = ("车辆", "pallet", "person", "floor", "slope", "ego",
          "other_robot", "cargo_floor")


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _safe_child(root, relative):
    root = Path(root).resolve()
    path = (root / relative).resolve()
    if root not in path.parents:
        raise ValueError(f"path escapes review packet: {relative}")
    return path


def validate_packet(packet_dir):
    root = Path(packet_dir).resolve()
    manifest_path, receipt_path = root / "manifest.json", root / "receipt.json"
    manifest, receipt = read_json(manifest_path), read_json(receipt_path)
    if manifest.get("schema_version") != MANIFEST_SCHEMA:
        raise ValueError("unsupported review manifest schema")
    for key, expected in {
        "candidate_only": True, "human_truth": False,
        "approved_for_training": False, "training_authorized": False,
        "test24_used": False, "production_deployed": False,
    }.items():
        if manifest.get(key) is not expected:
            raise ValueError(f"unsafe manifest boundary: {key}")
    rows = manifest.get("rows")
    if not isinstance(rows, list) or not 24 <= len(rows) <= 40:
        raise ValueError("review packet must contain 24-40 rows")
    if manifest.get("summary", {}).get("test24_intersection_count") != 0:
        raise ValueError("review packet intersects test24")
    if receipt.get("manifest_sha256") != sha256(manifest_path):
        raise ValueError("manifest receipt mismatch")
    receipt_files = {item["path"]: item for item in receipt.get("files", [])}
    sample_ids, image_hashes = set(), set()
    for row in rows:
        sample_id, image_hash = row.get("sample_id"), row.get("source_image_sha256")
        if not isinstance(sample_id, str) or not sample_id or sample_id in sample_ids:
            raise ValueError("invalid or duplicate sample id")
        if not isinstance(image_hash, str) or len(image_hash) != 64 or image_hash in image_hashes:
            raise ValueError("invalid or duplicate source image hash")
        sample_ids.add(sample_id)
        image_hashes.add(image_hash)
        for key in ("raw_image", "review_image"):
            relative = row.get(key)
            path = _safe_child(root, relative)
            item = receipt_files.get(relative)
            if not path.is_file() or not item or item.get("sha256") != sha256(path):
                raise ValueError(f"missing or mismatched receipt: {relative}")
        if (row.get("candidate_only") is not True
                or row.get("human_truth") is not False
                or row.get("approved_for_training") is not False):
            raise ValueError(f"unsafe row boundary: {sample_id}")
    return manifest, sha256(manifest_path)


def template_for(manifest, manifest_sha256):
    return {
        "schema_version": TEMPLATE_SCHEMA,
        "status": "pending_declared_human_review",
        "review_manifest_sha256": manifest_sha256,
        "sample_count": len(manifest["rows"]),
        "rows": [{
            "sample_id": row["sample_id"],
            "source_image_sha256": row["source_image_sha256"],
            "cohort": row["cohort"],
            "candidate_decision": None,
            "slope_presence": None,
            "class_conflict": None,
            "pallet_quality": None,
            "class_presence": {label: None for label in LABELS},
            "pallet_instance_count": None,
            "difficulty_flags": [],
            "note": "",
        } for row in manifest["rows"]],
        "reviewer_declared_id": None,
        "reviewed_at": None,
        "candidate_only": True,
        "human_truth": False,
        "approved_for_training": False,
        "training_authorized": False,
        "automatic_approval": False,
    }


def validate_submission(manifest, manifest_sha256, submission):
    schema = submission.get("schema_version")
    if schema not in {SUBMISSION_SCHEMA, LEGACY_SUBMISSION_SCHEMA}:
        raise ValueError("unsupported adjudication schema")
    if submission.get("review_manifest_sha256") != manifest_sha256:
        raise ValueError("adjudication is bound to another manifest")
    for key in ("candidate_only",):
        if submission.get(key) is not True:
            raise ValueError(f"unsafe adjudication boundary: {key}")
    for key in ("human_truth", "approved_for_training", "training_authorized",
                "automatic_approval"):
        if submission.get(key) is not False:
            raise ValueError(f"unsafe adjudication boundary: {key}")
    if not str(submission.get("reviewer_declared_id") or "").strip():
        raise ValueError("reviewer_declared_id is required")
    expected = {row["sample_id"]: row for row in manifest["rows"]}
    rows = submission.get("rows")
    if not isinstance(rows, list) or len(rows) != len(expected):
        raise ValueError("adjudication row count differs")
    allowed = {
        "candidate_decision": {"accept_candidate", "correct_candidate", "reject_candidate", "uncertain"},
        "slope_presence": {"present", "absent", "not_applicable", "uncertain"},
        "class_conflict": {"none", "vehicle_other_robot", "ego_other_robot",
                           "pallet_cargo_floor", "other", "uncertain"},
        "pallet_quality": {"complete", "hole_error", "merged_instances", "missed_instance",
                           "not_applicable", "uncertain"},
    }
    seen = set()
    benchmark_ready = schema == SUBMISSION_SCHEMA
    for row in rows:
        sample_id = row.get("sample_id")
        if sample_id in seen or sample_id not in expected:
            raise ValueError("unknown or duplicate adjudication sample")
        seen.add(sample_id)
        if row.get("source_image_sha256") != expected[sample_id]["source_image_sha256"]:
            raise ValueError(f"source image binding differs: {sample_id}")
        for key, values in allowed.items():
            if row.get(key) not in values:
                raise ValueError(f"invalid {key}: {sample_id}")
        if schema == SUBMISSION_SCHEMA:
            presence = row.get("class_presence")
            if (not isinstance(presence, dict) or set(presence) != set(LABELS)
                    or any(value not in {"present", "absent", "uncertain"}
                           for value in presence.values())):
                raise ValueError(f"invalid class_presence: {sample_id}")
            count = row.get("pallet_instance_count")
            if type(count) is not int or count < 0:
                raise ValueError(f"invalid pallet_instance_count: {sample_id}")
            if presence["pallet"] == "absent" and count != 0:
                raise ValueError(f"pallet count disagrees with absence: {sample_id}")
            if presence["pallet"] == "present" and count == 0:
                raise ValueError(f"pallet count disagrees with presence: {sample_id}")
            if "uncertain" in presence.values():
                benchmark_ready = False
        else:
            benchmark_ready = False
    return {"status": "valid_complete_review_record", "sample_count": len(rows),
            "benchmark_reference_ready": benchmark_ready,
            "human_truth": False, "approved_for_training": False}


def write_html(manifest, manifest_sha256, destination):
    items = [{
        "sample_id": row["sample_id"], "cohort": row["cohort"],
        "source_image_sha256": row["source_image_sha256"],
        "review_image": row["review_image"], "raw_image": row["raw_image"],
        "review_note": row.get("review_note") or "",
    } for row in manifest["rows"]]
    payload = html.escape(json.dumps(items, ensure_ascii=False))
    destination.write_text(f'''<!doctype html><html lang="zh-CN"><head><meta charset="utf-8">
<title>LabelAgent 困难帧受控复核</title><style>
body{{font:16px system-ui;margin:0;background:#10151b;color:#eef3f8}}header,main{{max-width:1180px;margin:auto;padding:18px}}
header{{position:sticky;top:0;background:#17202a;border-bottom:1px solid #39495a;z-index:2}}button,select,input,textarea{{font:inherit}}
.row{{display:grid;grid-template-columns:1fr 1fr;gap:18px}}img{{width:100%;border-radius:10px;background:#000}}
.panel{{background:#19232d;padding:16px;border-radius:12px}}label{{display:block;margin:10px 0}}select,textarea,input{{width:100%;box-sizing:border-box;padding:8px;background:#0f171f;color:#fff;border:1px solid #52677b;border-radius:6px}}
.flags label{{display:inline-block;margin-right:14px}}.flags input{{width:auto}}button{{padding:9px 15px;margin-right:8px}}.warn{{color:#ffd27a}}
</style></head><body><header><b>困难帧受控复核</b>　<span id="counter"></span><p class="warn">复核记录不是像素真值，不自动授权训练。</p></header>
<main><div class="row"><div class="panel"><h2 id="title"></h2><p id="meta"></p><img id="image"><p><a id="raw" target="_blank">打开原图</a></p></div>
<div class="panel"><label>复核人声明 ID<input id="reviewer"></label>
<label>候选结论<select id="candidate"><option value="">请选择</option><option>accept_candidate</option><option>correct_candidate</option><option>reject_candidate</option><option>uncertain</option></select></label>
<label>斜坡存在性<select id="slope"><option value="">请选择</option><option>present</option><option>absent</option><option>not_applicable</option><option>uncertain</option></select></label>
<label>类别冲突<select id="conflict"><option value="">请选择</option><option>none</option><option>vehicle_other_robot</option><option>ego_other_robot</option><option>pallet_cargo_floor</option><option>other</option><option>uncertain</option></select></label>
<label>托盘质量<select id="pallet"><option value="">请选择</option><option>complete</option><option>hole_error</option><option>merged_instances</option><option>missed_instance</option><option>not_applicable</option><option>uncertain</option></select></label>
<fieldset><legend>八类存在性（严格评测必填）</legend><div id="presence"></div></fieldset>
<label>可见托盘实例数<input id="palletCount" type="number" min="0" step="1" value="0"></label>
<div class="flags">困难因素：<label><input type="checkbox" value="edge">边缘</label><label><input type="checkbox" value="small">小目标</label><label><input type="checkbox" value="occluded">遮挡</label><label><input type="checkbox" value="material_or_light">材质/光照</label></div>
<label>备注<textarea id="note" rows="4"></textarea></label><button id="prev">上一张</button><button id="next">保存并下一张</button><button id="export">导出完整复核记录</button></div></div></main>
<script id="items" type="application/json">{payload}</script><script>
const items=JSON.parse(document.getElementById('items').textContent);const labels={json.dumps(list(LABELS), ensure_ascii=False)};const key='labelagent-mixed-hard-v2-{manifest_sha256}';let state=JSON.parse(localStorage.getItem(key)||'null')||{{index:0,reviewer:'',rows:{{}}}};
const $=id=>document.getElementById(id), selects=['candidate','slope','conflict','pallet'];
function presenceValues(){{return Object.fromEntries(labels.map(label=>[label,$('presence-'+labels.indexOf(label)).value]))}}
function save(){{const item=items[state.index];state.reviewer=$('reviewer').value.trim();state.rows[item.sample_id]={{sample_id:item.sample_id,source_image_sha256:item.source_image_sha256,candidate_decision:$('candidate').value,slope_presence:$('slope').value,class_conflict:$('conflict').value,pallet_quality:$('pallet').value,class_presence:presenceValues(),pallet_instance_count:Number($('palletCount').value),difficulty_flags:[...document.querySelectorAll('.flags input:checked')].map(x=>x.value),note:$('note').value}};localStorage.setItem(key,JSON.stringify(state))}}
function render(){{const item=items[state.index],row=state.rows[item.sample_id]||{{}};$('counter').textContent=`${{state.index+1}} / ${{items.length}}`;$('title').textContent=item.sample_id;$('meta').textContent=item.cohort+' · '+item.review_note;$('image').src=item.review_image;$('raw').href=item.raw_image;$('reviewer').value=state.reviewer||'';const values={{candidate:row.candidate_decision,slope:row.slope_presence,conflict:row.class_conflict,pallet:row.pallet_quality}};selects.forEach(x=>$(x).value=values[x]||'');labels.forEach((label,index)=>$('presence-'+index).value=(row.class_presence||{{}})[label]||'');$('palletCount').value=Number.isInteger(row.pallet_instance_count)?row.pallet_instance_count:0;document.querySelectorAll('.flags input').forEach(x=>x.checked=(row.difficulty_flags||[]).includes(x.value));$('note').value=row.note||''}}
$('prev').onclick=()=>{{save();state.index=Math.max(0,state.index-1);render()}};$('next').onclick=()=>{{save();state.index=Math.min(items.length-1,state.index+1);render()}};
$('export').onclick=()=>{{save();const rows=items.map(x=>state.rows[x.sample_id]).filter(Boolean);const missing=rows.some(r=>!r.candidate_decision||!r.slope_presence||!r.class_conflict||!r.pallet_quality||!r.class_presence||labels.some(label=>!r.class_presence[label])||!Number.isInteger(r.pallet_instance_count)||r.pallet_instance_count<0);if(!state.reviewer||rows.length!==items.length||missing)return alert('请填写复核人、全部类别存在性和托盘实例数');const disagreement=rows.some(r=>(r.class_presence.pallet==='absent'&&r.pallet_instance_count!==0)||(r.class_presence.pallet==='present'&&r.pallet_instance_count===0));if(disagreement)return alert('托盘存在性与实例数不一致');const out={{schema_version:'{SUBMISSION_SCHEMA}',status:'declared_human_review_complete',review_manifest_sha256:'{manifest_sha256}',reviewer_declared_id:state.reviewer,reviewed_at:new Date().toISOString(),rows,candidate_only:true,human_truth:false,approved_for_training:false,training_authorized:false,automatic_approval:false}};const a=document.createElement('a');a.href=URL.createObjectURL(new Blob([JSON.stringify(out,null,2)+'\\n'],{{type:'application/json'}}));a.download='mixed-hard-adjudication.json';a.click();URL.revokeObjectURL(a.href)}};
$('presence').innerHTML=labels.map((label,index)=>`<label>${{label}}<select id="presence-${{index}}"><option value="">请选择</option><option>present</option><option>absent</option><option>uncertain</option></select></label>`).join('');render();
</script></body></html>''', encoding="utf-8")


def copy_review_assets(manifest, packet_dir, output_dir):
    """Make the browser review workspace portable without changing its sources."""
    packet = Path(packet_dir).resolve()
    output = Path(output_dir).resolve()
    if packet == output:
        raise ValueError("review workspace must not overwrite its source packet")
    copied = {}
    for row in manifest["rows"]:
        for key in ("raw_image", "review_image"):
            relative = row[key]
            source = _safe_child(packet, relative)
            destination = _safe_child(output, relative)
            destination.parent.mkdir(parents=True, exist_ok=True)
            expected = sha256(source)
            if destination.is_file() and sha256(destination) == expected:
                copied[relative] = expected
                continue
            shutil.copy2(source, destination)
            actual = sha256(destination)
            if actual != expected:
                raise ValueError(f"copied review asset hash mismatch: {relative}")
            copied[relative] = actual
    return copied


def build_workspace(packet_dir, output_dir):
    manifest, digest = validate_packet(packet_dir)
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    copied_assets = copy_review_assets(manifest, packet_dir, output)
    template = template_for(manifest, digest)
    template_path = output / "adjudication-template.json"
    template_path.write_text(json.dumps(template, ensure_ascii=False, indent=2) + "\n",
                             encoding="utf-8")
    write_html(manifest, digest, output / "reviewer.html")
    files = {name: sha256(output / name) for name in
             ("adjudication-template.json", "reviewer.html")}
    files.update(copied_assets)
    receipt = {"schema_version": "labelagent.mixed-hard-adjudication-workspace-receipt.v2",
               "review_manifest_sha256": digest,
               "files": dict(sorted(files.items())),
               "copied_image_count": len(copied_assets),
               "candidate_only": True, "human_truth": False,
               "approved_for_training": False, "automatic_approval": False}
    (output / "receipt.json").write_text(
        json.dumps(receipt, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return receipt


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--packet", required=True)
    parser.add_argument("--output")
    parser.add_argument("--validate-submission")
    args = parser.parse_args()
    manifest, digest = validate_packet(args.packet)
    if args.validate_submission:
        print(json.dumps(validate_submission(
            manifest, digest, read_json(args.validate_submission)), ensure_ascii=False))
    else:
        if not args.output:
            parser.error("--output is required when building a workspace")
        print(json.dumps(build_workspace(args.packet, args.output), ensure_ascii=False))


if __name__ == "__main__":
    main()
