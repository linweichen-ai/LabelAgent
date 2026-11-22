"""Build a private, offline HTML reviewer for a hash-bound human signoff."""
from __future__ import annotations

import argparse
import base64
import hashlib
import json
from io import BytesIO
import shutil
import sys
import tempfile
from pathlib import Path, PurePosixPath
import re
from typing import Any

from PIL import Image


REVIEW_SCHEMA = "labelagent.cvat-manual-instance-review-pack.v1"
TEMPLATE_SCHEMA = "labelagent.cvat-human-instance-signoff.v1"
EXPECTED_COUNT = 81
BINDING_FIELDS = (
    "candidate_id",
    "frame_id",
    "source_image_sha256",
    "canonical_geometry_sha256",
    "focused_image_sha256",
)
SHA256 = re.compile(r"^[0-9a-fA-F]{64}$")
DECISIONS = ("pass", "reject", "uncertain")
MAX_JSON_BYTES = 4 * 1024 * 1024
MAX_IMAGE_BYTES = 8 * 1024 * 1024
MAX_TOTAL_IMAGE_BYTES = 128 * 1024 * 1024


def _read_json(path: Path, label: str) -> tuple[dict[str, Any], bytes]:
    try:
        with path.open("rb") as handle:
            raw = handle.read(MAX_JSON_BYTES + 1)
        if len(raw) > MAX_JSON_BYTES:
            raise ValueError(f"{label} exceeds the size limit")
        value = json.loads(raw.decode("utf-8"), object_pairs_hook=_unique_object)
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError(f"{label} is not valid UTF-8 JSON") from error
    if not isinstance(value, dict):
        raise ValueError(f"{label} must be an object")
    return value, raw


def _read_image(path: Path, index: int) -> bytes:
    try:
        with path.open("rb") as handle:
            raw = handle.read(MAX_IMAGE_BYTES + 1)
    except OSError:
        raise ValueError(f"review item {index} focused image cannot be read") from None
    if len(raw) > MAX_IMAGE_BYTES:
        raise ValueError(f"review item {index} focused image exceeds the size limit")
    return raw


def _unique_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("input JSON has duplicate keys")
        result[key] = value
    return result


def _sha256_bytes(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def _is_hash(value: Any) -> bool:
    return isinstance(value, str) and SHA256.fullmatch(value) is not None


def _safe_image_path(review_root: Path, relative: Any) -> Path:
    if not isinstance(relative, str) or not relative or "\\" in relative:
        raise ValueError("focused image path is invalid")
    rel = PurePosixPath(relative)
    if rel.is_absolute() or any(part in ("", ".", "..") for part in rel.parts):
        raise ValueError("focused image path escapes the review root")
    root = review_root.resolve(strict=True)
    if ":" in relative:
        raise ValueError("focused image path escapes the review root")
    try:
        candidate = (root / Path(*rel.parts)).resolve(strict=True)
    except OSError:
        raise ValueError("focused image is missing") from None
    if not candidate.is_relative_to(root) or not candidate.is_file():
        raise ValueError("focused image path escapes the review root")
    return candidate


def _validate_inputs(
    review_manifest: Path, signoff_template: Path, review_root: Path,
    expected_review_sha256: str, expected_template_sha256: str,
) -> tuple[list[dict[str, Any]], str, str, list[str]]:
    review_manifest = review_manifest.resolve(strict=True)
    signoff_template = signoff_template.resolve(strict=True)
    review_root = review_root.resolve(strict=True)
    review, review_raw = _read_json(review_manifest, "review manifest")
    template, template_raw = _read_json(signoff_template, "signoff template")
    if review.get("schema_version") != REVIEW_SCHEMA or review.get("status") != "pending_independent_review":
        raise ValueError("review manifest identity differs")
    if review.get("test24_intersection_count") != 0 or review.get("training_authorized") is not False:
        raise ValueError("review manifest isolation boundary differs")
    if template.get("schema_version") != TEMPLATE_SCHEMA or template.get("status") != "pending_human_review":
        raise ValueError("signoff template identity differs")
    if template.get("candidate_count") != EXPECTED_COUNT or template.get("allowed_decisions") != list(DECISIONS):
        raise ValueError("signoff template identity differs")
    if template.get("human_signoff_count") != 0:
        raise ValueError("signoff template already contains human signoff")
    if template.get("training_authorized") is not False or template.get("automatic_approval") is not False:
        raise ValueError("signoff template crosses the authorization boundary")
    review_items = review.get("review_items")
    rows = template.get("rows")
    if review.get("candidate_count") != EXPECTED_COUNT or not isinstance(review_items, list) or len(review_items) != EXPECTED_COUNT:
        raise ValueError("review manifest must contain exactly 81 items")
    if template.get("candidate_count") != EXPECTED_COUNT or not isinstance(rows, list) or len(rows) != EXPECTED_COUNT:
        raise ValueError("signoff template must contain exactly 81 rows")
    review_digest = _sha256_bytes(review_raw)
    template_digest = _sha256_bytes(template_raw)
    if not _is_hash(expected_review_sha256) or not _is_hash(expected_template_sha256):
        raise ValueError("frozen input hashes must be SHA-256")
    if review_digest != expected_review_sha256.lower() or template_digest != expected_template_sha256.lower():
        raise ValueError("review manifest or signoff template hash differs from the frozen input")
    if template.get("review_manifest_sha256") != review_digest:
        raise ValueError("signoff template review manifest hash differs")
    if template.get("allowed_decisions") != list(DECISIONS):
        raise ValueError("signoff template decision set differs")

    image_uris = []
    total_image_bytes = 0
    review_by_id: dict[str, dict[str, Any]] = {}
    for index, item in enumerate(review_items):
        if not isinstance(item, dict):
            raise ValueError(f"review item {index} is invalid")
        candidate_id = item.get("candidate_id")
        if not isinstance(candidate_id, str) or not candidate_id.strip() or candidate_id in review_by_id:
            raise ValueError(f"review item {index} id is missing or repeated")
        frame_id = item.get("frame_id")
        if type(frame_id) is not int or frame_id < 0:
            raise ValueError(f"review item {index} frame id is invalid")
        for field in ("source_image_sha256", "canonical_geometry_sha256", "focused_image_sha256"):
            if not _is_hash(item.get(field)):
                raise ValueError(f"review item {index} binding is invalid")
        image_path = _safe_image_path(review_root, item.get("focused_image_file"))
        image_bytes = _read_image(image_path, index)
        total_image_bytes += len(image_bytes)
        if total_image_bytes > MAX_TOTAL_IMAGE_BYTES:
            raise ValueError("focused images exceed the total bundle size limit")
        image_digest = _sha256_bytes(image_bytes)
        if image_digest != item["focused_image_sha256"].lower():
            raise ValueError(f"review item {index} focused image hash differs")
        try:
            with Image.open(BytesIO(image_bytes)) as decoded:
                decoded.verify()
        except Exception:
            raise ValueError(f"review item {index} focused image is invalid") from None
        image_uris.append("data:image/jpeg;base64," + base64.b64encode(image_bytes).decode("ascii"))
        review_by_id[candidate_id] = {field: item[field] for field in BINDING_FIELDS}

    template_by_id: dict[str, dict[str, Any]] = {}
    for index, row in enumerate(rows):
        if not isinstance(row, dict):
            raise ValueError(f"template row {index} is invalid")
        candidate_id = row.get("candidate_id")
        if not isinstance(candidate_id, str) or candidate_id not in review_by_id or candidate_id in template_by_id:
            raise ValueError(f"template row {index} id is unknown or repeated")
        expected = review_by_id[candidate_id]
        for field in BINDING_FIELDS:
            actual = row.get(field)
            if field.endswith("sha256") and isinstance(actual, str):
                actual = actual.lower()
            expected_value = expected[field]
            if field.endswith("sha256") and isinstance(expected_value, str):
                expected_value = expected_value.lower()
            if actual != expected_value:
                raise ValueError(f"template row {index} binding differs")
        if any(row.get(field) is not None for field in ("decision", "reviewer", "reviewed_at", "note")):
            raise ValueError("signoff template already contains human decisions")
        template_by_id[candidate_id] = expected
    if set(template_by_id) != set(review_by_id):
        raise ValueError("template does not exactly cover review items")
    ordered_rows = [template_by_id[item["candidate_id"]] for item in review_items]
    return ordered_rows, review_digest, template_digest, image_uris


def _page_html(rows: list[dict[str, Any]], image_files: list[str], manifest_sha: str, template_sha: str) -> str:
    if len(rows) != EXPECTED_COUNT or len(image_files) != EXPECTED_COUNT:
        raise ValueError("reviewer page requires exactly 81 bound items")
    items = []
    for index, (row, image_file) in enumerate(zip(rows, image_files)):
        items.append({**row, "focused_image_file": image_file, "index": index,
                      "decision": None, "reviewer": "", "reviewed_at": "", "note": ""})
    data = json.dumps(items, ensure_ascii=False).replace("</", "<\\/")
    data = data.replace("&", "\\u0026").replace("<", "\\u003c").replace(">", "\\u003e")
    manifest_sha = manifest_sha.replace("&", "&amp;").replace('"', "&quot;").replace("<", "&lt;")
    template_sha = template_sha.replace("&", "&amp;").replace('"', "&quot;").replace("<", "&lt;")
    return f'''<!doctype html>
<html lang="zh-CN"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>离线人工签字复核台</title><style>
:root{{color-scheme:light;--bg:#f3f5f4;--panel:#fff;--text:#202825;--muted:#65716c;--line:#d4dcd7;--accent:#176b56;--warn:#9b4b20}}
*{{box-sizing:border-box}}body{{margin:0;background:var(--bg);color:var(--text);font:15px/1.45 system-ui,"Segoe UI",sans-serif}}
header{{background:var(--panel);border-bottom:1px solid var(--line);padding:12px 18px;display:flex;gap:18px;align-items:center;flex-wrap:wrap;position:sticky;top:0;z-index:2}}
header strong{{font-size:17px}}.toolbar{{display:flex;gap:8px;align-items:center;flex-wrap:wrap}}button,select,input,textarea{{font:inherit;color:inherit}}button,select,input{{min-height:36px;border:1px solid var(--line);background:#fff;border-radius:4px;padding:6px 10px}}button{{cursor:pointer}}.button{{display:inline-flex;align-items:center;min-height:36px;border:1px solid var(--line);background:#fff;border-radius:4px;padding:6px 10px;cursor:pointer}}button.primary{{background:var(--accent);color:#fff;border-color:var(--accent)}}button:disabled{{opacity:.45;cursor:not-allowed}}.count{{font-weight:650;color:var(--accent)}}main{{max-width:1160px;margin:18px auto;padding:0 16px}}.grid{{display:grid;grid-template-columns:minmax(0,1.45fr) minmax(280px,1fr);gap:18px}}.evidence,.form{{background:var(--panel);border:1px solid var(--line);border-radius:6px;overflow:hidden}}.evidence img{{display:block;width:100%;max-height:72vh;object-fit:contain;background:#e7ece9}}.meta,.form{{padding:14px}}.meta{{display:flex;gap:20px;flex-wrap:wrap;color:var(--muted)}}.form label{{display:block;font-weight:600;margin:12px 0 5px}}.form input,.form textarea{{width:100%;border:1px solid var(--line);border-radius:4px;padding:8px}}.form textarea{{min-height:90px;resize:vertical}}fieldset{{border:0;padding:0;margin:10px 0}}.choices{{display:flex;gap:8px;flex-wrap:wrap}}.choice{{display:flex;align-items:center;gap:7px;border:1px solid var(--line);border-radius:4px;padding:8px 10px;cursor:pointer}}.choice input{{accent-color:var(--accent)}}.notice{{margin:14px 0;padding:10px 12px;border-left:3px solid var(--warn);background:#fff8f2;color:#603616}}.nav{{display:flex;justify-content:space-between;gap:8px;margin-top:12px}}.privacy{{color:var(--muted);font-size:13px}}@media(max-width:760px){{.grid{{grid-template-columns:1fr}}header{{position:static}}}}
</style></head><body>
<header><strong>离线人工签字复核台</strong><div class="toolbar"><label for="filter">筛选</label><select id="filter"><option value="all">全部</option><option value="pending">未完成</option><option value="pass">Pass</option><option value="reject">Reject</option><option value="uncertain">Uncertain</option></select><span class="count" id="counter">未完成 81 / 81</span></div><div class="toolbar"><button id="prev" type="button">上一项</button><span id="position">1 / 81</span><button id="next" type="button">下一项</button><button id="exportDraft" type="button">导出草稿</button><label for="importDraft" class="button">导入草稿</label><input id="importDraft" type="file" accept="application/json,.json" hidden><button id="export" class="primary" type="button" disabled>导出完整 JSON</button></div></header>
<main><p class="notice">人工 pass 仅表示人工复核结论，不构成训练授权。此页面不联网、不保存浏览器本地状态；请用草稿 JSON 保存和恢复进度，完成全部 81 项后才能导出正式签字结果。</p><div class="grid"><section class="evidence"><img id="evidence" alt="聚焦证据图"><div class="meta"><span>frame_id: <b id="frame"></b></span><span>candidate_id: <b id="candidate"></b></span></div></section><section class="form"><fieldset><legend><b>人工结论</b></legend><div class="choices"><label class="choice"><input type="radio" name="decision" value="pass">pass</label><label class="choice"><input type="radio" name="decision" value="reject">reject</label><label class="choice"><input type="radio" name="decision" value="uncertain">uncertain</label></div></fieldset><label for="reviewer">Reviewer（可统一填写）</label><input id="reviewer" autocomplete="off"><label for="reviewedAt">Reviewed at（ISO 时间，可统一填写）</label><input id="reviewedAt" autocomplete="off" placeholder="2026-11-22T10:00:00+08:00"><button id="applyIdentity" type="button">将 Reviewer 与时间填入全部未完成项</button><label for="note">Note</label><textarea id="note"></textarea><p class="privacy">manifest SHA-256: {manifest_sha}<br>template SHA-256: {template_sha}</p></section></div><div class="nav"><button id="prevBottom" type="button">上一项</button><span id="status"></span><button id="nextBottom" type="button">下一项</button></div></main>
<script id="items" type="application/json">{data}</script><script>
"use strict";
const items=JSON.parse(document.getElementById("items").textContent);let index=0;let importing=false;const byId=id=>document.getElementById(id);const filter=byId("filter");const DRAFT_SCHEMA="labelagent.cvat-human-signoff-draft.v1";const BINDINGS=["candidate_id","frame_id","source_image_sha256","canonical_geometry_sha256","focused_image_sha256"];const HUMAN=["decision","reviewer","reviewed_at","note"];const draftKeys=["schema_version","review_manifest_sha256","signoff_template_sha256","candidate_count","rows"];const rowKeys=[...BINDINGS,...HUMAN];
function validTime(value){{return typeof value==="string"&&/^\\d{{4}}-\\d{{2}}-\\d{{2}}T\\d{{2}}:\\d{{2}}:\\d{{2}}(?:\\.\\d+)?(?:Z|[+-]\\d{{2}}:\\d{{2}})$/.test(value.trim())&&!Number.isNaN(Date.parse(value))}}
function complete(x){{return Boolean(["pass","reject","uncertain"].includes(x.decision)&&x.reviewer&&x.reviewer.trim()&&validTime(x.reviewed_at))}}
function visible(){{return items.map((x,i)=>({{x,i}})).filter(o=>filter.value==="all"||(filter.value==="pending"&&!complete(o.x))||filter.value===o.x.decision).map(o=>o.i)}}
function save(){{if(importing)return;const x=items[index];x.decision=document.querySelector('input[name="decision"]:checked')?.value||null;for(const [field,id] of [["reviewer","reviewer"],["reviewed_at","reviewedAt"],["note","note"]]){{const value=byId(id).value;if(value!==(x[field]??""))x[field]=value}}}}
function strictObject(value,keys,label){{if(!value||typeof value!=="object"||Array.isArray(value)||Object.keys(value).some(k=>!keys.includes(k))||keys.some(k=>!Object.prototype.hasOwnProperty.call(value,k)))throw new Error(label);return value}}
function rejectDuplicateKeys(text){{let p=0;function ws(){{while(/\\s/.test(text[p]||""))p++}}function value(depth){{if(depth>64)throw new Error("depth");ws();if(text[p]==="{{"){{p++;ws();const keys=new Set();if(text[p]==="}}"){{p++;return}}while(true){{ws();if(text[p]!== '"')throw new Error("syntax");const start=p++;while(p<text.length){{if(text[p]==="\\\\")p+=2;else if(text[p++]==='"')break}}const key=JSON.parse(text.slice(start,p));if(keys.has(key))throw new Error("duplicate");keys.add(key);ws();if(text[p++]!==":")throw new Error("syntax");value(depth+1);ws();if(text[p++]==="}}")break;if(text[p-1]!==",")throw new Error("syntax")}}}}else if(text[p]==="["){{p++;ws();if(text[p]=="]"){{p++;return}}while(true){{value(depth+1);ws();if(text[p++]==="]")break;if(text[p-1]!==",")throw new Error("syntax")}}}}else if(text[p]==='"'){{p++;while(p<text.length){{if(text[p]==="\\\\")p+=2;else if(text[p++]==='"')break}}}}else{{while(p<text.length&&!/[\\s,\\]}}]/.test(text[p]))p++}}}}value(0);ws();if(p!==text.length)throw new Error("syntax")}}
function parseDraft(text){{if(new TextEncoder().encode(text).length>1048576)throw new Error("草稿超过大小限制");try{{rejectDuplicateKeys(text)}}catch(e){{throw new Error("草稿 JSON 无效")}}let value;try{{value=JSON.parse(text,(key,val)=>val)}}catch(e){{throw new Error("草稿 JSON 无效")}};strictObject(value,draftKeys,"草稿字段无效");if(value.schema_version!==DRAFT_SCHEMA||value.review_manifest_sha256!=="{manifest_sha}"||value.signoff_template_sha256!=="{template_sha}"||value.candidate_count!==81||!Array.isArray(value.rows)||value.rows.length!==81)throw new Error("草稿身份或数量无效");const seen=new Set();const next=[];for(const row of value.rows){{strictObject(row,rowKeys,"草稿行字段无效");if(seen.has(row.candidate_id)||!items.some(x=>x.candidate_id===row.candidate_id))throw new Error("草稿集合无效");seen.add(row.candidate_id);const base=items.find(x=>x.candidate_id===row.candidate_id);for(const key of BINDINGS)if(row[key]!==base[key])throw new Error("草稿绑定无效");if(!(row.decision===null||["pass","reject","uncertain"].includes(row.decision))||!["reviewer","reviewed_at","note"].every(k=>row[k]===null||typeof row[k]==="string"))throw new Error("草稿人工字段类型无效");if((row.reviewer||"").length>512||(row.reviewed_at||"").length>128||(row.note||"").length>20000)throw new Error("草稿人工字段过长");next.push(row)}}if(seen.size!==81)throw new Error("草稿集合无效");return next}}
function setBusy(value){{importing=value;document.querySelectorAll("button,input,textarea,select").forEach(control=>control.disabled=value)}}
function render(){{const x=items[index];byId("evidence").src=x.focused_image_file;byId("frame").textContent=x.frame_id;byId("candidate").textContent=x.candidate_id;byId("reviewer").value=x.reviewer||"";byId("reviewedAt").value=x.reviewed_at||"";byId("note").value=x.note||"";document.querySelectorAll('input[name="decision"]').forEach(r=>r.checked=r.value===x.decision);const n=items.filter(x=>!complete(x)).length;byId("counter").textContent=`未完成 ${{n}} / ${{items.length}}`;byId("position").textContent=`${{index+1}} / ${{items.length}}`;byId("status").textContent=x.decision?`已选择 ${{x.decision}}`:"尚未选择";byId("export").disabled=n!==0}}
function move(direction){{save();const list=visible();if(!list.length)return;let at=list.indexOf(index);if(at<0)at=0;else at=(at+direction+list.length)%list.length;index=list[at];render()}}
document.querySelectorAll('input[name="decision"]').forEach(r=>r.addEventListener("change",()=>{{save();render()}}));["reviewer","reviewedAt","note"].forEach(id=>byId(id).addEventListener("input",()=>{{save();render()}}));filter.addEventListener("change",()=>{{const list=visible();if(!list.includes(index)&&list.length)index=list[0];render()}});["prev","prevBottom"].forEach(id=>byId(id).addEventListener("click",()=>move(-1)));["next","nextBottom"].forEach(id=>byId(id).addEventListener("click",()=>move(1)));
byId("applyIdentity").addEventListener("click",()=>{{save();const reviewer=byId("reviewer").value.trim();const reviewedAt=byId("reviewedAt").value.trim();if(!reviewer||!validTime(reviewedAt)){{byId("status").textContent="请先填写 Reviewer 和带时区的合法 ISO 时间";return}}items.filter(x=>!complete(x)).forEach(x=>{{x.reviewer=reviewer;x.reviewed_at=reviewedAt}});render()}});
byId("exportDraft").addEventListener("click",()=>{{save();const rows=items.map(x=>Object.fromEntries(rowKeys.map(k=>[k,x[k]??null])));const output={{schema_version:DRAFT_SCHEMA,review_manifest_sha256:"{manifest_sha}",signoff_template_sha256:"{template_sha}",candidate_count:81,rows}};const blob=new Blob([JSON.stringify(output,null,2)+"\\n"],{{type:"application/json"}});const link=document.createElement("a");link.href=URL.createObjectURL(blob);link.download="human-signoff-draft.json";link.click();URL.revokeObjectURL(link.href)}});
byId("importDraft").addEventListener("change",async event=>{{if(importing)return;const file=event.target.files?.[0];event.target.value="";if(!file)return;setBusy(true);let message="草稿导入失败";try{{if(file.size>1048576)throw new Error("size");const text=await file.text();const next=parseDraft(text);const restored=items.map(item=>{{const row=next.find(x=>x.candidate_id===item.candidate_id);return Object.fromEntries(HUMAN.map(k=>[k,row[k]]))}});restored.forEach((row,i)=>Object.assign(items[i],row));message="草稿已导入"}}catch(error){{}}finally{{setBusy(false);render();byId("status").textContent=message}}}});
byId("export").addEventListener("click",()=>{{save();if(items.some(x=>!complete(x)))return;const rows=items.map(x=>({{candidate_id:x.candidate_id,frame_id:x.frame_id,source_image_sha256:x.source_image_sha256,canonical_geometry_sha256:x.canonical_geometry_sha256,focused_image_sha256:x.focused_image_sha256,decision:x.decision,reviewer:x.reviewer.trim(),reviewed_at:x.reviewed_at.trim(),note:x.note,approved_for_training:false}}));const output={{schema_version:"labelagent.cvat-human-instance-signoff.v1",status:"human_review_complete",review_manifest_sha256:"{manifest_sha}",candidate_count:81,allowed_decisions:["pass","reject","uncertain"],rows,human_signoff_count:81,training_authorized:false,automatic_approval:false,approved_for_training:false,human_pass_count:rows.filter(r=>r.decision==="pass").length,human_pass_is_training_authorization:false}};const blob=new Blob([JSON.stringify(output,null,2)+"\\n"],{{type:"application/json"}});const link=document.createElement("a");link.href=URL.createObjectURL(blob);link.download="human-signoff.json";link.click();URL.revokeObjectURL(link.href)}});render();
</script></body></html>'''


def build(
    review_manifest: Path,
    signoff_template: Path,
    review_root: Path,
    output_dir: Path,
    expected_review_sha256: str,
    expected_template_sha256: str,
    repository_root: Path | None = None,
) -> dict[str, Any]:
    repository_root = (repository_root or Path(__file__).resolve().parents[1]).resolve()
    private_root = (repository_root / ".repro").resolve()
    output_dir = output_dir.resolve()
    if output_dir.parent != private_root:
        raise ValueError("output directory must be a direct child of repository .repro")
    if output_dir.exists():
        raise ValueError("choose a fresh output directory")
    rows, review_sha, template_sha, image_uris = _validate_inputs(
        review_manifest, signoff_template, review_root,
        expected_review_sha256, expected_template_sha256,
    )
    page = _page_html(rows, image_uris, review_sha, template_sha)
    manifest = {
        "schema_version": "labelagent.cvat-offline-human-reviewer.v1",
        "status": "pending_human_review",
        "review_manifest_sha256": review_sha,
        "signoff_template_sha256": template_sha,
        "candidate_count": EXPECTED_COUNT,
        "pending_count": EXPECTED_COUNT,
        "human_decisions_filled": False,
        "training_authorized": False,
        "automatic_approval": False,
        "human_pass_is_training_authorization": False,
        "html_sha256": _sha256_bytes(page.encode("utf-8")),
        "items": rows,
    }
    private_root.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".{output_dir.name}.tmp-", dir=private_root))
    try:
        (staging / "index.html").write_text(page, encoding="utf-8", newline="\n")
        (staging / "manifest.json").write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8", newline="\n"
        )
        staging.rename(output_dir)
    except Exception:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    return manifest


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--review-manifest", required=True, type=Path)
    parser.add_argument("--signoff-template", required=True, type=Path)
    parser.add_argument("--review-root", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--expected-review-sha256", required=True)
    parser.add_argument("--expected-template-sha256", required=True)
    args = parser.parse_args()
    manifest = build(
        args.review_manifest, args.signoff_template, args.review_root, args.output_dir,
        args.expected_review_sha256, args.expected_template_sha256,
    )
    print(json.dumps({"status": manifest["status"], "candidate_count": manifest["candidate_count"], "pending_count": manifest["pending_count"], "training_authorized": False}, ensure_ascii=False))
    return 0


def cli() -> int:
    try:
        return main()
    except ValueError as error:
        print(f"error: {error}", file=sys.stderr)
        return 2
    except OSError:
        print("error: input or output is unavailable", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(cli())
