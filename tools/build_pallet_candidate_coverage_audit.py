#!/usr/bin/env python3
"""Build a fail-closed 12-view coverage audit for pallet conflict routing."""

from __future__ import annotations

import argparse
import hashlib
from html import escape
import json
from pathlib import Path
import shutil


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read(path: Path, schema: str) -> dict:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("schema_version") != schema:
        raise ValueError(f"unexpected schema: {path}")
    if (payload.get("production_deployed") is not False
            or payload.get("workbench_writes") != 0):
        raise ValueError(f"unsafe source flags: {path}")
    return payload


def _key(row: dict) -> tuple[str, int, str]:
    return str(row["sequence"]), int(row["camera"]), str(row["frame_id"])


def _render(report: dict, output: Path) -> None:
    rows = []
    for row in report["rows"]:
        triggered = row["conflict_triggered"]
        route = (f'{row["route_candidate_count"]} 个候选'
                 if triggered else "安全跳过")
        kind = "trigger" if triggered else "skip"
        rows.append(
            f'<tr><td>{escape(row["sequence"])}</td><td>camera{row["camera"]}</td>'
            f'<td class="mono">{escape(row["frame_id"])}</td>'
            f'<td>{row["target_obj_id"]}</td>'
            f'<td>{row["maximum_other_projected_box_fraction"]:.2%}</td>'
            f'<td><span class="{kind}">{route}</span></td></tr>')
    html = f'''<!doctype html><html lang="zh-CN"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1"><title>托盘冲突路由覆盖审计</title>
<style>
:root{{--ink:#102a2a;--muted:#687c78;--line:#dce9e5;--paper:#f4f8f6;--card:#fff;--accent:#0f8b68}}
*{{box-sizing:border-box}}body{{margin:0;background:linear-gradient(180deg,#edf7f3,#f7faf9 320px);color:var(--ink);font:14px/1.5 system-ui,"Microsoft YaHei",sans-serif}}
main{{max-width:1320px;margin:auto;padding:30px 28px 56px}}h1{{margin:3px 0 5px;font-size:27px}}h2{{margin:0;font-size:17px}}p{{margin:0;color:var(--muted)}}
.top{{display:flex;justify-content:space-between;gap:20px;align-items:flex-start}}.status{{padding:9px 13px;border:1px solid #9fcfbd;border-radius:10px;background:#e9f6f1;color:#087455;font-weight:800}}
.stats{{display:grid;grid-template-columns:repeat(4,1fr);gap:12px;margin:20px 0}}.stat,.panel{{border:1px solid var(--line);border-radius:15px;background:rgba(255,255,255,.92);box-shadow:0 10px 28px rgba(27,82,67,.06)}}
.stat{{padding:14px 16px}}.stat span{{color:var(--muted)}}.stat strong{{display:block;margin-top:2px;font-size:25px}}.panel{{padding:17px;margin-top:16px}}
.contact{{width:100%;display:block;margin-top:13px;border-radius:11px;background:#111b21}}table{{width:100%;border-collapse:collapse;margin-top:10px}}th,td{{padding:9px;border-top:1px solid #e7efec;text-align:left}}th{{border-top:0;color:var(--muted);font-size:12px}}.mono{{font:12px ui-monospace,Consolas,monospace}}
.skip,.trigger{{display:inline-block;padding:4px 8px;border-radius:8px;font-size:12px;font-weight:800}}.skip{{background:#e7f6ef;color:#087455}}.trigger{{background:#fff0ee;color:#a33f37}}
.note{{margin-top:13px;padding:11px 13px;border-left:3px solid var(--accent);background:#f0f8f5;color:#47615b}}@media(max-width:800px){{.stats{{grid-template-columns:1fr 1fr}}.panel{{overflow:auto}}table{{min-width:800px}}}}
</style></head><body><main><header class="top"><div><p>LabelAgent / Pallet Conflict Coverage</p><h1>托盘冲突路由覆盖审计</h1><p>3 个序列、4 路相机；检查触发与安全跳过是否一致。</p></div><div class="status">离线验证 · 未部署</div></header>
<section class="stats"><div class="stat"><span>跨视角图像</span><strong>{report['image_count']}</strong></div><div class="stat"><span>安全跳过</span><strong>{report['safe_skip_count']}</strong></div><div class="stat"><span>冲突触发</span><strong>{report['trigger_count']}</strong></div><div class="stat"><span>误触发 / 漏触发</span><strong>{report['false_trigger_count']} / {report['missed_trigger_count']}</strong></div></section>
<section class="panel"><h2>12 视角定位结果</h2><p>绿色代表无需拆分；红色代表进入深度、LiDAR 与时序候选路由。</p><img class="contact" src="contact-sheet.jpg" alt="12路托盘定位接触表"></section>
<section class="panel"><h2>逐视角路由清单</h2><table><thead><tr><th>序列</th><th>相机</th><th>帧 ID</th><th>目标</th><th>最大相邻框重叠</th><th>路由</th></tr></thead><tbody>{''.join(rows)}</tbody></table>
<div class="note">安全边界：安全跳过只表示本轮 3D 投影冲突门未触发；不等于人工像素真值。触发项也只是待复核候选，不会自动写入训练集或正式工作台。</div></section></main></body></html>'''
    (output / "index.html").write_text(html, encoding="utf-8")


def build(localization_path: Path, conflict_path: Path, route_path: Path,
          output: Path) -> dict:
    if output.exists():
        raise ValueError("output already exists")
    localization = _read(localization_path, "labelagent.pallet_box3d_roi_sam.v1")
    conflict = _read(conflict_path, "labelagent.pallet_box3d_overlap_pairs.v1")
    routes = _read(route_path, "labelagent.pallet_visibility_candidate_routes.v1")
    if conflict.get("source_report_sha256") != sha256(localization_path):
        raise ValueError("conflict report is not bound to localization report")
    base_rows = localization.get("rows", [])
    conflict_rows = conflict.get("rows", [])
    route_rows = routes.get("rows", [])
    base = {_key(row): row for row in base_rows}
    conflicts = {_key(row): row for row in conflict_rows}
    route_groups: dict[tuple[str, int, str], list[dict]] = {}
    for row in route_rows:
        route_groups.setdefault(_key(row), []).append(row)
    if (len(base) != len(base_rows) or len(conflicts) != len(conflict_rows)
            or set(conflicts) != set(route_groups)
            or not set(conflicts).issubset(base)):
        raise ValueError("image identities are incomplete, duplicated or misaligned")
    if (localization.get("image_count") != len(base)
            or conflict.get("selected_source_row_count") != len(base)
            or conflict.get("skipped_no_conflict_count") != len(base) - len(conflicts)):
        raise ValueError("source count contract failed")

    rows = []
    false_triggers = missed_triggers = 0
    for key, source in sorted(base.items()):
        if source.get("localization_pass") is not True:
            raise ValueError("coverage audit requires localized views")
        expected_trigger = source.get("instance_isolation_pass") is False
        triggered = key in conflicts
        if triggered and not expected_trigger:
            false_triggers += 1
        if expected_trigger and not triggered:
            missed_triggers += 1
        candidates = route_groups.get(key, [])
        if ((triggered and not candidates) or (not triggered and candidates)
                or any(item.get("requires_review") is not True
                       or item.get("write_human_truth") is not False
                       or item.get("write_training_data") is not False
                       for item in candidates)):
            raise ValueError("candidate routes violate review-only trigger contract")
        rows.append({
            "sequence": key[0], "camera": key[1], "frame_id": key[2],
            "target_obj_id": int(source["target_obj_id"]),
            "maximum_other_projected_box_fraction": float(
                source["metrics"]["maximum_other_projected_box_fraction"]),
            "localization_pass": True,
            "instance_isolation_pass": bool(source["instance_isolation_pass"]),
            "conflict_triggered": triggered,
            "route_candidate_count": len(candidates),
            "route_names": sorted(item["route"] for item in candidates),
        })
    if false_triggers or missed_triggers:
        raise ValueError("trigger alignment failed")

    contact = Path(localization["contact_sheet"])
    if not contact.is_file() or sha256(contact) != localization["contact_sheet_sha256"]:
        raise ValueError("contact sheet is missing or changed")
    output.mkdir(parents=True)
    shutil.copyfile(contact, output / "contact-sheet.jpg")
    report = {
        "schema_version": "labelagent.pallet_candidate_coverage_audit.v1",
        "status": "offline_cross_view_coverage_not_deployed",
        "localization_report_sha256": sha256(localization_path),
        "conflict_report_sha256": sha256(conflict_path),
        "route_report_sha256": sha256(route_path),
        "contact_sheet_sha256": sha256(output / "contact-sheet.jpg"),
        "sequence_count": len({row["sequence"] for row in rows}),
        "camera_count": len({row["camera"] for row in rows}),
        "image_count": len(rows),
        "safe_skip_count": sum(not row["conflict_triggered"] for row in rows),
        "trigger_count": sum(row["conflict_triggered"] for row in rows),
        "false_trigger_count": false_triggers,
        "missed_trigger_count": missed_triggers,
        "candidate_count": sum(row["route_candidate_count"] for row in rows),
        "human_truth_writes": 0, "training_data_writes": 0,
        "workbench_writes": 0, "production_deployed": False,
        "rows": rows,
    }
    (output / "coverage.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    _render(report, output)
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--localization", required=True, type=Path)
    parser.add_argument("--conflicts", required=True, type=Path)
    parser.add_argument("--routes", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    report = build(args.localization, args.conflicts, args.routes, args.output)
    print(json.dumps({key: report[key] for key in (
        "image_count", "safe_skip_count", "trigger_count",
        "false_trigger_count", "missed_trigger_count", "candidate_count")},
        ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

