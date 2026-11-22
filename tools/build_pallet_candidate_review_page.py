#!/usr/bin/env python3
"""Build a static, read-only review page for a pallet candidate bundle."""

from __future__ import annotations

import argparse
from html import escape
import json
from pathlib import Path


def build_page(bundle_path: Path, output: Path) -> None:
    if output.exists():
        raise ValueError("output already exists")
    bundle = json.loads(bundle_path.read_text(encoding="utf-8"))
    if (bundle.get("schema_version") != "labelagent.pallet_visibility_candidate_bundle.v1"
            or bundle.get("production_deployed") is not False
            or bundle.get("human_truth_writes") != 0
            or bundle.get("training_data_writes") != 0):
        raise ValueError("requires a review-only non-deployed candidate bundle")
    stats = [
        ("候选实例", bundle["candidate_count"]),
        ("输出掩码", bundle["mask_candidates_emitted"]),
        ("抑制掩码", bundle["current_2d_candidates_suppressed"]),
        ("等待复核", bundle["candidate_count"]),
    ]
    stat_html = "".join(
        f'<section class="stat" data-stat="{escape(label)}"><span>{escape(label)}</span>'
        f'<strong>{int(value)}</strong></section>' for label, value in stats)
    cards = []
    for item in bundle["previews"]:
        title = f"{item['sequence']} · camera{item['camera']} · {item['frame_id']}"
        cards.append(
            f'<article class="scene"><header><div><small>冻结复核场景</small>'
            f'<h2>{escape(title)}</h2></div><span class="chip">只读候选</span></header>'
            f'<img src="{escape(item["path"])}" alt="{escape(title)}候选预览">'
            f'</article>')
    table_rows = []
    for row in bundle["rows"]:
        state = "输出候选掩码" if row["mask_path"] else "抑制二维掩码"
        badge = "allow" if row["mask_path"] else "suppress"
        source = "、".join(row.get("propagation_source_frame_ids") or []) or "—"
        table_rows.append(
            f'<tr><td>{escape(str(row["sequence"]))} / camera{int(row["camera"])}</td>'
            f'<td>{escape(str(row["obj_id"]))}</td>'
            f'<td><span class="route {badge}">{escape(state)}</span></td>'
            f'<td>{int(row["mask_pixels"]):,}</td><td class="mono">{escape(source)}</td>'
            f'<td>必须复核</td></tr>')
    html = f'''<!doctype html>
<html lang="zh-CN"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>LabelAgent · 托盘候选复核</title>
<style>
:root{{--ink:#102a2a;--muted:#687c78;--line:#dce9e5;--paper:#f4f8f6;--card:#fff;--accent:#0f8b68;--warn:#b44d45}}
*{{box-sizing:border-box}} body{{margin:0;background:linear-gradient(180deg,#edf7f3 0,#f7faf9 280px);color:var(--ink);font:14px/1.55 system-ui,"Microsoft YaHei",sans-serif}}
main{{max-width:1320px;margin:auto;padding:30px 28px 56px}} .top{{display:flex;justify-content:space-between;gap:24px;align-items:flex-start;margin-bottom:22px}}
h1{{font-size:28px;margin:4px 0 6px;letter-spacing:-.4px}} h2{{font-size:16px;margin:2px 0 0}} p{{margin:0;color:var(--muted)}}
.status{{background:#fff3f1;color:#8b3832;border:1px solid #f0cec9;border-radius:12px;padding:10px 14px;font-weight:700}}
.stats{{display:grid;grid-template-columns:repeat(4,1fr);gap:12px;margin-bottom:18px}} .stat{{background:rgba(255,255,255,.88);border:1px solid var(--line);border-radius:14px;padding:14px 16px;box-shadow:0 8px 24px rgba(27,82,67,.05)}}
.stat span{{color:var(--muted)}} .stat strong{{display:block;font-size:25px;margin-top:2px}}
.scenes{{display:grid;grid-template-columns:1fr 1fr;gap:16px}} .scene{{background:var(--card);border:1px solid var(--line);border-radius:16px;overflow:hidden;box-shadow:0 12px 32px rgba(27,82,67,.07)}}
.scene header{{display:flex;justify-content:space-between;align-items:center;padding:14px 16px}} small{{color:var(--muted)}} .chip{{background:#e9f6f1;color:#087455;border-radius:999px;padding:5px 9px;font-size:12px;font-weight:700}}
.scene img{{width:100%;display:block;aspect-ratio:5/4;object-fit:contain;background:#111b21}}
.panel{{margin-top:18px;background:var(--card);border:1px solid var(--line);border-radius:16px;padding:18px;box-shadow:0 12px 32px rgba(27,82,67,.06)}}
.panel-head{{display:flex;justify-content:space-between;align-items:end;margin-bottom:12px}} table{{width:100%;border-collapse:collapse}} th,td{{padding:11px 10px;border-top:1px solid #e7efec;text-align:left;vertical-align:top}} th{{border-top:0;color:var(--muted);font-size:12px}}
.route{{display:inline-block;padding:4px 8px;border-radius:8px;font-weight:700;font-size:12px}} .allow{{background:#e7f6ef;color:#087455}} .suppress{{background:#fff0ee;color:#a33f37}} .mono{{font-family:ui-monospace,SFMono-Regular,Consolas,monospace;font-size:12px;max-width:360px}}
.note{{margin-top:14px;padding:12px 14px;border-left:3px solid var(--accent);background:#f0f8f5;color:#47615b}}
@media(max-width:900px){{.stats{{grid-template-columns:1fr 1fr}}.scenes{{grid-template-columns:1fr}}.top{{flex-direction:column}}.panel{{overflow:auto}}table{{min-width:850px}}}}
</style></head><body><main>
<header class="top"><div><p>LabelAgent / Pallet Visibility Review</p><h1>托盘可见性候选复核</h1><p>LiDAR、3D 深度与时序证据汇总；仅供人工检查。</p></div><div class="status" data-deployment="not-deployed">离线候选 · 未部署 · 不是真值</div></header>
<div class="stats">{stat_html}</div><section class="scenes">{''.join(cards)}</section>
<section class="panel"><div class="panel-head"><div><small>机器可读路由</small><h2>实例处理清单</h2></div><span class="chip">{len(bundle['rows'])} 项全部需复核</span></div>
<table><thead><tr><th>场景</th><th>obj_id</th><th>处理</th><th>掩码像素</th><th>传播源帧</th><th>状态</th></tr></thead><tbody>{''.join(table_rows)}</tbody></table>
<div class="note">安全边界：本页没有保存、写回、训练或部署入口。绿色区域仍是候选；红色抑制项只保留 3D 身份与时序证据，不生成当前帧二维掩码。</div></section>
</main></body></html>'''
    output.write_text(html, encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args(); build_page(args.bundle, args.output); return 0


if __name__ == "__main__":
    raise SystemExit(main())
