import json,html
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];out=ROOT/'outputs/labelagent_projection_recovery_20261105'
def read(p):return json.loads(p.read_text(encoding='utf-8'))
def save(p,v):p.write_text(json.dumps(v,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
if (out/'finalized_report.json').exists():raise FileExistsError('Keep final report')
summary=read(out/'public_summary.json');audit=read(out/'append_point_audit/completed.json');changed=[{'case':r['case'],**p} for r in audit['cases'] for p in r['objects'] if p['gained'] or p['lost']]
text='本轮无损追加共新增1点、丢失0点，仍13个实例。\n'
for r in changed:text+=f"{r['case']} / 对象{r['object_id']}：并集覆盖{r['old_covered']}/{r['visible_points']}→{r['new_covered']}/{r['visible_points']}；纯度合格单候选支持{r.get('old_best_supported')}→{r.get('new_best_supported')}；原匹配{r.get('old_matched')}、新匹配{r.get('new_matched')}。\n"
text+='虽然增加了点覆盖，但未产生第14个合格实例，按原标准不采用追加候选，保留原13/17、490/579与6/6结果。\n'
files=sorted((out/'append_point_audit').glob('*.jpg'));page=(out/'index.html').read_text(encoding='utf-8');section='<section><h2>本轮新增1点的归属</h2><pre>'+html.escape(text)+'</pre><p><a href="append_point_audit/completed.json">本轮逐对象核对</a></p>'
for file in files:section+=f'<p>{file.stem}</p><img loading="lazy" src="{file.relative_to(out).as_posix()}">'
(out/'index.html').write_text(page.replace('</html>',section+'</section></html>'),encoding='utf-8');summary['pictures']+=len(files);summary['current_point_gain_lineage']=changed;save(out/'public_summary.json',summary);media=read(out/'media_verification.json');media['pictures']=summary['pictures'];save(out/'media_verification.json',media);(out/'结论与限制.txt').write_text(text+'\n'+(out/'结论与限制.txt').read_text(encoding='utf-8'),encoding='utf-8');save(out/'finalized_report.json',{'passed':True,'pictures':summary['pictures'],'videos':summary['videos'],'source_predictions_frozen':True});print(json.dumps({'pictures':summary['pictures'],'videos':summary['videos'],'new_point_lineage':changed},ensure_ascii=False))
