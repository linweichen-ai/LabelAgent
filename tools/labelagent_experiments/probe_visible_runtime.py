"""Read listener identity and public health fields, never command credentials."""
import json,urllib.request,datetime
from pathlib import Path
import psutil
ROOT=Path(__file__).resolve().parents[2];out=ROOT/'outputs/labelagent_visible_base_20261107';r={'checked_at':datetime.datetime.now().isoformat(),'listeners':[]}
for c in psutil.net_connections(kind='tcp'):
 if c.status=='LISTEN' and c.laddr.port in [8765,8766,18868] and c.pid:
  p=psutil.Process(c.pid);args=p.cmdline();r['listeners'].append({'port':c.laddr.port,'pid':c.pid,'exe':p.exe(),'script_files':[a for a in args[1:] if a.endswith('.py')],'created':p.create_time()})
for name,url in [('SAM','http://127.0.0.1:8766/health'),('VLM','http://127.0.0.1:18868/health')]:
 try:
  with urllib.request.urlopen(url,timeout=15) as f:r[name]=json.load(f)
 except Exception as e:r[name]={'error':str(e)}
(out/'runtime_probe.json').write_text(json.dumps(r,ensure_ascii=False,indent=2)+'\n',encoding='utf-8');print(json.dumps(r,ensure_ascii=False),flush=True)
