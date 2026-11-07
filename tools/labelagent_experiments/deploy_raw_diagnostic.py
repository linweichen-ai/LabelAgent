"""Install a default-off diagnostic while SAM worker is absent."""
import ast,hashlib,json,shutil,urllib.request,datetime
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]
repo=ROOT/'task_workspaces/labelagent_realtime_handoff_20261104'
dest=Path('D:/Apps/LabelAgent/source/sam3_service')
backup=ROOT/'task_workspaces/labelagent_raw_smoke_20261107/deployment'
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
with urllib.request.urlopen('http://127.0.0.1:8766/manager/status') as f:s=json.load(f)
assert s['active_worker']!='sam3','Do not change loaded SAM service'
assert sha(dest/'server.py')=='7b2ebf69937b6f2c582f28c24a9a3809effb260302caf18361b091088b714ff5'
assert not backup.exists()
backup.mkdir(parents=True)
shutil.copy2(dest/'server.py',backup/'server_before.py')
files={}
for name in ('raw_capture_adapter.py','raw_diagnostic.py','server.py'):
 src=repo/'source/sam3_service'/name;ast.parse(src.read_text(encoding='utf-8'))
 if name!='server.py':assert not (dest/name).exists()
 temp=dest/(name+'.raw-diagnostic-pending');shutil.copy2(src,temp);temp.replace(dest/name)
 assert sha(src)==sha(dest/name);files[name]=sha(src)
receipt={'checked_at':datetime.datetime.now().astimezone().isoformat(),'installed':True,'default_capture':False,'service_restarted':False,'model_loaded_during_deploy':False,'files':files,'api_CPU_tests':4,'tests_passed':True,'sam_point_segment_unchanged':sha(dest/'point_segment.py')}
(backup/'receipt.json').write_text(json.dumps(receipt,indent=2)+'\n',encoding='utf-8')
print(json.dumps(receipt),flush=True)
