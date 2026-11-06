"""Stage only reviewed source and aggregate reports for the authorized mirrors."""
import subprocess,re,json
from publish_native_resume import ROOT,REPO,CONTROL,NAMES,OUT,read
paths=['tools/labelagent_experiments/'+n for n in NAMES if (REPO/'tools/labelagent_experiments'/n).exists()]+['tests/test_native_resume.py']+[p.relative_to(REPO).as_posix() for p in (REPO/'docs/test-results').glob('native-resume*')]
for rel in paths:
 p=REPO/rel;assert p.suffix in ['.py','.md','.json'] and p.stat().st_size<2*1024**2
 assert not re.search(r'(?:gh[pousr]_[A-Za-z0-9]{25,}|sk-[A-Za-z0-9]{30,}|AKIA[A-Z0-9]{16}|-----BEGIN (?:RSA |OPENSSH )?PRIVATE KEY-----)',p.read_text(encoding='utf-8'))
def git(*a):return subprocess.check_output(['git','-C',str(REPO),*a],text=True,encoding='utf-8').strip()
assert not git('diff','--cached','--name-only')
subprocess.run(['git','-C',str(REPO),'add','--',*paths],check=True)
subprocess.run(['git','-C',str(REPO),'diff','--cached','--check'],check=True)
staged=git('diff','--cached','--name-only').splitlines();assert set(staged)==set(paths)
message=CONTROL/'native_resume_commit_message.txt';s=read(OUT/'final_summary.json')
message.write_text('Resume native-coordinate grounding under unchanged instance gates\n\nFix source edge mapping and verify the actual 512-token request limit. Preserve all complete outputs, failures, originals and the incumbent. Record actual planning and SAM requests separately from reused Agent convergence. Keep image/video evidence local and publish aggregate checks.\n',encoding='utf-8')
subprocess.run(['git','-C',str(REPO),'commit','-F',str(message)],check=True)
print(json.dumps({'files':len(paths),'commit':git('rev-parse','HEAD'),'no_raw_media_labels_or_weights':True}),flush=True)
