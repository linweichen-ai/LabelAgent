"""Commit only reviewed code and aggregate diagnostics to the authorized branch."""
import subprocess,re,json
from publish_point_ids import ROOT,REPO,CONTROL,NAMES,TESTS
paths=['tools/labelagent_experiments/'+n for n in NAMES if (REPO/'tools/labelagent_experiments'/n).exists()]+['tests/'+n for n in TESTS]+[p.relative_to(REPO).as_posix() for p in (REPO/'docs/test-results').glob('point-ids*')]
def git(*args):return subprocess.check_output(['git','-C',str(REPO),*args],text=True,encoding='utf-8').strip()
assert not git('diff','--cached','--name-only')
for rel in paths:
 p=REPO/rel;assert p.suffix in ['.py','.md','.json'] and p.stat().st_size<2*1024**2
 assert not re.search(r'(?:gh[pousr]_[A-Za-z0-9]{25,}|sk-[A-Za-z0-9]{30,}|AKIA[A-Z0-9]{16}|-----BEGIN (?:RSA |OPENSSH )?PRIVATE KEY-----)',p.read_text(encoding='utf-8'))
subprocess.run(['git','-C',str(REPO),'add','--',*paths],check=True);subprocess.run(['git','-C',str(REPO),'diff','--cached','--check'],check=True);assert set(git('diff','--cached','--name-only').splitlines())==set(paths)
message=CONTROL/'point_ids_commit_message.txt';message.write_text('Audit fixed point IDs, blank hallucinations and stacked-base lineage\n\nKeep the original 13/17 and verified six Agent runs. Record 16 real plus four synthetic planning responses without inventing SAM outputs. Verify image-only temporal correspondences and the original per-layer pallet policy. Preserve complete masks, failed empty-output receipt and all original acceptance limits.\n',encoding='utf-8')
subprocess.run(['git','-C',str(REPO),'commit','-F',str(message)],check=True);print(json.dumps({'files':len(paths),'commit':git('rev-parse','HEAD'),'raw_media_labels_and_weights_not_published':True}),flush=True)
