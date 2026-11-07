"""Publish sanitized comparison and verify both remotes for complete folder."""
import re,shutil,subprocess,sys
import deliver_zoom_registry as d
from run_rgb_layers import read,save,sha
import maintain as m
CODE=d.CODE+['seal_zoom_registry_folder.py','publish_zoom_registry_folder.py']

def verify():
    r=read(d.C/'zoom_registry_bundle_verification.json');assert r['passed'] and r['artifact_type']=='complete_ordinary_file_directory';manifest=d.TARGET/'complete_evidence/evidence_manifest.json';assert sha(manifest)==r['evidence_manifest_sha256']
    rows=read(manifest)['files'];assert len(rows)==r['evidence_members_verified']
    for row in rows:assert sha(d.TARGET/'complete_evidence'/row['member'])==row['sha256']
    assert read(d.C/'zoom_registry_ui_verification.json')['passed'];return r

def publish():
    verify()
    for name in CODE:shutil.copy2(d.Path(__file__).with_name(name),d.REPO/'tools/labelagent_experiments'/name)
    for a,b in [('combined_summary.json','four-trained-checkpoints-native-masks.json'),('本轮结论.txt','four-trained-checkpoints-native-masks.md')]:shutil.copy2(d.TARGET/a,d.REPO/'docs/test-results'/b)
    paths=['tools/labelagent_experiments/'+n for n in CODE]+['docs/test-results/four-trained-checkpoints-native-masks.json','docs/test-results/four-trained-checkpoints-native-masks.md'];assert not d.git('diff','--cached','--name-only')
    for p in paths:assert not re.search(r'gh[pousr]_[A-Za-z0-9]{25,}|sk-[A-Za-z0-9]{30,}|-----BEGIN [A-Z ]*PRIVATE KEY-----',(d.REPO/p).read_text(encoding='utf-8'))
    subprocess.run(['git','-C',str(d.REPO),'add','--',*paths],check=True);subprocess.run(['git','-C',str(d.REPO),'diff','--cached','--check'],check=True);assert set(d.git('diff','--cached','--name-only').splitlines())==set(paths);subprocess.run(['git','-C',str(d.REPO),'commit','-m','Compare four trained teachers and preserve full native mask alternatives'],check=True)

def record():
    r=verify();cfg=read(d.C/'sync_config.json');head=d.git('rev-parse','HEAD');refs={n:d.git('ls-remote','--refs',n,'refs/heads/'+cfg['publish_branch']).split()[0] for n in ['origin','cnb']};assert all(v==head for v in refs.values()) and not d.git('status','--porcelain');receipt={**r,'checked_at':m.now(),'commit':head,'refs':refs,'both_verified':True,'summary':read(d.TARGET/'combined_summary.json')};save(d.C/'zoom_registry_final_sync_verification.json',receipt);save(d.HOME/'06_核验/四套训练权重与完整原生掩码_最终同步_20261108.json',receipt);m.refresh_manifest(d.HOME);print({'commit':head,'both_verified':True,'pictures':35,'videos':5,'complete_evidence_files':r['evidence_members_verified']},flush=True)

if __name__=='__main__':{'publish':publish,'record':record}[sys.argv[1]]()
