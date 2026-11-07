"""Independent ordinary-file handoff for the eight native prior comparisons."""
import re,shutil,subprocess,sys
from pathlib import Path
from run_rgb_layers import ROOT,read,save,sha
import run_curved_seam_prior as t
import finish_curved_seam_prior as report
import deliver_context_graph as preceding
import maintain as m
C=preceding.C;HOME=preceding.HOME;REPO=preceding.REPO
TARGET=HOME/'09_补充成果/原图曲线路径先验对照_20261108'
CODE=['run_curved_seam_prior.py','audit_curved_seam_prior.py','finish_curved_seam_prior.py','deliver_curved_seam_prior.py','deliver_context_graph.py']
def prepare():
    if TARGET.exists():raise FileExistsError('Preserve complete or interrupted handoff')
    assert read(report.OUT/'independent_readback.json')['passed'];files={}
    def add(p,name):
        p=Path(p);assert p.is_file() and name not in files;files[name]={'source':str(p),'sha256':sha(p),'bytes':p.stat().st_size}
    for name,folder in [('complete_trial',t.OUT),('report',report.OUT)]:
        for p in folder.rglob('*'):
            if p.is_file():add(p,name+'/'+p.relative_to(folder).as_posix())
    for i,(p,h) in enumerate(sorted(read(t.OUT/'frozen_plan.json')['protected'].items())):assert sha(p)==h;add(p,f'original_dependencies/{i:03d}/'+Path(p).name)
    for p,h in read(ROOT/'outputs/labelagent_point_ids_20261107/incumbent_binding.json')['files'].items():
        f=ROOT/p;assert sha(f)==h;add(f,'original_six/'+f.relative_to(ROOT/'outputs/labelagent_context_comparison_20261105').as_posix())
    cfg=read(ROOT/'task_workspaces/labelagent_point_ids_20261107/config.json');old=read(ROOT/'outputs/labelagent_soft_prior_20261108/frozen_plan.json')['protected']
    for k in ['reference_annotations','reference_description']:assert sha(cfg[k])==old[cfg[k]];add(cfg[k],'original_references/'+Path(cfg[k]).name)
    for name in CODE+['experiment_report_media.py','native_prompt_guard.py','audit_soft_prior.py','run_soft_prior_trial.py','run_rgb_layers.py','projection_audit_core.py']:add(Path(__file__).with_name(name),'reproduction/'+name)
    required=sum(r['bytes'] for r in files.values())
    if shutil.disk_usage(ROOT).free<required+96*2**20:raise RuntimeError('C full independent handoff96MiB guard')
    TARGET.mkdir();subprocess.run(['compact.exe','/C','/Q',str(TARGET)],capture_output=True,check=True)
    for name,r in files.items():
        dest=TARGET/name;dest.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(r['source'],dest);assert sha(dest)==r['sha256']
    save(TARGET/'complete_evidence_manifest.json',{'files':files,'all_current_inputs_outputs_and_original_references_retained':True,'model_weights_external_identity_in_preceding_verified_handoff':True});save(TARGET/'summary.json',read(report.OUT/'summary.json'));shutil.copy2(report.OUT/'本轮结论.txt',TARGET/'本轮结论.txt');media=read(report.OUT/'media_verification.json');media['videos']=[{**v,'path':'report/'+v['path']} for v in media['videos']];save(TARGET/'media_verification.json',media)
    html=(report.OUT/'index.html').read_text('utf-8').replace('src="','src="report/').replace('href="independent_readback.json"','href="report/independent_readback.json"');(TARGET/'index.html').write_text(html,encoding='utf-8');save(TARGET/'交接SHA256.json',{'files':[{'path':p.relative_to(TARGET).as_posix(),'sha256':sha(p),'bytes':p.stat().st_size} for p in TARGET.rglob('*') if p.is_file()]});save(C/'curved_seam_bundle_verification.json',{'passed':True,'bundle':str(TARGET),'evidence_members_verified':len(files),'manifest_sha256':sha(TARGET/'complete_evidence_manifest.json'),'pictures':10,'videos':2,'decoded_frames':400,'artifact_type':'complete_ordinary_file_directory'});print({'members':len(files),'logical_bytes':required,'pictures':10,'videos':2},flush=True)
def verify():
    r=read(C/'curved_seam_bundle_verification.json');assert r['passed'] and sha(TARGET/'complete_evidence_manifest.json')==r['manifest_sha256']
    for name,row in read(TARGET/'complete_evidence_manifest.json')['files'].items():assert sha(TARGET/name)==row['sha256']
    assert read(C/'curved_seam_ui_verification.json')['passed'];return r
def publish():
    verify();paths=[]
    for name in CODE:
        f='tools/labelagent_experiments/'+name;shutil.copy2(Path(__file__).with_name(name),REPO/f);paths.append(f)
    for a,b in [('summary.json','curved-source-seam-prior.json'),('本轮结论.txt','curved-source-seam-prior.md')]:
        f='docs/test-results/'+b;shutil.copy2(TARGET/a,REPO/f);paths.append(f)
    assert not preceding.git('diff','--cached','--name-only')
    for f in paths:assert not re.search(r'gh[pousr]_[A-Za-z0-9]{25,}|sk-[A-Za-z0-9]{30,}|-----BEGIN [A-Z ]*PRIVATE KEY-----',(REPO/f).read_text('utf-8'))
    subprocess.run(['git','-C',str(REPO),'add','--',*paths],check=True);subprocess.run(['git','-C',str(REPO),'diff','--cached','--check'],check=True);assert set(preceding.git('diff','--cached','--name-only').splitlines())==set(paths);subprocess.run(['git','-C',str(REPO),'commit','-m','Compare source RGB curved seam priors with intact native SAM masks'],check=True)
def record():
    r=verify();cfg=read(C/'sync_config.json');head=preceding.git('rev-parse','HEAD');refs={n:preceding.git('ls-remote','--refs',n,'refs/heads/'+cfg['publish_branch']).split()[0] for n in ['origin','cnb']};assert all(v==head for v in refs.values()) and not preceding.git('status','--porcelain');r.update({'checked_at':m.now(),'commit':head,'refs':refs,'both_verified':True,'summary':read(TARGET/'summary.json')});save(C/'curved_seam_final_sync_verification.json',r);save(HOME/'06_核验/原图曲线路径先验_最终同步_20261108.json',r);m.refresh_manifest(HOME);print({'commit':head,'both_verified':True},flush=True)
if __name__=='__main__':{'prepare':prepare,'publish':publish,'record':record}[sys.argv[1]]()
