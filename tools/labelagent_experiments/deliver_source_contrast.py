"""Full contrast trial handoff and sanitized dual-repository publication."""
import re,shutil,subprocess,sys
from pathlib import Path
from run_rgb_layers import ROOT,read,save,sha
import maintain as m
C=ROOT/'task_workspaces/labelagent_handoff_control_20261104';HOME=ROOT/'deliverables/LabelAgent_实时交接';REPO=ROOT/'task_workspaces/labelagent_realtime_handoff_20261104';TARGET=HOME/'09_补充成果/源图对比度与边缘增强_20261108';BASE=ROOT/'outputs/labelagent_source_contrast_20261108';REPORT=ROOT/'outputs/labelagent_source_contrast_report_20261108'
CODE=['run_source_contrast.py','audit_source_contrast.py','finish_source_contrast.py','deliver_source_contrast.py']

def git(*a):return subprocess.check_output(['git','-C',str(REPO),*a],text=True,encoding='utf-8').strip()

def prepare():
    if TARGET.exists():raise FileExistsError('Preserve complete contrast evidence')
    assert read(REPORT/'independent_readback.json')['passed'];p=read(BASE/'frozen_plan.json');assert all(sha(f)==h for f,h in p['protected'].items());files={}
    def add(source,name):
        source=Path(source);assert source.is_file() and name not in files;files[name]={'source':str(source),'sha256':sha(source),'bytes':source.stat().st_size}
    for name,folder in [('complete_trial',BASE),('report',REPORT)]:
        for f in folder.rglob('*'):
            if f.is_file():add(f,name+'/'+f.relative_to(folder).as_posix())
    for i,f in enumerate(sorted(p['protected'])):add(f,f'original_dependencies/{i:03d}/'+Path(f).name)
    for row in read(ROOT/'outputs/labelagent_two_sided_layers_20261108/SAM/generation_completed.json')['rows']:
        for name in ['complete_full_mask.npy','all_arrays.npz','metadata.json','response.json']:add(Path(row['folder'])/name,f'original_five_point_captures/cue{row["source_partition_index"]:02d}/'+name)
    binding=read(ROOT/'outputs/labelagent_point_ids_20261107/incumbent_binding.json')['files']
    for f,h in binding.items():assert sha(ROOT/f)==h;add(ROOT/f,'original_six/'+Path(f).relative_to('outputs/labelagent_context_comparison_20261105').as_posix())
    cfg=read(ROOT/'task_workspaces/labelagent_point_ids_20261107/config.json');old=read(ROOT/'outputs/labelagent_soft_prior_20261108/frozen_plan.json')['protected']
    for key in ['reference_annotations','reference_description']:assert sha(cfg[key])==old[cfg[key]];add(cfg[key],'original_references/'+Path(cfg[key]).name)
    for name in CODE+['native_prompt_guard.py','run_trained_point_sam.py','run_rgb_layers.py','audit_soft_prior.py','run_soft_prior_trial.py','projection_audit_core.py']:add(Path(__file__).with_name(name),'reproduction/'+name)
    if shutil.disk_usage(ROOT).free<sum(r['bytes'] for r in files.values())+96*2**20:raise RuntimeError('Complete independent evidence plus original96MiB reserve')
    TARGET.mkdir();subprocess.run(['compact.exe','/C','/Q',str(TARGET)],capture_output=True,check=True)
    for name,r in files.items():
        if shutil.disk_usage(ROOT).free<r['bytes']+96*2**20:raise RuntimeError('Original96MiB reserve during evidence copies')
        dest=TARGET/name;dest.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(r['source'],dest);assert sha(dest)==r['sha256']
    save(TARGET/'complete_evidence_manifest.json',{'files':files,'all_full_native_masks_and_original_sources_exact':True,'input_photometric_change_explicit_output_unchanged':True});save(TARGET/'combined_summary.json',read(REPORT/'summary.json'));media=read(REPORT/'media_verification.json');media['videos']=[{**v,'path':'report/'+v['path']} for v in media['videos']];save(TARGET/'media_verification.json',media);shutil.copy2(REPORT/'本轮结论.txt',TARGET/'本轮结论.txt');html=(REPORT/'index.html').read_text(encoding='utf-8').replace('src="','src="report/').replace('href="summary.json"','href="combined_summary.json"').replace('href="independent_readback.json"','href="report/independent_readback.json"');(TARGET/'index.html').write_text(html,encoding='utf-8');save(TARGET/'交接SHA256.json',{'files':[{'path':f.relative_to(TARGET).as_posix(),'sha256':sha(f)} for f in TARGET.rglob('*') if f.is_file()]});save(C/'source_contrast_bundle_verification.json',{'passed':True,'bundle':str(TARGET),'artifact_type':'complete_ordinary_file_directory','evidence_members_verified':len(files),'manifest_sha256':sha(TARGET/'complete_evidence_manifest.json'),'pictures':4,'videos':2,'decoded_frames':160});print({'evidence_members_verified':len(files),'pictures':4,'videos':2},flush=True)

def verify():
    r=read(C/'source_contrast_bundle_verification.json');assert r['passed'] and sha(TARGET/'complete_evidence_manifest.json')==r['manifest_sha256']
    for name,row in read(TARGET/'complete_evidence_manifest.json')['files'].items():assert sha(TARGET/name)==row['sha256']
    assert read(C/'source_contrast_ui_verification.json')['passed'];return r

def publish():
    verify()
    for name in CODE:shutil.copy2(Path(__file__).with_name(name),REPO/'tools/labelagent_experiments'/name)
    for a,b in [('combined_summary.json','source-contrast-native-masks.json'),('本轮结论.txt','source-contrast-native-masks.md')]:shutil.copy2(TARGET/a,REPO/'docs/test-results'/b)
    paths=['tools/labelagent_experiments/'+n for n in CODE]+['docs/test-results/source-contrast-native-masks.json','docs/test-results/source-contrast-native-masks.md'];assert not git('diff','--cached','--name-only')
    for path in paths:assert not re.search(r'gh[pousr]_[A-Za-z0-9]{25,}|sk-[A-Za-z0-9]{30,}|-----BEGIN [A-Z ]*PRIVATE KEY-----',(REPO/path).read_text(encoding='utf-8'))
    subprocess.run(['git','-C',str(REPO),'add','--',*paths],check=True);subprocess.run(['git','-C',str(REPO),'diff','--cached','--check'],check=True);assert set(git('diff','--cached','--name-only').splitlines())==set(paths);subprocess.run(['git','-C',str(REPO),'commit','-m','Compare fixed source contrast and sharpen inputs with full native masks'],check=True)

def record():
    r=verify();cfg=read(C/'sync_config.json');head=git('rev-parse','HEAD');refs={n:git('ls-remote','--refs',n,'refs/heads/'+cfg['publish_branch']).split()[0] for n in ['origin','cnb']};assert all(v==head for v in refs.values()) and not git('status','--porcelain');receipt={**r,'checked_at':m.now(),'commit':head,'refs':refs,'both_verified':True,'summary':read(TARGET/'combined_summary.json')};save(C/'source_contrast_final_sync_verification.json',receipt);save(HOME/'06_核验/源图对比度与边缘增强_最终同步_20261108.json',receipt);m.refresh_manifest(HOME);print({'commit':head,'both_verified':True,'pictures':4,'videos':2},flush=True)

if __name__=='__main__':{'prepare':prepare,'publish':publish,'record':record}[sys.argv[1]]()
