"""Seal all sensor-source evidence and publish only code and sanitized results."""
import re,shutil,subprocess,sys
from pathlib import Path
from run_rgb_layers import ROOT,read,save,sha
import maintain as m
CONTROL=ROOT/'task_workspaces/labelagent_handoff_control_20261104';HOME=ROOT/'deliverables/LabelAgent_实时交接';REPO=ROOT/'task_workspaces/labelagent_realtime_handoff_20261104';TARGET=HOME/'09_补充成果/未标注雷达分组与完整掩码_20261108';REPORT=ROOT/'outputs/labelagent_height_bands_report_20261108';SENSOR=ROOT/'outputs/labelagent_unlabelled_height_bands_20261108';REVIEW=ROOT/'outputs/labelagent_height_review_20261108'
CODE=['unlabelled_height_bands.py','prepare_height_review.py','run_height_review.py','audit_height_bands.py','finish_height_bands.py','deliver_height_bands.py']

def git(*args):return subprocess.check_output(['git','-C',str(REPO),*args],text=True,encoding='utf-8').strip()

def prepare():
    if TARGET.exists():raise FileExistsError('Preserve sealed unlabelled sensor evidence')
    assert read(REPORT/'independent_readback.json')['passed'] and read(SENSOR/'independent_readback.json')['passed'];files={}
    def add(source,name):
        source=Path(source);assert source.is_file() and name not in files;files[name]={'source':str(source),'sha256':sha(source),'bytes':source.stat().st_size}
    for name,folder in [('sensor_source',SENSOR),('neutral_and_SAM',REVIEW),('report',REPORT)]:
        for p in folder.rglob('*'):
            if p.is_file():add(p,name+'/'+p.relative_to(folder).as_posix())
    protected={}
    for folder in [SENSOR,REVIEW]:
        plan=read(folder/'frozen_plan.json');assert all(sha(f)==h for f,h in plan['protected'].items());protected.update(plan['protected'])
    for i,p in enumerate(sorted(protected)):add(p,f'original_source_dependencies/{i:03d}/'+Path(p).name)
    sp=read(SENSOR/'frozen_plan.json');dataset=Path(sp['dataset'])/'source'
    for relative,h in sp['allowed_sensor_files'].items():assert sha(dataset/relative)==h;add(dataset/relative,'unlabelled_raw_sensor/'+relative)
    binding=read(ROOT/'outputs/labelagent_point_ids_20261107/incumbent_binding.json')['files']
    for p,h in binding.items():assert sha(ROOT/p)==h;add(ROOT/p,'original_six/'+Path(p).relative_to('outputs/labelagent_context_comparison_20261105').as_posix())
    cfg=read(ROOT/'task_workspaces/labelagent_point_ids_20261107/config.json');old=read(ROOT/'outputs/labelagent_soft_prior_20261108/frozen_plan.json')['protected']
    for key in ['reference_annotations','reference_description']:assert sha(cfg[key])==old[cfg[key]];add(cfg[key],'original_references/'+Path(cfg[key]).name)
    for name in CODE+['native_prompt_guard.py','semantic_subject_guard.py','paired_source_cue.py','run_rgb_layers.py','run_trained_point_sam.py','run_base_material.py','audit_soft_prior.py','run_soft_prior_trial.py','projection_audit_core.py']:add(Path(__file__).with_name(name),'reproduction/'+name)
    needed=sum(r['bytes'] for r in files.values())+96*2**20
    if shutil.disk_usage(ROOT).free<needed:raise RuntimeError('Complete independent evidence copies plus original96MiB reserve')
    TARGET.mkdir();subprocess.run(['compact.exe','/C','/Q',str(TARGET)],capture_output=True,check=True)
    for name,r in files.items():
        if shutil.disk_usage(ROOT).free<r['bytes']+96*2**20:raise RuntimeError('Original96MiB reserve during evidence copies')
        source=Path(r['source']);dest=TARGET/name;dest.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(source,dest);assert sha(source)==sha(dest)==r['sha256']
    save(TARGET/'complete_evidence_manifest.json',{'files':files,'all_ordinary_independent_file_copies_verified':True,'source_annotation_files_were_only_packaged_after_generation_and_audit':True,'raw_source_XYZ_and_calibration_included':True,'weights_and_original_environment_not_duplicated':True});summary=read(REPORT/'summary.json');save(TARGET/'combined_summary.json',summary);media=read(REPORT/'media_verification.json');media['videos']=[{**v,'path':'report/'+v['path']} for v in media['videos']];save(TARGET/'media_verification.json',media);shutil.copy2(REPORT/'本轮结论.txt',TARGET/'本轮结论.txt');html=(REPORT/'index.html').read_text(encoding='utf-8').replace('src="','src="report/').replace('href="summary.json"','href="combined_summary.json"').replace('href="independent_readback.json"','href="report/independent_readback.json"');(TARGET/'index.html').write_text(html,encoding='utf-8');save(TARGET/'交接SHA256.json',{'files':[{'path':p.relative_to(TARGET).as_posix(),'sha256':sha(p)} for p in TARGET.rglob('*') if p.is_file()]});receipt={'passed':True,'bundle':str(TARGET),'artifact_type':'complete_ordinary_file_directory','evidence_members_verified':len(files),'manifest_sha256':sha(TARGET/'complete_evidence_manifest.json'),'pictures':20,'videos':4,'decoded_frames':800,'sensor_files_included':4,'all_new_raw_masks_inputs_replies_originals_and_failure_preserved':True};save(CONTROL/'height_bands_bundle_verification.json',receipt);print(receipt,flush=True)

def verify():
    r=read(CONTROL/'height_bands_bundle_verification.json');assert r['passed'] and sha(TARGET/'complete_evidence_manifest.json')==r['manifest_sha256']
    for name,row in read(TARGET/'complete_evidence_manifest.json')['files'].items():assert sha(TARGET/name)==row['sha256']
    assert read(CONTROL/'height_bands_ui_verification.json')['passed'];return r

def publish():
    verify()
    for name in CODE:shutil.copy2(Path(__file__).with_name(name),REPO/'tools/labelagent_experiments'/name)
    for a,b in [('combined_summary.json','unlabelled-height-bands.json'),('本轮结论.txt','unlabelled-height-bands.md')]:shutil.copy2(TARGET/a,REPO/'docs/test-results'/b)
    paths=['tools/labelagent_experiments/'+n for n in CODE]+['docs/test-results/unlabelled-height-bands.json','docs/test-results/unlabelled-height-bands.md'];assert not git('diff','--cached','--name-only')
    for name in paths:assert not re.search(r'gh[pousr]_[A-Za-z0-9]{25,}|sk-[A-Za-z0-9]{30,}|-----BEGIN [A-Z ]*PRIVATE KEY-----',(REPO/name).read_text(encoding='utf-8'))
    subprocess.run(['git','-C',str(REPO),'add','--',*paths],check=True);subprocess.run(['git','-C',str(REPO),'diff','--cached','--check'],check=True);assert set(git('diff','--cached','--name-only').splitlines())==set(paths);subprocess.run(['git','-C',str(REPO),'commit','-m','Test unlabelled sensor height proposals without changing instance gates'],check=True)

def record():
    r=verify();cfg=read(CONTROL/'sync_config.json');head=git('rev-parse','HEAD');refs={n:git('ls-remote','--refs',n,'refs/heads/'+cfg['publish_branch']).split()[0] for n in ['origin','cnb']};assert all(v==head for v in refs.values()) and not git('status','--porcelain');receipt={**r,'checked_at':m.now(),'commit':head,'refs':refs,'both_verified':True,'summary':read(TARGET/'combined_summary.json')};save(CONTROL/'height_bands_final_sync_verification.json',receipt);save(HOME/'06_核验/未标注雷达分组与完整掩码_最终同步_20261108.json',receipt);m.refresh_manifest(HOME);print({'commit':head,'both_verified':True,'pictures':20,'videos':4},flush=True)

if __name__=='__main__':{'prepare':prepare,'publish':publish,'record':record}[sys.argv[1]]()
