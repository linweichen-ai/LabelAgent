"""Local complete evidence ZIP; public code and sanitized counts only."""
import hashlib,json,re,shutil,subprocess,sys,zipfile,zlib
from pathlib import Path
from run_rgb_layers import ROOT,read,save,sha
import maintain as m
C=ROOT/'task_workspaces/labelagent_handoff_control_20261104';HOME=ROOT/'deliverables/LabelAgent_实时交接';REPO=ROOT/'task_workspaces/labelagent_realtime_handoff_20261104'
REPORT=ROOT/'outputs/labelagent_zoom_registry_report_20261108';TARGET=HOME/'09_补充成果/四套训练权重与完整原生掩码_20261108';ARCHIVE=TARGET/'LabelAgent_四套训练权重与完整原生掩码_完整证据_20261108.zip'
FOLDERS=['labelagent_source_layer_boxes_20261108','labelagent_source_surface_boxes_20261108','labelagent_zoom_teacher_20261108','labelagent_registry_zoom_teacher_20261108','labelagent_zoom_cues_20261108','labelagent_registry_zoom_cues_20261108','labelagent_registry_zoom_pool_20261108','labelagent_zoom_registry_report_20261108']
CODE=['source_layer_box_policy.py','run_source_layer_boxes.py','audit_source_layer_boxes.py','run_source_surface_boxes.py','audit_source_surface_boxes.py','run_zoom_teacher.py','run_registry_zoom_teacher.py','prepare_zoom_cues.py','run_zoom_cues.py','prepare_zoom_SAM_adapter.py','audit_zoom_candidates.py','prepare_registry_zoom_cues.py','prepare_registry_zoom_pool.py','run_registry_zoom_pool.py','audit_registry_zoom_pool.py','finish_zoom_registry.py','deliver_zoom_registry.py']
HELPERS=['m2f_query_replay.py','native_prompt_guard.py','semantic_subject_guard.py','paired_source_cue.py','run_rgb_layers.py','run_pallet_chain.py','run_trained_point_sam.py','run_base_material.py','audit_soft_prior.py','run_soft_prior_trial.py','projection_audit_core.py']

def git(*a):return subprocess.check_output(['git','-C',str(REPO),*a],text=True,encoding='utf-8').strip()

def prepare(resume=False,plan_only=False):
    if TARGET.exists() and not resume:raise FileExistsError('Preserve sealed trained checkpoint evidence')
    if resume:assert (C/'zoom_registry_archive_runtime_interruption.json').exists() and not (C/'zoom_registry_bundle_verification.json').exists()
    assert read(REPORT/'independent_readback.json')['passed'];summary=read(REPORT/'summary.json');files={};dependencies={};weights=[]
    sources={};aliases={}
    def add(p,name):
        p=Path(p);assert p.is_file() and name not in files;files[name]={'source':str(p),'sha256':sha(p),'bytes':p.stat().st_size}
        resolved=files[name]['sha256']
        if resolved in sources:
            aliases[name]=sources[resolved];del files[name]
        else:sources[resolved]=name
    for name in FOLDERS:
        folder=ROOT/'outputs'/name
        for p in folder.rglob('*'):
            if p.is_file():add(p,name+'/'+p.relative_to(folder).as_posix())
        for frozen in [folder/'frozen_plan.json',folder/'SAM/frozen_plan.json']:
            if not frozen.exists():continue
            for p,h in read(frozen).get('protected',{}).items():
                if Path(p).name in ['model.safetensors','sam3.pt']:weights.append({'source':p,'sha256':h,'bytes':Path(p).stat().st_size});continue
                assert sha(p)==h;dependencies[p]=h
        p=folder/'frozen_plan.json'
        if p.exists():
            for row in read(p).get('rows',read(p).get('views',[])):
                for key in ['input','source_image','prior']:
                    if row.get(key):dependencies[row[key]]=sha(row[key])
    for i,p in enumerate(sorted(dependencies)):add(p,f'original_dependencies/{i:03d}/'+Path(p).name)
    old=ROOT/'outputs/labelagent_two_sided_layers_20261108/SAM/generation_completed.json'
    add(old,'previous_five_point/generation_completed.json')
    for row in read(old)['rows']:
        for name in ['complete_full_mask.npy','all_arrays.npz','metadata.json','response.json']:add(Path(row['folder'])/name,f'previous_five_point/cue{row["source_partition_index"]:02d}/'+name)
    binding=read(ROOT/'outputs/labelagent_point_ids_20261107/incumbent_binding.json')['files']
    for p,h in binding.items():assert sha(ROOT/p)==h;add(ROOT/p,'original_six/'+Path(p).relative_to('outputs/labelagent_context_comparison_20261105').as_posix())
    cfg=read(ROOT/'task_workspaces/labelagent_point_ids_20261107/config.json');oldprotected=read(ROOT/'outputs/labelagent_soft_prior_20261108/frozen_plan.json')['protected']
    for key in ['reference_annotations','reference_description']:assert sha(cfg[key])==oldprotected[cfg[key]];add(cfg[key],'original_references/'+Path(cfg[key]).name)
    for name in CODE+HELPERS:add(Path(__file__).with_name(name),'reproduction/'+name)
    for name in ['registry_zoom_reviewer_warmup_20261108.json','manager_log_volume_recovery_20261108.json']:add(C/name,'runtime_receipts/'+name)
    oldzip=read(C/'source_context_trained_bundle_verification.json');assert sha(oldzip['archive'])==oldzip['archive_sha256']
    for name in ['zoom_registry_archive_preflight_v1.json','deliver_zoom_registry_raw_size_preflight_v1.py','zoom_registry_archive_runtime_interruption.json','deliver_zoom_registry_exact_preflight_v2.py']:add(C/name,'engineering_failures/'+name)
    save(C/'zoom_registry_archive_member_plan.json',{'files':files,'same_bytes_member_aliases':aliases})
    if plan_only:
        print({'members':len(files),'aliases':len(aliases),'raw_bytes':sum(r['bytes'] for r in files.values())},flush=True);return
    # Measure the exact ZIP DEFLATE stream in bounded RAM before writing.
    # Full-size sparse bitmaps are retained, rather than removed to fit storage.
    compressed=0
    for r in files.values():
        stream=zlib.compressobj(5,zlib.DEFLATED,-15);size=0
        with Path(r['source']).open('rb') as source:
            for block in iter(lambda:source.read(4*2**20),b''):size+=len(stream.compress(block))
        size+=len(stream.flush());r['measured_deflate_bytes']=size;compressed+=size
    media_copy=0 if resume else sum(p.stat().st_size for p in REPORT.rglob('*') if p.is_file());free=shutil.disk_usage(ARCHIVE.parent if ARCHIVE.parent.exists() else ROOT).free;needed=compressed+media_copy+8*2**20+96*2**20
    save(C/'zoom_registry_archive_exact_size_preflight.json',{'raw_bytes':sum(r['bytes'] for r in files.values()),'measured_deflate_bytes':compressed,'display_copy_bytes':media_copy,'metadata_and_zip_overhead_reserve':8*2**20,'unchanged_free_space_reserve_bytes':96*2**20,'available_bytes':free,'needed_bytes':needed,'passed':free>=needed})
    if free<needed:raise RuntimeError(f'Exact archive plus unchanged96MiB reserve needs{needed}, available{free}')
    TARGET.mkdir(exist_ok=resume);subprocess.run(['compact.exe','/C','/Q',str(TARGET)],capture_output=True,check=True)
    for p in REPORT.rglob('*'):
        if p.is_file():
            dest=TARGET/'report'/p.relative_to(REPORT);dest.parent.mkdir(parents=True,exist_ok=True)
            if dest.exists():assert sha(p)==sha(dest)
            else:shutil.copy2(p,dest)
            assert sha(p)==sha(dest)
    save(TARGET/'combined_summary.json',summary);shutil.copy2(REPORT/'本轮结论.txt',TARGET/'本轮结论.txt');media=read(REPORT/'media_verification.json');media['videos']=[{**v,'path':'report/'+v['path']} for v in media['videos']];save(TARGET/'media_verification.json',media)
    save(TARGET/'原始数据依赖.json',{'weights_not_duplicated':weights,'SAM_streamed_weights_identity':read(ROOT/'outputs/labelagent_registry_zoom_pool_20261108/SAM_weights_identity.json'),'previous_full_source_queries_inputs_archive':oldzip['archive'],'previous_archive_sha256':oldzip['archive_sha256'],'all_new_raw_queries_masks_inputs_replies_and_original_six_in_this_archive':True,'original_PCD_calibration_older_full_queries_follow_previous_archive_and_source_paths':True,'not_standalone_environment':True,'private_raw_data_only_local_publication_is_sanitized':True})
    html=(REPORT/'index.html').read_text(encoding='utf-8');html=html.replace('src="','src="report/').replace('href="summary.json"','href="combined_summary.json"').replace('href="independent_readback.json"','href="report/independent_readback.json"');(TARGET/'index.html').write_text(html,encoding='utf-8')
    for p in TARGET.rglob('*'):
        if p.is_file() and p.suffix in ['.json','.txt','.html']:add(p,'handoff/'+p.relative_to(TARGET).as_posix())
    with zipfile.ZipFile(ARCHIVE,'w',zipfile.ZIP_DEFLATED,compresslevel=5) as z:
        for name,r in files.items():
            if shutil.disk_usage(ARCHIVE.parent).free<96*2**20:raise RuntimeError('Original96MiB runtime reserve; preserve partial ZIP and originals')
            assert sha(r['source'])==r['sha256'];z.write(r['source'],name)
        z.writestr('archive_manifest.json',json.dumps({'files':files,'same_source_member_aliases':aliases},ensure_ascii=False,indent=2).encode())
    with zipfile.ZipFile(ARCHIVE) as z:
        assert set(z.namelist())==set(files)|{'archive_manifest.json'}
        for name,r in files.items():assert hashlib.sha256(z.read(name)).hexdigest()==r['sha256']
    local=[{'path':p.relative_to(TARGET).as_posix(),'sha256':sha(p)} for p in TARGET.rglob('*') if p.is_file()];save(TARGET/'交接SHA256.json',{'files':local});save(C/'zoom_registry_bundle_verification.json',{'passed':True,'bundle':str(TARGET),'archive':str(ARCHIVE),'archive_bytes':ARCHIVE.stat().st_size,'archive_sha256':sha(ARCHIVE),'members_verified':len(files)+1,'pictures':summary['pictures'],'videos':summary['videos'],'decoded_frames':media['fully_decoded_frames'],'full_new_queries_native_masks_and_originals_readback':True});print({'archive_bytes':ARCHIVE.stat().st_size,'members':len(files)+1,'pictures':summary['pictures'],'videos':summary['videos']},flush=True)

def publish():
    assert read(C/'zoom_registry_ui_verification.json')['passed'];assert sha(ARCHIVE)==read(C/'zoom_registry_bundle_verification.json')['archive_sha256']
    for name in CODE:shutil.copy2(Path(__file__).with_name(name),REPO/'tools/labelagent_experiments'/name)
    for a,b in [('combined_summary.json','four-trained-checkpoints-native-masks.json'),('本轮结论.txt','four-trained-checkpoints-native-masks.md')]:shutil.copy2(TARGET/a,REPO/'docs/test-results'/b)

def commit():
    paths=['tools/labelagent_experiments/'+n for n in CODE]+['docs/test-results/four-trained-checkpoints-native-masks.json','docs/test-results/four-trained-checkpoints-native-masks.md'];assert not git('diff','--cached','--name-only')
    for p in paths:assert not re.search(r'gh[pousr]_[A-Za-z0-9]{25,}|sk-[A-Za-z0-9]{30,}|-----BEGIN .*PRIVATE KEY',(REPO/p).read_text(encoding='utf-8'))
    subprocess.run(['git','-C',str(REPO),'add','--',*paths],check=True);subprocess.run(['git','-C',str(REPO),'diff','--cached','--check'],check=True);assert set(git('diff','--cached','--name-only').splitlines())==set(paths);subprocess.run(['git','-C',str(REPO),'commit','-m','Compare four trained teachers and preserve full native mask alternatives'],check=True)

def record():
    cfg=read(C/'sync_config.json');head=git('rev-parse','HEAD');refs={n:git('ls-remote','--refs',n,'refs/heads/'+cfg['publish_branch']).split()[0] for n in ['origin','cnb']};assert all(v==head for v in refs.values()) and not git('status','--porcelain');v=read(C/'zoom_registry_bundle_verification.json');assert sha(ARCHIVE)==v['archive_sha256'];receipt={**v,'checked_at':m.now(),'commit':head,'refs':refs,'both_verified':True,'summary':read(TARGET/'combined_summary.json')};save(C/'zoom_registry_final_sync_verification.json',receipt);save(HOME/'06_核验/四套训练权重与完整原生掩码_最终同步_20261108.json',receipt);m.refresh_manifest(HOME);print({'commit':head,'both_verified':True,'pictures':35,'videos':5},flush=True)

if __name__=='__main__':{'prepare':prepare,'publish':publish,'commit':commit,'record':record}[sys.argv[1]]()
