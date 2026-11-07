"""Seal current private evidence on D and publish only code and numeric results."""
import hashlib,json,re,shutil,subprocess,sys,zipfile
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]
C=ROOT/'task_workspaces/labelagent_handoff_control_20261104'
HOME=ROOT/'deliverables/LabelAgent_实时交接'
REPO=ROOT/'task_workspaces/labelagent_realtime_handoff_20261104'
BASE=Path('D:/AI/Codex/Home/visualizations/2026/11/04/01a1a56c-0d6f-7d61-8326-d3bbd344d2df')
REPORT=BASE/'labelagent_context_graph_report_20261108'
TARGET=HOME/'09_补充成果/源上下文分辨率与实例冲突图_20261108'
FULL=BASE/'LabelAgent_完整交接_上下文与实例冲突图_20261108'
CODE='run_aspect_context_teacher audit_aspect_context_geometry prepare_aspect_context_cues run_aspect_context_cues audit_aspect_context_cues run_aspect_context_audit_D review_aspect_semantic_extremes highres_context_cpu_engine run_highres_context_teacher prepare_highres_context_cues run_highres_context_cues audit_highres_context_cues prepare_mask_graph prepare_mask_graph_collision run_mask_graph_collision prepare_mask_graph_SAM_adapter audit_mask_graph_collision prepare_mask_graph_core run_mask_graph_core prepare_mask_graph_core_SAM_adapter audit_mask_graph_core run_native_constraint_repair audit_native_constraint_repair experiment_report_media finish_context_graph deliver_context_graph'.split()
def read(p):return json.loads(Path(p).read_text('utf-8-sig'))
def save(p,x):
    p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(x,ensure_ascii=False,indent=2),encoding='utf-8')
def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(4*2**20),b''):h.update(b)
    return h.hexdigest()
def git(*a):return subprocess.check_output(['git','-C',str(REPO),*a],text=True,encoding='utf-8').strip()
def prepare():
    assert read(REPORT/'independent_readback.json')['passed']
    if FULL.exists() or TARGET.exists():raise FileExistsError('Preserve prepared/interrupted handoff')
    folders={n:BASE/n for n in ['labelagent_aspect_context_teacher_20261108','labelagent_aspect_context_audit_20261108','labelagent_aspect_semantic_extremes_20261108','labelagent_highres_context_teacher_20261108','labelagent_highres_context_cues_20261108','labelagent_mask_graph_20261108','labelagent_mask_graph_collision_20261108','labelagent_mask_graph_core_20261108','labelagent_native_constraint_repair_20261108','labelagent_context_graph_report_20261108']}
    folders['labelagent_aspect_context_cues_20261108']=ROOT/'outputs/labelagent_aspect_context_cues_20261108'
    files={};digests={};weights=[]
    def add(p,name):
        p=Path(p);assert p.is_file() and name not in files
        h=sha(p);files[name]={'source':str(p),'sha256':h,'bytes':p.stat().st_size,'member':digests.setdefault(h,name)}
    for name,folder in folders.items():
        for p in sorted(folder.rglob('*')):
            if p.is_file():add(p,'experiments/'+name+'/'+p.relative_to(folder).as_posix())
    protected={}
    for folder in folders.values():
        for plan in folder.rglob('frozen_plan.json'):
            for p,h in read(plan).get('protected',{}).items():
                assert p not in protected or protected[p]==h
                protected[p]=h
    for i,(p,h) in enumerate(sorted(protected.items())):
        f=Path(p)
        if f.suffix in ['.safetensors','.pt']:
            weights.append({'path':p,'expected_sha256':h,'bytes':f.stat().st_size,'not_duplicated_in_archive':True});continue
        assert sha(f)==h;add(f,f'protected_dependencies/{i:04d}/'+f.name)
    for p,h in read(ROOT/'outputs/labelagent_point_ids_20261107/incumbent_binding.json')['files'].items():
        f=ROOT/p;assert sha(f)==h;add(f,'original_six/'+f.relative_to(ROOT/'outputs/labelagent_context_comparison_20261105').as_posix())
    cfg=read(ROOT/'task_workspaces/labelagent_point_ids_20261107/config.json');old=read(ROOT/'outputs/labelagent_soft_prior_20261108/frozen_plan.json')['protected']
    for k in ['reference_annotations','reference_description']:
        f=Path(cfg[k]);assert sha(f)==old[str(f)];add(f,'original_references/'+f.name)
    for f in Path(__file__).parent.glob('*.py'):add(f,'reproduction/'+f.name)
    for name in ['owned_sealed_media_dedup_completed_20261108.json','context_graph_export_worker_switch_20261108.json','aspect_context_reviewer_warmup_20261108.json']:
        if (C/name).is_file():add(C/name,'engineering_receipts/'+name)
    FULL.mkdir()
    manifest={'files':files,'weights_identity_not_weights_copy':weights,'includes_all_current_24_MF_query_captures_58_native_SAM_hypotheses_and_model_replies':True,'historical_training_data_and_model_weights_are_external_dependencies':True,'prepared_only_graph_v1_not_an_executed_experiment':True,'C_partial_audit_failure_kept_and_D_completed_audit_distinguished':True}
    save(FULL/'complete_evidence_manifest.json',manifest)
    archive=FULL/'完整原始证据.zip'
    unique=[(n,r) for n,r in files.items() if n==r['member']]
    if shutil.disk_usage(FULL).free<sum(r['bytes'] for _,r in unique)+512*2**20:raise RuntimeError('D output capacity guard')
    with zipfile.ZipFile(archive,'x',compression=zipfile.ZIP_DEFLATED,compresslevel=6,allowZip64=True) as z:
        for name,row in unique:
            assert sha(row['source'])==row['sha256'];z.write(row['source'],name)
        z.write(FULL/'complete_evidence_manifest.json','complete_evidence_manifest.json')
    verified=0
    with zipfile.ZipFile(archive) as z:
        assert set(z.namelist())=={n for n,_ in unique}|{'complete_evidence_manifest.json'}
        for name,row in unique:
            h=hashlib.sha256()
            with z.open(name) as f:
                for b in iter(lambda:f.read(4*2**20),b''):h.update(b)
            assert h.hexdigest()==row['sha256'];verified+=1
    shutil.copytree(REPORT,FULL/'report')
    text='完整原始证据在本目录的完整原始证据.zip；恢复所有逻辑路径请按 complete_evidence_manifest.json 的 member 字段展开同内容别名。\n模型权重和历史训练数据不重复装入包，记录原路径、大小及哈希。当前24次Mask2Former、46次VLM、30次SAM的输入、原答与原生掩码均保留。\n历史6/6 Agent未重跑；本轮7/7仅为原生点约束通过，仍剩4个漏检。\n'
    (FULL/'备份说明.txt').write_text(text,encoding='utf-8')
    receipt={'passed':True,'full_evidence_directory':str(FULL),'full_archive':str(archive),'archive_sha256':sha(archive),'archive_bytes':archive.stat().st_size,'logical_files':len(files),'unique_content_members_verified':verified,'manifest_sha256':sha(FULL/'complete_evidence_manifest.json'),'pictures':80,'videos':6,'decoded_frames':3200,'C_handoff_contains_media_and_explicit_D_backup_location':True,'C_handoff_alone_does_not_contain_full_private_archive':True}
    save(FULL/'archive_verification.json',receipt)
    if shutil.disk_usage(ROOT).free<sum(p.stat().st_size for p in REPORT.rglob('*') if p.is_file())+96*2**20:raise RuntimeError('C report96MiB guard')
    shutil.copytree(REPORT,TARGET)
    save(TARGET/'完整证据位置与核验.json',receipt)
    (TARGET/'完整证据备份说明.txt').write_text('C盘空间不足：本目录有全部80张图及6段视频。完整原始证据另存于：\n'+str(FULL)+'\n请备份上述D盘整个目录；仅备份C盘交接文件夹不包含本轮全部原始证据。\n'+text,encoding='utf-8')
    html=(TARGET/'index.html').read_text('utf-8');html=html.replace('</h1>','</h1><p style="padding:16px;background:#fff1b8">完整原始证据另存 D 盘；请同时备份 <a href="完整证据备份说明.txt">完整证据备份说明</a> 中的目录。</p>',1);assert '完整原始证据另存 D 盘' in html;(TARGET/'index.html').write_text(html,encoding='utf-8')
    save(TARGET/'交接SHA256.json',{'files':[{'path':p.relative_to(TARGET).as_posix(),'sha256':sha(p),'bytes':p.stat().st_size} for p in TARGET.rglob('*') if p.is_file()]})
    save(C/'context_graph_bundle_verification.json',receipt)
    print({k:v for k,v in receipt.items() if 'path' not in k},flush=True)
def verify():
    r=read(C/'context_graph_bundle_verification.json');assert r['passed'] and sha(r['full_archive'])==r['archive_sha256']
    assert sha(FULL/'complete_evidence_manifest.json')==r['manifest_sha256']
    for row in read(TARGET/'交接SHA256.json')['files']:assert sha(TARGET/row['path'])==row['sha256']
    assert read(C/'context_graph_ui_verification.json')['passed'];return r
def publish():
    verify();paths=[]
    for name in CODE:
        f='tools/labelagent_experiments/'+name+'.py';shutil.copy2(Path(__file__).with_name(name+'.py'),REPO/f);paths.append(f)
    summary=read(REPORT/'summary.json');f='docs/test-results/context-resolution-mask-graph.json';save(REPO/f,summary);paths.append(f)
    f='docs/test-results/context-resolution-mask-graph.md';shutil.copy2(REPORT/'本轮结论.txt',REPO/f);paths.append(f)
    f='docs/test-results/native-aspect-context-plan.json';save(REPO/f,{'status':'complete_rejected_no_new_accepted_instance','new_CPU_Mask2Former_forwards':12,'new_VLM_calls':14,'new_SAM_calls':8,'retained_support':'13/17','historical_Agent':'6/6_not_rerun','followup':'context-resolution-mask-graph.json'});paths.append(f)
    assert not git('diff','--cached','--name-only')
    for f in paths:assert not re.search(r'gh[pousr]_[A-Za-z0-9]{25,}|sk-[A-Za-z0-9]{30,}|-----BEGIN [A-Z ]*PRIVATE KEY-----',(REPO/f).read_text('utf-8'))
    subprocess.run(['git','-C',str(REPO),'add','--',*paths],check=True);subprocess.run(['git','-C',str(REPO),'diff','--cached','--check'],check=True)
    staged=set(git('diff','--cached','--name-only').splitlines());assert staged and staged<=set(paths)
    subprocess.run(['git','-C',str(REPO),'commit','-m','Audit trained context resolution and source mask conflicts with bounded native repair'],check=True)
def record():
    import maintain as m
    r=verify();cfg=read(C/'sync_config.json');head=git('rev-parse','HEAD');refs={n:git('ls-remote','--refs',n,'refs/heads/'+cfg['publish_branch']).split()[0] for n in ['origin','cnb']}
    assert all(v==head for v in refs.values()) and not git('status','--porcelain')
    r.update({'checked_at':m.now(),'commit':head,'refs':refs,'both_verified':True,'summary':read(REPORT/'summary.json')});save(C/'context_graph_final_sync_verification.json',r);save(HOME/'06_核验/上下文分辨率与实例冲突图_最终同步_20261108.json',r);m.refresh_manifest(HOME);print({'commit':head,'both_verified':True},flush=True)
if __name__=='__main__':{'prepare':prepare,'publish':publish,'record':record}[sys.argv[1]]()
