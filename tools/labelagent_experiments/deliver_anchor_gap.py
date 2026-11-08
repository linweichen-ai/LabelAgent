"""Seal complete local evidence; publish only code and sanitized numeric facts."""
import re,shutil,subprocess,sys
from pathlib import Path
from run_rgb_layers import ROOT,read,save,sha
import maintain as m
import finish_anchor_gap_report as r
import run_fixed_grid_teacher as grid
C=ROOT/'task_workspaces/labelagent_handoff_control_20261104';HOME=ROOT/'deliverables/LabelAgent_实时交接';REPO=ROOT/'task_workspaces/labelagent_realtime_handoff_20261104';TARGET=HOME/'09_补充成果/跨裁剪锚点与新实例间隙及覆盖缺口_20261108'
CODE=[n+'.py' for n in 'run_anchor_bracket_prior run_anchor_image_extent_prior audit_anchor_priors run_source_gap_instances review_source_gap_instances audit_source_gap_instances review_structure_layout finish_anchor_gap_report audit_source_cohort_coverage append_cohort_coverage deliver_anchor_gap deliver_anchor_gap_D fixed_grid_cpu_engine run_fixed_grid_teacher'.split()]
def git(*a):return subprocess.check_output(['git','-C',str(REPO),*a],encoding='utf-8').strip()
def prepare():
    resume=len(sys.argv)>2 and sys.argv[2]=='resume'
    if TARGET.exists() and not resume:raise FileExistsError('Preserve full evidence packet; explicit verified resume only')
    assert read(r.OUT/'independent_readback.json')['passed'];files={};weights=[]
    def add(p,name):
        p=Path(p);assert p.is_file() and name not in files;files[name]={'source':str(p),'sha256':sha(p),'bytes':p.stat().st_size}
    folders={'anchor_bracket':r.ONE,'anchor_image_extent':r.TWO,'source_gap_instances':r.gap.OUT,'structure_layout':r.layout.OUT,'report':r.OUT,'next_grid_preparation_only':grid.OUT}
    for name,folder in folders.items():
        for p in folder.rglob('*'):
            if p.is_file():add(p,name+'/'+p.relative_to(folder).as_posix())
    protected={}
    for folder in folders.values():
        for plan in folder.rglob('frozen_plan.json'):
            for p,h in read(plan).get('protected',{}).items():
                assert p not in protected or protected[p]==h;protected[p]=h
    for i,(p,h) in enumerate(sorted(protected.items())):
        f=Path(p)
        if f.suffix in ['.pt','.safetensors']:
            weights.append({'path':str(f),'expected_sha256':h,'bytes':f.stat().st_size,'not_duplicated':True});continue
        assert sha(f)==h;add(f,f'original_dependencies/{i:04d}/'+f.name)
    weightfile=r.BASE/'labelagent_highres_context_cues_20261108/SAM_weights_identity.json';add(weightfile,'model_identity/SAM_weights_original_streamed_identity.json')
    binding=read(ROOT/'outputs/labelagent_point_ids_20261107/incumbent_binding.json')['files']
    for p,h in binding.items():
        f=ROOT/p;assert sha(f)==h;add(f,'original_six/'+f.relative_to(ROOT/'outputs/labelagent_context_comparison_20261105').as_posix())
    cfg=read(ROOT/'task_workspaces/labelagent_point_ids_20261107/config.json');old=read(ROOT/'outputs/labelagent_soft_prior_20261108/frozen_plan.json')['protected'];dataset=Path(cfg['dataset']);manifest=read(dataset/'import_manifest.json');stamp=str(manifest['timestamps_ns'][76])
    for k in ['reference_annotations','reference_description']:assert sha(cfg[k])==old[cfg[k]];add(cfg[k],'original_references/'+Path(cfg[k]).name)
    for relative in ['import_manifest.json','source/lidar/'+stamp+'.pcd','source/labels/'+stamp+'.json','source/calib/lidar2camera.yaml']+[f'source/calib/camera{i}_calib.yaml' for i in range(4)]:add(dataset/relative,'original_projection_inputs/'+relative)
    for p in Path(__file__).parent.glob('*.py'):add(p,'reproduction/'+p.name)
    for relative in ['tools/evaluate_projected_support.py','source/agent/edit_agent.py']:add(Path(cfg['source_snapshot'])/relative,'original_evaluator/'+relative)
    add(C/'anchor_gap_C_copy_interruption_20261108.json','engineering_failures/C_copy_interruption.json')
    required=sum(v['bytes'] for v in files.values())
    if not resume and shutil.disk_usage(TARGET.anchor).free<required+96*2**20:raise RuntimeError('Target full packet original96MiB reserve')
    if resume:assert read(C/'anchor_gap_C_copy_interruption_20261108.json')['all_original_sources_and_model_results_retained']
    TARGET.mkdir(exist_ok=resume);subprocess.run(['compact.exe','/C','/Q',str(TARGET)],check=True,capture_output=True)
    for name,row in files.items():
        dest=TARGET/name
        if dest.is_file():
            assert resume and sha(dest)==row['sha256'];continue
        if shutil.disk_usage(TARGET.anchor).free<row['bytes']+96*2**20:raise RuntimeError('Original96MiB copy reserve; preserve partial packet')
        dest.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(row['source'],dest);assert sha(dest)==row['sha256']
    save(TARGET/'complete_evidence_manifest.json',{'files':files,'weights_identity_not_weight_copy':weights,'all_actual_current_inputs_replies_native_arrays_originals_and_references_copied':True,'next_grid_is_preparation_only_no_new_inference_at_sealing':True,'runtime_and_model_weights_external_not_claimed_self_contained_environment':True});save(TARGET/'summary.json',read(r.OUT/'summary.json'));shutil.copy2(r.OUT/'本轮结论.txt',TARGET/'本轮结论.txt');media=read(r.OUT/'media_verification.json');media['videos']=[{**v,'path':'report/'+v['path']} for v in media['videos']];save(TARGET/'media_verification.json',media);html=(r.OUT/'index.html').read_text('utf-8').replace('src="','src="report/').replace('href="','href="report/');(TARGET/'index.html').write_text(html,encoding='utf-8');(TARGET/'完整证据说明.txt').write_text('本目录为本轮完整普通文件证据包，可直接整体备份。全部源输入、原答、原生掩码/数组、失败、原6结果与参考及投影所需帧点云和标定均有独立副本。模型权重及运行环境不重复复制，按标识账记录。next_grid_preparation_only仅为下一协议冻结准备，不是已完成推理。参考坐标仅在事后诊断使用，不参与源候选生成。原13/17与历史6/6保留（Agent本轮未重跑），未采用任何新候选。\n',encoding='utf-8');save(TARGET/'交接SHA256.json',{'files':[{'path':p.relative_to(TARGET).as_posix(),'sha256':sha(p),'bytes':p.stat().st_size} for p in TARGET.rglob('*') if p.is_file()]});receipt={'passed':True,'bundle':str(TARGET),'artifact_type':'complete_ordinary_file_directory','evidence_members_verified':len(files),'logical_bytes':required,'manifest_sha256':sha(TARGET/'complete_evidence_manifest.json'),'pictures':27,'videos':4,'decoded_frames':1080};save(C/'anchor_gap_bundle_verification.json',receipt);print(receipt,flush=True)
def verify():
    receipt=read(C/'anchor_gap_bundle_verification.json');assert receipt['passed'] and sha(TARGET/'complete_evidence_manifest.json')==receipt['manifest_sha256']
    for name,row in read(TARGET/'complete_evidence_manifest.json')['files'].items():assert sha(TARGET/name)==row['sha256']
    assert read(C/'anchor_gap_ui_verification.json')['passed'];return receipt
def publish():
    verify();paths=[]
    for name in CODE:
        f='tools/labelagent_experiments/'+name;shutil.copy2(Path(__file__).with_name(name),REPO/f);paths.append(f)
    for a,b in [('summary.json','anchor-gap-structure-layout.json'),('本轮结论.txt','anchor-gap-structure-layout.md')]:
        f='docs/test-results/'+b;shutil.copy2(TARGET/a,REPO/f);paths.append(f)
    coverage=read(r.OUT/'source_cohort_coverage.json');f='docs/test-results/source-cohort-coverage-witness.json';save(REPO/f,{'status':'complete_post_generation_cohort_coverage_audit','scope':coverage['scope'],'cases':[{k:row[k] for k in ['case','object_id','visible_reference_points','points_inside_any_recent_six_teacher_ROIs']} for row in coverage['cases']],'coordinates_not_published':True,'reference_coordinates_not_generation_inputs':True,'original_full_frame_baseline_not_invalidated':True});paths.append(f)
    f='docs/test-results/fixed-grid-teacher-plan.json';save(REPO/f,{'status':'prepared_not_executed','real_views':10,'synthetic_control_views':2,'trained_checkpoints':2,'planned_CPU_forwards':24,'new_model_calls':0,'new_training':False,'GPU_calls':0,'source_grid':'whole frame plus fixed four quadrants for both original cameras; real context halo and explicit gray physical-frame padding','reference_inputs_for_grid_or_sampling':False,'all_original_final_criteria_unchanged':True,'native_outputs_must_not_be_cut_to_fit_observed_frame':True});paths.append(f)
    assert not git('diff','--cached','--name-only')
    for f in paths:assert not re.search(r'gh[pousr]_[A-Za-z0-9]{25,}|sk-[A-Za-z0-9]{30,}|-----BEGIN [A-Z ]*PRIVATE KEY-----',(REPO/f).read_text('utf-8'))
    subprocess.run(['git','-C',str(REPO),'add','--',*paths],check=True);subprocess.run(['git','-C',str(REPO),'diff','--cached','--check'],check=True);assert set(git('diff','--cached','--name-only').splitlines())==set(paths);subprocess.run(['git','-C',str(REPO),'commit','-m','Audit source gap instances, strict structure layouts and omitted camera ROI coverage'],check=True)
def record():
    receipt=verify();cfg=read(C/'sync_config.json');head=git('rev-parse','HEAD');refs={name:git('ls-remote','--refs',name,'refs/heads/'+cfg['publish_branch']).split()[0] for name in ['origin','cnb']};assert all(v==head for v in refs.values()) and not git('status','--porcelain');receipt.update(checked_at=m.now(),commit=head,refs=refs,both_verified=True,summary=read(TARGET/'summary.json'));save(C/'anchor_gap_final_sync_verification.json',receipt);save(HOME/'06_核验/跨裁剪锚点与间隙及覆盖缺口_最终同步_20261108.json',receipt);m.refresh_manifest(HOME);print({'commit':head,'both_verified':True},flush=True)
if __name__=='__main__':{'prepare':prepare,'publish':publish,'record':record}[sys.argv[1]]()
