"""Append complete private evidence; publish code and sanitized numbers only."""
import re,shutil,subprocess,sys
from pathlib import Path
from prepare_fixed_grid_followup import ROOT,BASE,OUT
from run_rgb_layers import read,save
from deliver_context_graph import sha
from finish_registered_grid_followup import REPORT
C=ROOT/'task_workspaces/labelagent_handoff_control_20261104';HOME=ROOT/'deliverables/LabelAgent_实时交接';REPO=ROOT/'task_workspaces/labelagent_realtime_handoff_20261104';FULL=BASE/'LabelAgent_完整交接_锚点间隙与覆盖缺口_20261108';APPEND=FULL/'grid';DISPLAY=HOME/'09_补充成果/跨裁剪锚点与新实例间隙及覆盖缺口_20261108';LEAF=DISPLAY/'registered_grid_followup_report'
CODE='prepare_fixed_grid_followup run_fixed_grid_followup audit_registered_fixed_grid review_fixed_grid_instances audit_fixed_grid_candidates prepare_single_teacher_grid run_single_teacher_grid finish_registered_grid_followup check_local_report_chrome deliver_registered_grid_followup'.split()

def git(*args):return subprocess.check_output(['git','-C',str(REPO),*args],text=True,encoding='utf-8').strip()
def extended(file):return '\\\\?\\'+str(Path(file).resolve())

def append():
    resume=len(sys.argv)>2 and sys.argv[2]=='resume'
    if LEAF.exists() or (APPEND/'append_manifest.json').exists() or (APPEND.exists() and not resume):raise FileExistsError('Preserve sealed/interrupted append; explicit verified resume only')
    assert read(REPORT/'independent_readback.json')['passed'] and read(C/'registered_grid_followup_ui_verification.json')['passed'];files={}
    folders={'cpu':BASE/'labelagent_registered_fixed_grid_20261108','pair':OUT,'report':REPORT,'next':BASE/'labelagent_single_teacher_grid_20261108'}
    for prefix,folder in folders.items():
        for file in folder.rglob('*'):
            if file.is_file():files[prefix+'/'+file.relative_to(folder).as_posix()]={'source':str(file),'sha256':sha(file),'bytes':file.stat().st_size}
    for name in CODE:
        file=Path(__file__).with_name(name+'.py');files['reproduction/'+file.name]={'source':str(file),'sha256':sha(file),'bytes':file.stat().st_size}
    for name in ['registered_grid_followup_ui_verification.json']:
        file=C/name;files['verification/'+name]={'source':str(file),'sha256':sha(file),'bytes':file.stat().st_size}
    assert shutil.disk_usage(FULL).free>sum(r['bytes'] for r in files.values())+96*2**20;APPEND.mkdir(exist_ok=resume)
    if resume:
        checked=[]
        for file in APPEND.rglob('*'):
            if file.is_file():
                name=file.relative_to(APPEND).as_posix();assert name in files and sha(file)==files[name]['sha256'];checked.append(name)
        save(C/'registered_grid_followup_partial_resume_readback.json',{'all_partial_files_exact_against_unchanged_sources':True,'verified_members':len(checked),'Windows_CopyFile2_uses_explicit_extended_absolute_paths':True,'no_original_files_deleted':True})
    for name,row in files.items():
        file=APPEND/name;file.parent.mkdir(parents=True,exist_ok=True)
        if not file.exists():shutil.copy2(extended(row['source']),extended(file))
        assert sha(file)==row['sha256']
    manifest={'files':files,'all_new_raw_queries_native_arrays_unmarked_inputs_raw_replies_masks_and_evaluation_adapters_preserved':True,'next_preparation_copy_is_historical_zero_call_snapshot_not_runtime':True,'models_and_runtime_are_external_dependencies_not_duplicated':True,'earlier_common_original_photos_PCD_calibration_references_and_annotations_still_in_parent_complete_manifest':True};save(APPEND/'append_manifest.json',manifest)
    assert shutil.disk_usage('C:/').free>sum(p.stat().st_size for p in REPORT.rglob('*') if p.is_file())+96*2**20;shutil.copytree(REPORT,LEAF)
    link='<p><a href="registered_grid_followup_report/index.html">新增：R5/R15同图对照与14张完整原生掩码，16图4段回放</a></p>';index=DISPLAY/'index.html';text=index.read_text('utf-8');assert link not in text;text=text.replace('</h1>','</h1>'+link,1).replace('本轮35图6视频（三份报告）','本轮51图10视频（四份报告）').replace('C盘保留全部27图4视频','C盘展示累计51图10视频');index.write_text(text,encoding='utf-8');index=FULL/'index.html';text=index.read_text('utf-8');index.write_text(text.replace('</h1>','</h1><p><a href="grid/report/index.html">新增：注册权重网格与原生配对掩码全部证据</a></p>',1),encoding='utf-8');receipt={'passed':True,'append_directory':str(APPEND),'manifest_sha256':sha(APPEND/'append_manifest.json'),'members':len(files),'logical_bytes':sum(r['bytes'] for r in files.values()),'new_pictures':16,'new_videos':4,'decoded_new_replay_frames':640,'C_display_new_report':str(LEAF),'C_display_phase_total_pictures':51,'C_display_phase_total_videos':10,'original_annotations_not_changed':True};save(C/'registered_grid_followup_append_verification.json',receipt);save(DISPLAY/'注册权重网格与原生配对_新增结果.json',read(REPORT/'summary.json'));backup=HOME/'00_本轮完整证据备份位置.txt';old=backup.read_text('utf-8');backup.write_text('新增注册权重网格与14张原生配对结果：本系列累计51张对比图、10段保存阶段回放。\nC盘仅展示图片视频及摘要；完整普通文件证据必须同时备份D盘整个目录：\n'+str(FULL)+'\n本次追加目录：'+str(APPEND)+'\n此前上下文与实例冲突图完整包也须保留。模型权重和运行环境是外部依赖，未冒称独立环境镜像。下一单模型提案目录在此包中只是0调用准备快照。\n\n此前历史备份记录：\n'+old,encoding='utf-8')
    for folder in [DISPLAY,FULL]:save(folder/'交接SHA256.json',{'files':[{'path':p.relative_to(folder).as_posix(),'sha256':sha(p),'bytes':p.stat().st_size} for p in folder.rglob('*') if p.is_file() and p.name!='交接SHA256.json']})
    print(receipt,flush=True)

def publish():
    receipt=read(C/'registered_grid_followup_append_verification.json');assert receipt['passed'] and sha(APPEND/'append_manifest.json')==receipt['manifest_sha256'];assert not git('diff','--cached','--name-only');assert not git('status','--porcelain');paths=[]
    for name in CODE:
        dest='tools/labelagent_experiments/'+name+'.py';shutil.copy2(Path(__file__).with_name(name+'.py'),REPO/dest);paths.append(dest)
    summary=read(REPORT/'summary.json')
    for name,value in [('registered-grid-native-followup.json',summary),('registered-fixed-grid-plan.json',{'status':'completed_24_CPU_forwards_not_candidate_acceptance','new_CPU':24,'real':20,'synthetic_controls':4,'new_training':False,'all48_saved_queries_original_postprocessor_pixels_exact':True,'R15_not_deployed':True,'new_accepted_instances':0}),('single-teacher-grid-plan.json',{'status':'prepared_not_executed_at_publication','source_points':21,'plain_controls':2,'new_calls_at_snapshot':0,'proposal_rule_changes':['up_to3_stock_PAL_per_registered_model_view','source_model_proposal_not_vetoed_by_second_model_before_neutral_review'],'all_final_acceptance_gates_unchanged':True,'no_output_mask_cuts':True,'max_unique_SAM_point_candidates':16,'max_paired_SAM_calls':32,'Agent_round_limit':3,'new_Agent_runs':0,'original_annotations_preserved':True,'not_an_independent_validation':True})]:
        dest='docs/test-results/'+name;save(REPO/dest,value);paths.append(dest)
    dest='docs/test-results/registered-grid-native-followup.md';text='R15原预测在左侧两例覆盖11/11、4/6个旧稀疏参考点，但固定最高分区域与双模型共识提案没有将这些区域送去复核。覆盖不等于独立实例找回。\n\n完成同图同配方24次CPU前向、18次源点及控制复核、14次完整原生SAM及23次完整语义新调用。全部48份保存输出与原后处理逐像素一致。14张SAM全部原生有效且留白前景0，但面积全部超出剩余预算；蓝色候选仍粘连多个实例，14候选完整语义均未通过。\n\n保留原13/17与历史6/6（未重跑Agent），不裁掩码、不放宽门槛。新增16图4段保存阶段回放，系列累计51图10回放；媒体和完整原答/数组仅在私人交接目录。代码和脱敏数值发布到双端同一分支。下一单模型提案分支此提交时仅准备，验收不变。\n';(REPO/dest).write_text(text,encoding='utf-8');paths.append(dest)
    for dest in paths:assert not re.search(r'gh[pousr]_[A-Za-z0-9]{25,}|sk-[A-Za-z0-9]{30,}|-----BEGIN [A-Z ]*PRIVATE KEY-----',(REPO/dest).read_text('utf-8'))
    subprocess.run(['git','-C',str(REPO),'add','--',*paths],check=True);subprocess.run(['git','-C',str(REPO),'diff','--cached','--check'],check=True);assert set(git('diff','--cached','--name-only').splitlines())==set(paths);subprocess.run(['git','-C',str(REPO),'commit','-m','Compare registered grid weights and full native SAM masks with unchanged gates'],check=True)

if __name__=='__main__':{'append':append,'publish':publish}[sys.argv[1]]()
