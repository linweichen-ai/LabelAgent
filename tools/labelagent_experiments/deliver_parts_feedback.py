"""Complete private packet and sanitized dual-mirror code publication."""
import re,shutil,subprocess,sys
from pathlib import Path
from prepare_palette_parts import ROOT,BASE
from finish_parts_feedback_report import REPORT
from run_rgb_layers import read,save
from deliver_context_graph import sha
C=ROOT/'task_workspaces/labelagent_handoff_control_20261104';HOME=ROOT/'deliverables/LabelAgent_实时交接';REPO=ROOT/'task_workspaces/labelagent_realtime_handoff_20261104';FULL=BASE/'LabelAgent_完整交接_部件与原生反馈_20261108';DISPLAY=HOME/'09_补充成果/部件选点与原生反馈_20261108';OLD=BASE/'LabelAgent_完整交接_锚点间隙与覆盖缺口_20261108'
CODE='palette_part_points prepare_palette_parts balanced_palette_part_points prepare_balanced_palette_parts run_palette_parts test_palette_parts audit_palette_parts neutral_prompt_pair native_review_feedback replay_native_feedback test_native_review_feedback run_concise_palette_candidates prepare_mf_first_parts finish_parts_feedback_report deliver_parts_feedback'.split()
def git(*a):return subprocess.check_output(['git','-C',str(REPO),*a],text=True,encoding='utf-8').strip()
def prepare():
    if FULL.exists() or DISPLAY.exists():raise FileExistsError('Preserve sealed or interrupted packet')
    assert read(REPORT/'independent_readback.json')['passed'] and read(C/'parts_feedback_ui_verification.json')['passed'];files={}
    def add(source,name):
        source=Path(source);assert source.is_file() and name not in files;files[name]={'source':str(source),'sha256':sha(source),'bytes':source.stat().st_size}
    folders={'base0':BASE/'labelagent_palette_parts_20261108','parts':BASE/'labelagent_balanced_palette_parts_20261108','prompt':BASE/'labelagent_neutral_prompt_pair_20261108','concise':BASE/'labelagent_concise_palette_candidates_20261108','feedback':BASE/'labelagent_native_feedback_replay_20261108','next0':BASE/'labelagent_MF_first_parts_20261108','report':REPORT,'mf':BASE/'labelagent_registered_fixed_grid_20261108'}
    for prefix,folder in folders.items():
        for p in folder.rglob('*'):
            if p.is_file():add(p,prefix+'/'+p.relative_to(folder).as_posix())
    old=read(OLD/'complete_evidence_manifest.json')['files']
    for name,r in old.items():
        if name.split('/')[0] in ['original_six','original_references','original_projection_inputs','original_evaluator']:
            assert sha(OLD/name)==r['sha256'];add(OLD/name,'original/'+name)
    # Every old response consulted by the zero-call replay is copied directly;
    # original NPZ arrays are embedded losslessly in its raw_capture field.
    protected=read(BASE/'labelagent_native_feedback_replay_20261108/completed.json')['protected'];number=0
    for p,digest in protected.items():
        if Path(p).name=='response.json':assert sha(p)==digest;add(p,f'feedback_dependencies/{number:02d}/response.json');number+=1
    for name in CODE:add(Path(__file__).with_name(name+'.py'),'code/'+name+'.py')
    add(C/'parts_feedback_ui_verification.json','verification/parts_feedback_ui_verification.json');longest=max(len(str(FULL/name)) for name in files);assert longest<245,longest;size=sum(r['bytes'] for r in files.values());assert shutil.disk_usage(BASE).free>size+192*2**20;FULL.mkdir()
    for name,r in files.items():
        p=FULL/name;p.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(r['source'],p);assert sha(p)==r['sha256']
    source_models=read(BASE/'labelagent_registered_fixed_grid_20261108/frozen_plan.json')['models'];sam=read(ROOT/'outputs/labelagent_registry_zoom_pool_20261108/SAM_weights_identity.json');save(FULL/'外部依赖身份.json',{'registered_MF_models':source_models,'SAM_prior_streamed_identity':sam,'weights_not_copied':True,'runtime_not_copied':True,'not_standalone_environment_image':True,'historical_old_28_captures_raw_replies_embedded_arrays_copied':True,'old_complete_packets_also_to_preserve':[str(OLD),str(BASE/'LabelAgent_完整交接_上下文与实例冲突图_20261108')]});save(FULL/'complete_evidence_manifest.json',{'files':files,'all_new_raw_arrays_masks_actual_requests_replies_original_sources_and_previous_control_annotations_copied':True,'registered_MF_raw_queries_copied_are_historical_reuse_not_new_forwards':True,'base0_and_next0_folders_are_preparation_snapshots_zero_calls':True,'model_weights_and_runtime_external':True})
    summary=read(REPORT/'summary.json');(FULL/'index.html').write_text('<meta charset="utf-8"><h1>部件选点与原生反馈完整证据</h1><p><a href="report/index.html">16图4回放及严格对比</a> · <a href="complete_evidence_manifest.json">完整原始证据清单</a> · <a href="外部依赖身份.json">外部模型与旧完整包依赖</a></p><p>没有新增合格实例；原13/17与历史6/6保留。next0是16点32请求准备快照，不是执行结果。MF原查询是历史复用，本轮0新前向；权重与运行环境未复制。</p>',encoding='utf-8');assert shutil.disk_usage('C:/').free>sum(p.stat().st_size for p in REPORT.rglob('*') if p.is_file())+96*2**20;shutil.copytree(REPORT,DISPLAY);(DISPLAY/'完整原始证据备份位置.txt').write_text(str(FULL)+'\nC盘此处为可播放媒体摘要。完整原答、原生数组、输入与旧原标注在上述D目录，需与旧两个完整D包一起备份。next0仅准备0调用。\n',encoding='utf-8');receipt={'passed':True,'full_private_packet':str(FULL),'manifest_sha256':sha(FULL/'complete_evidence_manifest.json'),'members':len(files),'logical_bytes':size,'max_file_path_length':longest,'pictures':16,'videos':4,'fully_decoded_frames':640,'display':str(DISPLAY),'C_display_media_only':True,'series_pictures':73,'series_videos':16,'new_accepted_instances':0};save(C/'parts_feedback_bundle_verification.json',receipt);save(DISPLAY/'完整证据核验.json',receipt)
    note=HOME/'00_本轮完整证据备份位置.txt';oldnote=note.read_text('utf-8');note.write_text('最新新增16图4回放，本系列累计73图16回放。完整新原始证据另在：\n'+str(FULL)+'\n请备份整个D目录及此前两个完整D包；C盘此处只媒体摘要。默认服务未部署新反馈适配器，所有原标注保留。next0只有准备，无新推理。\n\n此前历史记录：\n'+oldnote,encoding='utf-8')
    for folder in [FULL,DISPLAY]:save(folder/'交接SHA256.json',{'files':[{'path':p.relative_to(folder).as_posix(),'sha256':sha(p),'bytes':p.stat().st_size} for p in folder.rglob('*') if p.is_file() and p.name!='交接SHA256.json']})
    print(receipt,flush=True)
def publish():
    assert read(C/'parts_feedback_bundle_verification.json')['passed'] and read(C/'parts_feedback_C_ui_verification.json')['passed'];assert read(C/'watcher.json').get('status')=='stopped' and not git('status','--porcelain');paths=[]
    for name in CODE:
        dest='tools/labelagent_experiments/'+name+'.py';shutil.copy2(Path(__file__).with_name(name+'.py'),REPO/dest);paths.append(dest)
    summary=read(REPORT/'summary.json');dest='docs/test-results/parts-native-feedback.json';save(REPO/dest,summary);paths.append(dest);dest='docs/test-results/MF-first-parts-plan.json';save(REPO/dest,{'status':'prepared_not_executed','source_points':16,'logical_SAM_requests':32,'new_model_calls':0,'original_point_caption_failed_or_uncertain':15,'previous_captions_kept_faithfully_not_relabelled_as_PAL':True,'proposal_order_control':'original_trained_stock_PAL_then_complete_native_SAM_then_unchanged_whole_instance_final_review','no_NULL_promoted':True,'all_final_gates_unchanged':True,'Agent_round_limit':3,'new_Agent_runs':0,'not_adopted':True});paths.append(dest)
    dest='docs/test-results/parts-native-feedback.md';(REPO/dest).write_text('真实原生反馈回放找到了两条服务无警告但负点7×7仍含9/10前景像素的记录：选择器在无完全合格候选时回退到违反最少者，后续返回只检查负点中心。新实验复核适配器在28条真实捕获的零调用回放中明确拒收，并在原第3轮上限弃权、保留原标注；这不是新Agent收敛结果，也未部署到默认服务。9个原生约束与部件源点不变量测试通过。\n\n本轮22部位点、同照片提示配对与原完整语义共70次新VLM，10次新SAM。简洁提示通过六项PAL/非PAL校准，源提案1→5，但10张逻辑SAM（8新+2精确复用）中9张原生有效，完整语义0/9，没有新增合格实例。负点臂7961像素的新增占地163虽合预算，仍负点违反且纯度不足；5342像素候选支持的是旧实例，不增加缺失实例，新增占地233也超限。原13/17和历史6/6保留，未新跑Agent、未加轮、未裁掩码、未降门槛。\n\n新增16张对照图、4段保存阶段回放，系列累计73图16回放。图片、原答和数组仅本地完整交接，公开代码与脱敏数字。下一MF→SAM→完整复核提案顺序对照已准备16点32请求，提交时0调用，旧灰黑部件描述保留，最终全部原验收继续要求。\n',encoding='utf-8');paths.append(dest)
    for name in paths:assert not re.search(r'gh[pousr]_[A-Za-z0-9]{25,}|sk-[A-Za-z0-9]{30,}|-----BEGIN [A-Z ]*PRIVATE KEY-----',(REPO/name).read_text('utf-8'))
    subprocess.run(['git','-C',str(REPO),'add','--',*paths],check=True);subprocess.run(['git','-C',str(REPO),'diff','--cached','--check'],check=True);assert set(git('diff','--cached','--name-only').splitlines())==set(paths);subprocess.run(['git','-C',str(REPO),'commit','-m','Compare source parts and neutral prompts; expose native negative-neighborhood failures'],check=True)
if __name__=='__main__':{'prepare':prepare,'publish':publish}[sys.argv[1]]()
