"""Seal immutable native evidence with explicit same-disk link accounting."""
import os,re,shutil,subprocess,sys
from pathlib import Path
from prepare_palette_parts import ROOT,BASE
from prepare_mf_first_parts import OUT
from finish_MF_first_report import REPORT
from run_rgb_layers import read,save
from deliver_context_graph import sha
C=ROOT/'task_workspaces/labelagent_handoff_control_20261104'
HOME=ROOT/'deliverables/LabelAgent_实时交接'
FULL=BASE/'LA_MF32_20261108'
REPO=ROOT/'task_workspaces/labelagent_realtime_handoff_20261104'
CODE='run_MF_first_parts MF_first_semantics MF_first_native_pool MF_pool_semantics audit_MF_first_parts finish_MF_first_report deliver_MF_first test_handoff_space_reserve'.split()

def git(*a):return subprocess.check_output(['git','-C',str(REPO),*a],text=True,encoding='utf-8').strip()

def prepare():
    if FULL.exists():raise FileExistsError('Preserve sealed or interrupted packet')
    assert read(REPORT/'ui_verification.json')['passed'];summary=read(REPORT/'summary.json');files={}
    def add(source,name):
        source=Path(source);assert source.is_file() and name not in files;files[name]={'source':str(source),'sha256':sha(source),'bytes':source.stat().st_size}
    for prefix,folder in [('e',OUT),('report',REPORT)]:
        for p in folder.rglob('*'):
            if p.is_file():add(p,prefix+'/'+p.relative_to(folder).as_posix())
    prior=BASE/'LabelAgent_完整交接_部件与原生反馈_20261108'
    for p in (prior/'original').rglob('*'):
        if p.is_file():add(p,'original/'+p.relative_to(prior/'original').as_posix())
    for p in (prior/'mf').rglob('*'):
        if p.is_file():add(p,'historical_MF/'+p.relative_to(prior/'mf').as_posix())
    for name in CODE+['maintain']:add(Path(__file__).with_name(name+'.py'),'code/'+name+'.py')
    for i,r in enumerate(read(OUT/'generation_completed.json')['rows']):
        p=Path(r['original_neutral_response']);name='point_caption_dependencies/'+p.name
        if name not in files:add(p,name)
    assert max(len(str(FULL/name)) for name in files)<245
    assert shutil.disk_usage(BASE).free>192*2**20;FULL.mkdir();links=0;copies=0
    for name,r in files.items():
        p=FULL/name;p.parent.mkdir(parents=True,exist_ok=True);source=Path(r['source'])
        if source.drive.lower()==FULL.drive.lower():
            os.link(source,p);assert os.path.samefile(source,p);r['storage_mode']='immutable_same_disk_hardlink';links+=1
        else:shutil.copy2(source,p);r['storage_mode']='independent_file_copy';copies+=1
        assert sha(p)==r['sha256']
    save(FULL/'complete_evidence_manifest.json',{'files':files,'immutable_terminal_results_only':True,'same_disk_links_not_independent_physical_backup':True,'backup_action':'Copy this entire directory to another backup medium to store file contents independently. Keep prior three complete D packets too.','model_weights_runtime_external':True,'source_absolute_paths_preserved_as_provenance_not_portable_runtime_configuration':True})
    note='本机交接以不可变文件的同盘硬链接组装，不是独立物理备份。请把整个本目录复制到其他备份介质，并保留此前三个D完整交接包。C盘是入口索引，本轮媒体没有复制到C盘。模型权重与运行环境仍是外部依赖。所有源数组与原标注不改写。'
    (FULL/'index.html').write_text('<meta charset="utf-8"><h1>MF初稿与完整原生SAM完整证据</h1><p>'+note+'</p><p><a href="report/index.html">22图4段保存阶段回放</a> · <a href="complete_evidence_manifest.json">逐文件SHA清单</a></p>',encoding='utf-8');(FULL/'备份说明.txt').write_text(note,encoding='utf-8')
    receipt={'passed':True,'full_packet':str(FULL),'members':len(files),'manifest_sha256':sha(FULL/'complete_evidence_manifest.json'),'logical_bytes':sum(r['bytes'] for r in files.values()),'immutable_same_disk_links':links,'independent_file_copies':copies,'independent_physical_backup':False,'pictures':summary['pictures'],'videos':summary['saved_stage_replays'],'series_pictures':95,'series_saved_stage_replays':20,'C_contains_current_media':False,'all_members_read_back':True}
    save(C/'MF_first_bundle_verification.json',receipt);save(HOME/'06_核验/MF初稿与原生输出核验.json',receipt)
    (HOME/'00_剩余托盘推进_先读.txt').write_text('本轮MF→完整SAM→原语义复核已完成：32逻辑请求=29新SAM+3精确历史复用；默认25有效、固定索引池76有效，全部语义均0通过。新VLM '+str(summary['new_VLM_calls'])+'，新MF前向0、新Agent运行0。13/17和历史6/6保留；剩余4例；不裁输出、不加轮、不放宽验收。新增22图4回放；本系列95图20回放。\n当前22图4视频与完整证据入口：'+str(FULL/'index.html')+'\n'+note+'\n持续同步三段对话、GitHub和CNB。下一步从原RGB与未标注点云关联研究分层，不用参考点生成提示。\n',encoding='utf-8')
    note_file=HOME/'00_本轮完整证据备份位置.txt';old=note_file.read_text('utf-8');note_file.write_text('最新完整证据：'+str(FULL)+'\n'+note+'\n本轮22图4回放；本系列95图20回放。\n\n此前记录：\n'+old,encoding='utf-8');print(receipt,flush=True)

def publish():
    assert read(C/'MF_first_bundle_verification.json')['passed'];assert read(C/'watcher.json').get('status')=='stopped';assert not git('status','--porcelain');paths=[]
    for name in CODE:
        dest='tools/labelagent_experiments/'+name+'.py';shutil.copy2(Path(__file__).with_name(name+'.py'),REPO/dest);paths.append(dest)
    for name in ['maintain','test_handoff_space_reserve']:
        dest='tools/labelagent_live_handoff/'+name+'.py';shutil.copy2(Path(__file__).with_name(name+'.py'),REPO/dest);paths.append(dest)
    summary=read(REPORT/'summary.json');dest='docs/test-results/MF-first-parts-plan.json';save(REPO/dest,summary);paths.append(dest)
    dest='docs/test-results/MF-first-native-output-comparison.md';(REPO/dest).write_text('训练MF的stock PAL源提示经完整SAM后才进行原语义验收：32个逻辑请求，29次新SAM、3次精确历史复用。默认25张原生有效；冻结固定原生索引池96张中76有效，未按参考支持或面积选择。默认与固定索引池全部语义均0通过；没有新增合格实例。\n\n新VLM '+str(summary['new_VLM_calls'])+'次；新MF前向0、新Agent运行0。源点原局部caption忠实保留。左侧留白输入消除了部分输入触边，但不保证完整独立基座；固定原生索引p084达到8/11支持、纯度1，新增占地1004仍超过209.25余量，完整语义也未过。留白前景124/91的输出保留原生失败证据，没有裁切后计分；任何备选不能作为同请求的多个同时实例。\n\n原13/17与历史6/6保持，原标注不动，原三轮与全部门槛不变，未部署默认服务。本轮22图4段保存阶段回放，本系列95图20回放，仅本地交接；未公开媒体、原答、输入、数组或参考标注。C盘空间不足时保留96MiB媒体复制余量，交接索引明确列出D盘资料，空间恢复后补复制，真实回退与恢复测试通过。完整D包以终态文件同盘硬链接组装，须复制整个目录到外部介质才是独立备份；权重和环境外部。\n',encoding='utf-8');paths.append(dest)
    for name in paths:assert not re.search(r'gh[pousr]_[A-Za-z0-9]{25,}|sk-[A-Za-z0-9]{30,}|-----BEGIN [A-Z ]*PRIVATE KEY-----',(REPO/name).read_text('utf-8'))
    subprocess.run(['git','-C',str(REPO),'add','--',*paths],check=True);subprocess.run(['git','-C',str(REPO),'diff','--cached','--check'],check=True);assert set(git('diff','--cached','--name-only').splitlines())==set(paths);subprocess.run(['git','-C',str(REPO),'commit','-m','Compare complete native MF-first proposals and preserve media under low disk space'],check=True)

if __name__=='__main__':{'prepare':prepare,'publish':publish}[sys.argv[1]]()
