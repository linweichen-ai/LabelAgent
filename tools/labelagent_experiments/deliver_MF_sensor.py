"""Seal sensor diagnostics and publish only code and sanitized quantitative results."""
import os,re,shutil,subprocess,sys
from pathlib import Path
from run_MF_stock_sensor_SAM import OUT,SOURCE
from finish_MF_sensor_report import REPORT
from prepare_palette_parts import ROOT,BASE
from run_rgb_layers import read,save
from deliver_context_graph import sha
from deliver_MF_first import C,HOME,REPO,git
FULL=BASE/'LA_SENSOR32_20261108'
CODE='MF_stock_sensor_parts run_MF_stock_sensor_SAM MF_stock_sensor_review audit_MF_stock_sensor finish_MF_sensor_report deliver_MF_sensor'.split()

def prepare():
    if FULL.exists():raise FileExistsError('Preserve sealed or interrupted packet')
    assert read(REPORT/'ui_verification.json')['passed'];files={}
    def add(source,name):
        source=Path(source);assert source.is_file() and name not in files;files[name]={'source':str(source),'sha256':sha(source),'bytes':source.stat().st_size}
    for prefix,folder in [('e',SOURCE),('report',REPORT),('original',BASE/'LA_MF32_20261108/original'),('historical_MF',BASE/'LA_MF32_20261108/historical_MF')]:
        for p in folder.rglob('*'):
            if p.is_file():add(p,prefix+'/'+p.relative_to(folder).as_posix())
    inputs=set()
    for r in read(OUT/'generation_completed.json')['rows']:
        for key in ['input','source_image']:
            f=Path(r[key]);digest=sha(f)
            if digest not in inputs:add(f,'source_inputs/'+digest+'.png');inputs.add(digest)
    for name in CODE+['maintain','test_handoff_space_reserve']:add(Path(__file__).with_name(name+'.py'),'code/'+name+'.py')
    assert max(len(str(FULL/name)) for name in files)<245
    assert shutil.disk_usage(BASE).free>192*2**20;FULL.mkdir();links=copies=0
    for name,r in files.items():
        p=FULL/name;p.parent.mkdir(parents=True,exist_ok=True);s=Path(r['source'])
        if s.drive.lower()==FULL.drive.lower():os.link(s,p);assert os.path.samefile(s,p);r['storage_mode']='immutable_same_disk_hardlink';links+=1
        else:shutil.copy2(s,p);r['storage_mode']='independent_file_copy';copies+=1
        assert sha(p)==r['sha256']
    save(FULL/'complete_evidence_manifest.json',{'files':files,'same_disk_links_not_independent_physical_backup':True,'copy_whole_directory_to_another_medium_for_independent_backup':True,'keep_previous_four_D_packets_too':True,'model_weights_and_runtime_external':True,'sensor_modality_not_same_as_RGB_only_baseline':True})
    note='本目录是不可变结果的同盘文件链接组装，不是独立物理备份。请复制整个目录到另一备份介质，并保留此前四个D完整包。C盘入口不构成本轮完整媒体备份，完整内容以D包为准；自动同步仅在空间余量满足时补复制媒体。模型权重和运行环境外部。原图、原标注、模型输出不改写。'
    (FULL/'index.html').write_text('<meta charset="utf-8"><h1>点云辅助MF完整证据</h1><p>'+note+'</p><p><a href="report/index.html">27图4段保存阶段回放（另附9个传感器散点图）</a> · <a href="complete_evidence_manifest.json">SHA256清单</a></p>',encoding='utf-8');(FULL/'备份说明.txt').write_text(note,encoding='utf-8');summary=read(REPORT/'summary.json');receipt={'passed':True,'full_packet':str(FULL),'members':len(files),'logical_bytes':sum(r['bytes'] for r in files.values()),'immutable_same_disk_links':links,'independent_file_copies':copies,'independent_physical_backup':False,'manifest_sha256':sha(FULL/'complete_evidence_manifest.json'),'pictures':27,'additional_sensor_scatter_PNGs':9,'videos':4,'series_pictures':122,'series_saved_stage_replays':24,'C_contains_current_media':False,'all_members_read_back':True};save(C/'MF_sensor_bundle_verification.json',receipt);save(HOME/'06_核验/点云辅助MF完整核验.json',receipt)
    (HOME/'00_剩余托盘推进_先读.txt').write_text('本次已完成两批对照：MF-first 29新SAM/156新VLM，以及点云辅助32新SAM/'+str(summary['new_VLM_calls'])+'新VLM。默认原生有效25与19，另固定索引池76，完整语义均0通过；没有新增合格实例。原13/17与历史6/6保留，剩余4例，未裁掩码、未加轮、未降门槛、未部署。两批新增49张汇报图/8段保存阶段回放，另9个传感器散点图；本系列122图24回放。\n最新完整包：'+str(FULL)+'\n前一批：'+str(BASE/'LA_MF32_20261108')+'\n'+note+'\n当前文件总清单以06_核验/当前文件清单位置.json指向的文件为准；低空间时旧C清单作为历史保留。\n左q6 42回波BIC只支持1高度组，不能把高度强当实例身份；应继续研究RGB物理基座与跨视角可见性。\n',encoding='utf-8');p=HOME/'00_本轮完整证据备份位置.txt';p.write_text('最新点云辅助完整证据：'+str(FULL)+'\n'+note+'\n另需备份06_核验/当前文件清单位置.json所指的D盘最新清单。\n\n此前记录：\n'+p.read_text('utf-8'),encoding='utf-8');print(receipt,flush=True)

def publish():
    assert read(C/'MF_sensor_bundle_verification.json')['passed'];assert read(C/'watcher.json').get('status')=='stopped';assert not git('status','--porcelain');paths=[]
    for name in CODE:
        dest='tools/labelagent_experiments/'+name+'.py';shutil.copy2(Path(__file__).with_name(name+'.py'),REPO/dest);paths.append(dest)
    for name in ['maintain','test_handoff_space_reserve']:
        dest='tools/labelagent_live_handoff/'+name+'.py';shutil.copy2(Path(__file__).with_name(name+'.py'),REPO/dest);paths.append(dest)
    duplicate='tools/labelagent_experiments/test_handoff_space_reserve.py';subprocess.run(['git','-C',str(REPO),'rm','--',duplicate],check=True);paths.append(duplicate)
    dest='docs/test-results/MF-stock-sensor-parts.json';save(REPO/dest,read(REPORT/'summary.json'));paths.append(dest);dest='docs/test-results/MF-stock-sensor-parts.md';(REPO/dest).write_text('在全部符合原边界规则的训练R15 stock PAL中，9个源视图经原始未标注点云与标定白名单投影、高度BIC拟合生成16个正点，与其他高度组的假设负点组成32次新SAM请求。19张原生有效，全量原语义门0通过，仍13/17；历史6/6保留、未新跑Agent。输入包含传感器信息，不是与RGB基线同模态的公平收益证明。\n\n左侧真实q6源有42个可见回波，BIC只支持1组，没有强制拆分。其他区域不同高度也不等于不同基座；13个服务无警告输出仍被完整原生约束拒收。原输出、原标注、原三轮和全部验收不变。27张汇报图、9个原传感器散点图、4段保存阶段回放仅本地交接。\n\n磁盘保护保留96MiB媒体复制余量；大清单空间不足时写D盘、C盘保留明确当前位置与SHA指针，旧C清单作为历史保存。空间恢复后自动补复制及恢复本地清单。2项真实回退/恢复测试通过。完整D证据使用不可变终态文件同盘链接，须复制到其他介质才有独立备份，权重和环境外部。\n',encoding='utf-8');paths.append(dest)
    for name in paths:
        if (REPO/name).exists():assert not re.search(r'gh[pousr]_[A-Za-z0-9]{25,}|sk-[A-Za-z0-9]{30,}|-----BEGIN [A-Z ]*PRIVATE KEY-----',(REPO/name).read_text('utf-8'))
    subprocess.run(['git','-C',str(REPO),'add','--',*[p for p in paths if (REPO/p).exists()]],check=True);subprocess.run(['git','-C',str(REPO),'diff','--cached','--check'],check=True);assert set(git('diff','--cached','--name-only').splitlines())==set(paths);subprocess.run(['git','-C',str(REPO),'commit','-m','Test unlabelled sensor height prompts and preserve current manifest under low disk space'],check=True)

if __name__=='__main__':{'prepare':prepare,'publish':publish}[sys.argv[1]]()
