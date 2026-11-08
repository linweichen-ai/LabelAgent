"""Seal the completed broader proposal arm, retaining all earlier evidence."""
import re,shutil,subprocess,sys
from pathlib import Path
from prepare_single_teacher_grid import ROOT,BASE,OUT
from finish_single_teacher_grid import REPORT
from run_rgb_layers import read,save
from deliver_context_graph import sha
C=ROOT/'task_workspaces/labelagent_handoff_control_20261104';HOME=ROOT/'deliverables/LabelAgent_实时交接';REPO=ROOT/'task_workspaces/labelagent_realtime_handoff_20261104';FULL=BASE/'LabelAgent_完整交接_锚点间隙与覆盖缺口_20261108';APPEND=FULL/'single';DISPLAY=HOME/'09_补充成果/跨裁剪锚点与新实例间隙及覆盖缺口_20261108';LEAF=DISPLAY/'single_teacher_report';CODE=['continue_single_teacher_audit.py','finish_single_teacher_grid.py','deliver_single_teacher_grid.py']
def git(*a):return subprocess.check_output(['git','-C',str(REPO),*a],text=True,encoding='utf-8').strip()
def manifests():
 for folder in [FULL,DISPLAY]:save(folder/'交接SHA256.json',{'files':[{'path':p.relative_to(folder).as_posix(),'sha256':sha(p),'bytes':p.stat().st_size} for p in folder.rglob('*') if p.is_file() and p.name!='交接SHA256.json']})
def append():
 if APPEND.exists() or LEAF.exists():raise FileExistsError('Preserve completed or interrupted single-teacher packet')
 assert read(REPORT/'independent_readback.json')['passed'] and read(C/'single_teacher_ui_verification.json')['passed'];files={}
 for prefix,folder in [('run',OUT),('report',REPORT),('history_superseded_report',BASE/'labelagent_single_teacher_report_before_identity_correction_20261108')]:
  for p in folder.rglob('*'):
   if p.is_file():files[prefix+'/'+p.relative_to(folder).as_posix()]={'source':str(p),'sha256':sha(p),'bytes':p.stat().st_size}
 for name in CODE:
  p=Path(__file__).with_name(name);files['code/'+name]={'source':str(p),'sha256':sha(p),'bytes':p.stat().st_size}
 assert max(len(str(APPEND/name)) for name in files)<245;assert shutil.disk_usage(FULL).free>sum(r['bytes'] for r in files.values())+96*2**20
 for name,row in files.items():
  p=APPEND/name;p.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(row['source'],p);assert sha(p)==row['sha256']
 save(APPEND/'append_manifest.json',{'files':files,'all_8_new_native_arrays_all_actual_requests_source_photos_and_raw_replies_retained':True,'previous_next_progress_is_historical_pre_SAM_snapshot':True,'model_weights_and_runtime_external':True});assert shutil.disk_usage('C:/').free>sum(p.stat().st_size for p in REPORT.rglob('*') if p.is_file())+96*2**20;shutil.copytree(REPORT,LEAF);index=DISPLAY/'index.html';t=index.read_text('utf-8');t=t.replace('</h1>','</h1><p><a href="single_teacher_report/index.html">追加：单模型提案扩展，8张原生结果与源边界账本，6图2回放</a></p>',1).replace('本轮51图10视频（四份报告）','本轮57图12视频（五份报告）').replace('C盘展示累计51图10视频','C盘展示累计57图12视频');index.write_text(t,encoding='utf-8');index=FULL/'index.html';index.write_text(index.read_text('utf-8').replace('</h1>','</h1><p><a href="single/report/index.html">追加：单模型提案8张完整结果、31次新语义回复与完整账本</a></p>',1),encoding='utf-8');receipt={'passed':True,'complete_private_directory':str(APPEND),'manifest_sha256':sha(APPEND/'append_manifest.json'),'members':len(files),'logical_bytes':sum(r['bytes'] for r in files.values()),'new_pictures':6,'new_videos':2,'total_series_pictures':57,'total_series_videos':12,'new_accepted_instances':0};save(C/'single_teacher_append_verification.json',receipt);save(LEAF/'完整证据位置.json',receipt);manifests();print(receipt,flush=True)
def publish():
 assert read(C/'single_teacher_append_verification.json')['passed'] and not git('status','--porcelain') and read(C/'watcher.json').get('status')=='stopped';paths=[]
 for name in CODE:
  dest='tools/labelagent_experiments/'+name;shutil.copy2(Path(__file__).with_name(name),REPO/dest);paths.append(dest)
 summary=read(REPORT/'summary.json')
 for dest in ['docs/test-results/single-teacher-grid-plan.json','docs/test-results/single-teacher-grid-completed.json']:save(REPO/dest,summary);paths.append(dest)
 dest='docs/test-results/single-teacher-grid-completed.md';(REPO/dest).write_text('扩大为每个注册模型每源图最多3个原PAL提案，并去除提案阶段双模型一致要求后，21源点+2负对照完成23次中性VLM。仅4个蓝色堆叠点通过，8次配对SAM原生有效、留白前景0且所有前景完整保留，但8张均面积不通过，独立实例纯度和完整语义仍不通过。另完成8次新完整语义调用，19份精确旧回复复用。\n\n逐候选更正：覆盖左侧参考点的实际候选native和stock均不触输入边界，提示点落灰黑构件，中性语义未过；另一个触底边候选是不同物体，不能混称左侧失败原因。旧未封存解释及更正记录在私有证据历史中保留，原模型输出未修改。\n\n该联合提案变化没有带来新增合格实例；原13/17和历史6/6（未重跑Agent）保留，原验收未放宽。新增6张图2段保存阶段回放，系列累计57图12回放，图片/原答/数组仅在私人交接目录。\n',encoding='utf-8');paths.append(dest)
 for dest in paths:assert not re.search(r'gh[pousr]_[A-Za-z0-9]{25,}|sk-[A-Za-z0-9]{30,}|-----BEGIN [A-Z ]*PRIVATE KEY-----',(REPO/dest).read_text('utf-8'))
 subprocess.run(['git','-C',str(REPO),'add','--',*paths],check=True);subprocess.run(['git','-C',str(REPO),'diff','--cached','--check'],check=True);assert set(git('diff','--cached','--name-only').splitlines())==set(paths);subprocess.run(['git','-C',str(REPO),'commit','-m','Record completed single-teacher proposals and unchanged native acceptance failures'],check=True)
if __name__=='__main__':{'append':append,'publish':publish}[sys.argv[1]]()
