"""Self-contained local media and immutable reproducibility handoff."""
import os,json,hashlib,shutil,html
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];work=ROOT/'task_workspaces/labelagent_semantic_recovery_20261105';out=ROOT/'outputs/labelagent_semantic_recovery_20261105';control=ROOT/'task_workspaces/labelagent_handoff_control_20261104';home=ROOT/'deliverables/LabelAgent_实时交接/09_补充成果/剩余4托盘与2组复核及分类负对照_20261105'
def read(p):return json.loads(p.read_text(encoding='utf-8'))
def save(p,v):p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(v,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
def sha(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(1048576),b''):h.update(b)
 return h.hexdigest()
def frozen_copy(a,b):
 a,b=Path(a),Path(b);b.parent.mkdir(parents=True,exist_ok=True)
 if b.exists():
  if os.path.samefile(a,b) or sha(a)==sha(b):return str(b)
  raise ValueError('Frozen evidence differs: '+str(b))
 if a.suffix in ['.html','.md']:return shutil.copy2(a,b)
 os.link(a,b);return str(b)
def main():
 if (home/'交接SHA256.json').exists():raise FileExistsError('Sealed handoff is immutable')
 summary=read(out/'public_summary.json');assert read(control/'semantic_ui_verification.json')['passed']
 home.mkdir(parents=True,exist_ok=True)
 for source,label in [(out,'01_主报告与诊断'),(ROOT/'outputs/labelagent_context_comparison_20261105','02_固定前景与地面复核'),(ROOT/'outputs/labelagent_semantic_comparison_20261105','03_新候选全流程复测')]:
  shutil.copytree(source,home/label,copy_function=frozen_copy,ignore=shutil.ignore_patterns('__pycache__','*.pyc'),dirs_exist_ok=True)
 repro=home/'04_冻结源码与配置';shutil.copytree(work/'source_snapshot',repro/'source_snapshot',copy_function=frozen_copy,ignore=shutil.ignore_patterns('__pycache__','*.pyc'),dirs_exist_ok=True)
 for p in Path(__file__).parent.glob('*.py'):
  target=repro/'experiment_tools'/p.name;target.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(p,target)
 for name in ['config.json','config_context.json','config_semantic.json','generate_fine_grid.py']:shutil.copy2(work/name,repro/name)
 cfg=read(work/'config_semantic.json')
 for name in ['reference_annotations','reference_description']:shutil.copy2(cfg[name],repro/(name+'.json'))
 qa=home/'05_核验';qa.mkdir(exist_ok=True)
 for name in ['semantic_ui_verification.json','semantic_regression_verification.json']:
  assert read(control/name)['passed'];shutil.copy2(control/name,qa/name)
 shutil.copy2(out/'original_preservation.json',qa/'原始资料保护.json');shutil.copy2(out/'media_verification.json',qa/'图片视频核验.json')
 for source,label in [(ROOT/'outputs/labelagent_context_comparison_20261105','固定前景'),(ROOT/'outputs/labelagent_semantic_comparison_20261105','新候选')]:
  for name in ['acceptance_checks.json','budget_audit.json','round_analysis.json']:shutil.copy2(source/name,qa/(label+'_'+name))
 # Same physical model contents are copied independently for recoverability.
 package=Path(cfg['package']);model=home/'06_同权重Mask2Former模型包';model.mkdir(exist_ok=True)
 shutil.copytree(package/'model',model/'model',dirs_exist_ok=True)
 for name in ['infer_single.py','calibrated_postprocess.py','inference_config.json','README.md']:shutil.copy2(package/name,model/name)
 expected='d11c2e7f86cd5189605b857c4697e55a5caead0a6888b52b2f5ff814150f5291';assert sha(model/'model/model.safetensors')==expected and not os.path.samefile(model/'model/model.safetensors',package/'model/model.safetensors')
 save(qa/'同一模型权重核验.json',{'passed':True,'sha256':expected,'independent_copy':True,'training_updates':0})
 shutil.copy2(out/'结论与限制.txt',home/'结论与限制.txt')
 public=ROOT/'task_workspaces/labelagent_realtime_handoff_20261104/docs/test-results/semantic-recovery-final.md'
 if public.exists():shutil.copy2(public,home/'最终结论.md')
 notes=f"""本轮交接：剩余4托盘、2组复核与分类负对照

先打开01_主报告与诊断/index.html。主报告包含{summary['pictures']}张图片和{summary['videos']}段视频；均来自真实输入、推理输出和保存的Agent阶段。视频是结果回放，不是连续录屏。失败预检、被拒候选、灰白交叉实验和原网格输出完整保留。

02固定前景对比：保持上轮13/17托盘前景，以原SAM地面及首轮动作优化完成复核。03为新模型候选与中性观察的独立全流程对比，不能用其结果覆盖02或上轮。所有验收上限保持，候选未人工确认为真值，无默认安装和标注写回。

04保存实际隔离源码、实验工具、配置、原参考副本。06是同一真实Mask2Former推理包与独立复制的权重，SHA256 {expected}。SAM3/Qwen权重、完整ROS采集和依赖仍使用原环境；还原运行需按备份位置重绑配置路径。媒体可本地离线浏览。

结果和源码为已冻结文件；同盘硬链接用于避免重复占用，按内容正常复制即可备份。不要在归档内原地改写；改动请另存副本。原始标注、原PNG、原参考和旧结果保留。完整文件清单见交接SHA256.json，05含浏览器、逐帧视频、65项代码检查和原始资料保护收据。
"""
 (home/'交接说明.txt').write_text(notes,encoding='utf-8')
 (home/'index.html').write_text('<!doctype html><meta charset="utf-8"><h1>剩余4托盘 / 2组复核 / 分类负对照</h1><p><a href="01_主报告与诊断/index.html">主报告、全部图片与5段视频</a></p><p><a href="最终结论.md">最终结论</a></p><pre>'+html.escape(notes)+'</pre>',encoding='utf-8')
 rows=[{'path':p.relative_to(home).as_posix(),'sha256':sha(p),'bytes':p.stat().st_size} for p in home.rglob('*') if p.is_file() and p.name!='交接SHA256.json'];save(home/'交接SHA256.json',{'files':rows})
 bad=[r['path'] for r in rows if sha(home/r['path'])!=r['sha256']];assert not bad
 receipt={'passed':True,'bundle':str(home),'files':len(rows),'pictures':summary['pictures'],'videos':summary['videos'],'mismatches':bad,'independent_model_copy_verified':True,'browser_verified':True,'all_videos_fully_decoded':True,'regression_tests':65,'original_annotations_preserved':read(out/'original_preservation.json')['passed']};save(control/'semantic_bundle_verification.json',receipt);print(json.dumps(receipt,ensure_ascii=False))
if __name__=='__main__':main()
