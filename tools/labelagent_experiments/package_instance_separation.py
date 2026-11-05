"""Seal this complete experiment without altering earlier handoff bundles."""
import json,os,hashlib,shutil,html
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];work=ROOT/'task_workspaces/labelagent_instance_separation_20261105';out=ROOT/'outputs/labelagent_instance_separation_20261105';control=ROOT/'task_workspaces/labelagent_handoff_control_20261104';home=ROOT/'deliverables/LabelAgent_实时交接/09_补充成果/遮挡粘连实例分离_20261105'
def read(p):return json.loads(p.read_text(encoding='utf-8'))
def save(p,v):p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(v,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(1048576),b''):h.update(b)
 return h.hexdigest()
def frozen(a,b):
 a,b=Path(a),Path(b);b.parent.mkdir(parents=True,exist_ok=True)
 if b.exists():
  if os.path.samefile(a,b) or sha(a)==sha(b):return str(b)
  raise ValueError('Frozen file differs: '+str(b))
 if a.suffix in ['.html','.md']:return shutil.copy2(a,b)
 os.link(a,b);return str(b)
def main():
 if (home/'交接SHA256.json').exists():raise FileExistsError('Sealed bundle immutable')
 assert read(control/'instance_separation_ui_verification.json')['passed'] and read(control/'instance_separation_regression.json')['passed'];s=read(out/'public_summary.json');cfg=read(work/'config_comparison.json');assert read(out/'complete_generated_mask_proof.json')['masks']==37
 home.mkdir(parents=True,exist_ok=True)
 for folder,label in [(out,'01_主报告与全部试跑'),(Path(cfg['output']),'02_原三轮完整复测'),(ROOT/'outputs/labelagent_context_comparison_20261105','03_保留13实例与6组结果')]:shutil.copytree(folder,home/label,copy_function=frozen,dirs_exist_ok=True,ignore=shutil.ignore_patterns('__pycache__','*.pyc'))
 native=home/'04_原始资料只读副本';native.mkdir(exist_ok=True);data=Path(cfg['dataset']);manifest=read(data/'import_manifest.json');selected=['poses_aligned_motion.txt']+[p for p in manifest['extracted_sha256'] if p.startswith('calib/')]
 for frame in [74,75,76,77,78]:
  stamp=str(manifest['timestamps_ns'][frame]);selected+=[f'labels/{stamp}.json',f'lidar/{stamp}.pcd']+[f'cameras/camera_{cam}/{stamp}.png' for cam in range(4)]
 rows=[]
 for rel in selected:
  source=(data/'source'/rel).resolve();assert source.is_relative_to((data/'source').resolve());expected=manifest['extracted_sha256'][rel];assert sha(source)==expected;target=native/'source'/rel;target.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(source,target);assert sha(target)==expected;rows.append({'path':rel,'sha256':expected,'bytes':target.stat().st_size})
 for key in ['reference_annotations','reference_description']:shutil.copy2(cfg[key],native/(key+'.json'))
 rf,rc=cfg['reference_case'];record=next(r for r in manifest['records'] if r['frame']==rf and r['topic']==f'/capture/camera_{rc}/image');assert sha(data/record['source_file'])==record['sha256'];shutil.copy2(data/record['source_file'],native/'原参考图.png');shutil.copy2(data/'import_manifest.json',native/'import_manifest_original.json');save(native/'原始副本核验.json',{'passed':True,'files':rows,'originals_unchanged':True,'raw_MCAP_not_duplicated':True})
 repro=home/'05_冻结源码与配置';shutil.copytree(work/'source_snapshot',repro/'source_snapshot',copy_function=frozen,ignore=shutil.ignore_patterns('__pycache__','*.pyc'));(repro/'experiment_tools').mkdir()
 for f in Path(__file__).parent.glob('*.py'):shutil.copy2(f,repro/'experiment_tools'/f.name)
 for name in ['config.json','config_comparison.json']:shutil.copy2(work/name,repro/name)
 qa=home/'06_核验';qa.mkdir()
 for name in ['instance_separation_ui_verification.json','instance_separation_regression.json','instance_separation_regression.log']:shutil.copy2(control/name,qa/name)
 for name in ['whole_mask_preservation.json','original_preservation.json','complete_generated_mask_proof.json','acceptance_checks.json','budget_audit.json','round_analysis.json','media_verification.json']:shutil.copy2(out/name,qa/name)
 package=Path(cfg['package']);model=home/'07_同权重Mask2Former独立模型包';assert shutil.disk_usage(ROOT).free>512*1024**2;shutil.copytree(package/'model',model/'model')
 for name in ['infer_single.py','calibrated_postprocess.py','inference_config.json','README.md']:shutil.copy2(package/name,model/name)
 expected='d11c2e7f86cd5189605b857c4697e55a5caead0a6888b52b2f5ff814150f5291';assert sha(model/'model/model.safetensors')==expected and not os.path.samefile(model/'model/model.safetensors',package/'model/model.safetensors');save(qa/'模型独立复制核验.json',{'passed':True,'sha256':expected,'independent_copy':True,'training_updates':0})
 shutil.copy2(out/'本轮结论.md',home/'最终结论.md');notes=f'''遮挡粘连实例分离：本轮交接

打开01_主报告与全部试跑/index.html：{s['pictures']}张浏览图片、5段1080p视频，{s['fully_decoded_video_frames']}帧全部解码核验；均为真实输入、原始输出和保存阶段回放，不是连续录屏。
仍13/17、490/579点、6/6收敛，4个漏检未恢复。16次同权重模型前向、15次真实无先验SAM、2次明确标为合成的空白SAM诊断，30个原始查询和7个完整SAM候选全部保留。未通过的候选不替换保留结果。整摞写single的复核矛盾已加保守拦截，v1失败和v2解析都保留；未增加Agent轮数、放宽验收、裁掩码或写回原标注。

02是真实原三轮复测，A/B四组身份绑定复用、C两组真实新跑；03是保留的13实例与6组结果。04为四相机五邻帧PNG、PCD、原3D标签、标定、位姿及原人工参考的独立复制，逐一原摘要回读；07为同一Mask2Former独立权重包，SHA256 {expected}。
05含执行时隔离源码、实际实验工具及配置，路径需按恢复位置重绑；全量MCAP、SAM3/Qwen权重和系统运行依赖未重复复制，不宣称离线一键重跑完整环境。报告图片视频可以离线浏览。06含82项工程检查、原前景保护、37个原始输出逐像素证明及浏览器核验。

已冻结的结果与源码使用同盘硬链接节省重复占用，备份时正常按文件内容复制即可；不要在归档内原地改写，请另存新版本。原始输入和Mask2Former模型为独立复制。清单见交接SHA256.json。公共Git只含代码和去标识化汇总，不含真实媒体、原标注或权重。
''';(home/'交接说明.txt').write_text(notes,encoding='utf-8');(home/'index.html').write_text('<!doctype html><meta charset="utf-8"><h1>遮挡粘连实例分离</h1><p><a href="01_主报告与全部试跑/index.html">106张图与5段视频</a></p><p><a href="最终结论.md">本轮结论与限制</a></p><pre>'+html.escape(notes)+'</pre>',encoding='utf-8')
 files=[{'path':p.relative_to(home).as_posix(),'sha256':sha(p),'bytes':p.stat().st_size} for p in home.rglob('*') if p.is_file() and p.name!='交接SHA256.json'];save(home/'交接SHA256.json',{'files':files});bad=[r['path'] for r in files if sha(home/r['path'])!=r['sha256']];assert not bad
 receipt={'passed':True,'bundle':str(home),'files':len(files),'mismatches':bad,'pictures':s['pictures'],'videos':5,'fully_decoded_frames':s['fully_decoded_video_frames'],'native_files_readback_verified':len(rows),'independent_model_copy_verified':True,'incumbent_PAL_masks_preserved':27,'incumbent_person_masks_preserved':3,'tests':read(control/'instance_separation_regression.json')['tests'],'annotation_writes':0};save(control/'instance_separation_bundle_verification.json',receipt);print(json.dumps(receipt,ensure_ascii=False),flush=True)
if __name__=='__main__':main()
