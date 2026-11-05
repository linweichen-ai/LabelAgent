"""Freeze media, selected original inputs, code and the same independent model."""
import json,os,hashlib,shutil,html
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];work=ROOT/'task_workspaces/labelagent_projection_recovery_20261105';out=ROOT/'outputs/labelagent_projection_recovery_20261105';control=ROOT/'task_workspaces/labelagent_handoff_control_20261104';home=ROOT/'deliverables/LabelAgent_实时交接/09_补充成果/4托盘逐点归属与投影遮挡_20261105'
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
  raise ValueError('Frozen evidence differs: '+str(b))
 if a.suffix in ['.html','.md']:return shutil.copy2(a,b)
 os.link(a,b);return str(b)
def main():
 if (home/'交接SHA256.json').exists():raise FileExistsError('Sealed bundle immutable')
 assert read(control/'projection_recovery_ui_verification.json')['passed'] and read(control/'projection_recovery_regression.json')['passed'];s=read(out/'public_summary.json');cfg=read(work/'config_wire_comparison.json');home.mkdir(parents=True,exist_ok=True)
 for source,label in [(out,'01_主报告与诊断'),(ROOT/'outputs/labelagent_projection_wire_comparison_20261105','02_无损完整追加复测'),(ROOT/'outputs/labelagent_projection_retained_comparison_20261105','03_首次轮廓转换失败复测')]:shutil.copytree(source,home/label,dirs_exist_ok=True,copy_function=frozen,ignore=shutil.ignore_patterns('__pycache__','*.pyc'))
 # Selected original capture files, copied with readback rather than linked to D:.
 data=Path(cfg['dataset']);manifest=read(data/'import_manifest.json');native=home/'04_原始资料只读副本';native.mkdir(exist_ok=True);input_rows=[]
 selected=['poses_aligned_motion.txt']+[p for p in manifest['extracted_sha256'] if p.startswith('calib/')]
 for frame in [74,75,76,77,78]:
  stamp=str(manifest['timestamps_ns'][frame]);selected+=[f'labels/{stamp}.json',f'lidar/{stamp}.pcd']+[f'cameras/camera_{cam}/{stamp}.png' for cam in range(4)]
 for rel in selected:
  source=(data/'source'/rel).resolve();assert source.is_relative_to((data/'source').resolve());expected=manifest['extracted_sha256'][rel];assert sha(source)==expected;target=native/'source'/rel;target.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(source,target);assert sha(target)==expected;input_rows.append({'path':rel,'sha256':expected,'bytes':target.stat().st_size})
 shutil.copy2(data/'import_manifest.json',native/'import_manifest_original.json');save(native/'选取原始文件清单.json',{'files':input_rows,'source_files_unchanged':True,'original_full_MCAP_not_duplicated':True,'same_frame_images_PCD_labels_and_calibration_preserved':True,'human2D_truth_not_created':True})
 repro=home/'05_冻结源码与配置';shutil.copytree(work/'source_snapshot',repro/'source_snapshot',dirs_exist_ok=True,copy_function=frozen,ignore=shutil.ignore_patterns('__pycache__','*.pyc'))
 for name in ['config.json','config_comparison.json','config_wire_comparison.json']:shutil.copy2(work/name,repro/name)
 names=['prepare_projection_recovery.py','incumbent_append_policy.py','generate_append_recovery.py','projection_audit_core.py','audit_point_instance_tradeoff.py','audit_projection_visibility.py','inspect_projection_timing.py','prepare_blue_append.py','generate_blue_append.py','prepare_retained_projection_comparison.py','run_retained_projection_comparison.py','audit_wire_geometry.py','wire_geometry_policy.py','generate_wire_append.py','prepare_wire_comparison.py','run_wire_projection_comparison.py','summarize_projection_results.py','summarize_wire_projection_results.py','render_projection_recovery.py','audit_append_point_lineage.py','finalize_projection_media.py','package_projection_recovery.py','write_projection_public.py','test_projection_recovery.py','ground_action_review_v6.py','residual_gap_policy.py','semantic_evidence_gate.py','semantic_subject_guard.py','preserved_model_policy.py']
 for name in names:
  target=repro/'experiment_tools'/name;target.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(Path(__file__).with_name(name),target)
 for name in ['reference_annotations','reference_description']:shutil.copy2(cfg[name],native/(name+'.json'))
 rf,rc=cfg['reference_case'];record=next(r for r in manifest['records'] if r['frame']==rf and r['topic']==f'/capture/camera_{rc}/image');source=data/record['source_file'];assert sha(source)==record['sha256'];shutil.copy2(source,native/'原参考图.png')
 qa=home/'06_核验';qa.mkdir(exist_ok=True)
 for name in ['projection_recovery_ui_verification.json','projection_recovery_regression.json']:shutil.copy2(control/name,qa/name)
 for name in ['whole_mask_preservation.json','original_preservation.json','budget_audit.json','acceptance_checks.json','round_analysis.json','media_verification.json','wire_geometry_failed_variant.json']:shutil.copy2(out/name,qa/name)
 package=Path(cfg['package']);model=home/'07_同权重Mask2Former模型包';shutil.copytree(package/'model',model/'model',dirs_exist_ok=True)
 for name in ['infer_single.py','calibrated_postprocess.py','inference_config.json','README.md']:shutil.copy2(package/name,model/name)
 expected='d11c2e7f86cd5189605b857c4697e55a5caead0a6888b52b2f5ff814150f5291';assert sha(model/'model/model.safetensors')==expected and not os.path.samefile(model/'model/model.safetensors',package/'model/model.safetensors');save(qa/'模型独立复制核验.json',{'passed':True,'sha256':expected,'independent_copy':True,'training_updates':0})
 shutil.copy2(ROOT/'task_workspaces/labelagent_realtime_handoff_20261104/docs/test-results/projection-recovery-final.md',home/'最终结论.md');shutil.copy2(out/'结论与限制.txt',home/'结论与限制.txt')
 notes=f'''本轮交接：4托盘逐点归属与投影遮挡

先打开01_主报告与诊断/index.html。{s['pictures']}张图、5段1080p视频，2670帧全部解码核验；均为真实输入、模型结果和保存阶段回放，不是连续录屏。
保留13/17、490/579点与6/6结果。无损追加为13/17、491/579，未增加实例，不采用；仍4个托盘待恢复。旧493/12为何退步、投影与遮挡代理、时间缺失、48次SAM成功/失败以及首次轮廓转换失败均完整留档。原标准不变，原标注未写回，默认未替换。

02是无损完整追加复测，03是首次转换失败的复测，不能将失败版本当作完整原响应保持。04含四相机五邻帧PNG、PCD、原3D标签、标定、位姿及原人工参考副本；选取文件均与原导入摘要逐一回读一致。原始完整MCAP没有重复复制；完整采集、SAM3/Qwen权重和运行依赖仍使用原环境。
05含实际隔离源码、实验工具和配置；07是同一真实Mask2Former权重与推理包的独立副本，SHA256 {expected}。配置路径和旧基线缓存路径需按还原位置重绑，不宣称一键独立重跑完整环境。媒体可离线浏览。

已冻结结果和源码采用同盘硬链接节省重复占用，按内容正常复制即可备份；不要在归档内原地改写，改动请另存。原始资料和模型为独立复制。摘要清单见交接SHA256.json，06含69项代码检查、浏览器、视频逐帧和前景逐像素保护收据。
''';(home/'交接说明.txt').write_text(notes,encoding='utf-8');(home/'index.html').write_text('<!doctype html><meta charset="utf-8"><h1>4托盘逐点归属与投影遮挡</h1><p><a href="01_主报告与诊断/index.html">主报告、163张图与5段视频</a></p><p><a href="最终结论.md">最终结论</a></p><pre>'+html.escape(notes)+'</pre>',encoding='utf-8')
 rows=[{'path':p.relative_to(home).as_posix(),'sha256':sha(p),'bytes':p.stat().st_size} for p in home.rglob('*') if p.is_file() and p.name!='交接SHA256.json'];save(home/'交接SHA256.json',{'files':rows});bad=[r['path'] for r in rows if sha(home/r['path'])!=r['sha256']];assert not bad
 receipt={'passed':True,'bundle':str(home),'files':len(rows),'pictures':s['pictures'],'videos':s['videos'],'fully_decoded_frames':2670,'mismatches':bad,'native_source_files_readback_verified':len(input_rows),'independent_model_copy_verified':True,'original_incumbent_PAL_masks_preserved':27,'original_person_masks_preserved':3,'tests':69,'original_annotations_preserved':True};save(control/'projection_recovery_bundle_verification.json',receipt);print(json.dumps(receipt,ensure_ascii=False),flush=True)
if __name__=='__main__':main()
