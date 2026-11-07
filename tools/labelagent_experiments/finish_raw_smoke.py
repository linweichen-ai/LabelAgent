"""Read back actual captured arrays, encode browser media, seal local evidence."""
import hashlib,json,shutil,io,base64,datetime
from pathlib import Path
import numpy as np,cv2,av
ROOT=Path(__file__).resolve().parents[2];OUT=ROOT/'outputs/labelagent_raw_smoke_20261107';CONTROL=ROOT/'task_workspaces/labelagent_handoff_control_20261104';REPO=ROOT/'task_workspaces/labelagent_realtime_handoff_20261104'
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def save(p,v):Path(p).write_text(json.dumps(v,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
s=json.loads((OUT/'completed.json').read_text(encoding='utf-8'));checks=[]
for row in s['rows']:
 folder=OUT/row['kind'];manifest=json.loads((folder/'capture.json').read_text(encoding='utf-8'));a=np.load(folder/'received_arrays.npz',allow_pickle=False);raw=a[str(manifest['selected_prediction_call'])+'_masks'][manifest['selected_index']].squeeze()!=0
 wire=np.asarray(__import__('PIL.Image',fromlist=['Image']).open(folder/'wire.png'))!=0
 assert int(raw.sum())==row['raw_pixels'] and int(wire.sum())==row['wire_pixels']
 assert int((raw&~wire).sum())==row['removed_pixels'] and int((wire&~raw).sum())==row['added_pixels']
 assert manifest['image_sha256']==sha(folder/'input.png')
 checks.append({'kind':row['kind'],'exact_array_readback':True,'raw_to_wire_changes_recomputed':True})
v=OUT/'内部位图与返回轮廓_真实推理回放.mp4';v.rename(OUT/'原始阶段回放_mp4v.mp4')
with av.open(str(v),'w') as container:
 stream=container.add_stream('libx264',rate=10);stream.width=1920;stream.height=1080;stream.pix_fmt='yuv420p'
 for kind in ('real','gray'):
  for name in ('raw','wire'):
   image=cv2.cvtColor(cv2.imread(str(OUT/kind/(name+'_comparison.jpg'))),cv2.COLOR_BGR2RGB)
   for _ in range(50):
    for packet in stream.encode(av.VideoFrame.from_ndarray(image,format='rgb24')):container.mux(packet)
 for packet in stream.encode():container.mux(packet)
cap=cv2.VideoCapture(str(v));frames=0
while True:
 ok,_=cap.read()
 if not ok:break
 frames+=1
cap.release();assert frames==200
addendum={'passed':True,'checks':checks,'api_CPU_tests_passed':4,'decoded_frames':frames,'capture_route':'existing managed HTTP optional capture, private identical source uses shared service cache','service_diagnostic_installed':True,'default_prediction_path_unchanged':True,'capture_stage_service_modified':False,'summary_scope_correction':'completed.json service_modified=false refers only to inference stage; earlier opt-in installation is recorded separately','real_requests':1,'synthetic_gray_requests':1,'new_candidate_adopted':False,'no_instance_recovery_claim':True}
save(OUT/'verified_addendum.json',addendum)
text='''真实 SAM 内部掩码捕获（2026-11-07）

通过现有管理器真实执行原图一次及灰图一次。原图内部4024像素、10组件，返回3973像素：删除66、增加15，无warning；灰图内部944像素、7组件，返回927：删除25、增加8，无warning。内部float32只含0/1，全部候选及selected索引、原请求与实际返回均已保存并独立逐像素回算。该结果证明当前真实请求后处理会改像素；不能推算历史全部损失，也不能宣布四个漏检已恢复。

新include_raw_capture选项默认关闭，四项API合成检查通过。安装时SAM worker未加载，未重启管理器；通过原有管理器正常加载共享模型，未并发启动第二份模型。point_segment原处理函数保持原SHA，全部源输入与参考文件前后SHA相同。completed.json中service_modified=false仅指推理阶段；部署变更详见deployment/receipt.json及verified_addendum.json。

保留13/17及原6/6、剩4；没有新Agent运行或采用候选，不修改标注、轮数、面积和其他验收。下一步要用完整内部位图核对每个组件的实例归属，而不是拿被筛选后的轮廓面积当完整内部面积。
'''
(OUT/'本轮结论.txt').write_text(text,encoding='utf-8')
docs=REPO/'docs/test-results';(docs/'raw-smoke.md').write_text(text,encoding='utf-8');save(docs/'raw-smoke.json',{'real':s['rows'][0],'gray':s['rows'][1],'real_SAM_calls':1,'gray_SAM_calls':1,'adopted':False,'remaining_misses':4,'retained_support':'13/17','retained_original_Agent':'6/6','API_CPU_tests':4,'original_point_processor_unchanged':True})
for name in ('run_raw_smoke.py','finish_raw_smoke.py','deploy_raw_diagnostic.py'):shutil.copy2(ROOT/'tools/labelagent_live_handoff'/name,REPO/'tools/labelagent_experiments'/name)
home=ROOT/'deliverables/LabelAgent_实时交接/09_补充成果/真实SAM内部掩码捕获_20261107'
assert not home.exists();shutil.copytree(OUT,home);shutil.copytree(ROOT/'task_workspaces/labelagent_raw_smoke_20261107/deployment',home/'deployment')
files=[{'path':p.relative_to(home).as_posix(),'sha256':sha(p)} for p in home.rglob('*') if p.is_file()]
assert all(sha(home/f['path'])==f['sha256'] for f in files);save(home/'交接SHA256.json',{'files':files})
save(CONTROL/'raw_smoke_bundle_verification.json',{'passed':True,'bundle':str(home),'files':len(files),'pictures':4,'videos_for_gallery':1,'decoded_frames':200,'raw_readback':checks})
print(json.dumps({'passed':True,'bundle':str(home),'files':len(files),'frames':frames}),flush=True)
