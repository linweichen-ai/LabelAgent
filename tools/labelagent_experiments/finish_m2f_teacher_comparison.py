"""Real paired checkpoint evidence with unedited native logits and originals."""
import hashlib,json,shutil,textwrap
from pathlib import Path
import av,cv2,numpy as np,torch
from PIL import Image,ImageDraw,ImageFont,ImageOps
from run_rgb_layers import ROOT,INC,read,save,sha
from audit_m2f_frozen_queries import OUT,OLD,PACK
from trial_retained_m2f import OUT as TRIAL
from m2f_query_replay import replay,probability

CODE=['m2f_query_replay.py','audit_m2f_frozen_queries.py','trial_retained_m2f.py','audit_retained_teacher.py','finish_m2f_teacher_comparison.py']


def main():
    torch.set_num_threads(2);old=read(OUT/'completed.json');new=read(TRIAL/'audit/completed.json');a=read(OUT/'all_query_metrics.json');b=read(TRIAL/'audit/all_query_metrics.json');plan=read(OUT/'frozen_plan.json');newplan=read(TRIAL/'frozen_plan.json');params=read(PACK/'inference_config.json')['postprocessing'];font=ImageFont.truetype('C:/Windows/Fonts/msyh.ttc',27);small=ImageFont.truetype('C:/Windows/Fonts/msyh.ttc',22);assets=OUT/'assets';assets.mkdir(exist_ok=True);cards=[];checks=[]
    for p in [OUT/'frozen_plan.json',TRIAL/'frozen_plan.json',TRIAL/'audit/frozen_plan.json']:
        assert all(sha(x)==h for x,h in read(p)['protected'].items())
    def canvas(title):
        im=Image.new('RGB',(1920,1080),'#eef3f7');ImageDraw.Draw(im).text((35,25),title,font=font,fill='#152b40');return im
    def paste(im,k,photo,title,notes):
        d=ImageDraw.Draw(im);d.text((35+k*625,90),title,font=small,fill='#152b40');photo=ImageOps.contain(photo,(590,600));im.paste(photo,(35+k*625,135));lines=[]
        for note in notes:lines+=textwrap.wrap(str(note),44)
        for j,t in enumerate(lines[:10]):d.text((35+k*625,755+j*28),t,font=small,fill='#152b40')
    def store(im,name):
        p=assets/name;im.save(p);cards.append(p)
    def overlay(source,mask):
        rgb=np.array(source).copy();rgb[mask!=0]=(.5*rgb[mask!=0]+.5*np.array([0,210,160])).astype('uint8');return Image.fromarray(rgb)
    overview=canvas('训练过的Mask2Former：缓存回放与后续权重真实配对');d=ImageDraw.Draw(overview);lines=['旧d11权重：10份冻结视图 ×100查询，0新增模型推理','后续6820权重：同图/同裁块/同架构/同预处理/同阈值，10次新CPU推理','两版每视图均与冻结原后处理逐像素和segment元数据对照通过','两版均14个PAL查询通过类别门槛，14个全部到达全景输出','不存在“14个已判PAL的查询被后处理整组删除”的证据','原生完整mask另存，旧赢家裁分仅用于解释历史行为，绝不当新修复','新训练0、SAM/VLM调用0、新Agent运行0；原13/17及旧6/6保留','全部原标注和六个结果JSON保持；剩4，不采用新权重']
    for j,t in enumerate(lines):d.text((55,150+j*105),t,font=font,fill='#152b40')
    store(overview,'00_真实执行与查询保留.jpg')
    panel=canvas('四个剩余实例：原生模型查询仍未达到联合要求');d=ImageDraw.Draw(panel);notes=['目标42：原生PAL查询命中13/16，但混入邻43的50点，纯度约20.3%','目标48：原生PAL查询命中27/27，但同时混入47/49，纯度约34.6%','相机1目标47/49：两版已判PAL查询均未命中它们的参考点','目标49可支持4/6的残余查询判NULL、pPAL仅0.0017，且额外面积39160','现有14份PAL原生query均未满足新增209像素预算/完整边界等要求','竞争分配后保留原生像素比例最低87.4%，仍未解决上述实例粘连','以上是暴露采集数据的稀疏投影诊断，不是密集真值或泛化结果']
    for j,t in enumerate(notes):d.text((55,170+j*110),t,font=font,fill='#152b40')
    store(panel,'01_四个剩余目标的实际卡点.jpg')
    for view in plan['views']:
        source=Image.open(view['source']).convert('RGB').crop(tuple(view['source_box']));oldfolder=Path(view['folder']);newfolder=TRIAL/'query_diagnosis'/view['case']/view['view'];entries=[]
        for folder in [oldfolder,newfolder]:
            rgb=np.asarray(Image.open(folder/'panoptic.png')).astype('uint32');ids=rgb[:,:,0]+rgb[:,:,1]*256+rgb[:,:,2]*65536;arrays=np.load(folder/'raw_queries.npz',allow_pickle=False);replayed,segs,rows,_=replay(arrays['class_logits'],arrays['mask_logits'],ids.shape,params);assert np.array_equal(ids,replayed);palids=[s['id'] for s in segs if s['category_id']==1];mask=np.isin(ids,palids);entries.append((overlay(source,mask),len(palids),sum(r['top_class_id']==7 for r in rows)));checks.append({'kind':'full_view_panoptic','case':view['case'],'view':view['view'],'source_logits_sha256':sha(folder/'raw_queries.npz'),'stock_panoptic_exact':True})
        c=canvas(view['case']+' / '+view['view']+'：两版同视图完整全景输出');paste(c,0,source,'固定输入原图',['CPU输入448×576','没有改变分块、阈值或像素预算']);paste(c,1,entries[0][0],'旧训练权重 d11',[f'全景PAL实例 {entries[0][1]}',f'NULL槽位 {entries[0][2]}','历史输出回放，非新推理']);paste(c,2,entries[1][0],'后续保留权重 6820',[f'全景PAL实例 {entries[1][1]}',f'NULL槽位 {entries[1][2]}','本轮真实CPU推理，不训练、不替换默认']);store(c,view['case']+'_'+view['view']+'_paired_panoptic.jpg')
    # Reference-centered display is post-evaluation explanation only; no crop
    # or reference pixel is ever fed back into model/point generation.
    cfg=read(ROOT/'task_workspaces/labelagent_point_ids_20261107/config.json');import sys;sys.path[:0]=[cfg['source_snapshot'],str(Path(cfg['source_snapshot'])/'tools'),cfg['dependency_dir']]
    import evaluate_projected_support as ev
    from projection_audit_core import frame_data,cameras
    data=frame_data(Path(cfg['dataset']),read(Path(cfg['dataset'])/'import_manifest.json'),76,ev);cams=cameras(Path(cfg['dataset']),read(Path(cfg['dataset'])/'import_manifest.json'),ev)
    for target in old['targets']:
        case=target['case'];oid=target['object_id'];cam=0 if case.startswith('camera0') else 1;source=Image.open(INC/'inputs'/(case+'.png')).convert('RGB');src,pix=ev.project_visible(data['xyz'],cams[cam]);i=next(i for i,bx in enumerate(data['boxes']) if bx['obj_id']==oid);samples=pix[data['owner'][src]==i+1];assert len(samples)>=5;lo=samples.min(0);hi=samples.max(0);box=[max(0,int(lo[0])-90),max(0,int(lo[1])-90),min(1920,int(hi[0])+91),min(1536,int(hi[1])+91)];base=source.crop(tuple(box));marked=base.copy();d=ImageDraw.Draw(marked)
        for x,y in samples:d.ellipse((int(x)-box[0]-2,int(y)-box[1]-2,int(x)-box[0]+2,int(y)-box[1]+2),fill='#ff2222')
        for group in ['PAL_argmax_score_kept','all_native_slots']:
            c=canvas(f'{case} / 参考对象{oid} / {group}：仅作冻结后诊断，非生成提示');paste(c,0,marked,'原图及原参考投影点',['红点仅用于本轮事后诊断','原参考不改，不是新二维人工真值']);pairs=[]
            for j,(targets,rows,root,name) in enumerate([(old['targets'],a,OLD,'旧d11'),(new['targets'],b,TRIAL/'query_diagnosis','后续6820')],1):
                summary=next(t for t in targets if t['case']==case and t['object_id']==oid)['groups'][group];r=next(r for r in rows if r['key']==summary[0]['key']);arrays=np.load(root/case/r['view']/'raw_queries.npz',allow_pickle=False);x1,y1,x2,y2=r['source_box'];local=(probability(arrays['mask_logits'],r['query'],(y2-y1,x2-x1))>=.5).astype('uint8');full=np.zeros((1536,1920),np.uint8);full[y1:y2,x1:x2]=local;assert hashlib.sha256(full.tobytes()).hexdigest()==r['native']['full_mask_sha256'];path=OUT/'diagnostic_masks'/f'{case}_{oid}_{group}_{name}.npy';path.parent.mkdir(exist_ok=True);np.save(path,full);detail=next((o for o in r['native']['ownership'] if o['object_id']==oid),{'hits':0,'total':len(samples),'purity':0});paste(c,j,overlay(source,full).crop(tuple(box)),name+' 完整原生mask的局部显示',[r['key'],f'最高类别{r["top_class_id"]} / pPAL {r["p_pallet"]:.4f}',f'命中{detail["hits"]}/{detail["total"]} / 纯度{detail["purity"]:.3f}',f'完整mask面积{r["native"]["pixels"]}',f'新增面积{r["native"]["extra_pixels"]}','全图位图保留，显示裁块不是裁掩码']);pairs.append({'weight':name,'query':r['key'],'native_full_mask_sha256':r['native']['full_mask_sha256'],'reference_only_display_box':box})
            checks.append({'kind':'post_reference_diagnostic_only','case':case,'object_id':oid,'group':group,'whole_masks_exact':True,'pairs':pairs});store(c,f'{case}_obj{oid}_{group}.jpg')
    videos=[]
    for name,photos in [('01_查询归因与门槛说明.mp4',cards[:2]),('02_两版十视图配对全景.mp4',cards[2:12]),('03_四个剩余实例的原生查询.mp4',cards[12:])]:
        v=OUT/name
        with av.open(str(v),'w') as container:
            stream=container.add_stream('libx264',rate=10);stream.width=1920;stream.height=1080;stream.pix_fmt='yuv420p'
            for p in photos:
                arr=np.array(Image.open(p).convert('RGB'))
                for _ in range(40):
                    for packet in stream.encode(av.VideoFrame.from_ndarray(arr,format='rgb24')):container.mux(packet)
            for packet in stream.encode():container.mux(packet)
        cap=cv2.VideoCapture(str(v));count=0
        while True:
            ok,_=cap.read()
            if not ok:break
            count+=1
        cap.release();assert count==40*len(photos);videos.append({'path':name,'frames':count,'sha256':sha(v)})
    pal_a=[r for r in a if r['class_gate'] and r['top_class_id']==0];pal_b=[r for r in b if r['class_gate'] and r['top_class_id']==0];ratio=[r['legacy_selected_pixels']/r['native_pixels'] for r in pal_a if r['native_pixels']]
    summary={'status':'complete_frozen_replay_and_retained_weight_trial','old_cached_query_slots':1000,'retained_weight_query_slots':1000,'new_Mask2Former_CPU_forwards':10,'old_model_new_forwards':0,'new_training':False,'new_SAM_calls':0,'new_VLM_calls':0,'new_Agent_runs':0,'legacy_PAL_class_kept':[len(pal_a),len(pal_b)],'legacy_PAL_postprocess_kept':[sum(r.get('postprocess_gate',False) for r in pal_a),sum(r.get('postprocess_gate',False) for r in pal_b)],'PAL_native_area_person_border_qualified':[sum(r['native']['area_fits'] and not r['native']['person_conflict'] and not r['native']['touches_source_crop_border'] for r in group) for group in [pal_a,pal_b]],'old_selected_over_native_min':min(ratio),'old_selected_ratio_below_0_6':sum(v<.6 for v in ratio),'cached_old_panoptic_exact':10,'new_stock_panoptic_exact':10,'same_images_crops_architecture_processor_recipe_thresholds':True,'old_weight_sha256':'d11c2e7f86cd5189605b857c4697e55a5caead0a6888b52b2f5ff814150f5291','candidate_weight_sha256':newplan['candidate_sha256'],'retained_support':'13/17','retained_original_Agent':'6/6','remaining_misses':4,'adopted':False,'all_originals_and_old_six_exact':True,'audit_sparse_radius':3,'original_final_radii_still_required':[0,3,6],'pictures':len(cards),'videos':3,'engineering_tests':3,'not_dense_truth_or_generalization':True,'display_ROIs_reference_only_never_generation_inputs':True}
    save(OUT/'final_summary.json',summary);save(OUT/'independent_readback.json',{'passed':True,'checks':checks,'all_protected_sources_exact':True});save(OUT/'media_verification.json',{'passed':True,'pictures':len(cards),'videos':videos,'fully_decoded_frames':sum(v['frames'] for v in videos),'saved_stage_replay_not_screen_recording':True})
    text='Mask2Former缓存查询与后续权重对照：旧模型1000槽位只读回放，后续保留6820权重使用相同10份原图/裁块、架构、预处理和后处理做10次新CPU推理。两版各14个PAL查询全部进入原全景输出，0个被整个后处理拒绝；各14个原生完整PAL掩码均不满足新增面积/完整边界等联合条件。目标42/48的高分PAL查询有覆盖但混入相邻实例，纯度约20%/35%，新权重没有修复独立支持。目标49有4/6纯度80%的残余查询却为NULL且面积大，不能靠null提升改判。旧赢家分配保留原生像素比例最低87.4%；原assigned/native比可超过1，不能将其当成掩码保留率，但本批14份即使看真实保留率也全部大于0.6，没有额外可救查询。\n\n全部旧原始数组、失败、新权重结果和完整诊断mask保留。20张图片、3段阶段回放，共800帧完整解码。红参考点和局部显示框仅用于事后归因，绝不反馈到候选生成；不是新人工真值、独立盲测或泛化验证。没有新训练、SAM/VLM调用或新Agent运行。原标注、13/17与原6/6保持，仍4，未采用新模型，也没有用本次暴露采集数据替其他训练任务选检查点。'
    (OUT/'本轮结论.txt').write_text(text,encoding='utf-8');(OUT/'index.html').write_text('<meta charset="utf-8"><title>Mask2Former原生查询与后续权重</title><h1>Mask2Former原生查询归因与已训练权重实测</h1><p>'+text.replace('\n','<br>')+'</p><a href="final_summary.json">实测结论</a> · <a href="all_query_metrics.json">旧1000查询全部指标</a> · <a href="completed.json">旧权重归因</a> · <a href="independent_readback.json">独立回读</a>'+''.join('<p><video controls width="1100" src="'+v['path']+'"></video></p>' for v in videos)+''.join('<p><img width="1100" src="'+p.relative_to(OUT).as_posix()+'"></p>' for p in cards),encoding='utf-8')
    repo=ROOT/'task_workspaces/labelagent_realtime_handoff_20261104'
    for name in CODE:shutil.copy2(Path(__file__).with_name(name),repo/'tools/labelagent_experiments'/name)
    shutil.copy2(Path(__file__).with_name('test_m2f_query_replay.py'),repo/'tests/test_m2f_query_replay.py');shutil.copy2(PACK/'calibrated_postprocess.py',repo/'tests/fixtures/m2f_calibrated.py');save(repo/'docs/test-results/m2f-query-replay.json',summary);(repo/'docs/test-results/m2f-query-replay.md').write_text(text,encoding='utf-8');print(json.dumps(summary,ensure_ascii=False),flush=True)


if __name__=='__main__':main()
