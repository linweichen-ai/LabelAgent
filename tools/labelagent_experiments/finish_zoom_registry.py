"""Read back four trained teachers, complete native masks and real stage media."""
import hashlib,json,textwrap
from pathlib import Path
import av,cv2,numpy as np
from PIL import Image,ImageDraw,ImageFont,ImageOps
from run_rgb_layers import ROOT,read,save,sha
from paired_source_cue import body
from semantic_subject_guard import compatible_pallet_subject
from native_prompt_guard import evaluate

OUT=ROOT/'outputs/labelagent_zoom_registry_report_20261108'
Z=ROOT/'outputs/labelagent_zoom_teacher_20261108'
R=ROOT/'outputs/labelagent_registry_zoom_teacher_20261108'
CUES=[ROOT/'outputs/labelagent_zoom_cues_20261108',ROOT/'outputs/labelagent_registry_zoom_pool_20261108']
BOXES=[ROOT/'outputs/labelagent_source_layer_boxes_20261108',ROOT/'outputs/labelagent_source_surface_boxes_20261108']

def main():
    if OUT.exists():raise FileExistsError('Preserve completed four-checkpoint report')
    checks=[];allviews=[];calls={'CPU_MF':0,'VLM':0,'SAM':0};pool_total=0
    for folder in [Z,R]:
        p=read(folder/'frozen_plan.json');g=read(folder/'generation_completed.json')
        assert all(sha(f)==h for f,h in p['protected'].items())
        assert g['new_CPU_model_forwards']==12 and not g['new_training'] and g['GPU_calls']==0
        calls['CPU_MF']+=g['new_CPU_model_forwards'];allviews+=g['rows']
        for row in g['rows']:
            f=Path(row['folder']);assert all(sha(f/n)==h for n,h in row['artifacts'].items())
            np.testing.assert_array_equal(np.asarray(Image.open(row['source_image']).convert('RGB').crop(tuple(row['box']))),np.asarray(Image.open(row['input']).convert('RGB')))
            checks.append({'model':row['model'],'view':row['teacher_key'],'all_query_artifacts_and_source_crop_exact':True})
    for folder in CUES:
        p=read(folder/'frozen_plan.json');r=read(folder/'review_completed.json');g=read(folder/'SAM/generation_completed.json');a=read(folder/'audit_summary.json')
        assert all(sha(f)==h for f,h in p['protected'].items());calls['VLM']+=r['actual_new_VLM_calls'];calls['SAM']+=g['actual_new_SAM_calls']
        for cue in p['rows']:
            rr=next(x for x in r['rows'] if x['partition']==cue['index']);v=cue['new_point_views'][0]
            request=body(*[b['image'] for b in v['bindings']]);assert request==read(v['request_file'])
            assert hashlib.sha256(json.dumps(request,sort_keys=True).encode()).hexdigest()==v['request_sha256']
            assert sha(rr['response_file'])==rr['response_sha256'];reply=read(rr['response_file'])['raw_response']
            assert reply['local_vlm']['model_config_sha256']==p['model_config_sha256'] and reply['local_vlm']['actual_image_token_counts']==[256,256]
            assert json.loads(reply['choices'][0]['message']['content'])==rr['decision'] and compatible_pallet_subject(rr['decision'])==rr['PAL_subject_evidence']
            photo=Image.open(cue['source_image']).convert('RGB')
            for b in v['bindings']:
                side=b['side'];box=b['source_canvas_box'];clip=b['clipped_box'];im=Image.new('RGB',(side,side),(127,127,127));im.paste(photo.crop(tuple(clip)),(clip[0]-box[0],clip[1]-box[1]));np.testing.assert_array_equal(np.asarray(im.resize((512,512),Image.Resampling.LANCZOS)),np.asarray(Image.open(b['image']).convert('RGB')))
            checks.append({'cohort':folder.name,'cue':cue['index'],'exact_source_near_context_payload_reply_and_subject_guard':True})
        for arm in a['arms']:
            d=read(folder/'audits'/arm['arm']/'support_area_audit.json');assert all(c['old_matches']==c['geometry_area_person_valid_matches'] for case in d['cases'] for c in case['radii'])
        pool=read(folder/'native_pool_frozen_outputs.json')['rows'];pool_total+=len(pool)
        assert len(pool)==3*len(g['rows'])
        for row in g['rows']:
            raw=np.load(row['raw_array_file'],allow_pickle=False);meta=read(row['metadata_file']);full=np.load(Path(row['folder'])/'complete_full_mask.npy',allow_pickle=False)
            local=raw[f"{row['prediction_call']}_masks"][row['native_index']].squeeze();x1,y1,x2,y2=row['box'];np.testing.assert_array_equal(full[y1:y2,x1:x2],local)
            assert int(full.sum())==int((local!=0).sum()) and hashlib.sha256(full.tobytes()).hexdigest()==row['mask_sha256']
            assert meta['selected_index']==row['native_index'] and meta['points']==row['points'] and meta['image_sha256']==sha(row['input'])
            assert meta['prior_mask_sha256'] is None and evaluate(local,row['points'])['valid']
            checks.append({'cohort':folder.name,'SAM_cue':row['index'],'original_selected_full_native_exact':True})
    for folder in BOXES:
        p=read(folder/'frozen_plan.json');g=read(folder/'generation_completed.json');a=read(folder/'post_freeze_audit/support_area_audit.json');assert all(sha(f)==h for f,h in p['protected'].items());calls['SAM']+=g['actual_new_SAM_calls']
        for row in g['rows']:
            raw=np.load(row['raw_array_file'],allow_pickle=False);meta=read(row['metadata_file']);local=raw[f"{row['prediction_call']}_masks"][row['native_index']].squeeze();full=np.load(Path(row['folder'])/'complete_full_mask.npy',allow_pickle=False);x1,y1,x2,y2=row['box'];np.testing.assert_array_equal(full[y1:y2,x1:x2],local);assert int(full.sum())==int((local!=0).sum()) and hashlib.sha256(full.tobytes()).hexdigest()==row['mask_sha256'];assert evaluate(local,row['points'])['valid'];checks.append({'box_cohort':folder.name,'cue':row['index'],'full_native_mask_no_crop_and_prompt_guard':True})
        assert all(x['old_matches']==x['geometry_area_person_valid_matches'] for case in a['cases'] for x in case['radii'])
    binding=read(ROOT/'outputs/labelagent_point_ids_20261107/incumbent_binding.json')['files'];assert all(sha(ROOT/f)==h for f,h in binding.items())
    cfg=read(ROOT/'task_workspaces/labelagent_point_ids_20261107/config.json');old=read(ROOT/'outputs/labelagent_soft_prior_20261108/frozen_plan.json')['protected'];assert all(sha(cfg[k])==old[cfg[k]] for k in ['reference_annotations','reference_description'])
    assert calls=={'CPU_MF':24,'VLM':14,'SAM':13} and pool_total==24
    OUT.mkdir();assets=OUT/'assets';assets.mkdir();font=ImageFont.truetype('C:/Windows/Fonts/msyh.ttc',28);small=ImageFont.truetype('C:/Windows/Fonts/msyh.ttc',22);tiny=ImageFont.truetype('C:/Windows/Fonts/msyh.ttc',18);cards=[]
    def canvas(title):
        im=Image.new('RGB',(1920,1080),'#edf2f6');ImageDraw.Draw(im).text((28,22),title,font=font,fill='#193a50');return im
    def panel(im,k,photo,title,notes,grid=False):
        x=28+(k%3)*630;y=90+(k//3)*480 if grid else 110;d=ImageDraw.Draw(im);d.text((x,y),title,font=small,fill='#193a50');size=(595,340) if grid else (595,560);show=ImageOps.contain(photo,size);im.paste(show,(x,y+42));yy=y+390 if grid else 750
        for note in notes:
            for line in textwrap.wrap(str(note),55 if str(note).isascii() else 31):d.text((x,yy),line,font=tiny,fill='#193a50');yy+=24
    def overlay(photo,mask):
        a=np.asarray(photo.convert('RGB')).copy();mask=np.asarray(mask)!=0;assert mask.shape==a.shape[:2];a[mask]=(a[mask]*.48+np.array([25,210,235])*.52).astype('uint8');return Image.fromarray(a)
    def add(group,name,im):
        f=assets/(name+'.jpg');im.save(f,quality=93);cards.append((group,f))
    oldrows=read(ROOT/'outputs/labelagent_two_sided_layers_20261108/SAM/generation_completed.json')['rows']
    for i,oldrow in enumerate(oldrows):
        im=canvas('提示框输入对照：相同原图、相同五个点，完整输出均保留');photo=Image.open(oldrow['input']).convert('RGB');x1,y1,x2,y2=oldrow['box']
        for k,(title,folder,row) in enumerate([('原五点、无框',None,oldrow)]+[(title,f,next(x for x in read(f/'generation_completed.json')['rows'] if x['index']==oldrow['index'])) for title,f in zip(['全宽高度提示框','源材质宽度提示框'],BOXES)]):
            full=np.load(Path(row['folder'])/'complete_full_mask.npy',allow_pickle=False);notes=[f"完整返回面积 {row['raw_pixels']}",f"边界触碰 {row.get('touches_border',row.get('border','见原记录'))}"]
            if folder:
                aa=next(x for x in read(folder/'post_freeze_audit/support_area_audit.json')['candidates'] if x['index']==row['index']);notes += [f"额外面积 {aa['extra_pixels']} / 预算通过 {aa['area_fits']}",'提示框只送模型，未裁输出']
            panel(im,k,overlay(photo,full[y1:y2,x1:x2]),title,notes)
        add('boxes',f'box_{i+1:02d}',im)
    keys=[x['teacher_key'] for x in read(Z/'frozen_plan.json')['views']]
    labels={'old':'原单模型 d11c','retained':'保留候选 6820','R5_historical_boundary':'历史边界 R5 c625','R15_PAL_candidate':'托盘候选 R15 4b98'}
    for i,key in enumerate(keys):
        rows=[next(x for x in allviews if x['teacher_key']==key and x['model']==name) for name in labels];photo=Image.open(rows[0]['input']).convert('RGB');im=canvas('真实训练权重局部对照 '+key)
        panel(im,0,photo,'同一源图局部', ['448×576统一输入；CPU FP32','非新增训练；不等同原R5部署配方'],True)
        for k,row in enumerate(rows,1):
            rgb=np.asarray(Image.open(Path(row['folder'])/'panoptic.png')).astype('uint32');ids=rgb[:,:,0]+256*rgb[:,:,1]+65536*rgb[:,:,2];pal=[s['id'] for s in row['segments_info'] if s['category_id']==1];mask=np.isin(ids,pal);panel(im,k,overlay(photo,mask),labels[row['model']],[f'原后处理PAL候选 {len(pal)}；非新增实例',row['model_sha256'][:24]],True)
        panel(im,5,photo,'本局部结论',['保留所有100个原始查询','层间粘连或局部覆盖仍未解决','四权重均未获新增合格实例'],True);add('teachers',f'teacher_{i+1:02d}',im)
    for cohort,folder in enumerate(CUES,1):
        plan=read(folder/'frozen_plan.json');review=read(folder/'review_completed.json')
        for cue in plan['rows']:
            rr=next(x for x in review['rows'] if x['partition']==cue['index']);im=canvas(f"源点语义复核 {cohort}-{cue['index']}：{labels[cue['model']]}");photo=Image.open(cue['input']).convert('RGB');stock=np.asarray(Image.open(cue['MF_stock_mask']))!=0
            panel(im,0,overlay(photo,stock),'原MF后处理完整候选',[f"源图坐标 {cue['source_point']}",f"query {cue['query']}；主体初筛 {rr['PAL_subject_evidence']}",'初筛不是完整基座验收'])
            for k,b in enumerate(cue['new_point_views'][0]['bindings'],1):panel(im,k,Image.open(b['image']).convert('RGB'),'实际模型输入：'+('近景' if k==1 else '上下文'),[rr['decision']['object']] if k==1 else [f"历史完全相同复用 {rr['historical_exact_reuse']}",f"同批完全相同复用 {rr.get('same_batch_reuse',False)}",'原请求、原回复及配置均回读'])
            add('neutral',f'neutral_{cohort}_{cue["index"]:02d}',im)
        pool=read(folder/'native_pool_frozen_outputs.json')['rows'];g=read(folder/'SAM/generation_completed.json')
        for row in g['rows']:
            im=canvas(f"SAM原生三候选 {cohort}-{row['index']}：源选中与替代项分别保留");photo=Image.open(row['input']).convert('RGB');x1,y1,x2,y2=row['box'];options=[x for x in pool if x['sam_request_sha256']==row['sam_request_sha256']]
            for k,r in enumerate(options):
                full=np.load(Path(r['folder'])/'complete_full_mask.npy',allow_pickle=False);panel(im,k,overlay(photo,full[y1:y2,x1:x2]),f"原生索引 {r['native_index']}"+(' / 原服务选中' if r['source_selected'] else ' / 未选中'),[f"完整面积 {r['raw_pixels']}；分数 {r['native_score']:.5f}",f"原生提示守卫 {r['native_prompt_guard']['valid']}",'固定索引分开评估，未按参考挑选','未把三项同时计作三个实例'])
            add('native',f'native_{cohort}_{row["index"]:02d}',im)
    im=canvas('本轮实测：更换真实训练权重与提示方式，未产生可采用的新实例');d=ImageDraw.Draw(im);lines=['4套真实训练权重 × 6个原图局部 = 24次新CPU推理','17个逻辑源点复核：14次新VLM，2次同批复用，1次历史完全相同复用','6次新提示框SAM + 7次新单点SAM = 13次新SAM；另1次准确复用','8组单点请求的24个原生候选全部保留，源选中结果与替代项分开','窄掩码仍不足完整实例；大掩码仍混入相邻托盘层；不裁像素补分','原13/17、历史6/6、原始标注与三轮等门槛均保留；新增Agent运行0','局部/同批诊断，不构成独立场景泛化或稠密像素真值验证']
    for i,line in enumerate(lines):d.text((45,140+i*108),line,font=font,fill='#193a50')
    add('summary','summary',im)
    videos=[]
    for group in ['boxes','teachers','neutral','native','summary']:
        photos=[f for g,f in cards if g==group];name=group+'.mp4'
        with av.open(str(OUT/name),'w') as container:
            stream=container.add_stream('libx264',rate=10);stream.width=1920;stream.height=1080;stream.pix_fmt='yuv420p';stream.options={'crf':'23','preset':'veryfast'}
            for f in photos:
                arr=np.asarray(Image.open(f).convert('RGB'))
                for _ in range(40):
                    for packet in stream.encode(av.VideoFrame.from_ndarray(arr,format='rgb24')):container.mux(packet)
            for packet in stream.encode():container.mux(packet)
        cap=cv2.VideoCapture(str(OUT/name));count=0
        while True:
            ok,_=cap.read()
            if not ok:break
            count+=1
        cap.release();assert count==40*len(photos);videos.append({'path':name,'frames':count,'seconds':count/10,'sha256':sha(OUT/name)})
    summary={'status':'complete_rejected_four_trained_checkpoint_and_box_comparison','actual_new_MF_CPU_forwards':24,'trained_checkpoints':4,'source_ROIs':6,'actual_new_VLM_calls':14,'actual_new_SAM_calls':13,'logical_neutral_views':17,'same_batch_VLM_reuses':2,'historical_exact_VLM_reuses':1,'historical_exact_SAM_reuses':1,'single_point_native_hypotheses_preserved':24,'new_accepted_instances':0,'retained_support':'13/17','retained_historical_Agent':'6/6','new_Agent_runs':0,'Agent_round_limit':3,'remaining_misses':4,'original_six_and_reference_bytes_unchanged':True,'no_mask_clipping':True,'all_original_gates_unchanged':True,'adopted':False,'pictures':len(cards),'videos':len(videos),'saved_stage_replay_not_screen_recording':True}
    save(OUT/'summary.json',summary);save(OUT/'independent_readback.json',{'passed':True,'checks':checks,'calls_verified':calls,'originals_and_full_native_pixels_exact':True});save(OUT/'media_verification.json',{'passed':True,'pictures':len(cards),'videos':videos,'fully_decoded_frames':sum(v['frames'] for v in videos)})
    text='继续完成真实训练权重排查：d11c、6820、历史R5 c625与R15 4b98，同一六个原图局部共24次CPU推理，保留全部查询。R5/R15确实训练过，但原推荐记录未将PAL候选设为稳定默认；统一局部配方也不等同R5原部署配方。新做14次中性双图语义复核及13次SAM，完整返回不裁切。源材质宽度提示框能守住面积，却仍混入相邻层；单点三候选中的窄条覆盖不足，大掩码层间粘连，固定原生索引分臂比较也未新增独立实例。原13/17及历史6/6、原标注和所有门槛保留，新Agent运行0，未采用候选。视频为真实保存阶段回放。'
    (OUT/'本轮结论.txt').write_text(text,encoding='utf-8');(OUT/'index.html').write_text('<meta charset="utf-8"><title>四套训练权重与完整原生掩码对照</title><h1>四套训练权重与完整原生掩码对照</h1><p>'+text+'</p><a href="summary.json">实测汇总</a> · <a href="independent_readback.json">独立回读</a>'+''.join('<p><video controls width="1100" src="'+v['path']+'"></video></p>' for v in videos)+''.join('<p><img width="1100" src="assets/'+f.name+'"></p>' for _,f in cards),encoding='utf-8');print(summary,flush=True)

if __name__=='__main__':main()
