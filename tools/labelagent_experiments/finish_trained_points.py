"""Pixel-bound trained-point review and full native-mask diagnostic media."""
import hashlib,json,os,textwrap
from pathlib import Path
import av,cv2,numpy as np
from PIL import Image,ImageDraw,ImageFont,ImageOps
from run_rgb_layers import ROOT,read,save,sha
from paired_source_cue import body
from semantic_subject_guard import compatible_pallet_subject
BASE=Path(os.environ['LABELAGENT_TRAINED_POINTS_OUTPUT'])
REVIEW=Path(os.environ['LABELAGENT_TRAINED_REVIEW_OUTPUT'])
SAM=ROOT/'outputs/labelagent_trained_point_sam_20261108'
OUT=ROOT/'outputs/labelagent_trained_point_report_20261108'

def main():
    if OUT.exists():raise FileExistsError('Preserve final trained-point report')
    p=read(BASE/'frozen_plan.json');rp=read(REVIEW/'frozen_plan.json');sp=read(SAM/'frozen_plan.json');done=read(REVIEW/'completed.json');generation=read(SAM/'generation_completed.json');audit=read(SAM/'post_freeze_audit/support_area_audit.json')
    assert all(sha(f)==h for q in [p,rp,sp] for f,h in q['protected'].items());checks=[]
    for r in done['rows']:
        request=body(*[b['image'] for b in r['bindings']]);assert request==read(r['request_file']) and hashlib.sha256(json.dumps(request,sort_keys=True).encode()).hexdigest()==r['request_sha256'];saved=read(r['response_file']);reply=saved['raw_response'];assert sha(r['response_file'])==r['response_sha256'] and reply['local_vlm']['model_config_sha256']==rp['model_config_sha256'] and reply['local_vlm']['actual_image_token_counts']==[256,256];assert json.loads(reply['choices'][0]['message']['content'])==r['decision'] and compatible_pallet_subject(r['decision'])==r['PAL_subject_evidence'];photo=Image.open(r['source_image']).convert('RGB');x1,y1,x2,y2=r['box'];np.testing.assert_array_equal(np.asarray(photo.crop((x1,y1,x2,y2))),np.asarray(Image.open(r['input']).convert('RGB')));assert r['source_point']==[r['points'][0]['x']+x1,r['points'][0]['y']+y1]
        for b in r['bindings']:
            side=b['side'];box=b['source_canvas_box'];clip=b['clipped_box'];canvas=Image.new('RGB',(side,side),(127,127,127));canvas.paste(photo.crop(tuple(clip)),(clip[0]-box[0],clip[1]-box[1]));np.testing.assert_array_equal(np.asarray(canvas.resize((512,512),Image.Resampling.LANCZOS)),np.asarray(Image.open(b['image']).convert('RGB')));assert sha(b['image'])==b['sha256']
        checks.append({'index':r['index'],'source_pixel_payload_encoding_model_raw_reply_exact':True})
    for r in generation['rows']:
        a=np.load(r['raw_array_file'],allow_pickle=False);meta=read(r['metadata_file']);m=np.load(Path(r['folder'])/'complete_full_mask.npy',allow_pickle=False);raw=a[f"{r['prediction_call']}_masks"][r['native_index']].squeeze();x1,y1,x2,y2=r['box'];np.testing.assert_array_equal(m[y1:y2,x1:x2],raw);assert int(m.sum())==int((raw!=0).sum()) and hashlib.sha256(m.tobytes()).hexdigest()==r['mask_sha256'];assert meta['selected_index']==r['native_index'] and meta['selected_prediction_call']==r['prediction_call'] and meta['points']==r['points'] and meta['image_sha256']==sha(r['input']) and meta['prior_mask_sha256'] is None;np.testing.assert_array_equal(a[f"{r['prediction_call']}_point_coords"],[[q['x'],q['y']] for q in r['points']]);assert sha(Path(r['folder'])/'response.json')==r['response_sha256']
    binding=read(ROOT/'outputs/labelagent_point_ids_20261107/incumbent_binding.json')['files'];assert all(sha(ROOT/f)==h for f,h in binding.items());cfg=read(ROOT/'task_workspaces/labelagent_point_ids_20261107/config.json');reference=read(ROOT/'outputs/labelagent_soft_prior_20261108/frozen_plan.json')['protected'];assert all(sha(cfg[k])==reference[cfg[k]] for k in ['reference_annotations','reference_description'])
    OUT.mkdir();assets=OUT/'assets';assets.mkdir();font=ImageFont.truetype('C:/Windows/Fonts/msyh.ttc',25);small=ImageFont.truetype('C:/Windows/Fonts/msyh.ttc',21);cards=[]
    def canvas(title):
        c=Image.new('RGB',(1920,1080),'#eef3f7');ImageDraw.Draw(c).text((30,25),title,font=font,fill='#16364f');return c
    def panel(c,k,im,title,notes):
        x=30+k*630;d=ImageDraw.Draw(c);d.text((x,100),title,font=small,fill='#16364f');im=ImageOps.contain(im,(585,520));c.paste(im,(x,150));yy=715
        for n in notes:
            for line in textwrap.wrap(str(n),45 if str(n).isascii() else 26):d.text((x,yy),line,font=small,fill='#16364f');yy+=27
    for r in done['rows']:
        c=canvas(f"训练模型提示点 {r['index']:02d}：{r['teacher_key']} / {r['side']}");photo=Image.open(r['input']).convert('RGB');d=ImageDraw.Draw(photo);x,y=r['points'][0]['x'],r['points'][0]['y'];d.ellipse((x-5,y-5,x+5,y+5),fill='#ff3155');panel(c,0,photo,'原教师区域与选点（标记仅展示）',[f"原图点 {r['source_point']}",'旧/后续原生类别图都预测PAL','上/下区域各取最高原始logit点','仅选点，未切割输出掩码']);panel(c,1,Image.open(r['bindings'][0]['image']),'实际模型第一图：129近景',['无标记输入','两图实际编码 [256,256]','未使用参考点、面积预算']);panel(c,2,Image.open(r['bindings'][1]['image']),'实际模型第二图：385上下文',[r['decision']['object'],f"托盘主体初筛 {r['PAL_subject_evidence']}",'初筛不等于完整实例通过']);f=assets/f"point_{r['index']:02d}.jpg";c.save(f,quality=94);cards.append(('points',f))
    for r in audit['candidates']:
        c=canvas(f"完整原生掩码 {r['index']:02d}：不使用先验、不裁切、不换原生选中索引");x1,y1,x2,y2=r['box'];mask=np.load(Path(r['folder'])/'complete_full_mask.npy',allow_pickle=False);photo=np.asarray(Image.open(r['input']).convert('RGB')).copy();local=mask[y1:y2,x1:x2]!=0;overlay=photo.copy();overlay[local]=(0.5*overlay[local]+0.5*np.array([30,210,230])).astype('uint8');ownership=next(x['ownership'] for case in audit['cases'] for radius in case['radii'] if radius['radius']==3 for x in radius['new_mask_ownership'] if x['key']==r['key']);panel(c,0,Image.fromarray(photo),'原始输入区域',[f"原生选中mask面积 {r['raw_pixels']}",f"新增并集像素 {r['extra_pixels']}",f"面积门槛通过 {r['area_fits']}",f"person冲突 {r['person_conflict']}"]);panel(c,1,Image.fromarray(overlay),'完整原生mask覆盖（仅展示叠色）',['全部内部mask数组保留','未按参考点选掩码','未按预算修边或裁切']);notes=[f"对象{o['object_id']}: 覆盖{o['points']} / 纯度{o['purity']:.6f}" for o in ownership];panel(c,2,Image.fromarray(local.astype('uint8')*255),'完整二值掩码与r=3稀疏审计',notes+['稀疏参考不是稠密人工真值','本轮没有新增独立支持']);f=assets/f"mask_{r['index']:02d}.jpg";c.save(f,quality=94);cards.append(('masks',f))
    r3=[next(r for r in c['radii'] if r['radius']==3) for c in audit['cases']];assert sum(r['geometry_area_person_valid_matches'] for r in r3)==13 and all(r['old_matches']==r['geometry_area_person_valid_matches'] for c in audit['cases'] for r in c['radii'])
    summary={'status':'complete_trained_points_no_new_instance','cached_stock_panoptic_replays_exact':6,'old_teacher_sha256':p['old_teacher_sha256'],'later_teacher_sha256':p['later_teacher_sha256'],'new_Mask2Former_forwards':0,'new_training':False,'source_only_point_cues':12,'PAL_subject_cues':len(done['PAL_subject_cues']),'actual_new_VLM_calls':done['actual_VLM_calls_total'],'exact_VLM_cache_reuses':done['exact_prior_reuse'],'actual_new_SAM_calls':generation['actual_new_SAM_calls'],'new_independent_full_native_masks':len(generation['rows']),'complete_no_border_masks':sum(r['valid'] for r in generation['rows']),'area_person_valid_masks':sum(r['valid'] and r['area_fits'] and not r['person_conflict'] for r in audit['candidates']),'r0_r3_r6_instance_support_unchanged':True,'new_accepted_instances':0,'retained_support':'13/17','retained_original_Agent':'6/6','new_Agent_runs':0,'Agent_round_limit':3,'remaining_misses':4,'adopted':False,'no_output_mask_clipping':True,'all_original_six_and_references_exact':True,'area_cap':audit['cap'],'remaining_integer_pixels':209,'not_dense_truth_or_independent_generalization':True}
    c=canvas('两版训练结果同判PAL，仍需实际部位语义和独立实例验证');d=ImageDraw.Draw(c);notes=['六份原始查询回放逐像素复现原全景图：不是新训练或新Mask2Former推理','12个源图点：5个托盘主体初筛通过，7个描述为容器/箱体/地面等','5次新SAM，5个完整未裁切原生mask；4个满足面积与person约束','第3点覆盖目标26/27，但纯度26/79=0.3291，多个实例粘连','第4点更窄，目标只覆盖8/27；第8点覆盖旧实例36且新增481像素超209','第9点主要支持已覆盖对象50；第10点没有参考命中，不能算恢复漏检','r=0/3/6独立匹配均未改善；保留13/17、原6/6，新增Agent0','所有初筛、原生返回、失败、原始标注与三轮等验收门槛保留']
    for i,n in enumerate(notes):d.text((40,140+i*108),n,font=font,fill='#16364f')
    f=assets/'summary_trained_semantic_and_instance_limits.jpg';c.save(f,quality=94);cards.append(('summary',f));videos=[]
    for name,kinds in [('01_训练模型十二点部位实测.mp4',['points']),('02_五张完整掩码与实例验收.mp4',['masks','summary'])]:
        photos=[f for kind,f in cards if kind in kinds]
        with av.open(str(OUT/name),'w') as container:
            stream=container.add_stream('libx264',rate=10);stream.width=1920;stream.height=1080;stream.pix_fmt='yuv420p'
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
        cap.release();assert count==len(photos)*40;videos.append({'path':name,'frames':count,'sha256':sha(OUT/name)})
    summary.update(pictures=len(cards),videos=len(videos));save(OUT/'summary.json',summary);save(OUT/'independent_readback.json',{'passed':True,'source_pixel_request_reply_checks':checks,'source_selected_complete_masks_exact':5,'all_originals_exact':True});save(OUT/'media_verification.json',{'passed':True,'pictures':len(cards),'videos':videos,'fully_decoded_frames':sum(v['frames'] for v in videos),'saved_stage_replay_not_screen_recording':True});text='已训练Mask2Former原生类别约束选点：12点、12次新VLM、5次新SAM，五张完整原生掩码都保留。四张满足面积与person约束，但未提升独立实例支持，仍13/17及原6/6、剩4。高覆盖候选仍粘连，较窄候选支持不足；支持旧实例不算恢复漏检。没有新Agent运行或采用，不裁mask、不增加轮数或放宽门槛。';(OUT/'本轮结论.txt').write_text(text,encoding='utf-8');(OUT/'index.html').write_text('<meta charset="utf-8"><h1>已训练模型提示点与完整实例实测</h1><p>'+text+'</p><a href="summary.json">实测结论</a> · <a href="independent_readback.json">独立回读</a>'+''.join('<p><video controls width="1100" src="'+v['path']+'"></video></p>' for v in videos)+''.join('<p><img width="1100" src="'+f.relative_to(OUT).as_posix()+'"></p>' for _,f in cards),encoding='utf-8');print(summary,flush=True)

if __name__=='__main__':main()
