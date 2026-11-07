"""Exact native gain readback and unchanged full-mask instance comparison."""
import hashlib,json,textwrap
from pathlib import Path
import av,cv2,numpy as np
from PIL import Image,ImageDraw,ImageFont,ImageOps
from run_nearzero_prior import ROOT,OUT,PARENT,base

CODE=['run_nearzero_prior.py','audit_nearzero_prior.py','review_nearzero_controls.py','finish_nearzero_prior.py']


def main():
    plan=base.read(OUT/'frozen_plan.json');new=base.read(OUT/'generation_completed.json');old=base.read(PARENT/'generation_completed.json');audit=base.read(OUT/'support_area_audit.json');old_audit=base.read(PARENT/'support_area_audit.json');controls=base.read(OUT/'semantic_controls/completed.json');assert new['unique_real_SAM_calls']==14
    assert all(base.sha(p)==h for p,h in plan['protected'].items());binding=base.read(ROOT/'outputs/labelagent_point_ids_20261107/incumbent_binding.json')['files'];assert all(base.sha(ROOT/p)==h for p,h in binding.items())
    originals={r['pair']:r for r in old['rows'] if r['mode']=='soft'};raw={};checks=[];paired=[];assets=OUT/'assets';assets.mkdir(exist_ok=True)
    def native(r):
        if r['key'] in raw:return raw[r['key']]
        a=np.load(r['raw_array_file'],allow_pickle=False);m=base.read(r['metadata_file']);i=r['prediction_call'];b=a[f'{i}_masks'][r['native_index']].squeeze();full=np.load(Path(r['folder'])/'complete_full_mask.npy',allow_pickle=False);x1,y1,x2,y2=r['box'];assert b.shape==(y2-y1,x2-x1) and np.isin(b,[0,1]).all();np.testing.assert_array_equal(full[y1:y2,x1:x2],b);assert int(full.sum())==int(b.sum()) and hashlib.sha256(full.tobytes()).hexdigest()==r['mask_sha256']
        assert m['image_sha256']==base.sha(r['input']) and m['prior_mask_sha256']==base.sha(r['prior']) and m['points']==r['points'] and m['predict_inst_calls']==1 and m['source_sha256']=='ab8861c57fbd275e8d16b5cc9bfbf734b3e4d7226d02b93d2b0ef9173fc49a81'
        np.testing.assert_array_equal(a[f'{i}_point_coords'],[[p['x'],p['y']] for p in r['points']]);np.testing.assert_array_equal(a[f'{i}_point_labels'],[p['label'] for p in r['points']])
        for call,info in enumerate(m['prior_calls']):
            actual=a[f'{call}_actual_mask_input'];expected=cv2.resize(np.load(r['logits'],allow_pickle=False),(actual.shape[2],actual.shape[1]),interpolation=cv2.INTER_LINEAR)[None];np.testing.assert_array_equal(actual,expected);assert info['multimask_output'] is False
            for name in ['actual_mask_input','original_hard_mask_input','native_low_res_logits']:assert hashlib.sha256(a[f'{call}_{name}'].tobytes()).hexdigest()==info[name]['sha256']
        raw[r['key']]=(b,a,m);checks.append({'key':r['key'],'source_points_full_native_output_and_low_res_exact':True});return raw[r['key']]
    for r in new['rows']:
        o=originals[r['pair']];nr,na,nm=native(r);orr,oa,om=native(o);assert r['input_sha256']==o['input_sha256'] and r['prior_sha256']==o['prior_sha256'] and r['points']==o['points'] and r['box']==o['box'] and nm['adapter_sha256']==om['adapter_sha256']
        source=np.load(r['original_teacher_logits'],allow_pickle=False);scaled=np.load(r['logits'],allow_pickle=False);np.testing.assert_array_equal(scaled,(source*r['gain']).astype('float32'));np.testing.assert_array_equal(scaled>=0,source>=0)
        np.testing.assert_array_equal(na['0_original_hard_mask_input'],oa['0_original_hard_mask_input']);np.testing.assert_allclose(na['0_actual_mask_input'],oa['0_actual_mask_input']*r['gain'],rtol=1e-5,atol=1e-12)
        iou=float(((nr!=0)&(orr!=0)).sum()/max(1,((nr!=0)|(orr!=0)).sum()));rms=lambda a:float(np.sqrt(np.mean(a.astype('float64')**2)))
        paired.append({'pair':r['pair'],'kind':r['kind'],'same_RGB_binary_prior_points_single_mask_path_source':True,'gain':r['gain'],'actual_input_RMS_before':rms(oa['0_actual_mask_input']),'actual_input_RMS_after':rms(na['0_actual_mask_input']),'complete_output_IoU':iou,'old_new_area':[o['raw_pixels'],r['raw_pixels']],'old_new_valid':[o['valid'],r['valid']]})
    from run_visual_budget import payload
    cp=base.read(OUT/'semantic_controls/frozen_plan.json')
    for r in controls['rows']:
        body=payload(Path(r['image']).read_bytes(),256);assert hashlib.sha256(json.dumps(body,sort_keys=True).encode()).hexdigest()==r['request_sha256'];reply=base.read(r['response']);assert base.sha(r['response'])==r['response_sha256'] and reply['local_vlm']['model_config_sha256']==cp['model_config_sha256'] and reply['local_vlm']['actual_image_token_counts']==[256]
    assert all(base.sha(p)==h for p,h in cp['protected'].items())
    font=ImageFont.truetype('C:/Windows/Fonts/msyh.ttc',27);small=ImageFont.truetype('C:/Windows/Fonts/msyh.ttc',21);cards=[]
    def canvas(title):
        c=Image.new('RGB',(1920,1080),'#eef3f7');ImageDraw.Draw(c).text((30,25),title,font=font,fill='#14334d');return c
    def annotated(r,mask=True):
        rgb=np.array(Image.open(r['input']).convert('RGB'));m=native(r)[0]
        if mask:rgb[m!=0]=(.5*rgb[m!=0]+.5*np.array([0,215,160])).astype('uint8')
        im=Image.fromarray(rgb);d=ImageDraw.Draw(im)
        for p in r['points']:d.ellipse((p['x']-3,p['y']-3,p['x']+3,p['y']+3),fill='#ffcc00' if p['label'] else '#ff00cc')
        return im
    def panel(c,k,photo,title,notes):
        x=30+630*k;d=ImageDraw.Draw(c);d.text((x,100),title,font=small,fill='#14334d');photo=ImageOps.contain(photo,(590,560));c.paste(photo,(x,145));yy=745
        for note in notes:
            for line in textwrap.wrap(str(note),26):d.text((x,yy),line,font=small,fill='#14334d');yy+=28
    def store(c,name):
        p=assets/name;c.save(p);cards.append(p)
    remaining=[]
    for r in new['rows']:
        if r['kind']!='real':continue
        o=originals[r['pair']];metric=next(p for p in paired if p['pair']==r['pair']);c=canvas(r['pair']+'：只缩放连续先验幅度，完整原生输出')
        panel(c,0,annotated(r,False),'相同RGB / 相同正负点',['黄正点、紫负点；范围一致','完整原教师二值前景仍保留',f'源先验仅乘全局系数{r["gain"]:.3g}','近零输入不等于移除先验嵌入'])
        for k,(row,source) in enumerate([(o,old_audit),(r,audit)],1):
            detail=next(a for a in source['candidates'] if a['key']==row['key']);radius=next(a for a in source['cases'] if a['case']==row['case'])['radii'][1];own=next(a for a in radius['new_mask_ownership'] if a['key']==row['key'])['ownership'];hits=[a for a in own if a['object_id'] in ([42,48] if row['case'].startswith('camera0') else [47,49])]
            if k==2:remaining.append({'pair':r['pair'],'remaining_targets':hits,'area_fits':detail['area_fits'],'extra_pixels':detail['extra_pixels']})
            notes=[f'提示/边界合格 {row["valid"]}',f'原生面积 {row["raw_pixels"]} / 新增 {detail["extra_pixels"]}',f'面积合格 {detail["area_fits"]} / 人员冲突 {detail["person_conflict"]}']+[f'目标{a["object_id"]}: {a["points"]}点 纯度{a["purity"]:.3f}' for a in hits]+[f'实际输入RMS {metric["actual_input_RMS_before" if k==1 else "actual_input_RMS_after"]:.3g}']
            panel(c,k,annotated(row),'原连续先验' if k==1 else '近零幅度先验',notes)
        store(c,r['pair']+'_paired_complete_outputs.jpg')
    real_card_count=len(cards)
    for kind in ['gray_control','white_control']:
        r=next(r for r in new['rows'] if r['kind']==kind);c=canvas(kind+'：实际输出与中性类别复核');panel(c,0,annotated(r),'SAM完整输出',[f'完整面积 {r["raw_pixels"]}',f'提示/边界合格 {r["valid"]}','非空形状不等于托盘证据'])
        for k,v in enumerate([a for a in controls['rows'] if a['kind']==kind],1):panel(c,k,Image.open(v['image']),v['view'],[v['decision']['object'],f'托盘判定 {v["PAL_passed"]}',f'精确缓存复用 {v["reused"]}'])
        store(c,kind+'_neutral_review.jpg')
    oldreal=[r for r in originals.values() if r['kind']=='real'];newreal=[r for r in new['rows'] if r['kind']=='real'];individual=lambda a:sum(r['kind']=='real' and r['mode']=='soft' and r['valid'] and r['area_fits'] and not r['person_conflict'] for r in a['candidates'])
    stable=[]
    for case in audit['cases']:
        for radius in case['radii']:
            assert radius['geometry_area_person_valid_matches']==radius['old_matches'];stable.append({'case':case['case'],'radius':radius['radius'],'matches':radius['geometry_area_person_valid_matches']})
    summary={'status':'complete_rejected_nearzero_continuous_prior','new_actual_SAM_calls':14,'real_photo_SAM_calls':12,'gray_white_SAM_calls':2,'new_actual_VLM_calls':controls['fresh_unique_VLM_responses'],'control_logical_views':controls['logical_views'],'control_exact_reused_views':controls['reused_logical_views'],'control_PAL_false_positives':controls['PAL_false_positives'],'real_prompt_border_pass_old_new':[sum(r['valid'] for r in oldreal),sum(r['valid'] for r in newreal)],'individual_area_person_prompt_pass_old_new':[individual(old_audit),individual(audit)],'individual_counts_not_joint_budget_admission':True,'single_intervention':'global scalar on entire continuous source prior, bounded at 1e-6','same_source_RGB_prior_sign_points_precision_source_and_single_mask_path':True,'not_same_as_absent_mask_embedding':True,'old_original_point_segment_unchanged':True,'new_Mask2Former_forwards':0,'new_training':False,'one_SAM_call_per_chain':True,'Agent_round_limit':3,'new_Agent_runs':0,'retained_support':'13/17','retained_original_Agent':'6/6','remaining_misses':4,'adopted':False,'originals_and_all_six_results_exact':True,'no_post_mask_clipping':True,'dense_truth_or_independent_generalization':False,'later_real_semantic_extent_LiDAR_Agent_checks':'not run because remaining-instance sparse support failed','radii_readback':stable}
    c=canvas('近零先验幅度：完整输出改变，独立实例仍未通过');d=ImageDraw.Draw(c);notes=['仅把连续先验乘全局系数至最大绝对值1e-6；原图、点、前景符号不改','同6份教师、每份2组假设、每条一次SAM；未改原服务入口或Agent轮数',f'真实提示/边界通过 {summary["real_prompt_border_pass_old_new"]}；单个面积/人员/提示 {summary["individual_area_person_prompt_pass_old_new"]}','目标48：上组14/27点纯度41.2%，下组12/27点纯度42.9%','目标42：下组13/16点纯度32.5%；四个剩余实例仍无新增通过','幅度确实改变掩码，但减弱先验不能单独解决完整物理实例识别','输入仍经过mask编码器，不能当作没有先验，也不能只看面积变小','原13/17、旧6/6、原始标注和失败保留；全部完整输出，不裁掩码']
    for j,n in enumerate(notes):d.text((45,145+j*105),n,font=font,fill='#14334d')
    store(c,'summary_gain_effect_and_remaining_instance_failure.jpg')
    videos=[]
    for filename,photos in [('01_十二组完整掩码幅度对照.mp4',cards[:real_card_count]),('02_灰白复核与独立实例门槛.mp4',cards[real_card_count:])]:
        with av.open(str(OUT/filename),'w') as container:
            stream=container.add_stream('libx264',rate=10);stream.width=1920;stream.height=1080;stream.pix_fmt='yuv420p'
            for photo in photos:
                arr=np.array(Image.open(photo).convert('RGB'))
                for _ in range(40):
                    for packet in stream.encode(av.VideoFrame.from_ndarray(arr,format='rgb24')):container.mux(packet)
            for packet in stream.encode():container.mux(packet)
        cap=cv2.VideoCapture(str(OUT/filename));count=0
        while True:
            ok,_=cap.read()
            if not ok:break
            count+=1
        cap.release();assert count==40*len(photos);videos.append({'path':filename,'frames':count,'sha256':base.sha(OUT/filename)})
    summary.update(pictures=len(cards),videos=len(videos));base.save(OUT/'final_summary.json',summary);base.save(OUT/'paired_gain_readback.json',paired);base.save(OUT/'remaining_instance_readback.json',remaining);base.save(OUT/'independent_readback.json',{'passed':True,'native_checks':checks,'matched_pairs':len(paired),'all_source_points_prior_and_complete_outputs_exact':True,'actual_prior_gain_verified_rtol':1e-5,'actual_prior_gain_verified_atol':1e-12,'all_wire_neutral_controls_model_and_256_tokens_exact':True,'all_originals_and_old_six_exact':True});base.save(OUT/'media_verification.json',{'passed':True,'pictures':len(cards),'videos':videos,'fully_decoded_frames':sum(v['frames'] for v in videos),'saved_stage_replay_not_screen_recording':True})
    text='近零连续先验诊断：同6份教师、同RGB/二值前景/正负点，仅将整个源logits乘全局系数使最大绝对值为1e-6，仍是同一单mask先验路径，不等于移除mask嵌入。14次新SAM（12真图、2灰白）、2条新VLM回复及2条精确复用。真实提示/边界通过12→11，单个面积/人员/提示8→9；单个计数不是联合面积准入。目标48上组14/27点纯度41.2%、下组12/27点纯度42.9%，目标42下组13/16点纯度32.5%；四个剩余实例仍无新增通过。幅度改变完整输出范围，减弱先验不能单独解决物理实例识别。本批拒绝采用，原13/17、旧6/6、原始标注与全部失败保持，新Agent0，不裁掩码、不增加轮数或放宽门槛。真实候选未过原实例支持，未追加语义/范围/可见LiDAR/Agent，也未记作通过。灰白4个逻辑视图0托盘误报。全部实际先验输入、原生掩码及低分辨率logits、精确载荷与回复保留；视频为实测阶段回放，不是现场录屏。'
    (OUT/'本轮结论.txt').write_text(text,encoding='utf-8');(OUT/'index.html').write_text('<meta charset="utf-8"><title>近零连续先验完整对照</title><h1>幅度影响与独立实例检查</h1><p>'+text+'</p><a href="final_summary.json">实测结论</a> · <a href="independent_readback.json">独立原生回读</a> · <a href="support_area_audit.json">原门槛核验</a>'+''.join('<p><video controls width="1100" src="'+v['path']+'"></video></p>' for v in videos)+''.join('<p><img width="1100" src="'+p.relative_to(OUT).as_posix()+'"></p>' for p in cards),encoding='utf-8');print(json.dumps(summary,ensure_ascii=False),flush=True)


if __name__=='__main__':main()
