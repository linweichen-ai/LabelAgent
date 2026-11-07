"""Compare full native outputs, prior values and unchanged instance gates."""
import hashlib,json,shutil,textwrap
from pathlib import Path
import av,cv2,numpy as np
from PIL import Image,ImageDraw,ImageFont,ImageOps
from run_soft_prior_trial import ROOT,OUT,read,save,sha

STRENGTH=OUT/'strength_control'
CODE=['soft_prior_capture.py','run_soft_prior_trial.py','audit_soft_prior.py','review_soft_prior_controls.py','run_prior_strength_control.py','audit_prior_strength.py','review_prior_strength_controls.py','finish_soft_prior.py']


def main():
    plan=read(OUT/'frozen_plan.json');main=read(OUT/'generation_completed.json');control=read(STRENGTH/'generation_completed.json');audit=read(OUT/'support_area_audit.json');audit_strength=read(STRENGTH/'support_area_audit.json');semantic=read(OUT/'semantic_controls/completed.json');semantic_strength=read(STRENGTH/'semantic_controls/completed.json');assets=OUT/'assets';assets.mkdir(exist_ok=True);font=ImageFont.truetype('C:/Windows/Fonts/msyh.ttc',27);small=ImageFont.truetype('C:/Windows/Fonts/msyh.ttc',21);cards=[];checks=[]
    for file in [OUT/'frozen_plan.json',STRENGTH/'frozen_plan.json',OUT/'semantic_controls/frozen_plan.json',STRENGTH/'semantic_controls/frozen_plan.json']:
        assert all(sha(p)==h for p,h in read(file)['protected'].items())
    assert read(OUT/'provenance_readback.json')['passed']
    binding=read(ROOT/'outputs/labelagent_point_ids_20261107/incumbent_binding.json')['files'];assert all(sha(ROOT/p)==h for p,h in binding.items())
    groups={'hard':[r for r in main['rows'] if r['mode']=='hard'],'soft':[r for r in main['rows'] if r['mode']=='soft'],'rms_binary':control['rows']};sources={'hard':audit,'soft':audit,'rms_binary':audit_strength}
    def get(pair,arm):return next(r for r in groups[arm] if r['pair']==pair)
    def native(row):
        arrays=np.load(row['raw_array_file'],allow_pickle=False);meta=read(row['metadata_file']);i=row['prediction_call'];j=row['native_index'];raw=arrays[f'{i}_masks'][j].squeeze();full=np.load(Path(row['folder'])/'complete_full_mask.npy',allow_pickle=False);x1,y1,x2,y2=row['box'];assert np.isin(raw,[0,1]).all() and np.array_equal(full[y1:y2,x1:x2],raw) and int(full.sum())==int((raw!=0).sum());assert hashlib.sha256(full.tobytes()).hexdigest()==row['mask_sha256'];assert meta['image_sha256']==sha(row['input']) and meta['prior_mask_sha256']==sha(row['prior']) and meta['points']==row['points'];assert meta['all_native_low_res_logits_retained']
        np.testing.assert_array_equal(arrays[f'{i}_point_coords'],[[p['x'],p['y']] for p in row['points']]);np.testing.assert_array_equal(arrays[f'{i}_point_labels'],[p['label'] for p in row['points']])
        for call,info in enumerate(meta['prior_calls']):
            original=arrays[f'{call}_original_hard_mask_input'];actual=arrays[f'{call}_actual_mask_input'];expected=original if row['mode']=='hard' else cv2.resize(np.load(row['logits'],allow_pickle=False),(original.shape[2],original.shape[1]),interpolation=cv2.INTER_LINEAR)[None];assert np.array_equal(actual,expected)
            for name in ['original_hard_mask_input','actual_mask_input','native_low_res_logits']:
                a=arrays[f'{call}_{name}'];assert hashlib.sha256(a.tobytes()).hexdigest()==info[name]['sha256']
        checks.append({'key':row['key'],'pair':row['pair'],'arm':row.get('arm',row['mode']),'actual_points_prior_and_full_native_mask_exact':True,'native_low_res_logits_exact':True,'mask_sha256':row['mask_sha256']});return raw,arrays,meta
    def canvas(title):
        im=Image.new('RGB',(1920,1080),'#eef3f7');ImageDraw.Draw(im).text((35,25),title,font=font,fill='#152b40');return im
    def panel(c,k,im,title,notes):
        d=ImageDraw.Draw(c);d.text((35+625*k,90),title,font=small,fill='#152b40');im=ImageOps.contain(im,(590,590));c.paste(im,(35+k*625,135));lines=[]
        for t in notes:lines+=textwrap.wrap(str(t),44)
        for j,t in enumerate(lines[:10]):d.text((35+k*625,755+29*j),t,font=small,fill='#152b40')
    def store(c,name):
        p=assets/name;c.save(p);cards.append(p)
    def overlay(photo,m,points):
        rgb=np.array(photo);rgb[m!=0]=(.5*rgb[m!=0]+.5*np.array([0,210,160])).astype('uint8');im=Image.fromarray(rgb);d=ImageDraw.Draw(im)
        for q in points:d.ellipse((q['x']-3,q['y']-3,q['x']+3,q['y']+3),fill='#ffcc00' if q['label'] else '#ff00cc')
        return im
    real_pairs=[p['pair'] for p in plan['proposals'] if p['mode']=='hard' and p['kind']=='real'];teachers=[]
    for pair in real_pairs:
        if get(pair,'hard')['teacher_key'] not in [get(p,'hard')['teacher_key'] for p in teachers]:teachers.append(pair)
    for pair in teachers:
        c=canvas(get(pair,'hard')['teacher_key']+'：实际先验值热图 / 不是标注图')
        arrays_by_arm=[]
        for arm in ['hard','soft','rms_binary']:
            r=get(pair,arm);_,a,_=native(r);arrays_by_arm.append(a['0_actual_mask_input'][0])
        bound=max(float(np.abs(a).max()) for a in arrays_by_arm)
        for k,(arm,a) in enumerate(zip(['hard','soft','rms_binary'],arrays_by_arm)):
            heat=cv2.applyColorMap(np.rint((a/bound+1)*127.5).astype('uint8'),cv2.COLORMAP_TURBO)[:,:,::-1];panel(c,k,Image.fromarray(heat),arm,[f'实际RMS {float(np.sqrt(np.mean(a.astype("float64")**2))):.3f}',f'实际范围 {float(a.min()):.2f} .. {float(a.max()):.2f}','三图使用相同颜色刻度','源前景一致，全部实际数值另存','热图只作数值显示，不裁输出mask'])
        store(c,get(pair,'hard')['teacher_key']+'_prior_values.jpg')
    prior_card_count=len(cards);pair_metrics=[]
    for pair in real_pairs:
        c=canvas(pair+'：同图同点，单次SAM，完整原生输出');base=get(pair,'hard');per={}
        for k,arm in enumerate(['hard','soft','rms_binary']):
            r=get(pair,arm);raw,_,_=native(r);assert r['input_sha256']==base['input_sha256'] and r['prior_sha256']==base['prior_sha256'] and r['points']==base['points'] and r['box']==base['box'];detail=next(x for x in sources[arm]['candidates'] if x['key']==r['key']);radius3=next(x for x in sources[arm]['cases'] if x['case']==r['case'])['radii'][1];ownership=next(x for x in radius3['new_mask_ownership'] if x['key']==r['key'])['ownership'];target_ids=[42,48] if r['case'].startswith('camera0') else [47,49];hits=[x for x in ownership if x['object_id'] in target_ids];notes=[f'原提示约束/完整输出通过 {r["valid"]}',f'原生面积{r["raw_pixels"]} / 新增{detail["extra_pixels"]}',f'面积合格{detail["area_fits"]} / 人员冲突{detail["person_conflict"]}']+[f'目标{x["object_id"]}: {x["points"]}点，纯度{x["purity"]:.3f}' for x in hits]
            if r.get('source_error'):notes+=['原函数拒绝：无法排除负点']
            notes+=['黄正点、紫负点；无输出裁切'];panel(c,k,overlay(Image.open(r['input']).convert('RGB'),raw,r['points']),arm,notes);per[arm]={'valid':r['valid'],'area_fits':detail['area_fits'],'person_conflict':detail['person_conflict'],'extra_pixels':detail['extra_pixels'],'raw_pixels':r['raw_pixels'],'remaining_target_ownership':hits}
        pair_metrics.append({'pair':pair,'arms':per});store(c,pair+'_complete_masks.jpg')
    output_card_count=len(cards)
    for kind in ['gray_control','white_control']:
        pair=next(r['pair'] for r in groups['hard'] if r['kind']==kind);c=canvas(kind+'：提示分割的非空形状仍需语义拒判')
        for k,arm in enumerate(['hard','soft','rms_binary']):
            r=get(pair,arm);raw,_,_=native(r);sem=semantic_strength if arm=='rms_binary' else semantic;rr=[x for x in sem['rows'] if x['kind']==kind and x['mode']==r['mode']];notes=[f'SAM完整输出{r["raw_pixels"]}像素','形状不构成物理托盘证据']+[x['view']+': '+x['decision']['object'] for x in rr]+['本批中性语义均拒判托盘'];panel(c,k,overlay(Image.open(r['input']).convert('RGB'),raw,r['points']),arm,notes)
        store(c,kind+'_actual_outputs_and_semantics.jpg')
    counts={}
    for arm,rr in groups.items():
        real=[r for r in rr if r['kind']=='real'];eligible=[x for x in sources[arm]['candidates'] if x['key'] in {r['key'] for r in real}];counts[arm]={'real_pairs':len(real),'native_valid_prompt_and_border':sum(r['valid'] for r in real),'individual_area_person_prompt_pass':sum(x['valid'] and x['area_fits'] and not x['person_conflict'] for x in eligible),'negative_constraint_fail_pairs':[r['pair'] for r in real if r.get('source_error') and '负点' in r['source_error'].get('message','')]}
    diversities=[]
    for pair in teachers:
        key=get(pair,'hard')['teacher_key'];d={'teacher_key':key}
        for arm,rr in groups.items():
            rows=[r for r in rr if r['kind']=='real' and r['teacher_key']==key];a=np.load(Path(rows[0]['folder'])/'complete_full_mask.npy')!=0;b=np.load(Path(rows[1]['folder'])/'complete_full_mask.npy')!=0;d[arm]=float((a&b).sum()/max(1,(a|b).sum()))
        diversities.append(d)
    overview=canvas('连续先验与等强度对照：修正约束改善，不等于独立实例通过');d=ImageDraw.Draw(overview);notes=['每条链相同原图、原二值前景、正负点、SAM权重及一次调用','有效原提示/完整输出：hard 9/12，连续soft 12/12，等RMS二值 9/12','单个候选面积/人员/提示合格：3 → 8 → 3，全部使用完整原生mask','三组在同一4个剩余目标上均未增加独立实例支持，仍13/17','对照支持连续表示有用；RMS只控制整体强度，尚未分离全部局部因素','上下候选差异扩大，仍须原67%纯度、单基座、完整范围和可见性','42次真实SAM调用（36真实图、6灰白），4个新VLM回复、8逻辑复用','原始标注/六个结果保留；新Agent0，旧6/6保留且不算新版本验收']
    for j,t in enumerate(notes):d.text((55,150+j*105),t,font=font,fill='#152b40')
    store(overview,'summary_counts_and_limitations.jpg')
    overview2=canvas('仍未通过的原因：局部约束、面积与物理实例必须共同满足');d=ImageDraw.Draw(overview2);notes=['目标48：soft可支持21/27点，纯度35%至41.2%，仍混合邻实例','目标42：soft可支持13/16点，最好纯度约29.5%，仍未达到67%','query32上下完整mask IoU：约94% → 43%，不能直接计作两个实例','三个软先验修正的负点都位于连通外部背景，未见该点局部封闭孔洞','灰白12个逻辑中性视图均拒判托盘；不把非空SAM形状当类别证据','真实目标未过原实例支持，未追加单基座/范围/可见LiDAR/Agent','所有旧失败、第三组跟进对照和低分辨率原生logits都保留','保持三轮与面积等门槛；本批是暴露采集数据诊断，不是泛化验证']
    for j,t in enumerate(notes):d.text((55,150+j*105),t,font=font,fill='#152b40')
    store(overview2,'remaining_failures_and_physical_instance_gate.jpg')
    videos=[]
    for name,photos in [('01_实际先验值与等强度对照.mp4',cards[:prior_card_count]),('02_十二组完整原生掩码对比.mp4',cards[prior_card_count:output_card_count]),('03_灰白复核及剩余卡点.mp4',cards[output_card_count:])]:
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
    summary={'status':'complete_native_soft_prior_with_RMS_followup','SAM_calls_total':main['unique_real_SAM_calls']+control['unique_real_SAM_calls'],'real_photo_SAM_calls':36,'gray_white_SAM_calls':6,'new_Mask2Former_forwards':0,'new_training':False,'new_VLM_calls':semantic['fresh_unique_VLM_responses']+semantic_strength['fresh_unique_VLM_responses'],'control_logical_views':semantic['logical_views']+semantic_strength['logical_views'],'control_reused_logical_views':semantic['reused_logical_views']+semantic_strength['reused_logical_views'],'control_PAL_false_positives':semantic['PAL_false_positives']+semantic_strength['PAL_false_positives'],'new_Agent_runs':0,'one_SAM_call_per_chain':True,'Agent_round_limit':3,'counts':counts,'opposed_mask_IoU':diversities,'individual_pool_not_joint_budget_admission':True,'retained_support':'13/17','retained_original_Agent':'6/6','remaining_misses':4,'adopted':False,'all_originals_and_old_six_exact':True,'whole_model_source_unchanged':False,'original_point_segment_unchanged':True,'default_prior_mode':None,'RMS_followup_exploratory_after_pilot':True,'RMS_does_not_isolate_all_local_magnitude_or_interpolation_factors':True,'all_output_components_and_native_low_res_logits_preserved':True,'no_post_mask_clipping':True,'pictures':len(cards),'videos':3,'engineering_tests':19,'not_dense_truth_or_generalization':True}
    save(OUT/'paired_comparison.json',pair_metrics);save(OUT/'final_summary.json',summary);save(OUT/'independent_readback.json',{'passed':True,'complete_native_checks':checks,'all_protected_and_old_six_exact':True});save(OUT/'media_verification.json',{'passed':True,'pictures':len(cards),'videos':videos,'fully_decoded_frames':sum(v['frames'] for v in videos),'saved_stage_replay_not_screen_recording':True})
    text='硬/连续先验配对：冻结6份完整PAL教师原生query，RGB上/下正负点共12组，每组hard与soft各一次SAM，另2组灰白对照。再做只由实际输入RMS决定强度的二值跟进对照14次，合计42次SAM（36真图、6灰白）、4个新VLM回复、8条逻辑精确复用，无新Mask2Former推理或训练、新Agent0。\n\n原提示约束及边界合格9/12→12/12，等RMS二值仍9/12且同3组负点失败；单个候选面积/人员/提示合格3→8→3。全程保存完整原生位图、实际先验数组和SAM低分辨率logits，没有裁mask或按209预算改输入。连续表示改善不能仅用整体强度解释，但RMS并未控制所有局部数值/边界插值因素，此跟进是看到首批结果后的探索性对照。\n\n仍无新剩余实例通过原支持：目标48的21/27点纯度35%至41.2%，目标42的13/16点最高纯度29.5%，均未达67%。query32上下候选IoU94%→43%显示形状分开，不能据此当两个独立实例。三个被修正负点位于连通外部背景，不是其局部闭孔。灰白12逻辑中性复核全部拒判托盘；非空提示分割不是PAL类别证据。真实候选未过支持，单基座/范围/可见LiDAR及Agent未追加、未计通过。原标注、27托盘/3person、13/17和原6/6保持、剩4，无采用。\n\n新API默认关闭且原point_segment.py不改；完整服务源快照包含可选诊断入口变更，不能声称全源代码未变或旧6/6是新版本完整验收。全部旧失败、实际载荷、对照、灰白回复及空间/协议限制保留。本批是暴露采集开发诊断，不是密集真值或独立泛化验证。'
    (OUT/'本轮结论.txt').write_text(text,encoding='utf-8');(OUT/'index.html').write_text('<meta charset="utf-8"><title>硬与连续先验完整对照</title><h1>先验连续信息、等强度对照与完整实例门槛</h1><p>'+text.replace('\n','<br>')+'</p><a href="final_summary.json">完整实测结论</a> · <a href="paired_comparison.json">逐组原门槛比较</a> · <a href="independent_readback.json">全部原生输入输出回读</a> · <a href="semantic_controls/completed.json">灰白实际回复</a>'+''.join('<p><video controls width="1100" src="'+v['path']+'"></video></p>' for v in videos)+''.join('<p><img width="1100" src="'+p.relative_to(OUT).as_posix()+'"></p>' for p in cards),encoding='utf-8')
    repo=ROOT/'task_workspaces/labelagent_realtime_handoff_20261104'
    for name in CODE:shutil.copy2(Path(__file__).with_name(name),repo/'tools/labelagent_experiments'/name)
    for name in ['test_soft_prior.py','test_soft_prior_api.py']:shutil.copy2(Path(__file__).with_name(name),repo/'tests'/name)
    shutil.copy2(ROOT/'task_workspaces/labelagent_soft_prior_20261108/deployment/server.new.py',repo/'tests/fixtures/sam_server_prior.py');save(repo/'docs/test-results/soft-prior.json',summary);(repo/'docs/test-results/soft-prior.md').write_text(text,encoding='utf-8');print(json.dumps({k:v for k,v in summary.items() if k!='opposed_mask_IoU'},ensure_ascii=False),flush=True)


if __name__=='__main__':main()
