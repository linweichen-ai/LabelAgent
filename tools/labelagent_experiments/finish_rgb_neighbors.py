"""Read back every native input/output and render the failed layer comparison."""
import hashlib,json,shutil,textwrap
from pathlib import Path
import av,cv2,numpy as np
from PIL import Image,ImageDraw,ImageFont,ImageOps
from run_rgb_neighbors import ROOT,OUT,PREVIOUS,base

CODE=['rgb_neighbor_policy.py','run_rgb_neighbors.py','audit_rgb_neighbors.py','review_rgb_neighbor_controls.py','finish_rgb_neighbors.py']


def main():
    plan=base.read(OUT/'frozen_plan.json');generated=base.read(OUT/'generation_completed.json');audit=base.read(OUT/'support_area_audit.json');baseline=base.read(PREVIOUS/'generation_completed.json');old_audit=base.read(PREVIOUS/'support_area_audit.json');controls=base.read(OUT/'semantic_controls/completed.json');assets=OUT/'assets';assets.mkdir(exist_ok=True)
    assert generated['status']=='complete' and generated['unique_real_SAM_calls']==14
    assert all(base.sha(p)==h for p,h in plan['protected'].items())
    assert base.sha(Path(__file__).with_name('rgb_neighbor_policy.py'))==plan['policy_sha256']
    binding=base.read(ROOT/'outputs/labelagent_point_ids_20261107/incumbent_binding.json')['files'];assert all(base.sha(ROOT/p)==h for p,h in binding.items())
    checks=[];native={};rows=generated['rows'];old=[r for r in baseline['rows'] if r['mode']=='soft' and r['kind']=='real']
    for row in rows+old:
        arrays=np.load(row['raw_array_file'],allow_pickle=False);meta=base.read(row['metadata_file']);i=row['prediction_call'];raw=arrays[f'{i}_masks'][row['native_index']].squeeze();full=np.load(Path(row['folder'])/'complete_full_mask.npy',allow_pickle=False);x1,y1,x2,y2=row['box'];assert raw.shape==(y2-y1,x2-x1) and np.isin(raw,[0,1]).all() and np.array_equal(full[y1:y2,x1:x2],raw) and int(raw.sum())==int(full.sum());assert hashlib.sha256(full.tobytes()).hexdigest()==row['mask_sha256']
        assert meta['image_sha256']==base.sha(row['input']) and meta['prior_mask_sha256']==base.sha(row['prior']) and meta['points']==row['points'] and meta['predict_inst_calls']==1
        np.testing.assert_array_equal(arrays[f'{i}_point_coords'],[[p['x'],p['y']] for p in row['points']]);np.testing.assert_array_equal(arrays[f'{i}_point_labels'],[p['label'] for p in row['points']])
        for call,info in enumerate(meta['prior_calls']):
            actual=arrays[f'{call}_actual_mask_input'];expected=cv2.resize(np.load(row['logits'],allow_pickle=False),(actual.shape[2],actual.shape[1]),interpolation=cv2.INTER_LINEAR)[None];np.testing.assert_array_equal(actual,expected)
            for name in ['actual_mask_input','original_hard_mask_input','native_low_res_logits']:
                a=arrays[f'{call}_{name}'];assert hashlib.sha256(a.tobytes()).hexdigest()==info[name]['sha256']
        native[row['key']]=raw;checks.append({'key':row['key'],'actual_input_points_prior_and_full_output_exact':True,'native_low_res_logits_exact':True})
    # Independently reconstruct every neutral wire request, not merely class totals.
    from run_visual_budget import payload
    cp=base.read(OUT/'semantic_controls/frozen_plan.json')
    for r in controls['rows']:
        body=payload(Path(r['image']).read_bytes(),256);assert hashlib.sha256(json.dumps(body,sort_keys=True).encode()).hexdigest()==r['request_sha256'];reply=base.read(r['response']);assert base.sha(r['response'])==r['response_sha256'] and reply['local_vlm']['model_config_sha256']==cp['model_config_sha256'] and reply['local_vlm']['actual_image_token_counts']==[256]
    assert all(base.sha(p)==h for p,h in cp['protected'].items())
    font=ImageFont.truetype('C:/Windows/Fonts/msyh.ttc',27);small=ImageFont.truetype('C:/Windows/Fonts/msyh.ttc',22);cards=[];point_cards=[];comparison=[]
    def canvas(title):
        im=Image.new('RGB',(1920,1080),'#eef3f7');ImageDraw.Draw(im).text((30,25),title,font=font,fill='#18334c');return im
    def annotate(r,mask=True,seams=None):
        rgb=np.array(Image.open(r['input']).convert('RGB'));m=native[r['key']]
        if mask:rgb[m!=0]=(.5*rgb[m!=0]+.5*np.array([0,215,165])).astype('uint8')
        im=Image.fromarray(rgb);d=ImageDraw.Draw(im)
        if seams:
            center=(seams['parent_bbox'][0]+seams['parent_bbox'][2])/2;slope=seams['RGB_slope']
            for value in seams['RGB_seams']:d.line([(0,value-slope*center),(im.width-1,value+slope*(im.width-1-center))],fill='white',width=1)
        for p in r['points']:d.ellipse((p['x']-3,p['y']-3,p['x']+3,p['y']+3),fill='#ffca00' if p['label'] else '#ff00cc')
        return im
    def panel(im,k,photo,title,notes):
        x=30+945*(k%2);y=105+470*(k//2);d=ImageDraw.Draw(im);d.text((x,y),title,font=small,fill='#18334c');photo=ImageOps.contain(photo,(510,395));im.paste(photo,(x,y+40))
        yy=y+45
        for note in notes:
            for line in textwrap.wrap(str(note),17):d.text((x+530,yy),line,font=small,fill='#18334c');yy+=31
    def store(im,name):
        p=assets/name;im.save(p);cards.append(p);return p
    for diag in plan['teacher_diagnostics']:
        teacher=diag['teacher_key'];new=[r for r in rows if r['kind']=='real' and r['teacher_key']==teacher];previous=[r for r in old if r['teacher_key']==teacher];assert len(new)==len(previous)==2
        for r in new:assert r['input_sha256']==previous[0]['input_sha256'] and r['prior_sha256']==previous[0]['prior_sha256'] and r['source_logit_file_sha256']==previous[0]['source_logit_file_sha256'] and r['box']==previous[0]['box']
        c=canvas(teacher+'：图像层线与两组冻结提示（线不是实例真值）')
        for k,r in enumerate(new):panel(c,k,annotate(r,False,diag),r['side'],['黄：正点；紫：邻层负点',f'正负点总数 {len(r["points"])}','先验完整保留、不裁输出','每个教师仍限两组假设'])
        point_cards.append(store(c,teacher+'_frozen_layer_points.jpg'))
        c=canvas(teacher+'：旧连续先验 / 邻层负点，完整原生mask')
        for k,r in enumerate(previous+new):
            source=old_audit if k<2 else audit;detail=next(a for a in source['candidates'] if a['key']==r['key']);radius=next(a for a in source['cases'] if a['case']==r['case'])['radii'][1];ownership=next(a for a in radius['new_mask_ownership'] if a['key']==r['key'])['ownership'];targets=[42,48] if r['case'].startswith('camera0') else [47,49];hits=[a for a in ownership if a['object_id'] in targets]
            notes=[f'原提示/边界通过 {r["valid"]}',f'完整面积 {r["raw_pixels"]}',f'新增面积 {detail["extra_pixels"]}',f'面积通过 {detail["area_fits"]}']+[f'{a["object_id"]}: {a["points"]}点 / 纯度{a["purity"]:.3f}' for a in hits]
            if r.get('source_error'):notes+=['原函数拒绝：负点无法排除']
            panel(c,k,annotate(r),('旧soft ' if k<2 else '邻层负点 ')+r['side'],notes)
        store(c,teacher+'_complete_mask_comparison.jpg')
        def iou(rr):
            a=native[rr[0]['key']]!=0;b=native[rr[1]['key']]!=0;return float((a&b).sum()/max(1,(a|b).sum()))
        comparison.append({'teacher_key':teacher,'old_soft_IoU':iou(previous),'neighbor_IoU':iou(new),'same_RGB_prior_geometry_weights':True,'both_policies_two_hypotheses':True,'new_points':[{k:r[k] for k in ['side','points','valid','raw_pixels']} for r in new]})
    for kind in ['gray_control','white_control']:
        r=next(r for r in rows if r['kind']==kind);c=canvas(kind+'：非空形状不构成托盘类别证据');rr=[x for x in controls['rows'] if x['kind']==kind];panel(c,0,annotate(r),kind,[f'完整原生面积 {r["raw_pixels"]}',f'提示/边界合格 {r["valid"]}','没有把SAM非空输出直接计成托盘'])
        for k,x in enumerate(rr):panel(c,k+1,Image.open(x['image']),x['view'],[x['decision']['object'],f'托盘判定 {x["PAL_passed"]}',f'精确复用 {x["reused"]}'])
        store(c,kind+'_neutral_semantic_controls.jpg')
    newreal=[r for r in rows if r['kind']=='real'];oldreal=[r for r in old];individual=lambda a:sum(r['valid'] and r['area_fits'] and not r['person_conflict'] for r in a['candidates'] if r['kind']=='real' and r['mode']=='soft')
    summary={'status':'complete_rejected_RGB_neighbor_policy','new_actual_SAM_calls':14,'real_photo_SAM_calls':12,'blank_SAM_calls':2,'new_actual_VLM_calls':controls['fresh_unique_VLM_responses'],'control_logical_views':controls['logical_views'],'control_exact_reused_views':controls['reused_logical_views'],'control_PAL_false_positives':controls['PAL_false_positives'],'real_prompt_border_pass_old_new':[sum(r['valid'] for r in oldreal),sum(r['valid'] for r in newreal)],'individual_area_person_prompt_pass_old_new':[individual(old_audit),individual(audit)],'old_and_new_teacher_count':6,'hypotheses_per_teacher_each':2,'one_SAM_call_per_chain':True,'max_points':max(len(r['points']) for r in rows),'Agent_round_limit':3,'new_Agent_runs':0,'retained_support':'13/17','retained_original_Agent':'6/6','remaining_misses':4,'adopted':False,'no_output_mask_clipping':True,'all_originals_and_six_results_exact':True,'same_full_teacher_prior_and_RGB_crop':True,'new_prompt_policy_changes_multiple_point_locations':'exploratory combined policy, not one-factor attribution','independent_dense_truth_or_generalization':False,'real_remaining_semantic_extent_LiDAR_Agent_checks':'not run because original sparse support failed'}
    for case in audit['cases']:
        for radius in case['radii']:assert radius['geometry_area_person_valid_matches']==radius['old_matches']
    c=canvas('邻层负点没有解决独立基座：本批拒绝采用');d=ImageDraw.Draw(c);notes=['同6份完整教师先验、同原图范围、每份同样2组假设；每条仍1次SAM','12组真实候选：提示/边界通过12 → 10，单个面积/人员/提示通过8 → 8','目标48最好20/27点、纯度约42.6%；目标42仍13/16点、最高约24.1%','四个剩余目标均没有通过原67%纯度，保持13/17及旧6/6','两组真实候选又无法排除负点，说明层线/点可能与先验对象冲突','强图像边缘不能直接当物理实例分界；货物边缘和开孔也可能入选','灰白4个逻辑中性视图0托盘误报：1条新回复、3条精确复用','全部完整原生mask与原始低分辨率logits保留，没有裁掩码或增加轮数']
    for j,n in enumerate(notes):d.text((45,150+j*105),n,font=font,fill='#18334c')
    store(c,'summary_rejection_and_next_boundary_issue.jpg')
    comparison_cards=[p for p in cards if p not in point_cards];videos=[]
    for filename,photos in [('01_RGB层线与实际提示.mp4',point_cards),('02_完整输出失败对照与灰白复核.mp4',comparison_cards)]:
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
        cap.release();assert count==len(photos)*40;videos.append({'path':filename,'frames':count,'sha256':base.sha(OUT/filename)})
    summary.update(pictures=len(cards),videos=len(videos));base.save(OUT/'final_summary.json',summary);base.save(OUT/'paired_parent_comparison.json',comparison);base.save(OUT/'independent_readback.json',{'passed':True,'checks':checks,'actual_new_native_outputs_verified':14,'previous_native_outputs_verified':12,'all_originals_and_old_six_exact':True,'all_actual_prior_points_and_native_logits_exact':True,'all_neutral_wire_requests_model_and_256_tokens_exact':True});base.save(OUT/'media_verification.json',{'passed':True,'pictures':len(cards),'videos':videos,'fully_decoded_frames':sum(v['frames'] for v in videos),'saved_stage_replay_not_screen_recording':True})
    text='邻层负点跟进：冻结同6份完整教师先验与RGB输入，每份仍2组假设，每条一次SAM，最多3个点。12次真实图+2次灰白SAM，1条新VLM回复、3条精确复用。真实提示/边界通过12→10，单个面积/人员/提示通过8→8；没有新增剩余实例通过。目标48最好20/27点、纯度42.6%，目标42仍13/16点、最高约24.1%，均未达67%。两组真实候选负点失败；RGB强边缘不等于物理托盘分界，层提示可能与完整先验对象冲突。原13/17、旧6/6、原始标注、全部失败和完整原生输出保留，不裁掩码，不增加轮数或放宽门槛，新Agent0，未采用。该探索同时改变点位置/邻层数量，不能作单因素因果结论或泛化验证。尚未通过原支持的候选没有追加语义/完整范围/可见LiDAR/Agent，也没有记为通过。视频是实测阶段回放，不是现场屏幕录制。'
    (OUT/'本轮结论.txt').write_text(text,encoding='utf-8');(OUT/'index.html').write_text('<meta charset="utf-8"><title>RGB邻层负点完整对照</title><h1>完整输出及失败原因</h1><p>'+text+'</p><p><a href="final_summary.json">实测结论</a> · <a href="independent_readback.json">独立原生回读</a> · <a href="support_area_audit.json">原门槛核验</a></p>'+''.join('<p><video controls width="1100" src="'+v['path']+'"></video></p>' for v in videos)+''.join('<p><img width="1100" src="'+p.relative_to(OUT).as_posix()+'"></p>' for p in cards),encoding='utf-8')
    print(json.dumps(summary,ensure_ascii=False),flush=True)


if __name__=='__main__':main()
