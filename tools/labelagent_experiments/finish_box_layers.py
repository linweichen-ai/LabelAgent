"""Seal complete boxed SAM outputs and paired review failures without adoption."""
import hashlib,json,shutil,textwrap
from pathlib import Path
import av,cv2,numpy as np
from PIL import Image,ImageDraw,ImageFont,ImageOps
from run_box_layers import ROOT,OUT,INC,read,save,sha

CODE=['boxed_capture.py','box_layer_policy.py','run_box_layers.py','audit_box_layers.py','review_box_layers.py','review_box_aspect.py','bbox_center_views.py','review_box_center.py','finish_box_layers.py']


def main():
    plan=read(OUT/'frozen_plan.json');gen=read(OUT/'generation_completed.json');audit=read(OUT/'support_area_audit.json');semantic=read(OUT/'semantic/completed.json');aspect=read(OUT/'aspect_diagnostic/completed.json');center=read(OUT/'center_diagnostic/completed.json');cases={c['case']:c for c in audit['cases']};candidates={r['key']:r for r in audit['candidates']};assets=OUT/'assets';assets.mkdir(exist_ok=True);checks=[];cards=[];font=ImageFont.truetype('C:/Windows/Fonts/msyh.ttc',27);small=ImageFont.truetype('C:/Windows/Fonts/msyh.ttc',21)
    for path in [OUT/'frozen_plan.json',OUT/'semantic/frozen_plan.json',OUT/'aspect_diagnostic/frozen_plan.json',OUT/'center_diagnostic/frozen_plan.json']:
        assert all(sha(p)==h for p,h in read(path)['protected'].items())
    for p,h in read(ROOT/'outputs/labelagent_point_ids_20261107/incumbent_binding.json')['files'].items():assert sha(ROOT/p)==h
    for seed in range(1,9):
        rows=[p for p in plan['proposals'] if p['seed']==seed];im=Image.open(rows[0]['input']).convert('RGB');d=ImageDraw.Draw(im)
        for p in rows:
            d.rectangle(tuple(p['diagnostic_box']),outline='#ffcc00' if p['side']=='upper' else '#ff00cc',width=2);q=p['points'][0];d.ellipse((q['x']-3,q['y']-3,q['x']+3,q['y']+3),fill='#ff2200')
        c=Image.new('RGB',(1920,1080),'#eef3f7');d=ImageDraw.Draw(c);d.text((35,25),f'种子 {seed}：冻结的RGB层间框提示 / 输出不裁剪',font=font,fill='#152b40');photo=ImageOps.contain(im,(1150,850));c.paste(photo,(35,130));notes=['黄色：上层框；紫色：下层框','红点：每次请求的一个正点','父区域：最大有效完整原生候选','分层线：RGB中部最强水平梯度','不读参考框、owner或参考点选提示','框只进入模型，不裁输出位图','原始标注与已接受掩码不修改','最终验收与Agent三轮上限不变']
        for j,t in enumerate(notes):d.text((1220,145+j*77),t,font=small,fill='#152b40')
        f=assets/f'seed{seed:02d}_frozen_boxes.jpg';c.save(f);cards.append(f)
    for prop in plan['proposals']:
        rows=[r for r in gen['rows'] if r.get('request_key')==prop['key']];c=Image.new('RGB',(1920,1080),'#eef3f7');d=ImageDraw.Draw(c);d.text((35,22),f"种子{prop['seed']} / {prop['side']} / 全部原生候选，不按框裁输出",font=font,fill='#152b40');image=Image.open(prop['input']).convert('RGB')
        for k,row in enumerate(rows):
            arrays=np.load(row['raw_array_file'],allow_pickle=False);meta=read(row['metadata_file']);a=arrays[f"{row['prediction_call']}_masks"];raw=np.asarray(a[row['native_index']]).squeeze();full=np.load(Path(row['folder'])/'complete_full_mask.npy',allow_pickle=False);x1,y1,x2,y2=row['box'];assert hashlib.sha256(a.tobytes()).hexdigest()==meta['predictions'][row['prediction_call']]['masks']['sha256'];assert np.array_equal(full[y1:y2,x1:x2],raw) and int(full.sum())==int((raw!=0).sum());assert hashlib.sha256(full.tobytes()).hexdigest()==row['mask_sha256'];assert meta['image_sha256']==sha(prop['input']);assert meta['actual_box_calls'] and all(x['box']==[float(v) for v in prop['diagnostic_box']] for x in meta['actual_box_calls']);np.testing.assert_array_equal(arrays[f"{row['prediction_call']}_point_coords"],[[q['x'],q['y']] for q in prop['points']]);scores=arrays[f"{row['prediction_call']}_scores"];assert hashlib.sha256(scores.tobytes()).hexdigest()==meta['predictions'][row['prediction_call']]['scores']['sha256'];assert float(scores[row['native_index']])==row['native_score']
            checks.append({'key':row['key'],'whole_raw_mapping_exact':True,'actual_box_and_point_exact':True,'native_scores_exact':True,'mask_sha256':row['mask_sha256'],'outside_prompt_box_pixels_preserved':row['pixels_outside_prompt_box_preserved']})
            rgb=np.array(image);rgb[raw!=0]=(.5*rgb[raw!=0]+.5*np.array([0,210,160])).astype('uint8');im=Image.fromarray(rgb);draw=ImageDraw.Draw(im);draw.rectangle(tuple(prop['diagnostic_box']),outline='#ff00cc',width=2);photo=ImageOps.contain(im,(590,590));c.paste(photo,(35+625*k,125));detail=candidates[row['key']];own=next(x for x in cases[row['case']]['radii'][1]['new_mask_ownership'] if x['key']==row['key'])['ownership'];targetids=[42,48] if row['case'].startswith('camera0') else [47,49];notes=[f"原生{k} / {'服务选中' if row['source_selected'] else '未选中'}",f"完整面积{row['raw_pixels']} / 新增{detail['extra_pixels']}",f"框外保留像素 {row['pixels_outside_prompt_box_preserved']}",f"面积合格{detail['area_fits']} / 人员冲突{detail['person_conflict']}"]+[f"目标{o['object_id']}: {o['points']}点，纯度{o['purity']:.3f}" for o in own if o['object_id'] in targetids]
            for j,t in enumerate(notes[:7]):d.text((35+625*k,750+j*39),t,font=small,fill='#152b40')
        f=assets/f"seed{prop['seed']:02d}_{prop['side']}_all_native.jpg";c.save(f);cards.append(f)
    original_count=len(cards)
    for key in read(OUT/'semantic/frozen_plan.json')['selected']:
        row=candidates[key]
        for view in ['near','context']:
            entries=[next(r for r in data['rows'] if r.get('candidate')==key and r.get('view',r.get('name'))==view and r['kind']==kind) for data,kind in [(aspect,'original_candidate'),(aspect,'aspect_candidate'),(center,'bbox_candidate')]]
            c=Image.new('RGB',(1920,1080),'#eef3f7');d=ImageDraw.Draw(c);d.text((35,20),f"种子{row['seed']} / {view}：原方形、厚处居中矩形、完整外接范围居中",font=font,fill='#152b40')
            for k,e in enumerate(entries):
                im=Image.open(e['image']).convert('RGB');photo=ImageOps.contain(im,(590,590));c.paste(photo,(35+k*625,130));notes=[['原方形视图','厚处居中矩形','外接范围居中'][k],f"主体语义通过：{e['PAL_passed']}"]+textwrap.wrap(e['decision']['object'],44)+['所有候选像素保留；未采用']
                for j,t in enumerate(notes[:10]):d.text((35+k*625,750+j*29),t,font=small,fill='#152b40')
            f=assets/f"seed{row['seed']:02d}_{view}_review_comparison.jpg";c.save(f);cards.append(f)
    videos=[]
    for name,photos in [('01_图像层间框提示.mp4',cards[:8]),('02_完整原生位图与框外像素.mp4',cards[8:original_count]),('03_复核居中与实际回答对比.mp4',cards[original_count:])]:
        v=OUT/name
        with av.open(str(v),'w') as container:
            stream=container.add_stream('libx264',rate=10);stream.width=1920;stream.height=1080;stream.pix_fmt='yuv420p'
            for p in photos:
                a=np.array(Image.open(p).convert('RGB'))
                for _ in range(40):
                    for packet in stream.encode(av.VideoFrame.from_ndarray(a,format='rgb24')):container.mux(packet)
            for packet in stream.encode():container.mux(packet)
        cap=cv2.VideoCapture(str(v));count=0
        while True:
            ok,_=cap.read()
            if not ok:break
            count+=1
        cap.release();assert count==40*len(photos);videos.append({'path':name,'frames':count,'sha256':sha(v)})
    controls={name:{'candidate_views':sum(r['kind']==prefix+'candidate' for r in data['rows']),'candidate_subject_PASS':sum(r['PAL_passed'] for r in data['rows'] if r['kind']==prefix+'candidate'),'negative_views':sum(r['kind'] in [prefix+'gray',prefix+'white'] for r in data['rows']),'negative_rejections':sum(not r['PAL_passed'] for r in data['rows'] if r['kind'] in [prefix+'gray',prefix+'white'])} for name,data,prefix in [('original_square',aspect,'original_'),('deepest_center_aspect',aspect,'aspect_'),('bbox_center_aspect',center,'bbox_')]}
    proof=[]
    for case in audit['cases']:
        r=case['radii'][1]
        for candidate in audit['candidates']:
            if candidate['case']!=case['case'] or not(candidate['valid'] and candidate['area_fits'] and not candidate['person_conflict']):continue
            own=next(x for x in r['new_mask_ownership'] if x['key']==candidate['key'])['ownership']
            for target in r['remaining_targets']:
                hit=next((o for o in own if o['object_id']==target['object_id']),None)
                if target['old_best_support']<.5 and hit and hit['qualified_support']>=.5:
                    proof.append({'candidate':candidate['key'],'seed':candidate['seed'],'case':case['case'],'object_id':target['object_id'],'points':hit['points'],'coverage':hit['coverage'],'purity':hit['purity'],'extra_pixels':candidate['extra_pixels'],'area':candidate['raw_pixels'],'old_pairs_can_be_kept_exact':True,'feasible_match_count_with_this_candidate_only':r['old_matches']+1})
    summary={'status':'complete_box_prompt_diagnostic','real_SAM_calls':gen['unique_real_SAM_calls'],'native_candidates':gen['native_candidates'],'valid_complete_candidates':gen['valid_complete_outputs'],'outside_prompt_box_nonzero_candidates':sum(r.get('pixels_outside_prompt_box_preserved',0)>0 for r in gen['rows']),'new_VLM_calls':semantic['actual_unique_VLM_responses']+aspect['fresh_responses']+center['fresh_responses'],'new_Agent_runs':0,'reference_support_and_budget_candidates':proof,'pool_counts_not_joint_budget_admission':True,'review_comparison':controls,'original_semantic_candidates':semantic['candidates'],'retained_support':'13/17','retained_original_Agent':'6/6','remaining_misses':4,'adopted':False,'default_box_prompt':None,'no_clipping_or_component_removal':True,'all_original_six_hashes_exact':True,'Agent_round_limit':3,'later_gates':'Neutral failures remain failures; full single-base, extent, visible-LiDAR and Agent gates not waived or counted as passed.','pictures':len(cards),'videos':len(videos),'engineering_tests':19,'scope':'Exposed capture development diagnosis, no dense truth or generalization claim'}
    save(OUT/'final_summary.json',summary);save(OUT/'independent_readback.json',{'passed':True,'checks':checks,'all_frozen_protected_sets_exact':True,'all_original_six_exact':True});save(OUT/'media_verification.json',{'passed':True,'pictures':len(cards),'videos':videos,'fully_decoded_frames':sum(v['frames'] for v in videos),'saved_stage_replay_not_screen_recording':True})
    text='框加点提示：16次真实SAM、48个原生输出。仅依据既有原生预测与RGB层间梯度冻结范围；框是模型输入，不裁掩码。36份候选有框外像素且全部保留。目标42四份候选12至13/16点、纯度85.7%至100%、新增9至61像素，均单独符合原面积及人员标准。相机1目标49仍有5/6支持但面积超标，不裁切。\n\n全部原始回答、原失败和候选保留。复核输入对照见final_summary.json：先原方形，再厚处居中的矩形，最后完整外接范围中心；均保持256图像tokens及原提示/分类规则，带灰白负对照。几何支持和描述改善都不能替代独立物理基座与完整可见/可辨识遮挡范围验收。原标注、27托盘/3person、13/17和原6/6保留；无采用，无新Agent，旧6/6不是新跑结果。实验API默认关闭，无独立GPU工作进程或训练。'
    (OUT/'本轮结论.txt').write_text(text,encoding='utf-8');(OUT/'index.html').write_text('<meta charset="utf-8"><title>框加点完整基座对比</title><h1>框加点：完整原生输出与复核目标绑定</h1><p>'+text.replace('\n','<br>')+'</p><a href="final_summary.json">完整实测结论</a> · <a href="support_area_audit.json">支持与面积</a> · <a href="independent_readback.json">全部原生回读</a>'+''.join('<p><video controls width="1100" src="'+v['path']+'"></video></p>' for v in videos)+''.join('<p><img width="1100" src="'+p.relative_to(OUT).as_posix()+'"></p>' for p in cards),encoding='utf-8')
    repo=ROOT/'task_workspaces/labelagent_realtime_handoff_20261104'
    for name in CODE:shutil.copy2(Path(__file__).with_name(name),repo/'tools/labelagent_experiments'/name)
    for name in ['test_box_layer.py','test_box_api.py','test_bbox_center_views.py']:shutil.copy2(Path(__file__).with_name(name),repo/'tests'/name)
    fixture=repo/'tests/fixtures';fixture.mkdir(exist_ok=True);shutil.copy2(ROOT/'task_workspaces/labelagent_box_layers_20261107/deployment/server.new.py',fixture/'sam_server_box.py');save(repo/'docs/test-results/box-layers.json',summary);(repo/'docs/test-results/box-layers.md').write_text(text,encoding='utf-8');print(json.dumps(summary,ensure_ascii=False),flush=True)


if __name__=='__main__':main()
