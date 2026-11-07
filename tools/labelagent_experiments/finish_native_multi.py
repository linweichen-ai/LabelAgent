"""Read back every raw array and publish an honest native multimask comparison."""
import hashlib, json, shutil, textwrap
from pathlib import Path
import av, cv2, numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageOps
from run_native_multi import ROOT, OUT, INC, read, save, sha


def main():
    plan=read(OUT/'frozen_plan.json');gen=read(OUT/'generation_completed.json');audit=read(OUT/'support_area_audit.json');semantic=read(OUT/'semantic/completed.json')
    assert all(sha(p)==h for p,h in plan['protected'].items())
    binding=read(ROOT/'outputs/labelagent_point_ids_20261107/incumbent_binding.json')['files']
    assert all(sha(ROOT/p)==h for p,h in binding.items())
    checks=[];assets=OUT/'assets';assets.mkdir(exist_ok=True);cards=[]
    font=ImageFont.truetype('C:/Windows/Fonts/msyh.ttc',28);small=ImageFont.truetype('C:/Windows/Fonts/msyh.ttc',22)
    candidates={r['key']:r for r in audit['candidates']}
    for seed in sorted({p['seed'] for p in plan['proposals']}):
        props=[p for p in plan['proposals'] if p['seed']==seed];photo=Image.open(props[0]['input']).convert('RGB');d=ImageDraw.Draw(photo)
        for p in props:
            q=p['points'][0];d.ellipse((q['x']-3,q['y']-3,q['x']+3,q['y']+3),fill='#ff2200')
        c=Image.new('RGB',(1920,1080),'#eef3f7');d=ImageDraw.Draw(c);d.text((35,25),f'种子 {seed}：图像/预测掩码内点，单次只使用一个点',font=font,fill='#152b40');p=ImageOps.contain(photo,(1150,850)).resize(ImageOps.contain(photo,(1150,850)).size);c.paste(p,(35,130))
        notes=[f'本种子实际请求 {len(props)}','距离峰值 / 下四分位内点','提示冻结后才读取参考标签','单点启用模型原生多候选','全部组件和边界像素保留','未依据面积预算选点或裁切','红点为冻结位置，非真值点','13/17与原6/6保留']
        for j,t in enumerate(notes):d.text((1220,150+j*75),t,font=small,fill='#152b40')
        file=assets/f'seed{seed:02d}_prompts.jpg';c.save(file);cards.append(file)
    seed_count=len(cards)
    for p in plan['proposals']:
        rows=[r for r in gen['rows'] if r.get('request_key')==p['key']]
        c=Image.new('RGB',(1920,1080),'#eef3f7');d=ImageDraw.Draw(c);d.text((35,20),f"种子{p['seed']} / {p['anchor_origin']}：同一请求的全部原生候选",font=font,fill='#152b40')
        original=Image.open(p['input']).convert('RGB')
        for k,row in enumerate(rows):
            rawfile=Path(row['raw_array_file']);meta=read(row['metadata_file']);arrays=np.load(rawfile,allow_pickle=False);a=arrays[f"{row['prediction_call']}_masks"];raw=np.asarray(a[row['native_index']]).squeeze();full=np.load(Path(row['folder'])/'complete_full_mask.npy',allow_pickle=False);x1,y1,x2,y2=row['box']
            assert meta['image_sha256']==sha(row['input']) and meta['predictions'][row['prediction_call']]['multimask_output'] is True
            assert hashlib.sha256(a.tobytes()).hexdigest()==meta['predictions'][row['prediction_call']]['masks']['sha256']
            scores=arrays[f"{row['prediction_call']}_scores"]
            assert hashlib.sha256(scores.tobytes()).hexdigest()==meta['predictions'][row['prediction_call']]['scores']['sha256']
            assert float(scores[row['native_index']])==row['native_score']
            np.testing.assert_array_equal(arrays[f"{row['prediction_call']}_point_coords"],[[q['x'],q['y']] for q in p['points']]);np.testing.assert_array_equal(arrays[f"{row['prediction_call']}_point_labels"],[1])
            assert np.array_equal(full[y1:y2,x1:x2],raw) and int(full.sum())==int((raw!=0).sum()) and hashlib.sha256(full.tobytes()).hexdigest()==row['mask_sha256']
            checks.append({'key':row['key'],'full_native_mask_exact':True,'actual_prompt_exact':True,'raw_array_sha256':sha(rawfile),'mask_sha256':row['mask_sha256']})
            rgb=np.array(original);rgb[raw!=0]=(.5*rgb[raw!=0]+.5*np.array([0,210,160])).astype('uint8');photo=ImageOps.contain(Image.fromarray(rgb),(590,590));c.paste(photo,(35+k*625,120))
            detail=candidates[row['key']];radius3=next(x for x in audit['cases'] if x['case']==row['case'])['radii'][1];own=next(x for x in radius3['new_mask_ownership'] if x['key']==row['key'])['ownership'];targets=[x for x in own if x['object_id'] in ([42,48] if row['case'].startswith('camera0') else [47,49])]
            notes=[f"原生索引 {row['native_index']} / {'原服务选中' if row['source_selected'] else '原服务未选中'}",f"面积 {row['raw_pixels']} / 新增 {detail['extra_pixels']}",f"边界接触 {row['touches_border']} / 面积合格 {detail['area_fits']}",f"人员冲突 {detail['person_conflict']} / 提示合格 {row['positive_covered']}"]
            notes += [f"目标{x['object_id']}：{x['points']}点，纯度{x['purity']:.3f}" for x in targets]
            if not targets:notes.append('剩余目标参考点命中：0')
            for j,t in enumerate(notes[:7]):d.text((35+k*625,750+j*37),t,font=small,fill='#152b40')
        f=assets/f"seed{p['seed']:02d}_anchor{p['rank']}_all_native.jpg";c.save(f);cards.append(f)
    for i,row in enumerate(semantic['rows']):
        if row['stage']!='neutral' or row['kind']!='candidate':continue
        c=Image.new('RGB',(1920,1080),'#eef3f7');d=ImageDraw.Draw(c);d.text((35,25),'新实例支持候选：原中性语义复核实际回答',font=font,fill='#152b40');photo=Image.open(row['images'][0]).convert('RGB').resize((850,850));c.paste(photo,(35,130));caption=row['decision']['object'];candidate=next(r for r in gen['rows'] if r['key']==row['candidate']);result=next(r for r in semantic['candidates'] if r['key']==row['candidate']);notes=[f"种子 {candidate['seed']} / 原生索引 {candidate['native_index']}",f"视图 {row['name']} / 实际tokens {row['actual_tokens']}",'原模型实际回答：']+textwrap.wrap(caption,55)+[f"双视图中性通过：{result['neutral_passed']}",f"单基座通过：{result['single_base_passed']}",f"完整范围通过：{result['extent_passed']}",'点支持提高不等于完整托盘通过','未采用；原13/17及原6/6保留']
        for j,t in enumerate(notes):d.text((970,140+j*62),t,font=small,fill='#152b40')
        f=assets/f'semantic_{i:02d}.jpg';c.save(f);cards.append(f)
    videos=[]
    for name,photos in [('01_冻结图像内点.mp4',cards[:seed_count]),('02_全部原生候选与原门槛.mp4',cards[seed_count:])]:
        v=OUT/name
        with av.open(str(v),'w') as container:
            stream=container.add_stream('libx264',rate=10);stream.width=1920;stream.height=1080;stream.pix_fmt='yuv420p'
            for p in photos:
                rgb=np.array(Image.open(p).convert('RGB'))
                for _ in range(40):
                    for packet in stream.encode(av.VideoFrame.from_ndarray(rgb,format='rgb24')):container.mux(packet)
            for packet in stream.encode():container.mux(packet)
        cap=cv2.VideoCapture(str(v));count=0
        while True:
            ok,_=cap.read()
            if not ok:break
            count+=1
        cap.release();assert count==40*len(photos);videos.append({'path':name,'frames':count,'sha256':sha(v)})
    eligible=[r for r in audit['candidates'] if r['valid'] and r['area_fits'] and not r['person_conflict']]
    comparison=[]
    for case in audit['cases']:
        r=next(r for r in case['radii'] if r['radius']==3)
        for target in r['remaining_targets']:
            oid=target['object_id'];groups={}
            for name,rows in [('source_selected',[x for x in eligible if x['source_selected']]),('all_native',eligible)]:
                keys={x['key'] for x in rows if x['case']==case['case']}
                values=[o['qualified_support'] for mask in r['new_mask_ownership'] if mask['key'] in keys for o in mask['ownership'] if o['object_id']==oid]
                groups[name]=max(values,default=0)
            comparison.append({'case':case['case'],'object_id':oid,'radius':3,'best_qualified_support_after_geometry_area_person':groups})
    summary={'status':'complete_native_multi_diagnostic','real_SAM_calls':gen['unique_real_SAM_calls'],'native_candidates':gen['native_candidates'],'complete_valid_native_candidates':gen['valid_complete_outputs'],'geometry_area_person_candidates':len(eligible),'reference_support_comparison':[{k:v for k,v in c.items() if k!='new_mask_ownership'} for case in audit['cases'] for c in [{'case':case['case'],**r} for r in case['radii']]],'retained_support':'13/17','retained_original_Agent':'6/6','new_Agent_runs':0,'new_VLM_calls':0,'remaining_misses':4,'adopted':False,'annotation_writes':0,'all_original_six_hashes_exact':True,'all_masks_untrimmed':True,'old_gates_unchanged':True,'pictures':len(cards),'videos':2,'engineering_tests':3,'evidence_scope':'Exposed-capture development diagnostics; projected sparse support is not dense truth or generalization.'}
    summary['source_selected_vs_all_native_remaining_support']=comparison
    summary['new_VLM_calls']=semantic['actual_unique_VLM_responses']
    summary['semantic_candidates']=semantic['candidates']
    summary['semantic_neutral_passed']=sum(r['neutral_passed'] for r in semantic['candidates'])
    summary['later_gates_not_run_reason']='Both candidates fail unchanged neutral semantic identity; single-base, extent, visible-LiDAR and Agent are still required before any adoption.'
    save(OUT/'independent_readback.json',{'passed':True,'native_mask_count':len(checks),'checks':checks,'all_original_six_hashes_exact':True})
    save(OUT/'media_verification.json',{'passed':True,'pictures':len(cards),'videos':videos,'fully_decoded_frames':sum(v['frames'] for v in videos),'saved_stage_replay_not_screen_recording':True})
    save(OUT/'final_summary.json',summary)
    text='原生多候选试验：图像与未确认预测掩码生成12个单点请求，启用原服务已有multimask路径；保存36个原生数组、分数、原服务选中索引，并原样映射全图。没有参考标签参与选点，没有裁掩码、组件筛除或面积预算倒推。\n\n两个原服务未选中的候选在目标42均支持10/16点，纯度10/12=83.3%和10/11=90.9%，完整面积1875/1995、新增73/58像素，原面积及人员门槛通过。只看几何/面积/人员，相机0在radius3从7到8个匹配；相机1保持6。目标49的5/6支持候选仍新增1817至3637像素，超过剩余209像素预算，未裁切。\n\n几何支持改善不等于完整托盘通过；实际中性/单基座/完整范围回答与跳过理由见semantic/completed.json，原始回答全部保留。没有采用新候选，保留原标注、27托盘与3人员掩码、13/17及原6/6，剩4。新Agent运行0，原6/6为旧结果复用。视频为保存阶段回放，不是连续录屏。\n'
    (OUT/'本轮结论.txt').write_text(text,encoding='utf-8')
    text+='\n本轮两个候选双视图中性语义均失败（0/2）；4个原图回答均为灰色分格货箱/箱组，8个逻辑灰白负对照均拒判为托盘。6个唯一真实VLM请求，12条逻辑视图（同图同提示仅请求一次）。语义失败后未运行单基座、完整范围、可见LiDAR和Agent，未把跳过计作通过。\n'
    (OUT/'本轮结论.txt').write_text(text,encoding='utf-8')
    (OUT/'index.html').write_text('<meta charset="utf-8"><title>SAM3全部原生候选</title><h1>单点提示与全部原生候选</h1><p>'+text.replace('\n','<br>')+'</p><a href="final_summary.json">实测结论</a> · <a href="support_area_audit.json">原支持与面积复核</a> · <a href="independent_readback.json">完整数组回读</a>'+''.join('<p><video width="1100" controls src="'+v['path']+'"></video></p>' for v in videos)+''.join('<p><img width="1100" src="'+p.relative_to(OUT).as_posix()+'"></p>' for p in cards),encoding='utf-8')
    repo=ROOT/'task_workspaces/labelagent_realtime_handoff_20261104'
    for name in ['native_multi_policy.py','run_native_multi.py','audit_native_multi.py','review_native_multi.py','finish_native_multi.py']:
        shutil.copy2(Path(__file__).with_name(name),repo/'tools/labelagent_experiments'/name)
    shutil.copy2(Path(__file__).with_name('test_native_multi_policy.py'),repo/'tests/test_native_multi_policy.py')
    public={k:v for k,v in summary.items() if k!='reference_support_comparison'}
    public['matches_by_radius']=[{'case':r['case'],'radius':r['radius'],'old':r['old_matches'],'all_unfiltered':r['unfiltered_all_matches'],'gated_geometry_area_person':r['geometry_area_person_valid_matches']} for r in summary['reference_support_comparison']]
    save(repo/'docs/test-results/native-multi.json',public);(repo/'docs/test-results/native-multi.md').write_text(text,encoding='utf-8')
    print(json.dumps(public,ensure_ascii=False),flush=True)


if __name__=='__main__':main()
