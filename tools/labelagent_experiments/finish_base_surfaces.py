"""Read back source bands, every prompt, native arrays and paired failure media."""
import hashlib,json,textwrap,sys
from pathlib import Path
import av,cv2,numpy as np
from PIL import Image,ImageDraw,ImageFont,ImageOps
from run_rgb_layers import ROOT,read,save,sha
from paired_source_cue import body
from semantic_subject_guard import compatible_pallet_subject
from base_structure_points import propose as edges
from base_material_band import propose as material
from base_blue_surface import propose as blue
BASE=ROOT/'outputs/labelagent_trained_partition_20261108';M=ROOT/'outputs/labelagent_base_material_20261108';B=ROOT/'outputs/labelagent_base_blue_surface_20261108';OUT=ROOT/'outputs/labelagent_base_surfaces_report_20261108'

def main():
    if OUT.exists():raise FileExistsError('Preserve completed source-band report')
    base=read(BASE/'partition_plan.json');previous=read(BASE/'SAM/generation_completed.json');trials={};checks=[]
    for name,folder,policy in [('material',M,material),('blue',B,blue)]:
        plan=read(folder/'frozen_plan.json');review=read(folder/'review_completed.json');g=read(folder/'SAM/generation_completed.json');audit=read(folder/'post_freeze_audit/support_area_audit.json');assert all(sha(f)==h for f,h in plan['protected'].items())
        for r in base['rows']:
            photo=Image.open(r['input']).convert('RGB');parent=np.asarray(Image.open(r['prior']).convert('L'))>127;anchor=[r['points'][0]['x'],r['points'][0]['y']];p,d=policy(np.asarray(photo),parent,anchor);ep,ed=edges(np.asarray(photo),parent,anchor);saved=next(x for x in plan['diagnostics'] if x['partition']==r['index']);assert p==saved['material_proposal'] and d==saved['material_diagnostics'] and ep==saved['edge_proposal'] and ed==saved['edge_diagnostics'];checks.append({'trial':name,'partition':r['index'],'source_only_policy_replayed_exact':True})
        for r in plan['rows']:
            original=next(x for x in base['rows'] if x['index']==r['source_partition_index']);assert r['points'][:2]==original['points'] and r['points'][2:]==r['new_positive_points'] and len(r['points'])==4 and r['input']==original['input'];photo=Image.open(r['source_image']).convert('RGB');x1,y1,x2,y2=r['box'];np.testing.assert_array_equal(np.asarray(photo.crop((x1,y1,x2,y2))),np.asarray(Image.open(r['input']).convert('RGB')))
            for v in r['new_point_views']:
                assert v['source_point']==[v['new_point']['x']+x1,v['new_point']['y']+y1];req=body(*[b['image'] for b in v['bindings']]);assert req==read(v['request_file']) and hashlib.sha256(json.dumps(req,sort_keys=True).encode()).hexdigest()==v['request_sha256'];rr=next(x for x in review['rows'] if x['partition']==r['source_partition_index'] and x['request_sha256']==v['request_sha256']);reply=read(rr['response_file'])['raw_response'];assert sha(rr['response_file'])==rr['response_sha256'] and reply['local_vlm']['model_config_sha256']==plan['model_config_sha256'] and reply['local_vlm']['actual_image_token_counts']==[256,256];assert json.loads(reply['choices'][0]['message']['content'])==rr['decision'] and compatible_pallet_subject(rr['decision'])==rr['PAL_subject_evidence']
                for b in v['bindings']:
                    side=b['side'];box=b['source_canvas_box'];clip=b['clipped_box'];c=Image.new('RGB',(side,side),(127,127,127));c.paste(photo.crop(tuple(clip)),(clip[0]-box[0],clip[1]-box[1]));np.testing.assert_array_equal(np.asarray(c.resize((512,512),Image.Resampling.LANCZOS)),np.asarray(Image.open(b['image']).convert('RGB')));assert sha(b['image'])==b['sha256']
        for r in g['rows']:
            views=[v for v in review['rows'] if v['partition']==r['source_partition_index']];assert len(views)==2 and all(v['PAL_subject_evidence'] for v in views);a=np.load(r['raw_array_file'],allow_pickle=False);meta=read(r['metadata_file']);m=np.load(Path(r['folder'])/'complete_full_mask.npy',allow_pickle=False);raw=a[f"{r['prediction_call']}_masks"][r['native_index']].squeeze();x1,y1,x2,y2=r['box'];np.testing.assert_array_equal(m[y1:y2,x1:x2],raw);assert int(m.sum())==int((raw!=0).sum()) and hashlib.sha256(m.tobytes()).hexdigest()==r['mask_sha256'];assert meta['selected_index']==r['native_index'] and meta['points']==r['points'] and meta['image_sha256']==sha(r['input']) and meta['prior_mask_sha256'] is None;np.testing.assert_array_equal(a[f"{r['prediction_call']}_point_coords"],[[q['x'],q['y']] for q in r['points']]);np.testing.assert_array_equal(a[f"{r['prediction_call']}_point_labels"],[q['label'] for q in r['points']]);assert sha(Path(r['folder'])/'response.json')==r['response_sha256'];checks.append({'trial':name,'partition':r['source_partition_index'],'all_four_points_and_source_selected_native_mask_exact':True})
        assert all(rr['old_matches']==rr['geometry_area_person_valid_matches'] for c in audit['cases'] for rr in c['radii']);trials[name]={'plan':plan,'review':review,'generation':g,'audit':audit}
    binding=read(ROOT/'outputs/labelagent_point_ids_20261107/incumbent_binding.json')['files'];assert all(sha(ROOT/f)==h for f,h in binding.items());cfg=read(ROOT/'task_workspaces/labelagent_point_ids_20261107/config.json');old_protected=read(ROOT/'outputs/labelagent_soft_prior_20261108/frozen_plan.json')['protected'];assert all(sha(cfg[k])==old_protected[cfg[k]] for k in ['reference_annotations','reference_description']);sys.path[:0]=[cfg['source_snapshot'],str(Path(cfg['source_snapshot'])/'tools'),cfg['dependency_dir']]
    from source.agent.edit_agent import _rasterize
    old_masks={}
    for cam in [0,1]:old_masks[f'camera{cam}_frame76']=[_rasterize(s,1920,1536) for s in read(ROOT/f'outputs/labelagent_context_comparison_20261105/C_hybrid/camera{cam}_frame76/result.json')['shapes'] if s['label']=='pallet']
    combined_extra=0
    for case,ms in old_masks.items():
        old=np.maximum.reduce(ms);new=old.copy()
        for t in trials.values():
            for r in t['generation']['rows']:
                if r['case']==case:new|=np.load(Path(r['folder'])/'complete_full_mask.npy',allow_pickle=False)
        combined_extra+=int(((new!=0)&(old==0)).sum())
    assert 122302+combined_extra<=122511.25
    OUT.mkdir();assets=OUT/'assets';assets.mkdir();font=ImageFont.truetype('C:/Windows/Fonts/msyh.ttc',25);small=ImageFont.truetype('C:/Windows/Fonts/msyh.ttc',20);cards=[]
    def canvas(title):
        c=Image.new('RGB',(1920,1080),'#eef3f7');ImageDraw.Draw(c).text((30,25),title,font=font,fill='#16364f');return c
    def panel(c,k,im,title,notes):
        x=30+k*630;d=ImageDraw.Draw(c);d.text((x,100),title,font=small,fill='#16364f');im=ImageOps.contain(im,(585,520));c.paste(im,(x,150));yy=715
        for n in notes:
            for line in textwrap.wrap(str(n),47 if str(n).isascii() else 28):d.text((x,yy),line,font=small,fill='#16364f');yy+=26
    def overlay(photo,m):
        a=np.asarray(photo).copy();a[m]=(a[m]*.5+np.array([30,210,230])*.5).astype('uint8');return Image.fromarray(a)
    comparisons=[]
    for r in base['rows']:
        index=r['index'];photo=Image.open(r['input']).convert('RGB');marked=photo.copy();d=ImageDraw.Draw(marked)
        for q in r['points']:
            x,y=q['x'],q['y'];d.ellipse((x-4,y-4,x+4,y+4),fill='#ee3566' if q['label'] else '#ffc533')
        c=canvas(f'基座部位提示 {index}：源图补点，逐点复核后才生成');panel(c,0,marked,'原输入、原正点与未确认负点',['红=原正点，黄=负点假设','源图标记仅用于展示','线段方法未找到合格成对边界','补充点不读取参考或预算'])
        for col,(name,title) in enumerate([('material','与原点相连的材质带'),('blue','避开过暗部位的蓝色表面')],1):
            t=trials[name];prop=next((x for x in t['plan']['rows'] if x['source_partition_index']==index),None);diag=next(x for x in t['plan']['diagnostics'] if x['partition']==index)['material_diagnostics'];im=photo.copy();notes=[]
            if prop:
                dd=ImageDraw.Draw(im)
                for q in prop['new_positive_points']:
                    x,y=q['x'],q['y'];dd.ellipse((x-4,y-4,x+4,y+4),fill='#2fc87c')
                for rr in [x for x in t['review']['rows'] if x['partition']==index]:notes += [rr['decision']['object'],f"新点{rr['new_point_index']}主体初筛 {rr['PAL_subject_evidence']}"]
            else:notes=[diag.get('reason','No valid distributed source proposal')]
            notes += [f"源材质跨度 {diag.get('span','NA')} 像素",'连通区域只选提示，不裁输出']
            panel(c,col,im,title+'（绿色新正点）',notes)
        f=assets/f'points_{index:02d}.jpg';c.save(f,quality=94);cards.append(('points',f));c=canvas(f'完整返回前后对照 {index}：同一原输入、源选中索引规则不变');old=next(x for x in previous['rows'] if x['index']==index);oldmask=np.load(Path(old['folder'])/'complete_full_mask.npy',allow_pickle=False);assert hashlib.sha256(oldmask.tobytes()).hexdigest()==old['mask_sha256'];x1,y1,x2,y2=r['box'];panel(c,0,overlay(photo,oldmask[y1:y2,x1:x2]!=0),'原两点完整SAM返回',[f"面积 {old['raw_pixels']}",'每组新方法仅增加2个正点','原正点、原负点仍在','全返回数组保留，不裁掩码'])
        for col,(name,title) in enumerate([('material','材质带补点'),('blue','蓝色表面补点')],1):
            t=trials[name];new=next((x for x in t['audit']['candidates'] if x['source_partition_index']==index),None)
            if new:
                full=np.load(Path(new['folder'])/'complete_full_mask.npy',allow_pickle=False);ownership=next(x['ownership'] for case in t['audit']['cases'] for radius in case['radii'] if radius['radius']==3 for x in radius['new_mask_ownership'] if x['key']==new['key']);im=overlay(photo,full[y1:y2,x1:x2]!=0);notes=[f"面积{new['raw_pixels']} / 新增{new['extra_pixels']}",f"面积通过{new['area_fits']} / 人员冲突{new['person_conflict']}"]+[f"对象{o['object_id']}: 命中{o['points']} / 纯度{o['purity']:.6f}" for o in ownership];comparisons.append({'trial':name,'partition':index,'old_pixels':old['raw_pixels'],'new_pixels':new['raw_pixels'],'extra_pixels':new['extra_pixels'],'r3_ownership':ownership})
            else:im=photo;notes=['源图无合格提案或新增点初筛失败','没有发起SAM','不使用缺失结果充当成功']
            panel(c,col,im,title+'完整返回',notes+['稀疏支持不是稠密人工真值'])
        f=assets/f'masks_{index:02d}.jpg';c.save(f,quality=94);cards.append(('masks',f))
    summary={'status':'complete_source_surface_expansion_no_new_instance','source_pairs':6,'edge_pair_proposals':0,'material_proposals':len(trials['material']['plan']['rows']),'blue_proposals':len(trials['blue']['plan']['rows']),'logical_new_point_reviews':sum(t['review']['logical_views'] for t in trials.values()),'actual_new_VLM_calls':sum(t['review']['actual_new_VLM_calls'] for t in trials.values()),'exact_historical_VLM_reuses':sum(t['review']['exact_historical_reuses'] for t in trials.values()),'same_batch_VLM_reuses':sum(t['review']['same_batch_reuses'] for t in trials.values()),'actual_new_SAM_calls':sum(t['generation']['actual_new_SAM_calls'] for t in trials.values()),'new_complete_native_masks':sum(len(t['generation']['rows']) for t in trials.values()),'blue_hypotheses_rejected_for_new_point_semantics':2,'combined_new_union_pixels':combined_extra,'combined_union':122302+combined_extra,'area_cap':122511.25,'all_new_masks_area_person_border_valid':all(r['valid'] and r['area_fits'] and not r['person_conflict'] for t in trials.values() for r in t['audit']['candidates']),'r0_r3_r6_instance_matches_unchanged':True,'new_accepted_instances':0,'retained_support':'13/17','retained_original_Agent':'6/6','remaining_misses':4,'Agent_round_limit':3,'new_Agent_runs':0,'new_Mask2Former_forwards':0,'adopted':False,'no_output_mask_clipping':True,'all_original_six_and_references_exact':True,'all_new_source_cues_remain_unconfirmed_hypotheses':True,'model_manager_recovered_same_source_args_log_and_tmp_only_changed':True,'not_dense_truth_or_independent_generalization':True}
    notesets=[('可见结构补点：更大掩码仍需独立实例验证',['6个原两点样例，原输入、原正负点与原生选择规则保持','线段边界提案0；材质带提案3、蓝面提案5，全部源图规则提前冻结','16逻辑新点复核，10实际VLM、6同批复用；两组蓝面点因地面描述拦下','6次新SAM、6完整原生mask，均满足原面积/person/边界要求',f'六张共同新增并集{combined_extra}像素；总{122302+combined_extra}小于122511.25','分离小部位可扩大，但混入相邻参考实例；原13/17及旧6/6保持','新增Agent0、无新Mask2Former前向，不裁mask、不放宽任何最终门槛']),('完整部位与纯度：本轮仍未通过',['旧358/390像素掩码：r3目标48命中7/27、纯度1.0','蓝面补点后3641/3937像素：目标分别9/27、8/27','同时混入对象49等，目标纯度降为0.45和0.380952','材质带3299像素候选：对象47命中16，但纯度16/24=2/3<0.67','其余候选覆盖不足或主要支持已有对象；r0/3/6独立匹配均无增益','这些失败反映实例归属与完整性问题，不能用面积合规替代','所有候选/拒绝/实际原答保留，原人工标注不改']),('满盘导致管理器卡住：恢复和限制',['D盘满导致退出事件写入失败，反复记录同一已退出PID，未进入新推理','完整事件快照、原日志和确切503磁盘错误全部保留','目录在线切换因日志占用未执行，未改原目录或建立连接','确认没有模型工作进程后，以相同源码/模型/GPU/端口恢复管理器','仅日志和临时文件换到有余量的工作盘，原服务配置与标注保留','恢复后6个VLM原始回复核验同模型及[256,256]，3次SAM实际完成','磁盘仍紧张；保留全部原实验与失败，只回收已核验重复副本'])]
    for i,(title,notes) in enumerate(notesets):
        c=canvas(title);d=ImageDraw.Draw(c)
        for j,n in enumerate(notes):d.text((40,145+j*120),n,font=font,fill='#16364f')
        f=assets/f'summary_{i+1:02d}.jpg';c.save(f,quality=94);cards.append(('summary',f))
    videos=[]
    for name,kinds in [('01_六组源图补点与拒绝记录.mp4',['points']),('02_六组完整掩码扩大前后.mp4',['masks']),('03_验收结果与运行恢复.mp4',['summary'])]:
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
        cap.release();assert count==40*len(photos);videos.append({'path':name,'frames':count,'sha256':sha(OUT/name)})
    summary.update(pictures=len(cards),videos=len(videos));save(OUT/'summary.json',summary);save(OUT/'paired_comparison.json',comparisons);save(OUT/'independent_readback.json',{'passed':True,'checks':checks,'source_pixels_full_payload_model_tokens_all_labels_native_masks_originals_exact':True});save(OUT/'media_verification.json',{'passed':True,'pictures':len(cards),'videos':videos,'fully_decoded_frames':sum(v['frames'] for v in videos),'saved_stage_replay_not_screen_recording':True});text='源图边界无合格提案后，分别冻结材质带与可见蓝面补点。16逻辑点复核、10次新VLM、6同批复用；6次新SAM，全部完整未裁切原生输出。蓝面两组因地面描述被拦下；有效返回虽扩大了原窄部位，但混入邻近参考实例或覆盖不足，r0/3/6独立支持无提升。原13/17、原6/6与三轮等门槛保持，未采用。磁盘满导致管理器记录退出事件失败；保留全事件/原日志后，在零模型工作进程下以同配置恢复，日志及临时文件换盘，真实模型身份和推理已回读。';(OUT/'本轮结论.txt').write_text(text,encoding='utf-8');(OUT/'index.html').write_text('<meta charset="utf-8"><h1>完整可见基座补点与纯度对照</h1><p>'+text+'</p><a href="summary.json">实测汇总</a> · <a href="paired_comparison.json">逐项对照</a> · <a href="independent_readback.json">独立回读</a>'+''.join('<p><video controls width="1100" src="'+v['path']+'"></video></p>' for v in videos)+''.join('<p><img width="1100" src="assets/'+f.name+'"></p>' for _,f in cards),encoding='utf-8');print(summary,flush=True)

if __name__=='__main__':main()
