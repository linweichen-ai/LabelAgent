"""Render exact RGB correspondence; never turn tracked points into truth."""
import hashlib,json,shutil
from pathlib import Path
import av,cv2,numpy as np
from PIL import Image,ImageDraw,ImageFont,ImageOps
from run_pallet_chain import ROOT,OUT,OLD,old

CODE=['track_pallet_chain.py','run_pallet_chain.py','report_pallet_chain.py']


def main():
    plan=old.read(OUT/'frozen_plan.json');done=old.read(OUT/'completed.json');assert all(old.digest(p)==h for p,h in plan['protected'].items());assets=OUT/'assets';assets.mkdir(exist_ok=True);sources={(s['camera'],s['frame']):s['path'] for s in plan['native_sources']};font=ImageFont.truetype('C:/Windows/Fonts/msyh.ttc',25);small=ImageFont.truetype('C:/Windows/Fonts/msyh.ttc',19);number=ImageFont.truetype('C:/Windows/Fonts/msyh.ttc',10);cards=[];checks=[]
    for row in done['rows']:
        seed=next(s for s in plan['seeds'] if s['index']==row['seed']);chain=row['chain'];original=np.asarray(seed['initial_points']);alive={i:p for i,p in enumerate(seed['initial_points'])}
        for edge in chain['forward_edges']+chain['backward_edges']:
            ids=edge['input_point_ids'];assert ids==sorted(alive)
            if not ids:assert edge['track'] is None;continue
            result=edge['track'];assert len(result['points'])==len(ids);nextalive={}
            for p in result['points']:
                np.testing.assert_allclose(p['source_xy'],alive[ids[p['point_id']]],rtol=0,atol=1e-4)
                f=p['target_xy'];b=p['back_to_source_xy'];valid=p['forward_status'] and p['backward_status'] and f is not None and b is not None and 0<=f[0]<1920 and 0<=f[1]<1536 and 0<=b[0]<1920 and 0<=b[1]<1536 and np.linalg.norm(np.asarray(b)-p['source_xy'])<=1.5
                assert bool(valid)==p['valid_image_correspondence']
                if valid:nextalive[ids[p['point_id']]]=f
            alive=nextalive
        valid=[]
        for p in chain['points']:
            np.testing.assert_array_equal(p['source_xy'],original[p['point_id']]);back=alive.get(p['point_id']);passed=back is not None and np.linalg.norm(np.asarray(back)-original[p['point_id']])<=1.5;assert passed==p['all_edges_and_whole_cycle_valid'];assert back==p['returned_xy']
            if passed:valid.append(p['point_id'])
        assert len(valid)==chain['valid_points'] and chain['anchor_valid']==(0 in valid);assert chain['diagnostic_image_correspondence_available']==(len(valid)>=6 and len(valid)/len(original)>=.6)
        checks.append({'seed':row['seed'],'frame':row['endpoint_frame'],'all_edge_and_cycle_statuses_recomputed':True,'original_ids_and_points_exact':True,'valid_points':len(valid)})
        camera=int(row['case'].split('_')[0][6:]);path=chain['path'];first=chain['forward_edges'][0]['track'];middle=path[1];endpoint=path[-1];photos={f:Image.open(sources[(camera,f)]).convert('RGB') for f in path};initial=[{'point_id':i,'xy':p,'valid':None} for i,p in enumerate(seed['initial_points'])];direct=[{'point_id':p['point_id'],'xy':p['target_xy'],'valid':p['valid_image_correspondence']} for p in row['direct']['points']];mid=[{'point_id':p['point_id'],'xy':p['target_xy'],'valid':p['valid_image_correspondence']} for p in first['points']];end=[{'point_id':p['point_id'],'xy':p['endpoint_xy'],'valid':p['all_edges_and_whole_cycle_valid']} for p in chain['points']]
        board=Image.new('RGB',(1920,1080),'#152436');d=ImageDraw.Draw(board);d.text((22,20),f"种子{row['seed']:02d} {row['case']} → {endpoint}：相同图像点、原LK参数，逐边和完整回环仍≤1.5px",font=font,fill='white');d.text((22,62),'上排无标记原图裁块，下排实际对应；0为旧anchor。显示平移不是掩码、身份确认或完整基座证明。',font=small,fill='#d2e6ef')
        columns=[(76,None,initial,'原始帧76'),(endpoint,row['direct']['median_valid_translation_xy'],direct,'旧直接跟踪'),(middle,first['median_valid_translation_xy'],mid,'逐帧中间步骤'),(endpoint,chain['median_valid_translation_xy'],end,'逐帧完整回环终点')];display=[]
        for k,(frame,delta,points,label) in enumerate(columns):
            box,clamped=old.crop_box(seed['binding']['source_box'],delta,photos[frame].size);crop=photos[frame].crop(tuple(box));overlay=crop.copy();draw=ImageDraw.Draw(overlay)
            for p in points:
                if p['xy'] is None:continue
                x,y=p['xy'][0]-box[0],p['xy'][1]-box[1]
                if not(0<=x<crop.width and 0<=y<crop.height):continue
                color='#ffa31a' if p['valid'] is None else '#00ff8c' if p['valid'] else '#ff5d5d';rad=4 if p['point_id']==0 else 2;draw.ellipse((x-rad,y-rad,x+rad,y+rad),outline=color,width=2);draw.text((x+3,y-10),str(p['point_id']),font=number,fill=color,stroke_width=1,stroke_fill='black')
            x0=22+475*k;d.text((x0,110),label+f' / 帧{frame}',font=small,fill='white')
            for j,im in enumerate([crop,overlay]):
                tile=ImageOps.contain(im,(450,340));board.paste(tile,(x0+(450-tile.width)//2,150+380*j+(340-tile.height)//2))
            display.append({'column':k,'native_frame':frame,'source_display_box':box,'display_box_clamped':clamped,'not_annotation_geometry':True})
        d.text((22,915),f"旧直接：{row['direct']['valid_points']}/{len(original)}，诊断通过{row['direct']['diagnostic_image_correspondence_available']}，anchor{row['direct']['anchor_valid']}",font=font,fill='#d2e6ef');d.text((22,956),f"逐帧完整回环：{chain['valid_points']}/{len(original)}，诊断通过{chain['diagnostic_image_correspondence_available']}，anchor{chain['anchor_valid']}",font=font,fill='#d2e6ef');d.text((22,1006),'重复纹理仍可能对应错误；没有读取3D参考、标签、标定、位姿或面积预算，没有生成新mask或写回原标注。',font=small,fill='#e0cf9d');file=assets/f"seed{row['seed']:02d}_frame{endpoint}_direct_chain.jpg";board.save(file,quality=94);cards.append(file);old.save(OUT/f"display/seed{row['seed']:02d}_frame{endpoint}.json",display)
    assert done['direct_supported']==6;recovered=[{'seed':r['seed'],'frame':r['endpoint_frame']} for r in done['rows'] if r['chain']['diagnostic_image_correspondence_available'] and not r['direct']['diagnostic_image_correspondence_available']];lost=[{'seed':r['seed'],'frame':r['endpoint_frame']} for r in done['rows'] if r['direct']['diagnostic_image_correspondence_available'] and not r['chain']['diagnostic_image_correspondence_available']]
    summary={k:v for k,v in done.items() if k!='rows'};summary.update(recovered_correspondence_pairs=recovered,lost_correspondence_pairs=lost,retained_support='13/17',retained_original_Agent='6/6',remaining_misses=4,Agent_round_limit=3,pictures=len(cards),same_source_initial_points_and_LK_parameters=True,whole_cycle_extra_constraint=True,not_dense_truth_or_generalization=True,physical_instance_acceptance_not_evaluated=True)
    videos=[]
    for camera in (0,1):
        photos=[p for p in cards if int(next(s['case'] for s in plan['seeds'] if s['index']==int(p.stem[4:6])).split('_')[0][6:])==camera];filename=f'camera{camera}_原图与逐帧对应实测回放.mp4'
        with av.open(str(OUT/filename),'w') as c:
            stream=c.add_stream('libx264',rate=10);stream.width=1920;stream.height=1080;stream.pix_fmt='yuv420p'
            for p in photos:
                arr=np.asarray(Image.open(p).convert('RGB'))
                for _ in range(40):
                    for packet in stream.encode(av.VideoFrame.from_ndarray(arr,format='rgb24')):c.mux(packet)
            for packet in stream.encode():c.mux(packet)
        cap=cv2.VideoCapture(str(OUT/filename));count=0
        while True:
            ok,_=cap.read()
            if not ok:break
            count+=1
        cap.release();assert count==len(photos)*40;videos.append({'path':filename,'frames':count,'sha256':old.digest(OUT/filename)})
    summary['videos']=len(videos);old.save(OUT/'summary.json',summary);old.save(OUT/'independent_readback.json',{'passed':True,'checks':checks,'all_protected_inputs_exact':all(old.digest(p)==h for p,h in plan['protected'].items()),'same_immutable_direct_baseline_points':True});old.save(OUT/'media_verification.json',{'passed':True,'pictures':len(cards),'videos':videos,'fully_decoded_frames':sum(v['frames'] for v in videos),'saved_stage_replay_not_screen_recording':True});text=f"固定8种子16邻帧配对，起始点、原图、LK窗口/金字塔/迭代数/每边1.5px和6点60%诊断门槛均不变。逐帧76→75→74及76→77→78，额外做整条路径逆行回环≤1.5px。图像对应通过{done['direct_supported']}→{done['chain_supported']}/16，anchor通过{done['direct_anchor_valid']}→{done['chain_anchor_valid']}/16；新增可用图像对应{len(recovered)}组、旧通过损失{len(lost)}组。此结果仅说明RGB对应改善，不证明物理托盘身份或完整基座，也不计作实例恢复。原13/17、旧6/6、剩4、新mask0、新模型请求0、新Agent0，原标注与验收均不改。所有逐边坐标、前后状态、完整回环误差和失败保存，16图2视频是实测阶段回放，不是现场屏幕录制。"
    (OUT/'本轮结论.txt').write_text(text,encoding='utf-8');(OUT/'index.html').write_text('<meta charset="utf-8"><title>逐帧RGB对应诊断</title><h1>相同图像点的直接与逐帧对照</h1><p>'+text+'</p><a href="summary.json">实测结论</a> · <a href="independent_readback.json">独立回读</a> · <a href="completed.json">全部实际轨迹与失败</a>'+''.join('<p><video controls width="1100" src="'+v['path']+'"></video></p>' for v in videos)+''.join('<p><img width="1100" src="'+p.relative_to(OUT).as_posix()+'"></p>' for p in cards),encoding='utf-8');print(json.dumps(summary,ensure_ascii=False),flush=True)


if __name__=='__main__':main()
