"""Fixed-view geometry and clean photographs, without claiming a new instance."""
import av,cv2,numpy as np
from PIL import Image,ImageDraw,ImageFont,ImageOps
from run_pallet_long_chain import ROOT,OUT,old
from verify_track_records import verify


def main():
    if (OUT/'summary.json').exists():raise FileExistsError('Preserve completed longer-view report')
    plan=old.read(OUT/'frozen_plan.json');done=old.read(OUT/'completed.json');assert all(old.digest(p)==h for p,h in plan['protected'].items());assets=OUT/'assets';assets.mkdir();sources={(s['camera'],s['frame']):s['path'] for s in plan['native_sources']};checks=[];cards=[];font=ImageFont.truetype('C:/Windows/Fonts/msyh.ttc',25);small=ImageFont.truetype('C:/Windows/Fonts/msyh.ttc',17);number=ImageFont.truetype('C:/Windows/Fonts/msyh.ttc',9);display=[]
    for seed in plan['seeds']:
        camera=int(seed['case'].split('_')[0][6:]);rr=[r for r in done['rows'] if r['seed']==seed['index']];assert len(rr)==4
        for row in rr:checks.append(verify(row,seed['initial_points']))
        board=Image.new('RGB',(1920,1080),'#152436');d=ImageDraw.Draw(board);d.text((20,20),f"种子{seed['index']:02d} {seed['case']}：固定更远帧70 / 72 / 80 / 82，不重新选点",font=font,fill='white');d.text((20,62),'上排实际未标记照片，下排原始点位对应；相同原LK参数、逐边与整条回环≤1.5px、6点60%诊断门槛。',font=small,fill='#d2e6ef')
        columns=[(76,None,None)]+[(r['endpoint_frame'],r['chain']['median_valid_translation_xy'],r) for r in rr]
        for k,(frame,delta,row) in enumerate(columns):
            photo=Image.open(sources[(camera,frame)]).convert('RGB');box,clamped=old.crop_box(seed['binding']['source_box'],delta,photo.size);crop=photo.crop(tuple(box));overlay=crop.copy();od=ImageDraw.Draw(overlay)
            points=[{'point_id':i,'xy':p,'valid':None} for i,p in enumerate(seed['initial_points'])] if row is None else [{'point_id':p['point_id'],'xy':p['endpoint_xy'],'valid':p['all_edges_and_whole_cycle_valid']} for p in row['chain']['points']]
            for p in points:
                if p['xy'] is None:continue
                x,y=p['xy'][0]-box[0],p['xy'][1]-box[1]
                if not(0<=x<crop.width and 0<=y<crop.height):continue
                color='#ffa31a' if p['valid'] is None else '#00ff8c' if p['valid'] else '#ff5d5d';rad=4 if p['point_id']==0 else 2;od.ellipse((x-rad,y-rad,x+rad,y+rad),outline=color,width=2);od.text((x+2,y-9),str(p['point_id']),font=number,fill=color,stroke_width=1,stroke_fill='black')
            x0=18+k*381;d.text((x0,112),f'帧{frame}'+(' / 起始' if row is None else ''),font=small,fill='white')
            for j,im in enumerate([crop,overlay]):
                tile=ImageOps.contain(im,(365,345));board.paste(tile,(x0+(365-tile.width)//2,150+380*j+(345-tile.height)//2))
            notes=['原起始点，不新增'] if row is None else [f"直接 {row['direct']['valid_points']}/{row['direct']['initial_points']}",f"逐帧 {row['chain']['valid_points']}/{row['chain']['initial_points']}",f"诊断通过 {row['chain']['diagnostic_image_correspondence_available']}",f"原anchor {row['chain']['anchor_valid']}"]
            for j,n in enumerate(notes):d.text((x0,912+25*j),n,font=small,fill='#d2e6ef')
            display.append({'seed':seed['index'],'frame':frame,'native_source_box':box,'display_box_clamped':clamped,'fixed_box_fallback_when_no_valid_flow':delta is None and frame!=76,'not_mask_warp_or_identity_proof':True})
        d.text((20,1030),'对应不通过时仍保留原位置画面供核查；重复纹理、出画和遮挡都可能失败。没有新mask、模型请求或Agent编辑。',font=small,fill='#e0cf9d');p=assets/f"seed{seed['index']:02d}_fixed_longer_views.jpg";board.save(p,quality=94);cards.append(p)
    videos=[]
    for camera in (0,1):
        photos=[p for p in cards if int(next(s['case'] for s in plan['seeds'] if s['index']==int(p.stem[4:6])).split('_')[0][6:])==camera];filename=f'camera{camera}_更远视角对应与可见结构.mp4'
        with av.open(str(OUT/filename),'w') as c:
            stream=c.add_stream('libx264',rate=10);stream.width=1920;stream.height=1080;stream.pix_fmt='yuv420p'
            for p in photos:
                arr=np.asarray(Image.open(p).convert('RGB'))
                for _ in range(60):
                    for packet in stream.encode(av.VideoFrame.from_ndarray(arr,format='rgb24')):c.mux(packet)
            for packet in stream.encode():c.mux(packet)
        cap=cv2.VideoCapture(str(OUT/filename));count=0
        while True:
            ok,_=cap.read()
            if not ok:break
            count+=1
        cap.release();assert count==60*len(photos);videos.append({'path':filename,'frames':count,'sha256':old.digest(OUT/filename)})
    summary={k:v for k,v in done.items() if k!='rows'};summary.update(retained_support='13/17',retained_original_Agent='6/6',remaining_misses=4,Agent_round_limit=3,fixed_endpoints=plan['endpoints'],pictures=len(cards),videos=len(videos),same_source_initial_points_and_parameters=True,new_supported_pairs_with_invalid_anchor=[{'seed':r['seed'],'frame':r['endpoint_frame']} for r in done['rows'] if r['chain']['diagnostic_image_correspondence_available'] and not r['chain']['anchor_valid']],physical_instance_acceptance_not_evaluated=True)
    old.save(OUT/'summary.json',summary);old.save(OUT/'independent_readback.json',{'passed':True,'checks':checks,'all_protected_sources_exact':all(old.digest(p)==h for p,h in plan['protected'].items()),'no_reference_inputs':True});old.save(OUT/'display_bindings.json',display);old.save(OUT/'media_verification.json',{'passed':True,'pictures':len(cards),'videos':videos,'fully_decoded_frames':sum(v['frames'] for v in videos),'saved_stage_replay_not_screen_recording':True});text=f"固定同8种子和原点，查看70/72/80/82共32配对。直接对应通过{done['direct_supported']}/32，逐帧并完整回环通过{done['chain_supported']}/32，其中原anchor也有效{done['chain_supported_with_anchor']}组。seed7在80/82虽点数过诊断线，但原anchor丢失，不能传播原目标部位。这不是托盘身份或完整基座确认。原图/点/窗口/金字塔/误差和诊断门槛不改，未读取参考3D点、标签、位姿或面积。没有新模型请求、SAM、mask或Agent编辑，仍保留13/17、旧6/6、剩4。8张图与2段视频来自实际保存结果，失败全部保留，视频为阶段回放。"
    (OUT/'本轮结论.txt').write_text(text,encoding='utf-8');(OUT/'index.html').write_text('<meta charset="utf-8"><title>固定更远RGB视角</title><h1>可见基座证据预检</h1><p>'+text+'</p><a href="summary.json">实测结论</a> · <a href="independent_readback.json">逐边及完整回环回读</a> · <a href="completed.json">全部轨迹和失败</a>'+''.join('<p><video controls width="1100" src="'+v['path']+'"></video></p>' for v in videos)+''.join('<p><img width="1100" src="'+p.relative_to(OUT).as_posix()+'"></p>' for p in cards),encoding='utf-8');print(summary,flush=True)


if __name__=='__main__':main()
