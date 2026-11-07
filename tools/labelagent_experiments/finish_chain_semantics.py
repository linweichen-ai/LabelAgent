"""Source-pixel and raw-response readback for new chain correspondence pairs."""
import hashlib,json,textwrap
from pathlib import Path
import av,cv2,numpy as np
from PIL import Image,ImageDraw,ImageFont,ImageOps
from run_pallet_chain import ROOT,OUT,old
from run_visual_budget import payload
from semantic_subject_guard import compatible_pallet_subject


def main():
    plan=old.read(OUT/'frozen_plan.json');sp=old.read(OUT/'semantic/frozen_plan.json');s=old.read(OUT/'semantic/completed.json');done=old.read(OUT/'completed.json');summary=old.read(OUT/'summary.json')
    if 'new_unique_VLM_calls' in summary:raise FileExistsError('Semantic report already complete; preserve counts and media')
    checks=[];seen=set();historical=0;within=0
    for r in s['rows']:
        raw=Path(r['image']).read_bytes();assert old.digest(r['image'])==r['image_sha256'];body=payload(raw,256);key=hashlib.sha256(json.dumps(body,sort_keys=True).encode()).hexdigest();assert key==r['request_sha256'];reply=old.read(r['response']);assert reply['local_vlm']['actual_image_token_counts']==[256] and reply['local_vlm']['model_config_sha256']==sp['model_config_sha256'] and old.digest(r['response'])==r['response_sha256'];assert json.loads(reply['choices'][0]['message']['content'])==r['decision'] and compatible_pallet_subject(r['decision'])==r['PAL_lexical_subject_guard']
        if r['kind']=='real_anchor_review':
            photo=Image.open(r['source_image']).convert('RGB');box=r['canvas_box'];clip=r['clipped_box'];canvas=Image.new('RGB',(129,129),(127,127,127));canvas.paste(photo.crop(tuple(clip)),(clip[0]-box[0],clip[1]-box[1]));canvas=canvas.resize((512,512),Image.Resampling.LANCZOS);np.testing.assert_array_equal(np.asarray(canvas),np.asarray(Image.open(r['image']).convert('RGB')))
            seed=next(x for x in plan['seeds'] if x['index']==r['seed']);anchor=seed['initial_points'][0] if r['frame']==76 else next(x['chain']['points'][0]['endpoint_xy'] for x in done['rows'] if x['seed']==r['seed'] and x['endpoint_frame']==r['frame']);np.testing.assert_array_equal(anchor,r['anchor_source_xy'])
        else:
            expected=127 if r['kind']=='gray_control' else 255;assert (np.asarray(Image.open(r['image']))==expected).all()
        if r['reused']:historical+=1
        elif key in seen:within+=1
        seen.add(key);checks.append({'name':r['name'],'request_image_model_reply_and_point_binding_exact':True,'request_sha256':key})
    assert all(old.digest(p)==h for p,h in sp['protected'].items()) and all(old.digest(p)==h for p,h in plan['protected'].items());fresh=len({r['request_sha256'] for r in s['rows'] if not r['reused']});assert fresh==s['fresh_unique_VLM_calls']
    assets=OUT/'assets';font=ImageFont.truetype('C:/Windows/Fonts/msyh.ttc',26);small=ImageFont.truetype('C:/Windows/Fonts/msyh.ttc',21);cards=[];pair_decisions=[]
    def canvas(title):
        c=Image.new('RGB',(1920,1080),'#eef3f7');ImageDraw.Draw(c).text((30,25),title,font=font,fill='#16364e');return c
    def panel(c,k,photo,title,notes):
        x=30+945*(k%2);y=100+470*(k//2);d=ImageDraw.Draw(c);d.text((x,y),title,font=small,fill='#16364e');photo=ImageOps.contain(photo,(470,385));c.paste(photo,(x,y+40));yy=y+45
        for note in notes:
            for line in textwrap.wrap(str(note),33 if str(note).isascii() else 17):d.text((x+490,yy),line,font=small,fill='#16364e');yy+=28
    for pair in sp['selected_pairs']:
        seed=next(x for x in plan['seeds'] if x['index']==pair['seed']);row=next(x for x in done['rows'] if x['seed']==pair['seed'] and x['endpoint_frame']==pair['frame']);rr=[x for x in s['rows'] if x.get('seed')==pair['seed']];assert len(rr)==2;c=canvas(f"种子{pair['seed']:02d} 新增图像对应 / 中心实际像素的中性物理描述")
        for j,r in enumerate(rr):
            photo=Image.open(r['source_image']).convert('RGB');delta=None if r['frame']==76 else row['chain']['median_valid_translation_xy'];box,_=old.crop_box(seed['binding']['source_box'],delta,photo.size);panel(c,j*2,photo.crop(tuple(box)),f"帧{r['frame']} 未标记上下文",['仅用于观察周边结构','上下文里有物体不构成中心点身份证据','图像对应已通过，身份仍需核查']);panel(c,j*2+1,Image.open(r['image']),f"帧{r['frame']} 实际VLM输入",[r['decision']['object'],f'词面托盘证据 {r["PAL_lexical_subject_guard"]}','129原生像素放大，未增加细节','没有定位箭头或预期类别'])
        file=assets/f"semantic_seed{pair['seed']:02d}_frame{pair['frame']}.jpg";c.save(file,quality=94);cards.append(file);pair_decisions.append({**pair,'source_guard':rr[0]['PAL_lexical_subject_guard'],'neighbor_guard':rr[1]['PAL_lexical_subject_guard'],'both_guard':all(r['PAL_lexical_subject_guard'] for r in rr),'physical_identity_or_full_base_confirmed':False})
    c=canvas('灰白中性控制：历史完全相同载荷的复用，未计作新实测')
    for k,r in enumerate([r for r in s['rows'] if r['kind']!='real_anchor_review']):panel(c,k,Image.open(r['image']),r['kind'],[r['decision']['object'],f'词面托盘证据 {r["PAL_lexical_subject_guard"]}','模型、载荷SHA、256图像令牌均核对一致','这两条是历史精确复用'])
    file=assets/'semantic_gray_white_exact_cached_controls.jpg';c.save(file,quality=94);cards.append(file)
    filename='03_新增对应的物理部位中性复核.mp4'
    with av.open(str(OUT/filename),'w') as container:
        stream=container.add_stream('libx264',rate=10);stream.width=1920;stream.height=1080;stream.pix_fmt='yuv420p'
        for p in cards:
            arr=np.asarray(Image.open(p).convert('RGB'))
            for _ in range(40):
                for packet in stream.encode(av.VideoFrame.from_ndarray(arr,format='rgb24')):container.mux(packet)
        for packet in stream.encode():container.mux(packet)
    cap=cv2.VideoCapture(str(OUT/filename));count=0
    while True:
        ok,_=cap.read()
        if not ok:break
        count+=1
    cap.release();assert count==40*len(cards);media=old.read(OUT/'media_verification.json');media['pictures']+=len(cards);media['videos'].append({'path':filename,'frames':count,'sha256':old.digest(OUT/filename)});media['fully_decoded_frames']+=count;old.save(OUT/'media_verification.json',media)
    summary.update(pictures=media['pictures'],videos=len(media['videos']),new_unique_VLM_calls=fresh,semantic_logical_views=len(s['rows']),historical_exact_reused_views=historical,within_batch_identical_request_reuses=within,total_exact_reused_logical_views=historical+within,new_pairs_with_both_neutral_PAL_guard=sum(x['both_guard'] for x in pair_decisions),new_pairs_reviewed=len(pair_decisions),blank_control_PAL_false_positives=s['blank_PAL_false_positives'],blank_controls_are_historical_exact_reuse=True,lexical_evidence_not_physical_identity_or_complete_base=True,engineering_tests=4,new_model_calls=fresh,new_SAM_calls=0)
    old.save(OUT/'summary.json',summary);old.save(OUT/'semantic_readback.json',{'passed':True,'checks':checks,'pair_decisions':pair_decisions,'fresh_unique_requests':fresh,'historical_reuses':historical,'within_batch_reuses':within,'not_mask_or_identity_acceptance':True});old.save(OUT/'assistant_visual_review.json',{'reviewer':'assistant visual inspection, not human truth','images_inspected':['seed01_frame78','seed03_frame78','seed07_frame74','seed05_frame74','seed06_frame74'],'observations':[{'seeds':[1],'observation':'邻帧可见货物下方的浅色木质基座与开孔，原图和邻帧中性描述均出现pallet；尚未确认新增独立实例或完成掩码。'},{'seeds':[3,7],'observation':'浅色基座后方与前景深色分格容器同时可见，跟踪点位于遮挡边缘附近。中性模型描述深色表面/容器或浅色矩形条，新增对应没有补足中心部位的托盘证据；不能据此认定真实类别为非PAL。'},{'seeds':[5,6],'observation':'远处横向结构的对应改善，但纹理小且模糊，画面还包含架体与邻近器材。没有足够证据确认该中心部位是独立托盘基座。'}],'physical_identity_confirmations':0,'new_dense_human_truth':0,'new_masks_approved':0})
    text=(OUT/'本轮结论.txt').read_text(encoding='utf-8')+f"\n\n对新增5组且anchor有效的对应，另做原帧/邻帧中心原生129像素无标记中性复核，共12逻辑视图：{fresh}条新独立载荷回复、{historical}条历史精确复用、{within}条同批相同请求复用。仅1/5组在两帧均获得词面托盘证据，其余4组未获得充分中心部位证据；这不等于证明它们不是托盘。灰白2视图均历史精确复用且拒判PAL，不当作新灰白实测。3D参考/标签/面积未参与生成。词面出现托盘仍不证明物理身份、完整基座或实例新增通过。没有新SAM、新mask或Agent编辑。4项功能检查通过；初版小尺寸合成平移用例因金字塔边界拒绝2点，保留失败后改用有完整金字塔支持的内部点验证平移，并新增边界保守拒绝回归，运行参数和门槛均未改变。"
    (OUT/'本轮结论.txt').write_text(text,encoding='utf-8');html=(OUT/'index.html').read_text(encoding='utf-8');html+='<h2>新增对应的中性物理部位复核</h2><p>'+text.split('\n\n')[-1]+'</p><p><a href="semantic_readback.json">载荷与源像素回读</a> · <a href="assistant_visual_review.json">辅助视觉观察及限制</a></p><video controls width="1100" src="'+filename+'"></video>'+''.join('<p><img width="1100" src="'+p.relative_to(OUT).as_posix()+'"></p>' for p in cards);(OUT/'index.html').write_text(html,encoding='utf-8');print(json.dumps(summary,ensure_ascii=False),flush=True)


if __name__=='__main__':main()
