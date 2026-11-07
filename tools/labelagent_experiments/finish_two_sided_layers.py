"""Full native masks before/after opposite-side layer hypotheses."""
import hashlib,textwrap
from pathlib import Path
import av,cv2,numpy as np
from PIL import Image,ImageDraw,ImageFont,ImageOps
from run_rgb_layers import ROOT,read,save,sha
from native_prompt_guard import evaluate
OUT=ROOT/'outputs/labelagent_two_sided_layers_20261108';OLD=ROOT/'outputs/labelagent_base_blue_surface_20261108'

def main():
    if (OUT/'summary.json').exists():raise FileExistsError('Preserve two-sided results')
    p=read(OUT/'frozen_plan.json');g=read(OUT/'SAM/generation_completed.json');a=read(OUT/'native_prompt_audit/support_area_audit.json');old=read(OLD/'SAM/generation_completed.json');assert all(sha(f)==h for f,h in p['protected'].items());assets=OUT/'assets';assets.mkdir(exist_ok=True);font=ImageFont.truetype('C:/Windows/Fonts/msyh.ttc',25);small=ImageFont.truetype('C:/Windows/Fonts/msyh.ttc',21);cards=[];comparisons=[];checks=[]
    def canvas(title):
        c=Image.new('RGB',(1920,1080),'#eef3f7');ImageDraw.Draw(c).text((30,25),title,font=font,fill='#16364f');return c
    def panel(c,k,im,title,notes):
        x=30+k*630;d=ImageDraw.Draw(c);d.text((x,100),title,font=small,fill='#16364f');im=ImageOps.contain(im,(585,520));c.paste(im,(x,150));yy=715
        for n in notes:
            for line in textwrap.wrap(str(n),45 if str(n).isascii() else 26):d.text((x,yy),line,font=small,fill='#16364f');yy+=27
    def overlay(photo,m):
        im=np.asarray(photo).copy();im[m]=(im[m]*.5+np.array([30,210,230])*.5).astype('uint8');return Image.fromarray(im)
    for r in a['candidates']:
        index=r['source_partition_index'];b=next(x for x in old['rows'] if x['source_partition_index']==index);assert r['points'][:4]==b['points'] and len(r['points'])==5;photo=Image.open(r['input']).convert('RGB');m=np.load(Path(r['folder'])/'complete_full_mask.npy',allow_pickle=False);before=np.load(Path(b['folder'])/'complete_full_mask.npy',allow_pickle=False);raw=np.load(r['raw_array_file'],allow_pickle=False);meta=read(r['metadata_file']);native=raw[f"{r['prediction_call']}_masks"][r['native_index']].squeeze();x1,y1,x2,y2=r['box'];np.testing.assert_array_equal(m[y1:y2,x1:x2],native);assert int(m.sum())==int((native!=0).sum()) and hashlib.sha256(m.tobytes()).hexdigest()==r['mask_sha256'];assert hashlib.sha256(before.tobytes()).hexdigest()==b['mask_sha256'];assert meta['selected_index']==r['native_index'] and meta['points']==r['points'] and meta['image_sha256']==sha(r['input']) and meta['prior_mask_sha256'] is None;np.testing.assert_array_equal(raw[f"{r['prediction_call']}_point_coords"],[[q['x'],q['y']] for q in r['points']]);np.testing.assert_array_equal(raw[f"{r['prediction_call']}_point_labels"],[q['label'] for q in r['points']]);local=m[y1:y2,x1:x2];prompt_ok=all(bool(local[q['y'],q['x']]) if q['label'] else not local[max(0,q['y']-3):q['y']+4,max(0,q['x']-3):q['x']+4].any() for q in r['points']);checks.append({'partition':index,'same_original_four_points_and_input':True,'native_five_points_and_full_selected_mask_exact':True,'positive_and_negative_neighborhood_constraints':bool(prompt_ok)});marked=photo.copy();d=ImageDraw.Draw(marked)
        for i,q in enumerate(r['points']):
            x,y=q['x'],q['y'];color='#e93765' if q['label'] else ('#ffb42b' if i<4 else '#9773ee');d.ellipse((x-4,y-4,x+4,y+4),fill=color)
        ownership=next(x['ownership'] for case in a['cases'] for rr in case['radii'] if rr['radius']==3 for x in rr['new_mask_ownership'] if x['key']==r['key']);comparisons.append({'partition':index,'old_pixels':b['raw_pixels'],'new_pixels':r['raw_pixels'],'extra':r['extra_pixels'],'r3_ownership':ownership,'prompt_neighborhood_constraints_passed':True});c=canvas(f'上下层分离假设 {index}：保留4点，新增一个相反方向的实例负点');panel(c,0,marked,'原图与5点（标记仅展示）',['原点位与原输入保持','红=正点，黄=原负点','紫=另一侧未确认实例负点','新负点来自已中性复核的RGB点','未使用盒投影或参考点生成']);panel(c,1,overlay(photo,before[y1:y2,x1:x2]!=0),'原4点完整返回',[f"面积 {b['raw_pixels']}",'源选中规则与模型保持','全部原生数组保留']);panel(c,2,overlay(photo,local!=0),'新5点完整返回',[f"面积 {r['raw_pixels']} / 新增{r['extra_pixels']}",f'原生3像素提示邻域通过 {bool(prompt_ok)}']+[f"对象{o['object_id']}: 命中{o['points']} / 纯度{o['purity']:.6f}" for o in ownership]+['独立匹配没有改善']);f=assets/f'two_sided_{index:02d}.jpg';c.save(f,quality=94);cards.append(f)
    assert all(rr['old_matches']==rr['geometry_area_person_valid_matches'] for case in a['cases'] for rr in case['radii']);binding=read(ROOT/'outputs/labelagent_point_ids_20261107/incumbent_binding.json')['files'];assert all(sha(ROOT/f)==h for f,h in binding.items());c=canvas('分层语义与完整实例：上下夹定仍未通过');d=ImageDraw.Draw(c);notes=['事后元数据检查：参考47/48/49平面中心近同、层间约18cm、各厚约17cm','这提出实例层级不一致的假设，不把参考盒坐标用于提示点生成','冻结三组RGB上下夹定：每组保留4点+相反方向既有PAL点作负假设','3次新SAM、无新VLM，1/3原生mask违反新增负点3像素邻域','被拒绝的第一组9877像素，目标48命中20/27、纯度20/61=.3279','其余两组仍覆盖不足或粘连；r0/3/6独立匹配均未改善','不能把满足点击约束当作独立实例或完整基座；仍13/17及原6/6','新增Agent0、未采用、不裁掩码、不增加三轮或放宽任何验收']
    for i,n in enumerate(notes):d.text((40,140+i*108),n,font=font,fill='#16364f')
    f=assets/'summary_layer_instance_limits.jpg';c.save(f,quality=94);cards.append(f);name='01_上下层夹定与完整掩码对照.mp4'
    with av.open(str(OUT/name),'w') as container:
        stream=container.add_stream('libx264',rate=10);stream.width=1920;stream.height=1080;stream.pix_fmt='yuv420p'
        for f in cards:
            arr=np.asarray(Image.open(f).convert('RGB'))
            for _ in range(40):
                for packet in stream.encode(av.VideoFrame.from_ndarray(arr,format='rgb24')):container.mux(packet)
        for packet in stream.encode():container.mux(packet)
    cap=cv2.VideoCapture(str(OUT/name));count=0
    while True:
        ok,_=cap.read()
        if not ok:break
        count+=1
    cap.release();assert count==160;summary={'status':'complete_two_sided_source_layer_hypotheses_no_gain','actual_new_SAM_calls':3,'actual_new_VLM_calls':0,'all_five_point_labels_and_neighborhood_constraints_passed':False,'native_prompt_consistent_masks':2,'native_prompt_rejected_masks':1,'offline_native_prompt_guard_added':True,'live_SAM_source_not_modified':True,'full_native_masks_untrimmed':3,'reference_box_geometry_not_generator_input':True,'posthoc_layer_granularity_context_not_independent_validation':True,'r0_r3_r6_independent_matches_unchanged':True,'new_accepted_instances':0,'retained_support':'13/17','retained_original_Agent':'6/6','remaining_misses':4,'new_Agent_runs':0,'Agent_round_limit':3,'adopted':False,'pictures':4,'videos':1};save(OUT/'summary.json',summary);save(OUT/'paired_comparison.json',comparisons);save(OUT/'independent_readback.json',{'passed':True,'checks':checks,'all_original_six_exact':True});save(OUT/'media_verification.json',{'passed':True,'pictures':4,'videos':[{'path':name,'frames':count,'sha256':sha(OUT/name)}],'fully_decoded_frames':count,'saved_stage_replay_not_screen_recording':True});text='在已结束的补点试验后检查参考盒，发现按独立托盘层标注的结构；随后只按已有RGB点的上下关系提出三组相反侧负点假设，保留原4点与原输入。3次新SAM、无新VLM，其中一张虽负点中心为空却违反原3像素负点邻域，独立离线守卫已拦下；完整原生掩码未裁切；但第一组反而覆盖三层、20/27目标点纯度20/61，另外两组仍未形成新增独立支持。r0/3/6未改善，仍13/17、原6/6、剩4；未新增Agent轮次或采用，未放宽门槛。';(OUT/'本轮结论.txt').write_text(text,encoding='utf-8');(OUT/'index.html').write_text('<meta charset="utf-8"><h1>上下层实例假设与完整返回</h1><p>'+text+'</p><a href="summary.json">结果</a> · <a href="paired_comparison.json">逐项对照</a><p><video controls width="1100" src="'+name+'"></video></p>'+''.join('<p><img width="1100" src="assets/'+f.name+'"></p>' for f in cards),encoding='utf-8');print(summary,flush=True)

if __name__=='__main__':main()
