"""Side-by-side original and negative-point SAM returns, without mask editing."""
import hashlib,json,textwrap
from pathlib import Path
import av,cv2,numpy as np
from PIL import Image,ImageDraw,ImageFont,ImageOps
from run_rgb_layers import ROOT,read,save,sha
OUT=ROOT/'outputs/labelagent_trained_partition_20261108'

def main():
    if (OUT/'summary.json').exists():raise FileExistsError('Preserve partition results')
    p=read(OUT/'partition_plan.json');g=read(OUT/'SAM/generation_completed.json');a=read(OUT/'post_freeze_audit/support_area_audit.json');old=read(ROOT/'outputs/labelagent_trained_point_sam_20261108/generation_completed.json');assert all(sha(f)==h for f,h in p['protected'].items());binding=read(ROOT/'outputs/labelagent_point_ids_20261107/incumbent_binding.json')['files'];assert all(sha(ROOT/f)==h for f,h in binding.items());assets=OUT/'assets';assets.mkdir();font=ImageFont.truetype('C:/Windows/Fonts/msyh.ttc',25);small=ImageFont.truetype('C:/Windows/Fonts/msyh.ttc',21);cards=[];checks=[]
    def canvas(title):
        c=Image.new('RGB',(1920,1080),'#eef3f7');ImageDraw.Draw(c).text((30,25),title,font=font,fill='#16364f');return c
    def panel(c,k,im,title,notes):
        x=30+k*630;d=ImageDraw.Draw(c);d.text((x,100),title,font=small,fill='#16364f');im=ImageOps.contain(im,(585,520));c.paste(im,(x,150));yy=715
        for n in notes:
            for line in textwrap.wrap(str(n),45 if str(n).isascii() else 26):d.text((x,yy),line,font=small,fill='#16364f');yy+=27
    def overlay(photo,mask):
        arr=np.asarray(photo).copy();arr[mask]=(arr[mask]*.5+np.array([30,210,230])*.5).astype('uint8');return Image.fromarray(arr)
    comparisons=[]
    for r in a['candidates']:
        b=next(x for x in old['rows'] if x['index']==r['positive_source_index']);assert sha(b['input'])==sha(r['input']) and b['points']==r['points'][:1];x1,y1,x2,y2=r['box'];m=np.load(Path(r['folder'])/'complete_full_mask.npy',allow_pickle=False);raw=np.load(r['raw_array_file'],allow_pickle=False);meta=read(r['metadata_file']);native=raw[f"{r['prediction_call']}_masks"][r['native_index']].squeeze();np.testing.assert_array_equal(m[y1:y2,x1:x2],native);assert m.sum()==(native!=0).sum() and hashlib.sha256(m.tobytes()).hexdigest()==r['mask_sha256'];assert meta['selected_index']==r['native_index'] and meta['points']==r['points'] and meta['image_sha256']==sha(r['input']) and meta['prior_mask_sha256'] is None;np.testing.assert_array_equal(raw[f"{r['prediction_call']}_point_coords"],[[q['x'],q['y']] for q in r['points']]);np.testing.assert_array_equal(raw[f"{r['prediction_call']}_point_labels"],[q['label'] for q in r['points']]);assert sha(Path(r['folder'])/'response.json')==r['response_sha256'];checks.append({'index':r['index'],'same_positive_input_unchanged_new_negative_and_selected_raw_mask_exact':True});before=np.load(Path(b['folder'])/'complete_full_mask.npy',allow_pickle=False);assert hashlib.sha256(before.tobytes()).hexdigest()==b['mask_sha256'];photo=Image.open(r['input']).convert('RGB');source=photo.copy();d=ImageDraw.Draw(source)
        for q in r['points']:
            x,y=q['x'],q['y'];d.ellipse((x-4,y-4,x+4,y+4),fill='#ef3565' if q['label'] else '#ffc533')
        ownership=next(x['ownership'] for c in a['cases'] for rr in c['radii'] if rr['radius']==3 for x in rr['new_mask_ownership'] if x['key']==r['key']);comparisons.append({'index':r['index'],'positive':r['positive_source_index'],'negative':r['negative_source_index'],'old_pixels':b['raw_pixels'],'new_pixels':r['raw_pixels'],'extra':r['extra_pixels'],'r3_ownership':ownership});c=canvas(f"跨教师分离假设 {r['index']}：正点{r['positive_source_index']} / 假设负点{r['negative_source_index']}");panel(c,0,source,'相同原始区域，仅新增一个提示点',['红=正点，黄=未确认实例负点','点标记只用于此展示','不是人工确认的背景/实例边界','不使用参考点或预算生成提示']);panel(c,1,overlay(photo,before[y1:y2,x1:x2]!=0),'原单点SAM完整返回',[f"原面积 {b['raw_pixels']}",'保留同一原生选择规则','没有裁切或修改任何输出像素']);panel(c,2,overlay(photo,m[y1:y2,x1:x2]!=0),'新两点SAM完整返回',[f"面积{r['raw_pixels']} / 新增{r['extra_pixels']}"]+[f"对象{o['object_id']}: 命中{o['points']} / 纯度{o['purity']:.6f}" for o in ownership]+['独立匹配未改善']);f=assets/f"partition_{r['index']:02d}.jpg";c.save(f,quality=94);cards.append(f)
    assert all(rr['old_matches']==rr['geometry_area_person_valid_matches'] for c in a['cases'] for rr in c['radii']);assert sum(next(rr for rr in c['radii'] if rr['radius']==3)['geometry_area_person_valid_matches'] for c in a['cases'])==13
    c=canvas('分离提示能减少粘连，但没有恢复完整基座');d=ImageDraw.Draw(c);notes=['全部6个源图跨教师点对在推理前冻结，每组1正点+1未确认实例负点','6次新SAM、6张完整未裁切原生mask，全部满足面积/person/边界要求','原高覆盖26/27且纯度0.329的mask，变为358/390像素的小部位','r=3目标纯度可达1.0，但只覆盖7/27=25.9%，仍低于50%','另有候选12/18=2/3，严格小于0.67，未四舍五入通过','支持旧对象50不算新增实例；r=0/3/6独立匹配均无提升','仍13/17、原6/6、剩4；本批未采用，新增Agent0','负点不是新真值；完整返回、原始输入、前后对照和失败均保留']
    for i,n in enumerate(notes):d.text((40,140+i*108),n,font=font,fill='#16364f')
    f=assets/'summary_partition_completeness_limits.jpg';c.save(f,quality=94);cards.append(f);name='01_六组完整生成前后对照.mp4'
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
    cap.release();assert count==280;summary={'status':'complete_rejected_partition_hypotheses','actual_new_SAM_calls':6,'new_full_native_masks':6,'actual_new_VLM_calls':0,'all_area_person_border_guards_passed':True,'reference_free_generation':True,'r3_target48_high_purity_coverage':'7/27','r0_r3_r6_independent_matches_unchanged':True,'new_accepted_instances':0,'retained_support':'13/17','retained_original_Agent':'6/6','remaining_misses':4,'new_Agent_runs':0,'Agent_round_limit':3,'not_human_truth':True,'no_output_mask_clipping':True,'adopted':False,'pictures':7,'videos':1};save(OUT/'summary.json',summary);save(OUT/'paired_comparison.json',comparisons);save(OUT/'independent_readback.json',{'passed':True,'checks':checks,'original_six_exact':True});save(OUT/'media_verification.json',{'passed':True,'pictures':7,'videos':[{'path':name,'frames':count,'sha256':sha(OUT/name)}],'fully_decoded_frames':count,'saved_stage_replay_not_screen_recording':True});text='六个跨教师分离提示均为未确认假设，每组两点、同原输入。6次新SAM的完整原生输出均保留，未裁切；全部满足面积/person/边界，但高纯度候选仅支持7/27目标点，窄条完整基座问题仍未解决。2/3严格小于0.67，不四舍五入放行。r=0/3/6独立匹配均无改善，仍13/17、原6/6、剩4，未新增Agent轮次或采用。';(OUT/'本轮结论.txt').write_text(text,encoding='utf-8');(OUT/'index.html').write_text('<meta charset="utf-8"><h1>跨教师分离提示：完整生成前后对照</h1><p>'+text+'</p><a href="summary.json">结果</a> · <a href="paired_comparison.json">逐项前后对比</a><p><video controls width="1100" src="'+name+'"></video></p>'+''.join('<p><img width="1100" src="assets/'+f.name+'"></p>' for f in cards),encoding='utf-8');print(summary,flush=True)

if __name__=='__main__':main()
