"""Matched original/contrast/sharpen source pixels and complete native outputs."""
import hashlib
from pathlib import Path
import av,cv2,numpy as np
from PIL import Image,ImageDraw,ImageFont,ImageOps
from run_rgb_layers import ROOT,read,save,sha
BASE=ROOT/'outputs/labelagent_source_contrast_20261108';OUT=ROOT/'outputs/labelagent_source_contrast_report_20261108'

def main():
    if OUT.exists():raise FileExistsError('Preserve source contrast comparison')
    p=read(BASE/'frozen_plan.json');g=read(BASE/'generation_completed.json');summary=read(BASE/'audit_summary.json');assert all(sha(f)==h for f,h in p['protected'].items());assert g['actual_new_SAM_calls']==6 and all(r['valid'] and r['area_fits'] and not r['person_conflict'] for arm in summary['arms'] for r in arm['candidates']);assert all(r['old']==r['with_candidates'] for a in summary['arms'] for c in a['radii'].values() for r in c)
    checks=[]
    for row in g['rows']:
        original=np.asarray(Image.open(row['original_untransformed_input']).convert('RGB'))
        if row['photometric_policy'].startswith('source_LAB'):
            lab=cv2.cvtColor(original,cv2.COLOR_RGB2LAB);lab[:,:,0]=cv2.createCLAHE(clipLimit=2.0,tileGridSize=(8,8)).apply(lab[:,:,0]);expected=cv2.cvtColor(lab,cv2.COLOR_LAB2RGB)
        else:expected=cv2.addWeighted(original,2.0,cv2.GaussianBlur(original,(0,0),1.0),-1.0,0)
        np.testing.assert_array_equal(expected,np.asarray(Image.open(row['input']).convert('RGB')));a=np.load(row['raw_array_file'],allow_pickle=False);meta=read(row['metadata_file']);mask=a[f"{row['prediction_call']}_masks"][row['native_index']].squeeze();full=np.load(Path(row['folder'])/'complete_full_mask.npy',allow_pickle=False);x1,y1,x2,y2=row['box'];np.testing.assert_array_equal(mask,full[y1:y2,x1:x2]);assert int(full.sum())==int((mask!=0).sum()) and hashlib.sha256(full.tobytes()).hexdigest()==row['mask_sha256'];assert meta['image_sha256']==sha(row['input']) and meta['points']==row['points'] and meta['selected_index']==row['native_index'];checks.append({'cue':row['index'],'photometric_policy_reproduced_exact':True,'input_coordinates_and_complete_selected_native_exact':True})
    OUT.mkdir();(OUT/'assets').mkdir();font=ImageFont.truetype('C:/Windows/Fonts/msyh.ttc',27);small=ImageFont.truetype('C:/Windows/Fonts/msyh.ttc',20);cards=[]
    def canvas(t):
        c=Image.new('RGB',(1920,1080),'#eef3f7');ImageDraw.Draw(c).text((28,22),t,font=font,fill='#173a50');return c
    def panel(c,k,im,title,notes):
        x=28+k%3*630;y=100+k//3*480;dr=ImageDraw.Draw(c);dr.text((x,y),title,font=small,fill='#173a50');im=ImageOps.contain(im,(595,320));c.paste(im,(x,y+40))
        for i,line in enumerate(notes):dr.text((x,y+370+i*25),line,font=small,fill='#173a50')
    old=read(ROOT/'outputs/labelagent_two_sided_layers_20261108/SAM/generation_completed.json')['rows']
    for base in old:
        rows=[base]+[next(r for r in g['rows'] if r['source_partition_index']==base['source_partition_index'] and r['photometric_policy']==policy) for policy in ['source_LAB_CLAHE_clip2_grid8','source_unsharp_sigma1_amount1']];im=canvas(f"输入对比度/边缘试验：源组{base['source_partition_index']}，同原图几何与全部五点")
        for k,(title,r) in enumerate(zip(['原始输入 / 历史完整mask','LAB亮度局部对比度增强','固定轻度边缘增强'],rows)):
            photo=Image.open(r['input']).convert('RGB');panel(im,k,photo,title,['顶部为实际送入模型的图像','增强图不是原始照片，原图单独保留']);raw=np.asarray(Image.open(base['input']).convert('RGB')).copy();x1,y1,x2,y2=r['box'];mask=np.load(Path(r['folder'])/'complete_full_mask.npy',allow_pickle=False)[y1:y2,x1:x2]!=0;raw[mask]=(raw[mask]*.45+np.array([25,210,235])*.55).astype('uint8');notes=[f"完整面积 {r['raw_pixels']}",'未裁掩码、未变源选中索引规则']
            if k:notes.append('面积/提示约束合格，独立支持未提升')
            panel(im,k+3,Image.fromarray(raw),'完整返回叠回原始照片',notes)
        f=OUT/'assets'/f"source_{base['source_partition_index']:02d}.jpg";im.save(f,quality=93);cards.append(f)
    im=canvas('固定图像增强：提示约束改善，完整独立实例仍未解决');dr=ImageDraw.Draw(im)
    for i,t in enumerate(['3张原图局部 × 2种固定增强 = 6次新SAM；0新VLM','全部同一五点、同一坐标与原始分割模型；源选中索引保持','增强输入与原始输入分别保存，输出完整像素不裁切','六张新mask均通过原生提示、面积、边界及人员冲突检查','目标48在r3仍只有7..12/27点，纯度约.41..48；仍混入相邻层','r0/3/6无新增独立实例；原13/17和历史6/6保持','新Agent运行0，三轮等所有门槛不变，未采用候选']):dr.text((40,140+i*110),t,font=font,fill='#173a50')
    f=OUT/'assets/summary.jpg';im.save(f,quality=93);cards.append(f);videos=[]
    for name,photos in [('paired.mp4',cards[:3]),('summary.mp4',cards[3:])]:
        with av.open(str(OUT/name),'w') as c:
            stream=c.add_stream('libx264',rate=10);stream.width=1920;stream.height=1080;stream.pix_fmt='yuv420p';stream.options={'crf':'23','preset':'veryfast'}
            for photo in photos:
                arr=np.asarray(Image.open(photo).convert('RGB'))
                for _ in range(40):
                    for packet in stream.encode(av.VideoFrame.from_ndarray(arr,format='rgb24')):c.mux(packet)
            for packet in stream.encode():c.mux(packet)
        cap=cv2.VideoCapture(str(OUT/name));n=0
        while True:
            ok,_=cap.read()
            if not ok:break
            n+=1
        cap.release();assert n==40*len(photos);videos.append({'path':name,'frames':n,'seconds':n/10,'sha256':sha(OUT/name)})
    result={'status':'complete_rejected_fixed_source_contrast_trial','actual_new_SAM_calls':6,'new_VLM_calls':0,'new_Agent_runs':0,'source_images':3,'fixed_photometric_arms':2,'complete_native_masks':6,'native_prompt_guard_passed':6,'new_accepted_instances':0,'retained_support':'13/17','retained_historical_Agent':'6/6','remaining_misses':4,'Agent_round_limit':3,'no_mask_clipping':True,'all_original_acceptance_gates_unchanged':True,'adopted':False,'pictures':4,'videos':2,'saved_stage_replay_not_screen_recording':True};save(OUT/'summary.json',result);save(OUT/'independent_readback.json',{'passed':True,'checks':checks});save(OUT/'media_verification.json',{'passed':True,'pictures':4,'videos':videos,'fully_decoded_frames':160});text='在同一三张原局部、全部五点不变条件下，冻结LAB局部对比度增强和固定边缘增强，实际6次新SAM、0新VLM。六张完整原生mask提示/边界/面积/person检查合格，原先一组负点邻域冲突在增强输入下消失，但仍混入相邻层，目标48覆盖与纯度均不足，r0/3/6无新增独立匹配。原13/17、历史6/6、原标注及全部门槛保持，新Agent0，未采用。原始照片、实际增强输入、完整数组、源选中索引及首次空间守卫记录全部保留；视频为保存阶段回放。';(OUT/'本轮结论.txt').write_text(text,encoding='utf-8');(OUT/'index.html').write_text('<meta charset="utf-8"><h1>源图对比度与边缘增强对照</h1><p>'+text+'</p><a href="summary.json">结果</a> · <a href="independent_readback.json">回读</a>'+''.join('<p><video controls width="1100" src="'+v['path']+'"></video></p>' for v in videos)+''.join('<p><img width="1100" src="assets/'+p.name+'"></p>' for p in cards),encoding='utf-8');print(result,flush=True)

if __name__=='__main__':main()
