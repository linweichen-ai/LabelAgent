"""Real sensor/caption/full-mask evidence; physical identity remains unconfirmed."""
import hashlib,json,textwrap
from pathlib import Path
import av,cv2,numpy as np
from PIL import Image,ImageDraw,ImageFont,ImageOps
from run_rgb_layers import ROOT,read,save,sha
from paired_source_cue import body
from semantic_subject_guard import compatible_pallet_subject
S=ROOT/'outputs/labelagent_unlabelled_height_bands_20261108';H=ROOT/'outputs/labelagent_height_review_20261108';OUT=ROOT/'outputs/labelagent_height_bands_report_20261108'

def main():
    if OUT.exists():raise FileExistsError('Preserve completed unlabelled sensor report')
    p=read(S/'frozen_plan.json');d=read(S/'completed.json');review=read(H/'review_completed.json');hp=read(H/'frozen_plan.json');generation=read(H/'SAM/generation_completed.json');audit=read(H/'post_freeze_audit/support_area_audit.json');assert all(sha(f)==h for f,h in p['protected'].items()) and all(sha(f)==h for f,h in hp['protected'].items());assert read(S/'independent_readback.json')['passed'];checks=[]
    for cue in hp['rows']:
        v=cue['new_point_views'][0];rr=next(r for r in review['rows'] if r['partition']==cue['index']);request=body(*[b['image'] for b in v['bindings']]);assert request==read(v['request_file']) and hashlib.sha256(json.dumps(request,sort_keys=True).encode()).hexdigest()==v['request_sha256'];reply=read(rr['response_file'])['raw_response'];assert sha(rr['response_file'])==rr['response_sha256'] and reply['local_vlm']['actual_image_token_counts']==[256,256] and reply['local_vlm']['model_config_sha256']==hp['model_config_sha256'];assert rr['decision']==json.loads(reply['choices'][0]['message']['content']) and rr['PAL_subject_evidence']==compatible_pallet_subject(rr['decision']);photo=Image.open(cue['source_image']).convert('RGB')
        for b in v['bindings']:
            side=b['side'];box=b['source_canvas_box'];clip=b['clipped_box'];im=Image.new('RGB',(side,side),(127,127,127));im.paste(photo.crop(tuple(clip)),(clip[0]-box[0],clip[1]-box[1]));np.testing.assert_array_equal(np.asarray(im.resize((512,512),Image.Resampling.LANCZOS)),np.asarray(Image.open(b['image']).convert('RGB')))
        checks.append({'cue':cue['index'],'source_pixels_payload_model_tokens_and_decision_exact':True})
    assert review['actual_new_VLM_calls']==10 and generation['actual_new_SAM_calls']==3;assert all(r['old_matches']==r['geometry_area_person_valid_matches'] for c in audit['cases'] for r in c['radii']);assert sum(not r['valid'] for r in audit['candidates'])==1
    OUT.mkdir();(OUT/'assets').mkdir();font=ImageFont.truetype('C:/Windows/Fonts/msyh.ttc',28);small=ImageFont.truetype('C:/Windows/Fonts/msyh.ttc',21);cards=[]
    def canvas(title):
        c=Image.new('RGB',(1920,1080),'#eef3f7');ImageDraw.Draw(c).text((28,22),title,font=font,fill='#183b50');return c
    def panel(c,k,im,title,notes):
        x=28+k*630;dr=ImageDraw.Draw(c);dr.text((x,110),title,font=small,fill='#183b50');im=ImageOps.contain(im,(595,540));c.paste(im,(x,155));yy=725
        for n in notes:
            for line in textwrap.wrap(str(n),54 if str(n).isascii() else 29):dr.text((x,yy),line,font=small,fill='#183b50');yy+=25
    def add(group,name,im):
        f=OUT/'assets'/(name+'.jpg');im.save(f,quality=93);cards.append((group,f))
    colors=['#ff715e','#29cba1','#b78bff','#efbc38']
    for i,view in enumerate(p['views'],1):
        diag=next(x for x in d['diagnostics'] if x['teacher_key']==view['teacher_key']);photo=Image.open(view['input']).convert('RGB');im=canvas('未标注雷达 z 分组：'+view['teacher_key']);panel(im,0,photo,'原图局部与MF父区',[f"父区内可见原始回波 {diag['observed_sensor_points_in_MF_prior']}",'只读PCD XYZ与标定；不读参考标注','雷达z分组未证实为重力高度或物理实例'])
        file=S/(view['teacher_key']+'_sensor_components.npz')
        if file.exists():
            a=np.load(file,allow_pickle=False);marked=photo.copy();dr=ImageDraw.Draw(marked);x1,y1,_,_=view['box']
            for (x,y),j in zip(a['pixels'],a['component']):dr.ellipse((int(x)-x1-2,int(y)-y1-2,int(x)-x1+2,int(y)-y1+2),fill=colors[int(j)])
            panel(im,1,marked,'源传感器投影与颜色分组',[f"BIC选中 {diag['chosen_k']} 组",'颜色仅供展示，不是生成掩码','8px/.15m可见性规则与原行为一致'])
            plot=Image.new('RGB',(590,470),'white');pd=ImageDraw.Draw(plot);z=a['xyz'][:,2];low=float(z.min());span=max(float(z.max()-low),.01)
            for j in np.unique(a['component']):
                zz=z[a['component']==j]
                for t,val in enumerate(zz):
                    x=35+int(520*(val-low)/span);y=70+int(j)*100+t%7*4;pd.ellipse((x-2,y-2,x+2,y+2),fill=colors[int(j)])
                pd.text((15,90+int(j)*100),f'group{j} n={len(zz)}',font=small,fill='#183b50')
            pd.text((15,425),f'z: {low:.3f} ... {low+span:.3f} m',font=small,fill='#183b50');panel(im,2,plot,'实际 z 分布（非参考实例）',['1..4组，最小BIC且每组至少8点','仅按回波数量排序，最多2个候选','z分组不等同完整独立托盘'])
        add('sensor',f'sensor_{i:02d}',im)
    for cue in hp['rows']:
        rr=next(r for r in review['rows'] if r['partition']==cue['index']);im=canvas(f"源点主体复核 {cue['index']}：几何分组之后仍须检查实际RGB");photo=Image.open(cue['input']).convert('RGB');mark=photo.copy();dr=ImageDraw.Draw(mark)
        for q in cue['points']:
            x,y=q['x'],q['y'];dr.ellipse((x-4,y-4,x+4,y+4),fill='#37c891' if q['label'] else '#f5b834')
        panel(im,0,mark,'绿=源回波正点；黄=负点假设',[f"主体初筛 {rr['PAL_subject_evidence']}",f"源坐标 {cue['source_point']}",'负点跨z组不证明另一个实例','全部参考坐标未参与提示生成'])
        for k,b in enumerate(cue['new_point_views'][0]['bindings'],1):panel(im,k,Image.open(b['image']).convert('RGB'),'实际近景输入' if k==1 else '实际上下文输入',[rr['decision']['object']] if k==1 else ['同一模型、同一中性提示','实际图像token [256,256]','PAL文字初筛不是最终完整基座验收'])
        add('neutral',f'neutral_{cue["index"]:02d}',im)
    for row in audit['candidates']:
        photo=Image.open(row['input']).convert('RGB');full=np.load(Path(row['folder'])/'complete_full_mask.npy',allow_pickle=False);x1,y1,x2,y2=row['box'];mask=full[y1:y2,x1:x2]!=0;raw=np.asarray(photo).copy();raw[mask]=(raw[mask]*.45+np.array([25,210,235])*.55).astype('uint8');marked=Image.fromarray(raw);dr=ImageDraw.Draw(marked)
        for q in row['points']:
            x,y=q['x'],q['y'];dr.ellipse((x-3,y-3,x+3,y+3),fill='#34ce85' if q['label'] else '#ffc440')
            if not q['label']:dr.rectangle((x-3,y-3,x+3,y+3),outline='#ff304f',width=1)
        im=canvas(f"完整SAM返回 {row['index']}：接口成功不替代原生约束与独立实例验收");panel(im,0,photo,'同一原图、源回波提示',[f"完整mask面积 {row['raw_pixels']}",f"额外面积 {row['extra_pixels']}",'未裁切输出或通过预算反推轮廓'])
        panel(im,1,marked,'原生掩码及负点7×7检查区',[f"接口有效 {row['service_reported_valid']}",f"独立原生守卫 {row['native_prompt_guard']['valid']}",f"面积通过 {row['area_fits']}",f"人员冲突 {row['person_conflict']}"])
        own=next(x['ownership'] for case in audit['cases'] for radius in case['radii'] if radius['radius']==3 for x in radius['new_mask_ownership'] if x['key']==row['key']);notes=[f"对象{x['object_id']} 命中{x['points']} / 纯度{x['purity']:.3f}" for x in own];panel(im,2,Image.fromarray((mask*255).astype('uint8')),'完整原生bitmap',notes+['r0/3/6未增加独立匹配','没有采用候选'])
        add('SAM',f'SAM_{row["index"]:02d}',im)
    im=canvas('未标注传感器辅助结果：高度分组未解决完整单层实例');dr=ImageDraw.Draw(im)
    for i,line in enumerate(['4个白名单传感器文件；6个既有MF源区域；不读取labels或参考点','z混合分组选中2/2/3/2/1/3组，仅形成10个源提示假设','10次新双图VLM，3个正点初筛通过；3次新完整SAM','3张完整mask面积均合规；其中1张违反原3像素负点邻域，严格拒绝','另两张仍混入相邻参考实例：目标48纯度约.328，目标42约.206','r0/3/6均无新增匹配；原13/17、历史6/6及三轮等门槛保持','输入新增了未标注传感器，不能宣称与纯RGB方案完全同条件','这是源传感器辅助诊断；不是独立场景泛化或稠密真值验证']):dr.text((40,130+i*105),line,font=font,fill='#183b50')
    add('summary','summary',im);videos=[]
    for group in ['sensor','neutral','SAM','summary']:
        photos=[f for g,f in cards if g==group];name=group+'.mp4'
        with av.open(str(OUT/name),'w') as container:
            stream=container.add_stream('libx264',rate=10);stream.width=1920;stream.height=1080;stream.pix_fmt='yuv420p';stream.options={'crf':'23','preset':'veryfast'}
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
        cap.release();assert count==40*len(photos);videos.append({'path':name,'frames':count,'seconds':count/10,'sha256':sha(OUT/name)})
    summary={'status':'complete_rejected_unlabelled_sensor_height_trial','sensor_files':4,'source_views':6,'source_hypotheses':10,'actual_new_VLM_calls':10,'actual_new_SAM_calls':3,'new_training':False,'new_Agent_runs':0,'native_masks_rejected_by_original_negative_neighborhood':1,'new_accepted_instances':0,'retained_support':'13/17','retained_historical_Agent':'6/6','remaining_misses':4,'Agent_round_limit':3,'no_mask_clipping':True,'all_original_gates_unchanged':True,'extra_unlabelled_sensor_input_not_RGB_only':True,'no_reference_annotation_files_opened_by_generation':True,'not_independent_ground_truth':True,'adopted':False,'pictures':len(cards),'videos':len(videos),'saved_stage_replay_not_screen_recording':True};save(OUT/'summary.json',summary);save(OUT/'independent_readback.json',{'passed':True,'checks':checks,'source_whitelist_and_projection_check':str(S/'independent_readback.json'),'complete_native_masks_original_guard_audited':True});save(OUT/'media_verification.json',{'passed':True,'pictures':len(cards),'videos':videos,'fully_decoded_frames':sum(v['frames'] for v in videos)})
    text='追加未标注雷达z坐标分组，不读取labels或参考坐标，6个MF父区形成10个源点假设，实际10VLM、3SAM。分组不等同重力高度或物理托盘实例。完整返回虽面积合规，仍混入相邻层/相邻物；另1张接口成功却违反原负点邻域，被原生守卫拒绝。r0/3/6无新增独立匹配，保留13/17、历史6/6、原标注及所有门槛，新Agent0，未采用。源数据与评价共享传感器，不能当独立真值验证；新增传感器输入需与纯RGB对照分别表述。视频为保存阶段回放。';(OUT/'本轮结论.txt').write_text(text,encoding='utf-8');(OUT/'index.html').write_text('<meta charset="utf-8"><h1>未标注雷达分组与完整原生掩码</h1><p>'+text+'</p><a href="summary.json">实测汇总</a> · <a href="independent_readback.json">独立回读</a>'+''.join('<p><video controls width="1100" src="'+v['path']+'"></video></p>' for v in videos)+''.join('<p><img width="1100" src="assets/'+f.name+'"></p>' for _,f in cards),encoding='utf-8');print(summary,flush=True)

if __name__=='__main__':main()
