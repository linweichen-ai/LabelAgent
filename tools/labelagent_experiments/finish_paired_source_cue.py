"""Actual paired source pixels, budgets, replies, and conservative SAM admission."""
import hashlib,json,textwrap
from pathlib import Path
import av,cv2,numpy as np
from PIL import Image,ImageDraw,ImageFont,ImageOps
from paired_source_cue import ROOT,OUT,old,body
from semantic_subject_guard import compatible_pallet_subject


def main():
    if (OUT/'summary.json').exists():raise FileExistsError('Preserve completed paired comparison')
    p=old.read(OUT/'frozen_plan.json');done=old.read(OUT/'completed.json');assert all(old.digest(f)==h for f,h in p['protected'].items());checks=[]
    original=old.read(OUT/'engineering_failures/frozen_plan_v1.json');current_protected={k:v for k,v in p['protected'].items() if not k.endswith('paired_source_cue.py')};old_protected={k:v for k,v in original['protected'].items() if not k.endswith('paired_source_cue.py')};assert current_protected==old_protected and original['requests']==p['requests'];assert p['pre_inference_guard_amendment']['model_responses_before_amendment']==0
    for r in done['rows']:
        request=body(r['first_image'],r['second_image']);assert old.read(r['request_file'])==request and hashlib.sha256(json.dumps(request,sort_keys=True).encode()).hexdigest()==r['request_sha256'];reply=old.read(r['response']);assert reply['local_vlm']['model_config_sha256']==p['model_config_sha256'] and reply['local_vlm']['actual_image_token_counts']==[256,256] and old.digest(r['response'])==r['response_sha256'];assert json.loads(reply['choices'][0]['message']['content'])==r['decision'] and compatible_pallet_subject(r['decision'])==r['PAL_guard'];photo=Image.open(r['source']).convert('RGB')
        for b in r['bindings']:
            side=b['source_side'];canvas=Image.new('RGB',(side,side),(127,127,127));box=b['source_canvas_box'];clip=b['clipped_box'];canvas.paste(photo.crop(tuple(clip)),(clip[0]-box[0],clip[1]-box[1]));expected=canvas.resize((512,512),Image.Resampling.LANCZOS);file=Path(r['first_image']).with_name(b['view']+'.png');np.testing.assert_array_equal(np.asarray(expected),np.asarray(Image.open(file).convert('RGB')));assert b['same_center_source_point']==[int(round(v)) for v in r['point']]
        checks.append({'case':r['name'],'arm':r['arm'],'source_points_images_payload_model_tokens_reply_exact':True})
    names=[]
    for r in done['rows']:
        if r['name'] not in names:names.append(r['name'])
    pairs=[];assets=OUT/'assets';assets.mkdir();cards=[];font=ImageFont.truetype('C:/Windows/Fonts/msyh.ttc',26);small=ImageFont.truetype('C:/Windows/Fonts/msyh.ttc',21)
    def canvas(title):
        c=Image.new('RGB',(1920,1080),'#eef3f7');ImageDraw.Draw(c).text((30,25),title,font=font,fill='#16364f');return c
    def panel(c,k,im,title,notes):
        x=30+k*630;d=ImageDraw.Draw(c);d.text((x,105),title,font=small,fill='#16364f');im=ImageOps.contain(im,(585,520));c.paste(im,(x,150));yy=715
        for n in notes:
            for line in textwrap.wrap(str(n),45 if str(n).isascii() else 26):d.text((x,yy),line,font=small,fill='#16364f');yy+=27
    for name in names:
        a=next(r for r in done['rows'] if r['name']==name and r['arm']=='repeated_near');b=next(r for r in done['rows'] if r['name']==name and r['arm']=='near_plus_context');ra=body(a['first_image'],a['second_image']);rb=body(b['first_image'],b['second_image']);ra['messages'][0]['content'][2]=rb['messages'][0]['content'][2];assert ra==rb and a['first_sha256']==b['first_sha256'];pairs.append({'case':name,'kind':a['kind'],'PAL_guard_repeated_near':a['PAL_guard'],'PAL_guard_near_plus_context':b['PAL_guard'],'only_second_image_changed':True,'known_class_truth':a.get('known_class_truth',False)})
        c=canvas(name+'：相同中心点、两图512视觉令牌，仅第二张视野改变');notes=['第一张图在两组完全相同','同提示词、schema、温度0','同模型与256输出上限','真实点位不由类别词推定']
        if a['kind']=='control':notes+=['派生控制图，非传感器原生尺度']
        else:notes+=[f'原图点 {a["point"]}','未读取3D参考或面积预算']
        panel(c,0,Image.open(a['first_image']),'共同第一张：129像素近景',notes)
        for k,r in enumerate([a,b],1):panel(c,k,Image.open(r['second_image']),'第二张：重复近景' if k==1 else '第二张：385像素宽景',[r['decision']['object'],f'主体托盘证据 {r["PAL_guard"]}','实际编码 [256,256]','未增加Agent编辑轮数'])
        f=assets/(name+'.jpg');c.save(f,quality=94);cards.append((a['kind'],f))
    counter=[r for r in done['rows'] if r['kind']=='counter_positive_cue' and r['arm']=='near_plus_context'];admitted=[]
    for r in counter:
        original=next(x for x in done['rows'] if x['name']==f"original_seed{r['seed']:02d}" and x['arm']=='near_plus_context')
        if r['PAL_guard'] and not original['PAL_guard']:admitted.append(r['original_counter_key'])
    assert not admitted;old.save(OUT/'counter_cue_generation_admission.json',{'status':'no_new_qualified_counter_point','candidate_keys':admitted,'rule':'new counter positive must have PAL subject evidence and original point must not be assigned PAL; still only an unconfirmed proposal before original full-mask gates','new_SAM_calls':0,'no_clipping_or_reference_generation':True});old.save(OUT/'generation_completed.json',{'status':'no_qualified_new_point','rows':[],'actual_new_SAM_calls':0,'new_independent_masks':0,'new_Agent_runs':0,'adopted':False})
    controls=[r for r in done['rows'] if r['kind']=='control'];wood={arm:next(r['PAL_guard'] for r in controls if r['name']=='control_wood_pallet_base' and r['arm']==arm) for arm in ['repeated_near','near_plus_context']};negative_FP={arm:sum(r['PAL_guard'] for r in controls if not r['expected_PAL'] and r['arm']==arm) for arm in ['repeated_near','near_plus_context']};real={arm:sum(r['PAL_guard'] for r in done['rows'] if r['kind']=='original_seed' and r['arm']==arm) for arm in ['repeated_near','near_plus_context']};corner={arm:sum(r['PAL_guard'] for r in done['rows'] if r['kind']=='counter_positive_cue' and r['arm']==arm) for arm in ['repeated_near','near_plus_context']};binding=old.read(ROOT/'outputs/labelagent_point_ids_20261107/incumbent_binding.json')['files'];assert all(old.digest(ROOT/f)==h for f,h in binding.items())
    summary={'status':'complete_no_new_qualified_source_point','cases':p['cases'],'logical_conditions':len(done['rows']),'actual_unique_new_VLM_calls':done['unique_model_calls_total'],'exact_same_batch_reused_conditions':done['exact_reused_logical_conditions'],'matched_image_counts':[2,2],'matched_actual_visual_tokens':[512,512],'only_second_image_changed':True,'same_source_point_prompt_schema_model_and_decode':True,'original_seed_PAL_subject_evidence':real,'counter_point_PAL_subject_evidence':corner,'wood_fixture_PAL_subject_evidence':wood,'known_non_PAL_controls_false_PAL':negative_FP,'real_point_dense_truth_or_class_accuracy_not_established':True,'person_description_not_full_person_category_validation':True,'neutral_padding_is_not_object_evidence':True,'new_SAM_calls':0,'new_independent_masks':0,'new_Agent_runs':0,'retained_support':'13/17','retained_original_Agent':'6/6','remaining_misses':4,'Agent_round_limit':3,'adopted':False,'all_original_six_bytes_exact':True,'pre_inference_guard_null_PID_bug_and_fix_retained':True,'same_requests_before_and_after_guard_fix':True,'guard_tests':3,'not_independent_generalization':True}
    c=canvas('宽景帮助正控，但未恢复四个漏检的真实基座部位');d=ImageDraw.Draw(c);notes=['18个样例、36逻辑条件，30条真实独立载荷回复、6条同批精确复用','两组都2图/512视觉令牌，只改变第二张图，提示词与schema均相同','木托盘控制：重复近景无托盘主体证据，加入宽景后获得托盘描述','8个旧点仍仅原木基座点获得证据；4个纹理点两组均未获得托盘主体证据','宽景中提及背景托盘的回答被主体检查拦下，未借给当前点位','5个非PAL控制两组均无错误PAL；不能据此声称完整7类分类验证通过','没有合格新点，因此新SAM0、新mask0、新Agent0，仍13/17及原6/6','补边和派生控制尺度、工程首版空PID失败及零推理修复均保留']
    for i,n in enumerate(notes):d.text((45,145+i*105),n,font=font,fill='#16364f')
    f=assets/'summary_matched_context_and_instance_limits.jpg';c.save(f,quality=94);cards.append(('summary',f));videos=[]
    for filename,kinds in [('01_八个原点的近景宽景对照.mp4',['original_seed']),('02_四个纹理点的部位复核.mp4',['counter_positive_cue']),('03_控制图与原验收结果.mp4',['control','summary'])]:
        photos=[f for kind,f in cards if kind in kinds]
        with av.open(str(OUT/filename),'w') as container:
            stream=container.add_stream('libx264',rate=10);stream.width=1920;stream.height=1080;stream.pix_fmt='yuv420p'
            for file in photos:
                arr=np.asarray(Image.open(file).convert('RGB'))
                for _ in range(40):
                    for packet in stream.encode(av.VideoFrame.from_ndarray(arr,format='rgb24')):container.mux(packet)
            for packet in stream.encode():container.mux(packet)
        cap=cv2.VideoCapture(str(OUT/filename));count=0
        while True:
            ok,_=cap.read()
            if not ok:break
            count+=1
        cap.release();assert count==40*len(photos);videos.append({'path':filename,'frames':count,'sha256':old.digest(OUT/filename)})
    summary.update(pictures=len(cards),videos=len(videos));old.save(OUT/'summary.json',summary);old.save(OUT/'paired_comparison.json',pairs);old.save(OUT/'independent_readback.json',{'passed':True,'checks':checks,'all_paired_requests_differ_only_in_second_image':True,'all_original_six_exact':True,'all_original_source_and_driver_hashes_exact':True,'guard_fix_changed_no_cohort_or_model_body':True});old.save(OUT/'media_verification.json',{'passed':True,'pictures':len(cards),'videos':videos,'fully_decoded_frames':sum(v['frames'] for v in videos),'saved_stage_replay_not_screen_recording':True});text='固定18样例、36逻辑条件，30条实际新载荷回复与6条同批精确复用。两组均2图/512视觉令牌、同提示词/模式/温度/输出上限/模型，仅第二图从重复129近景改385宽景。木托盘正控获得改善，但8原点两组都仅1个具托盘主体证据，4纹理点两组都0，不能将附近或背景里的托盘身份借给实际容器/地面点。5个已知非PAL控制两组均无错误PAL；这里未验证完整person类别，更不能当完整7类回归。部分控制宽景含灰色补边，补边不是物体证据；控制是派生图，非原传感器尺度。新提案0，新SAM0、新mask0、新Agent0，原13/17、原6/6、三轮与所有最终门槛保持，没有采用。外部运行结束时child_pid置空导致首次执行保护异常，发生在任何模型调用前；原脚本/原冻结计划/失败均保留，仅修正空PID保护，36条件及30请求完全未变，并通过3项检查。全部实际输入、原始回复和截断描述保留。视频是实际保存阶段回放。';(OUT/'本轮结论.txt').write_text(text,encoding='utf-8');(OUT/'index.html').write_text('<meta charset="utf-8"><title>同一点近景与宽景配对</title><h1>结构上下文与真实基座证据</h1><p>'+text+'</p><a href="summary.json">实测结论</a> · <a href="independent_readback.json">源像素及实际编码回读</a> · <a href="counter_cue_generation_admission.json">新提案准入</a>'+''.join('<p><video controls width="1100" src="'+v['path']+'"></video></p>' for v in videos)+''.join('<p><img width="1100" src="'+f.relative_to(OUT).as_posix()+'"></p>' for _,f in cards),encoding='utf-8');print(summary,flush=True)


if __name__=='__main__':main()
