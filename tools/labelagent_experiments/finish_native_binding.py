"""Append target-binding diagnostics without replacing failed gate evidence."""
import json,shutil,textwrap
from pathlib import Path
import numpy as np,cv2,av
from PIL import Image,ImageDraw,ImageFont,ImageOps
from run_native_multi import ROOT,OUT,read,save,sha

def main():
    diag=read(OUT/'target_binding_diagnostic/completed.json');cards=[];font=ImageFont.truetype('C:/Windows/Fonts/msyh.ttc',28);small=ImageFont.truetype('C:/Windows/Fonts/msyh.ttc',22)
    for i,row in enumerate(diag['rows']):
        c=Image.new('RGB',(1920,1080),'#eef3f7');d=ImageDraw.Draw(c);d.text((35,25),f"目标绑定诊断 / 种子{row['seed']} / {row['kind']} / {row['stage']}",font=font,fill='#152b40')
        ims=[Image.open(p).convert('RGB') for p in row['images']]
        joined=Image.new('RGB',(512*len(ims),512),'#7f7f7f')
        for k,im in enumerate(ims):joined.paste(im,(k*512,0))
        joined=ImageOps.contain(joined,(1150,830));c.paste(joined,(35,130));notes=['原提示实际回答：']
        for key,value in row['decision'].items():notes+=textwrap.wrap(f'{key}: {value}',48)
        notes+=['诊断不能覆盖已失败的中性门槛','没有运行新Agent或采用候选']
        for j,t in enumerate(notes):d.text((1220,140+j*48),t,font=small,fill='#152b40')
        p=OUT/'assets'/f'binding_diagnostic_{i:02d}.jpg';c.save(p);cards.append(p)
    v=OUT/'03_指示目标与片段范围诊断.mp4'
    with av.open(str(v),'w') as container:
        stream=container.add_stream('libx264',rate=10);stream.width=1920;stream.height=1080;stream.pix_fmt='yuv420p'
        for p in cards:
            a=np.array(Image.open(p).convert('RGB'))
            for _ in range(40):
                for packet in stream.encode(av.VideoFrame.from_ndarray(a,format='rgb24')):container.mux(packet)
        for packet in stream.encode():container.mux(packet)
    cap=cv2.VideoCapture(str(v));count=0
    while True:
        ok,_=cap.read()
        if not ok:break
        count+=1
    cap.release();assert count==40*len(cards)
    media=read(OUT/'media_verification.json');media['pictures']+=len(cards);media['videos'].append({'path':v.name,'frames':count,'sha256':sha(v)});media['fully_decoded_frames']+=count;save(OUT/'media_verification.json',media)
    interpretation={'real_extent_edge_fragment':sum(r['kind']=='real' and r['stage']=='extent' and r['decision'].get('target_extent')=='edge_fragment' for r in diag['rows']),'real_pointer_pallet_base':sum(r['kind']=='real' and r['stage']=='pointer' and r['decision'].get('category')=='pallet_base' for r in diag['rows']),'gray_extent_describes_white_shape':sum(r['kind']=='marked_gray' and r['stage']=='extent' and 'white' in r['decision'].get('object','').lower() for r in diag['rows']),'visual_readback_note':'Raw mask overlays indicate white base lower-edge pieces, while wide neutral views include foreground gray containers. Gray extent inputs retain the exact white mask on the second image; white-object descriptions are not evidence of a physical object.','adopted':False}
    save(OUT/'target_binding_diagnostic/interpretation.json',interpretation)
    summary=read(OUT/'final_summary.json');summary['pictures']=media['pictures'];summary['videos']=len(media['videos']);summary['binding_diagnostic_VLM_calls']=diag['unique_actual_VLM_responses'];summary['new_VLM_calls']+=diag['unique_actual_VLM_responses'];summary['binding_diagnostic_does_not_override_neutral_failure']=True;summary['binding_findings']=interpretation;save(OUT/'final_summary.json',summary)
    html=(OUT/'index.html').read_text(encoding='utf-8');html+='<h2>原指示与范围提示的失败后诊断</h2><p>掩码主要位于白色基座下沿；中性回答灰箱不能等同于SAM掩码位于灰箱。实际回复见下图。诊断不会把中性失败改为通过。</p><a href="target_binding_diagnostic/completed.json">诊断全部原始回答</a><p><video width="1100" controls src="'+v.name+'"></video></p>'+''.join('<p><img width="1100" src="'+p.relative_to(OUT).as_posix()+'"></p>' for p in cards);(OUT/'index.html').write_text(html,encoding='utf-8')
    with (OUT/'本轮结论.txt').open('a',encoding='utf-8') as f:f.write('\n可视回读纠正：1875/1995像素候选主要是白色基座下沿，不能把中性模型回答灰箱等同于SAM分到了灰箱。对完整叠图与中性视图的交叉检查暴露了目标绑定问题。另运行8次原指示/范围及带标记灰图诊断：两个真实候选均判edge_fragment，两个原指示分类均非pallet_base；两组灰图+白色二值mask范围复核都描述了白色物体，暴露把mask形状当物体证据的风险。没有通过声明。全部回答保留；诊断不会覆盖中性失败，也没有完整单基座或Agent通过声明。\n')
    repo=ROOT/'task_workspaces/labelagent_realtime_handoff_20261104';public=read(repo/'docs/test-results/native-multi.json');public.update(pictures=summary['pictures'],videos=summary['videos'],new_VLM_calls=summary['new_VLM_calls'],binding_diagnostic_VLM_calls=diag['unique_actual_VLM_responses'],binding_diagnostic_does_not_override_neutral_failure=True,binding_findings=interpretation);save(repo/'docs/test-results/native-multi.json',public);shutil.copy2(OUT/'本轮结论.txt',repo/'docs/test-results/native-multi.md')
    for name in ['diagnose_native_binding.py','finish_native_binding.py']:shutil.copy2(Path(__file__).with_name(name),repo/'tools/labelagent_experiments'/name)
    print({'pictures':summary['pictures'],'videos':summary['videos'],'frames':media['fully_decoded_frames'],'VLM_calls':summary['new_VLM_calls']},flush=True)

if __name__=='__main__':main()
