"""Source geometry, decoder resolution, mask conflicts and bounded repair evidence."""
import hashlib,json
from pathlib import Path
import numpy as np
from PIL import Image,ImageDraw
from run_rgb_layers import ROOT,read,save
from experiment_report_media import Report,overlay
from paired_source_cue import body
from semantic_subject_guard import compatible_pallet_subject
from native_prompt_guard import evaluate
BASE=Path('D:/AI/Codex/Home/visualizations/2026/11/04/01a1a56c-0d6f-7d61-8326-d3bbd344d2df')
LOW=BASE/'labelagent_aspect_context_teacher_20261108';HIGH=BASE/'labelagent_highres_context_teacher_20261108';CLOW=ROOT/'outputs/labelagent_aspect_context_cues_20261108';CHIGH=BASE/'labelagent_highres_context_cues_20261108';ALOW=BASE/'labelagent_aspect_context_audit_20261108';COLL=BASE/'labelagent_mask_graph_collision_20261108';CORE=BASE/'labelagent_mask_graph_core_20261108';SEM=BASE/'labelagent_aspect_semantic_extremes_20261108/semantic';REPAIR=BASE/'labelagent_native_constraint_repair_20261108';OUT=BASE/'labelagent_context_graph_report_20261108'

def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(4*2**20),b''):h.update(b)
    return h.hexdigest()

def main():
    if OUT.exists():raise FileExistsError('Preserve complete or interrupted experiment report')
    checks=[];counts={'new_CPU_MF':0,'new_VLM':0,'new_SAM':0};views={};captured=[];native_count=0
    for folder in [LOW,HIGH]:
        p=read(folder/'frozen_plan.json');g=read(folder/'generation_completed.json');assert all(sha(f)==h for f,h in p['protected'].items());assert g['new_CPU_model_forwards']==12 and not g['new_training'] and g['GPU_calls']==0;counts['new_CPU_MF']+=g['new_CPU_model_forwards'];views[folder.name]=g['rows']
        for r in g['rows']:
            assert r['new_forward_call'] and all(sha(Path(r['folder'])/n)==h for n,h in r['artifacts'].items());source=Image.open(r['source_image']).convert('RGB');np.testing.assert_array_equal(np.asarray(source.crop(tuple(r['box']))),np.asarray(Image.open(r['input']).convert('RGB')));checks.append({'CPU_model':r['model'],'input_shape':r['actual_input_tensor_shape'],'view':r['teacher_key'],'source_pixels_and_all_query_files_exact':True})
    assert read(LOW/'actual_geometry_readback.json')['passed'] and read(HIGH/'paired_resolution_readback.json')['passed'];assert read(LOW/'frozen_plan.json')['postprocessing']==read(HIGH/'frozen_plan.json')['postprocessing']
    cohort_rows=[]
    for folder in [CLOW,CHIGH,COLL,CORE]:
        p=read(folder/'frozen_plan.json');review=read(folder/'review_completed.json');g=read(folder/'SAM/generation_completed.json');assert all(sha(f)==h for f,h in p['protected'].items());counts['new_VLM']+=review['actual_new_VLM_calls'];counts['new_SAM']+=g['actual_new_SAM_calls']
        for cue in p['rows']:
            for v in cue['new_point_views']:
                rr=next(x for x in review['rows'] if x['partition']==cue['index'] and x['request_sha256']==v['request_sha256']);payload=body(*[b['image'] for b in v['bindings']]);assert payload==read(v['request_file']) and hashlib.sha256(json.dumps(payload,sort_keys=True).encode()).hexdigest()==v['request_sha256'];raw=read(rr['response_file'])['raw_response'];assert sha(rr['response_file'])==rr['response_sha256'] and raw['local_vlm']['model_config_sha256']==p['model_config_sha256'] and raw['local_vlm']['actual_image_token_counts']==[256,256];assert json.loads(raw['choices'][0]['message']['content'])==rr['decision'] and compatible_pallet_subject(rr['decision'])==rr['PAL_subject_evidence'];source=Image.open(cue['source_image']).convert('RGB')
                for b in v['bindings']:
                    side=b['side'];box=b['source_canvas_box'];clip=b['clipped_box'];im=Image.new('RGB',(side,side),(127,127,127));im.paste(source.crop(tuple(clip)),(clip[0]-box[0],clip[1]-box[1]));np.testing.assert_array_equal(np.asarray(im.resize((512,512),Image.Resampling.LANCZOS)),np.asarray(Image.open(b['image']).convert('RGB')))
                cohort_rows.append({'cohort':folder.name,'cue':cue,'view':v,'review':rr});checks.append({'cohort':folder.name,'cue':cue['index'],'exact_payload_source_crops_reply_model_and_tokens':True})
        for r in g['rows']:
            a=np.load(r['raw_array_file'],allow_pickle=False);meta=read(r['metadata_file']);local=a[f"{r['prediction_call']}_masks"][r['native_index']].squeeze();full=np.load(Path(r['folder'])/'complete_full_mask.npy',allow_pickle=False);x1,y1,x2,y2=r['box'];np.testing.assert_array_equal(full[y1:y2,x1:x2],local);assert int(full.sum())==int((local!=0).sum()) and hashlib.sha256(full.tobytes()).hexdigest()==r['mask_sha256'];assert meta['points']==r['points'] and meta['selected_index']==r['native_index'] and meta['image_sha256']==sha(r['input']);np.testing.assert_array_equal(a[f"{r['prediction_call']}_point_coords"],[[q['x'],q['y']] for q in r['points']]);np.testing.assert_array_equal(a[f"{r['prediction_call']}_point_labels"],[q['label'] for q in r['points']]);native_count+=sum(len(a[n]) for n in a.files if n.endswith('_masks'));captured.append((folder,r));checks.append({'cohort':folder.name,'SAM_cue':r['index'],'actual_all_labels_and_selected_full_native_exact':True})
    sem=read(SEM/'completed.json');counts['new_VLM']+=sem['fresh_this_execution'];assert sem['fresh_this_execution']==8 and not any(r['all_semantic_stages_passed'] for r in sem['candidates']);assert not any(r['stage']=='extent' for r in sem['rows'])
    for r in sem['rows']:
        response=read(r['response']);assert sha(r['response'])==r['response_sha256'] and response['local_vlm']['actual_image_token_counts']==[256]*len(r['images']) and response['local_vlm']['model_config_sha256']==read(SEM/'frozen_plan.json')['model_config_sha256']
    repaired=read(REPAIR/'completed.json');repair_readback=read(REPAIR/'independent_readback.json');assert repair_readback['passed'] and repaired['native_constraint_cases_passed']==7 and repaired['actual_new_SAM_requests']==1 and repaired['total_point_annotations']==25;counts['new_SAM']+=repaired['actual_new_SAM_requests'];new_repair=[c['rounds'][-1]['row'] for c in repaired['cases'] if len(c['rounds'])>1];assert len(new_repair)==1
    for r in new_repair:
        a=np.load(r['raw_array_file'],allow_pickle=False);native_count+=sum(len(a[n]) for n in a.files if n.endswith('_masks'))
    assert counts=={'new_CPU_MF':24,'new_VLM':46,'new_SAM':30} and native_count==58
    audits=[ALOW,CHIGH]
    for folder in audits:
        a=read(folder/'audit_summary.json')
        for arm in a['arms']:
            d=read(folder/'audits'/arm['arm']/'support_area_audit.json');assert all(r['old_matches']==r['geometry_area_person_valid_matches'] for c in d['cases'] for r in c['radii'])
    for folder in [COLL,CORE]:
        for arm in read(folder/'audit_summary.json')['arms']:
            assert all(r['old']==r['with_candidates'] for case in arm['radii'].values() for r in case)
    assert all(r['old']==r['final'] for case in repair_readback['geometry_counts'].values() for r in case)
    binding=read(ROOT/'outputs/labelagent_point_ids_20261107/incumbent_binding.json')['files'];assert all(sha(ROOT/f)==h for f,h in binding.items());cfg=read(ROOT/'task_workspaces/labelagent_point_ids_20261107/config.json');old=read(ROOT/'outputs/labelagent_soft_prior_20261108/frozen_plan.json')['protected'];assert all(sha(cfg[k])==old[cfg[k]] for k in ['reference_annotations','reference_description'])
    report=Report(OUT)
    previous=read(ROOT/'outputs/labelagent_zoom_teacher_20261108/generation_completed.json')['rows'];all_phases=[previous,views[LOW.name],views[HIGH.name]];phase_names=['原局部直接拉伸','补真实上下文、保持比例','同源上下文、双倍输入分辨率'];keys=[r['teacher_key'] for r in read(LOW/'frozen_plan.json')['views']]
    for number,key in enumerate(keys,1):
        card=report.canvas('真实训练权重：源上下文/比例与编码分辨率对照 '+key)
        for model_index,model in enumerate(['old','retained']):
            for phase,rows in enumerate(all_phases):
                r=next(x for x in rows if x['teacher_key']==key and x['model']==model);photo=Image.open(r['input']).convert('RGB');rgb=np.asarray(Image.open(Path(r['folder'])/'panoptic.png')).astype('uint32');ids=rgb[:,:,0]+256*rgb[:,:,1]+65536*rgb[:,:,2];pal=[s['id'] for s in r['segments_info'] if s['category_id']==1];im=overlay(photo,np.isin(ids,pal));height,width=r['actual_input_tensor_shape'][-2:];im=im.resize((width,height),Image.Resampling.BILINEAR);report.panel(card,model_index*3+phase,im,('原训练d11c / ' if model=='old' else '保留候选6820 / ')+phase_names[phase],[f"模型输入{height}×{width}；PAL原候选{len(pal)}",'候选数不等同新增实例'],True)
        report.add('teachers',f'teacher_{number:02d}',card)
    unique={}
    for row in cohort_rows:unique.setdefault(row['view']['request_sha256'],[]).append(row)
    for index,group in enumerate(unique.values(),1):
        first=group[0];cue=first['cue'];rr=first['review'];v=first['view'];card=report.canvas(f'源点中性复核 {index}：相同请求合并展示，仍须完整单基座验收');photo=Image.open(cue['input']).convert('RGB');marked=photo.copy();q=v['new_point'];draw=ImageDraw.Draw(marked);draw.ellipse((q['x']-3,q['y']-3,q['x']+3,q['y']+3),fill='#36c889');report.panel(card,0,marked,'源点位置（绿色仅展示）',[f"原源坐标{v['source_point']}",f"主体初筛 {rr['PAL_subject_evidence']}",f"本批逻辑复用 {len(group)} 行",'原请求图没有此绿色标记'])
        for k,b in enumerate(v['bindings'],1):report.panel(card,k,Image.open(b['image']).convert('RGB'),'实际近景输入' if k==1 else '实际上下文输入',[rr['decision']['object']] if k==1 else ['同一Qwen3VL2B NF4模型','实际视觉token [256,256]','PAL主体初筛不证明实例完整性'])
        report.add('neutral',f'neutral_{index:02d}',card)
    for cohort_number,(folder,audit_folder) in enumerate([(CLOW,ALOW),(CHIGH,CHIGH)],1):
        pool=read(audit_folder/'native_pool_frozen_outputs.json')['rows']
        for row in read(folder/'SAM/generation_completed.json')['rows']:
            card=report.canvas(f"SAM完整三候选 {cohort_number}-{row['index']}：原选中与固定替代项分别保留");photo=Image.open(row['input']).convert('RGB');x1,y1,x2,y2=row['box'];options=[p for p in pool if p['sam_request_sha256']==row['sam_request_sha256']]
            for k,r in enumerate(options):
                full=np.load(Path(r['folder'])/'complete_full_mask.npy',allow_pickle=False);report.panel(card,k,overlay(photo,full[y1:y2,x1:x2]),f"原生索引{r['native_index']}"+(' / 原服务选中' if r['source_selected'] else ' / 未选中'),[f"完整面积{r['raw_pixels']}；原分数{r['native_score']:.5f}",f"点约束{r['native_prompt_guard']['valid']} / 触边{r['touches_border']}",'固定索引独立分臂，未按参考择优','未把三项同时算成三个实例'])
            report.add('native',f'native_{cohort_number}_{row["index"]:02d}',card)
    parent_catalogue={folder.name+'/'+r['sam_request_sha256']:r for folder,r in captured if folder in [CLOW,CHIGH]}
    for trial_number,folder in enumerate([COLL,CORE],1):
        p=read(folder/'frozen_plan.json');generation=read(folder/'SAM/generation_completed.json');by_index={r['index']:r for r in generation['rows']};audits={r['index']:r for arm in ['neighbor_negatives','exclusive_surface_positives'] for r in read(folder/'audits'/arm/'support_area_audit.json')['candidates']}
        for detail in p['details']:
            rank=detail['parent'];hypotheses=[r for r in p['rows'] if r['parent_rank']==rank];base=parent_catalogue[detail['source_parent_uid']];photo=Image.open(base['input']).convert('RGB');card=report.canvas(f'掩码冲突提示 {trial_number}-{rank}：源模型关系选点，不读参考或面积预算')
            for k,row in enumerate(hypotheses):
                marked=photo.copy();d=ImageDraw.Draw(marked)
                for q in row['points']:
                    x,y=q['x'],q['y'];d.ellipse((x-3,y-3,x+3,y+3),fill='#38c984' if q['label'] else '#ffbc39')
                    if not q['label']:d.rectangle((x-3,y-3,x+3,y+3),outline='#f02e5f')
                report.panel(card,k,marked,'仅加邻近负点' if k==0 else '补充同区域两正点',[f"实际点数{len(row['points'])}",'标记仅用于展示，实际原图无标记','负点所属实例仍只是源图假设'])
            region=Image.open(folder/'source_sampling'/f'parent{rank:02d}'/'positive_sampling_region_NOT_OUTPUT.png').convert('RGB');report.panel(card,2,region,'正点采样区：不是分割结果',['只用于选择点，未作为mask或先验提交','原完整父掩码和新返回均保留','核心方案要求邻近候选支持整个7×7' if trial_number==2 else '碰撞方案只要求重叠像素支持']);report.add('graph',f'graph_points_{trial_number}_{rank:02d}',card)
            card=report.canvas(f'源候选完整返回 {trial_number}-{rank}：所有模型像素原样保留');x1,y1,x2,y2=base['box'];full=np.load(Path(base['folder'])/'complete_full_mask.npy',allow_pickle=False);report.panel(card,0,overlay(photo,full[y1:y2,x1:x2]),'原父候选完整输出',[f"面积{base['raw_pixels']}",'没有用采样区替换原输出'])
            for k,row in enumerate(hypotheses,1):
                if row['index'] not in by_index:report.panel(card,k,photo,'正点初筛未通过，未调用SAM',['新增正点被描述为混凝土地面','没有为此候选伪造返回']);continue
                generated=by_index[row['index']];a=audits[row['index']];full=np.load(Path(generated['folder'])/'complete_full_mask.npy',allow_pickle=False);report.panel(card,k,overlay(photo,full[y1:y2,x1:x2]),'仅加邻近负点' if k==1 else '补充同区域正点',[f"完整面积{a['raw_pixels']} / 新增{a['extra_pixels']}",f"接口成功{a['service_reported_valid']} / 原生守卫{a['native_prompt_guard']['valid']}",f"面积通过{a['area_fits']} / 人员冲突{a['person_conflict']}",'独立实例支持未增加'])
            report.add('graph',f'graph_outputs_{trial_number}_{rank:02d}',card)
    semantic_plan=read(SEM/'frozen_plan.json')
    for number,key in enumerate(semantic_plan['selected'],1):
        rows=[r for r in sem['rows'] if r['candidate']==key];card=report.canvas('完整单基座复核：'+('窄小返回' if number==1 else '整摞返回'))
        neutral=next(r for r in rows if r['stage']=='neutral' and r['kind']=='candidate');report.panel(card,0,Image.open(neutral['images'][0]).convert('RGB'),'完整原照片主体复核',[neutral['decision']['object'],'近景/上下文主体初筛均通过','灰白原控制答复按完全相同请求复用'])
        for k,row in enumerate([r for r in rows if r['stage']=='structure'],1):report.panel(card,k,Image.open(row['images'][0]).convert('RGB'),'原指针结构视图' if row['name']=='pointer' else '原角框结构视图',[str(row['decision'])])
        report.add('semantics',f'semantic_{number:02d}',card)
    repaired_case=next(c for c in repaired['cases'] if len(c['rounds'])>1);before=repaired_case['rounds'][0]['row'];after=repaired_case['rounds'][-1]['row'];photo=Image.open(before['input']).convert('RGB');x1,y1,x2,y2=before['box'];card=report.canvas('原生约束修复：总请求仍≤3、总点数25≤32，未裁改任何输出像素')
    for k,row in enumerate([before,after]):
        full=np.load(Path(row['folder'])/'complete_full_mask.npy',allow_pickle=False);a=np.load(row['raw_array_file'],allow_pickle=False);local=a[f"{row['prediction_call']}_masks"][row['native_index']].squeeze();guard=evaluate(local,row['points']);report.panel(card,k,overlay(photo,full[y1:y2,x1:x2]),'首次原答（准确复用）' if k==0 else '第二次真实模型返回',[f"完整面积{row['raw_pixels']} / 原生点约束{guard['valid']}",f"保留所有既有点；本次点数{len(row['points'])}",'增加的负点来自模型违规前景像素' if k else '负点中心空，但7×7仍有前景'])
    report.panel(card,2,photo,'边界与独立验收',['只新增1次SAM，复用原7个首答','7个原生点约束案例通过，非Agent7/7','新模型输出增加159像素、减少177像素','原13/17和历史6/6未变，仍剩4个']);report.add('semantics','bounded_native_repair',card)
    report.text_card('summary','summary','本轮结论：生成约束更可核查，剩余实例仍未达标',['24次新CPU Mask2Former、46次新VLM、30次新SAM请求','两套真实训练权重；无新增训练，原始六结果及参考字节保持','源比例/上下文与双倍分辨率均没有新增严格实例支持','碰撞源点增加覆盖仍混层；核心采样仍有1个原生约束失败','模型反例修复只追加1次请求，在第二次真实返回通过原生约束','原生约束通过不等于完整基座或Agent收敛；原6/6未重跑','保留13/17、原有门槛和原始标注；所有新候选未采用'])
    report.text_card('summary','limits','原标准保持，局部诊断的边界明确',['生成不读取参考点坐标或以面积预算反推掩码','SAM原生位图、分数、点坐标及源选中索引全部保留','SAM低分辨率logits未另行捕获；MF所有100查询logits均保存','初筛只说明PAL主体；小/大掩码单基座复核均未通过','原全范围复核因此未继续请求；不假装已经执行','邻近源候选的身份未被人工确认；采样区不是输出或真值','图像重采样未创造原始细节，同批结果不是独立泛化验证'])
    videos=report.videos(['teachers','neutral','native','graph','semantics','summary']);summary={'status':'complete_rejected_context_resolution_graph_and_bounded_native_repair','actual_new_MF_CPU_forwards':24,'actual_new_VLM_calls':46,'actual_new_SAM_requests':30,'trained_checkpoints':2,'native_SAM_hypotheses_preserved':native_count,'raw_MF_queries_preserved':2400,'native_constraint_repair_initial_cases':7,'native_constraint_repair_initial_passes':6,'native_constraint_repair_final_passes':7,'native_constraint_repair_new_SAM_requests':1,'native_constraint_repair_max_requests_per_case':2,'native_constraint_repair_total_point_annotations':25,'native_constraint_success_not_Agent_convergence':True,'new_Agent_runs':0,'Agent_round_limit':3,'retained_support':'13/17','retained_historical_Agent':'6/6','historical_Agent_not_rerun':True,'remaining_misses':4,'new_accepted_instances':0,'all_original_gates_unchanged':True,'original_six_and_references_exact':True,'no_output_mask_clipping':True,'adopted':False,'pictures':len(report.cards),'videos':len(videos),'saved_stage_replay_not_screen_recording':True};save(OUT/'summary.json',summary);save(OUT/'independent_readback.json',{'passed':True,'checks':checks,'counts':counts,'native_SAM_masks':native_count,'originals_exact':True,'actual_geometry_table_overrides_inherited_unused_scale_diagnostics':True,'repaired_case_raw_pixel_delta':next(r for r in repair_readback['checks'] if r['requests_total']>1),'not_Agent_or_instance_adoption':True});text='本轮24CPU、46VLM、30SAM，完整保存两权重100查询、58个SAM原生位图及全部实际输入/原答。源比例/上下文、双倍编码分辨率均未新增独立实例；源掩码冲突采样提高目标覆盖仍混入邻层，原生守卫拦下不合格返回。核心采样还剩1例原生约束失败，在原三次请求/32点上限内用违规像素补1负点，只新增1次真实SAM，第二次返回通过：增加159像素、减少177像素，全部原样保存，没有后裁。7/7仅指本批原生点约束，不是Agent收敛或实例验收；原13/17与历史6/6、原标注和所有门槛保持，剩4，未采用。小/大掩码的原单基座复核均未过，未调用后续全范围阶段。视频是实际保存阶段回放。';(OUT/'本轮结论.txt').write_text(text,encoding='utf-8');report.index('源图比例、编码分辨率、掩码冲突与原生约束修复',text,videos);print(summary,flush=True)

if __name__=='__main__':main()
