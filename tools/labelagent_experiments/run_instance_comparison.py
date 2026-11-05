"""Frozen, isolated three-arm Mask2Former/SAM3/production-Agent comparison."""
import argparse, base64, copy, hashlib, io, json, os, subprocess, sys, time
from contextlib import contextmanager
from pathlib import Path
import urllib.request, urllib.parse


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def save(path, value):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + '.pending')
    temp.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    os.replace(temp, path)


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for part in iter(lambda: f.read(1048576), b''): h.update(part)
    return h.hexdigest()


def value_sha(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False).encode()).hexdigest()


def get(url):
    with urllib.request.urlopen(url, timeout=12) as response: return json.load(response)


def main():
    parser = argparse.ArgumentParser(description=__doc__); parser.add_argument('--config', required=True)
    parser.add_argument('--prepare-only', action='store_true')
    options = parser.parse_args(); cfg = read(options.config)
    for key in ('sam_url', 'vlm_url'):
        if urllib.parse.urlparse(cfg[key]).hostname not in ('127.0.0.1', 'localhost', '::1'):
            raise ValueError('This experiment only permits existing loopback model services')
    snapshot, out, dataset, package = (Path(cfg[k]) for k in ('source_snapshot','output','dataset','package'))
    sys.path[:0] = [str(snapshot), str(snapshot/'tools'), cfg['dependency_dir']]
    import cv2, numpy as np, psutil
    from PIL import Image
    from source.agent import workflow
    from source.agent.edit_agent import _shape_from_mask, _rasterize, valid_shape
    from run_local_agent_acceptance import load_image
    psutil.Process().nice(psutil.BELOW_NORMAL_PRIORITY_CLASS)
    out.mkdir(parents=True, exist_ok=True)
    if (out/'completed.json').exists(): raise FileExistsError('Completed comparison is immutable')
    stopped = out/'STOP'
    def check_stop():
        if stopped.exists(): raise RuntimeError('Stopped by experiment STOP file; existing outputs retained')
    def progress(phase, **extra):
        save(out/'progress.json', dict(phase=phase, pid=os.getpid(), time=time.time(), **extra))
        print(json.dumps(dict(phase=phase, **extra), ensure_ascii=False), flush=True)
    def yield_to_workbench():
        while True:
            check_stop()
            try: busy = get('http://127.0.0.1:8765/api/run/status').get('status') == 'running'
            except Exception: busy = False
            if not busy: return
            progress('waiting_for_user_workbench'); time.sleep(5)
    def resource_slot():
        for _ in range(120):
            check_stop(); yield_to_workbench()
            free, util = map(float, subprocess.check_output(['nvidia-smi','--query-gpu=memory.free,utilization.gpu','--format=csv,noheader,nounits'],text=True).strip().split(','))
            available = psutil.virtual_memory().available
            if free >= 3584 and util <= 15 and available >= 768*2**20: return
            progress('waiting_for_bounded_model_slot', free_gpu_MiB=free, utilization=util, available_RAM_MiB=round(available/2**20))
            time.sleep(10)
        raise RuntimeError('Resource wait exhausted; no other process stopped')

    manifest = read(dataset/'import_manifest.json')
    records = {(r['frame'], r['topic']): r for r in manifest['records'] if r['topic'].endswith('/image')}
    cases = [(f, c, f'/capture/camera_{c}/image') for f,c in cfg['cases']]
    annotations, original_description = read(cfg['reference_annotations']), read(cfg['reference_description'])
    rf, rc = cfg['reference_case']; reference_image = load_image(dataset, records[rf,f'/capture/camera_{rc}/image'])
    labels = ['pallet','person','floor','slope','ego','other_robot','cargo_floor']
    queries = {'pallet':'pallet','person':'person','floor':'floor','slope':'ramp',
               'ego':'vehicle hood','other_robot':'mobile robot','cargo_floor':'truck floor'}
    for item in original_description['prompts']:
        if item['label'] in ('pallet','person'): queries[item['label']] = item['text']
    rules = ('Exploratory seven-class bootstrap, NOT human-confirmed seven-class truth. '
             'pallet: visible pallet base only, never cargo body. person: visible human body. '
             'floor: visible ordinary horizontal warehouse floor, never vertical wall, rack or cargo. '
             'slope: visible ramp. ego: visible own robot body near the camera. other_robot: a separate mobile robot. '
             'cargo_floor: vehicle cargo-compartment floor, not ordinary warehouse ground. '
             'Leave uncertain regions unlabelled. Do not invent occluded structure. '
             'Image evidence is needed for edits; candidate masks are hypotheses, not references.')
    if cfg.get('brief_hole_reasons'):
        rules += (' Output formatting: each hole_assessment reason uses at most THREE English words; '
                  'other reasons use at most EIGHT English words. Include every required hole_id exactly once. '
                  'Do not omit evidence, decisions or required operations to save tokens.')
    bootstrap_description = {'description':rules,'prompts':[{'label':k,'text':queries[k]} for k in labels]}
    model_sha = sha(package/'model/model.safetensors')
    assert model_sha == 'd11c2e7f86cd5189605b857c4697e55a5caead0a6888b52b2f5ff814150f5291'
    before = get('http://127.0.0.1:8765/api/state')
    protected = lambda s: value_sha({k:s.get(k) for k in ('path','truth','reference_frames','topic')})
    images = {topic:load_image(dataset, records[f,topic]) for f,c,topic in cases}
    inputs = {topic:{str(f):records[f,topic]['sha256']} for f,c,topic in cases}
    plan = {'scope':'exploratory same-capture two-image comparison; not independent accuracy',
            'source_commit':cfg['source_commit'],'cases':cfg['cases'],'source_image_sha256':inputs,
            'archive_sha256':manifest['archive_sha256'],'mask2former_model_sha256':model_sha,
            'mask2former_config':read(package/'inference_config.json'),
            'inference_script_sha256':sha(package/'infer_single.py'),
            'agent_workflow_sha256':sha(snapshot/'source/agent/workflow.py'),
            'reference_annotation_sha256':sha(cfg['reference_annotations']),
            'reference_description_sha256':sha(cfg['reference_description']),
            'A':'existing two-class SAM3 text seeds plus production Agent and confirmed reference',
            'B':'original SAM seeds plus whole fixed-grid SAM proposals; image-only ranking; bounded residual gap recovery',
            'C':'all previous13-support PAL/person masks retained; append complete independently inferred instances under remaining original budget; unchanged original floor seed and three-round v6 review',
            'queries':queries,'bootstrap_rules':rules,'max_review_rounds':cfg['max_rounds'],
            'compact_response_max_tokens':cfg.get('review_max_tokens',768),
            'brief_hole_reasons':bool(cfg.get('brief_hole_reasons',False)),
            'hole_review_batch_size':cfg.get('hole_review_batch_size'),
            'hole_review_adapter_sha256':sha(Path(__file__).with_name('ground_action_review_v6.py')),
            'batch_acceptance':'Only actual SAM masks covering 100% of an approved gap may change that gap. Unmet regions stay untouched and are explicitly pending; warnings retained. No other warnings accepted.',
            'negative_prompt_policy':cfg['sam_gap_negative_policy'],
            'instance_children_sha256':{f'camera{c}_frame{f}':{a:sha(Path(cfg['instance_generation'])/f'camera{c}_frame{f}'/(a+'_children.json')) for a in ['B','C']} for f,c,_ in cases},
            'generation_script_sha256':sha(Path(__file__).with_name('review_instance_consistency.py')),
            'whole_mask_policy_sha256':sha(Path(__file__).with_name('incumbent_append_policy.py')),
            'generation_protocol_sha256':sha(Path(cfg['instance_generation'])/'generation_protocol.json'),
            'protocol_amendment_sha256':sha(out/'protocol_amendment.json'),
            'new_fixed_model_grid_sha256':sha(Path(cfg['generation_source'])/'completed.json'),
            'new_fixed_model_grid_forwards':16,
            'residual_policy_sha256':sha(Path(__file__).with_name('residual_gap_policy.py')),
            'instance_geometry_sha256':sha(Path(__file__).with_name('instance_separation_geometry.py')),
            'instance_evidence_sha256':sha(Path(__file__).with_name('instance_evidence_views.py')),
            'instance_consistency_sha256':sha(Path(__file__).with_name('instance_identity_consistency.py')),
            'wire_geometry_policy_sha256':sha(Path(__file__).with_name('wire_geometry_policy.py')),
            'append_policy_sha256':sha(Path(__file__).with_name('incumbent_append_policy.py')),
            'semantic_subject_sha256':sha(Path(__file__).with_name('semantic_subject_guard.py')),
            'optimization_plan_sha256':sha(Path(cfg['optimization_plan'])),
            'semantic_guard':'Raw-photo-only category must independently confirm ordinary floor before fill; conflicts veto edits. Initial candidates protected from destructive global proposals.',
            'hole_transport':'one image-bound gap action per real request; original IDs attached by exact input binding',
            'gap_reply_budget':256,'gap_image_budget':256,'global_object_only_review':True,
            'geometry_validation_sha256':sha(snapshot/'source/agent/edit_agent.py'),
            'keyed_schema_sha256':sha(snapshot/'source/agent/protocol_schema.py'),
            'keyed_decoder_sha256':sha(snapshot/'source/agent/keyed_holes.py'),
            'tiled_model_additions_sha256':{f'camera{c}_frame{f}':sha(Path(cfg['additional_pal_shapes'])/f'camera{c}_frame{f}'/'shapes.json') for f,c,_ in cases},
            'repair_rule':'Retain every incumbent foreground mask. Append only complete new raw-query or no-prior seam-SAM masks passing unchanged neutral checks plus single-base controls; reject whole on budget. No reference-based selection.',
            'operation_cap_per_whole_agent_round':3,
            'sam_text_reused_across_arms_with_identical_input_and_query':True,
            'mask_refinement':'up to three component-interior positive clicks, two automatic exterior clicks, original prior mask; reject invalid or warned output',
            'missing_class_completion':'same seven fixed SAM3 text queries, only for classes absent from Mask2Former seeds; accounted separately',
            'ground_truth_limits':'Only pallet/person have sparse projected 3D reference; floor/other classes have qualitative previews, not dense accuracy.',
            'no_default_model_or_annotation_changes':True,'no_automatic_adoption_from_two_images':True}
    if (out/'frozen_plan.json').exists() and read(out/'frozen_plan.json') != plan:
        raise ValueError('Frozen plan changed; do not resume under a different protocol')
    save(out/'frozen_plan.json', plan); plan_sha = sha(out/'frozen_plan.json')
    if options.prepare_only:
        progress('prepared_no_model_requests', plan_sha256=plan_sha)
        return
    reference = {'image':reference_image,'marked_image':workflow.reference_overlay(reference_image, annotations),'annotations':annotations}
    config = {'sam3_url':cfg['sam_url'],'llm_url':cfg['vlm_url'],'llm_model':cfg['vlm_model'],
              'llm_key':'local','llm_prompt_profile':'compact','llm_duplicate_review':False,
              'llm_max_tokens':cfg.get('review_max_tokens',768),'llm_keyed_holes':True,'llm_batch_reviewed_gaps':True,
              'sam_accept_complete_gap_subsets':True,'sam_gap_negative_policy':cfg['sam_gap_negative_policy'],'llm_residual_gap_recovery':True}
    trace = {r['id']:r for r in read(out/'model_trace.json')} if (out/'model_trace.json').exists() else {}
    inherited_trace_ids=set(trace);new_mask_forwards=0
    def log_call(event):
        trace[event['id']] = copy.deepcopy(event); save(out/'model_trace.json', list(trace.values()))
    def sam(endpoint, image, **prompt):
        for attempt in range(40):
            check_stop(); yield_to_workbench()
            try: return workflow.sam_request(config, endpoint, image, **prompt)
            except ValueError as error:
                if 'Another model request' not in str(error): raise
                progress('waiting_for_existing_model_request', attempt=attempt+1);time.sleep(3)
        raise RuntimeError('Existing model request remained busy')
    original_chat = workflow._chat
    def cooperative_chat(*args, **kwargs):
        for attempt in range(40):
            check_stop();yield_to_workbench()
            try:return original_chat(*args, **kwargs)
            except ValueError as error:
                if 'Another model request' not in str(error):raise
                progress('waiting_for_existing_model_request', attempt=attempt+1);time.sleep(3)
        raise RuntimeError('Existing model request remained busy')
    workflow._chat = cooperative_chat
    original_sam=workflow.sam_request
    def cooperative_sam(*args,**kwargs):
        for attempt in range(80):
            check_stop();yield_to_workbench()
            try:return original_sam(*args,**kwargs)
            except ValueError as error:
                if 'Another model request' not in str(error):raise
                progress('waiting_for_existing_model_request',attempt=attempt+1);time.sleep(3)
        raise RuntimeError('Existing model request remained busy')
    workflow.sam_request=cooperative_sam
    if cfg.get('hole_review_batch_size'):
        from ground_action_review_v6 import install_review_batches
        batches=[]
        def batch_record(value):
            batches.append(value);save(out/'batch_reviews.json',batches)
        install_review_batches(workflow,cfg['hole_review_batch_size'],3,batch_record,cache_dir=out/'gap_action_cache')
    all_seeds, drafts, refinement = {}, {}, []
    with workflow.trace_calls(log_call):
        # All model-network stages precede VLM review to limit normal GPU swaps.
        for frame,camera,topic in cases:
            case = f'camera{camera}_frame{frame}';folder=out/'mask2former'/case
            source=out/'inputs'/(case+'.png');source.parent.mkdir(exist_ok=True);source.write_bytes(images[topic])
            if not (folder/'prediction.json').exists():
                resource_slot();progress('mask2former_original_png',case=case)
                result=subprocess.run([cfg['python'],str(package/'infer_single.py'),'--image',str(source),'--outdir',str(folder),
                    '--device','cuda','--dependency-dir',cfg['dependency_dir']],capture_output=True,text=True,timeout=300)
                (out/(case+'_mask2former.log')).write_text(result.stdout+result.stderr,encoding='utf-8')
                if result.returncode:raise RuntimeError('Delivered Mask2Former inference failed; private log retained')
                new_mask_forwards+=1
            else:progress('exact_frozen_mask2former_cache_reused',case=case)
            pixels=np.asarray(Image.open(folder/'panoptic.png')).astype(np.uint32)
            ids=pixels[:,:,0]+256*pixels[:,:,1]+65536*pixels[:,:,2];meta=read(folder/'prediction.json')
            assert meta['model_forward_passes']==1
            shapes=[]
            for s in meta['segments_info']:
                shape=_shape_from_mask((ids==s['id']).astype(np.uint8),labels[s['category_id']-1],len(shapes)+1)
                if shape:
                    shape['score']=s.get('score');shapes.append(shape)
            drafts[topic]=shapes;save(folder/'shapes.json',shapes)
        text_seeds = {}
        for frame,camera,topic in cases:
            case=f'camera{camera}_frame{frame}';image=images[topic]
            creator=workflow.AnnotationAgent(config,image,bootstrap_description,labels,1)
            text_seeds[topic]={}
            for label in labels:
                progress('sam3_text',case=case,label=label)
                cached=out/'sam_text'/case/(label+'.json')
                response=read(cached) if cached.exists() else sam('/segment_text',image,prompt=queries[label])
                save(cached,response)
                text_seeds[topic][label]=creator.candidates(response.get('results',[]),label)
        for frame,camera,topic in cases:
            case=f'camera{camera}_frame{frame}';image=images[topic];im=Image.open(io.BytesIO(image));w,h=im.size
            raw=copy.deepcopy(drafts[topic]);refined=[]
            save(out/'seed_stages'/case/'mask_whole.json',raw)
            tiled=[s for s in read(Path(cfg['additional_pal_shapes'])/case/'shapes.json') if s['label']=='pallet']
            save(out/'seed_stages'/case/'mask_all_tiles.json',raw+copy.deepcopy(tiled))
            children={a:read(Path(cfg['instance_generation'])/case/(a+'_children.json')) for a in ['B','C']}
            save(out/'seed_stages'/case/'repair_proposals.json',{'trained_crop_cues':len(tiled),'B_children':len(children['B']),'C_children':len(children['C']),'reference_used':False,'whole_unclipped_masks':True})
            for source_shape in raw:
                mask=_rasterize(source_shape,w,h);n,cc,stats,_=cv2.connectedComponentsWithStats(mask,connectivity=8)
                components=sorted(range(1,n),key=lambda i:int(stats[i,cv2.CC_STAT_AREA]),reverse=True)[:3]
                clicks=[]
                for component in components:
                    distance=cv2.distanceTransform((cc==component).astype(np.uint8),cv2.DIST_L2,5)
                    y,x=np.unravel_index(np.argmax(distance),distance.shape)
                    clicks.append({'x':int(x),'y':int(y),'label':1})
                ring=cv2.dilate(mask,np.ones((21,21),np.uint8))&~cv2.dilate(mask,np.ones((7,7),np.uint8))
                for _ in range(2):
                    if not ring.any():break
                    distance=cv2.distanceTransform(ring.astype(np.uint8),cv2.DIST_L2,5);y,x=np.unravel_index(np.argmax(distance),distance.shape)
                    clicks.append({'x':int(x),'y':int(y),'label':0});cv2.circle(ring,(int(x),int(y)),25,0,-1)
                buffer=io.BytesIO();Image.fromarray(mask*255).save(buffer,'PNG')
                progress('sam3_mask_point_refinement',case=case,label=source_shape['label'],shape_id=source_shape['id'])
                cached=out/'refinement'/case/(str(source_shape['id'])+'.json')
                if cached.exists():
                    saved=read(cached)
                    assert saved['record']['clicks']==clicks
                    response=saved['response']
                else:
                    failed=[v for v in trace.values() if v.get('service')=='SAM 3' and v.get('status')=='failed'
                        and v.get('input',{}).get('endpoint')=='/segment_points'
                        and v.get('input',{}).get('image',{}).get('sha256')==hashlib.sha256(image).hexdigest()
                        and v.get('input',{}).get('mask',{}).get('sha256')==hashlib.sha256(buffer.getvalue()).hexdigest()
                        and v.get('input',{}).get('points')==clicks]
                    if failed and any(word in failed[-1].get('error','') for word in ('无法排除','负点','提示点')):
                        response={'recorded_rejection':failed[-1]['error'],'reused_failed_request_without_repeating':True}
                    else:
                        try:response=sam('/segment_points',image,points=clicks,mask=base64.b64encode(buffer.getvalue()).decode())
                        except ValueError as error:
                            if not any(word in str(error) for word in ('无法排除','负点','提示点')):raise
                            response={'recorded_rejection':str(error)}
                row={'case':case,'shape_id':source_shape['id'],'label':source_shape['label'],'clicks':clicks,'warnings':response.get('warnings',[])}
                if response.get('recorded_rejection'):row['rejected_reason']=response['recorded_rejection']
                result=copy.deepcopy(source_shape)
                if response.get('parts') and not response.get('warnings'):
                    candidate={'id':source_shape['id'],'label':source_shape['label'],'parts':response['parts'],'score':response.get('score')}
                    if valid_shape(candidate,w,h):result=candidate;row['accepted_technical_output']=True
                row.setdefault('accepted_technical_output',False)
                row['changed_pixels']=int((_rasterize(result,w,h)!=mask).sum());refinement.append(row);refined.append(result)
                save(out/'refinement'/case/(str(source_shape['id'])+'.json'),{'record':row,'response':response})
            present={s['label'] for s in raw};complete=[copy.deepcopy(s) for s in refined if s['label']!='floor']
            complete.extend(copy.deepcopy(text_seeds[topic]['floor']))
            for label in labels:
                if label not in present or label=='pallet':complete.extend(copy.deepcopy(text_seeds[topic][label]))
            def renumber(shapes):
                for i,s in enumerate(shapes,1):s['id']=i
                return shapes
            baseline=[copy.deepcopy(s) for label in ('pallet','person') for s in text_seeds[topic][label]]
            seven=[copy.deepcopy(s) for label in labels for s in text_seeds[topic][label]]
            seven.extend(copy.deepcopy(children['B']))
            complete.extend(copy.deepcopy(children['C']))
            all_seeds[topic]={'A_current':renumber(baseline),'B_sam7':renumber(seven),'C_hybrid':renumber(complete)}
            save(out/'seed_stages'/case/'mask_raw.json',raw);save(out/'seed_stages'/case/'mask_refined.json',refined)
            save(out/'seed_stages'/case/'all_arms.json',all_seeds[topic])
        arms={name:{} for name in ('A_current','B_sam7','C_hybrid')};run_rows=[]
        old_rows={(r['case'],r['arm']):r for r in read(out/'agent_runs.json')} if (out/'agent_runs.json').exists() else {}
        for frame,camera,topic in cases:
            case=f'camera{camera}_frame{frame}';image=images[topic]
            for arm in arms:
                seed=copy.deepcopy(all_seeds[topic][arm]);directory=out/arm/case;directory.mkdir(parents=True,exist_ok=True)
                if (directory/'result.json').exists() and (case,arm) in old_rows:
                    result=read(directory/'result.json');arms[arm].setdefault(topic,{})[str(frame)]=result['shapes']
                    run_rows.append(old_rows[case,arm]);progress('completed_agent_cache_reused',case=case,arm=arm)
                    continue
                labels_run=['pallet','person'] if arm=='A_current' else labels
                description=original_description if arm=='A_current' else bootstrap_description
                ref=reference if arm=='A_current' else None
                class SeededAgent(workflow.AnnotationAgent):
                    def generate(self):
                        self.shapes=copy.deepcopy(seed);self.generated=len(seed)
                        self.config['protected_initial_ids']=[s['id'] for s in seed]
                        self.next_id=max([s['id'] for s in seed]+[0])+1;self.progress('real_seed_cache_loaded')
                events=[]
                def event(row):
                    check_stop();events.append({'time':time.time(),'phase':row['phase'],'round':row['round'],'shape_count':len(row['shapes'])})
                    save(directory/'events.json',events)
                    if row['phase'] in ('real_seed_cache_loaded','observing','edited','finished'):
                        name=f"{len(events):03d}_{row['phase']}.jpg"
                        (directory/name).write_bytes(workflow.candidate_overlay(image,row['shapes']))
                old_rules=workflow.COMPACT_REFERENCE_RULES
                if arm!='A_current':workflow.COMPACT_REFERENCE_RULES=rules+'\nOnly listed classes; no human seven-class reference was provided. Never call model predictions truth.'
                started=time.monotonic();before_calls=set(trace)
                progress('agent_review',case=case,arm=arm)
                try:
                    result=SeededAgent(config,image,description,labels_run,cfg['max_rounds'],event,reference=ref).execute()
                finally:workflow.COMPACT_REFERENCE_RULES=old_rules
                save(directory/'result.json',result);arms[arm].setdefault(topic,{})[str(frame)]=result['shapes']
                calls=[v for k,v in trace.items() if k not in before_calls]
                applied=[v for audit in result.get('agent',{}).get('audit',[]) for v in audit.get('tool_results',[]) if v.get('status')=='applied']
                run_rows.append({'case':case,'arm':arm,'initial_counts':{label:sum(s['label']==label for s in seed) for label in labels_run},
                    'final_counts':{label:sum(s['label']==label for s in result['shapes']) for label in labels_run},
                    'rounds':result.get('agent',{}).get('rounds'),'converged':result.get('agent',{}).get('converged'),
                    'stop_reason':result.get('agent',{}).get('stop_reason'),'applied_edits':len(applied),
                    'successful_vlm_calls':sum(v.get('service')=='GPT' and v.get('status')=='succeeded' for v in calls),
                    'elapsed_seconds':round(time.monotonic()-started,2)})
                save(out/'agent_runs.json',run_rows)
    # Freeze every arm before reading the selected 3D reference for scoring.
    score_dirs={}
    for arm,mapping in arms.items():
        common={topic:{f:[s for s in shapes if s['label'] in ('pallet','person')] for f,shapes in frames.items()} for topic,frames in mapping.items()}
        save(out/arm/'all_candidates.json',mapping);save(out/arm/'candidates_by_topic.json',common)
        save(out/arm/'prediction_receipt.json',{'archive_sha256':manifest['archive_sha256'],
            'prediction_json_sha256':sha(out/arm/'candidates_by_topic.json'),'image_sha256':inputs,'confirmed_as_truth':False})
    for arm in arms:
        progress('original_sparse_reference_scoring',arm=arm);directory=out/arm/'evaluation'
        if (directory/'summary.json').exists() and read(directory/'summary.json').get('status')=='complete':
            score_dirs[arm]=read(directory/'summary.json')['groups']['all'];continue
        process=subprocess.run([cfg['python'],str(snapshot/'tools/evaluate_projected_support.py'),'--dataset',str(dataset),
            '--candidates-json',str(out/arm/'candidates_by_topic.json'),'--prediction-receipt',str(out/arm/'prediction_receipt.json'),
            '--output',str(directory),'--frames','76','--cameras','0,1'],capture_output=True,text=True,timeout=240,
            env={**os.environ,'PYTHONPATH':cfg['dependency_dir']})
        (out/arm/'evaluation.log').write_text(process.stdout+process.stderr,encoding='utf-8')
        if process.returncode:raise RuntimeError('Original sparse reference scorer failed; private log retained')
        score_dirs[arm]=read(directory/'summary.json')['groups']['all']
    after=get('http://127.0.0.1:8765/api/state')
    assert sha(package/'model/model.safetensors')==model_sha and sha(out/'frozen_plan.json')==plan_sha
    assert all(hashlib.sha256(load_image(dataset,records[f,topic])).hexdigest()==inputs[topic][str(f)] for f,c,topic in cases)
    final={'status':'complete','cases':cfg['cases'],'plan_sha256':plan_sha,'model_sha256':model_sha,
           'source_images_unchanged':True,'weights_unchanged':True,'protected_state_before_after_equal':protected(before)==protected(after),
           'annotation_writes':0,'model_training_updates':0,'dense_pixel_accuracy_evaluated':False,
           'real_mask2former_forwards_source':2,'diagnostic_only_mask2former_forwards':16,
           'fresh_mask2former_forwards_this_run':new_mask_forwards,
           'inherited_real_model_trace_entries':len(inherited_trace_ids),
           'sam_calls_succeeded':sum(v.get('service')=='SAM 3' and v.get('status')=='succeeded' for v in trace.values()),
           'vlm_calls_succeeded':sum(v.get('service')=='GPT' and v.get('status')=='succeeded' for v in trace.values()),
           'refinement':refinement,'agent_runs':run_rows,'common_class_scores':score_dirs,
           'scope':'two exposed same-capture images, exploratory comparison; floor/other classes lack dense truth; no automatic adoption'}
    save(out/'completed.json',final);progress('complete',completed_agent_runs=len(run_rows),converged_agent_runs=sum(bool(r['converged']) for r in run_rows))


if __name__=='__main__':
    main()
