const $=id=>document.getElementById(id);let candidates={},agentRuns={},candidateReviewProposals=[],candidateReviewBindings={},truthReviewStatus={},topics={},topic='',frame=0,shapes={},draft=[],playing=null,labels=[],autoLabels=[],referenceOnlyLabels=[],labelStateSha256='',configStateSha256='',fileLibraryStateSha256='',imageWidth=0,imageHeight=0,loading=false,frameLoadGeneration=0,frameNavigationGeneration=0,playback=null,playGeneration=0,presentedTime=0,catalog={},recordDuration=0,clockTime=0,clockAnchor=0,clockBase=0,rate=1,tickHandle=null,extraWindows=new Map(),windowSerial=0,annotationMode=false,annotationView=null,pointMode=false,negativeMode=false,promptPoints=[],segmentBusy=false,pointPromptStartedAt=0,pointPromptTimer=null,pointPromptNotice='',pointPromptAutoRunOnce=false,pointPromptGeneratedPointCount=0,pointPromptRequiresRefinement=false,generatedShapeFrame=null,generatedShapeTopic='',generatedShapeIndex=-1,regionMode='',regionEditGeneration=0,selectedShapeIndex=-1,zoomScale=1,referencePreparationIntent=null,referenceFinalizeBusy=false,referenceFinalizeNotice='',referenceReviewSolo=false,referenceReviewFocused=false,referenceReviewOriginalOnly=false,referenceOriginalSweepVisited=false,referenceOpenSetReviewed=false,referenceReviewedShapeKeys=new Set(),referenceDismissedMissingLabels=new Set(),candidateStateSignature='',loadedDataPath='',loadedFileActivationGeneration=0,pendingDatasetLoad=null,restoringFrameNavigation=false,compactGeometryMode=false,truthFrameIndex=new Set(),candidateCountIndex={},workbenchConfig={},runResultHandled=new Set(),runResultAutoRestoreDeferred=new Set(),bootstrapComplete=false,workspaceRestoreState='booting',workspaceReadyResolved=false,resolveWorkspaceReady,retryFrames=[],retryScope=null,retryMode='';
const workspaceReadyDeferred=new Promise(resolve=>{resolveWorkspaceReady=resolve});
async function api(path,body){const response=await fetch('/api/'+path,{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(body)});const raw=await response.text();let data;try{data=JSON.parse(raw)}catch{throw Error(`接口 ${path} 返回非 JSON 响应（HTTP ${response.status}），请重启标注服务`)}if(!response.ok){const error=Error(data.error||`请求失败（HTTP ${response.status}）`);error.status=response.status;error.payload=data;throw error}return data}
function applyCompactFrameState(data,target,expectedTopic=topic,expectedPath=loadedDataPath){
 if(!data?.compact_geometry||data.topic!==expectedTopic||data.path!==expectedPath||Number(data.frame)!==Number(target))throw Error('按需标注层与当前文件、相机或帧不一致，已拒绝显示');
 const key=String(target),replace=(store,source)=>{if(Object.prototype.hasOwnProperty.call(source||{},key))store[key]=source[key];else delete store[key]};
 replace(shapes,data.truth);replace(truthReviewStatus,data.truth_review_status);replace(candidates,data.candidates);replace(agentRuns,data.agent_runs);replace(candidateReviewBindings,data.candidate_review_bindings);
 candidateCountIndex[key]=Array.isArray(data.candidates?.[key])?data.candidates[key].length:0;
 candidateReviewProposals=(candidateReviewProposals||[]).filter(item=>!(item?.file===expectedPath&&item?.topic===expectedTopic&&Number(item?.frame)===Number(target))).concat(data.candidate_review_proposals||[]);
 if(Array.isArray(data.truth_frames))truthFrameIndex=new Set(data.truth_frames.map(String));
 if(data.candidate_counts&&typeof data.candidate_counts==='object')candidateCountIndex={...data.candidate_counts};
 return data;
}
async function readCompactFrameStateSnapshot(target,expectedTopic=topic,expectedPath=loadedDataPath,expectedGeneration=loadedFileActivationGeneration){
 const response=await fetch(`/api/frame_state/${target}?topic=${encodeURIComponent(expectedTopic)}&path=${encodeURIComponent(expectedPath)}&file_activation_generation=${encodeURIComponent(expectedGeneration)}`,{cache:'no-store'}),data=await response.json();
 if(!response.ok)throw Error(data.error||'按需读取当前帧标注失败');
 if(!data?.compact_geometry||data.topic!==expectedTopic||data.path!==expectedPath||data.file_activation_generation!==expectedGeneration||Number(data.frame)!==Number(target))throw Error('按需标注层与当前文件载入代次、相机或帧不一致，已拒绝显示');
 return data;
}
async function fetchCompactFrameState(target,generation=frameLoadGeneration){
 const expectedTopic=topic,expectedPath=loadedDataPath,data=await readCompactFrameStateSnapshot(target,expectedTopic,expectedPath);
 if(generation!==frameLoadGeneration||topic!==expectedTopic||loadedDataPath!==expectedPath)throw Error('读取期间文件、相机或帧已变化，已丢弃旧标注层');
 applyCompactFrameState(data,target,expectedTopic,expectedPath);return data;
}
window.readCompactFrameStateSnapshot=readCompactFrameStateSnapshot;
window.fetchCompactFrameState=fetchCompactFrameState;
window.currentWorkbenchFrameState=()=>({compact_geometry:compactGeometryMode,path:loadedDataPath,topic,file_activation_generation:loadedFileActivationGeneration,labels:[...labels],auto_labels:[...autoLabels],reference_only_labels:[...referenceOnlyLabels],label_state_sha256:labelStateSha256,config:{...workbenchConfig},config_state_sha256:configStateSha256,truth:shapes,truth_review_status:truthReviewStatus,truth_frames:[...truthFrameIndex],candidates,agent_runs:agentRuns,candidate_counts:{...candidateCountIndex},candidate_review_proposals:candidateReviewProposals,candidate_review_bindings:candidateReviewBindings,reference_frames:window.referenceFrames||{}});
window.reviewPacketWorkbenchSnapshot=()=>({path:loadedDataPath,topic,frame,
 shapes:JSON.parse(JSON.stringify(selectedShapes())),truth_verified:truthIsVerified(frame),
 annotation_state:frameAnnotationState(frame),loading:Boolean(loading),playing:Boolean(playing)});
function publishWorkbenchFrameState(){document.dispatchEvent(new CustomEvent('labelagent:frame-state',{detail:{state:window.currentWorkbenchFrameState()}}))}
const viewedFrameStorageKey='labelagentViewedFrameV105';
function readViewedFrames(){try{const value=JSON.parse(sessionStorage.getItem(viewedFrameStorageKey)||'{}');return value&&typeof value==='object'&&!Array.isArray(value)?value:{}}catch{return {}}}
function viewedFrameScope(selectedTopic=topic,sourcePath=loadedDataPath){return sourcePath&&selectedTopic?`${sourcePath}|${selectedTopic}`:''}
function rememberedViewedFrame(selectedTopic=topic,sourcePath=loadedDataPath){const scope=viewedFrameScope(selectedTopic,sourcePath),entry=scope?readViewedFrames()[scope]:null,value=Number(entry?.frame),count=Number(topics?.[selectedTopic]?.count||0);return Number.isInteger(value)&&value>=0&&value<count?value:null}
function rememberViewedFrame(){if(restoringFrameNavigation)return;const scope=viewedFrameScope(),count=Number(topics?.[topic]?.count||0);if(!scope||!Number.isInteger(frame)||frame<0||frame>=count)return;const all=readViewedFrames();all[scope]={frame,recorded_at:new Date().toISOString(),browser_session_only:true};try{sessionStorage.setItem(viewedFrameStorageKey,JSON.stringify(all))}catch{}}
function focusPointPromptCanvas(){requestAnimationFrame(()=>{const status=$('pointPromptCanvasStatus'),stage=$('stage');stage?.scrollIntoView({behavior:'smooth',block:'center'});status?.focus({preventScroll:true})})}
function toast(message){$('toast').textContent=message;$('toast').style.display='block';setTimeout(()=>$('toast').style.display='none',3500)}
window.labelAgentToast=toast;
window.workspaceMode='manual';let manualDrafts={},autoDrafts={};
const workbenchHeights={},workbenchSplitter=$('workbenchSplitter');
let workbenchDrag=null;
function workbenchHeightLimits(){
 const viewer=document.querySelector('.viewer');
 const reserved=document.querySelector('.windowToolbar').getBoundingClientRect().height+document.querySelector('.playbackBar').getBoundingClientRect().height+$('imageInfo').getBoundingClientRect().height+workbenchSplitter.getBoundingClientRect().height+180;
 const max=Math.max(0,Math.floor(viewer.clientHeight-reserved));
 return {min:Math.min(160,max),max};
}
window.resizeWorkbench=()=>{
 const panel=$('workbench'),limits=workbenchHeightLimits(),preferred=workbenchHeights[window.workspaceMode];
 panel.style.maxHeight=limits.max+'px';
 if(preferred===undefined)panel.style.removeProperty('height');
 else panel.style.height=Math.max(limits.min,Math.min(limits.max,preferred))+'px';
 workbenchSplitter.setAttribute('aria-valuemin',limits.min);
 workbenchSplitter.setAttribute('aria-valuemax',limits.max);
 workbenchSplitter.setAttribute('aria-valuenow',Math.round(panel.getBoundingClientRect().height));
};
function setWorkbenchHeight(height){
 const limits=workbenchHeightLimits();
 workbenchHeights[window.workspaceMode]=Math.max(limits.min,Math.min(limits.max,height));
 window.resizeWorkbench();
}
workbenchSplitter.addEventListener('pointerdown',event=>{
 if(!event.isPrimary||event.button!==0)return;
 event.preventDefault();
 workbenchDrag={id:event.pointerId,y:event.clientY,height:$('workbench').getBoundingClientRect().height};
 workbenchSplitter.setPointerCapture(event.pointerId);
 document.body.classList.add('resizingWorkbench');
});
workbenchSplitter.addEventListener('pointermove',event=>{
 if(workbenchDrag?.id===event.pointerId)setWorkbenchHeight(workbenchDrag.height+workbenchDrag.y-event.clientY);
});
function finishWorkbenchResize(event){
 if(workbenchDrag?.id!==event.pointerId)return;
 workbenchDrag=null;
 document.body.classList.remove('resizingWorkbench');
 if(workbenchSplitter.hasPointerCapture(event.pointerId))workbenchSplitter.releasePointerCapture(event.pointerId);
}
for(const name of ['pointerup','pointercancel','lostpointercapture'])workbenchSplitter.addEventListener(name,finishWorkbenchResize);
workbenchSplitter.addEventListener('keydown',event=>{
 const height=$('workbench').getBoundingClientRect().height,limits=workbenchHeightLimits();
 const next={ArrowUp:height+20,ArrowDown:height-20,Home:limits.min,End:limits.max}[event.key];
 if(next===undefined)return;
 event.preventDefault();setWorkbenchHeight(next);
});
new ResizeObserver(()=>window.resizeWorkbench()).observe(document.querySelector('.viewer'));
window.resizeWorkbench();
function selectedShapes(){const key=String(frame);if(window.workspaceMode==='auto'&&candidates[key]!==undefined)return candidates[key];return shapes[key]??(window.workspaceMode==='manual'?(manualDrafts[key]??(manualDrafts[key]=[])):(autoDrafts[key]??(autoDrafts[key]=[])))}
function truthIsVerified(index=frame){return truthReviewStatus[String(index)]?.reusable_as_reference===true}
function frameAnnotationState(index=frame){const key=String(index);if(window.workspaceMode==='auto'&&candidates[key]!==undefined)return shapes[key]!==undefined?'comparison_candidate':'candidate';return shapes[key]!==undefined?(truthIsVerified(index)?'truth':'legacy_truth'):window.workspaceMode==='manual'&&Array.isArray(manualDrafts[key])&&manualDrafts[key].length?'manual_draft':'empty'}
const labelVisuals={
 '车辆':{stroke:'#ff7a18',fill:'rgba(255,122,24,.27)'},vehicle:{stroke:'#ff7a18',fill:'rgba(255,122,24,.27)'},
 pallet:{stroke:'#39d353',fill:'rgba(57,211,83,.24)'},person:{stroke:'#ff4f9a',fill:'rgba(255,79,154,.25)'},
 floor:{stroke:'#3b9cff',fill:'rgba(59,156,255,.24)'},slope:{stroke:'#ffb020',fill:'rgba(255,176,32,.28)'},
 ego:{stroke:'#b46cff',fill:'rgba(180,108,255,.27)'},other_robot:{stroke:'#19d3e6',fill:'rgba(25,211,230,.25)'},
 cargo_floor:{stroke:'#f3d43b',fill:'rgba(243,212,59,.27)'}
};
const fallbackLabelVisuals=[
 {stroke:'#38d6a2',fill:'rgba(56,214,162,.25)'},{stroke:'#ff8364',fill:'rgba(255,131,100,.25)'},
 {stroke:'#70a5ff',fill:'rgba(112,165,255,.25)'},{stroke:'#d07cff',fill:'rgba(208,124,255,.25)'}
];
function labelVisual(label){if(labelVisuals[label])return labelVisuals[label];let hash=0;for(const char of String(label))hash=(Math.imul(hash,31)+char.codePointAt(0))>>>0;return fallbackLabelVisuals[hash%fallbackLabelVisuals.length]}
function machineDraftVisible(){return ['machine_candidate','sam_point_prompt'].includes(referencePreparationIntent?.source)||(window.workspaceMode==='auto'&&candidates[String(frame)]!==undefined)}
function refreshClassColorLegend(items=selectedShapes()){
 const legend=$('classColorLegend');if(!legend)return;const names=[...new Set((items||[]).map(item=>item.label).filter(Boolean))];legend.hidden=!referencePreparationIntent||!names.length;legend.replaceChildren();
 if(legend.hidden)return;const title=document.createElement('strong');title.textContent='类别颜色';legend.append(title);
 for(const name of names){const chip=document.createElement('span'),swatch=document.createElement('i');swatch.style.background=labelVisual(name).stroke;chip.append(swatch,document.createTextNode(name));legend.append(chip)}
 const source=document.createElement('span');source.className='draftLineKey';const line=document.createElement('i');line.className=machineDraftVisible()?'dashed':'solid';source.append(line,document.createTextNode(machineDraftVisible()?'虚线＝机器草稿':'实线＝人工轮廓'));legend.append(source)
}
async function openCandidateFrame(index,selectedTopic=topic){
 await window.setWorkspaceMode?.('auto',false);
 if(selectedTopic!==topic)throw Error('候选所属图像源与当前图像源不一致');
 let view=annotationView;
 if(!view||view.sources[0]!==selectedTopic){
  view=[...extraWindows.values()].find(item=>item.type==='image'&&item.sources[0]===selectedTopic&&!item.closed);
  if(!view){await addWindow('image',[selectedTopic]);view=[...extraWindows.values()].find(item=>item.type==='image'&&item.sources[0]===selectedTopic&&!item.closed)}
  if(!view)throw Error('无法打开候选帧的图像窗口');
  await activateAnnotation(view);
  if(annotationView!==view)throw Error('无法进入标注复核模式');
 }
 const opened=await showFrame(index);
 return opened===true&&frame===Number(index)&&topic===selectedTopic;
}
function refreshLabels(){let select=$('labelSelect');select.innerHTML='';labels.forEach(x=>select.add(new Option(x,x)))}
function refreshAutoLabels(){
 const host=$('autoLabelOptions');host.innerHTML='';
 labels.forEach(label=>{const wrapper=document.createElement('label'),input=document.createElement('input');input.type='checkbox';input.value=label;input.checked=autoLabels.includes(label);wrapper.className=referenceOnlyLabels.includes(label)?'referenceLabel':'unreferencedLabel';wrapper.title=referenceOnlyLabels.includes(label)?'参考帧已有此类别':'参考帧没有此类别，将使用通用对象关键词';wrapper.append(input,document.createTextNode(label));host.append(wrapper)});
}
function applyInferenceLabelState(data,refresh=true){
 if(!data||!Array.isArray(data.labels))return false;
 labels=[...data.labels];
 autoLabels=Array.isArray(data.auto_labels)?data.auto_labels.filter(label=>labels.includes(label)):[...labels];
 referenceOnlyLabels=Array.isArray(data.reference_only_labels)?data.reference_only_labels.filter(label=>labels.includes(label)):[];
 labelStateSha256=typeof data.label_state_sha256==='string'?data.label_state_sha256:'';
 refreshLabels();refreshAutoLabels();$('labelsInput').value=labels.join(',');if(refresh)refreshCandidateSummary();
 return true;
}
function datasetIdentityForPath(pathValue){
 const value=String(pathValue||''),lower=value.toLowerCase(),basename=value.split(/[\\/]/).pop()||'未命名数据';
 if(lower.includes('sfl_cpd15y_20261104')||lower.includes('20261104_smoke')){const sample=basename.match(/^(003|004)(?:[-_.]|$)/i)?.[1];return {kind:'new',badge:'问题包抽样',name:`SFL-CPD15-Y · 批次 B（1104）· 问题包 ${sample||'003/004'} · 单图抽样`,file:basename}}
 if(lower.includes('capture_20261103'))return {kind:'new',badge:'新实采数据',name:'Capture_20261103 · 常熟实采 · 四相机',file:basename};
 if(lower.includes('sfl_cpd15y_20261111')||lower.includes('batcha_0920'))return {kind:'new',badge:'新数据集',name:'SFL-CPD15-Y · 批次 A（0920）· 000',file:basename};
 if(lower.includes('cvat_task3_20261028')||lower.includes('cvat')&&lower.includes('task3'))return {kind:'old',badge:'旧验证集',name:'CVAT Task3 · 斜坡复核集',file:basename};
 return value?{kind:'other',badge:'已加载数据',name:basename,file:basename}:{kind:'unloaded',badge:'尚未加载数据',name:'请先选择图片或 MCAP 数据',file:''};
}
function updateDatasetIdentity(){
 const host=$('datasetIdentity');if(!host)return;const pending=pendingDatasetLoad,identity=datasetIdentityForPath(pending?.path||loadedDataPath),count=Number(topics?.[topic]?.count||0),imageTopics=Object.entries(catalog||{}).filter(([,item])=>item?.type==='image').length;
 if(pending){host.className=`datasetIdentity ${identity.kind} loading`;$('datasetIdentityKind').textContent='正在验证';$('datasetIdentityName').textContent=identity.name;$('datasetIdentityDetail').textContent=`${identity.file} · 正在读取媒体索引并检查首帧；完成前保留当前工作区`;host.title=pending.path;return}
 const restoring=workspaceRestoreState!=='ready';host.className=`datasetIdentity ${identity.kind}${restoring?' restoring':''}`;$('datasetIdentityKind').textContent=restoring&&loadedDataPath?'正在恢复':identity.badge;$('datasetIdentityName').textContent=identity.name;
 const detail=loadedDataPath?`${identity.file} · ${topic||'尚未选择相机'}${count?` · ${count} 帧`:''}${imageTopics?` · ${imageTopics} 路图像`:''}`:'加载后会在这里持续显示数据集、批次、相机话题和帧数';
 $('datasetIdentityDetail').textContent=restoring&&loadedDataPath?`正在恢复上次工作帧，请稍候 · ${detail}`:detail;
 host.title=loadedDataPath||'';
}
function markWorkspaceReady(){
 if(workspaceReadyResolved)return;workspaceRestoreState='ready';workspaceReadyResolved=true;updateDatasetIdentity();updatePointPromptCoach();publishWorkbenchFrameState();
 const detail={frame,topic,path:loadedDataPath};resolveWorkspaceReady(detail);document.dispatchEvent(new CustomEvent('labelagent:workspace-ready',{detail}));
}
window.currentInferenceLabels=()=>[...autoLabels];
function candidateCoverageEvidence(label,count,selected,assessment,visible){
 const confidence=Number(assessment?.confidence),confidenceText=Number.isFinite(confidence)?` · 置信度 ${(confidence*100).toFixed(0)}%`:'',reason=String(assessment?.reason||'').trim();
 if(count)return {state:'candidate',title:`${label} 已生成 ${count} 个机器候选`,detail:`依据：当前帧保存了 ${count} 个通过分割门槛的 ${label} 候选。机器候选仍需人工检查，不是人工真值。`};
 if(!selected)return {state:'unselected',title:`${label} 未进入本轮检查`,detail:'依据：该类别不在当前 auto_labels 中。若需要检查，请先在上方勾选并应用类别。'};
 if(visible)return {state:'missing',title:`${label} 可能漏分`,detail:`依据：当前帧 VLM 逐类判断为 ${assessment?.evidence||'present/uncertain'}${confidenceText}${reason?`；理由：${reason}`:''}。SAM 没有留下合格候选，需要人工确认或补标。`};
 if(assessment?.evidence==='absent')return {state:'absent',title:`${label} 被机器判断为未见`,detail:`依据：当前帧 VLM 逐类判断为 absent${confidenceText}${reason?`；理由：${reason}`:''}。这只是机器判断，仍可转人工检查。`};
 return {state:'pending',title:`${label} 尚未形成可追溯结论`,detail:'当前运行记录既没有该类别的有效掩码，也没有 present、uncertain 或 absent 的逐类证据；不能据此判断画面中没有该类别。'};
}
function showCandidateCoverageDetail(label,count,selected,assessment,visible){
 const detail=$('candidateCoverageDetail');if(!detail)return;const evidence=candidateCoverageEvidence(label,count,selected,assessment,visible);detail.replaceChildren();detail.hidden=false;detail.dataset.coverageLabel=label;
 const copy=document.createElement('div'),heading=document.createElement('strong'),text=document.createElement('span');heading.textContent=evidence.title;text.textContent=evidence.detail;copy.append(heading,text);detail.append(copy);
 if(selected&&!count){const action=document.createElement('button');action.type='button';action.dataset.coverageRepair=label;action.textContent=evidence.state==='absent'?'仍要人工检查':evidence.state==='missing'?'补标这个类别':'人工检查并补标';action.onclick=event=>{event.stopPropagation();startCandidateCoverageRepair(label).catch(error=>toast(error.message))};detail.append(action)}
 const close=document.createElement('button');close.type='button';close.className='candidateCoverageClose';close.textContent='收起';close.onclick=()=>{detail.hidden=true};detail.append(close);
}
async function startCandidateCoverageRepair(label){
 if(segmentBusy)throw Error('SAM 正在处理上一项请求，请完成后再补标');
 const sourceFrame=frame,sourceTopic=topic,sourcePath=loadedDataPath;if(!labels.includes(label))throw Error(`当前类别表中没有 ${label}`);
 localStorage.setItem('labelagentWorkspaceMode','manual');await window.setWorkspaceMode?.('manual');
 if(frame!==sourceFrame||topic!==sourceTopic||loadedDataPath!==sourcePath)throw Error('进入补标期间数据、相机或帧已经切换，请在当前帧重新选择类别');
 if(!annotationMode)throw Error('图像标注窗口尚未准备好，请先打开当前相机图像');
 startMissingCandidatePointPrompt(label);toast(`已进入 ${label} 人工补标；机器候选仍独立保留。请在目标内部点一下，点下后才会运行 SAM`);
}
function refreshCandidateSummary(){
 const host=$('candidateSummary');if(!host)return;const keys=Object.keys(candidates),current=candidates[String(frame)]||[],indexedKeys=Object.keys(candidateCountIndex).filter(key=>Number(candidateCountIndex[key])>0),run=agentRuns[String(frame)]||{},discovery=normalizeReferenceDiscovery(run.visible_label_discovery||{}),visible=new Set([...(discovery.visible_labels||[]),...(discovery.uncertain_labels||[])]),assessments=new Map((discovery.assessments||[]).filter(item=>item?.label).map(item=>[item.label,item])),counts=new Map();
 for(const item of current)counts.set(item.label,(counts.get(item.label)||0)+1);
 host.replaceChildren();const title=document.createElement('strong');title.textContent=`自动候选总览 · 已选 ${autoLabels.length}/${labels.length} 类 · ${compactGeometryMode?indexedKeys.length:keys.length} 帧有结果 · 当前帧 ${current.length} 个`;host.append(title);
 for(const label of labels){const count=counts.get(label)||0,selected=autoLabels.includes(label),assessment=assessments.get(label),isVisible=visible.has(label),chip=document.createElement('button');chip.type='button';chip.className='candidateCoverageChip ';chip.dataset.coverageLabel=label;if(count){chip.className+='candidatePresent';chip.textContent=`${label}：已生成 ${count} 个`}else if(!selected){chip.className+='candidateUnselected';chip.textContent=`${label}：未选`}else if(isVisible){chip.className+='candidateMissing';chip.textContent=`${label}：VLM 看见，SAM 漏分`}else if(assessment?.evidence==='absent'){chip.className+='candidateAbsent';chip.textContent=`${label}：本帧未见`}else{chip.className+='candidatePending';chip.textContent=`${label}：尚未评估 / 无有效掩码`}chip.title='点击查看判定依据和下一步';chip.onclick=()=>showCandidateCoverageDetail(label,count,selected,assessment,isVisible);host.append(chip)}
 const note=document.createElement('span');note.className='candidateCoverageNote';note.textContent='点击任一类别可查看当前帧依据；缺失类别可转人工点选补标，但不会自动运行模型或写入人工真值。';host.append(note);const detail=document.createElement('div');detail.id='candidateCoverageDetail';detail.className='candidateCoverageDetail';detail.hidden=true;host.append(detail);
}
function updateCandidateReuseBanner(){
 const banner=$('candidateReuseBanner'),button=$('candidateReuseButton'),summary=$('candidateReuseSummary');if(!banner||!button||!summary)return;
 const key=String(frame),items=candidates[key]||[],hasTruth=shapes[key]!==undefined,hasDraft=Array.isArray(manualDrafts[key])&&manualDrafts[key].length>0;
 const visible=window.workspaceMode==='manual'&&annotationMode&&items.length>0&&!hasTruth&&!hasDraft&&!referencePreparationIntent;
 banner.hidden=!visible;if(!visible){banner.classList.remove('hasWarnings');return}
 const counts=new Map();for(const item of items)counts.set(item.label,(counts.get(item.label)||0)+1);
 const classes=[...counts].map(([label,count])=>`${label} ${count}`).join('、');
 const run=agentRuns[key]||{},queue=Array.isArray(run.review_queue)?run.review_queue:[];
 const missing=[...new Set(queue.filter(item=>item?.reason_code==='vlm_visible_label_missing_candidate'&&item.label).map(item=>item.label))].filter(label=>!counts.has(label));
 const openSetIncomplete=run.visible_label_discovery?.open_set_scan_complete===false||queue.some(item=>item?.reason_code==='vlm_open_set_output_truncated');
 const warnings=[];if(missing.length)warnings.push(`VLM 看见但 SAM 未生成：${missing.join('、')}`);if(openSetIncomplete)warnings.push('新类别扫描未完整返回');
 banner.classList.toggle('hasWarnings',warnings.length>0);
 summary.textContent=`当前帧已有 ${items.length} 个机器候选（${classes}）。无需重新运行 SAM；载入后仍需逐个检查并人工确认。${warnings.length?` 注意：${warnings.join('；')}，已进入复核队列。`:''}`;
 button.disabled=false;button.textContent=`载入 ${items.length} 个候选并开始复核`;
}
async function reuseCurrentFrameCandidates(){
 const key=String(frame),items=candidates[key];if(!Array.isArray(items)||!items.length){toast('当前帧没有可载入的机器候选');updateCandidateReuseBanner();return}
 const button=$('candidateReuseButton');button.disabled=true;button.textContent='正在载入候选…';
 try{await window.prepareReferenceCandidateDraft(frame,JSON.parse(JSON.stringify(items)),JSON.parse(JSON.stringify(agentRuns[key]||{})))}catch(error){toast(error.message);updateCandidateReuseBanner()}
}
function refreshShapeList(){cancelChangedCandidateRefinement(selectedShapes(),frame);if(referencePreparationIntent?.source==='machine_candidate'&&referencePreparationIntent.frame===frame&&referencePreparationIntent.topic===topic)invalidateChangedCandidateAudits(selectedShapes(),frame);let list=$('shapeList');list.innerHTML='';selectedShapes().forEach((s,i)=>{let row=document.createElement('div');row.className='shapeItem'+(i===selectedShapeIndex?' selected':'');row.onclick=()=>{selectedShapeIndex=i;refreshShapeList();draw();document.dispatchEvent(new CustomEvent('labelagent:review-selection-changed',{detail:{frame,index:i}}))};let name=document.createElement('span');name.textContent=`${i+1}. ${s.label} · ${s.points.length} 点`;let remove=document.createElement('button');remove.textContent='×';remove.title='删除标注';remove.onclick=e=>{e.stopPropagation();selectedShapes().splice(i,1);if(selectedShapeIndex>=selectedShapes().length)selectedShapeIndex=-1;refreshShapeList();draw()};row.append(name,remove);list.append(row)});refreshClassColorLegend();refreshCandidateSummary();updateCandidateReuseBanner();updateReferenceFinalizeGuide()}
function draw(){let canvas=$('canvas'),img=$('frameImage');if(!img.naturalWidth)return;imageWidth=img.naturalWidth;imageHeight=img.naturalHeight;// Viewport rectangles already include the stage zoom; canvas CSS uses local coordinates.
const box=img.getBoundingClientRect(),stage=$('stage').getBoundingClientRect();canvas.style.left=((box.left-stage.left)/zoomScale)+'px';canvas.style.top=((box.top-stage.top)/zoomScale)+'px';canvas.style.width=(box.width/zoomScale)+'px';canvas.style.height=(box.height/zoomScale)+'px';canvas.width=imageWidth;canvas.height=imageHeight;let ctx=canvas.getContext('2d');ctx.clearRect(0,0,canvas.width,canvas.height);const machineDraft=machineDraftVisible();if(!referenceReviewOriginalOnly)selectedShapes().forEach((s,i)=>{const selected=i===selectedShapeIndex,visual=labelVisual(s.label);if(referenceReviewSolo&&selectedShapeIndex>=0&&!selected)return;ctx.fillStyle=selected?'rgba(255,190,40,.38)':visual.fill;ctx.strokeStyle=selected?'#ffd447':visual.stroke;ctx.lineWidth=Math.max(selected?4:2.5,imageWidth/(selected?500:650));ctx.setLineDash(machineDraft&&!selected?[Math.max(10,imageWidth/170),Math.max(6,imageWidth/260)]:[]);for(const part of s.parts||[s]){ctx.beginPath();part.points.forEach(([x,y],j)=>j?ctx.lineTo(x,y):ctx.moveTo(x,y));ctx.closePath();for(const hole of part.holes||[]){hole.forEach(([x,y],j)=>j?ctx.lineTo(x,y):ctx.moveTo(x,y));ctx.closePath()}ctx.fill('evenodd');ctx.stroke()}ctx.setLineDash([]);ctx.fillStyle=selected?'#ffd447':visual.stroke;ctx.strokeStyle='#081317';ctx.lineWidth=Math.max(2,imageWidth/900);ctx.font=`700 ${Math.max(18,imageWidth/90)}px Arial`;ctx.strokeText(`${i+1} ${s.label}`,s.points[0][0]+5,s.points[0][1]-8);ctx.fillText(`${i+1} ${s.label}`,s.points[0][0]+5,s.points[0][1]-8)});if(promptPoints.length){ctx.font=`${Math.max(20,imageWidth/70)}px Arial`;promptPoints.forEach(p=>{ctx.beginPath();ctx.arc(p.x,p.y,Math.max(6,imageWidth/350),0,Math.PI*2);ctx.fillStyle=p.label?'#21e2a0':'#ff615d';ctx.fill();ctx.strokeStyle='#102028';ctx.lineWidth=2;ctx.stroke();ctx.fillStyle='#102028';ctx.textAlign='center';ctx.textBaseline='middle';ctx.fillText(p.label?'+':'−',p.x,p.y)})}if(draft.length){ctx.strokeStyle='#ffd053';ctx.fillStyle='#ffd053';ctx.lineWidth=Math.max(2,imageWidth/700);ctx.beginPath();draft.forEach(([x,y],j)=>{j?ctx.lineTo(x,y):ctx.moveTo(x,y);ctx.beginPath();ctx.arc(x,y,Math.max(4,imageWidth/450),0,Math.PI*2);ctx.fill();ctx.beginPath();if(j)ctx.moveTo(...draft[j-1]);ctx.lineTo(x,y);ctx.stroke()})}}
function updatePointPromptCoach(){const coach=$('pointPromptCoach');if(!coach)return;const positive=promptPoints.filter(point=>point.label===1).length,negative=promptPoints.length-positive,hasResult=generatedShapeFrame===frame&&generatedShapeTopic===topic&&generatedShapeIndex>=0,pendingPointChanges=hasResult&&promptPoints.length!==pointPromptGeneratedPointCount,awaitingPoint=annotationMode&&pointMode&&!positive&&!segmentBusy;coach.hidden=!annotationMode||!pointMode;coach.classList.toggle('awaitingPoint',awaitingPoint);coach.classList.toggle('pendingChanges',pendingPointChanges);coach.classList.toggle('needsRefinement',pointPromptRequiresRefinement&&!pendingPointChanges);$('positivePointCount').textContent=`目标点 ${positive}`;$('negativePointCount').textContent=`排除点 ${negative}`;const active=segmentBusy||pendingPointChanges||pointPromptRequiresRefinement?3:!positive?1:negativeMode?2:3;for(const [id,step] of [['pointPromptStep1',1],['pointPromptStep2',2],['pointPromptStep3',3]]){const item=$(id);item.classList.toggle('active',step===active);item.classList.toggle('done',step<active||(step===3&&hasResult&&!pendingPointChanges&&!pointPromptRequiresRefinement))}const elapsed=segmentBusy&&pointPromptStartedAt?Math.max(0,Math.round((Date.now()-pointPromptStartedAt)/1000)):0;const running=elapsed>=30?`SAM 已确认收到请求 · 正在生成 ${elapsed} 秒，请勿重复点击`:`SAM 正在生成 · ${elapsed} 秒`,pendingCount=Math.max(0,promptPoints.length-pointPromptGeneratedPointCount);const status=pointPromptNotice||(segmentBusy?running:pendingPointChanges?`已新增 ${pendingCount} 个提示点，点击“重新生成轮廓”才会应用`:pointPromptRequiresRefinement?'当前轮廓未通过提示点检查，请继续调点并重新生成':hasResult?'轮廓已生成，可继续调点或进入人工确认':negativeMode?'请在非目标背景点红色排除点':positive?'提示点已记录，请点击“生成轮廓”':'尚未提交：请在图片中的目标内部点一下');$('pointPromptStatus').textContent=status;const canvasStatus=$('pointPromptCanvasStatus');if(canvasStatus){canvasStatus.hidden=!annotationMode||!pointMode;canvasStatus.classList.toggle('awaitingPoint',awaitingPoint);canvasStatus.classList.toggle('running',segmentBusy);canvasStatus.classList.toggle('done',hasResult&&!segmentBusy&&!pendingPointChanges&&!pointPromptRequiresRefinement);canvasStatus.classList.toggle('error',pointPromptNotice.startsWith('生成失败'));canvasStatus.classList.toggle('warning',pointPromptRequiresRefinement&&!pendingPointChanges);canvasStatus.querySelector('span').textContent=segmentBusy?running:pendingPointChanges?`新增提示点尚未应用 · 请重新生成`:pointPromptRequiresRefinement?'轮廓未覆盖全部提示点 · 请继续精修':hasResult?'轮廓已生成 · 请检查边界':pointPromptNotice.startsWith('生成失败')?pointPromptNotice:negativeMode?'请点击需要排除的背景':positive?'提示点已记录 · 点击“生成轮廓”':'第 2 步：尚未提交 · 请直接点击目标内部，点击后才会运行 SAM'}const enabled=annotationMode&&pointMode&&positive>0&&!segmentBusy&&!loading&&!playing;for(const id of ['generateMaskButton','pointPromptGenerate']){$(id).disabled=!enabled;$(id).textContent=segmentBusy?`生成中 ${elapsed}秒`:hasResult?'重新生成轮廓':'生成轮廓'}const review=$('pointPromptReview');review.hidden=!hasResult;review.disabled=!hasResult||segmentBusy||pendingPointChanges||pointPromptRequiresRefinement;review.title=pointPromptRequiresRefinement?'SAM 警告尚未解决，请继续调点并重新生成':''}
function resetPrompts(){promptPoints=[];pointPromptNotice='';pointPromptAutoRunOnce=false;pointPromptGeneratedPointCount=0;pointPromptRequiresRefinement=false;generatedShapeFrame=null;generatedShapeTopic='';generatedShapeIndex=-1;updatePointPromptCoach();draw()}
function refreshPointTools(){const pointSegmentationAvailable=catalog[topic]?.type===undefined||catalog[topic]?.type==='image';if(annotationMode&&pointMode&&pointSegmentationAvailable)for(const id of ['pointButton','negativeButton','generateMaskButton'])$(id).hidden=false;for(const [id,selected] of [['pointButton',pointMode],['negativeButton',pointMode&&negativeMode],['polygonButton',!pointMode&&!regionMode&&$('polygonButton').classList.contains('selected')],['addRegionButton',regionMode==='add'],['eraseRegionButton',regionMode==='erase']])$(id).classList.toggle('selected',selected);$('canvas').classList.toggle('drawing',annotationMode&&(pointMode||$('polygonButton').classList.contains('selected')||regionMode));$('finishButton').hidden=pointMode;updatePointPromptCoach()}
async function generatePointMask(){if(workspaceRestoreState!=='ready'){pointPromptNotice='正在恢复上次工作帧，完成后才能运行 SAM';updatePointPromptCoach();toast(pointPromptNotice);return}if(segmentBusy||loading||playing||!annotationMode||!pointMode)return;if(!promptPoints.some(p=>p.label===1)){pointPromptNotice='请先在目标内部点一个绿色目标点';updatePointPromptCoach();toast('先在目标内部点一下');return}const currentFrame=frame,currentTopic=topic,currentPath=loadedDataPath,currentActivationGeneration=loadedFileActivationGeneration,requestLabel=$('labelSelect').value,requestPoints=promptPoints.map(point=>({x:Number(point.x),y:Number(point.y),label:Number(point.label)}));segmentBusy=true;pointPromptStartedAt=Date.now();pointPromptNotice='';pointPromptTimer=setInterval(updatePointPromptCoach,1000);updatePointPromptCoach();try{const result=await api('segment_points',{frame:currentFrame,points:requestPoints,path:currentPath,topic:currentTopic,file_activation_generation:currentActivationGeneration});if(frame!==currentFrame||topic!==currentTopic||loadedDataPath!==currentPath||loadedFileActivationGeneration!==currentActivationGeneration||result.path!==currentPath||result.topic!==currentTopic||Number(result.frame)!==Number(currentFrame)||result.file_activation_generation!==currentActivationGeneration){toast('文件载入代次、数据集、相机或帧已切换，旧 SAM 结果已安全丢弃');return}const current=selectedShapes(),shape={label:requestLabel,points:result.points,holes:result.holes||[],parts:result.parts||[{points:result.points,holes:result.holes||[]}]};if(generatedShapeFrame===currentFrame&&generatedShapeTopic===currentTopic&&generatedShapeIndex>=0&&generatedShapeIndex<current.length)current[generatedShapeIndex]=shape;else{generatedShapeIndex=current.push(shape)-1;generatedShapeFrame=currentFrame;generatedShapeTopic=currentTopic}selectedShapeIndex=generatedShapeIndex;referenceReviewOriginalOnly=false;pointPromptGeneratedPointCount=requestPoints.length;const seconds=Number(result.inference_elapsed_seconds),warnings=Array.isArray(result.warnings)?result.warnings.filter(Boolean):[],score=Number(result.score),lowScore=Number.isFinite(score)&&score<0.5,scoreText=Number.isFinite(score)?` · 置信度 ${(score*100).toFixed(1)}%`:'',timing=Number.isFinite(seconds)?` · ${seconds.toFixed(1)} 秒`:'',labelChanged=$('labelSelect').value!==requestLabel;const qualityNotices=warnings.slice();if(lowScore)qualityNotices.push(`SAM 置信度偏低 ${(score*100).toFixed(1)}%，请把绿色点移到目标主体内部`);if(labelChanged)qualityNotices.push(`生成期间类别已切换，本轮仍按提交时类别“${requestLabel}”保存候选`);pointPromptRequiresRefinement=qualityNotices.length>0;pointPromptNotice=`轮廓已生成并自动选中${timing}${scoreText}${qualityNotices.length?` · 注意：${qualityNotices.join('；')}；修好前不能进入下一步`:'；请检查黄色边界：漏区加绿色点，粘到人或地面加红色排除点，再重新生成'}`;refreshShapeList();draw();toast(pointPromptNotice)}catch(error){pointPromptRequiresRefinement=true;pointPromptNotice=`生成失败：${error.message}`;toast(error.message)}finally{segmentBusy=false;pointPromptStartedAt=0;if(pointPromptTimer){clearInterval(pointPromptTimer);pointPromptTimer=null}updatePointPromptCoach()}}
function frameAtTime(time,metadata=playback){let times=metadata?.times||[],low=0,high=times.length;while(low<high){let mid=(low+high)>>1;if(times[mid]<=time+0.00001)low=mid+1;else high=mid}return Math.max(0,low-1)}
function currentClock(){return playing===true?Math.min(recordDuration,clockBase+(performance.now()-clockAnchor)/1000*rate):clockTime}
function stopPlayback(){clockTime=currentClock();playing=null;playGeneration++;cancelAnimationFrame(tickHandle);$('playbackVideo').pause();for(const view of extraWindows.values())for(const layer of view.layers||[]){layer.video.pause();layer.video.hidden=true;if(layer.image){layer.image.hidden=false;alignStillLayer(layer,clockTime)}}$('playButton').textContent='▶';$('playbackVideo').hidden=true;$('frameImage').hidden=false;$('canvas').hidden=!annotationMode;window.updateCandidateSave?.()}
function formatTime(seconds){const hundredths=Math.floor(Math.max(0,seconds)*100);return `${String(Math.floor(hundredths/6000)).padStart(2,'0')}:${String(Math.floor(hundredths/100)%60).padStart(2,'0')}.${String(hundredths%100).padStart(2,'0')}`}
function updateControls(){$('playbackTime').textContent=`${formatTime(clockTime)} / ${formatTime(recordDuration)}`;$('timeTimeline').max=recordDuration;$('timeTimeline').value=clockTime;if(topic){if(annotationView?.directStill){$('frameNumber').value=frame;return}const offset=playback?.offset??catalog[topic]?.offset??0;frame=frameAtTime(Math.max(0,clockTime-offset));$('frameNumber').value=frame;}}
async function prepareVideo(selected,onProgress){while(true){const response=await fetch('/api/playback?topic='+encodeURIComponent(selected)),data=await response.json();if(!response.ok||data.status==='error')throw Error(data.error||'视频准备失败');if(data.status==='ready')return data;if(onProgress&&!onProgress(data.progress))throw Error('已取消');await new Promise(resolve=>setTimeout(resolve,400))}}
async function prepareTimeline(selected){const response=await fetch('/api/timeline?topic='+encodeURIComponent(selected)),data=await response.json();if(!response.ok)throw Error(data.error||'时间轴读取失败');return data}
async function setMedia(video,metadata){const url=new URL(metadata.url,location.href).href;if(video.src===url&&video.readyState>=1)return;video.src=url;await new Promise((resolve,reject)=>{video.onloadedmetadata=resolve;video.onerror=()=>reject(Error('浏览器无法解码视频'));video.load()})}
function alignMedia(video,metadata,time,run){if(!metadata||video.readyState<1)return;const local=time-metadata.offset;const valid=local>=0&&local<=metadata.duration;const target=Math.max(0,Math.min(local,metadata.duration));if(Math.abs(video.currentTime-target)>(run?0.20:0.00001))video.currentTime=target;video.playbackRate=rate;if(run&&valid&&video.paused)video.play().catch(()=>{});else if(!run||!valid)video.pause()}
function stillFrameForTime(metadata,time){return frameAtTime(Math.max(0,time-(metadata?.offset||0)),metadata)}
async function alignStillLayer(layer,time,wait=false){if(!layer?.metadata||!layer.image)return;const target=stillFrameForTime(layer.metadata,time);layer.video.hidden=true;layer.image.hidden=false;if(layer.stillFrame===target&&layer.image.complete&&layer.image.naturalWidth)return;layer.stillFrame=target;const generation=++layer.stillGeneration,ready=new Promise((resolve,reject)=>{const loaded=()=>{cleanup();resolve()},failed=()=>{cleanup();if(generation===layer.stillGeneration)reject(Error('单帧图像载入失败'));else resolve()},cleanup=()=>{layer.image.removeEventListener('load',loaded);layer.image.removeEventListener('error',failed)};layer.image.addEventListener('load',loaded,{once:true});layer.image.addEventListener('error',failed,{once:true});layer.image.src='/api/frame/'+target+'?topic='+encodeURIComponent(layer.source)});if(wait)await ready;else ready.catch(error=>{if(generation===layer.stillGeneration)layer.view.status.textContent=error.message})}
function updateWindows(force=false){for(const view of extraWindows.values()){if(view.element.hidden)continue;if(view.type==='image'){for(const layer of view.layers||[]){if(!layer.metadata)continue;if(playing===true&&layer.playback?.url){layer.image.hidden=true;layer.video.hidden=false;alignMedia(layer.video,layer.playback,clockTime,true)}else alignStillLayer(layer,clockTime)}}else updateCloud(view,force)}}
function playbackTick(){if(!playing)return;clockTime=currentClock();updateControls();if(playback&&!$('primaryWindow').hidden)alignMedia($('playbackVideo'),playback,clockTime,true);updateWindows();if(clockTime>=recordDuration){pauseAtClock();return}tickHandle=requestAnimationFrame(playbackTick)}
function beginFrameNavigation(){return {generation:++frameNavigationGeneration,topic,path:loadedDataPath}}
function frameNavigationIsCurrent(navigation){return navigation.generation===frameNavigationGeneration&&navigation.topic===topic&&navigation.path===loadedDataPath}
function invalidateFrameNavigation(){frameNavigationGeneration++}
async function pauseAtClock(){const navigation=beginFrameNavigation();stopPlayback();if(topic){const target=frameAtTime(Math.max(0,clockTime-(playback?.offset??catalog[topic]?.offset??0)));await loadStill(target)}if(!frameNavigationIsCurrent(navigation))return false;updateControls();updateWindows(true);return true}
async function startPlayback(){
 if(playing){await pauseAtClock();return}
 if(!recordDuration)return;
 const generation=++playGeneration;playing='preparing';window.updateCandidateSave?.();clockBase=clockTime;clockAnchor=performance.now();$('playButton').textContent='⏸';
 try{
  if(topic&&!$('primaryWindow').hidden){if(!playback?.url)playback=await prepareVideo(topic,progress=>{if(generation!==playGeneration)return false;$('imageInfo').textContent=`正在准备标注窗口：${Math.round(progress*100)}%`;return true});if(generation!==playGeneration)return;await setMedia($('playbackVideo'),playback)}
  for(const view of extraWindows.values()){if(generation!==playGeneration)return;if(view.closed||view.element.hidden||view.type!=='image')continue;for(const layer of view.layers||[]){if(!layer.playback?.url)layer.playback=await prepareVideo(layer.source,progress=>{if(generation!==playGeneration||view.closed)return false;view.status.textContent=`首次播放准备中：${Math.round(progress*100)}%`;return true});if(generation!==playGeneration||view.closed)return;await setMedia(layer.video,layer.playback);layer.image.hidden=true;layer.video.hidden=false}}
  if(generation!==playGeneration)return;
  if(clockTime>=recordDuration)clockTime=0;
  clockBase=clockTime;clockAnchor=performance.now();playing=true;draft=[];
  if(topic&&!$('primaryWindow').hidden){$('canvas').hidden=true;$('frameImage').hidden=true;$('playbackVideo').hidden=false;alignMedia($('playbackVideo'),playback,clockTime,true)}
  updateWindows(true);playbackTick();
 }catch(error){if(generation===playGeneration){stopPlayback();toast(error.message)}}
}
async function loadStill(n,compactPrefetched=false){if(!topic)return;const generation=++frameLoadGeneration,img=$('frameImage');selectedShapeIndex=-1;regionMode='';zoomScale=1;$('stage').style.transform='';frame=Math.max(0,Math.min(Number(n)||0,topics[topic].count-1));if(compactGeometryMode&&!compactPrefetched){shapes={};truthReviewStatus={};candidates={};agentRuns={};candidateReviewProposals=[];candidateReviewBindings={}}loading=true;window.updateCandidateSave?.();$('frameNumber').value=frame;$('frameTotal').textContent='/ '+(topics[topic].count-1);const stateReady=compactGeometryMode&&!compactPrefetched?fetchCompactFrameState(frame,generation):Promise.resolve(),ready=new Promise((resolve,reject)=>{const cleanup=()=>{img.removeEventListener('load',loaded);img.removeEventListener('error',failed)},loaded=()=>{cleanup();resolve()},failed=()=>{cleanup();reject(Error('帧图像载入失败'))};img.addEventListener('load',loaded,{once:true});img.addEventListener('error',failed,{once:true})});img.src='/api/frame/'+frame+'?topic='+encodeURIComponent(topic)+'&v='+Date.now();draft=[];resetPrompts();refreshShapeList();updateReferenceStatus();try{await Promise.all([ready,stateReady])}catch(error){if(generation===frameLoadGeneration)throw error}if(generation!==frameLoadGeneration)return;loading=false;refreshCandidateSummary();refreshShapeList();updateReferenceStatus();publishWorkbenchFrameState();document.dispatchEvent(new CustomEvent('labelagent:frame-changed',{detail:{frame,topic}}));window.refreshInspection?.();rememberViewedFrame();updatePointPromptCoach();updateControls();draw()}
async function showFrame(n){if(!topic)return false;const navigation=beginFrameNavigation();stopPlayback();const target=Math.max(0,Math.min(Number(n)||0,topics[navigation.topic].count-1));try{const timeline=playback||await prepareTimeline(navigation.topic);if(!frameNavigationIsCurrent(navigation))return false;if(!playback)playback=timeline;clockTime=(timeline.offset||0)+timeline.times[target];await loadStill(target);if(!frameNavigationIsCurrent(navigation))return false;updateControls();updateWindows(true);return true}catch(e){if(frameNavigationIsCurrent(navigation))toast(e.message);return false}}
async function showTime(value){if(!topic)return false;const navigation=beginFrameNavigation();stopPlayback();const requestedTime=Number(value)||0;try{const timeline=playback||await prepareTimeline(navigation.topic);if(!frameNavigationIsCurrent(navigation))return false;if(!playback)playback=timeline;clockTime=requestedTime;const target=frameAtTime(clockTime-(timeline.offset||0),timeline);await loadStill(target);if(!frameNavigationIsCurrent(navigation))return false;updateControls();updateWindows(true);return true}catch(error){if(frameNavigationIsCurrent(navigation))toast(error.message);return false}}
function refreshTopics(){setAnnotationMode(false);$('primaryTitle').textContent=topic||'无图像话题';$('primaryWindow').hidden=true;refreshMessageList();updateGrid();frame=window.referenceFrames?.[topic]?.[0]??0;$('frameNumber').value=frame;$('frameTotal').textContent='/ '+(topic?topics[topic].count-1:0);updateControls();updateReferenceStatus();updateDatasetIdentity();}
function refreshMessageList(){refreshSourceChoices()}
async function openImportedImage(){
 if(catalog[topic]?.type!=='image')return;
 await addWindow('image',[topic],true);
 const view=[...extraWindows.values()].find(item=>item.sources[0]===topic&&!item.closed);
 if(view)await activateAnnotation(view);
 await window.setWorkspaceMode?.(window.workspaceMode||'manual',false);
 if(window.workspaceMode==='auto'){
  const firstCandidate=Object.keys(compactGeometryMode?candidateCountIndex:candidates).filter(key=>!compactGeometryMode||Number(candidateCountIndex[key])>0).map(Number).filter(Number.isFinite).sort((a,b)=>a-b)[0];
  if(firstCandidate!==undefined)await showFrame(firstCandidate);
 }
}
function setAnnotationMode(enabled){annotationMode=!!enabled;window.updateCandidateSave?.();$('annotatePrimary').classList.toggle('selected',annotationMode);$('annotatePrimary').setAttribute('aria-pressed',String(annotationMode));$('annotatePrimary').title=annotationMode?'退出标注模式':'进入标注模式';$('annotatePrimary').setAttribute('aria-label',$('annotatePrimary').title);$('primaryWindow').classList.toggle('annotating',annotationMode);$('canvas').hidden=!annotationMode||!!playing;if(!annotationMode){draft=[];resetPrompts();draw()}}
function leaveAnnotation(){stopPlayback();setAnnotationMode(false);$('primaryWindow').hidden=true;$('primaryWindow').classList.remove('maximized');if(annotationView&&!annotationView.closed){annotationView.element.hidden=false;if(annotationView.wasMaximized)annotationView.element.classList.add('maximized');for(const layer of annotationView.layers)alignStillLayer(layer,clockTime)}annotationView=null;updateGrid()}
async function activateAnnotation(view){if(view.closed)return;if(annotationView===view){leaveAnnotation();return}const navigation=beginFrameNavigation();if(annotationView)leaveAnnotation();const time=currentClock();stopPlayback();try{const selected=view.sources[0],metadata=view.layers?.[0]?.metadata,targetFrame=metadata?stillFrameForTime(metadata,time):0;window.cancelReferencePrefetch?.();const data=await api('topic',{topic:selected,frame:targetFrame,compact:true,path:loadedDataPath,file_activation_generation:loadedFileActivationGeneration});if(view.closed||navigation.generation!==frameNavigationGeneration)return;if(topic!==selected){manualDrafts={};autoDrafts={}}topic=selected;loadedDataPath=data.path||loadedDataPath;loadedFileActivationGeneration=data.file_activation_generation??loadedFileActivationGeneration;navigation.topic=topic;navigation.path=loadedDataPath;compactGeometryMode=data.compact_geometry===true;shapes={};truthReviewStatus={};candidates={};agentRuns={};candidateReviewProposals=[];candidateReviewBindings={};applyCompactFrameState(data,targetFrame,topic,loadedDataPath);window.referenceFrames=data.reference_frames||{};applyInferenceLabelState(data);currentDescription=data.description||null;draft=[];const timeline=metadata||await prepareTimeline(topic);if(!frameNavigationIsCurrent(navigation))return;playback=timeline;annotationView=view;view.wasMaximized=view.element.classList.contains('maximized');view.element.classList.remove('maximized');view.element.hidden=true;for(const layer of view.layers)layer.video.pause();view.element.before($('primaryWindow'));$('primaryWindow').hidden=false;$('primaryWindow').classList.toggle('maximized',view.wasMaximized);$('primaryTitle').textContent=topic;updateDatasetIdentity();setAnnotationMode(true);updateGrid();if(annotationView!==view||!frameNavigationIsCurrent(navigation))return;clockTime=time;await loadStill(targetFrame,true);if(!frameNavigationIsCurrent(navigation))return;updateControls();updateWindows(true);if(data.description_cache_hit)toast('已恢复该相机已验证的场景描述，无需重新生成')}catch(error){if(navigation.generation===frameNavigationGeneration){if(annotationView===view)leaveAnnotation();toast(error.message)}}}
function updateGrid(){const count=[...extraWindows.values()].filter(view=>!view.element.hidden).length+(!$('primaryWindow').hidden?1:0);const columns=count<=1?1:count<=4?2:3;$('windowGrid').style.setProperty('--columns',columns);$('windowGrid').style.setProperty('--rows',Math.max(1,Math.ceil(count/columns)));$('emptyWindows').hidden=count>0;if(!count)$('windowGrid').classList.remove('hasMaximized')}
function maximizeWindow(element){const expanded=element.classList.contains('maximized');document.querySelectorAll('.sensorWindow.maximized').forEach(x=>x.classList.remove('maximized'));element.classList.toggle('maximized',!expanded);$('windowGrid').classList.toggle('hasMaximized',!expanded);requestAnimationFrame(draw)}
function resetWindows(){annotationView=null;$('primaryWindow').hidden=true;for(const view of extraWindows.values()){view.closed=true;view.abort?.abort();view.renderer?.destroy();for(const layer of view.layers||[])layer.video.pause();view.element.remove()}extraWindows.clear();document.querySelectorAll('.maximized').forEach(x=>x.classList.remove('maximized'));$('windowGrid').classList.remove('hasMaximized');updateGrid()}
async function updateCloud(view,force=false){if(view.closed)return;view.desiredTime=clockTime;if(view.busy)return;if(!force&&(!playing||performance.now()-view.lastRequest<100))return;const time=clockTime;view.busy=true;view.lastRequest=performance.now();view.abort=new AbortController();try{const results=await Promise.all(view.sources.map(async source=>{const response=await fetch('/api/cloud?topic='+encodeURIComponent(source)+'&time='+time.toFixed(6),{signal:view.abort.signal});if(!response.ok){const data=await response.json();throw Error(data.error||'点云读取失败')}return {points:new Float32Array(await response.arrayBuffer()),count:Number(response.headers.get('X-Point-Count'))}}));if(view.closed)return;if(Math.abs(clockTime-time)<0.5||!playing&&Math.abs(clockTime-time)<0.01){const total=results.reduce((n,r)=>n+r.points.length,0),merged=new Float32Array(total);let offset=0;for(const result of results){merged.set(result.points,offset);offset+=result.points.length}view.renderer.setPoints(merged);view.status.textContent=`${time.toFixed(2)} 秒 · ${merged.length/4} / ${results.reduce((n,r)=>n+r.count,0)} 点 · ${view.sources.length} 源叠加`}}catch(e){if(!view.closed&&e.name!=='AbortError')view.status.textContent=e.message}finally{view.busy=false;if(!view.closed&&!playing&&Math.abs(view.desiredTime-time)>0.001)updateCloud(view,true)}}
function refreshSourceChoices(){const type=$('windowType').value,list=$('windowSources');list.replaceChildren();for(const [name,info] of Object.entries(catalog)){if(info.type!==type)continue;const label=document.createElement('label');label.className='sourceChoice';const input=document.createElement('input');input.type='checkbox';input.value=name;const span=document.createElement('span');span.textContent=name;label.append(input,span);list.append(label)}if(!list.children.length){const empty=document.createElement('p');empty.textContent='当前文件没有此类型的消息';list.append(empty)}$('windowHint').textContent=type==='image'?'多个图像源按顺序透明叠加，并按采集时间同步。':'多个点云源合并到同一个三维画面；请确保坐标系一致。'}
async function addWindow(type,sources,deferInitialLoad=false){if(!sources.length||sources.some(name=>catalog[name]?.type!==type))throw Error('请选择有效的消息源');const id=++windowSerial,element=document.createElement('article');element.className='sensorWindow';const header=document.createElement('div');header.className='windowHeader';const title=document.createElement('span');title.textContent=(type==='image'?'图像 · ':'点云 · ')+sources.join(' + ');title.title=sources.join(' + ');const expand=document.createElement('button');expand.textContent='⛶';expand.title='放大 / 还原';const close=document.createElement('button');close.textContent='×';close.title='关闭窗口';header.append(title);if(type==='image'){const annotate=document.createElement('button');annotate.textContent='✎';annotate.title='进入标注模式';annotate.onclick=()=>activateAnnotation(view);header.append(annotate)}header.append(expand,close);const body=document.createElement('div');body.className='sensorBody';const status=document.createElement('div');status.className='sensorStatus';element.append(header,body,status);$('windowGrid').append(element);const view={id,sources,type,element,status,closed:false,lastRequest:0,busy:false,layers:[]};extraWindows.set(id,view);expand.onclick=()=>maximizeWindow(element);element.ondblclick=e=>{e.preventDefault();maximizeWindow(element)};close.onclick=()=>{if(annotationView===view)leaveAnnotation();view.closed=true;view.abort?.abort();view.renderer?.destroy();for(const layer of view.layers)layer.video.pause();element.remove();extraWindows.delete(id);if(element.classList.contains('maximized'))$('windowGrid').classList.remove('hasMaximized');updateGrid()};updateGrid();try{if(type==='image'){for(const [index,source] of sources.entries()){if(view.closed)return;const image=document.createElement('img'),video=document.createElement('video');image.className='imageLayer'+(index?' overlay':'');image.alt=source;video.className='imageLayer'+(index?' overlay':'');video.muted=true;video.playsInline=true;video.preload='metadata';video.hidden=true;body.append(image,video);const layer={source,image,video,metadata:null,playback:null,stillFrame:-1,stillGeneration:0,view};view.layers.push(layer);if(deferInitialLoad){status.textContent=`正在直接打开标注窗口：${source}`;continue}status.textContent=`正在读取 ${index+1}/${sources.length}：${source}`;layer.metadata=await prepareTimeline(source);if(view.closed)return;await alignStillLayer(layer,clockTime,true)}if(!deferInitialLoad)status.textContent=`${sources.length} 个图像源 · 单帧已就绪 · 点击播放时才准备视频`}else{const canvas=document.createElement('canvas');body.append(canvas);view.renderer=new PointCloudView(canvas);status.textContent='拖动旋转 · 滚轮缩放';updateCloud(view,true)}}catch(e){if(!view.closed)status.textContent=e.message}}
$('addWindowButton').onclick=()=>{refreshSourceChoices();$('addWindowDialog').showModal()};$('closeWindowDialog').onclick=()=>$('addWindowDialog').close();$('cancelWindowDialog').onclick=()=>$('addWindowDialog').close();$('windowType').onchange=refreshSourceChoices;$('addWindowForm').onsubmit=e=>{e.preventDefault();const sources=[...$('windowSources').querySelectorAll('input:checked')].map(input=>input.value);if(!sources.length){toast('请至少选择一个消息源');return}const type=$('windowType').value;$('addWindowDialog').close();addWindow(type,sources).catch(error=>toast(error.message))};
$('timeTimeline').onchange=e=>showTime(e.target.value);
$('annotatePrimary').onclick=leaveAnnotation;$('maximizePrimary').onclick=()=>maximizeWindow($('primaryWindow'));$('primaryWindow').ondblclick=e=>{if(e.target===$('canvas')&&$('polygonButton').classList.contains('selected'))return;e.preventDefault();maximizeWindow($('primaryWindow'))};$('closePrimary').onclick=()=>{const view=annotationView;leaveAnnotation();view?.element.querySelector('.windowHeader button:last-child')?.click()};document.addEventListener('keydown',e=>{if(e.key==='Escape'){if($('addWindowDialog').open)return;document.querySelectorAll('.maximized').forEach(x=>x.classList.remove('maximized'));$('windowGrid').classList.remove('hasMaximized');return}const target=e.target;if(['INPUT','TEXTAREA','SELECT'].includes(target.tagName))return;if(e.key==='ArrowLeft'||e.key==='ArrowRight'){e.preventDefault();showFrame(frame+(e.key==='ArrowLeft'?-1:1))}});
$('playbackRate').onchange=()=>{const time=currentClock();rate=Number($('playbackRate').value);clockTime=clockBase=time;clockAnchor=performance.now();$('playbackVideo').playbackRate=rate;for(const view of extraWindows.values())for(const layer of view.layers||[])layer.video.playbackRate=rate;updateControls()};
$('polygonButton').classList.remove('selected');$('polygonButton').onclick=()=>{if(segmentBusy){toast('SAM 正在生成，请完成后再切换绘制方式');return}pointMode=false;$('polygonButton').classList.toggle('selected');resetPrompts();refreshPointTools()};$('pointButton').onclick=()=>{if(segmentBusy){toast('SAM 正在生成，请勿重复提交');return}const hasResult=generatedShapeFrame===frame&&generatedShapeTopic===topic&&generatedShapeIndex>=0;pointMode=true;negativeMode=false;pointPromptNotice=hasResult?'继续精修：在漏区加绿色点后，点击“重新生成轮廓”':'第 2 步：SAM 尚未启动，请现在点击图片中的目标内部';pointPromptAutoRunOnce=!hasResult;draft=[];$('polygonButton').classList.remove('selected');refreshPointTools();draw();focusPointPromptCanvas();toast(hasResult?'已进入 SAM 精修模式；加点后请重新生成':'尚未运行 SAM：现在请点击图片中的目标，出现绿色点和计时才算开始')};$('negativeButton').onclick=()=>{if(segmentBusy){toast('SAM 正在生成，请完成后再调整提示点');return}pointMode=true;pointPromptNotice='';pointPromptAutoRunOnce=false;draft=[];negativeMode=!negativeMode;$('polygonButton').classList.remove('selected');refreshPointTools();draw()};$('generateMaskButton').onclick=generatePointMask;$('pointPromptGenerate').onclick=generatePointMask;$('pointPromptReview').onclick=()=>{const guide=$('referenceFinalizeGuide');if(guide&&!guide.hidden){const review=$('referenceReviewApproveNext'),target=review&&!review.disabled?review:$('referenceFinalizePrimary');$('referenceReviewNavigation')?.scrollIntoView({behavior:'smooth',block:'center'});target.focus({preventScroll:true});toast(target===review?'已定位到黄色轮廓；检查边界后点“检查通过并下一个”':'请先完成当前轮廓检查和漏标处理')}else toast('当前没有可确认的参考帧草稿')};$('pointPromptFullPipeline').onclick=async()=>{const button=$('pointPromptFullPipeline'),label=button.textContent;if(button.disabled)return;button.disabled=true;button.textContent='正在进入整帧全流程…';try{if(typeof window.startReferenceAutoDraft!=='function')throw Error('整帧 SAM + VLM 助手尚未加载，请刷新页面后重试');pointMode=false;negativeMode=false;refreshPointTools();draw();await window.startReferenceAutoDraft()}catch(error){toast(error.message)}finally{button.disabled=false;button.textContent=label}};refreshPointTools();function beginRegionMode(mode){const items=selectedShapes();if(regionMode===mode){regionMode='';draft=[];refreshPointTools();draw();return}if(!items.length){toast('当前帧还没有可编辑的标注；如需 SAM，请点击“SAM 点选”')}else if(selectedShapeIndex<0){if(items.length===1)selectedShapeIndex=0;else{toast('请先在标注列表中选择要编辑的标注');return}}if(!items.length)return;regionMode=mode;pointMode=false;draft=[];$('polygonButton').classList.remove('selected');refreshShapeList();refreshPointTools();draw()}$('addRegionButton').onclick=()=>beginRegionMode('add');$('eraseRegionButton').onclick=()=>beginRegionMode('erase');
async function finish(){
 if(pointMode){generatePointMask();return}
 if(!annotationMode||loading||playing)return;
 if(draft.length<3){toast('至少需要 3 个点');return}
 if(regionMode){
  if(selectedShapeIndex<0){toast('请先选择要编辑的标注');return}
  const requestGeneration=++regionEditGeneration,requestFrame=frame,requestTopic=topic,requestPath=loadedDataPath,requestActivationGeneration=loadedFileActivationGeneration,requestMode=window.workspaceMode,requestIndex=selectedShapeIndex,items=selectedShapes();
  if(!items[requestIndex]){toast('当前轮廓不存在，请重新选择后再编辑');return}
  const requestShape=JSON.parse(JSON.stringify(items[requestIndex])),requestShapeSnapshot=candidateAuditFullSnapshot(requestShape),requestOperation=regionMode,requestPolygon=draft.map(point=>point.map(value=>Math.round(value)));
  try{
   const result=await api('edit_region',{frame:requestFrame,operation:requestOperation,polygon:requestPolygon,shape:requestShape,path:requestPath,topic:requestTopic,file_activation_generation:requestActivationGeneration});
   if(requestGeneration!==regionEditGeneration||frame!==requestFrame||topic!==requestTopic||loadedDataPath!==requestPath||loadedFileActivationGeneration!==requestActivationGeneration||window.workspaceMode!==requestMode||result.path!==requestPath||result.topic!==requestTopic||Number(result.frame)!==Number(requestFrame)||result.file_activation_generation!==requestActivationGeneration){toast('文件载入代次、帧、相机、数据集或模式已切换，旧区域编辑结果已安全丢弃');return}
   const current=selectedShapes();
   if(requestIndex>=current.length||candidateAuditFullSnapshot(current[requestIndex])!==requestShapeSnapshot){toast('原轮廓已变化，旧区域编辑结果已安全丢弃');return}
   current[requestIndex]=result;draft=[];regionMode='';refreshPointTools();refreshShapeList();draw();toast('区域已更新');
  }catch(e){toast(e.message)}
  return;
 }
 selectedShapes().push({label:$('labelSelect').value,points:draft.map(p=>p.map(Math.round))});draft=[];refreshShapeList();draw();
}
$('frameImage').onload=()=>{loading=false;window.updateCandidateSave?.();updatePointPromptCoach();if(playing)return;updateControls();draw()};$('frameImage').onerror=()=>{loading=false;window.updateCandidateSave?.();updatePointPromptCoach();$('imageInfo').textContent=`帧 ${frame} 载入失败`;toast('帧载入失败')};window.addEventListener('resize',draw);$('stage').addEventListener('wheel',e=>{if(!annotationMode)return;e.preventDefault();zoomScale=Math.max(1,Math.min(5,zoomScale*(e.deltaY<0?1.15:0.87)));$('stage').style.transform=`scale(${zoomScale})`;$('stage').style.transformOrigin='center center';draw()},{passive:false});new ResizeObserver(draw).observe($('stage'));$('canvas').onclick=e=>{if(!annotationMode||!pointMode&&!$('polygonButton').classList.contains('selected')&&!regionMode)return;if(pointMode&&segmentBusy){toast('SAM 正在生成，本次点位已锁定；完成后可继续加点');return}let box=e.currentTarget.getBoundingClientRect();if(loading){pointPromptNotice='图片正在加载，请稍等后再点';updatePointPromptCoach();toast(pointPromptNotice);return}if(!imageWidth||playing)return;const x=(e.clientX-box.left)*imageWidth/box.width,y=(e.clientY-box.top)*imageHeight/box.height;if(pointMode){promptPoints.push({x:Math.round(x),y:Math.round(y),label:negativeMode?0:1});pointPromptNotice='';updatePointPromptCoach();if(pointPromptAutoRunOnce&&!negativeMode){pointPromptAutoRunOnce=false;queueMicrotask(generatePointMask)}}else draft.push([x,y]);draw()};$('canvas').ondblclick=e=>{e.preventDefault()};$('undoButton').onclick=()=>{if(pointMode&&segmentBusy){toast('SAM 正在生成，本次点位已锁定；完成后再撤销');return}if(pointMode){promptPoints.pop();pointPromptNotice='';pointPromptAutoRunOnce=false;updatePointPromptCoach()}else draft.pop();draw()};$('finishButton').onclick=finish;$('clearButton').onclick=()=>{if(segmentBusy){toast('SAM 正在生成，完成后再清空本帧');return}selectedShapes().splice(0);draft=[];resetPrompts();refreshShapeList();draw()};
function resetReferenceReviewView(){referenceReviewSolo=false;referenceReviewFocused=false;referenceReviewOriginalOnly=false;zoomScale=1;$('stage').style.transform='';$('stage').style.transformOrigin='center center'}
function referenceReviewKeys(items){const seen=new Map();return items.map(shape=>{const geometry=JSON.stringify({label:shape.label,points:shape.points||[],holes:shape.holes||[],parts:shape.parts||[]}),occurrence=(seen.get(geometry)||0)+1;seen.set(geometry,occurrence);return `${geometry}#${occurrence}`})}
async function applyCandidateReviewProposals(target,items,runMetadata={}){
 if(!Array.isArray(items)||!items.length)return 0;const binding=candidateReviewBindings?.[String(target)],actualArtifact=binding?.candidate_artifact_sha256,declaredArtifact=runMetadata?.candidate_artifact_sha256;let applied=0;
 for(const proposal of candidateReviewProposals||[]){
  if(proposal?.file!==loadedDataPath||proposal?.topic!==topic||Number(proposal?.frame)!==Number(target)||proposal?.candidate_artifact_sha256!==actualArtifact||proposal?.route!=='human_review_required')continue;
  if(proposal?.declared_candidate_artifact_sha256&&proposal.declared_candidate_artifact_sha256!==declaredArtifact)continue;
  const shape=items.find(item=>Number(item?.id)===Number(proposal.candidate_id));if(!shape||shape.label!==proposal.from_label)continue;
  const itemBindings=(binding?.candidates||[]).filter(item=>Number(item?.candidate_id)===Number(proposal.candidate_id));if(itemBindings.length!==1||itemBindings[0].candidate_sha256!==proposal.candidate_sha256)continue;
  shape.review_status='assistant_reviewed_model_conflict';shape.provenance={...(shape.provenance||{}),vlm_conflict:true,vlm_conflict_project_label:proposal.from_label,vlm_observed_label:proposal.proposed_label,vlm_confidence:proposal.confidence,vlm_reason:proposal.reason,assistant_review_proposal_id:proposal.proposal_id,assistant_review_evidence_sha256:proposal.evidence_sha256,assistant_review_source:proposal.source,route:'human_review_required',identity_assurance:'assistant_reviewed_not_human_truth',candidate_mutated:false,human_truth:false,approved_for_training:false};applied++;
 }
 if(actualArtifact&&actualArtifact===declaredArtifact){
  for(const audit of runMetadata?.candidate_class_audits||[]){
   if(audit?.candidate_artifact_sha256!==actualArtifact||audit?.route!=='human_review_required')continue;
   const decision=audit.decision||{},observed=decision.observed_label,matches=items.filter(item=>Number(item?.id)===Number(audit.candidate_id));
   if(matches.length!==1||matches[0].label!==audit.bound_current_label||!observed||observed===audit.bound_current_label||!labels.includes(observed))continue;
   const shape=matches[0];shape.review_status='assistant_reviewed_model_conflict';shape.provenance={...(shape.provenance||{}),vlm_conflict:true,vlm_conflict_project_label:audit.bound_current_label,vlm_observed_label:observed,vlm_confidence:decision.confidence,vlm_reason:decision.reason,assistant_review_evidence_sha256:audit.audit_sha256,assistant_review_source:'candidate_class_blind_audit',route:'human_review_required',identity_assurance:'machine_review_not_human_truth',candidate_mutated:false,human_truth:false,approved_for_training:false};applied++;
  }
 }
 return applied;
}
function candidateModelConflict(shape){if(shape?.provenance?.candidate_audit_stale===true)return null;const provenance=shape?.provenance||{},explicit=provenance.vlm_conflict===true||String(shape?.review_status||'').includes('model_conflict');if(!shape||!explicit)return null;const confidence=Number(provenance.vlm_confidence);return {projectLabel:String(provenance.vlm_conflict_project_label||shape.label||'未命名'),vlmLabel:String(provenance.vlm_observed_label||'未给出类别'),confidence:Number.isFinite(confidence)?confidence:null,resolved:provenance.vlm_conflict_resolved===true,resolution:provenance.vlm_conflict_resolution||null}}
function unresolvedCandidateModelConflict(shape){const conflict=candidateModelConflict(shape);return conflict&&!conflict.resolved?conflict:null}
function resolveSelectedModelConflict(choice){const items=selectedShapes(),shape=items[selectedShapeIndex],conflict=unresolvedCandidateModelConflict(shape);if(!shape||!conflict){toast('当前轮廓没有待处理的模型类别冲突');return false}if(!['keep_project_label','use_vlm_label'].includes(choice)){toast('未知的冲突处理方式，已停止');return false}if(choice==='use_vlm_label'&&!labels.includes(conflict.vlmLabel)){toast(`VLM 建议类别 ${conflict.vlmLabel} 不在当前标签表中，必须进入新类别人工门禁`);return false}const keys=referenceReviewKeys(items),oldKey=keys[selectedShapeIndex],priorLabel=shape.label;shape.provenance={...(shape.provenance||{}),vlm_conflict:true,vlm_conflict_project_label:conflict.projectLabel,vlm_conflict_resolved:true,vlm_conflict_resolution:choice,resolved_label:choice==='use_vlm_label'?conflict.vlmLabel:priorLabel,conflict_resolution_source:'explicit_operator_ui_choice',identity_assurance:'operator_choice_pending_authenticated_human_review',conflict_resolved_at:new Date().toISOString()};if(choice==='use_vlm_label')shape.label=conflict.vlmLabel;referenceReviewedShapeKeys.delete(oldKey);persistReferenceReviewProgress(items);refreshShapeList();draw();updateReferenceFinalizeGuide();toast(choice==='use_vlm_label'?`已将当前机器草稿类别改为 ${conflict.vlmLabel}；仍需检查轮廓并通过人工真值门禁`:`已明确保留 ${priorLabel}；仍需检查轮廓并通过人工真值门禁`);return true}
window.resolveSelectedModelConflict=resolveSelectedModelConflict;
const referenceReviewStorageKey='labelagentReferenceReviewProgressV106';
const candidateAuditSnapshots=new Map();
const candidateRefinementSnapshots=new Map();
function candidateAuditScope(target=frame,sourceTopic=topic,sourcePath=loadedDataPath){return JSON.stringify([sourcePath,sourceTopic,Number(target)])}
function candidateAuditSnapshot(items){return JSON.stringify(items.map(shape=>({id:shape.id,label:shape.label,score:shape.score,points:shape.points||[],holes:shape.holes||[],parts:shape.parts||[]})))}
function candidateAuditFullSnapshot(value){if(Array.isArray(value))return '['+value.map(candidateAuditFullSnapshot).join(',')+']';if(value&&typeof value==='object')return '{'+Object.keys(value).sort().map(key=>JSON.stringify(key)+':'+candidateAuditFullSnapshot(value[key])).join(',')+'}';return JSON.stringify(value)}
function candidateClassAuditRequired(shape){
 if(['ego','person','other_robot'].includes(shape?.label))return true;
 if(String(shape?.label||'').toLowerCase()!=='pallet')return false;
 try{
  const parts=Array.isArray(shape.parts)&&shape.parts.length?shape.parts:[shape],rings=parts.map(part=>({points:part.points,holes:Array.isArray(part.holes)?part.holes:[]}));
  const validRing=ring=>Array.isArray(ring)&&ring.length>=3&&ring.every(point=>Array.isArray(point)&&point.length===2&&point.every(Number.isFinite));
  if(rings.some(part=>!validRing(part.points)||part.holes.some(hole=>!validRing(hole))))return true;
  const points=rings.flatMap(part=>part.points);if(!points.length)return true;
   const xs=points.map(point=>point[0]),ys=points.map(point=>point[1]),width=Math.max(...xs)-Math.min(...xs)+1,height=Math.max(...ys)-Math.min(...ys)+1,bboxArea=width*height;
  const area=ring=>Math.abs(ring.reduce((sum,point,index)=>{const next=ring[(index+1)%ring.length];return sum+point[0]*next[1]-next[0]*point[1]},0)/2);
  const foreground=rings.reduce((sum,part)=>sum+area(part.points)-part.holes.reduce((holes,hole)=>holes+area(hole),0),0);
   const longSide=Math.max(width,height),shortSide=Math.min(width,height);if(longSide>=120&&shortSide>=4&&shortSide/Math.max(1,longSide)<=.08&&foreground>=500)return true;
   if(width<120||height<80||bboxArea<5000)return false;
  return foreground/Math.max(1,bboxArea)>=.82&&width/Math.max(1,height)<=2.10;
 }catch{return true}
}
function newCandidateRefinementId(){
 if(typeof crypto.randomUUID==='function')return crypto.randomUUID().replaceAll('-','');
 const bytes=crypto.getRandomValues(new Uint8Array(16));bytes[6]=(bytes[6]&15)|64;bytes[8]=(bytes[8]&63)|128;return Array.from(bytes,value=>value.toString(16).padStart(2,'0')).join('');
}
function cancelChangedCandidateRefinement(items,target=frame){
 const intent=referencePreparationIntent,key=String(target),draft=manualDrafts[key];
 if(intent?.source!=='machine_candidate'||intent.frame!==Number(target)||intent.topic!==topic||!Array.isArray(draft))return false;
 const scope=candidateAuditScope(target),entry=candidateRefinementSnapshots.get(scope),snapshot=candidateAuditFullSnapshot(draft);
 if(!entry?.pending||entry.editSnapshot===snapshot)return false;
 entry.pending=false;entry.stale=true;entry.generation++;
 fetch('/api/reference_candidates/refine/stop',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({frame:Number(target),topic:entry.sourceTopic,refinement_id:entry.refinementId}),keepalive:true}).catch(()=>{});
 referenceFinalizeNotice='人工草稿已变化，已取消过时的后台 Agent 精修；当前修改不会被覆盖';return true;
}
async function auditReferenceCandidateSnapshot(target,items){
 const sourceTopic=topic,sourcePath=loadedDataPath,key=String(target),scope=candidateAuditScope(target),snapshot=candidateAuditFullSnapshot(items);
 if(candidateAuditSnapshots.get(scope)?.pending)throw Error('当前候选正在轻量复审，请等待本轮完成');
 const entry={snapshot:candidateAuditSnapshot(items),generation:(candidateAuditSnapshots.get(scope)?.generation||0)+1,pending:true,stale:false};candidateAuditSnapshots.set(scope,entry);const generation=entry.generation;
 const assertCurrent=()=>{if(topic!==sourceTopic||loadedDataPath!==sourcePath||candidateAuditSnapshots.get(scope)!==entry||entry.generation!==generation||entry.stale||candidateAuditFullSnapshot(candidates[key])!==snapshot)throw Error('候选或数据作用域已变化，复审结果已过期')};
 try{
   const fresh=await readCompactFrameStateSnapshot(target,sourceTopic,sourcePath);assertCurrent();
  const binding=fresh.candidate_review_bindings?.[key],artifact=binding?.candidate_artifact_sha256;
   if(candidateAuditFullSnapshot(fresh.candidates?.[key])!==snapshot||!artifact)throw Error('候选服务器绑定不一致，已拒绝复审');
  const imageResponse=await fetch(`/api/frame/${target}?topic=${encodeURIComponent(sourceTopic)}`,{cache:'no-store'});if(!imageResponse.ok)throw Error('无法读取复审原图');
  const imageHash=Array.from(new Uint8Array(await crypto.subtle.digest('SHA-256',await imageResponse.arrayBuffer())),value=>value.toString(16).padStart(2,'0')).join('');assertCurrent();
  const version=fresh.candidate_class_audit_version,validBinding=value=>value?.candidate_artifact_sha256===artifact&&Array.isArray(value?.candidates)&&value.candidates.length===(binding.candidates||[]).length&&(binding.candidates||[]).every(expected=>{const matches=value.candidates.filter(item=>item.candidate_id===expected.candidate_id);return matches.length===1&&(matches[0].candidate_fingerprint_sha256||matches[0].candidate_fingerprint)===(expected.candidate_fingerprint_sha256||expected.candidate_fingerprint)&&matches[0].candidate_sha256===expected.candidate_sha256});
  const highRisk=items.filter(candidateClassAuditRequired),validVersion=value=>value?.schema_version==='labelagent.candidate_class_audit_version.v1'&&['config_sha256','protocol_sha256','labels_sha256'].every(name=>typeof value[name]==='string'&&/^[a-f0-9]{64}$/.test(value[name])),validateAudits=run=>{
   const audits=run?.candidate_class_audits;if(run?.candidate_class_audit_status!=='complete'||run.candidate_artifact_sha256!==artifact||run.input_sha256!==imageHash||!validVersion(version)||!validVersion(run.candidate_class_audit_version)||candidateAuditFullSnapshot(run.candidate_class_audit_version)!==candidateAuditFullSnapshot(version)||!Array.isArray(audits))return false;
   const ids=highRisk.map(shape=>shape.id);if(ids.some(id=>!Number.isInteger(id))||new Set(ids).size!==ids.length||audits.length!==highRisk.length)return false;
   return highRisk.every(shape=>{const rows=audits.filter(audit=>audit?.candidate_id===shape.id);if(rows.length!==1)return false;const audit=rows[0],matches=(binding.candidates||[]).filter(item=>item.candidate_id===shape.id),classification=audit.decision?.classification;return matches.length===1&&audit.bound_current_label===shape.label&&audit.candidate_sha256===matches[0].candidate_fingerprint_sha256&&audit.candidate_artifact_sha256===artifact&&audit.source_image_sha256===imageHash&&audit.schema_version==='labelagent.blind_candidate_class_audit.v1'&&candidateAuditFullSnapshot(audit.audit_version)===candidateAuditFullSnapshot(version)&&typeof audit.prompt_sha256==='string'&&/^[a-f0-9]{64}$/.test(audit.prompt_sha256)&&typeof audit.audit_sha256==='string'&&/^[a-f0-9]{64}$/.test(audit.audit_sha256)&&audit.human_truth===false&&!audit.error&&audit.decision&&['known_label','uncertain','conflict'].includes(classification)&&['candidate_preserved','human_review_required'].includes(audit.route)});
  };
  const fromCache=()=>{const run=fresh.agent_runs?.[key];return validateAudits(run)?run:null};
  let resultAgent=fromCache();
  if(!resultAgent){
   const result=await api('reference_candidates/audit',{frame:Number(target),topic:sourceTopic,expected_candidate_artifact_sha256:artifact,expected_frame_sha256:imageHash});assertCurrent();
   if(result.frame!==Number(target)||result.topic!==sourceTopic||result.candidate_only!==true||result.human_truth!==false||result.input_sha256!==imageHash||result.candidate_artifact_sha256!==artifact||candidateAuditFullSnapshot(result.candidates)!==snapshot||!validBinding(result.binding)||!validateAudits(result.agent))throw Error('复审响应绑定不一致，已拒绝采用');
   resultAgent=result.agent;
  }
   const readback=await readCompactFrameStateSnapshot(target,sourceTopic,sourcePath);assertCurrent();
  const readbackRun=readback.agent_runs?.[key];
   if(candidateAuditFullSnapshot(readback.candidates?.[key])!==snapshot||!validBinding(readback.candidate_review_bindings?.[key])||candidateAuditFullSnapshot(readback.candidate_class_audit_version)!==candidateAuditFullSnapshot(version)||!validateAudits(readbackRun)||candidateAuditFullSnapshot(readbackRun)!==candidateAuditFullSnapshot(resultAgent))throw Error('复审服务器读回不一致，已拒绝采用');
  candidateReviewBindings=readback.candidate_review_bindings;entry.imageHash=imageHash;entry.artifact=artifact;entry.bindingSnapshot=candidateAuditFullSnapshot(readback.candidate_review_bindings?.[key]);entry.stale=false;return readbackRun;
 }catch(error){entry.stale=true;throw error}finally{entry.pending=false}
}
async function refineReferenceCandidateSnapshot(target,items){
 const sourceTopic=topic,sourcePath=loadedDataPath,key=String(target),scope=candidateAuditScope(target),snapshot=candidateAuditFullSnapshot(items);
 if(candidateRefinementSnapshots.get(scope)?.pending)throw Error('当前候选正在后台精修，请等待本轮完成');
 const entry={pending:true,stale:false,snapshot,editSnapshot:candidateAuditFullSnapshot(manualDrafts[key]??items),sourceTopic,sourcePath,refinementId:newCandidateRefinementId(),generation:(candidateRefinementSnapshots.get(scope)?.generation||0)+1};candidateRefinementSnapshots.set(scope,entry);const generation=entry.generation;
 const assertClientCurrent=()=>{if(topic!==sourceTopic||loadedDataPath!==sourcePath||candidateRefinementSnapshots.get(scope)!==entry||entry.generation!==generation||entry.stale||candidateAuditFullSnapshot(candidates[key])!==snapshot)throw Error('候选或数据作用域已变化，后台精修结果已过期')};
 try{
  const fresh=await readCompactFrameStateSnapshot(target,sourceTopic,sourcePath);assertClientCurrent();
  const binding=fresh.candidate_review_bindings?.[key],artifact=binding?.candidate_artifact_sha256;
  if(candidateAuditFullSnapshot(fresh.candidates?.[key])!==snapshot||fresh.agent_runs?.[key]?.agent_review_status!=='pending'||!artifact)throw Error('候选服务器绑定不一致，已拒绝后台精修');
  const imageResponse=await fetch(`/api/frame/${target}?topic=${encodeURIComponent(sourceTopic)}`,{cache:'no-store'});if(!imageResponse.ok)throw Error('无法读取精修原图');
  const imageHash=Array.from(new Uint8Array(await crypto.subtle.digest('SHA-256',await imageResponse.arrayBuffer())),value=>value.toString(16).padStart(2,'0')).join('');assertClientCurrent();
  const result=await api('reference_candidates/refine',{frame:Number(target),topic:sourceTopic,refinement_id:entry.refinementId,expected_candidate_artifact_sha256:artifact,expected_frame_sha256:imageHash});assertClientCurrent();
  if(result.stopped)throw Error('后台 Agent 精修已停止，初始候选保持不变');
  if(result.refinement_id!==entry.refinementId||result.frame!==Number(target)||result.topic!==sourceTopic||result.candidate_only!==true||result.human_truth!==false||result.approved_for_training!==false||result.input_sha256!==imageHash||result.initial_candidate_artifact_sha256!==artifact||!Array.isArray(result.candidates)||!result.candidates.length||result.agent?.agent_review_status!=='complete'||result.agent?.candidate_artifact_sha256!==result.candidate_artifact_sha256)throw Error('后台精修响应绑定不一致，已拒绝采用');
  const readback=await readCompactFrameStateSnapshot(target,sourceTopic,sourcePath);
  if(topic!==sourceTopic||loadedDataPath!==sourcePath||candidateAuditFullSnapshot(readback.candidates?.[key])!==candidateAuditFullSnapshot(result.candidates)||candidateAuditFullSnapshot(readback.agent_runs?.[key])!==candidateAuditFullSnapshot(result.agent)||readback.candidate_review_bindings?.[key]?.candidate_artifact_sha256!==result.candidate_artifact_sha256)throw Error('后台精修服务器读回不一致，已停止采用');
  candidates[key]=JSON.parse(JSON.stringify(result.candidates));agentRuns[key]=JSON.parse(JSON.stringify(result.agent));candidateReviewBindings=readback.candidate_review_bindings||{};entry.pending=false;return {candidates:candidates[key],agent:agentRuns[key],mutated:result.candidate_mutated===true};
 }finally{entry.pending=false}
}
let deferredRunRefinementActive=false;
async function refineDeferredRunFrames(rows,sourceTopic,sourcePath){
 const pending=(rows||[]).filter(item=>item?.agent_review_status==='pending'&&item?.agent_review_deferred===true&&Number(item.remaining)>0);
 if(!pending.length)return {completed:0,failed:0};
 deferredRunRefinementActive=true;renderRunButton();let completed=0,failed=0;
 try{
  for(const row of pending){
   if(topic!==sourceTopic||loadedDataPath!==sourcePath)break;
   const target=Number(row.frame),key=String(target);let entry=null;
   try{
    const fresh=await readCompactFrameStateSnapshot(target,sourceTopic,sourcePath),items=fresh.candidates?.[key],run=fresh.agent_runs?.[key],binding=fresh.candidate_review_bindings?.[key],artifact=binding?.candidate_artifact_sha256;
    if(!Array.isArray(items)||!items.length||run?.agent_review_status!=='pending'||!artifact)throw Error('初始候选服务器绑定已变化');
    const initialSnapshot=candidateAuditFullSnapshot(items),scope=candidateAuditScope(target,sourceTopic,sourcePath);
    if(candidateRefinementSnapshots.get(scope)?.pending)throw Error('该帧已有 Agent 后台精修');
    entry={pending:true,stale:false,snapshot:initialSnapshot,editSnapshot:candidateAuditFullSnapshot(manualDrafts[key]??items),sourceTopic,sourcePath,refinementId:newCandidateRefinementId(),generation:(candidateRefinementSnapshots.get(scope)?.generation||0)+1};candidateRefinementSnapshots.set(scope,entry);const generation=entry.generation;
    const assertRefinementCurrent=()=>{if(candidateRefinementSnapshots.get(scope)!==entry||entry.generation!==generation||entry.stale)throw Error('人工草稿已变化，过时的后台精修已取消')};
    const imageResponse=await fetch(`/api/frame/${target}?topic=${encodeURIComponent(sourceTopic)}`,{cache:'no-store'});
    if(!imageResponse.ok)throw Error('无法读取后台精修原图');
    const imageHash=Array.from(new Uint8Array(await crypto.subtle.digest('SHA-256',await imageResponse.arrayBuffer())),value=>value.toString(16).padStart(2,'0')).join('');
    if(topic!==sourceTopic||loadedDataPath!==sourcePath)break;
    assertRefinementCurrent();const result=await api('reference_candidates/refine',{frame:target,topic:sourceTopic,refinement_id:entry.refinementId,expected_candidate_artifact_sha256:artifact,expected_frame_sha256:imageHash});assertRefinementCurrent();
    if(result.stopped)throw Error('后台精修已停止');
    if(result.refinement_id!==entry.refinementId||result.frame!==target||result.topic!==sourceTopic||result.candidate_only!==true||result.human_truth!==false||result.approved_for_training!==false||result.input_sha256!==imageHash||result.initial_candidate_artifact_sha256!==artifact||!Array.isArray(result.candidates)||!result.candidates.length||result.agent?.agent_review_status!=='complete'||result.agent?.candidate_artifact_sha256!==result.candidate_artifact_sha256)throw Error('后台精修响应绑定不一致');
    const readback=await readCompactFrameStateSnapshot(target,sourceTopic,sourcePath);
    if(candidateAuditFullSnapshot(readback.candidates?.[key])!==candidateAuditFullSnapshot(result.candidates)||candidateAuditFullSnapshot(readback.agent_runs?.[key])!==candidateAuditFullSnapshot(result.agent)||readback.candidate_review_bindings?.[key]?.candidate_artifact_sha256!==result.candidate_artifact_sha256)throw Error('后台精修服务器读回不一致');
    if(topic===sourceTopic&&loadedDataPath===sourcePath){
     const localSnapshot=candidateAuditFullSnapshot(candidates[key]);
     if(!candidates[key]||localSnapshot===initialSnapshot){candidates[key]=JSON.parse(JSON.stringify(result.candidates));agentRuns[key]=JSON.parse(JSON.stringify(result.agent));candidateReviewBindings=readback.candidate_review_bindings||candidateReviewBindings;candidateCountIndex[key]=result.candidates.length;if(frame===target){refreshCandidateSummary();refreshShapeList();updateReferenceStatus();draw()}}
    }
    completed++;
   }catch(error){failed++;if(topic===sourceTopic&&loadedDataPath===sourcePath)toast(`帧 ${row.frame} 的 Agent 后台精修未采用：${error.message}；初始机器候选已保留`)}finally{if(entry)entry.pending=false}
  }
  if(topic===sourceTopic&&loadedDataPath===sourcePath&&completed)toast(`Agent 已在后台完成 ${completed} 帧精修；候选仍需人工确认${failed?`，${failed} 帧保留初始候选`:''}`);
  return {completed,failed};
 }finally{deferredRunRefinementActive=false;await refreshAgentProgress();renderRunButton()}
}
function invalidateChangedCandidateAudits(items,target=frame){
 const scope=candidateAuditScope(target),entry=candidateAuditSnapshots.get(scope);if(!entry||entry.snapshot===candidateAuditSnapshot(items))return false;
 entry.stale=true;entry.snapshot=candidateAuditSnapshot(items);entry.generation++;
 const run=agentRuns[String(target)];if(run){run.candidate_class_audits=[];run.review_queue=(run.review_queue||[]).filter(item=>item?.source!=='candidate_class_audit');run.candidate_class_audit_status='stale'}
 for(const shape of items)if(shape.provenance?.assistant_review_source==='candidate_class_blind_audit')shape.provenance={...shape.provenance,candidate_audit_stale:true};
 referenceReviewedShapeKeys=new Set();referenceOriginalSweepVisited=false;referenceOpenSetReviewed=false;referenceDismissedMissingLabels=new Set();
 referenceFinalizeNotice='候选集合已变化，旧类别盲审已失效；请重新逐个检查，机器审核不代表人工确认';return true;
}
function referenceReviewScope(items,target=frame,sourceTopic=topic){const payload=JSON.stringify({file:loadedDataPath,topic:sourceTopic,frame:Number(target),keys:referenceReviewKeys(items)});let hash=2166136261;for(let index=0;index<payload.length;index++){hash^=payload.charCodeAt(index);hash=Math.imul(hash,16777619)}return `${sourceTopic}|${Number(target)}|${(hash>>>0).toString(16).padStart(8,'0')}`}
function readReferenceReviewProgress(){try{return JSON.parse(sessionStorage.getItem(referenceReviewStorageKey)||'{}')}catch{return {}}}
function restoreReferenceReviewProgress(items,target=frame,sourceTopic=topic){const keys=new Set(referenceReviewKeys(items)),stored=readReferenceReviewProgress()[referenceReviewScope(items,target,sourceTopic)]||{},reviewed=Array.isArray(stored.reviewed_keys)?stored.reviewed_keys:[];referenceReviewedShapeKeys=new Set(reviewed.filter(key=>keys.has(key)));referenceOriginalSweepVisited=stored.original_sweep_visited===true;referenceOpenSetReviewed=stored.open_set_reviewed===true;return referenceReviewedShapeKeys.size}
function persistReferenceReviewProgress(items=selectedShapes()){const intent=referencePreparationIntent;if(!intent||!Array.isArray(items)||!items.length)return;const keys=new Set(referenceReviewKeys(items)),valid=[...referenceReviewedShapeKeys].filter(key=>keys.has(key)),all=readReferenceReviewProgress(),scope=referenceReviewScope(items,intent.frame,intent.topic);all[scope]={reviewed_keys:valid,original_sweep_visited:referenceOriginalSweepVisited,open_set_reviewed:referenceOpenSetReviewed,recorded_at:new Date().toISOString(),browser_session_only:true};try{sessionStorage.setItem(referenceReviewStorageKey,JSON.stringify(all))}catch{}}
function clearStoredReferenceReviewProgress(items,intent){if(!intent||!Array.isArray(items)||!items.length)return;const all=readReferenceReviewProgress(),scope=referenceReviewScope(items,intent.frame,intent.topic);delete all[scope];try{sessionStorage.setItem(referenceReviewStorageKey,JSON.stringify(all))}catch{}}
function agentGeometryWarnings(run){
 const warnings=[];
 for(const entry of run?.audit||[]){
  for(const operation of entry?.decision?.operations||[]){
   if(operation?.operation!=='sam_points')continue;
   const points=Array.isArray(operation.points)?operation.points:[],pointCoordinates=new Map();
   for(const point of points){const key=`${Number(point?.x)},${Number(point?.y)}`,labels=pointCoordinates.get(key)||new Set();labels.add(point?.label);pointCoordinates.set(key,labels)}
   const polygon=Array.isArray(operation.polygon)?operation.polygon:[],uniqueVertices=new Set(polygon.map(point=>`${Number(point?.[0])},${Number(point?.[1])}`));let area=0;
   if(polygon.length>=3)for(let index=0;index<polygon.length;index++){const current=polygon[index],next=polygon[(index+1)%polygon.length];area+=Number(current?.[0])*Number(next?.[1])-Number(next?.[0])*Number(current?.[1])}
   if(points.length>1&&pointCoordinates.size===1)warnings.push('duplicate_only_points');
   if([...pointCoordinates.values()].some(labels=>labels.size>1))warnings.push('conflicting_point_labels');
   if(polygon.length<3||uniqueVertices.size<3||Math.abs(area)/2<0.5)warnings.push('zero_area_polygon');
  }
 }
 return [...new Set(warnings)];
}
function normalizeReferenceDiscovery(discovery){
 const normalized=discovery&&typeof discovery==='object'?JSON.parse(JSON.stringify(discovery)):{};
 normalized.visible_labels=Array.isArray(normalized.visible_labels)?normalized.visible_labels:[];
 normalized.uncertain_labels=Array.isArray(normalized.uncertain_labels)?normalized.uncertain_labels:[];
 normalized.semantic_warnings=[];
 const assessment=(Array.isArray(normalized.assessments)?normalized.assessments:[]).find(item=>item?.label==='cargo_floor');
 if(assessment?.evidence==='present'){
  const reason=' '+String(assessment.reason||'').toLowerCase().replace(/\s+/g,' ').trim(),palletTerms=['pallet','托盘','栈板','卡板'],attachmentTerms=['attached to a robot','attached to the robot','attached to robot','mounted on a robot','mounted on the robot','mounted on robot','part of a robot','part of the robot','part of robot','integrated with a robot','integrated with the robot','robot chassis','robot body','robot cargo deck','robot cargo platform','mobile robot chassis','mobile platform chassis','mobile platform deck','连接在机器人','安装在机器人','机器人底盘','机器人车体','机器人货台','机器人载货平台','移动平台的货台'];
  const palletConfusion=palletTerms.some(term=>reason.includes(term)),attachmentSupported=attachmentTerms.some(term=>reason.includes(term));
  if(palletConfusion||!attachmentSupported){
   normalized.visible_labels=normalized.visible_labels.filter(label=>label!=='cargo_floor');
   if(!normalized.uncertain_labels.includes('cargo_floor'))normalized.uncertain_labels.push('cargo_floor');
   normalized.semantic_warnings.push({label:'cargo_floor',reason_code:palletConfusion?'cargo_floor_pallet_confusion':'cargo_floor_missing_robot_attachment_evidence',original_evidence:'present',normalized_evidence:'uncertain',reason:String(assessment.reason||'')});
  }
 }
 return normalized;
}
function beginReferenceFinalize(source,target){referencePreparationIntent={source,frame:Number(target),topic,path:loadedDataPath};referenceDismissedMissingLabels=new Set();referenceOriginalSweepVisited=false;referenceOpenSetReviewed=false;const restored=restoreReferenceReviewProgress(selectedShapes(),target,topic);referenceFinalizeNotice=restored||referenceOpenSetReviewed?`已从当前浏览器会话恢复检查进度 ${restored}/${selectedShapes().length}${referenceOpenSetReviewed?'，原图漏标检查已完成':''}`:'';resetReferenceReviewView();updateReferenceFinalizeGuide()}
function clearReferenceFinalize(){referencePreparationIntent=null;referenceFinalizeNotice='';referenceReviewedShapeKeys=new Set();referenceDismissedMissingLabels=new Set();referenceOriginalSweepVisited=false;referenceOpenSetReviewed=false;resetReferenceReviewView();updateReferenceFinalizeGuide();document.dispatchEvent(new CustomEvent('labelagent:reference-draft-cleared'))}
function appendCandidateClassAuditAlerts(alerts,items,run){
 const audits=Array.isArray(run?.candidate_class_audits)?run.candidate_class_audits:[],queue=(Array.isArray(run?.review_queue)?run.review_queue:[]).filter(item=>item?.source==='candidate_class_audit'),records=[...audits];
 for(const item of queue)if(!audits.some(audit=>String(audit.candidate_id)===String(item.candidate_id)))records.push(item);
 for(const item of items)if(candidateClassAuditRequired(item)&&!records.some(audit=>String(audit.candidate_id)===String(item.id)))records.push({candidate_id:item.id,bound_current_label:item.label,route:'human_review_required',reason:'缺少类别盲审记录；复用不会触发新的 VLM 盲审，请人工核对类别与外观'});
 if(!records.length)return;
 alerts.hidden=false;const heading=document.createElement('strong');heading.textContent='候选类别盲审 · 原类别与 VLM 结果';alerts.append(heading);
 for(const audit of records){
  const index=items.findIndex(item=>String(item.id)===String(audit.candidate_id)),candidate=index>=0?items[index]:null,decision=audit.decision||{},required=audit.route==='human_review_required'||audit.human_review_required===true,note=document.createElement('span');
  const reason=decision.reason||audit.reason||audit.error||decision.reason_code||audit.reason_code||'未提供原因';
  note.textContent=`候选 #${audit.candidate_id??'未知'} · 原类别：${audit.bound_current_label??candidate?.label??'未记录'} · VLM observed_label：${decision.observed_label??audit.observed_label??'未确定'} · 原因：${reason} · human_review_required：${required?'true（必须人工复核）':'false（仍需逐个检查）'}${candidate?'':' · 当前草稿中未找到该候选'}`;
  alerts.append(note);if(required)alerts.classList.add('geometryRisk');
  if(candidate){const inspect=document.createElement('button');inspect.type='button';inspect.textContent=`查看候选 #${audit.candidate_id}`;inspect.onclick=()=>{selectedShapeIndex=index;referenceReviewOriginalOnly=false;if(referenceReviewFocused)applyReferenceReviewFocus();refreshShapeList();draw();updateReferenceFinalizeGuide();$('shapeList').children[index]?.scrollIntoView({behavior:'smooth',block:'nearest'})};alerts.append(inspect)}
 }
}
function updateReferenceFinalizeGuide(){
 if(referencePreparationIntent?.source==='machine_candidate'&&referencePreparationIntent.frame===frame&&referencePreparationIntent.topic===topic)invalidateChangedCandidateAudits(selectedShapes(),frame);
 const guide=$('referenceFinalizeGuide');if(!guide)return;
 const intent=referencePreparationIntent;guide.hidden=!intent;if(!intent)return;
 const same=intent.path===loadedDataPath&&intent.topic===topic&&intent.frame===frame,key=String(intent.frame),items=same?(shapes[key]??manualDrafts[key]??autoDrafts[key]??candidates[key]??[]):[],truthReady=same&&truthIsVerified(intent.frame)&&Array.isArray(shapes[key])&&shapes[key].length>0,draftReady=same&&Array.isArray(items)&&items.length>0,reference=window.referenceFrames?.[intent.topic]?.[0],locked=reference===intent.frame,run=agentRuns[key],openSetIncomplete=run?.visible_label_discovery?.open_set_scan_complete===false||(run?.review_queue||[]).some(item=>item?.reason_code==='vlm_open_set_output_truncated'),unresolvedModelConflicts=items.map((shape,index)=>({index,conflict:unresolvedCandidateModelConflict(shape)})).filter(item=>item.conflict),reviewKeys=referenceReviewKeys(items),reviewedCount=reviewKeys.filter(item=>referenceReviewedShapeKeys.has(item)).length,allReviewed=items.length>0&&reviewedCount===items.length&&unresolvedModelConflicts.length===0;refreshClassColorLegend(items);
 const active=locked?3:truthReady?3:draftReady?2:1;
 for(const [id,step]of[['referenceFinalizeStep1',1],['referenceFinalizeStep2',2],['referenceFinalizeStep3',3]]){const item=$(id);item.classList.toggle('active',step===active&&!locked);item.classList.toggle('done',locked||step<active)}
 const status=referenceFinalizeNotice||(locked?'参考帧已锁定':!same?`正在准备帧 ${intent.frame}，请先返回该帧`:truthReady?'人工真值已保存且回执已核验，最后锁定参考帧':intent.source==='legacy_truth'?'历史标注已载入，请检查/精修后完成人工确认':draftReady?'轮廓草稿待人工确认':'当前没有轮廓，请先完成标注');$('referenceFinalizeStatus').textContent=status;
 const review=$('referenceFinalizeReview'),advice=$('referenceFinalizeAdvice'),navigation=$('referenceReviewNavigation');let unresolvedMissing=[];review.hidden=!same||(!draftReady&&!truthReady);advice.hidden=review.hidden;navigation.hidden=review.hidden||locked;
 if(!review.hidden){
  $('referenceFinalizeCount').textContent=String(items.length);$('referenceReviewProgress').textContent=truthReady||locked?'逐个检查已完成':`已检查 ${reviewedCount} / ${items.length}`;
  review.classList.toggle('verified',truthReady||locked);
  const geometryWarnings=agentGeometryWarnings(run),invalidModelOutput=run?.stop_reason==='invalid_model_output'||(run?.review_queue||[]).some(item=>item?.reason_code==='agent_invalid_model_output');let aiStatus='草稿待逐个检查',next='逐个点击轮廓，检查漏标、粘连和边界；必要时用点提示 SAM 或橡皮擦修正。';
  if(intent.source==='machine_candidate'&&invalidModelOutput){aiStatus='Agent 未形成有效修正，已安全保留 SAM 候选';next='机器修正没有通过协议门禁；请逐个人工检查候选，必要时使用有效点提示或橡皮擦修正。'}
  else if(intent.source==='machine_candidate'&&geometryWarnings.length){aiStatus='Agent 历史修正含无效几何，不能视为已通过';next='候选仍可作为草稿，但请逐个人工检查；建议重新一键起稿，或用有效点提示/橡皮擦修正。'}
  else if(intent.source==='machine_candidate'&&run?.stop_reason==='no_progress'){aiStatus='VLM 已复核，但未产生可安全执行的修改';next='请重点检查小目标、遮挡处和轮廓边缘；机器没有安全修改不等于轮廓一定正确。'}
  else if(intent.source==='machine_candidate'&&run?.converged){aiStatus='Agent 流程已结束（不代表正确）';next='机器流程结束只表示停止迭代；仍需逐个检查类别、边界、漏标和多标。'}
  else if(intent.source==='legacy_truth'){aiStatus='历史标注已载入，来源需重新确认';next='历史轮廓可以省去重画，但请重新检查类别与边界，再完成人工确认。'}
  else if(intent.source==='sam_point_prompt')aiStatus='点提示 SAM 草稿';
  else if(intent.source==='manual_polygon')aiStatus='人工绘制草稿';
  if(truthReady){aiStatus='人工真值与回执已核验';next='当前真值已通过人工门禁，可以继续锁定为参考帧。'}
  if(locked){aiStatus='人工真值已锁定';next='该帧已经是参考帧，后续机器流程不会修改它。'}
   $('referenceFinalizeAiStatus').textContent=aiStatus;$('referenceFinalizeTruthStatus').textContent=locked?'参考帧已锁定':truthReady?'人工真值已核验':'仍是草稿 · 未写入人工真值';advice.textContent=next;
const egoFallback=run?.ego_bottom_point_fallback||{},egoFallbackCandidate=egoFallback.status==='candidate_generated';
const alerts=$('referenceReviewAlerts'),discovery=normalizeReferenceDiscovery(run?.visible_label_discovery||{}),uncertain=discovery.uncertain_labels,visible=discovery.visible_labels,semanticWarnings=discovery.semantic_warnings||[],unknown=Array.isArray(discovery.unknown_candidates)?discovery.unknown_candidates:[],conflicts=Array.isArray(discovery.registry_conflicts)?discovery.registry_conflicts:[],candidateLabels=new Set(items.map(item=>item.label).filter(Boolean)),missingVisible=[...new Set([...visible,...uncertain])].filter(label=>!candidateLabels.has(label));unresolvedMissing=missingVisible.filter(label=>!referenceDismissedMissingLabels.has(label));alerts.replaceChildren();alerts.classList.toggle('geometryRisk',geometryWarnings.length>0||invalidModelOutput||semanticWarnings.length>0||unresolvedMissing.length>0);alerts.hidden=truthReady||locked||(!invalidModelOutput&&!geometryWarnings.length&&!semanticWarnings.length&&!uncertain.length&&!unknown.length&&!conflicts.length&&!unresolvedMissing.length);if(unresolvedMissing.length){const semanticLabels=unresolvedMissing.filter(label=>semanticWarnings.some(item=>item.label===label)),ordinaryLabels=unresolvedMissing.filter(label=>!semanticLabels.includes(label)),currentDraftLabels=[...candidateLabels].join('、')||'暂无类别';advice.textContent=semanticLabels.length?`当前草稿已有 ${currentDraftLabels}。旧 VLM 对 ${semanticLabels.join('、')} 的证据不合格，已降级为不确定项；请单独核对，不会修改已有轮廓。`:`当前草稿已有 ${currentDraftLabels}；另有 ${ordinaryLabels.join('、')} 可能漏标。下方“补标缺失类别”只会新增轮廓，不会改动已有类别。`}if(!alerts.hidden){const heading=document.createElement('strong');heading.textContent=geometryWarnings.length||invalidModelOutput||semanticWarnings.length||unresolvedMissing.length?'安全提醒':'优先复核';alerts.append(heading);if(invalidModelOutput){const note=document.createElement('span');note.textContent='Agent 连续返回无效动作，系统未执行修改；SAM 候选仍需逐个人工检查';alerts.append(note)}else if(geometryWarnings.length){const note=document.createElement('span');note.textContent='检测到重复点或零面积修正：旧 Agent 完成状态已作废，请勿直接确认';alerts.append(note)}for(const warning of semanticWarnings){const note=document.createElement('span');note.className='semanticEvidenceWarning';note.textContent=warning.reason_code==='cargo_floor_pallet_confusion'?'cargo_floor 旧证据引用了托盘；塑料托盘不等于与机器人相连的载货平台，不能直接补标，已降级为不确定':'cargo_floor 旧证据未证明货台与机器人相连；塑料托盘不等于与机器人相连的载货平台，只有可见的机器人载货平台才算 cargo_floor，已降级为不确定';alerts.append(note)}for(const label of uncertain.filter(label=>candidateLabels.has(label))){const count=items.filter(item=>item.label===label).length;const button=document.createElement('button');button.type='button';button.dataset.reviewLabel=label;button.textContent=`先检查 ${label}（${count} 个候选）`;button.onclick=()=>selectReferenceReviewLabel(label);alerts.append(button)}for(const label of unresolvedMissing){const note=document.createElement('span');note.className='missingCandidateAlert';note.dataset.missingLabel=label;const semanticWarning=semanticWarnings.find(item=>item.label===label);note.textContent=semanticWarning?`需单独核对缺失的 ${label}：旧 VLM 证据不足，当前草稿没有这个类别`:`当前草稿没有 ${label}：${visible.includes(label)?'VLM 判断画面中可能可见，需补标或确认不存在':'需确认是否漏标'}`;alerts.append(note);const action=document.createElement('button');action.type='button';action.dataset.missingCandidateAction=label;action.textContent=`补标缺失的 ${label}`;action.title=`只会新增 ${label} 轮廓，不会修改已有的 ${[...candidateLabels].join('、')||'轮廓'}`;action.onclick=()=>startMissingCandidatePointPrompt(label);alerts.append(action);const dismiss=document.createElement('button');dismiss.dataset.dismissMissingLabel=label;dismiss.textContent=`确认画面中没有 ${label}`;dismiss.onclick=()=>resolveMissingCandidateAsAbsent(label);alerts.append(dismiss)}if(unknown.length){const note=document.createElement('span');note.textContent=`${unknown.length} 个疑似新类待人工命名`;alerts.append(note)}if(conflicts.length){const note=document.createElement('span');note.textContent=`${conflicts.length} 个标签别名冲突待核对`;alerts.append(note)}}
if(egoFallbackCandidate&&!truthReady&&!locked){alerts.hidden=false;alerts.classList.add('geometryRisk');if(!alerts.querySelector('strong')){const heading=document.createElement('strong');heading.textContent='安全提醒';alerts.append(heading)}const note=document.createElement('span');note.className='egoFallbackCandidateAlert';const profile=egoFallback.evidence?.selected_profile||'受限底部结构';note.textContent=`VLM 看见 ego，但文本分割漏检；系统用 ${profile} 点扫描补出 1 个候选。该结果已进入高风险复核，仍须逐个检查后才能成为人工真值。`;alerts.append(note)}
  if(!truthReady&&!locked)appendCandidateClassAuditAlerts(alerts,items,run);
  if(openSetIncomplete&&!truthReady&&!locked){alerts.hidden=false;alerts.classList.add('geometryRisk');if(!alerts.children.length){const heading=document.createElement('strong');heading.textContent='安全提醒';alerts.append(heading)}const note=document.createElement('span');note.className='openSetIncompleteWarning';note.textContent=referenceOpenSetReviewed?'✓ 已完成人工原图漏标检查；该记录仅保存在当前浏览器会话，最终仍需人工真值门禁。':'VLM 的新类别扫描输出未完整返回；必须先只看原图检查未覆盖设备、远处小目标和贴边区域，再明确确认完成。';alerts.append(note);const inspect=document.createElement('button');inspect.type='button';inspect.dataset.openSetReviewAction='inspect';inspect.textContent=referenceOriginalSweepVisited?'重新只看原图检查':'第 1 步：只看原图检查';inspect.onclick=()=>{referenceOriginalSweepVisited=true;referenceReviewOriginalOnly=true;referenceReviewSolo=false;persistReferenceReviewProgress(items);updateReferenceFinalizeGuide();draw();toast('已隐藏全部候选；请检查漏标、远处小目标和标签表外目标')};alerts.append(inspect);const confirm=document.createElement('button');confirm.type='button';confirm.dataset.openSetReviewAction='confirm';confirm.disabled=!referenceOriginalSweepVisited||referenceOpenSetReviewed;confirm.textContent=referenceOpenSetReviewed?'✓ 原图漏标检查已确认':'第 2 步：确认已检查未覆盖区域';confirm.onclick=()=>{if(!referenceOriginalSweepVisited){toast('请先进入“只看原图”完成检查');return}referenceOpenSetReviewed=true;referenceReviewOriginalOnly=false;persistReferenceReviewProgress(items);updateReferenceFinalizeGuide();draw();toast('已记录本次原图漏标检查；仍需逐个检查轮廓并通过人工真值门禁')};alerts.append(confirm)}
  const validSelection=selectedShapeIndex>=0&&selectedShapeIndex<items.length,currentReviewKey=validSelection?reviewKeys[selectedShapeIndex]:null,currentReviewed=currentReviewKey&&referenceReviewedShapeKeys.has(currentReviewKey);$('referenceReviewPosition').textContent=validSelection?`${selectedShapeIndex+1} / ${items.length} · ${items[selectedShapeIndex].label}`:`尚未选择 · 共 ${items.length} 个`;
  $('referenceReviewPrevious').disabled=!items.length;$('referenceReviewNext').disabled=!items.length;const riskOrder=window.labelAgentCandidateRiskOrder?.(items,run,labels,Number($('candidateConfidenceThreshold')?.value||0.65),imageWidth,imageHeight)||[],riskButton=$('referenceReviewRisk');riskButton.disabled=!riskOrder.length;riskButton.textContent=riskOrder.length?`先查高风险（${riskOrder.length}）`:'暂无显式风险';const solo=$('referenceReviewSolo');solo.disabled=!items.length;solo.classList.toggle('selected',referenceReviewSolo&&validSelection);solo.setAttribute('aria-pressed',String(referenceReviewSolo&&validSelection));solo.textContent=referenceReviewSolo&&validSelection?'仅看当前：开':'仅看当前：关';const focus=$('referenceReviewFocus');focus.disabled=!items.length;focus.classList.toggle('selected',referenceReviewFocused&&validSelection);focus.setAttribute('aria-pressed',String(referenceReviewFocused&&validSelection));focus.textContent=referenceReviewFocused&&validSelection?`聚焦当前：${zoomScale.toFixed(1)}×`:'聚焦当前：关';const original=$('referenceReviewOriginal');original.disabled=!items.length;original.classList.toggle('selected',referenceReviewOriginalOnly);original.setAttribute('aria-pressed',String(referenceReviewOriginalOnly));original.textContent=referenceReviewOriginalOnly?'只看原图：开':'只看原图：关';
  const currentModelConflict=validSelection?unresolvedCandidateModelConflict(items[selectedShapeIndex]):null,approve=$('referenceReviewApprove');approve.disabled=!validSelection||truthReady||locked||Boolean(currentModelConflict);approve.classList.toggle('selected',Boolean(currentReviewed));approve.setAttribute('aria-pressed',String(Boolean(currentReviewed)));approve.textContent=currentModelConflict?'先明确处理标签冲突':currentReviewed?'✓ 当前已检查':'当前轮廓检查通过';const approveNext=$('referenceReviewApproveNext');approveNext.disabled=!validSelection||truthReady||locked||Boolean(currentReviewed)||Boolean(currentModelConflict);approveNext.textContent=currentModelConflict?'先明确处理标签冲突':allReviewed?'全部已检查':currentReviewed?'当前已检查':'检查通过并下一个 →';
 }
 const missingLabelPending=same&&!truthReady&&draftReady&&unresolvedMissing.length>0,openSetPending=same&&!truthReady&&draftReady&&openSetIncomplete&&!referenceOpenSetReviewed,modelConflictPending=same&&!truthReady&&draftReady&&unresolvedModelConflicts.length>0,checklistPending=same&&!truthReady&&draftReady&&!allReviewed,primary=$('referenceFinalizePrimary');primary.textContent=!same?'返回准备帧':truthReady?'锁定为参考帧':missingLabelPending?`先处理可能漏标：${unresolvedMissing.join('、')}`:openSetPending?'先完成原图漏标检查':modelConflictPending?`先处理标签冲突 (${unresolvedModelConflicts.length})`:checklistPending?`先完成逐个检查 (${reviewedCount}/${items.length})`:'人工确认并设为参考帧';primary.disabled=referenceFinalizeBusy||locked||(same&&!truthReady&&!draftReady)||missingLabelPending||openSetPending||modelConflictPending||checklistPending||(intent.source==='machine_candidate'&&Boolean(candidateAuditSnapshots.get(candidateAuditScope(intent.frame))?.pending||candidateAuditSnapshots.get(candidateAuditScope(intent.frame))?.stale));$('referenceFinalizeCancel').disabled=referenceFinalizeBusy
}
function applyReferenceReviewFocus(){const shape=selectedShapes()[selectedShapeIndex];if(!shape||!imageWidth||!imageHeight)return false;const points=(shape.parts||[shape]).flatMap(part=>part.points||[]);if(!points.length)return false;const xs=points.map(point=>point[0]),ys=points.map(point=>point[1]),left=Math.min(...xs),right=Math.max(...xs),top=Math.min(...ys),bottom=Math.max(...ys),width=Math.max(1,right-left),height=Math.max(1,bottom-top),cx=(left+right)/2,cy=(top+bottom)/2;zoomScale=Math.max(1.5,Math.min(5,0.62/Math.max(width/imageWidth,height/imageHeight)));$('stage').style.transformOrigin=`${cx/imageWidth*100}% ${cy/imageHeight*100}%`;$('stage').style.transform=`scale(${zoomScale})`;referenceReviewFocused=true;draw();return true}
function selectReferenceReviewShape(direction){const items=selectedShapes();if(!items.length)return;referenceReviewOriginalOnly=false;selectedShapeIndex=selectedShapeIndex<0?(direction<0?items.length-1:0):(selectedShapeIndex+direction+items.length)%items.length;if(referenceReviewFocused)applyReferenceReviewFocus();refreshShapeList();draw();updateReferenceFinalizeGuide();$('shapeList').children[selectedShapeIndex]?.scrollIntoView({behavior:'smooth',block:'nearest'});toast(`正在检查 ${selectedShapeIndex+1}/${items.length}：${items[selectedShapeIndex].label}`)}
function selectReferenceReviewLabel(label){const items=selectedShapes(),keys=referenceReviewKeys(items),matches=items.map((item,index)=>item.label===label?index:-1).filter(index=>index>=0);if(!matches.length){toast(`当前草稿没有 ${label} 候选，请人工补标`);return}referenceReviewOriginalOnly=false;selectedShapeIndex=matches.find(index=>!referenceReviewedShapeKeys.has(keys[index]))??matches[0];if(referenceReviewFocused)applyReferenceReviewFocus();refreshShapeList();draw();updateReferenceFinalizeGuide();$('shapeList').children[selectedShapeIndex]?.scrollIntoView({behavior:'smooth',block:'nearest'});toast(`优先检查 ${label}：${selectedShapeIndex+1}/${items.length}`)}
function startMissingCandidatePointPrompt(label){const select=$('labelSelect'),available=[...select.options].some(option=>option.value===label);if(!available){toast(`类别表中没有 ${label}，已停止点选补标`);return}select.value=label;select.dispatchEvent(new Event('change',{bubbles:true}));$('pointButton').click();pointPromptAutoRunOnce=true;pointPromptNotice=`尚未提交：请在 ${label} 目标内部点一下，点击后才会运行 SAM`;updatePointPromptCoach();requestAnimationFrame(()=>{const coach=$('pointPromptCoach');coach?.scrollIntoView({behavior:'smooth',block:'center'});coach?.focus({preventScroll:true})});toast(`已切换到 ${label} 点选补标；尚未提交，请在目标内部点一下`)}
function resolveMissingCandidateAsAbsent(label){referenceDismissedMissingLabels.add(label);updateReferenceFinalizeGuide();const items=selectedShapes(),keys=referenceReviewKeys(items),run=agentRuns[String(frame)],riskOrder=window.labelAgentCandidateRiskOrder?.(items,run,labels,Number($('candidateConfidenceThreshold')?.value||0.65),imageWidth,imageHeight)||[],nextRisk=riskOrder.find(item=>!referenceReviewedShapeKeys.has(keys[item.index])),nextAny=items.map((item,index)=>index).find(index=>!referenceReviewedShapeKeys.has(keys[index])),nextIndex=nextRisk?.index??nextAny;if(nextIndex!==undefined){selectedShapeIndex=nextIndex;refreshShapeList();draw();updateReferenceFinalizeGuide();requestAnimationFrame(()=>{$('referenceReviewNavigation')?.scrollIntoView({behavior:'smooth',block:'center'});$('referenceReviewApproveNext')?.focus({preventScroll:true})});toast(`已记录本次目视判断：画面中没有 ${label}；接下来检查 ${nextIndex+1}/${items.length}`);return}requestAnimationFrame(()=>{$('referenceFinalizePrimary')?.scrollIntoView({behavior:'smooth',block:'center'});$('referenceFinalizePrimary')?.focus({preventScroll:true})});toast(`已记录本次目视判断：画面中没有 ${label}；现在可进入人工确认门禁`)}
function selectReferenceRiskShape(){const items=selectedShapes(),run=agentRuns[String(frame)],order=window.labelAgentCandidateRiskOrder?.(items,run,labels,Number($('candidateConfidenceThreshold')?.value||0.65),imageWidth,imageHeight)||[],keys=referenceReviewKeys(items);if(!order.length){toast('当前草稿没有检测到显式风险，仍需逐个人工检查');return}referenceReviewOriginalOnly=false;const target=order.find(item=>!referenceReviewedShapeKeys.has(keys[item.index]))||order[0];selectedShapeIndex=target.index;if(referenceReviewFocused)applyReferenceReviewFocus();refreshShapeList();draw();updateReferenceFinalizeGuide();$('shapeList').children[selectedShapeIndex]?.scrollIntoView({behavior:'smooth',block:'nearest'});toast(`优先检查高风险轮廓 ${selectedShapeIndex+1}/${items.length}：${items[selectedShapeIndex].label}`)}
function selectInitialReferenceReviewTarget(items,run){const order=window.labelAgentCandidateRiskOrder?.(items,run,labels,Number($('candidateConfidenceThreshold')?.value||0.65),imageWidth,imageHeight)||[],keys=referenceReviewKeys(items),nextRisk=order.find(item=>!referenceReviewedShapeKeys.has(keys[item.index])),nextAny=items.map((item,index)=>index).find(index=>!referenceReviewedShapeKeys.has(keys[index]));selectedShapeIndex=nextRisk?.index??nextAny??order[0]?.index??(items.length?0:-1);return Boolean(nextRisk)}
function approveAndAdvanceReferenceReview(){const items=selectedShapes();if(!items.length)return;if(selectedShapeIndex<0){selectReferenceReviewShape(1);return}if(unresolvedCandidateModelConflict(items[selectedShapeIndex])){toast('当前存在模型类别冲突，请先明确选择保留当前类别或采用 VLM 建议');updateReferenceFinalizeGuide();return}const keys=referenceReviewKeys(items),currentKey=keys[selectedShapeIndex];if(!referenceReviewedShapeKeys.has(currentKey))referenceReviewedShapeKeys.add(currentKey);persistReferenceReviewProgress(items);const run=agentRuns[String(frame)],riskOrder=window.labelAgentCandidateRiskOrder?.(items,run,labels,Number($('candidateConfidenceThreshold')?.value||0.65),imageWidth,imageHeight)||[],nextRisk=riskOrder.find(item=>!referenceReviewedShapeKeys.has(keys[item.index])),nextAny=items.map((item,index)=>index).find(index=>!referenceReviewedShapeKeys.has(keys[index])),nextIndex=nextRisk?.index??nextAny;if(nextIndex===undefined){refreshShapeList();draw();updateReferenceFinalizeGuide();requestAnimationFrame(()=>{$('referenceFinalizePrimary')?.scrollIntoView({behavior:'smooth',block:'center'});$('referenceFinalizePrimary')?.focus({preventScroll:true})});toast(`已逐个检查 ${items.length}/${items.length}；请进入人工确认门禁`);return}selectedShapeIndex=nextIndex;if(referenceReviewFocused)applyReferenceReviewFocus();refreshShapeList();draw();updateReferenceFinalizeGuide();$('shapeList').children[selectedShapeIndex]?.scrollIntoView({behavior:'smooth',block:'nearest'});toast(`当前已检查，下一项 ${selectedShapeIndex+1}/${items.length}：${items[selectedShapeIndex].label}`)}
$('referenceReviewPrevious').onclick=()=>selectReferenceReviewShape(-1);$('referenceReviewNext').onclick=()=>selectReferenceReviewShape(1);
$('referenceReviewRisk').onclick=selectReferenceRiskShape;document.addEventListener('labelagent:risk-engine-ready',updateReferenceFinalizeGuide);
$('referenceReviewApproveNext').onclick=approveAndAdvanceReferenceReview;
$('referenceReviewApprove').onclick=()=>{const items=selectedShapes();if(!items.length)return;if(selectedShapeIndex<0)selectReferenceReviewShape(1);if(unresolvedCandidateModelConflict(items[selectedShapeIndex])){toast('当前存在模型类别冲突，请先明确选择保留当前类别或采用 VLM 建议');updateReferenceFinalizeGuide();return}const keys=referenceReviewKeys(items),key=keys[selectedShapeIndex];if(referenceReviewedShapeKeys.has(key)){referenceReviewedShapeKeys.delete(key);toast('已撤销当前轮廓的检查状态')}else{referenceReviewedShapeKeys.add(key);toast(`已检查 ${selectedShapeIndex+1}/${items.length}：${items[selectedShapeIndex].label}`)}persistReferenceReviewProgress(items);updateReferenceFinalizeGuide()};
$('referenceReviewSolo').onclick=()=>{if(selectedShapeIndex<0)selectReferenceReviewShape(1);referenceReviewOriginalOnly=false;referenceReviewSolo=!referenceReviewSolo;updateReferenceFinalizeGuide();draw();toast(referenceReviewSolo?'已隐藏其他轮廓，只检查当前黄色轮廓':'已恢复显示全部轮廓')};
$('referenceReviewOriginal').onclick=()=>{referenceReviewOriginalOnly=!referenceReviewOriginalOnly;if(referenceReviewOriginalOnly){referenceReviewSolo=false;referenceOriginalSweepVisited=true;persistReferenceReviewProgress()}updateReferenceFinalizeGuide();draw();toast(referenceReviewOriginalOnly?'已临时隐藏全部候选；请检查漏标、远处小目标和标签表外目标':'已恢复显示候选轮廓')};
$('referenceReviewFocus').onclick=()=>{if(selectedShapeIndex<0)selectReferenceReviewShape(1);if(referenceReviewFocused){referenceReviewFocused=false;zoomScale=1;$('stage').style.transform='';$('stage').style.transformOrigin='center center';draw();updateReferenceFinalizeGuide();toast('已还原完整画面');return}if(applyReferenceReviewFocus()){updateReferenceFinalizeGuide();toast(`已放大当前轮廓到 ${zoomScale.toFixed(1)} 倍`)}};
function updateReferenceStatus(){const reference=window.referenceFrames?.[topic]?.[0],status=$('referenceStatus'),kind=frameAnnotationState(),run=agentRuns[String(frame)],comparison=kind==='comparison_candidate',geometryRisk=agentGeometryWarnings(run).length>0,invalidModelOutput=run?.stop_reason==='invalid_model_output';status.textContent=(reference===undefined?'':`当前参考帧：${reference}`)+(kind==='truth'?' · 当前帧真值已保存':kind==='legacy_truth'?' · 当前帧是历史标注，尚未通过当前人工门禁':comparison?' · 正在显示机器评测候选；人工真值保持不变':kind==='candidate'?' · 当前帧是自动候选，尚未保存为真值':kind==='manual_draft'?' · 当前帧是可编辑人工草稿，尚未通过人工门禁':'')+((kind==='candidate'||comparison)&&run?` · Agent ${invalidModelOutput?'无有效修正，SAM 候选待人工复核':geometryRisk?'结果含无效修正，需人工复核':run.converged?'复查完成，候选待复核':'流程未完成'} (${run.rounds} 轮)`: '');$('referenceButton').textContent=reference===frame?'已是参考帧':'设为参考帧';$('truthSave').disabled=comparison;$('truthSave').textContent=comparison?'评测候选不可覆盖人工真值':window.workspaceMode==='auto'&&candidates[String(frame)]&&!shapes[String(frame)]?'确认并保存当前帧真值':'保存当前帧真值';$('openReferenceButton').hidden=reference===undefined;updateReferenceFinalizeGuide()}
function candidateServerSignature(state){const runs={};for(const [key,run] of Object.entries(state?.agent_runs||{}))runs[key]=[run?.rounds,run?.stop_reason,run?.converged,run?.candidate_event_id,run?.candidate_sha256,run?.candidate_class_audits,run?.review_queue];return JSON.stringify([state?.topic||'',state?.candidates||{},runs,state?.candidate_review_proposals||[],state?.candidate_review_bindings||{},state?.reference_frames||{}])}
window.syncWorkbenchCandidateState=state=>{
 if(!state||!state.loaded||!state.topic||state.topic!==topic||state.path!==loadedDataPath)return false;
 for(const key of new Set([...Object.keys(candidateReviewBindings),...Object.keys(state.candidate_review_bindings||{})])){const binding=state.candidate_review_bindings?.[key],entry=candidateAuditSnapshots.get(candidateAuditScope(Number(key)));if(entry?.bindingSnapshot&&entry.bindingSnapshot!==candidateAuditFullSnapshot(binding)){entry.stale=true;entry.generation++;if(Number(key)===frame){referenceReviewedShapeKeys=new Set();referenceOpenSetReviewed=false;referenceOriginalSweepVisited=false;referenceFinalizeNotice='服务器候选绑定已变化，旧复审已失效'}}}
 const signature=candidateServerSignature(state);if(signature===candidateStateSignature)return false;
 const key=String(frame),hadCurrent=Array.isArray(candidates[key])&&candidates[key].length>0;
 compactGeometryMode=false;workbenchConfig=state.config||workbenchConfig;configStateSha256=state.config_state_sha256||configStateSha256;truthFrameIndex=new Set(Object.keys(state.truth||{}));candidateCountIndex=Object.fromEntries(Object.entries(state.candidates||{}).map(([name,items])=>[name,Array.isArray(items)?items.length:0]));
 candidates=state.candidates||{};agentRuns=state.agent_runs||{};candidateReviewProposals=state.candidate_review_proposals||[];candidateReviewBindings=state.candidate_review_bindings||{};window.referenceFrames=state.reference_frames||{};candidateStateSignature=signature;
 refreshCandidateSummary();refreshShapeList();updateReferenceStatus();window.updateCandidateSave?.();draw();
 const hasCurrent=Array.isArray(candidates[key])&&candidates[key].length>0;
 if(!hadCurrent&&hasCurrent)toast(`后台已同步帧 ${frame} 的 ${candidates[key].length} 个机器候选；仍需人工检查后才能成为参考真值`);
 document.dispatchEvent(new CustomEvent('labelagent:candidate-state-synced',{detail:{frame,topic,candidate_count:Object.keys(candidates).length}}));
 return true;
};
document.addEventListener('labelagent:review-state',event=>window.syncWorkbenchCandidateState?.(event.detail?.state));
async function openFrameForReferencePreparation(target){
 if(!topic||catalog[topic]?.type!=='image')throw Error('当前没有可用的图像话题');
 const navigation=beginFrameNavigation();
 stopPlayback();
 const targetFrame=Math.max(0,Math.min(Number(target)||0,topics[topic].count-1));
 let view=annotationView;
 if(!view||view.sources[0]!==topic){
  if(annotationView)leaveAnnotation();
  const element=document.createElement('article');
  element.className='sensorWindow directStillReference';
  view={id:'direct-still-reference',sources:[topic],type:'image',element,layers:[],closed:false,wasMaximized:false,directStill:true};
  annotationView=view;
 }
 playback=null;
 view.element.hidden=true;
 $('primaryWindow').hidden=false;
 $('primaryWindow').classList.remove('maximized');
 $('primaryTitle').textContent=topic;
 setAnnotationMode(true);
 updateGrid();
 if(!view.directStill){
  const timeline=await prepareTimeline(topic);
  if(!frameNavigationIsCurrent(navigation))return false;
  playback=timeline;
  clockTime=(timeline.offset||0)+(timeline.times[targetFrame]??0);
 }
 await loadStill(targetFrame);
 if(!frameNavigationIsCurrent(navigation))return false;
 updateControls();
 updateWindows(true);
 return true;
}
async function openReferenceFrame(){const reference=window.referenceFrames?.[topic]?.[0];if(reference===undefined)return;try{await openFrameForReferencePreparation(reference)}catch(error){toast(error.message)}}
window.openFrameForReferencePreparation=openFrameForReferencePreparation;
async function enterReferencePreparationFrame(target,expectedGeneration,expectedTopic,expectedPath){
 if(frameNavigationGeneration!==expectedGeneration||topic!==expectedTopic||loadedDataPath!==expectedPath)throw Error('您已切换到其他帧、相机或数据集，本次参考起稿未进入');
 const opened=await openFrameForReferencePreparation(target);
 if(!opened||frame!==Number(target)||topic!==expectedTopic||loadedDataPath!==expectedPath)throw Error('您已切换到其他帧、相机或数据集，本次参考起稿未进入');
 return true;
}
function referencePreparationIsActive(target,expectedTopic,expectedPath){const intent=referencePreparationIntent;return frame===Number(target)&&topic===expectedTopic&&loadedDataPath===expectedPath&&intent?.frame===Number(target)&&intent?.topic===expectedTopic&&intent?.path===expectedPath}
window.prepareReferenceCandidateDraft=async(target,candidate,runMetadata=null)=>{
 const key=String(target),sourceTopic=topic,sourcePath=loadedDataPath,requestGeneration=frameNavigationGeneration;
 if(shapes[key]!==undefined)throw Error(truthIsVerified(target)?'该帧已有已核验人工真值，请直接复用已有标注':'该帧已有历史标注，请从参考帧助手载入并复核');
 if(!Array.isArray(candidate)||!candidate.length)throw Error('该帧没有可精修的机器候选');
 if(Array.isArray(candidates[key])&&candidateAuditFullSnapshot(candidates[key])!==candidateAuditFullSnapshot(candidate))throw Error('候选集合与当前服务器快照不一致，已拒绝载入');
 const agentPending=runMetadata?.agent_review_status==='pending';
 const auditPending=['pending','blocked_on_agent_review'].includes(runMetadata?.candidate_class_audit_status);
 const auditedRun=(agentPending||auditPending)?runMetadata:await auditReferenceCandidateSnapshot(target,candidate),preparedDraft=JSON.parse(JSON.stringify(candidate)),preparedRun=JSON.parse(JSON.stringify(auditedRun));
 const proposalCount=await applyCandidateReviewProposals(target,preparedDraft,preparedRun||{});
 await window.setWorkspaceMode?.('manual',false);
 await enterReferencePreparationFrame(target,requestGeneration,sourceTopic,sourcePath);
 manualDrafts[key]=preparedDraft;agentRuns[key]=preparedRun;
 selectedShapeIndex=-1;pointMode=false;negativeMode=false;regionMode='';draft=[];resetPrompts();$('polygonButton').classList.add('selected');refreshPointTools();beginReferenceFinalize('machine_candidate',target);const initialRisk=selectInitialReferenceReviewTarget(manualDrafts[key],agentRuns[key]);refreshShapeList();draw();
 document.dispatchEvent(new CustomEvent('labelagent:reference-draft-loaded',{detail:{frame:target,source:'machine_candidate',candidate_count:candidate.length}}));
 requestAnimationFrame(()=>{$('referenceFinalizeGuide')?.scrollIntoView({behavior:'smooth',block:'center'});$('referenceReviewApproveNext')?.focus({preventScroll:true})});
 toast(agentPending?`已载入 ${candidate.length} 个初始候选，可立即检查轮廓；Agent 正在后台精修，不会写成人工真值`:auditPending?`已载入 ${candidate.length} 个候选，可立即检查轮廓；高风险类别正在后台复核，结果不会自动变成人工真值`:initialRisk?`已载入 ${candidate.length} 个候选${proposalCount?`，其中 ${proposalCount} 个有隔离模型改类建议`:''}，并定位到第一条高风险轮廓；检查通过后点“检查通过并下一个”`:`已载入 ${candidate.length} 个候选，并选中第一条；请逐个检查后再保存真值`);
 const active=()=>referencePreparationIsActive(target,sourceTopic,sourcePath);
 const signalBackgroundComplete=stage=>document.dispatchEvent(new CustomEvent('labelagent:candidate-state-synced',{detail:{frame:target,topic:sourceTopic,candidate_count:(manualDrafts[key]||candidate).length,source:stage}}));
 const finishAudit=items=>auditReferenceCandidateSnapshot(target,items).then(result=>{if(!active()||candidateAuditFullSnapshot(manualDrafts[key])!==candidateAuditFullSnapshot(items))return false;agentRuns[key]=JSON.parse(JSON.stringify(result));updateReferenceFinalizeGuide();refreshShapeList();toast('高风险类别后台复核已完成；候选仍需人工逐个确认');signalBackgroundComplete('reference_background_audit_complete');return true}).catch(error=>{if(!active())return false;updateReferenceFinalizeGuide();toast(`高风险类别后台复核未采用：${error.message}；候选仍保留给人工检查`);signalBackgroundComplete('reference_background_audit_terminal');return false});
 if(agentPending)refineReferenceCandidateSnapshot(target,candidate).then(result=>{if(!active())return false;const untouched=candidateAuditFullSnapshot(manualDrafts[key])===candidateAuditFullSnapshot(candidate);if(!untouched){toast('Agent 后台精修已完成，但您已开始编辑；系统没有覆盖当前人工草稿');signalBackgroundComplete('reference_background_refinement_preserved_manual_edit');return false}manualDrafts[key]=JSON.parse(JSON.stringify(result.candidates));agentRuns[key]=JSON.parse(JSON.stringify(result.agent));selectedShapeIndex=-1;referenceReviewedShapeKeys=new Set();referenceOriginalSweepVisited=false;referenceOpenSetReviewed=false;refreshShapeList();draw();toast(result.mutated?'Agent 后台精修已安全应用；仍需人工逐个确认':'Agent 后台检查完成，初始轮廓无需改动；仍需人工逐个确认');if(result.agent?.candidate_class_audit_status==='pending')return finishAudit(result.candidates);signalBackgroundComplete('reference_background_refinement_complete');return true}).catch(error=>{if(!active())return false;updateReferenceFinalizeGuide();toast(`Agent 后台精修未采用：${error.message}；已保留初始候选供人工检查`);if(candidateAuditFullSnapshot(manualDrafts[key])!==candidateAuditFullSnapshot(candidate)){signalBackgroundComplete('reference_background_refinement_terminal');return false}if(candidateAuditFullSnapshot(candidates[key])===candidateAuditFullSnapshot(candidate))return finishAudit(candidate);signalBackgroundComplete('reference_background_refinement_terminal');return false});
 else if(auditPending)finishAudit(candidate);
};
window.prepareReferenceLegacyTruthDraft=async(target,legacyTruth,provenance={})=>{
 const key=String(target),sourceTopic=topic,sourcePath=loadedDataPath,requestGeneration=frameNavigationGeneration;
 if(!Array.isArray(legacyTruth)||!legacyTruth.length)throw Error('该帧没有可复核的历史标注');
 if(provenance.reusable_as_reference===true)throw Error('该帧已经通过当前人工门禁，请直接采用已核验真值');
 const preparedTruth=JSON.parse(JSON.stringify(legacyTruth)),preparedStatus={...provenance,reusable_as_reference:false,current_human_review_verified:false};
 await window.setWorkspaceMode?.('manual',false);await enterReferencePreparationFrame(target,requestGeneration,sourceTopic,sourcePath);
 shapes[key]=preparedTruth;truthReviewStatus[key]=preparedStatus;
 selectedShapeIndex=-1;pointMode=false;negativeMode=false;regionMode='';draft=[];resetPrompts();$('polygonButton').classList.add('selected');refreshPointTools();refreshShapeList();beginReferenceFinalize('legacy_truth',target);draw();
 toast(`已载入帧 ${target} 的历史标注；请检查/精修并通过当前人工门禁，不需要从零重画`);
};
window.prepareReferenceSamDraft=async target=>{
 const key=String(target),sourceTopic=topic,sourcePath=loadedDataPath,requestGeneration=frameNavigationGeneration;if(shapes[key]!==undefined)throw Error(truthIsVerified(target)?'该帧已有已核验人工真值，请直接复用已有标注':'该帧已有历史标注，请从参考帧助手载入并复核');
 await window.setWorkspaceMode?.('manual',false);await enterReferencePreparationFrame(target,requestGeneration,sourceTopic,sourcePath);
 pointMode=true;negativeMode=false;regionMode='';draft=[];$('polygonButton').classList.remove('selected');resetPrompts();pointPromptAutoRunOnce=true;pointPromptNotice='第 2 步：尚未提交，请现在点击图片中的目标内部';refreshPointTools();beginReferenceFinalize('sam_point_prompt',target);draw();
 focusPointPromptCanvas();
 toast('第 1 步已完成，但 SAM 尚未运行；现在请点击图片中的目标内部');
};
window.prepareReferenceManualDraft=async target=>{
 const key=String(target),sourceTopic=topic,sourcePath=loadedDataPath,requestGeneration=frameNavigationGeneration;if(shapes[key]!==undefined)throw Error(truthIsVerified(target)?'该帧已有已核验人工真值，请直接复用已有标注':'该帧已有历史标注，请从参考帧助手载入并复核');
 await window.setWorkspaceMode?.('manual',false);await enterReferencePreparationFrame(target,requestGeneration,sourceTopic,sourcePath);
 pointMode=false;negativeMode=false;regionMode='';draft=[];resetPrompts();$('polygonButton').classList.add('selected');refreshPointTools();beginReferenceFinalize('manual_polygon',target);draw();
 toast('已进入多边形绘制；保存真值并通过人工门禁后，再点击“设为参考帧”');
};
$('openReferenceButton').onclick=openReferenceFrame;async function loadVideo(data){
 invalidateFrameNavigation();clearReferenceFinalize();manualDrafts={};autoDrafts={};window.referenceFrames=data.reference_frames||{};stopPlayback();setAnnotationMode(false);resetWindows();
 catalog=data.catalog||{};recordDuration=data.duration||0;clockTime=0;playback=null;topics=data.topics;topic=data.topic;loadedDataPath=data.path||$('mcapPath').value||'';loadedFileActivationGeneration=data.file_activation_generation??0;workbenchConfig=data.config||workbenchConfig;configStateSha256=data.config_state_sha256||configStateSha256;
 const resumeFrame=rememberedViewedFrame(topic,loadedDataPath);shapes=data.truth||{};truthReviewStatus=data.truth_review_status||{};candidates=data.candidates||{};agentRuns=data.agent_runs||{};candidateReviewProposals=data.candidate_review_proposals||[];candidateReviewBindings=data.candidate_review_bindings||{};compactGeometryMode=data.compact_geometry===true;truthFrameIndex=new Set((data.truth_frames||Object.keys(shapes)).map(String));candidateCountIndex=data.candidate_counts&&typeof data.candidate_counts==='object'?{...data.candidate_counts}:Object.fromEntries(Object.entries(candidates).map(([name,items])=>[name,Array.isArray(items)?items.length:0]));
 applyInferenceLabelState(data,false);updateReferenceStatus();loading=false;$('mcapPath').value=loadedDataPath;refreshTopics();refreshCandidateSummary();
 restoringFrameNavigation=true;try{await window.setWorkspaceMode?.(window.workspaceMode||'manual',false);await openImportedImage();if(resumeFrame!==null&&frame!==resumeFrame)await showFrame(resumeFrame)}finally{restoringFrameNavigation=false}rememberViewedFrame();toast(resumeFrame!==null?`文件已载入 · 已回到上次查看的第 ${resumeFrame} 帧`:'文件已载入')
}
async function readCurrentDataScope(){const response=await fetch('/api/state/summary',{cache:'no-store'}),data=await response.json();if(!response.ok)throw Error(data.error||'无法确认服务器当前数据集');return data}
function sameLoadedDataScope(data,current){return Boolean(data?.path&&data?.topic&&Number.isInteger(data?.file_activation_generation)&&current?.loaded&&current.path===data.path&&current.topic===data.topic&&current.file_activation_generation===data.file_activation_generation)}
async function loadVerifiedVideoData(data){
 const before=await readCurrentDataScope();if(!sameLoadedDataScope(data,before))throw Error('数据已在另一个窗口切换，本次过期加载结果未显示；请重新选择当前数据');
 await loadVideo(data);const after=await readCurrentDataScope();if(sameLoadedDataScope(data,after))return data;
 try{const response=await fetch('/api/bootstrap',{cache:'no-store'}),current=await response.json();if(!response.ok)throw Error(current.error||'读取当前数据失败');if(current.loaded)await loadVideo(current);else clearLoadedVideo(current.file_activation_generation??0)}catch{clearLoadedVideo()}
 throw Error('加载画面期间另一个窗口切换了数据，工作台已恢复服务器当前作用域；请重新确认后操作');
}
window.loadVideoData=loadVerifiedVideoData;
window.loadVerifiedVideoData=loadVerifiedVideoData;
let fileManagerBusy=false,fileUploadRequest=null;
function fileStatus(message){$('fileManagerStatus').textContent=message}
function setFileUploadControls(active,canCancel=active){$('chooseButton').disabled=active;$('openButton').disabled=active;$('cancelUploadButton').hidden=!canCancel}
function uploadFilesWithProgress(files){
 return new Promise((resolve,reject)=>{
  const form=new FormData();for(const file of files)form.append('files',file,file.name);
  const xhr=new XMLHttpRequest();fileUploadRequest=xhr;xhr.open('POST','/api/uploads');
  xhr.upload.onprogress=event=>{if(!event.lengthComputable){fileStatus(`正在上传 ${files.length} 个文件…`);return}const percent=Math.min(100,Math.round(event.loaded/event.total*100));if(percent>=100){setFileUploadControls(true,false);fileStatus('文件已传完，正在校验并写入列表…')}else fileStatus(`正在上传 ${files.length} 个文件 · ${percent}%`)};
  xhr.onload=()=>{let data={};try{data=JSON.parse(xhr.responseText||'{}')}catch{reject(Error(`批量上传返回异常（HTTP ${xhr.status}）`));return}if(xhr.status<200||xhr.status>=300){reject(Error(data.error||`添加失败（HTTP ${xhr.status}）`));return}resolve(data)};
  xhr.onerror=()=>reject(Error('上传连接中断，请重试；未完成批次不会写入文件列表'));
  xhr.onabort=()=>reject(Error('已取消上传；未完成批次不会写入文件列表'));
  xhr.send(form);
 });
}
function applyFileLibrary(data){if(!Array.isArray(data?.files)||typeof data.library_state_sha256!=='string'||data.library_state_sha256.length!==64)throw Error('文件列表回执缺少完整版本绑定');fileLibraryStateSha256=data.library_state_sha256;renderFiles(data.files)}
function renderFiles(files){
 const list=$('fileList');list.replaceChildren();if(!files.length){const empty=document.createElement('p');empty.className='emptyFileList';empty.textContent='尚未添加图片或 MCAP 文件';list.append(empty)}
 for(const file of files){const identity=datasetIdentityForPath(file.path),row=document.createElement('div');row.className='fileRow';const info=document.createElement('div');info.className='fileInfo';const badge=document.createElement('span');badge.className=`fileDatasetBadge ${identity.kind}`;badge.textContent=identity.badge;const name=document.createElement('strong');name.textContent=identity.kind==='new'||identity.kind==='old'?identity.name:file.name;const path=document.createElement('span');path.textContent=file.path;path.title=file.path;const detail=document.createElement('small');detail.textContent=`${file.name} · ${(file.size/1024/1024).toFixed(1)} MB${file.loaded?' · 当前已加载':''}${!file.available?' · 文件不存在':''}`;info.append(badge,name,path,detail);const load=document.createElement('button');load.textContent=file.loaded?'已加载':'加载';load.disabled=file.loaded||!file.available||fileManagerBusy;load.onclick=async()=>{if(fileManagerBusy)return;fileManagerBusy=true;pendingDatasetLoad={path:file.path,name:file.name};updateDatasetIdentity();fileStatus('正在加载 '+file.name);try{window.cancelReferencePrefetch?.();const data=await api('files/'+file.id+'/load',{});await loadVerifiedVideoData(data);const checked=data.data_compatibility?.ready?` · 首帧兼容检查通过（${data.data_compatibility.width}×${data.data_compatibility.height}）`:'';fileStatus('已加载 '+file.name+checked)}catch(error){fileStatus(error.message)}finally{pendingDatasetLoad=null;updateDatasetIdentity();fileManagerBusy=false;await refreshFiles()}};const remove=document.createElement('button');remove.textContent='删除';remove.disabled=fileManagerBusy;remove.title='从列表移除，保留磁盘文件';remove.onclick=async()=>{if(fileManagerBusy)return;const requestPath=loadedDataPath,requestTopic=topic||null,requestGeneration=loadedFileActivationGeneration,expectedLibrary=fileLibraryStateSha256,expectedEntry=file.entry_sha256,expectedLoaded=file.loaded===true;fileManagerBusy=true;try{if(!expectedLibrary||!expectedEntry)throw Error('文件列表版本尚未载入完成，请刷新后重试');const response=await fetch('/api/files/'+file.id,{method:'DELETE',headers:{'Content-Type':'application/json'},body:JSON.stringify({expected_path:requestPath,expected_topic:requestTopic,expected_file_activation_generation:requestGeneration,expected_library_state_sha256:expectedLibrary,expected_file_entry_sha256:expectedEntry,expected_loaded:expectedLoaded})}),data=await response.json();if(!response.ok)throw Error(data.error||'删除失败');if(data.removed_library_state_sha256!==expectedLibrary||data.removed_file_entry_sha256!==expectedEntry||data.removed_file_id!==file.id||data.file_activation_generation!==requestGeneration||data.previous_path!==requestPath||data.previous_topic!==requestTopic||data.disk_file_deleted!==false)throw Error('文件移除回执与提交时作用域不一致，当前页面保持不变');if(loadedDataPath!==requestPath||(topic||null)!==requestTopic||loadedFileActivationGeneration!==requestGeneration){fileStatus('文件已从服务器列表移除；当前页面已切换，保持现有画面');return}if(data.unloaded){const current=await fetch('/api/bootstrap',{cache:'no-store'}),currentData=await current.json();if(!current.ok)throw Error(currentData.error||'读取当前工作区失败');if(currentData.loaded)await loadVideo(currentData);else clearLoadedVideo(currentData.file_activation_generation??requestGeneration)}applyFileLibrary(data);fileStatus('已从列表移除 '+file.name+'；磁盘原文件已保留')}catch(error){fileStatus(error.message)}finally{fileManagerBusy=false;await refreshFiles()}};row.append(info,load,remove);list.append(row)}
}
async function refreshFiles(){try{const response=await fetch('/api/files',{cache:'no-store'}),data=await response.json();if(!response.ok)throw Error(data.error||'读取文件列表失败');applyFileLibrary(data)}catch(error){fileStatus(error.message)}}
function clearLoadedVideo(currentGeneration=0){invalidateFrameNavigation();clearReferenceFinalize();manualDrafts={};autoDrafts={};candidates={};agentRuns={};truthReviewStatus={};compactGeometryMode=false;truthFrameIndex=new Set();candidateCountIndex={};stopPlayback();setAnnotationMode(false);resetWindows();catalog={};topics={};topic=null;loadedDataPath='';loadedFileActivationGeneration=Number.isInteger(currentGeneration)?currentGeneration:0;playback=null;shapes={};draft=[];frame=0;recordDuration=0;clockTime=0;loading=false;$('frameImage').removeAttribute('src');$('playbackVideo').removeAttribute('src');$('playbackVideo').load();$('primaryWindow').hidden=true;$('frameNumber').value=0;$('frameTotal').textContent='/ 0';refreshMessageList();refreshShapeList();updateGrid();updateControls();updateDatasetIdentity()}
$('chooseButton').onclick=()=>$('mcapFile').click();$('cancelUploadButton').onclick=()=>fileUploadRequest?.abort();$('mcapFile').onchange=async()=>{if(fileManagerBusy)return;const files=[...$('mcapFile').files];if(!files.length)return;fileManagerBusy=true;setFileUploadControls(true,true);try{fileStatus(`准备上传 ${files.length} 个文件…`);const data=await uploadFilesWithProgress(files);fileStatus(`已添加 ${data.added?.length||files.length} 个文件，请点击「加载」`)}catch(error){fileStatus(error.message)}finally{fileUploadRequest=null;fileManagerBusy=false;setFileUploadControls(false,false);$('mcapFile').value='';await refreshFiles()}};
$('openButton').onclick=async()=>{if(fileManagerBusy)return;fileManagerBusy=true;try{await api('files',{path:$('mcapPath').value});fileStatus('已添加，请点击「加载」');$('mcapPath').value=''}catch(error){fileStatus(error.message)}finally{fileManagerBusy=false;await refreshFiles()}};
$('frameNumber').onchange=e=>showFrame(e.target.value);$('playButton').onclick=startPlayback;$('stopButton').onclick=()=>showTime(0);
$('candidateReuseButton').onclick=reuseCurrentFrameCandidates;
$('labelsSave').onclick=async()=>{const requestPath=loadedDataPath,requestTopic=topic||null,requestGeneration=loadedFileActivationGeneration,expectedLabelState=labelStateSha256;try{if(!expectedLabelState)throw Error('类别状态尚未载入完成，请刷新后重试');const data=await api('labels',{labels:$('labelsInput').value.split(','),path:requestPath,topic:requestTopic,file_activation_generation:requestGeneration,expected_label_state_sha256:expectedLabelState});if(loadedDataPath!==requestPath||(topic||null)!==requestTopic||loadedFileActivationGeneration!==requestGeneration||data.path!==requestPath||data.topic!==requestTopic||data.file_activation_generation!==requestGeneration)throw Error('类别保存期间数据文件、相机或载入代次已变化，拒绝更新当前页面');applyInferenceLabelState(data);currentDescription=null;toast('类别已保存；参考描述需要重新确认')}catch(e){toast(e.message)}};
$('autoLabelsAll').onclick=()=>{for(const input of $('autoLabelOptions').querySelectorAll('input'))input.checked=true};
$('autoLabelsReference').onclick=()=>{for(const input of $('autoLabelOptions').querySelectorAll('input'))input.checked=referenceOnlyLabels.includes(input.value)};
$('autoLabelsSave').onclick=async()=>{const button=$('autoLabelsSave'),selected=[...$('autoLabelOptions').querySelectorAll('input:checked')].map(input=>input.value),requestPath=loadedDataPath,requestTopic=topic||null,requestGeneration=loadedFileActivationGeneration,expectedLabelState=labelStateSha256;button.disabled=true;try{if(!expectedLabelState)throw Error('类别状态尚未载入完成，请刷新后重试');const data=await api('auto_labels',{labels:selected,path:requestPath,topic:requestTopic,file_activation_generation:requestGeneration,expected_label_state_sha256:expectedLabelState});if(loadedDataPath!==requestPath||(topic||null)!==requestTopic||loadedFileActivationGeneration!==requestGeneration||data.path!==requestPath||data.topic!==requestTopic||data.file_activation_generation!==requestGeneration)throw Error('推理类别保存期间数据文件、相机或载入代次已变化，拒绝更新当前页面');applyInferenceLabelState({...data,labels:data.project_labels});currentDescription=null;toast('推理类别已应用；请在参考帧重新生成真值描述')}catch(e){toast(e.message)}finally{button.disabled=false}};
async function lockReferenceFrame(target=frame,targetTopic=topic){const targetPath=loadedDataPath,targetGeneration=loadedFileActivationGeneration,key=String(target),expectedTruth=candidateAuditFullSnapshot(shapes[key]),expectedHash=truthReviewStatus[key]?.truth_sha256,originFrame=frame;const assertScope=()=>{if(loadedDataPath!==targetPath||loadedFileActivationGeneration!==targetGeneration||topic!==targetTopic||frame!==originFrame)throw Error('锁定参考帧期间文件载入代次、数据作用域或帧已变化，拒绝更新当前工作区')};assertScope();await api('reference_frame',{frame:target,expected_topic:targetTopic,expected_path:targetPath,expected_file_activation_generation:targetGeneration});assertScope();const readback=await readCompactFrameStateSnapshot(target,targetTopic,targetPath,targetGeneration);assertScope();const readbackTruth=readback.truth?.[key],review=readback.truth_review_status?.[key];if(readback.reference_frames?.[targetTopic]?.[0]!==target||readback.file_activation_generation!==targetGeneration||!Array.isArray(readbackTruth)||!readbackTruth.length||candidateAuditFullSnapshot(readbackTruth)!==expectedTruth||!expectedHash||review?.truth_sha256!==expectedHash||review?.reusable_as_reference!==true)throw Error('服务器读回未同时确认真值内容、文件载入代次、话题、帧、人工回执和参考帧，已停止宣称成功');window.referenceFrames=readback.reference_frames||{};shapes[key]=readbackTruth;truthReviewStatus[key]=review;updateReferenceStatus();return readback}
$('referenceButton').onclick=async()=>{try{await lockReferenceFrame();toast('服务器读回通过，已锁定参考帧，后续 AI 不会修改此帧')}catch(e){toast(e.message)}};
async function saveCurrentTruth(candidateOnly=false){
 const targetFrame=frame,targetTopic=topic,targetPath=loadedDataPath,targetGeneration=loadedFileActivationGeneration,key=String(targetFrame);
 if(window.workspaceMode==='auto'&&shapes[key]!==undefined&&candidates[key]!==undefined)throw Error('当前显示的是评测候选；人工真值已存在，禁止用候选覆盖');
 if(candidateOnly&&(window.workspaceMode!=='auto'||shapes[key]!==undefined||!Object.hasOwn(candidates,key)))throw Error('当前帧没有待保存的自动候选');
 if(!annotationMode)throw Error('请先在主画面打开待复核的图像帧');
 if(loading||playing)throw Error('请暂停并等待原始帧加载后再保存真值');
 const savedShapes=JSON.parse(JSON.stringify(selectedShapes())),savedPayload=candidateAuditFullSnapshot(savedShapes);
 if(typeof window.commitHumanReviewedTruth!=='function')throw Error('安全人工复核门禁未加载，已阻止直接保存真值');
 const candidateSource=(referencePreparationIntent?.source==='sam_point_prompt'||(generatedShapeFrame===targetFrame&&generatedShapeTopic===targetTopic))?'sam_point_prompt':null;
 const assertSaveScope=()=>{if(frame!==targetFrame||topic!==targetTopic||loadedDataPath!==targetPath||loadedFileActivationGeneration!==targetGeneration)throw Error('人工确认期间文件载入代次、数据作用域或帧已切换；拒绝更新当前工作区')};
 const data=await window.commitHumanReviewedTruth({frame:targetFrame,topic:targetTopic,fileActivationGeneration:targetGeneration,shapes:savedShapes,candidateOnly,candidateSource});assertSaveScope();if(data?.rejected){if(data.review&&(data.review.file!==targetPath||data.review.topic!==targetTopic||Number(data.review.frame)!==Number(targetFrame)))throw Error('驳回回执作用域不匹配');if(frame===targetFrame&&topic===targetTopic&&loadedDataPath===targetPath&&loadedFileActivationGeneration===targetGeneration){delete manualDrafts[key];delete autoDrafts[key];delete candidates[key];candidateCountIndex[key]=0;refreshShapeList();updateReferenceStatus();draw()}toast('机器候选已驳回并进入修正队列，未写入人工真值');return data}if(!data||data.review?.frame===undefined||Number(data.review.frame)!==Number(targetFrame)||data.review.topic!==targetTopic||data.review.file!==targetPath||data.readback?.verified!==true||data.readback?.truth_sha256_matches!==true||!data.review.truth_sha256||(data.readback.truth_sha256!==undefined&&data.readback.truth_sha256!==data.review.truth_sha256))throw Error('人工复核回执未绑定原文件、话题、帧及真值内容，已拒绝本地更新');
 const stateReadback=await readCompactFrameStateSnapshot(targetFrame,targetTopic,targetPath,targetGeneration);assertSaveScope();
 const readbackTruth=stateReadback.truth?.[key],review=stateReadback.truth_review_status?.[key],serverPath=stateReadback.path,expectedTruthHash=data.review.truth_sha256;
 if(stateReadback.topic!==targetTopic||serverPath!==targetPath||candidateAuditFullSnapshot(readbackTruth)!==savedPayload||review?.reusable_as_reference!==true||!expectedTruthHash||review?.truth_sha256!==expectedTruthHash||data.review.truth_sha256!==expectedTruthHash||data.review.topic!==stateReadback.topic||Number(data.review.frame)!==Number(targetFrame))throw Error('真值内容、文件/话题/帧绑定或人工回执读回不一致，已拒绝本地提交');
 shapes[key]=savedShapes;truthFrameIndex.add(key);delete manualDrafts[key];delete autoDrafts[key];delete candidates[key];candidateCountIndex[key]=0;
 truthReviewStatus[key]=review;window.referenceFrames=stateReadback.reference_frames||window.referenceFrames;
 updateReferenceStatus();if(typeof document!=='undefined'&&typeof CustomEvent==='function')document.dispatchEvent(new CustomEvent('labelagent:human-truth-saved',{detail:{frame:targetFrame,topic:targetTopic,path:targetPath,review:JSON.parse(JSON.stringify(review))}}));toast(`人工门禁读回通过，已保存 ${data.saved} 个真值标注`);return data;
}
$('truthSave').onclick=()=>saveCurrentTruth().catch(e=>toast(e.message));
async function finalizeReferencePreparation(){const intent=referencePreparationIntent;if(!intent||referenceFinalizeBusy)return;if(intent.topic!==topic||intent.frame!==frame){try{await openFrameForReferencePreparation(intent.frame);updateReferenceFinalizeGuide()}catch(error){referenceFinalizeNotice=error.message;updateReferenceFinalizeGuide()}return}updateReferenceFinalizeGuide();if($('referenceFinalizePrimary').disabled){toast('请先完成逐个轮廓检查及待处理提醒；机器审核不能代替人工检查');return}referenceFinalizeBusy=true;referenceFinalizeNotice='正在进入人工确认…';updateReferenceFinalizeGuide();try{if(!truthIsVerified(intent.frame)){const data=await saveCurrentTruth();if(data?.rejected){clearReferenceFinalize();return}}referenceFinalizeNotice='人工真值与回执已核验，正在锁定参考帧…';updateReferenceFinalizeGuide();await lockReferenceFrame(intent.frame,intent.topic);clearStoredReferenceReviewProgress(selectedShapes(),intent);clearReferenceFinalize();toast(`帧 ${intent.frame} 已经人工确认并锁定为参考帧`)}catch(error){referenceFinalizeNotice=truthIsVerified(intent.frame)?`真值已保存；参考帧尚未锁定：${error.message}`:error.message;updateReferenceFinalizeGuide();toast(error.message)}finally{referenceFinalizeBusy=false;updateReferenceFinalizeGuide()}}
$('referenceFinalizePrimary').onclick=finalizeReferencePreparation;
$('referenceFinalizeCancel').onclick=()=>{clearReferenceFinalize();toast('已退出参考帧收尾；当前草稿仍保留')};
$('openConfigDialog').onclick=()=>$('config').showModal();
$('closeConfigDialog').onclick=()=>$('config').close();
$('openFilesDialog').onclick=()=>{$('fileManager').showModal();refreshFiles()};
$('datasetIdentitySwitch').onclick=()=>{$('fileManager').showModal();refreshFiles()};
$('closeFilesDialog').onclick=()=>$('fileManager').close();
$('doneFilesButton').onclick=()=>$('fileManager').close();
$('configSave').onclick=async()=>{const requestPath=loadedDataPath,requestTopic=topic||null,requestGeneration=loadedFileActivationGeneration,expectedConfigState=configStateSha256;try{if(!expectedConfigState)throw Error('配置版本尚未载入完成，请刷新后重试');const data=await api('config',{llm_url:$('llmUrl').value,llm_model:$('llmModel').value,llm_key:$('llmKey').value,expected_path:requestPath,expected_topic:requestTopic,expected_file_activation_generation:requestGeneration,expected_config_state_sha256:expectedConfigState});if(loadedDataPath!==requestPath||(topic||null)!==requestTopic||loadedFileActivationGeneration!==requestGeneration||data.path!==requestPath||data.topic!==requestTopic||data.file_activation_generation!==requestGeneration||data.previous_config_state_sha256!==expectedConfigState||typeof data.config_state_sha256!=='string')throw Error('配置保存期间数据文件、相机、载入代次或配置版本已变化，拒绝更新当前页面');workbenchConfig=data.config||workbenchConfig;configStateSha256=data.config_state_sha256;currentDescription=null;$('config').close();toast('配置已保存；参考描述需要重新确认')}catch(e){toast(e.message)}};
let currentDescription=null,descriptionRequest=null;
async function regenerateReferenceDescription(targetFrame=frame){
 const selectedFrame=Number(targetFrame),targetTopic=topic,targetPath=loadedDataPath;
 if(!Number.isInteger(selectedFrame)||selectedFrame<0)throw Error('参考帧号无效，无法生成场景描述');
 const requestKey=`${targetPath}|${targetTopic}|${selectedFrame}`;
 if(descriptionRequest){
  if(descriptionRequest.key===requestKey)return descriptionRequest.promise;
  throw Error('正在为另一个参考帧生成描述，请等待当前请求完成');
 }
 const promise=(async()=>{
  const result=await api('describe',{frame:selectedFrame});
  if(result?.stopped||result?.description_saved===false)throw Error('场景描述生成已停止，尚未保存');
  if(topic!==targetTopic||loadedDataPath!==targetPath)throw Error('场景描述已生成，但当前文件或相机已经切换；请在当前数据上重新检查');
  currentDescription=result;
  if(typeof CustomEvent==='function')document.dispatchEvent(new CustomEvent('labelagent:description-ready',{detail:{frame:selectedFrame,topic:targetTopic,path:targetPath}}));
  return result;
 })();
 descriptionRequest={key:requestKey,promise};
 try{return await promise}finally{if(descriptionRequest?.promise===promise)descriptionRequest=null}
}
window.regenerateReferenceDescription=regenerateReferenceDescription;
$('describeButton').onclick=async()=>{const button=$('describeButton');if(button.disabled)return;button.disabled=true;button.textContent='生成中…';toast('正在生成真值描述…');try{await regenerateReferenceDescription(frame);toast('描述已生成，可以开始自动标注')}catch(e){toast(e.message)}finally{button.disabled=false;button.textContent='生成真值描述'}};
 const agentPhase={preflight:'检查模型与数据是否就绪',submitting:'请求已提交，等待后台确认',claiming:'后台已接收任务',starting:'准备任务',discovering:'VLM 识别当前帧可见类别',discovering_retry:'VLM 正在重试标签识别',describing:'VLM 生成真值描述',loading:'读取图像',generating:'SAM 3 生成候选',generated:'候选已生成',observing:'VLM 观察图像和候选',agent_refining:'VLM 后台精修候选',candidate_class_audit:'隔离复核疑似箱体栈板',vlm_waiting:'VLM 正忙，任务正在排队等待',editing:'执行标注工具',edited:'工具执行结束',finished:'本帧结束',retrying:'服务短暂异常，正在自动重试本帧',saved:'候选已保存',completed:'本次处理结束',stopping:'正在停止',stopped:'已停止',failed:'运行失败'};
const agentStop={done:'Agent 流程已结束（不代表正确）',needs_review:'旧记录：请求人工复核',max_rounds:'达到最大复查轮数',no_progress:'没有有效修改',cycle:'检测到循环修改',error:'模型或工具调用失败'};
 const agentProgressStage={preflight:0,submitting:0,claiming:0,starting:0,discovering:0,discovering_retry:0,describing:0,loading:0,generating:1,generated:1,observing:2,agent_refining:2,candidate_class_audit:3,vlm_waiting:2,editing:3,edited:3,finished:4,retrying:1,saved:4,completed:4,stopping:4,stopped:4,failed:4};
 const agentProgressWeight={preflight:0.01,submitting:0.02,claiming:0.03,starting:0.04,discovering:0.06,discovering_retry:0.08,describing:0.10,loading:0.12,generating:0.22,generated:0.38,observing:0.58,agent_refining:0.58,candidate_class_audit:0.90,vlm_waiting:0.58,editing:0.74,edited:0.86,finished:0.94,retrying:0.18,saved:1};
 const agentPhaseTypicalSeconds={discovering:16,discovering_retry:16,describing:16,observing:34,agent_refining:34,candidate_class_audit:10};
 const agentPhaseProgressCeiling={discovering:0.20,discovering_retry:0.20,describing:0.20,observing:0.72,agent_refining:0.72,candidate_class_audit:0.94};
let progressTaskId=null,progressHighWater=0,runStatusConnectionMessage='',runStatusScopeMismatch=null;
let progressFetching=false,progressFetchPromise=null;
$('downloadDebug').onclick=async()=>{
 const button=$('downloadDebug');if(button.disabled)return;button.disabled=true;
 try{
  const response=await fetch('/api/run/debug',{cache:'no-store'});
  if(!response.ok)throw Error('无法下载调用记录');
  const data=await response.json(),blob=new Blob([JSON.stringify(data.trace||[])],{type:'application/json'}),url=URL.createObjectURL(blob),link=document.createElement('a');
  link.href=url;link.download='autolabel-model-debug.json';link.click();setTimeout(()=>URL.revokeObjectURL(url),1000);
 }catch(error){toast(error.message)}finally{button.disabled=false}
};
let latestAgentJob={status:'idle'},runRequestStartedAt=null,runRequestPhase='preflight',runRequestFrame=null,runRequestCount=null,runRequestMaxRounds=6,runStatusFailureCount=0;
function pendingRunResult(job=latestAgentJob){const id=job?.result_task_id||job?.task_id;return !!(job?.result_available&&id&&!runResultHandled.has(id))}
function markRunResultHandled(taskId){if(taskId)runResultHandled.add(taskId)}
function runJobIdentity(job=latestAgentJob){return job?.task_id||job?.result_task_id||null}
function competingRunJob(taskId){const currentId=runJobIdentity(latestAgentJob);return !!(currentId&&currentId!==taskId&&(['running','stopping'].includes(latestAgentJob?.status)||latestAgentJob?.result_available))}
function pendingRetry(){return retryFrames.length>0&&typeof retryScope?.taskId==='string'&&retryScope.taskId&&retryScope.file===loadedDataPath&&retryScope.topic===topic}
async function recoverRunResult(job=latestAgentJob,{automatic=false}={}){
 const taskId=job?.result_task_id||job?.task_id;if(!taskId||runResultHandled.has(taskId))return false;
 if(automatic&&runResultAutoRestoreDeferred.has(taskId))return false;
 if(!automatic)runResultAutoRestoreDeferred.delete(taskId);
 const recoveryPath=loadedDataPath,recoveryTopic=topic,recoveryFrame=frame,recoveryNavigationGeneration=frameNavigationGeneration,recoveryWorkspaceMode=window.workspaceMode;
 const response=await fetch('/api/run/result?task_id='+encodeURIComponent(taskId),{cache:'no-store'}),data=await response.json();
 if(!response.ok)throw Error(data.error||'无法恢复上次运行结果');
 if(data.task_id!==taskId)throw Error('上次运行结果的任务号不匹配，已拒绝更新当前页面');
 if(data.file!==recoveryPath||data.topic!==recoveryTopic)throw Error('上次结果不属于请求恢复时的数据文件和图像源');
 const operatorChanged=loadedDataPath!==recoveryPath||topic!==recoveryTopic||frameNavigationGeneration!==recoveryNavigationGeneration||window.workspaceMode!==recoveryWorkspaceMode;
 if(competingRunJob(taskId)||operatorChanged){runResultAutoRestoreDeferred.add(taskId);renderRunButton();toast(competingRunJob(taskId)?'新的自动标注任务已经开始；旧候选仍保留在原帧，当前页面保持不变':'恢复期间您已切换数据、帧或工作模式；旧结果仍保留，当前页面保持不变');return false}
 const first=(data.frames||[]).find(item=>!item.failed&&item.remaining>0)||(data.frames||[]).find(item=>item.remaining>0);
 const firstKey=first?String(first.frame):null,hadFirstCount=firstKey!==null&&Object.prototype.hasOwnProperty.call(candidateCountIndex,firstKey),firstCountBefore=firstKey===null?undefined:candidateCountIndex[firstKey];
 const restoreFirstCount=()=>{if(firstKey===null)return;if(hadFirstCount)candidateCountIndex[firstKey]=firstCountBefore;else delete candidateCountIndex[firstKey];refreshCandidateSummary()};
 const opened=first?await openCandidateFrame(first.frame,data.topic):true;
 if(first&&!opened){restoreFirstCount();runResultAutoRestoreDeferred.add(taskId);if((latestAgentJob?.result_task_id||latestAgentJob?.task_id)===taskId)latestAgentJob.result_available=true;renderRunButton();toast(`帧 ${first.frame} 的结果已保留；您已切换到其他位置，点击“查看上次结果”可再次打开`);return false}
 if(competingRunJob(taskId)){
  const sameScope=loadedDataPath===recoveryPath&&topic===recoveryTopic,restored=sameScope&&(frame===recoveryFrame||await showFrame(recoveryFrame));
  if(restored&&window.workspaceMode!==recoveryWorkspaceMode)await window.setWorkspaceMode?.(recoveryWorkspaceMode,false);
  restoreFirstCount();
  runResultAutoRestoreDeferred.add(taskId);renderRunButton();toast(restored?'新的自动标注任务已经开始；已返回您原来的帧和工作模式，旧候选仍保留':'新的自动标注任务已经开始；旧候选仍保留在原帧，当前任务状态不变');return false
 }
 for(const item of data.frames||[])if(Number.isInteger(item.frame))candidateCountIndex[String(item.frame)]=Math.max(0,Number(item.remaining)||0);
 refreshCandidateSummary();
 runResultAutoRestoreDeferred.delete(taskId);
 markRunResultHandled(taskId);if((latestAgentJob?.result_task_id||latestAgentJob?.task_id)===taskId)latestAgentJob.result_available=false;renderRunButton();
 const failed=data.failed_frames||[],pending=(data.status==='running'||data.status==='failed')?(data.pending_frames||[]):[];
 retryFrames=[...new Set([...pending,...failed])];retryScope=retryFrames.length?{file:data.file,topic:data.topic,taskId}:null;retryMode=pending.length?'resume':failed.length?'failed':'';renderRunButton();
 toast(pending.length?`已恢复 ${data.completed} 帧结果，另有 ${retryFrames.length} 帧待继续；已完成帧不会重跑`:first?(failed.length?`已恢复 ${data.completed} 帧结果，其中 ${failed.length} 帧失败；旧候选已保留`:`已恢复 ${data.completed} 帧结果，请继续人工复核`):'上次任务没有可显示的候选');
 return true;
}
function formatAgentDuration(seconds){
 const value=Math.max(0,Math.floor(seconds||0)),hours=Math.floor(value/3600),minutes=Math.floor(value%3600/60),rest=value%60;
 if(hours)return `${hours} 小时 ${minutes} 分`;
 if(minutes)return `${minutes} 分 ${rest} 秒`;
 return `${rest} 秒`;
}
function estimatedAgentPhaseWeight(phase,phaseElapsed){
 const base=agentProgressWeight[phase]||0,typical=agentPhaseTypicalSeconds[phase],ceiling=agentPhaseProgressCeiling[phase];
 if(!typical||!Number.isFinite(ceiling)||ceiling<=base)return base;
 const fraction=Math.min(0.92,Math.max(0,Number(phaseElapsed)||0)/typical);
 return base+(ceiling-base)*fraction;
}
function renderAgentProgressPanel(job,elapsed){
 const status=job.status||'idle',running=status==='running'||status==='stopping';
 const total=Math.max(0,Number(job.total)||0),completed=Math.min(total,Math.max(0,Number(job.completed)||0));
 const phase=job.phase||(status==='completed'?'completed':status==='failed'?'failed':status==='stopped'?'stopped':'starting');
 const phaseElapsed=Math.max(0,Number(job.phase_elapsed_seconds)||0);
 const currentWeight=running&&total?estimatedAgentPhaseWeight(phase,phaseElapsed):0;
 const estimated=total?Math.round((completed+currentWeight)/total*100):0;
 const taskIdentity=job.task_id||null;
 if(taskIdentity!==progressTaskId){progressTaskId=taskIdentity;progressHighWater=0}
 let percent=(status==='completed'||total&&completed>=total)?100:total?Math.min(99,Math.max(running?1:0,estimated)):0;
 if(running){progressHighWater=Math.max(progressHighWater,percent);percent=progressHighWater}
 const stateText=phase==='preflight'?'检查中':phase==='submitting'?'已提交':({idle:'等待开始',running:'推理中',stopping:'停止中',completed:'已完成',stopped:'已停止',failed:'运行失败'}[status]||status);
 const title=status==='idle'?'尚未开始标注推理':agentPhase[phase]||'正在处理';
 const latest=job.events?.at(-1),details=[];
 if(job.frame!==null&&job.frame!==undefined)details.push(`第 ${job.frame} 帧`);
 if(phase==='preflight')details.push('正在确认 SAM、VLM 和数据状态');
 if(phase==='submitting')details.push('请求已提交，请勿重复点击');
 if(phase==='vlm_waiting')details.push(`共享模型正在处理其他请求，系统将自动重试${job.retry_in_seconds?`（约 ${job.retry_in_seconds} 秒后）`:''}`);
 if(phase==='candidate_class_audit')details.push(`正在隔离检查疑似箱体候选 ${Number(job.audit_completed)||0}/${Number(job.audit_total)||0}`);
 if(latest?.label)details.push(`类别：${latest.label}`);
 const typicalSeconds=agentPhaseTypicalSeconds[phase];
 if(running&&typicalSeconds)details.push(`本阶段已进行 ${formatAgentDuration(phaseElapsed)} · 当前 4090 基线通常约 ${formatAgentDuration(typicalSeconds)}`);
 if(running&&phaseElapsed>=60)details.push(`该阶段已持续 ${formatAgentDuration(phaseElapsed)}，后台仍在线，可安全停止`);
 if(status==='failed'&&job.error)details.push(job.error);
 if(runStatusConnectionMessage)details.push(runStatusConnectionMessage);
 if(!details.length)details.push(status==='completed'?`已完成 ${completed} 帧`:status==='idle'?'点击“开始自动标注”后，这里会显示实时进度。':`已完成 ${completed}/${total} 帧`);
 const state=$('runProgressState');state.textContent=stateText;state.className=`runProgressState ${status}`;
 $('runProgressTitle').textContent=title;
 $('runProgressDetail').textContent=details.join(' · ');
 $('runProgressPercent').textContent=status==='idle'?'待开始':running?`约 ${percent}%`:`${percent}%`;
 $('runProgressFill').style.width=`${percent}%`;
 const progressBar=$('runProgressBar');
 progressBar.classList.toggle('running',running);
 progressBar.classList.toggle('longPhase',running&&phaseElapsed>=60);
 progressBar.classList.toggle('failed',status==='failed');
 progressBar.setAttribute('aria-valuenow',String(percent));
 $('runProgressFrames').textContent=`${completed} / ${total}`;
 $('runProgressFrame').textContent=job.frame===null||job.frame===undefined?'—':String(job.frame);
 $('runProgressRound').textContent=job.round?`${job.round} / ${Math.max(1,Number(job.max_rounds)||6)}`:'—';
 $('runProgressShapes').textContent=String(Math.max(0,Number(job.shape_count)||0));
 $('runProgressElapsed').textContent=formatAgentDuration(elapsed);
 const activeStage=agentProgressStage[phase]??0,finished=status==='completed';
 for(const [index,item] of [...$('runProgressStages').children].entries()){
  item.classList.toggle('active',running&&index===activeStage);
  item.classList.toggle('done',finished||index<activeStage||(index===4&&phase==='saved'));
  item.classList.toggle('failed',status==='failed'&&index===activeStage);
 }
}
function renderRunButton(){
 const button=$('runButton');
 const pending=runRequestStartedAt!==null&&(!latestAgentJob.started_at||latestAgentJob.started_at<runRequestStartedAt-1);
 const job=pending?{status:'running',phase:runRequestPhase,started_at:runRequestStartedAt,phase_started_at:runRequestStartedAt,phase_elapsed_seconds:Math.max(0,Date.now()/1000-runRequestStartedAt),frame:runRequestFrame??Number($('frameNumber').value),total:runRequestCount??Number($('taskCount').value),max_rounds:runRequestMaxRounds,completed:0,shape_count:0}:latestAgentJob;
 const stopping=job.status==='stopping',running=job.status==='running'||stopping;
 const elapsed=job.started_at?Math.max(0,Math.floor((running?Date.now()/1000:(job.events?.at(-1)?.time||job.started_at))-job.started_at)):null;
  const recoverable=pendingRunResult(job),retryable=pendingRetry();
  const launchingRefinement=typeof deferredRunRefinementActive!=='undefined'&&deferredRunRefinementActive&&!running;
  const scopeBlocked=typeof runStatusScopeMismatch!=='undefined'&&runStatusScopeMismatch!==null;
  button.disabled=pending||stopping||launchingRefinement||scopeBlocked;
  button.textContent=(scopeBlocked?'数据已在其他窗口切换 · 请重新载入':pending?'启动中…':stopping?'正在停止…':running?'停止自动标注':launchingRefinement?'正在启动后台精修…':recoverable?'查看上次结果':retryable?(retryMode==='resume'?`继续 ${retryFrames.length} 个未完成帧`:`重试 ${retryFrames.length} 个失败帧`):'开始自动标注')+(running&&!pending&&!scopeBlocked?` · 第 ${job.round||0}/${Math.max(1,Number(job.max_rounds)||6)} 轮`: '')+(elapsed===null||scopeBlocked?'':` · ${elapsed} 秒`);
 button.setAttribute('aria-busy',String(pending||running||launchingRefinement));
 renderAgentProgressPanel(job,elapsed||0);
}
function renderAgentProgress(job){const changed=latestAgentJob?.status!==job.status||latestAgentJob?.phase!==job.phase||latestAgentJob?.frame!==job.frame;latestAgentJob=job;renderRunButton();if(changed)document.dispatchEvent(new CustomEvent('labelagent:job-status',{detail:job}))}
async function fetchAgentStatus(){
 const controller=typeof AbortController==='function'?new AbortController():null,timer=controller?setTimeout(()=>controller.abort(),8000):null;
 try{return await fetch('/api/run/status',{cache:'no-store',...(controller?{signal:controller.signal}:{})})}
 catch(error){if(error?.name==='AbortError')throw Error('状态读取超过 8 秒');throw error}
 finally{if(timer)clearTimeout(timer)}
}
async function refreshAgentProgress(){
 if(progressFetchPromise)return progressFetchPromise;
 progressFetching=true;
 const requestPath=typeof loadedDataPath==='undefined'?'':loadedDataPath,requestTopic=typeof topic==='undefined'?'':topic;
 progressFetchPromise=(async()=>{try{const response=await fetchAgentStatus();if(!response.ok)throw Error(`HTTP ${response.status}`);const job=await response.json(),currentPath=typeof loadedDataPath==='undefined'?'':loadedDataPath,currentTopic=typeof topic==='undefined'?'':topic;const currentScopeReady=Boolean(currentPath&&currentTopic),requestScopeChanged=currentPath!==requestPath||currentTopic!==requestTopic,serverScopeMismatch=currentScopeReady&&(job.selected_file!==currentPath||job.selected_topic!==currentTopic);if(requestScopeChanged||serverScopeMismatch){runStatusScopeMismatch={file:job.selected_file||'',topic:job.selected_topic||'',current_file:currentPath,current_topic:currentTopic};runStatusConnectionMessage='数据集已在另一个窗口切换，当前页面不会沿用其任务状态；请重新载入需要操作的数据';latestAgentJob={status:'idle'};runStatusFailureCount=0;$('runButton').title=runStatusConnectionMessage;renderRunButton();scheduleRunStatusPoll(250);return false}runStatusScopeMismatch=null;runStatusConnectionMessage='';renderAgentProgress(job);runStatusFailureCount=0;$('runButton').removeAttribute('title');if(bootstrapComplete&&runRequestStartedAt===null&&pendingRunResult(job))await recoverRunResult(job,{automatic:true});if(bootstrapComplete&&workspaceRestoreState!=='ready')markWorkspaceReady();return true}catch(error){runStatusFailureCount++;runStatusConnectionMessage=runStatusFailureCount>=3?`连续无法读取推理状态：${error.message}；后台任务不会重复提交`:`暂时无法读取推理状态：${error.message}，正在自动恢复`;$('runButton').title=runStatusConnectionMessage;renderRunButton();return false}finally{progressFetching=false;progressFetchPromise=null}})();
 return progressFetchPromise;
}
async function ensureAutoLabelReadiness(requestedFrames,startFrame,count){
 const [response,workflowResponse,dataResponse]=await Promise.all([
  fetch('/api/model_readiness',{cache:'no-store'}),
  fetch('/api/run/readiness',{cache:'no-store'}),
  fetch('/api/run/data-readiness',{method:'POST',headers:{'Content-Type':'application/json'},
   body:JSON.stringify(requestedFrames?{frames:requestedFrames}:{frame:startFrame,count})}),
 ]);
 if(!response.ok)throw Error(`模型就绪检查失败（HTTP ${response.status}）`);
 if(!workflowResponse.ok)throw Error(`标注流程检查失败（HTTP ${workflowResponse.status}）`);
 if(!dataResponse.ok)throw Error(`数据兼容检查失败（HTTP ${dataResponse.status}）`);
  const readiness=await response.json();let workflow=await workflowResponse.json();const data=await dataResponse.json();
 const failures=[];
 if(!readiness.sam?.ready)failures.push(`SAM：${readiness.sam?.error||readiness.sam?.status||'未就绪'}`);
 if(!readiness.vlm?.ready)failures.push(`VLM：${readiness.vlm?.error||readiness.vlm?.status||'未就绪'}`);
 if(!data.ready)failures.push(`数据：${data.message||'当前帧无法读取'}`);
 if(data.ready&&data.storage?.warning)toast(data.storage.message||'保存空间偏低，建议只运行少量帧');
  const descriptionRecoverable=!workflow.ready&&['description_missing','description_stale'].includes(workflow.code);
  if(descriptionRecoverable&&!failures.length){
   if(!Number.isInteger(workflow.reference_frame))throw Error('尚未启动推理。参考描述需要更新，但服务没有返回有效参考帧');
   runRequestPhase='describing';renderRunButton();toast('参考描述需要更新，正在自动生成；完成后会继续推理');
   try{
    await window.regenerateReferenceDescription(workflow.reference_frame);
    const refreshed=await fetch('/api/run/readiness',{cache:'no-store'});
    if(!refreshed.ok)throw Error(`重新检查失败（HTTP ${refreshed.status}）`);
    workflow=await refreshed.json();
   }catch(error){throw Error(`尚未启动推理。参考描述自动生成失败：${error.message}`)}
   finally{runRequestPhase='preflight';renderRunButton()}
  }
  if(!workflow.ready)failures.push(`流程：${workflow.message||'参考数据未就绪'}`);
 if(failures.length)throw Error(`尚未启动推理。${failures.join('；')}`);
 return {models:readiness,workflow,data};
}
$('runButton').onclick=async()=>{
 const button=$('runButton');
 if(latestAgentJob.status==='running'){
  const stopTaskId=latestAgentJob.task_id;
  if(!stopTaskId){toast('尚未确认本轮任务编号，请稍候再停止');scheduleRunStatusPoll(0);return}
  button.disabled=true;button.textContent='正在停止…';
  try{const stopped=await api('run/stop',{expected_task_id:stopTaskId});if(stopped.task_id!==stopTaskId)throw Error('停止回执与当前任务不一致，未确认停止');toast('已请求停止当前自动标注任务')}
  catch(e){toast(e.message);button.disabled=false;renderRunButton();scheduleRunStatusPoll(0)}
  return;
 }
  if(pendingRunResult()){
   button.disabled=true;button.textContent='正在恢复结果…';
   try{await recoverRunResult()}catch(e){toast(e.message)}finally{button.disabled=false;renderRunButton()}
   return;
  }
 if(runRequestStartedAt!==null)return;
 let requestedFrames=null,retryTaskId=null,staleRetryRejected=false;
 try{
   requestedFrames=pendingRetry()?[...retryFrames]:null;
   retryTaskId=requestedFrames?retryScope.taskId:null;
   const startFrame=requestedFrames?.[0]??Number($('frameNumber').value),count=requestedFrames?.length??Number($('taskCount').value),maxRounds=Number($('reviewRounds').value),jobTopic=topic,jobFile=loadedDataPath,jobNavigationGeneration=frameNavigationGeneration,jobWorkspaceMode=window.workspaceMode;
  if(!Number.isInteger(startFrame)||startFrame<0||startFrame>=(topics[topic]?.count||0))throw Error('请选择有效的当前帧');
  if(!Number.isInteger(count)||count<1||count>100)throw Error('处理帧数须为 1～100');
  if(!Number.isInteger(maxRounds)||maxRounds<1||maxRounds>12)throw Error('复查轮数须为 1～12');
   runRequestStartedAt=Date.now()/1000;runRequestPhase='preflight';runRequestFrame=startFrame;runRequestCount=count;runRequestMaxRounds=maxRounds;renderRunButton();scheduleRunStatusPoll(250);
  await ensureAutoLabelReadiness(requestedFrames,startFrame,count);
  if(requestedFrames&&(!pendingRetry()||retryScope.taskId!==retryTaskId))throw Error('预检期间重试来源已经更新，本次旧重试未启动');
  if(topic!==jobTopic||loadedDataPath!==jobFile||frameNavigationGeneration!==jobNavigationGeneration||window.workspaceMode!==jobWorkspaceMode)throw Error('预检期间数据集、相机、当前帧或工作模式已经切换，本次自动标注未启动；请在当前画面重新开始');
  runRequestPhase='submitting';renderRunButton();
  stopPlayback();
    const comparison=requestedFrames?requestedFrames.some(item=>truthFrameIndex.has(String(item))):truthFrameIndex.has(String(startFrame));
    const runScope={expected_file:jobFile,expected_topic:jobTopic};
    const data=await api('run',requestedFrames?{...runScope,frames:requestedFrames,expected_result_task_id:retryTaskId,max_rounds:maxRounds,defer_agent_review:false,compare_with_truth:comparison}:{...runScope,frame:startFrame,count,max_rounds:maxRounds,defer_agent_review:false,compare_with_truth:comparison});
   if(data.file!==jobFile||data.topic!==jobTopic)throw Error('自动标注回执与提交的数据文件或图像源不一致，已拒绝更新当前页面');
   const sameDataScope=topic===jobTopic&&loadedDataPath===jobFile;
   const sameOperatorScope=sameDataScope&&frameNavigationGeneration===jobNavigationGeneration&&window.workspaceMode===jobWorkspaceMode;
   if(!sameOperatorScope){
    if(data.task_id)runResultAutoRestoreDeferred.add(data.task_id);
    if(sameDataScope)latestAgentJob.result_available=true;
    toast(sameDataScope?'自动标注结果已保留；您已切换帧或工作模式，点击“查看上次结果”可打开':'自动标注结果已保存到原数据作用域；当前页面保持不变，切回原数据后可查看');
    return;
   }
   markRunResultHandled(data.task_id);latestAgentJob.result_available=false;
  if(data.description)currentDescription=data.description;
   for(const item of data.frames||[])if(Number.isInteger(item.frame))candidateCountIndex[String(item.frame)]=Math.max(0,Number(item.remaining)||0);
   refreshCandidateSummary();
   if(data.stopped){toast('自动标注已停止；已完成帧的候选已保存');return}
  if(!data.frames?.length)throw Error('没有实际处理任何帧。参考帧和已有真值会跳过，请增大处理帧数。');
  const failed=data.failed_frames||data.frames.filter(item=>item.failed).map(item=>item.frame);
   retryFrames=[...failed];retryScope=failed.length?{file:jobFile,topic:jobTopic,taskId:data.task_id}:null;retryMode=failed.length?'failed':'';
  const first=data.frames.find(item=>!item.failed&&item.remaining>0)||data.frames.find(item=>item.remaining>0);
  if(first){
   await openCandidateFrame(first.frame,jobTopic);
    toast(failed.length?`${failed.length} 帧失败，旧候选已保留；已打开结果，请先查看错误`:'SAM3、GPT 复查与 Agent 决策均已完成；已打开候选，请检查后点击“保存结果”');
  }else toast('处理结束，但没有可显示的候选，请查看自动标注过程');
 }catch(e){if(requestedFrames&&e.status===409){retryFrames=[];retryScope=null;retryMode='';staleRetryRejected=true;toast(`${e.message}；正在读取最新任务结果`)}else toast(e.message)}
  finally{await refreshAgentProgress();runRequestStartedAt=null;runRequestFrame=null;runRequestCount=null;renderRunButton();if(staleRetryRejected)scheduleRunStatusPoll(0)}
};
let runStatusPollTimer=null,runStatusPollDelay=3000,runStatusFetching=false;
function scheduleRunStatusPoll(delay=runStatusPollDelay){
 if(runStatusPollTimer)clearTimeout(runStatusPollTimer);
 runStatusPollTimer=null;
 if(!document.hidden)runStatusPollTimer=setTimeout(pollRunStatus,delay);
}
async function pollRunStatus(){
 runStatusPollTimer=null;
 if(document.hidden)return;
 if(runStatusFetching)return;
 runStatusFetching=true;
 try{
  await refreshAgentProgress();
  const active=['running','stopping'].includes(latestAgentJob?.status);
   runStatusPollDelay=typeof runStatusScopeMismatch!=='undefined'&&runStatusScopeMismatch?1000:active?1000:15000;
 }finally{
  runStatusFetching=false;
  scheduleRunStatusPoll(runStatusPollDelay);
 }
}
document.addEventListener('visibilitychange',()=>{if(document.hidden){if(runStatusPollTimer)clearTimeout(runStatusPollTimer);runStatusPollTimer=null}else if(!runStatusPollTimer)pollRunStatus()});
pollRunStatus();
window.labelAgentBootstrapReady=fetch('/api/bootstrap').then(r=>r.json()).then(async data=>{labels=data.labels||[];autoLabels=data.auto_labels||labels;referenceOnlyLabels=data.reference_only_labels||[];labelStateSha256=data.label_state_sha256||'';configStateSha256=data.config_state_sha256||'';loadedFileActivationGeneration=data.file_activation_generation??0;shapes=data.truth||{};truthReviewStatus=data.truth_review_status||{};candidates=data.candidates||{};agentRuns=data.agent_runs||{};candidateReviewProposals=data.candidate_review_proposals||[];candidateReviewBindings=data.candidate_review_bindings||{};truthFrameIndex=new Set((data.truth_frames||Object.keys(shapes)).map(String));candidateCountIndex=data.candidate_counts&&typeof data.candidate_counts==='object'?{...data.candidate_counts}:Object.fromEntries(Object.entries(candidates).map(([name,items])=>[name,Array.isArray(items)?items.length:0]));compactGeometryMode=data.compact_geometry===true;workbenchConfig=data.config||{};window.referenceFrames=data.reference_frames||{};refreshLabels();refreshAutoLabels();refreshCandidateSummary();$('labelsInput').value=labels.join(',')||$('labelsInput').value;for(let [key,id] of Object.entries({llm_url:'llmUrl',llm_model:'llmModel'}))$(id).value=data.config[key]||'';currentDescription=data.description||null;const savedMode=localStorage.getItem('labelagentWorkspaceMode'),hasCandidates=Object.values(candidateCountIndex).some(value=>Number(value)>0);await window.setWorkspaceMode?.(savedMode||(hasCandidates?'auto':'manual'),false);if(data.loaded)await loadVideo(data);publishWorkbenchFrameState();return data});
window.labelAgentWorkspaceReady=window.labelAgentBootstrapReady.then(async data=>{workspaceRestoreState='restoring';bootstrapComplete=true;updateDatasetIdentity();updatePointPromptCoach();await refreshAgentProgress();await workspaceReadyDeferred;return {data,frame,topic,path:loadedDataPath}});

// Keep point-prompt mode unmistakable: entering the mode is not an inference request.
// The existing coach owns the inference state; this presentation layer mirrors it on
// the image and gives operators a safe, explicit exit without saving anything.
const updatePointPromptCoachBase=updatePointPromptCoach;
updatePointPromptCoach=function(){
 updatePointPromptCoachBase();
 const positive=promptPoints.filter(point=>point.label===1).length;
 const hasResult=generatedShapeFrame===frame&&generatedShapeTopic===topic&&generatedShapeIndex>=0;
 const awaitingPoint=annotationMode&&pointMode&&!positive&&!segmentBusy;
 $('stage')?.classList.toggle('pointPromptAwaiting',awaitingPoint);
 const button=$('pointButton');
 if(button){
  const elapsed=segmentBusy&&pointPromptStartedAt?Math.max(0,Math.round((Date.now()-pointPromptStartedAt)/1000)):0;
  const restoring=workspaceRestoreState!=='ready';button.disabled=restoring;
  button.textContent=restoring?'正在恢复工作帧…':segmentBusy?`SAM 生成中 ${elapsed}秒`:awaitingPoint?'等待点选：请点画面':hasResult?'SAM 精修':'SAM 点选（点目标即运行）';
  button.setAttribute('aria-pressed',String(pointMode));
 }
 const cancel=$('pointPromptCancel');
 if(cancel){cancel.disabled=segmentBusy;cancel.textContent=segmentBusy?'正在生成，暂不可退出':'退出点选'}
};
$('pointPromptCancel').onclick=()=>{
 if(segmentBusy){toast('SAM 正在生成，完成后即可退出');return}
 pointMode=false;negativeMode=false;draft=[];resetPrompts();
 $('polygonButton').classList.add('selected');
 refreshPointTools();draw();
 toast('已退出 SAM 点选；没有提交或保存人工真值');
};
updatePointPromptCoach();
