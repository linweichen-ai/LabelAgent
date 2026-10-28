/* The automatic workspace shows recorded model calls and immutable stage images. */
const inspectionPanel=$('inspectionPanel');
const inspectionStages=['输入 GPT','GPT 描述与关键词','SAM 3 分割','GPT 复查输入','GPT 决策输出','Agent 标注结果'];
let inspectionData={trace:[],steps:[]},inspectionStage=0,inspectionRunFrame=null,inspectionBusy=false,inspectionVersion='',inspectionStageManual=false,inspectionDisclosureScope='';
let inspectionReferenceMode=null;
function configureInspectionStages(isReference){
 if(inspectionReferenceMode===isReference)return;
 inspectionReferenceMode=isReference;inspectionStage=isReference?0:2;inspectionStageManual=false;
 const tabs=inspectionPanel.querySelector('.inspectionStages');tabs.replaceChildren();
 for(const i of (isReference?[0,1]:[2,3,4,5])){
  const button=document.createElement('button');button.type='button';button.role='tab';button.id='inspectionStage'+i;button.dataset.stage=i;
  button.textContent=(isReference?i+1:i-1)+' · '+inspectionStages[i];
  button.onclick=()=>{inspectionStage=i;inspectionStageManual=true;renderInspection(true)};tabs.append(button);
 }
 inspectionPanel.querySelector('.inspectionTitle strong').textContent=isReference?'参考帧准备':'自动标注过程';
 $('describeButton').hidden=!isReference;
 $('prepareReference').hidden=isReference;
 $('taskCount').closest('label').hidden=isReference;
 inspectionPanel.querySelector('.runActions').hidden=isReference;
}
function inspectionNode(tag,text,className){const element=document.createElement(tag);if(text!==undefined)element.textContent=text;if(className)element.className=className;return element}
function inspectionBlock(parent,title,value,raw=false){
 const details=inspectionNode('details',undefined,'inspectionData');
 details.append(inspectionNode('summary',title));
 const text=typeof value==='string'?value:JSON.stringify(value,null,2);
 details.append(inspectionNode(raw?'pre':'p',text||'尚无记录',raw?'inspectionText':'inspectionSummary'));
 parent.append(details);
}
function inspectionRaw(parent,title,value){inspectionBlock(parent,title,value,true)}
function inspectionImage(parent,url,title,annotations=null){
 if(!url)return;
 const figure=document.createElement('figure'),wrap=inspectionNode('div',undefined,'inspectionImage');const img=new Image();img.alt=title;img.src=url;img.loading='lazy';wrap.append(img);
 if(annotations){const canvas=document.createElement('canvas');wrap.append(canvas);img.onload=()=>{canvas.width=img.naturalWidth;canvas.height=img.naturalHeight;const ctx=canvas.getContext('2d');for(const [i,shape] of annotations.entries()){ctx.fillStyle='rgba(20,180,122,.32)';ctx.strokeStyle='#21e2a0';ctx.lineWidth=Math.max(2,canvas.width/700);for(const part of shape.parts||[shape]){ctx.beginPath();for(const ring of [part.points,...(part.holes||[])]){ring.forEach(([x,y],j)=>j?ctx.lineTo(x,y):ctx.moveTo(x,y));ctx.closePath()}ctx.fill('evenodd');ctx.stroke()}if(shape.points?.length){ctx.font=`${Math.max(18,canvas.width/80)}px sans-serif`;ctx.fillStyle='white';ctx.fillText(`${shape.id??i+1} ${shape.label}`,shape.points[0][0]+4,shape.points[0][1]+22)}}}}
 figure.append(wrap,inspectionNode('figcaption',title+' · 点击放大'));wrap.onclick=()=>{const dialog=document.createElement('dialog');dialog.className='inspectionZoom';const close=inspectionNode('button','关闭');close.onclick=()=>dialog.close();dialog.append(close,wrap.cloneNode(true));if(annotations){const source=wrap.querySelector('canvas'),target=dialog.querySelector('canvas');target.width=source.width;target.height=source.height;target.getContext('2d').drawImage(source,0,0)}dialog.onclose=()=>dialog.remove();dialog.onclick=e=>{if(e.target===dialog)dialog.close()};document.body.append(dialog);dialog.showModal()};parent.append(figure);
}
function inspectionImages(parent,images,titles){const gallery=inspectionNode('div',undefined,'inspectionGallery');images.forEach((item,i)=>inspectionImage(gallery,item?.url,titles[i]||`输入图像 ${i+1}`));if(gallery.children.length)parent.append(gallery);else parent.append(inspectionNode('p','此旧记录没有保存过程图像；后续运行会保留实际输入和输出图像。','inspectionEmpty'))}
function inspectionCard(parent,title,status){const card=inspectionNode('article',undefined,'inspectionCard');const heading=inspectionNode('div',undefined,'inspectionCardHeading');heading.append(inspectionNode('strong',title),inspectionNode('span',status||''));card.append(heading);parent.append(card);return card}
function inspectionCallState(call){return (call.status==='rejected'?(call.retrying?'校验未通过，已请求修正':'校验未通过，重试已耗尽'):({running:'调用中',succeeded:'已返回',failed:'调用失败'}[call.status]||call.status))+(call.elapsed_ms===undefined?'':` · ${(call.elapsed_ms/1000).toFixed(1)} 秒`)}
function inspectionReviewAttempt(call){return call.review_attempt?` · 第 ${call.review_attempt} 次响应${call.review_attempt>1?'（协议修正）':''}`:''}
function inspectionCallError(card,call){if(call.error)card.append(inspectionNode('p',call.error,'inspectionError'))}
function updateCandidateSave(){
 $('saveCandidate').disabled=window.workspaceMode!=='auto'||!annotationMode||loading||!!playing||!$('frameImage').naturalWidth||['running','stopping'].includes(inspectionData.job?.status)||shapes[String(frame)]!==undefined||!Object.hasOwn(candidates,String(frame));
}
window.updateCandidateSave=updateCandidateSave;
function renderInspection(force=false){
 if(window.workspaceMode!=='auto')return;
 const target=inspectionRunFrame??frame;
 configureInspectionStages((window.referenceFrames?.[topic]||[]).includes(target));
 const version=JSON.stringify([inspectionData,topic,target,inspectionStage,candidates[String(target)],shapes[String(target)],agentRuns[String(target)]]);
 if(!force&&version===inspectionVersion)return;inspectionVersion=version;
 $('inspectionContext').textContent=`${topic||'尚未加载图像源'} · 帧 ${target}`+(['running','stopping'].includes(inspectionData.job?.status)?` · 正在处理 · ${agentPhase[inspectionData.job.phase]||inspectionData.job.phase} · 第 ${inspectionData.job.round||0} 轮`:'');
 const trace=inspectionData.trace,describes=trace.filter(x=>x.purpose==='生成真值描述'),calls=trace.filter(x=>x.frame===target&&x.purpose!=='生成真值描述'),sams=calls.filter(x=>x.service==='SAM 3'),reviews=calls.filter(x=>x.service==='GPT'),steps=inspectionData.steps.filter(x=>x.frame===target),description=inspectionData.description;
 const availability=[describes.length,description||describes.length,sams.length,reviews.length,reviews.length,steps.length||candidates[String(target)]||agentRuns[String(target)]];
 inspectionPanel.querySelectorAll('[role=tab]').forEach(button=>{const i=Number(button.dataset.stage);button.classList.toggle('selected',i===inspectionStage);button.setAttribute('aria-selected',String(i===inspectionStage));button.dataset.available=Boolean(availability[i]);button.title=availability[i]?'已有记录':'等待此阶段运行'});
 const body=$('inspectionBody'),scope=JSON.stringify([topic,target,inspectionStage]);
 const expanded=scope===inspectionDisclosureScope?new Set([...body.querySelectorAll('.inspectionData[open]')].map(node=>node.dataset.disclosureKey)):new Set();
 inspectionDisclosureScope=scope;
 body.setAttribute('aria-labelledby','inspectionStage'+inspectionStage);body.replaceChildren();
 if(inspectionStage===0){
  for(const call of describes){const card=inspectionCard(body,`参考帧 ${call.frame} → GPT`,inspectionCallState(call));inspectionBlock(card,'发送给 GPT 的完整提示词',call.input?.instruction,true);inspectionImages(card,call.input?.images||[],['GPT 输入图 1：原始参考帧','GPT 输入图 2：仅画真值轮廓的参考帧']);inspectionCallError(card,call)}
  if(!describes.length)body.append(inspectionNode('p',description?'本次使用已生成的描述，旧记录未保留生成描述时的完整调用。重新生成真值描述后可查看。':'设置参考帧并生成真值描述后，这里显示发送给 GPT 的提示词与参考帧真值图。','inspectionEmpty'));
 }else if(inspectionStage===1){
  for(const call of describes){const card=inspectionCard(body,'GPT 原始输出',inspectionCallState(call));const parsed=call.output?.parsed;if(parsed){inspectionBlock(card,'完整真值描述（供 GPT 复查）',parsed.description);inspectionBlock(card,'GPT 生成的 SAM 3 关键词',parsed.prompts,true)}else if(call.output?.raw_text)inspectionBlock(card,'GPT 返回原文',call.output.raw_text,true);inspectionRaw(card,'完整原始响应',call.output||{});inspectionCallError(card,call)}
  if(description){const card=inspectionCard(body,'流程实际采用的描述与关键词','包含格式及参考帧类别校验');inspectionBlock(card,'真值描述',description.description);inspectionBlock(card,'发送给 SAM 3 的关键词',description.prompts,true)}
 }else if(inspectionStage===2){
  for(const call of sams){const card=inspectionCard(body,`${call.input?.endpoint==='/segment_points'?'点提示':'文本提示'} · 第 ${call.round||0} 轮`,inspectionCallState(call));inspectionBlock(card,'SAM 3 实际输入',{prompt:call.input?.prompt,points:call.input?.points,polygon:call.input?.polygon,mask:call.input?.mask},true);inspectionImages(card,[call.input?.image,call.input?.mask,call.output?.segmentation],['输入原图','输入的已有候选 mask','SAM 3 返回分割叠加图']);if(call.output){const {segmentation,...summary}=call.output;inspectionBlock(card,'SAM 3 输出摘要',summary,true)}inspectionCallError(card,call)}
 }else if(inspectionStage===3){
  for(const call of reviews){const card=inspectionCard(body,`第 ${call.round||1} 轮 → GPT${inspectionReviewAttempt(call)}`,inspectionCallState(call));const prompt=call.input?.instruction||'',images=call.input?.images||[];const titles=images.map((_,i)=>{const n=i+1;if(n===1)return 'GPT 输入图 1：原始帧';if(n===2)return 'GPT 输入图 2：当前候选与 ID';if(prompt.includes(`图像 ${n} 是缺口局部证据页`))return `GPT 输入图 ${n}：局部原图 / 二值 mask / 待判断缺口`;if(prompt.includes(`图像 ${n} 是修改前候选`))return `GPT 输入图 ${n}：SAM 点提示修改前候选`;if(n===3&&prompt.includes('图像 3 左半'))return 'GPT 输入图 3：缺口编号与二值 mask';if(n===3&&prompt.includes('图片 3 是当前内部空洞'))return 'GPT 输入图 3：空洞编号';if(n===4&&prompt.includes('图像 4 是未着色的原始参考帧'))return 'GPT 输入图 4：原始参考帧';if(n===5&&prompt.includes('图像 5 只用洋红色'))return 'GPT 输入图 5：参考帧真值轮廓';return `GPT 输入图 ${n}（含义见本轮 Prompt）`});inspectionImages(card,images,titles);if(call.retry_feedback)inspectionBlock(card,'Agent 反馈给 GPT 的校验错误',call.retry_feedback);inspectionBlock(card,'再次发送给 GPT 的完整 Prompt',call.input?.instruction,true);inspectionCallError(card,call)}
 }else if(inspectionStage===4){
  for(const call of reviews){const card=inspectionCard(body,`GPT 第 ${call.round||1} 轮决策${inspectionReviewAttempt(call)}`,inspectionCallState(call));const output=call.output?.parsed;if(output){inspectionBlock(card,'判断与依据',`${output.status}\n${output.reason||''}`);if(output.hole_assessment)inspectionBlock(card,'空洞与开放缺口判断',output.hole_assessment,true);if(output.sam_assessment)inspectionBlock(card,'SAM 点提示是否改进',output.sam_assessment,true);inspectionBlock(card,'要求 Agent 执行的操作',output.operations,true)}else if(call.output?.raw_text)inspectionBlock(card,'GPT 返回原文',call.output.raw_text,true);inspectionCallError(card,call);inspectionRaw(card,'完整输出',call.output||{})}
  if(!reviews.length&&agentRuns[String(target)])inspectionBlock(body,'历史 GPT 决策与执行记录（未保留完整 Prompt）',agentRuns[String(target)].audit,true);
 }else{
  const final=steps.filter(x=>x.phase==='finished').at(-1),run=agentRuns[String(target)]||final?.result?.agent;
  if(run){const card=inspectionCard(body,'Agent 最终结果',run.converged?'复查完成':'流程未完成');inspectionBlock(card,'停止原因',`${agentStop[run.stop_reason]||run.stop_reason} · ${run.rounds} 轮\n${shapes[String(target)]!==undefined?'此帧已保存真值':'此帧候选尚未保存为真值，可检查后保存结果。'}`);if(final)inspectionImage(card,final.image,`帧 ${target} · Agent 最终候选 · ${final.shape_count} 个标注`);else if(candidates[String(target)]||shapes[String(target)])inspectionImage(card,'/api/frame/'+target+'?topic='+encodeURIComponent(topic),'当前已保存的候选 / 真值',candidates[String(target)]||shapes[String(target)]);if(run.review_queue?.length)inspectionBlock(card,'流程错误与未完成原因',run.review_queue,true)}
  for(const step of steps.filter(x=>x.phase!=='finished')){const card=inspectionCard(body,step.phase==='generated'?'SAM 3 初步候选':`第 ${step.round} 轮工具执行后`,`${step.shape_count} 个标注`);inspectionImage(card,step.image,'此阶段的标注结果');if(step.outcome)inspectionBlock(card,'实际执行结果',step.outcome,true)}
  if(run?.audit?.length)inspectionRaw(body,'逐轮 GPT 决策与 Agent 实际执行记录',run.audit);
 }
 if(!body.children.length)body.append(inspectionNode('p',`帧 ${target} 暂无此阶段记录。开始自动标注后，会实时显示提示词、图像和调用结果。`,'inspectionEmpty'));
 const occurrences=new Map();
 for(const details of body.querySelectorAll('.inspectionData')){
  const title=details.querySelector('summary').textContent,index=occurrences.get(title)||0;
  occurrences.set(title,index+1);
  details.dataset.disclosureKey=JSON.stringify([title,index]);
  details.open=expanded.has(details.dataset.disclosureKey);
 }
 updateCandidateSave();
 $('clearAutoResult').disabled=!(Object.hasOwn(candidates,String(target))||Object.hasOwn(agentRuns,String(target))||steps.length||calls.length)||['running','stopping'].includes(inspectionData.job?.status);
}
async function refreshInspection(){
 if(window.workspaceMode!=='auto'||inspectionBusy)return;inspectionBusy=true;
 const requestedTopic=topic,requestedFrame=inspectionRunFrame??frame;
 try{
  const response=await fetch('/api/run/inspection?topic='+encodeURIComponent(topic||'')+'&frame='+requestedFrame,{cache:'no-store'});
  if(!response.ok)throw Error('无法读取过程记录');
  const data=await response.json();
  if(topic!==requestedTopic||(inspectionRunFrame??frame)!==requestedFrame)return;
  const active=['running','stopping'].includes(data.job?.status)?data.active_frame:null;
  if(active!==null&&active!==undefined&&active!==requestedFrame){inspectionRunFrame=active;inspectionVersion='';return}
  if((active===null||active===undefined)&&inspectionRunFrame!==null){inspectionRunFrame=null;inspectionVersion='';return}
  inspectionRunFrame=active??null;
  inspectionData=data;
  configureInspectionStages((window.referenceFrames?.[topic]||[]).includes(requestedFrame));
  if(active!==null&&active!==undefined&&!inspectionStageManual){
   const calls=data.trace.filter(x=>x.purpose!=='生成真值描述'&&x.frame===requestedFrame),last=calls.at(-1);
   inspectionStage=data.steps.some(x=>x.frame===requestedFrame&&x.phase==='finished')?5:last?.service==='GPT'?(last.status==='running'?3:4):2;
  }
  renderInspection();
 }
 catch(error){$('inspectionContext').textContent=error.message}finally{inspectionBusy=false}
}
window.refreshInspection=refreshInspection;
async function ensureWorkspaceImage(){
 if(!topic)return;
 if(annotationMode)return;
 const current=Number($('frameNumber').value)||0;
 let view=[...extraWindows.values()].find(x=>x.type==='image'&&!x.closed&&x.sources[0]===topic);
 if(!view){await addWindow('image',[topic]);view=[...extraWindows.values()].find(x=>x.type==='image'&&!x.closed&&x.sources[0]===topic)}
 if(view){await activateAnnotation(view);await showFrame(current)}
}
window.setWorkspaceMode=async(mode,openImage=true)=>{
 const changed=window.workspaceMode!==mode;
 if(changed&&mode==='manual')inspectionRunFrame=null;
 window.workspaceMode=mode;document.body.dataset.workspace=mode;
 $('manualModeButton').classList.toggle('selected',mode==='manual');$('autoModeButton').classList.toggle('selected',mode==='auto');$('manualModeButton').setAttribute('aria-pressed',String(mode==='manual'));$('autoModeButton').setAttribute('aria-pressed',String(mode==='auto'));
 $('workspaceModeHint').textContent=mode==='auto'?'候选复核 · 可查看模型输入、分割和执行结果':'手工绘制与编辑真值';
 $('autoDesk').hidden=mode!=='auto';
 inspectionPanel.hidden=mode!=='auto';
 $('truth').hidden=mode==='auto';
 for(const id of ['pointButton','negativeButton','generateMaskButton'])$(id).hidden=mode==='manual';
 $('clearButton').hidden=mode==='auto';
 $('truthSave').textContent=mode==='auto'&&candidates[String(frame)]&&shapes[String(frame)]===undefined?'确认并保存当前帧真值':'保存当前帧真值';
 if(changed){pointMode=false;negativeMode=false;regionMode='';draft=[];resetPrompts();refreshPointTools();refreshShapeList();updateReferenceStatus();draw()}
 if(openImage)await ensureWorkspaceImage();
 if(mode==='auto'){await refreshInspection();renderInspection(true)}
 window.resizeWorkbench();
 requestAnimationFrame(draw);
};
$('manualModeButton').onclick=()=>window.setWorkspaceMode('manual').catch(e=>toast(e.message));
$('autoModeButton').onclick=()=>window.setWorkspaceMode('auto').catch(e=>toast(e.message));
$('runButton').addEventListener('click',()=>{inspectionRunFrame=null;inspectionStageManual=false;inspectionStage=2;refreshInspection()});
$('prepareReference').onclick=()=>{if(window.referenceFrames?.[topic]?.length)openReferenceFrame();else toast('请先在人工标注模式保存真值并设为参考帧')};
$('describeButton').addEventListener('click',()=>{inspectionStage=0;refreshInspection()});
$('saveCandidate').onclick=async()=>{try{await saveCurrentTruth(true);renderInspection(true);refreshInspection()}catch(error){toast(error.message)}};
$('clearAutoResult').onclick=async()=>{
 const selected=frame,selectedTopic=topic,button=$('clearAutoResult');
 if(button.disabled)return;
 button.disabled=true;button.textContent='清除中…';
 try{
  const response=await fetch('/api/auto_result/'+selected,{method:'DELETE',headers:{'Content-Type':'application/json'},body:JSON.stringify({topic:selectedTopic})});
  const result=await response.json();
  if(!response.ok)throw Error(result.error||'清除失败');
  if(topic===selectedTopic){
   delete candidates[String(selected)];delete agentRuns[String(selected)];delete autoDrafts[String(selected)];
   if(frame===selected){draft=[];resetPrompts();refreshShapeList();draw();updateReferenceStatus()}
   inspectionData={...inspectionData,description:null,trace:inspectionData.trace.filter(call=>call.frame!==selected&&call.purpose!=='生成真值描述'),steps:inspectionData.steps.filter(step=>step.frame!==selected),frames:(inspectionData.frames||[]).filter(n=>n!==selected)};
   inspectionVersion='';renderInspection(true);
   await Promise.all([refreshInspection(),refreshAgentProgress()]);
  }
  toast(result.truth_preserved?`帧 ${selected} 的自动结果已清除，人工真值保留`:`帧 ${selected} 的自动结果已清除`);
 }catch(error){toast(error.message)}
 finally{button.textContent='清除结果';renderInspection(true)}
};
window.setWorkspaceMode('manual',false);
setInterval(()=>{if(window.workspaceMode==='auto')refreshInspection()},1000);
