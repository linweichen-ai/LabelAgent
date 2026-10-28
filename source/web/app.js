const $=id=>document.getElementById(id);let candidates={},agentRuns={},topics={},topic='',frame=0,shapes={},draft=[],playing=null,labels=[],imageWidth=0,imageHeight=0,loading=false,playback=null,playGeneration=0,presentedTime=0,catalog={},recordDuration=0,clockTime=0,clockAnchor=0,clockBase=0,rate=1,tickHandle=null,extraWindows=new Map(),windowSerial=0,annotationMode=false,annotationView=null,pointMode=false,negativeMode=false,promptPoints=[],segmentBusy=false,generatedShapeFrame=null,generatedShapeTopic='',generatedShapeIndex=-1,regionMode='',selectedShapeIndex=-1,zoomScale=1;
async function api(path,body){const response=await fetch('/api/'+path,{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(body)});const raw=await response.text();let data;try{data=JSON.parse(raw)}catch{throw Error(`接口 ${path} 返回非 JSON 响应（HTTP ${response.status}），请重启标注服务`)}if(!response.ok)throw Error(data.error||`请求失败（HTTP ${response.status}）`);return data}
function toast(message){$('toast').textContent=message;$('toast').style.display='block';setTimeout(()=>$('toast').style.display='none',3500)}
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
function selectedShapes(){const key=String(frame);return shapes[key]??(window.workspaceMode==='manual'?(manualDrafts[key]??(manualDrafts[key]=[])):(candidates[key]??(autoDrafts[key]??(autoDrafts[key]=[]))))}
function frameAnnotationState(index=frame){const key=String(index);return shapes[key]!==undefined?'truth':window.workspaceMode==='auto'&&candidates[key]!==undefined?'candidate':'empty'}
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
 await showFrame(index);
 return frameAnnotationState(index);
}
function refreshLabels(){let select=$('labelSelect');select.innerHTML='';labels.forEach(x=>select.add(new Option(x,x)))}
function refreshShapeList(){let list=$('shapeList');list.innerHTML='';selectedShapes().forEach((s,i)=>{let row=document.createElement('div');row.className='shapeItem'+(i===selectedShapeIndex?' selected':'');row.onclick=()=>{selectedShapeIndex=i;refreshShapeList();draw()};let name=document.createElement('span');name.textContent=`${i+1}. ${s.label} · ${s.points.length} 点`;let remove=document.createElement('button');remove.textContent='×';remove.title='删除标注';remove.onclick=e=>{e.stopPropagation();selectedShapes().splice(i,1);if(selectedShapeIndex>=selectedShapes().length)selectedShapeIndex=-1;refreshShapeList();draw()};row.append(name,remove);list.append(row)})}
function draw(){let canvas=$('canvas'),img=$('frameImage');if(!img.naturalWidth)return;imageWidth=img.naturalWidth;imageHeight=img.naturalHeight;// Viewport rectangles already include the stage zoom; canvas CSS uses local coordinates.
const box=img.getBoundingClientRect(),stage=$('stage').getBoundingClientRect();canvas.style.left=((box.left-stage.left)/zoomScale)+'px';canvas.style.top=((box.top-stage.top)/zoomScale)+'px';canvas.style.width=(box.width/zoomScale)+'px';canvas.style.height=(box.height/zoomScale)+'px';canvas.width=imageWidth;canvas.height=imageHeight;let ctx=canvas.getContext('2d');ctx.clearRect(0,0,canvas.width,canvas.height);selectedShapes().forEach((s,i)=>{ctx.fillStyle='rgba(20,180,122,.25)';ctx.strokeStyle='#21e2a0';ctx.lineWidth=Math.max(2,imageWidth/700);for(const part of s.parts||[s]){ctx.beginPath();part.points.forEach(([x,y],j)=>j?ctx.lineTo(x,y):ctx.moveTo(x,y));ctx.closePath();for(const hole of part.holes||[]){hole.forEach(([x,y],j)=>j?ctx.lineTo(x,y):ctx.moveTo(x,y));ctx.closePath()}ctx.fill('evenodd');ctx.stroke()}ctx.fillStyle='#fff';ctx.font=`${Math.max(18,imageWidth/90)}px Arial`;ctx.fillText(`${i+1} ${s.label}`,s.points[0][0]+5,s.points[0][1]-8)});if(promptPoints.length){ctx.font=`${Math.max(20,imageWidth/70)}px Arial`;promptPoints.forEach(p=>{ctx.beginPath();ctx.arc(p.x,p.y,Math.max(6,imageWidth/350),0,Math.PI*2);ctx.fillStyle=p.label?'#21e2a0':'#ff615d';ctx.fill();ctx.strokeStyle='#102028';ctx.lineWidth=2;ctx.stroke();ctx.fillStyle='#102028';ctx.textAlign='center';ctx.textBaseline='middle';ctx.fillText(p.label?'+':'−',p.x,p.y)})}if(draft.length){ctx.strokeStyle='#ffd053';ctx.fillStyle='#ffd053';ctx.lineWidth=Math.max(2,imageWidth/700);ctx.beginPath();draft.forEach(([x,y],j)=>{j?ctx.lineTo(x,y):ctx.moveTo(x,y);ctx.beginPath();ctx.arc(x,y,Math.max(4,imageWidth/450),0,Math.PI*2);ctx.fill();ctx.beginPath();if(j)ctx.moveTo(...draft[j-1]);ctx.lineTo(x,y);ctx.stroke()})}}
function resetPrompts(){promptPoints=[];generatedShapeFrame=null;generatedShapeTopic='';generatedShapeIndex=-1;draw()}
function refreshPointTools(){for(const [id,selected] of [['pointButton',pointMode],['negativeButton',pointMode&&negativeMode],['polygonButton',!pointMode&&!regionMode&&$('polygonButton').classList.contains('selected')],['addRegionButton',regionMode==='add'],['eraseRegionButton',regionMode==='erase']])$(id).classList.toggle('selected',selected);$('canvas').classList.toggle('drawing',annotationMode&&(pointMode||$('polygonButton').classList.contains('selected')||regionMode));$('finishButton').hidden=pointMode}
async function generatePointMask(){if(segmentBusy||loading||playing||!annotationMode||!pointMode)return;if(!promptPoints.some(p=>p.label===1)){toast('先在目标内部点一下');return}segmentBusy=true;$('generateMaskButton').disabled=true;$('generateMaskButton').textContent='生成中…';const currentFrame=frame,currentTopic=topic;try{const result=await api('segment_points',{frame:currentFrame,points:promptPoints});if(frame!==currentFrame||topic!==currentTopic)return;const current=selectedShapes(),shape={label:$('labelSelect').value,points:result.points,holes:result.holes||[],parts:result.parts||[{points:result.points,holes:result.holes||[]}]};if(generatedShapeFrame===currentFrame&&generatedShapeTopic===currentTopic&&generatedShapeIndex>=0&&generatedShapeIndex<current.length)current[generatedShapeIndex]=shape;else{generatedShapeIndex=current.push(shape)-1;generatedShapeFrame=currentFrame;generatedShapeTopic=currentTopic}refreshShapeList();draw();toast(result.warnings?.join('；')||'已更新候选轮廓，请继续添加提示点或保存真值')}catch(error){toast(error.message)}finally{segmentBusy=false;$('generateMaskButton').disabled=false;$('generateMaskButton').textContent='生成轮廓'}}
function frameAtTime(time,metadata=playback){let times=metadata?.times||[],low=0,high=times.length;while(low<high){let mid=(low+high)>>1;if(times[mid]<=time+0.00001)low=mid+1;else high=mid}return Math.max(0,low-1)}
function currentClock(){return playing===true?Math.min(recordDuration,clockBase+(performance.now()-clockAnchor)/1000*rate):clockTime}
function stopPlayback(){clockTime=currentClock();playing=null;playGeneration++;cancelAnimationFrame(tickHandle);$('playbackVideo').pause();for(const view of extraWindows.values())for(const layer of view.layers||[])layer.video.pause();$('playButton').textContent='▶';$('playbackVideo').hidden=true;$('frameImage').hidden=false;$('canvas').hidden=!annotationMode;window.updateCandidateSave?.()}
function formatTime(seconds){const hundredths=Math.floor(Math.max(0,seconds)*100);return `${String(Math.floor(hundredths/6000)).padStart(2,'0')}:${String(Math.floor(hundredths/100)%60).padStart(2,'0')}.${String(hundredths%100).padStart(2,'0')}`}
function updateControls(){$('playbackTime').textContent=`${formatTime(clockTime)} / ${formatTime(recordDuration)}`;$('timeTimeline').max=recordDuration;$('timeTimeline').value=clockTime;if(topic){const offset=playback?.offset??catalog[topic]?.offset??0;frame=frameAtTime(Math.max(0,clockTime-offset));$('frameNumber').value=frame;}}
async function prepareVideo(selected,onProgress){while(true){const response=await fetch('/api/playback?topic='+encodeURIComponent(selected)),data=await response.json();if(!response.ok||data.status==='error')throw Error(data.error||'视频准备失败');if(data.status==='ready')return data;if(onProgress&&!onProgress(data.progress))throw Error('已取消');await new Promise(resolve=>setTimeout(resolve,400))}}
async function setMedia(video,metadata){const url=new URL(metadata.url,location.href).href;if(video.src===url&&video.readyState>=1)return;video.src=url;await new Promise((resolve,reject)=>{video.onloadedmetadata=resolve;video.onerror=()=>reject(Error('浏览器无法解码视频'));video.load()})}
function alignMedia(video,metadata,time,run){if(!metadata||video.readyState<1)return;const local=time-metadata.offset;const valid=local>=0&&local<=metadata.duration;const target=Math.max(0,Math.min(local,metadata.duration));if(Math.abs(video.currentTime-target)>(run?0.20:0.00001))video.currentTime=target;video.playbackRate=rate;if(run&&valid&&video.paused)video.play().catch(()=>{});else if(!run||!valid)video.pause()}
function updateWindows(force=false){for(const view of extraWindows.values()){if(view.element.hidden)continue;if(view.type==='image'){for(const layer of view.layers||[])if(layer.metadata)alignMedia(layer.video,layer.metadata,clockTime,!!playing)}else updateCloud(view,force)}}
function playbackTick(){if(!playing)return;clockTime=currentClock();updateControls();if(playback&&!$('primaryWindow').hidden)alignMedia($('playbackVideo'),playback,clockTime,true);updateWindows();if(clockTime>=recordDuration){pauseAtClock();return}tickHandle=requestAnimationFrame(playbackTick)}
async function pauseAtClock(){stopPlayback();if(topic){frame=frameAtTime(Math.max(0,clockTime-(playback?.offset??catalog[topic]?.offset??0)));await loadStill(frame)}updateControls();updateWindows(true)}
async function startPlayback(){
 if(playing){await pauseAtClock();return}
 if(!recordDuration)return;
 const generation=++playGeneration;playing='preparing';window.updateCandidateSave?.();clockBase=clockTime;clockAnchor=performance.now();$('playButton').textContent='⏸';
 try{
  if(topic&&!$('primaryWindow').hidden){if(!playback)playback=await prepareVideo(topic,progress=>{if(generation!==playGeneration)return false;$('imageInfo').textContent=`正在准备标注窗口：${Math.round(progress*100)}%`;return true});if(generation!==playGeneration)return;await setMedia($('playbackVideo'),playback)}
  if(generation!==playGeneration)return;
  if(clockTime>=recordDuration)clockTime=0;
  clockBase=clockTime;clockAnchor=performance.now();playing=true;draft=[];
  if(topic&&!$('primaryWindow').hidden){$('canvas').hidden=true;$('frameImage').hidden=true;$('playbackVideo').hidden=false;alignMedia($('playbackVideo'),playback,clockTime,true)}
  updateWindows(true);playbackTick();
 }catch(error){if(generation===playGeneration){stopPlayback();toast(error.message)}}
}
async function loadStill(n){if(!topic)return;selectedShapeIndex=-1;regionMode='';zoomScale=1;$('stage').style.transform='';frame=Math.max(0,Math.min(Number(n)||0,topics[topic].count-1));loading=true;window.updateCandidateSave?.();$('frameNumber').value=frame;$('frameTotal').textContent='/ '+(topics[topic].count-1);$('frameImage').src='/api/frame/'+frame+'?topic='+encodeURIComponent(topic)+'&v='+Date.now();draft=[];resetPrompts();refreshShapeList();updateReferenceStatus();window.refreshInspection?.()}
async function showFrame(n){if(!topic)return;stopPlayback();const target=Math.max(0,Math.min(Number(n)||0,topics[topic].count-1));try{if(!playback)playback=await prepareVideo(topic);clockTime=(playback.offset||0)+playback.times[target];await loadStill(target);updateControls();updateWindows(true)}catch(e){toast(e.message)}}
function refreshTopics(){setAnnotationMode(false);$('primaryTitle').textContent=topic||'无图像话题';$('primaryWindow').hidden=true;refreshMessageList();updateGrid();frame=window.referenceFrames?.[topic]?.[0]??0;$('frameNumber').value=frame;$('frameTotal').textContent='/ '+(topic?topics[topic].count-1:0);updateControls();updateReferenceStatus();}
function refreshMessageList(){refreshSourceChoices()}
function setAnnotationMode(enabled){annotationMode=!!enabled;window.updateCandidateSave?.();$('annotatePrimary').classList.toggle('selected',annotationMode);$('annotatePrimary').setAttribute('aria-pressed',String(annotationMode));$('annotatePrimary').title=annotationMode?'退出标注模式':'进入标注模式';$('annotatePrimary').setAttribute('aria-label',$('annotatePrimary').title);$('primaryWindow').classList.toggle('annotating',annotationMode);$('canvas').hidden=!annotationMode||!!playing;if(!annotationMode){draft=[];resetPrompts();draw()}}
function leaveAnnotation(){stopPlayback();setAnnotationMode(false);$('primaryWindow').hidden=true;$('primaryWindow').classList.remove('maximized');if(annotationView&&!annotationView.closed){annotationView.element.hidden=false;if(annotationView.wasMaximized)annotationView.element.classList.add('maximized');for(const layer of annotationView.layers)alignMedia(layer.video,layer.metadata,clockTime,false)}annotationView=null;updateGrid()}
async function activateAnnotation(view){if(view.closed)return;if(annotationView===view){leaveAnnotation();return}if(annotationView)leaveAnnotation();const time=currentClock();stopPlayback();try{const selected=view.sources[0],data=await api('topic',{topic:selected});if(view.closed)return;if(topic!==selected){manualDrafts={};autoDrafts={}}topic=selected;shapes=data.truth||{};candidates=data.candidates||{};agentRuns=data.agent_runs||{};draft=[];playback=null;annotationView=view;view.wasMaximized=view.element.classList.contains('maximized');view.element.classList.remove('maximized');view.element.hidden=true;for(const layer of view.layers)layer.video.pause();view.element.before($('primaryWindow'));$('primaryWindow').hidden=false;$('primaryWindow').classList.toggle('maximized',view.wasMaximized);$('primaryTitle').textContent=topic;setAnnotationMode(true);updateGrid();playback=await prepareVideo(topic);if(annotationView!==view)return;clockTime=time;await loadStill(frameAtTime(time-playback.offset));updateControls();updateWindows(true)}catch(error){if(annotationView===view)leaveAnnotation();toast(error.message)}}
function updateGrid(){const count=[...extraWindows.values()].filter(view=>!view.element.hidden).length+(!$('primaryWindow').hidden?1:0);const columns=count<=1?1:count<=4?2:3;$('windowGrid').style.setProperty('--columns',columns);$('windowGrid').style.setProperty('--rows',Math.max(1,Math.ceil(count/columns)));$('emptyWindows').hidden=count>0;if(!count)$('windowGrid').classList.remove('hasMaximized')}
function maximizeWindow(element){const expanded=element.classList.contains('maximized');document.querySelectorAll('.sensorWindow.maximized').forEach(x=>x.classList.remove('maximized'));element.classList.toggle('maximized',!expanded);$('windowGrid').classList.toggle('hasMaximized',!expanded);requestAnimationFrame(draw)}
function resetWindows(){annotationView=null;$('primaryWindow').hidden=true;for(const view of extraWindows.values()){view.closed=true;view.abort?.abort();view.renderer?.destroy();for(const layer of view.layers||[])layer.video.pause();view.element.remove()}extraWindows.clear();document.querySelectorAll('.maximized').forEach(x=>x.classList.remove('maximized'));$('windowGrid').classList.remove('hasMaximized');updateGrid()}
async function updateCloud(view,force=false){if(view.closed)return;view.desiredTime=clockTime;if(view.busy)return;if(!force&&(!playing||performance.now()-view.lastRequest<100))return;const time=clockTime;view.busy=true;view.lastRequest=performance.now();view.abort=new AbortController();try{const results=await Promise.all(view.sources.map(async source=>{const response=await fetch('/api/cloud?topic='+encodeURIComponent(source)+'&time='+time.toFixed(6),{signal:view.abort.signal});if(!response.ok){const data=await response.json();throw Error(data.error||'点云读取失败')}return {points:new Float32Array(await response.arrayBuffer()),count:Number(response.headers.get('X-Point-Count'))}}));if(view.closed)return;if(Math.abs(clockTime-time)<0.5||!playing&&Math.abs(clockTime-time)<0.01){const total=results.reduce((n,r)=>n+r.points.length,0),merged=new Float32Array(total);let offset=0;for(const result of results){merged.set(result.points,offset);offset+=result.points.length}view.renderer.setPoints(merged);view.status.textContent=`${time.toFixed(2)} 秒 · ${merged.length/4} / ${results.reduce((n,r)=>n+r.count,0)} 点 · ${view.sources.length} 源叠加`}}catch(e){if(!view.closed&&e.name!=='AbortError')view.status.textContent=e.message}finally{view.busy=false;if(!view.closed&&!playing&&Math.abs(view.desiredTime-time)>0.001)updateCloud(view,true)}}
function refreshSourceChoices(){const type=$('windowType').value,list=$('windowSources');list.replaceChildren();for(const [name,info] of Object.entries(catalog)){if(info.type!==type)continue;const label=document.createElement('label');label.className='sourceChoice';const input=document.createElement('input');input.type='checkbox';input.value=name;const span=document.createElement('span');span.textContent=name;label.append(input,span);list.append(label)}if(!list.children.length){const empty=document.createElement('p');empty.textContent='当前文件没有此类型的消息';list.append(empty)}$('windowHint').textContent=type==='image'?'多个图像源按顺序透明叠加，并按采集时间同步。':'多个点云源合并到同一个三维画面；请确保坐标系一致。'}
async function addWindow(type,sources){if(!sources.length||sources.some(name=>catalog[name]?.type!==type))throw Error('请选择有效的消息源');const id=++windowSerial,element=document.createElement('article');element.className='sensorWindow';const header=document.createElement('div');header.className='windowHeader';const title=document.createElement('span');title.textContent=(type==='image'?'图像 · ':'点云 · ')+sources.join(' + ');title.title=sources.join(' + ');const expand=document.createElement('button');expand.textContent='⛶';expand.title='放大 / 还原';const close=document.createElement('button');close.textContent='×';close.title='关闭窗口';header.append(title);if(type==='image'){const annotate=document.createElement('button');annotate.textContent='✎';annotate.title='进入标注模式';annotate.onclick=()=>activateAnnotation(view);header.append(annotate)}header.append(expand,close);const body=document.createElement('div');body.className='sensorBody';const status=document.createElement('div');status.className='sensorStatus';element.append(header,body,status);$('windowGrid').append(element);const view={id,sources,type,element,status,closed:false,lastRequest:0,busy:false,layers:[]};extraWindows.set(id,view);expand.onclick=()=>maximizeWindow(element);element.ondblclick=e=>{e.preventDefault();maximizeWindow(element)};close.onclick=()=>{if(annotationView===view)leaveAnnotation();view.closed=true;view.abort?.abort();view.renderer?.destroy();for(const layer of view.layers)layer.video.pause();element.remove();extraWindows.delete(id);if(element.classList.contains('maximized'))$('windowGrid').classList.remove('hasMaximized');updateGrid()};updateGrid();try{if(type==='image'){for(const [index,source] of sources.entries()){if(view.closed)return;const video=document.createElement('video');video.className='imageLayer'+(index?' overlay':'');video.muted=true;video.playsInline=true;video.preload='auto';body.append(video);const layer={source,video,metadata:null};view.layers.push(layer);status.textContent=`正在准备 ${index+1}/${sources.length}：${source}`;layer.metadata=await prepareVideo(source,progress=>{if(view.closed)return false;status.textContent=`正在准备 ${index+1}/${sources.length}：${Math.round(progress*100)}%`;return true});if(view.closed)return;await setMedia(video,layer.metadata);if(view.closed)return;alignMedia(video,layer.metadata,clockTime,!!playing)}status.textContent=`${sources.length} 个图像源 · 时间戳同步`}else{const canvas=document.createElement('canvas');body.append(canvas);view.renderer=new PointCloudView(canvas);status.textContent='拖动旋转 · 滚轮缩放';updateCloud(view,true)}}catch(e){if(!view.closed)status.textContent=e.message}}
$('addWindowButton').onclick=()=>{refreshSourceChoices();$('addWindowDialog').showModal()};$('closeWindowDialog').onclick=()=>$('addWindowDialog').close();$('cancelWindowDialog').onclick=()=>$('addWindowDialog').close();$('windowType').onchange=refreshSourceChoices;$('addWindowForm').onsubmit=e=>{e.preventDefault();const sources=[...$('windowSources').querySelectorAll('input:checked')].map(input=>input.value);if(!sources.length){toast('请至少选择一个消息源');return}const type=$('windowType').value;$('addWindowDialog').close();addWindow(type,sources).catch(error=>toast(error.message))};
$('timeTimeline').onchange=async e=>{stopPlayback();clockTime=Number(e.target.value);if(topic){try{if(!playback)playback=await prepareVideo(topic);frame=frameAtTime(clockTime-playback.offset);loadStill(frame)}catch(error){toast(error.message)}}updateControls();updateWindows(true)};
$('annotatePrimary').onclick=leaveAnnotation;$('maximizePrimary').onclick=()=>maximizeWindow($('primaryWindow'));$('primaryWindow').ondblclick=e=>{if(e.target===$('canvas')&&$('polygonButton').classList.contains('selected'))return;e.preventDefault();maximizeWindow($('primaryWindow'))};$('closePrimary').onclick=()=>{const view=annotationView;leaveAnnotation();view?.element.querySelector('.windowHeader button:last-child')?.click()};document.addEventListener('keydown',e=>{if(e.key==='Escape'){if($('addWindowDialog').open)return;document.querySelectorAll('.maximized').forEach(x=>x.classList.remove('maximized'));$('windowGrid').classList.remove('hasMaximized');return}const target=e.target;if(['INPUT','TEXTAREA','SELECT'].includes(target.tagName))return;if(e.key==='ArrowLeft'||e.key==='ArrowRight'){e.preventDefault();showFrame(frame+(e.key==='ArrowLeft'?-1:1))}});
$('playbackRate').onchange=()=>{const time=currentClock();rate=Number($('playbackRate').value);clockTime=clockBase=time;clockAnchor=performance.now();$('playbackVideo').playbackRate=rate;for(const view of extraWindows.values())for(const layer of view.layers||[])layer.video.playbackRate=rate;updateControls()};
$('polygonButton').classList.remove('selected');$('polygonButton').onclick=()=>{pointMode=false;$('polygonButton').classList.toggle('selected');resetPrompts();refreshPointTools()};$('pointButton').onclick=()=>{pointMode=true;negativeMode=false;draft=[];$('polygonButton').classList.remove('selected');refreshPointTools();draw()};$('negativeButton').onclick=()=>{pointMode=true;draft=[];negativeMode=!negativeMode;$('polygonButton').classList.remove('selected');refreshPointTools();draw()};$('generateMaskButton').onclick=generatePointMask;refreshPointTools();function beginRegionMode(mode){const items=selectedShapes();if(regionMode===mode){regionMode='';draft=[];refreshPointTools();draw();return}if(!items.length){toast('当前帧还没有可编辑的标注');return}if(selectedShapeIndex<0){if(items.length===1)selectedShapeIndex=0;else{toast('请先在标注列表中选择要编辑的标注');return}}regionMode=mode;pointMode=false;draft=[];$('polygonButton').classList.remove('selected');refreshShapeList();refreshPointTools();draw()}$('addRegionButton').onclick=()=>beginRegionMode('add');$('eraseRegionButton').onclick=()=>beginRegionMode('erase');
async function finish(){if(pointMode){generatePointMask();return}if(!annotationMode||loading||playing)return;if(draft.length<3){toast('至少需要 3 个点');return}if(regionMode){if(selectedShapeIndex<0){toast('请先选择要编辑的标注');return}try{const result=await api('edit_region',{frame,operation:regionMode,polygon:draft.map(p=>p.map(Math.round)),shape:selectedShapes()[selectedShapeIndex]});selectedShapes()[selectedShapeIndex]=result;draft=[];regionMode='';refreshPointTools();refreshShapeList();draw();toast('区域已更新')}catch(e){toast(e.message)}return}selectedShapes().push({label:$('labelSelect').value,points:draft.map(p=>p.map(Math.round))});draft=[];refreshShapeList();draw()}
$('frameImage').onload=()=>{loading=false;window.updateCandidateSave?.();if(playing)return;updateControls();draw()};$('frameImage').onerror=()=>{loading=false;window.updateCandidateSave?.();$('imageInfo').textContent=`帧 ${frame} 载入失败`;toast('帧载入失败')};window.addEventListener('resize',draw);$('stage').addEventListener('wheel',e=>{if(!annotationMode)return;e.preventDefault();zoomScale=Math.max(1,Math.min(5,zoomScale*(e.deltaY<0?1.15:0.87)));$('stage').style.transform=`scale(${zoomScale})`;$('stage').style.transformOrigin='center center';draw()},{passive:false});new ResizeObserver(draw).observe($('stage'));$('canvas').onclick=e=>{if(!annotationMode||!pointMode&&!$('polygonButton').classList.contains('selected')&&!regionMode)return;let box=e.currentTarget.getBoundingClientRect();if(!imageWidth||loading||playing)return;const x=(e.clientX-box.left)*imageWidth/box.width,y=(e.clientY-box.top)*imageHeight/box.height;if(pointMode)promptPoints.push({x:Math.round(x),y:Math.round(y),label:negativeMode?0:1});else draft.push([x,y]);draw()};$('canvas').ondblclick=e=>{e.preventDefault()};$('undoButton').onclick=()=>{if(pointMode)promptPoints.pop();else draft.pop();draw()};$('finishButton').onclick=finish;$('clearButton').onclick=()=>{selectedShapes().splice(0);draft=[];resetPrompts();refreshShapeList();draw()};
function updateReferenceStatus(){const reference=window.referenceFrames?.[topic]?.[0],status=$('referenceStatus'),kind=frameAnnotationState();status.textContent=(reference===undefined?'':`当前参考帧：${reference}`)+(kind==='truth'?' · 当前帧真值已保存':kind==='candidate'?' · 当前帧是自动候选，尚未保存为真值':'')+(kind==='candidate'&&agentRuns[String(frame)]?` · Agent ${agentRuns[String(frame)].converged?'复查完成，候选待保存':'流程未完成'} (${agentRuns[String(frame)].rounds} 轮)`: '');$('referenceButton').textContent=reference===frame?'已是参考帧':'设为参考帧';$('truthSave').textContent=window.workspaceMode==='auto'&&candidates[String(frame)]&&!shapes[String(frame)]?'确认并保存当前帧真值':'保存当前帧真值';$('openReferenceButton').hidden=reference===undefined}async function openReferenceFrame(){const reference=window.referenceFrames?.[topic]?.[0];if(reference===undefined)return;try{let view=annotationView;if(!view||view.sources[0]!==topic){view=[...extraWindows.values()].find(item=>item.type==='image'&&item.sources[0]===topic&&!item.closed);if(!view){await addWindow('image',[topic]);view=[...extraWindows.values()].find(item=>item.type==='image'&&item.sources[0]===topic&&!item.closed)}if(!view)throw Error('无法打开参考帧窗口');await activateAnnotation(view)}await showFrame(reference)}catch(error){toast(error.message)}}$('openReferenceButton').onclick=openReferenceFrame;async function loadVideo(data){manualDrafts={};autoDrafts={};window.referenceFrames=data.reference_frames||{};stopPlayback();setAnnotationMode(false);resetWindows();catalog=data.catalog||{};recordDuration=data.duration||0;clockTime=0;playback=null;topics=data.topics;topic=data.topic;shapes=data.truth||{};candidates=data.candidates||{};agentRuns=data.agent_runs||{};updateReferenceStatus();loading=false;$('mcapPath').value=data.path||$('mcapPath').value;refreshTopics();toast('MCAP 已载入')}
let fileManagerBusy=false;
function fileStatus(message){$('fileManagerStatus').textContent=message}
function renderFiles(files){const list=$('fileList');list.replaceChildren();if(!files.length){const empty=document.createElement('p');empty.className='emptyFileList';empty.textContent='尚未添加 MCAP 文件';list.append(empty)}for(const file of files){const row=document.createElement('div');row.className='fileRow';const info=document.createElement('div');info.className='fileInfo';const name=document.createElement('strong');name.textContent=file.name;const path=document.createElement('span');path.textContent=file.path;path.title=file.path;const detail=document.createElement('small');detail.textContent=`${(file.size/1024/1024).toFixed(1)} MB${file.loaded?' · 当前已加载':''}${!file.available?' · 文件不存在':''}`;info.append(name,path,detail);const load=document.createElement('button');load.textContent=file.loaded?'已加载':'加载';load.disabled=file.loaded||!file.available||fileManagerBusy;load.onclick=async()=>{if(fileManagerBusy)return;fileManagerBusy=true;fileStatus('正在加载 '+file.name);try{const data=await api('files/'+file.id+'/load',{});await loadVideo(data);fileStatus('已加载 '+file.name)}catch(error){fileStatus(error.message)}finally{fileManagerBusy=false;await refreshFiles()}};const remove=document.createElement('button');remove.textContent='删除';remove.disabled=fileManagerBusy;remove.title='从列表移除，保留磁盘文件';remove.onclick=async()=>{if(fileManagerBusy)return;fileManagerBusy=true;try{const response=await fetch('/api/files/'+file.id,{method:'DELETE'}),data=await response.json();if(!response.ok)throw Error(data.error||'删除失败');if(data.unloaded)clearLoadedVideo();fileStatus('已从列表移除 '+file.name);renderFiles(data.files)}catch(error){fileStatus(error.message)}finally{fileManagerBusy=false;await refreshFiles()}};row.append(info,load,remove);list.append(row)}}
async function refreshFiles(){try{const response=await fetch('/api/files'),data=await response.json();if(!response.ok)throw Error(data.error||'读取文件列表失败');renderFiles(data.files)}catch(error){fileStatus(error.message)}}
function clearLoadedVideo(){manualDrafts={};autoDrafts={};candidates={};agentRuns={};stopPlayback();setAnnotationMode(false);resetWindows();catalog={};topics={};topic=null;playback=null;shapes={};draft=[];frame=0;recordDuration=0;clockTime=0;loading=false;$('frameImage').removeAttribute('src');$('playbackVideo').removeAttribute('src');$('playbackVideo').load();$('primaryWindow').hidden=true;$('frameNumber').value=0;$('frameTotal').textContent='/ 0';refreshMessageList();refreshShapeList();updateGrid();updateControls()}
$('chooseButton').onclick=()=>$('mcapFile').click();$('mcapFile').onchange=async()=>{if(fileManagerBusy)return;const files=[...$('mcapFile').files];if(!files.length)return;fileManagerBusy=true;try{for(const [index,file]of files.entries()){fileStatus(`正在添加 ${index+1}/${files.length}：${file.name} (${(file.size/1024/1024).toFixed(0)} MB)`);const form=new FormData();form.append('file',file);const response=await fetch('/api/upload',{method:'POST',body:form}),data=await response.json();if(!response.ok)throw Error(data.error||'添加失败')}fileStatus(`已添加 ${files.length} 个文件，请点击「加载」`)}catch(error){fileStatus(error.message)}finally{fileManagerBusy=false;$('mcapFile').value='';await refreshFiles()}};
$('openButton').onclick=async()=>{if(fileManagerBusy)return;fileManagerBusy=true;try{await api('files',{path:$('mcapPath').value});fileStatus('已添加，请点击「加载」');$('mcapPath').value=''}catch(error){fileStatus(error.message)}finally{fileManagerBusy=false;await refreshFiles()}};
$('frameNumber').onchange=e=>showFrame(e.target.value);$('playButton').onclick=startPlayback;$('stopButton').onclick=async()=>{stopPlayback();clockTime=0;if(topic)await loadStill(0);updateControls();updateWindows(true)};
$('labelsSave').onclick=async()=>{try{let data=await api('labels',{labels:$('labelsInput').value.split(',')});labels=data.labels;refreshLabels();toast('类别已保存')}catch(e){toast(e.message)}};
$('referenceButton').onclick=async()=>{try{const data=await api('reference_frame',{frame});window.referenceFrames[topic]=[data.frame];updateReferenceStatus();toast('已设为参考帧，后续 AI 不会修改此帧')}catch(e){toast(e.message)}};
async function saveCurrentTruth(candidateOnly=false){
 if(candidateOnly&&(window.workspaceMode!=='auto'||shapes[String(frame)]!==undefined||!Object.hasOwn(candidates,String(frame))))throw Error('当前帧没有待保存的自动候选');
 if(!annotationMode)throw Error('请先在主画面打开待复核的图像帧');
 if(loading||playing)throw Error('请暂停并等待原始帧加载后再保存真值');
 const savedShapes=selectedShapes(),data=await api('truth',{frame,shapes:savedShapes});
 shapes[String(frame)]=savedShapes;delete manualDrafts[String(frame)];delete autoDrafts[String(frame)];delete candidates[String(frame)];
 updateReferenceStatus();toast(`已保存 ${data.saved} 个真值标注`);
}
$('truthSave').onclick=()=>saveCurrentTruth().catch(e=>toast(e.message));
$('openConfigDialog').onclick=()=>$('config').showModal();
$('closeConfigDialog').onclick=()=>$('config').close();
$('openFilesDialog').onclick=()=>{$('fileManager').showModal();refreshFiles()};
$('closeFilesDialog').onclick=()=>$('fileManager').close();
$('doneFilesButton').onclick=()=>$('fileManager').close();
$('configSave').onclick=async()=>{try{await api('config',{llm_url:$('llmUrl').value,llm_model:$('llmModel').value,llm_key:$('llmKey').value});$('config').close();toast('配置已保存')}catch(e){toast(e.message)}};
let currentDescription=null;
$('describeButton').onclick=async()=>{const button=$('describeButton');if(button.disabled)return;button.disabled=true;button.textContent='生成中…';toast('正在生成真值描述…');try{currentDescription=await api('describe',{frame});toast('描述已生成')}catch(e){toast(e.message)}finally{button.disabled=false;button.textContent='生成真值描述'}};
const agentPhase={starting:'准备任务',describing:'GPT 生成真值描述',loading:'读取图像',generating:'SAM 3 生成候选',generated:'候选已生成',observing:'GPT 观察图像和候选',editing:'执行标注工具',edited:'工具执行结束',finished:'本帧结束',saved:'候选已保存',completed:'本次处理结束',stopping:'正在停止',stopped:'已停止',failed:'运行失败'};
const agentStop={done:'Agent 已结束',needs_review:'旧记录：请求人工复核',max_rounds:'达到最大复查轮数',no_progress:'没有有效修改',cycle:'检测到循环修改',error:'模型或工具调用失败'};
let progressFetching=false;
$('downloadDebug').onclick=async()=>{
 const button=$('downloadDebug');if(button.disabled)return;button.disabled=true;
 try{
  const response=await fetch('/api/run/debug',{cache:'no-store'});
  if(!response.ok)throw Error('无法下载调用记录');
  const data=await response.json(),blob=new Blob([JSON.stringify(data.trace||[])],{type:'application/json'}),url=URL.createObjectURL(blob),link=document.createElement('a');
  link.href=url;link.download='autolabel-model-debug.json';link.click();setTimeout(()=>URL.revokeObjectURL(url),1000);
 }catch(error){toast(error.message)}finally{button.disabled=false}
};
let latestAgentJob={status:'idle'},runRequestStartedAt=null;
function renderRunButton(){
 const button=$('runButton');
 const job=runRequestStartedAt!==null&&(!latestAgentJob.started_at||latestAgentJob.started_at<runRequestStartedAt-1)?{status:'running',started_at:runRequestStartedAt}:latestAgentJob;
 const stopping=job.status==='stopping',running=job.status==='running'||stopping;
 const pending=runRequestStartedAt!==null&&!running&&!['stopped','completed','failed'].includes(job.status);
 const elapsed=job.started_at?Math.max(0,Math.floor((running?Date.now()/1000:(job.events?.at(-1)?.time||job.started_at))-job.started_at)):null;
 button.disabled=pending||stopping;
 button.textContent=(pending?'启动中…':stopping?'正在停止…':running?'停止自动标注':'开始自动标注')+(running?` · 第 ${job.round||0}/100 轮`: '')+(elapsed===null?'':` · ${elapsed} 秒`);
 button.setAttribute('aria-busy',String(running));
}
function renderAgentProgress(job){latestAgentJob=job;renderRunButton()}
async function refreshAgentProgress(){if(progressFetching)return;progressFetching=true;try{const response=await fetch('/api/run/status',{cache:'no-store'});if(response.ok){renderAgentProgress(await response.json());$('runButton').removeAttribute('title')}}catch{$('runButton').title='无法读取 Agent 状态，请检查工作台服务连接'}finally{progressFetching=false}}
$('runButton').onclick=async()=>{
 const button=$('runButton');
 if(latestAgentJob.status==='running'){
  button.disabled=true;button.textContent='正在停止…';
  try{await api('run/stop',{});toast('已请求停止自动标注')}catch(e){toast(e.message);button.disabled=false;renderRunButton()}
  return;
 }
 if(runRequestStartedAt!==null)return;
 runRequestStartedAt=Date.now()/1000;renderRunButton();
 try{
  const startFrame=Number($('frameNumber').value),count=Number($('taskCount').value);
  if(!Number.isInteger(startFrame)||startFrame<0||startFrame>=(topics[topic]?.count||0))throw Error('请选择有效的当前帧');
  if(!Number.isInteger(count)||count<1||count>100)throw Error('处理帧数须为 1～100');
  const jobTopic=topic;
  stopPlayback();
  const data=await api('run',{frame:startFrame,count});
  if(data.description)currentDescription=data.description;
  if(data.stopped){toast('自动标注已停止；已完成帧的候选已保存');return}
  if(!data.frames?.length)throw Error('没有实际处理任何帧。参考帧和已有真值会跳过，请增大处理帧数。');
  const state=await(await fetch('/api/state')).json();candidates=state.candidates||{};agentRuns=state.agent_runs||{};refreshShapeList();updateReferenceStatus();draw();
  const first=data.frames.find(item=>item.remaining>0);
  if(first){
   await openCandidateFrame(first.frame,jobTopic);
   toast('已打开候选，请检查后点击“保存结果”');
  }else toast('处理结束，但没有可显示的候选，请查看自动标注过程');
 }catch(e){toast(e.message)}
 finally{await refreshAgentProgress();runRequestStartedAt=null;renderRunButton()}
};
setInterval(()=>{renderRunButton();refreshAgentProgress()},1000);refreshAgentProgress();
fetch('/api/state').then(r=>r.json()).then(data=>{labels=data.labels||[];shapes=data.truth||{};candidates=data.candidates||{};agentRuns=data.agent_runs||{};window.referenceFrames=data.reference_frames||{};refreshLabels();$('labelsInput').value=labels.join(',')||$('labelsInput').value;for(let [key,id] of Object.entries({llm_url:'llmUrl',llm_model:'llmModel'}))$(id).value=data.config[key]||'';currentDescription=data.description||null;if(data.loaded){catalog=data.catalog||{};recordDuration=data.duration||0;topics=data.topics;topic=data.topic;$('mcapPath').value=data.path;refreshTopics()}});
