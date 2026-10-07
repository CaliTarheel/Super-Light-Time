/* Timing and orientation journeys; no model or external service is run. */
const fs = require('node:fs');
const vm = require('node:vm');
const assert = require('node:assert/strict');
const path = require('node:path');
const root = path.resolve(__dirname, '..');
const html = fs.readFileSync(path.join(root, 'web/index.html'), 'utf8');
const nodes = new Map(), outputs = new Map();
function element(tagName = 'div') {
  const listeners = new Map();
  return { tagName: String(tagName).toUpperCase(), textContent: '', innerHTML: '', hidden: false, checked: false,
    value: '', disabled: false, dataset: {}, style: {}, children: [], options: [], firstChild: {nodeValue:''},
    classList: { toggle() {}, add() {}, remove() {} }, setAttribute() {}, removeAttribute() {}, setPointerCapture() {},
    addEventListener(type, fn) { if (!listeners.has(type)) listeners.set(type, []); listeners.get(type).push(fn); },
    async emit(type, event = {}) { for (const fn of listeners.get(type) || []) await fn({target:this,preventDefault(){},...event}); },
    append(...children) { this.children.push(...children); }, add(option) { this.options.push(option); },
    replaceChildren(...children) { this.children = children; if (this.tagName === 'SELECT') this.options = children; },
    querySelectorAll() { return []; }, reportValidity() { return true; },
    click() { downloaded.push(this.href); }, remove() {}, getContext() { return {}; }, getBoundingClientRect() { return {left:0,top:0,width:640,height:320}; } };
}
for (const match of html.matchAll(/<(\w+)\b([^>]*\bid="([^"]+)"[^>]*)>/g)) {
  const [, tag, attributes, id] = match, node = element(tag);
  node.value = attributes.match(/\bvalue="([^"]*)"/)?.[1] || '';
  node.checked = /\bchecked\b/.test(attributes);
  if (tag === 'select') {
    const content = html.slice(match.index + match[0].length).split('</select>')[0];
    node.options = [...content.matchAll(/<option\b([^>]*)>([^<]*)/g)].map(option => ({value:option[1].match(/value="([^"]*)"/)?.[1] || '', textContent:option[2], selected:/\bselected\b/.test(option[1])}));
    node.value = (node.options.find(option => option.selected) || node.options[0])?.value || '';
  }
  nodes.set(id, node);
}
const get = (id) => { assert.ok(nodes.has(id), `Referenced HTML control exists: ${id}`); return nodes.get(id); };
const document = { getElementById:get, createElement:element, createElementNS:(_,tag)=>element(tag),
  querySelector(selector) { if (!outputs.has(selector)) outputs.set(selector, element()); return outputs.get(selector); },
  querySelectorAll:()=>[], addEventListener(){}, body:element(), activeElement:null };
let fetchBody; const downloaded = []; let clockMs = 0; const storage = new Map();
const sandbox = {URL, URLSearchParams, structuredClone, location:{search:'',href:'http://test/'}, history:{replaceState(){}},
  performance:{now:()=>clockMs}, document, window:{addEventListener(){},localStorage:{getItem:key=>storage.get(key)??null,setItem:(key,value)=>storage.set(key,value)}}, setTimeout(){}, clearTimeout(){}, setInterval(){},
  requestAnimationFrame(){}, console, Option:function(text,value){return {textContent:text,value};},
  fetch:async (_,options)=>{fetchBody=options?.body;return {ok:true,json:async()=>({})};} };
vm.createContext(sandbox);
const source = fs.readFileSync(path.join(root, 'web/app.js'),'utf8').replace('  initialize();',
  '  globalThis.app={state,terrain,gospl,globeOrientation,editHistory,normalizedOrientation,sameOrientation,outputOrientation,orientedPath,frameCacheKey,updateOrientationControls,applyGlobeOrientation,applyStatus,showFrame,refreshRecord,loadInspection,generateTerrain,generateGospl,savedOrientationNote,terrainSourceLabel,terrainPath,restoreEdit,timings,captureTiming,timingLabel,updateTimingDisplays,applyTerrainStatus,applyGosplStatus,updateTerrainControls,updateGosplControls,pauseOrResume,setRunningControls,setApi(fn){api=fn;}};');
vm.runInContext(source,sandbox);
const app = sandbox.app;
let checks=0;
async function check(name,action){await action();checks++;console.log(`PASS ${name}`);}
const snapshot=(elapsed,eta,state='running')=>({elapsed_seconds:elapsed,eta_seconds:eta,state,timing_state:state==='running'?(eta==null?'estimating':'running'):state});
const plain=value=>JSON.parse(JSON.stringify(value));
const angles={yaw:60,pitch:20,roll:-15};
const fakeFrame=(index=0)=>({index,time_myr:index*2,width:4,height:2,crust:Array(8).fill(1),elevation:Array(8).fill(100),plate:Array(8).fill(0),age:Array(8).fill(0),boundary:Array(8).fill(0),events:[],plates:[],stats:{}});
const deferred=()=>{let resolve;const promise=new Promise(done=>{resolve=done;});return{promise,resolve};};
const setAngle=async(axis,value)=>{get(`orientation-${axis}`).value=String(value);await get(`orientation-${axis}`).emit('input');};
const poseFromPath=path=>JSON.parse(new URL(path,'http://test').searchParams.get('orientation')||'{"yaw":0,"pitch":0,"roll":0}');
(async()=>{
  await check('elapsed increases and ETA counts down without another HTTP response',()=>{
    clockMs=1000;app.captureTiming('simulation',snapshot(60,180),'world-a');
    clockMs=11000;app.updateTimingDisplays();assert.match(get('simulation-timing').textContent,/1m 10s elapsed.*2m 50s remaining/);
  });
  await check('pause freezes the clock and resume continues the accumulated runtime',()=>{
    app.captureTiming('simulation',snapshot(75,165,'paused'),'world-a');clockMs+=600000;app.updateTimingDisplays();
    assert.equal(get('simulation-timing').textContent,'1m 15s elapsed · paused');
    app.captureTiming('simulation',snapshot(75,165),'world-a');clockMs+=5000;app.updateTimingDisplays();
    assert.match(get('simulation-timing').textContent,/1m 20s elapsed.*2m 40s remaining/);
  });
  await check('new experiments reset their own elapsed time and completed timing stops',()=>{
    app.captureTiming('simulation',snapshot(0,null),'world-b');assert.match(get('simulation-timing').textContent,/0s elapsed.*estimating/);
    app.captureTiming('simulation',snapshot(42,0,'complete'),'world-b');clockMs+=90000;app.updateTimingDisplays();
    assert.equal(get('simulation-timing').textContent,'42s elapsed · complete');
  });
  await check('null estimates remain explicitly uncertain; expired estimates do not imply completion',()=>{
    app.captureTiming('terrain',snapshot(5,null),'terrain-a');clockMs+=10000;app.updateTimingDisplays();
    assert.match(get('terrain-timing').textContent,/15s elapsed.*estimating/);
    app.captureTiming('terrain',snapshot(15,2),'terrain-a');clockMs+=5000;app.updateTimingDisplays();
    assert.match(get('terrain-timing').textContent,/20s elapsed.*updating estimate/);assert.doesNotMatch(get('terrain-timing').textContent,/complete/);
  });
  await check('simulation, terrain and goSPL clocks progress independently',()=>{
    app.captureTiming('simulation',snapshot(100,null,'paused'),'world-a');app.captureTiming('terrain',snapshot(200,30),'terrain-a');app.captureTiming('gospl',snapshot(400,60),'export-a');
    clockMs+=10000;app.updateTimingDisplays();
    assert.equal(get('simulation-timing').textContent,'1m 40s elapsed · paused');
    assert.match(get('terrain-timing').textContent,/3m 30s elapsed.*20s remaining/);
    assert.match(get('gospl-timing').textContent,/6m 50s elapsed.*50s remaining/);
  });
  await check('missing legacy runtime is hidden, never represented as zero',()=>{
    app.captureTiming('simulation',{state:'complete'},'legacy');assert.equal(get('simulation-timing').hidden,true);
  });
  await check('applying globe orientation requests a spherical frame and persists the chosen pose',async()=>{
    app.state.runId='oriented-world';app.state.frames=[{index:0,time_myr:0},{index:1,time_myr:2}];app.state.frame=fakeFrame();app.state.currentIndex=0;app.state.mode='history';app.state.status={state:'complete'};app.updateOrientationControls();
    const requests=[];app.setApi(async(path)=>{requests.push(path);return path.startsWith('/api/frame')?{...fakeFrame(),orientation:poseFromPath(path)}:{events:[]};});
    for(const [axis,value] of Object.entries(angles)) await setAngle(axis,value);
    assert.deepEqual(plain(app.outputOrientation()),{yaw:0,pitch:0,roll:0});
    await get('orientation-apply').emit('click');
    assert.deepEqual(poseFromPath(requests.find(path=>path.startsWith('/api/frame'))),angles);
    assert.deepEqual(poseFromPath(requests.find(path=>path.startsWith('/api/record'))),angles);
    assert.deepEqual(JSON.parse(storage.get('deep-time.orientation.oriented-world')),angles);
    assert.deepEqual(plain(app.state.frame.orientation),angles);
    assert.equal(get('download-heightmap').disabled,false);
  });
  await check('snapshot downloads and region inspection use exactly the displayed orientation',async()=>{
    for(const id of ['download-heightmap','download-export','download-preview']) await get(id).emit('click');
    for(const path of downloaded.slice(-3)) assert.deepEqual(poseFromPath(path),angles);
    let request;app.setApi(async(path)=>{request=path;return {mode:'location',points:[],selection:{frame:0,cell:3}};});
    await app.loadInspection(0,3);assert.match(request,/\/api\/history\?/);assert.deepEqual(poseFromPath(request),angles);
    assert.match(request,/cell=3/);
  });
  await check('switching experiments restores each pose, and reset requests the original sphere',async()=>{
    app.state.runId='other-world';app.updateOrientationControls();assert.deepEqual(plain(app.outputOrientation()),{yaw:0,pitch:0,roll:0});
    app.state.runId='oriented-world';app.updateOrientationControls();assert.deepEqual(plain(app.outputOrientation()),angles);
    const requests=[];app.setApi(async(path)=>{requests.push(path);return path.startsWith('/api/frame')?fakeFrame():{events:[]};});
    await get('orientation-reset').emit('click');assert.deepEqual(plain(app.outputOrientation()),{yaw:0,pitch:0,roll:0});
    assert.equal(requests.find(path=>path.startsWith('/api/frame')),'/api/frame?index=0');
  });
  await check('late old-orientation frames cannot repopulate the cache or replace the new view',async()=>{
    const old=deferred();app.state.cache.clear();app.state.frame=null;
    app.setApi(async(path)=>path.includes('orientation=')?{...fakeFrame(),view:'new'}:old.promise);
    const loadingOld=app.showFrame(0);
    app.globeOrientation.output={...angles};app.globeOrientation.draftOutput={...angles};app.state.cache.clear();
    await app.showFrame(0);old.resolve({...fakeFrame(),view:'old'});await loadingOld;
    assert.equal(app.state.frame.view,'new');assert.equal(app.state.cache.size,1);
    assert.ok([...app.state.cache.keys()].every(key=>key.includes('/60,20,-15')));
  });
  await check('live status polling cannot jump away from the epoch being rotated',async()=>{
    app.state.runId='live-world';app.state.status={state:'running',run_id:'live-world'};app.state.frames=[{index:0,time_myr:0},{index:1,time_myr:2}];app.state.frame=fakeFrame();app.state.currentIndex=0;app.state.cache.clear();app.updateOrientationControls();
    const pending=deferred(),requests=[];
    app.setApi(async(path)=>{requests.push(path);return path.startsWith('/api/frame')?pending.promise:{events:[]};});
    await setAngle('yaw',60);const rotating=get('orientation-apply').emit('click');
    await app.applyStatus({run_id:'live-world',state:'running',frames:[{index:0,time_myr:0},{index:1,time_myr:2},{index:2,time_myr:4}],elapsed_seconds:10,eta_seconds:100,timing_state:'running',time_myr:4,progress:.01});
    assert.equal(requests.filter(path=>path.startsWith('/api/frame')).length,1);
    pending.resolve(fakeFrame());await rotating;assert.equal(app.state.currentIndex,0);
    app.globeOrientation.output={...angles};app.globeOrientation.draftOutput={...angles};
  });
  await check('late old-orientation event fallback cannot replace the current record',async()=>{
    const old=deferred();app.state.cache.clear();app.globeOrientation.output={yaw:0,pitch:0,roll:0};
    app.setApi(async(path)=>{if(path.startsWith('/api/record')){const error=new Error('old API');error.status=404;throw error;}return old.promise;});
    const loading=app.refreshRecord();await Promise.resolve();await Promise.resolve();
    app.globeOrientation.output={...angles};app.state.record={events:[{type:'current'}]};old.resolve({...fakeFrame(),events:[{type:'old'}]});await loading;
    assert.equal(app.state.record.events[0].type,'current');
  });
  await check('new terrain and goSPL builds freeze orientation; saved assets keep their own coordinate system',async()=>{
    app.state.frame=fakeFrame();app.state.mode='history';app.state.currentIndex=0;app.state.frames=[{index:0,time_myr:0},{index:1,time_myr:2}];
    app.terrain.job=null;app.gospl.job=null;app.updateTerrainControls();app.updateGosplControls();
    const requests=[];app.setApi(async(path,data)=>{requests.push({path,data:plain(data)});return{job_id:path.includes('gospl')?'new-export':'new-terrain',run_id:app.state.runId,state:'running',progress:0,width:8192,time_myr:0,start_myr:0,end_myr:2,orientation:plain(data.orientation)};});
    await get('generate-terrain').emit('click');await get('generate-gospl').emit('click');
    assert.deepEqual(requests[0].data.orientation,angles);assert.deepEqual(requests[1].data.orientation,angles);
    app.globeOrientation.output={yaw:-90,pitch:0,roll:0};
    assert.deepEqual(plain(app.terrain.job.orientation),angles);assert.deepEqual(plain(app.gospl.job.orientation),angles);
    assert.match(app.terrainSourceLabel(app.terrain.job),/Orientation differs from current view/);
    assert.doesNotMatch(app.terrainPath(app.terrain.job,'preview'),/orientation=/);
    assert.doesNotMatch(app.terrainPath(app.terrain.job,'download','&format=zip'),/orientation=/);
  });
  await check('initial rotation posts the painted world, is one undo step, and does not change a paused checkpoint',async()=>{
    app.state.mode='initial';app.state.status={state:'paused',can_resume:true,run_id:app.state.runId};
    app.state.initial={width:4,height:2,crust:Uint8Array.from([0,1,2,1,0,0,1,2])};app.editHistory.undo=[];app.editHistory.redo=[];app.updateOrientationControls();
    const before=Array.from(app.state.initial.crust),rotated=[2,1,0,1,2,1,0,0];let request;
    app.setApi(async(path,data)=>{request={path,data};return{width:4,height:2,crust:rotated};});
    await setAngle('yaw',90);await get('orientation-apply').emit('click');
    assert.equal(request.path,'/api/orient-initial');assert.deepEqual(Array.from(request.data.initial.crust),before);
    assert.deepEqual(plain(request.data.orientation),{yaw:90,pitch:0,roll:0});assert.deepEqual(Array.from(app.state.initial.crust),rotated);
    assert.equal(app.editHistory.undo.length,1);assert.equal(app.state.status.state,'paused');assert.equal(get('orientation-yaw').value,0);
    await get('undo-edit').emit('click');assert.deepEqual(Array.from(app.state.initial.crust),before);
    assert.deepEqual(plain(app.outputOrientation()),{yaw:-90,pitch:0,roll:0});
  });
  console.log(`${checks} orientation and timing UI checks passed`);
})().catch(error=>{console.error(error);process.exitCode=1;});
