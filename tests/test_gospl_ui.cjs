/* goSPL export journeys with deferred HTTP responses and experiment switching. */
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
let fetchBody; const downloaded = [];
const sandbox = {document, window:{addEventListener(){}}, setTimeout(){}, clearTimeout(){}, setInterval(){},
  requestAnimationFrame(){}, console, Option:function(text,value){return {textContent:text,value};},
  fetch:async (_,options)=>{fetchBody=options?.body;return {ok:true,json:async()=>({})};} };
vm.createContext(sandbox);
const source = fs.readFileSync(path.join(root, 'web/app.js'),'utf8').replace('  initialize();',
  '  globalThis.app={state,gospl,updateGosplControls,refreshGosplJobs,pollGospl,applyGosplStatus,gosplFormError,setApi(fn){api=fn;}};');
vm.runInContext(source,sandbox);
const app = sandbox.app;
let checks = 0;
async function check(name, action) { await action(); checks++; console.log(`PASS ${name}`); }
const plain = value => JSON.parse(JSON.stringify(value));
const frames = (...times) => times.map((time_myr,index)=>({index,time_myr}));
const complete = (job_id,run_id,start_myr=0,end_myr=10) => ({job_id,run_id,start_myr,end_myr,node_count:163842,subdivisions:7,state:'complete',progress:1,filename:'history.zip'});
const selectRun = (id,times) => {app.state.runId=id;app.state.frames=frames(...times);app.updateGosplControls();};
const choose = async (id,value,type='change') => { get(id).value=String(value);await get(id).emit(type); };
const deferred = () => { let resolve,reject; const promise=new Promise((yes,no)=>{resolve=yes;reject=no;});return {promise,resolve,reject}; };

(async()=>{
  await check('evolving export requires two saved epochs and defaults to the requested mesh and climate',()=>{
    assert.equal(get('gospl-subdivisions').value,'7');assert.equal(get('gospl-dt').value,'100000');assert.equal(get('gospl-rainfall').value,'1');
    selectRun('a',[0]);assert.equal(get('generate-gospl').disabled,true);
    selectRun('a',[0,2]);assert.equal(get('generate-gospl').disabled,false);
    assert.deepEqual([get('gospl-start').value,get('gospl-end').value],['0','1']);
    assert.match(get('gospl-selected-source').textContent,/0–2 Myr elapsed/);
    assert.deepEqual(get('gospl-subdivisions').options.map(option=>option.value),['6','7','8','9']);
  });
  await check('default range grows with saved history; deliberate range survives scrubbing and polling',async()=>{
    selectRun('a',[0,2,4,6]);assert.equal(get('gospl-end').value,'3');
    await choose('gospl-start',1);await choose('gospl-end',2);
    app.state.currentIndex=0;app.state.mode='initial';app.updateGosplControls();
    selectRun('a',[0,2,4,6,8]);
    assert.deepEqual([get('gospl-start').value,get('gospl-end').value],['1','2']);
    selectRun('b',[0,10,20]);assert.deepEqual([get('gospl-start').value,get('gospl-end').value],['0','2']);
  });
  await check('invalid ranges and numeric inputs cannot start; zero rainfall is valid',async()=>{
    await choose('gospl-start',2);assert.equal(get('generate-gospl').disabled,true);assert.match(get('gospl-form-note').textContent,/later/);
    await choose('gospl-start',0);await choose('gospl-dt',0,'input');assert.equal(get('generate-gospl').disabled,true);
    await choose('gospl-dt',100000.5,'input');assert.equal(get('generate-gospl').disabled,true);assert.match(get('gospl-form-note').textContent,/whole-year/);
    await choose('gospl-dt',300000,'input');assert.equal(get('generate-gospl').disabled,true);assert.match(get('gospl-form-note').textContent,/divide every/);
    await choose('gospl-dt',100000,'input');await choose('gospl-rainfall',-1,'input');assert.equal(get('generate-gospl').disabled,true);
    await choose('gospl-rainfall',20.1,'input');assert.equal(get('generate-gospl').disabled,true);
    await choose('gospl-rainfall',0,'input');assert.equal(get('generate-gospl').disabled,false);
    await choose('gospl-rainfall',1,'input');
  });
  await check('a running simulation can export; its POST source and time range stay frozen across a run switch',async()=>{
    const wait=deferred();let request;
    app.state.status={state:'running'};app.state.playing=true;
    app.setApi((path,data)=>{request={path,data:plain(data)};return wait.promise;});
    const starting=get('generate-gospl').emit('click');
    assert.deepEqual(request,{path:'/api/gospl/start',data:{run_id:'b',start_index:0,end_index:2,subdivisions:7,dt_years:100000,rainfall_m_yr:1,orientation:{yaw:0,pitch:0,roll:0}}});
    selectRun('c',[0,2]);
    assert.match(get('gospl-progress-source').textContent,/Different experiment · b · 0–20/);
    await choose('gospl-dt',50000,'input');
    wait.resolve({...complete('build-b','b',0,20),state:'running',progress:.2,message:'Sampling history'});await starting;
    assert.equal(app.gospl.job.run_id,'b');assert.equal(request.data.dt_years,100000);
    assert.equal(app.state.playing,true);assert.equal(app.state.status.state,'running');
    assert.match(get('gospl-progress-label').textContent,/20%/);
  });
  await check('other-world completion stays saved without replacing the current experiment',async()=>{
    app.setApi(async path=>{assert.match(path,/status\?job_id=build-b$/);return complete('build-b','b',0,20);});
    await app.pollGospl();assert.equal(app.gospl.result,null);
    assert.equal(app.gospl.jobs.filter(job=>job.state==='complete').length,1);
    assert.equal(get('gospl-result').hidden,true);assert.equal(get('saved-gospl').disabled,false);
  });
  await check('explicit cross-experiment export is labeled, downloadable, and remains selected',async()=>{
    await choose('saved-gospl','build-b');assert.equal(app.gospl.selectionManual,true);
    assert.match(get('gospl-result-source').textContent,/Different experiment · b/);
    await get('download-gospl').emit('click');assert.equal(downloaded.at(-1),'/api/gospl/download?job_id=build-b');
    selectRun('b',[0,10,20]);assert.match(get('gospl-result-source').textContent,/Selected experiment · b/);
    selectRun('c',[0,2]);assert.equal(app.gospl.result.job_id,'build-b');
  });
  await check('automatic restoration chooses the newest matching export, not a newer unrelated world',async()=>{
    app.gospl.selectionManual=false;
    app.setApi(async()=>({jobs:[complete('300-other','b'),complete('100-matching','c'),complete('200-matching','c',0,2)]}));
    await get('refresh-gospl').emit('click');assert.equal(app.gospl.result.job_id,'200-matching');
    selectRun('empty',[0,2]);assert.equal(app.gospl.result,null);
    assert.ok(get('saved-gospl').options.length>=4);
  });
  await check('a refreshed page recovers an in-progress export with its own source',async()=>{
    app.gospl.job=null;
    app.setApi(async()=>({jobs:[{...complete('recover / job','c'),state:'running',progress:.6}]}));
    await app.refreshGosplJobs();assert.equal(app.gospl.job.job_id,'recover / job');
    assert.match(get('gospl-progress-source').textContent,/Different experiment · c/);
    assert.equal(get('generate-gospl').disabled,true);
  });
  await check('late status response cannot undo cancellation or remove completed exports',async()=>{
    const wait=deferred();let cancelled;
    app.setApi(async(path,data)=>{
      if(path.startsWith('/api/gospl/status')) {assert.match(path,/recover%20%2F%20job/);return wait.promise;}
      assert.equal(path,'/api/gospl/cancel');cancelled=plain(data);return {...app.gospl.job,state:'cancelled'};
    });
    const polling=app.pollGospl();await get('cancel-gospl').emit('click');
    assert.deepEqual(cancelled,{job_id:'recover / job'});assert.equal(app.gospl.job.state,'cancelled');
    wait.resolve({...app.gospl.job,state:'running',progress:.7});await polling;
    assert.equal(app.gospl.job.state,'cancelled');assert.match(get('gospl-progress-label').textContent,/cancelled/);
    assert.equal(get('generate-gospl').disabled,false);
  });
  await check('deliberate saved selection survives a new completion',async()=>{
    selectRun('c',[0,2]);
    app.gospl.jobs.push(complete('old-keep','c'));await choose('saved-gospl','old-keep');
    app.applyGosplStatus(complete('new-build','c',0,2));
    assert.equal(app.gospl.result.job_id,'old-keep');assert.equal(get('saved-gospl').value,'old-keep');
  });
  await check('failed list refresh preserves the selected archive and later refresh recovers',async()=>{
    app.setApi(async()=>{throw new Error('offline');});await app.refreshGosplJobs();
    assert.equal(app.gospl.result.job_id,'old-keep');assert.equal(get('download-gospl').disabled,false);
    assert.match(get('saved-gospl-status').textContent,/could not be refreshed/);
    app.setApi(async()=>({jobs:[]}));await app.refreshGosplJobs();assert.equal(app.gospl.notice,'');
  });
  await check('failed start unlocks export controls and preserves completed ZIPs',async()=>{
    app.setApi(async()=>{throw new Error('Not enough free disk space');});
    await get('generate-gospl').emit('click');assert.equal(app.gospl.starting,false);
    assert.equal(get('generate-gospl').disabled,false);assert.equal(app.gospl.result.job_id,'old-keep');
    assert.match(get('saved-gospl-status').textContent,/disk space/);
  });
  await check('a stale list response cannot overwrite a newer completed job',async()=>{
    const wait=deferred();app.setApi(async()=>wait.promise);const refreshing=app.refreshGosplJobs();
    app.applyGosplStatus(complete('new-complete','c',0,2));
    wait.resolve({jobs:[{...complete('new-complete','c'),state:'running'}]});await refreshing;
    assert.equal(app.gospl.job.state,'complete');assert.equal(app.gospl.jobs.find(job=>job.job_id==='new-complete').state,'complete');
  });
  await check('all form controls have unique HTML ids and the approximation is stated',()=>{
    const ids=[...html.matchAll(/\bid="([^"]+)"/g)].map(match=>match[1]);assert.equal(new Set(ids).size,ids.length);
    assert.match(html,/Motion and geometric vertical forcing/);assert.match(html,/Mesh spacing differs from heightmap pixels/);
    assert.match(html,/app\.js\?v=[a-z0-9-]+/);
  });
  console.log(`${checks} goSPL UI checks passed`);
})().catch(error=>{console.error(error);process.exitCode=1;});
