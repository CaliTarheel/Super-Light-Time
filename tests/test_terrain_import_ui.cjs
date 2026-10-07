/* User journeys for native and goSPL final surfaces, including stale replies. */
const fs = require('node:fs'), vm = require('node:vm'), assert = require('node:assert/strict'), path = require('node:path');
const root = path.resolve(__dirname, '..'), html = fs.readFileSync(path.join(root, 'web/index.html'), 'utf8');
const nodes = new Map(), outputs = new Map();
function element(tag = 'div') {
  const listeners = new Map();
  return { tagName:tag.toUpperCase(), value:'', textContent:'', hidden:false, disabled:false, checked:false, options:[], children:[], dataset:{}, style:{}, firstChild:{nodeValue:''},
    classList:{toggle(){},add(){},remove(){}}, setAttribute(){},removeAttribute(){},setPointerCapture(){},
    addEventListener(type, fn){if(!listeners.has(type))listeners.set(type,[]);listeners.get(type).push(fn);},
    async emit(type){for(const fn of listeners.get(type)||[])await fn({target:this,preventDefault(){}});},
    replaceChildren(...children){this.children=children;if(this.tagName==='SELECT')this.options=children;},
    add(option){this.options.push(option);},append(...children){this.children.push(...children);},querySelectorAll(){return[];},
    getContext(){return{};},getBoundingClientRect(){return{left:0,top:0,width:640,height:320};},remove(){},click(){},reportValidity(){return true;} };
}
for(const match of html.matchAll(/<(\w+)\b([^>]*\bid="([^"]+)"[^>]*)>/g)){
  const[,tag,attrs,id]=match,node=element(tag);node.value=attrs.match(/\bvalue="([^"]*)"/)?.[1]||'';
  if(tag==='select'){
    const content=html.slice(match.index+match[0].length).split('</select>')[0];
    node.options=[...content.matchAll(/<option\b([^>]*)>([^<]*)/g)].map(o=>({value:o[1].match(/value="([^"]*)"/)?.[1]||'',textContent:o[2],selected:/\bselected\b/.test(o[1])}));
    node.value=(node.options.find(o=>o.selected)||node.options[0])?.value||'';
  }nodes.set(id,node);
}
const get=id=>{assert.ok(nodes.has(id),id);return nodes.get(id);};
const document={getElementById:get,createElement:element,createElementNS:(_,t)=>element(t),querySelector(s){if(!outputs.has(s))outputs.set(s,element());return outputs.get(s);},querySelectorAll(){return[];},addEventListener(){},body:element()};
const sandbox={document,window:{addEventListener(){}},setTimeout(){},clearTimeout(){},setInterval(){},requestAnimationFrame(){},console,
  Option:function(text,value){return{textContent:text,value:String(value)};},fetch:async()=>({ok:true,json:async()=>({})})};
vm.createContext(sandbox);
vm.runInContext(fs.readFileSync(path.join(root,'web/app.js'),'utf8').replace('  initialize();','  globalThis.app={state,terrain,terrainImport,globeOrientation,updateTerrainControls,applyTerrainStatus,reconcileTerrainExperiment,terrainSourceLabel,setApi(fn){api=fn;}};'),sandbox);
const app=sandbox.app, plain=x=>JSON.parse(JSON.stringify(x));
const deferred=()=>{let resolve;const promise=new Promise(r=>resolve=r);return{promise,resolve};};
const info={epochs:[{index:0,time_years:0},{index:20,time_years:2000000}],default_epoch:20};
const job={job_id:'solver-test',state:'running',source_type:'gospl_result',run_id:'gospl-result',source_result_epoch:20,time_myr:2,width:2048,height:1024,detail:0,source_path:'C:/World/output',orientation:{yaw:0,pitch:0,roll:0}};
let checks=0;
async function check(name,action){await action();checks++;console.log(`PASS ${name}`);}
(async()=>{
  await check('goSPL mode works without a tectonic run and starts with no added texture',async()=>{
    get('terrain-source-mode').value='gospl';await get('terrain-source-mode').emit('change');
    assert.equal(get('terrain-detail').value,'0');assert.equal(get('generate-terrain').disabled,true);
    assert.equal(get('terrain-import').hidden,false);assert.equal(get('terrain-native-options').hidden,true);
    app.setApi(async()=>info);get('terrain-import-path').value='C:/World/output';await get('terrain-import-path').emit('input');await get('inspect-gospl-result').emit('click');
    assert.equal(get('terrain-import-epoch').value,'20');assert.equal(get('generate-terrain').disabled,false);
  });
  await check('changing a path invalidates its epoch list and ignores the old response',async()=>{
    const wait=deferred();app.setApi(()=>wait.promise);const pending=get('inspect-gospl-result').emit('click');
    get('terrain-import-path').value='C:/World/other';await get('terrain-import-path').emit('input');
    wait.resolve(info);await pending;assert.equal(app.terrainImport.info,null);assert.equal(get('generate-terrain').disabled,true);
  });
  await check('import captures selected solver epoch without applying the tectonic globe rotation twice',async()=>{
    app.setApi(async()=>info);await get('inspect-gospl-result').emit('click');
    app.globeOrientation.output={yaw:75,pitch:20,roll:10};get('terrain-width').value='2048';
    let submitted;app.setApi(async(route,data)=>{submitted={route,data:plain(data)};return job;});
    await get('generate-terrain').emit('click');
    assert.deepEqual(submitted,{route:'/api/terrain/gospl',data:{path:'C:/World/other',epoch_index:20,width:2048,detail:0}});
    assert.equal(get('terrain-source-mode').disabled,true);assert.equal(get('terrain-import-path').disabled,true);
  });
  await check('completed landscapes stay reviewable across tectonic experiment switches with accurate source labels',async()=>{
    app.setApi(async()=>({jobs:[{...job,state:'complete'}]}));app.applyTerrainStatus({...job,state:'complete'});
    assert.equal(app.terrain.result.job_id,job.job_id);assert.equal(app.terrain.selectionManual,true);
    app.state.runId='a-different-world';app.reconcileTerrainExperiment();assert.equal(app.terrain.result.job_id,job.job_id);
    assert.match(get('terrain-result-source').textContent,/goSPL landscape.*epoch 20/);
    assert.doesNotMatch(get('terrain-result-source').textContent,/Different experiment|null/);
    assert.match(get('terrain-result-title').textContent,/solver time/);
  });
  await check('an older native epoch requires explicit opt-in to derived continental margins',async()=>{
    get('terrain-source-mode').value='history';await get('terrain-source-mode').emit('change');
    app.state.mode='history';app.state.currentIndex=3;app.state.frame={index:3,time_myr:6,width:512,height:256};
    app.globeOrientation.runId=app.state.runId;app.globeOrientation.output={yaw:75,pitch:20,roll:10};
    get('terrain-reconstruct-margins').checked=true;app.updateTerrainControls();
    let submitted;app.setApi(async(route,data)=>{submitted={route,data:plain(data)};return{...job,job_id:'native-test',source_type:'native_history',run_id:app.state.runId};});
    await get('generate-terrain').emit('click');assert.equal(submitted.route,'/api/terrain');
    assert.equal(submitted.data.reconstruct_margins,true);assert.deepEqual(submitted.data.orientation,{yaw:75,pitch:20,roll:10});
  });
  console.log(`${checks} terrain import UI journeys passed`);
})().catch(error=>{console.error(error);process.exitCode=1;});
