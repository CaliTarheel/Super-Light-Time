/* Review progressive rifts and preserve independent mechanics detail without a server. */
const fs = require('node:fs');
const vm = require('node:vm');
const assert = require('node:assert/strict');
const path = require('node:path');
const root = path.resolve(__dirname, '..');
const html = fs.readFileSync(path.join(root, 'web/index.html'), 'utf8');
const nodes = new Map(), outputs = new Map(), layers = [];
function element(tagName = 'div') {
  const listeners = new Map(), classes = new Set();
  return {tagName:String(tagName).toUpperCase(),textContent:'',innerHTML:'',hidden:false,checked:false,
    value:'',disabled:false,dataset:{},style:{},children:[],options:[],firstChild:{nodeValue:''},
    classList:{toggle(name,on){if(on)classes.add(name);else classes.delete(name);},add(name){classes.add(name);},remove(name){classes.delete(name);},contains:name=>classes.has(name)},
    setAttribute(){},removeAttribute(){},setPointerCapture(){},
    addEventListener(type,fn){if(!listeners.has(type))listeners.set(type,[]);listeners.get(type).push(fn);},
    async emit(type,event={}){for(const fn of listeners.get(type)||[])await fn({target:this,preventDefault(){},...event});},
    append(...children){this.children.push(...children);},add(option){this.options.push(option);},
    replaceChildren(...children){this.children=children;if(this.tagName==='SELECT')this.options=children;},
    querySelectorAll(){return [];},reportValidity(){return true;},click(){},remove(){},getContext(){return {};},
    getBoundingClientRect(){return {left:0,top:0,width:640,height:320};}};
}
for(const match of html.matchAll(/<(\w+)\b([^>]*\bid="([^"]+)"[^>]*)>/g)){
  const [,tag,attributes,id]=match,node=element(tag);
  node.value=attributes.match(/\bvalue="([^"]*)"/)?.[1]||'';node.checked=/\bchecked\b/.test(attributes);
  node.hidden=/\bhidden\b/.test(attributes);node.disabled=/\bdisabled\b/.test(attributes);
  if(tag==='select'){
    const content=html.slice(match.index+match[0].length).split('</select>')[0];
    node.options=[...content.matchAll(/<option\b([^>]*)>([^<]*)/g)].map(option=>({value:option[1].match(/value="([^"]*)"/)?.[1]||'',textContent:option[2],selected:/\bselected\b/.test(option[1])}));
    node.value=(node.options.find(option=>option.selected)||node.options[0])?.value||'';
  }
  nodes.set(id,node);
}
for(const match of html.matchAll(/<button\b([^>]*\bdata-layer="([^"]+)"[^>]*)>/g)){
  const id=match[1].match(/\bid="([^"]+)"/)?.[1],node=id?nodes.get(id):element('button');
  node.dataset.layer=match[2];layers.push(node);
}
const get=id=>{assert.ok(nodes.has(id),`HTML control exists: ${id}`);return nodes.get(id);};
const document={getElementById:get,createElement:element,createElementNS:(_,tag)=>element(tag),
  querySelector(selector){if(!outputs.has(selector))outputs.set(selector,element());return outputs.get(selector);},
  querySelectorAll:selector=>selector==='[data-layer]'?layers:[],addEventListener(){},body:element(),activeElement:null};
const sandbox={document,window:{addEventListener(){},localStorage:{getItem(){return null;},setItem(){}}},
  performance:{now:()=>0},setTimeout(){},clearTimeout(){},setInterval(){},requestAnimationFrame(){},console,
  Option:function(text,value){return {textContent:text,value};},fetch:async()=>({ok:true,json:async()=>({})})};
vm.createContext(sandbox);
const source=fs.readFileSync(path.join(root,'web/app.js'),'utf8').replace('  initialize();',
  '  globalThis.app={state,riftReview,readConfig,fillConfig,hasGridField,foundationColor,riftLoadingInfo,riftLoadingColor,riftLoadingHover,updateLegend,updateFoundationLayers,updateRiftHistory,riftMilestones,eventButton,updateStats,updateEvents,showHover,setVisit(fn){visitTime=fn;}};');
vm.runInContext(source,sandbox);
const app=sandbox.app;
const textOf=node=>[node.textContent,...node.children.map(textOf)].filter(Boolean).join('\n');
const baseFrame=()=>({width:4,height:2,time_myr:40,crust:[0,0,1,2,0,0,1,1],plate:Array(8).fill(0),
  elevation:[-4000,-3000,100,1000,-3000,-4000,300,500],age:Array(8).fill(20),boundary:Array(8).fill(0),
  plates:[{id:0,uid:7,name:'Incoming'},{id:1,uid:12,name:'Overriding'}]});
const rift=(id=2,phase='active')=>({id,phase,plate_uid:7,started_myr:10,last_active_myr:38,
  extension_km:34.5,peak_damage:.72,mean_strength:.82,geometry_xyz:[[1,0,0],[0,1,0]],
  history:[{time_myr:10,phase:'incipient',reason:'local_extension'},
    {time_myr:22,phase:'active',reason:'persistent_loading'},
    {time_myr:28,phase:'failed',reason:'unloading'},
    {time_myr:34,phase:'active',reason:'renewed_extension'}]});
function show(systems,time=40){app.state.mode='history';app.state.runId='fixture';
  app.state.frame={...baseFrame(),time_myr:time,rift_systems:systems};app.updateRiftHistory();}
let checks=0;
async function check(name,action){await action();checks++;console.log(`PASS ${name}`);}
(async()=>{
  await check('mechanical detail defaults independently and preserves arbitrary imported values',()=>{
    assert.equal(app.readConfig().mechanics_nodes,1024);
    const width=app.readConfig().width;
    app.fillConfig({mechanics_nodes:1536});assert.equal(app.readConfig().mechanics_nodes,1536);
    assert.equal(app.readConfig().width,width);
    assert.equal(get('mechanics_nodes').options.filter(row=>Number(row.value)===1536).length,1);
    app.fillConfig({mechanics_nodes:1536});assert.equal(get('mechanics_nodes').options.filter(row=>Number(row.value)===1536).length,1);
    app.fillConfig({width:96});assert.equal(app.readConfig().mechanics_nodes,1536);
    assert.equal(app.readConfig().height,48);
    assert.match(get('mechanics_nodes').options.find(row=>Number(row.value)===1536).textContent,/imported/);
  });
  await check('damage and strength layers require complete grids and clear in legacy or editor',()=>{
    app.state.mode='history';app.state.frame=baseFrame();app.updateFoundationLayers();
    for(const id of ['layer-damage','layer-weakness']){assert.equal(get(id).hidden,true);assert.equal(get(id).disabled,true);}
    app.state.frame.rift_damage=[.2];app.updateFoundationLayers();assert.equal(get('layer-damage').hidden,true);
    app.state.frame.rift_damage=Array(8).fill(.2);app.state.frame.rift_strength_relative=Array(8).fill(.8);
    app.updateFoundationLayers();assert.equal(get('layer-damage').hidden,false);assert.equal(get('layer-weakness').disabled,false);
    app.state.layer='damage';app.state.frame=baseFrame();app.updateFoundationLayers();assert.equal(app.state.layer,'elevation');
    app.state.layer='weakness';app.state.mode='initial';app.updateFoundationLayers();assert.equal(app.state.layer,'crust');
  });
  await check('relative quantities use distinct bounded colors and exact hover meaning',()=>{
    for(const layer of ['damage','weakness'])for(const value of [0,.3,1,3,5,100,NaN]){
      const rgb=app.foundationColor(layer,value);assert.equal(rgb.length,3);assert.ok(rgb.every(v=>Number.isInteger(v)&&v>=0&&v<=255));
    }
    assert.notDeepEqual(Array.from(app.foundationColor('damage',0)),Array.from(app.foundationColor('damage',1)));
    app.state.mode='history';app.state.frame={...baseFrame(),rift_damage:Array(8).fill(.72),rift_strength_relative:Array(8).fill(.82)};
    app.state.layer='damage';app.showHover({x:2,y:0});assert.match(get('map-hover').textContent,/Rift damage: 72%/);
    app.state.layer='weakness';app.showHover({x:2,y:0});assert.match(get('map-hover').textContent,/Relative lithospheric strength: 0.82 · lower is weaker/);
    assert.doesNotMatch(get('map-hover').textContent,/Foreland deflection|Crust thickness/);
  });
  await check('review shows local stretching, failure, strength and daughter identity without mutation',()=>{
    app.state.record=null;const row={...rift(),phase:'broken_through',new_plate_uid:12,source_rift_id:19};const before=JSON.stringify(row);
    show([row]);const text=textOf(get('rift-detail'));
    assert.equal(get('rift-history').hidden,false);assert.match(text,/Parent: Incoming \(UID 7\)/);
    assert.match(text,/New plate: Overriding \(UID 12\)/);assert.match(text,/Phase: Broken through/);
    assert.match(text,/Cumulative stretching: 34.5 km/);assert.match(text,/Peak damage: 72%/);
    assert.match(text,/Mean relative strength: 0.82/);assert.equal(JSON.stringify(row),before);
    assert.equal(get('rift-select').options[0].textContent,'Rift system 2 · Broken through');
    assert.match(text,/Linked rift scar: 19/);assert.match(text,/separate scar identity.*rift-inversion history/);
    assert.match(get('rift-description').textContent,/not current basin width/);
  });
  await check('system event labels distinguish structural scars without rewriting saved records',()=>{
    const event={type:'rift_active',time_myr:22,description:'Continental rift 2: continued stretching.',details:{rift_system_id:2}};
    const original=JSON.stringify(event);assert.match(textOf(app.eventButton(event)),/Rift system 2: continued stretching/);
    assert.equal(JSON.stringify(event),original);
    const legacy={...event,type:'rift_inversion',details:{source_rift_id:2}};
    assert.match(textOf(app.eventButton(legacy)),/Continental rift 2: continued stretching/);
    const mismatched={...event,details:{rift_system_id:19}};
    assert.match(textOf(app.eventButton(mismatched)),/Continental rift 2: continued stretching/);
    show([rift()]);assert.doesNotMatch(textOf(get('rift-detail')),/Linked rift scar/);
  });
  await check('milestones combine local phases and linked catalogue events, retaining future navigation',async()=>{
    const row=rift();const events=[
      {type:'rift_reactivated',time_myr:34,description:'Stretching resumed',details:{rift_system_id:2}},
      {type:'rift_active',time_myr:22,description:'Recorded active phase',details:{rift_system_id:2}},
      {type:'rift_breakthrough',time_myr:64,description:'Continental separation',details:{rift_system_id:2}},
      {type:'rift_failed',time_myr:50,details:{rift_system_id:9}},
      {type:'trench_mature',time_myr:60,details:{rift_system_id:2}}];
    const milestones=app.riftMilestones(row,events);
    assert.equal(milestones.length,5);assert.equal(milestones[1].description,'Recorded active phase');
    app.state.record={events};show([row]);const list=get('rift-detail').children.at(-1);
    assert.equal(list.children.at(-1).classList.contains('future-event'),true);
    let visited=null;app.setVisit(time=>visited=time);await list.children.at(-1).emit('click');assert.equal(visited,64);
  });
  await check('selection stays with its system across frames and resets for a new experiment',async()=>{
    app.state.record=null;app.riftReview.selectedId=null;
    const rows=[rift(3,'failed'),rift(2,'active'),rift(1,'incipient')];show(rows);
    assert.deepEqual(get('rift-select').options.map(row=>row.value),['1','2','3']);assert.equal(get('rift-select').value,'2');
    get('rift-select').value='3';await get('rift-select').emit('change');show(rows,42);
    assert.equal(get('rift-select').value,'3');assert.match(textOf(get('rift-detail')),/Phase: Failed \/ dormant/);
    app.state.runId='other';app.updateRiftHistory();assert.equal(get('rift-select').value,'2');
  });
  await check('mechanics diagnostics remain reviewable even before any rift forms',()=>{
    show([]);app.state.frame.rift_mechanics={mesh_nodes:900,mesh_edges:2800,target_nodes:1024,solver_iterations:15,residual:1e-6,converged:false};app.updateRiftHistory();
    assert.equal(get('rift-select').disabled,true);assert.equal(get('rift-mechanics-detail').hidden,false);
    assert.match(get('rift-mechanics-status').textContent,/900 nodes/);assert.match(get('rift-mechanics-status').textContent,/did not converge/);
    assert.match(get('rift-mechanics-status').textContent,/independent of the map/);
  });
  await check('local continental mechanics never inherits a plate-clock percentage or cooldown',()=>{
    const plate={id:0,internal_loading:{model:'local material extension and damage',eligible:true,
      accumulated_load_myr:999,rupture_threshold_myr:75,cooldown_remaining_myr:110}};
    const info=app.riftLoadingInfo(plate);assert.equal(info.kind,'local');assert.equal(info.ratio,null);
    assert.ok(app.riftLoadingColor(info).every(Number.isFinite));
    const hover=app.riftLoadingHover(plate).join('\n');assert.match(hover,/local stretching/);
    assert.doesNotMatch(hover,/999|110|%|cooldown/);
    app.state.mode='history';app.state.layer='loading';app.state.frame={...baseFrame(),plates:[plate]};app.updateLegend();
    assert.match(get('loading-map-note').textContent,/applies only to oceanic breakup/);
    assert.match(get('layer-legend').innerHTML,/Local continental mechanics/);
    const legacy={internal_loading:{eligible:true,accumulated_load_myr:37.5,rupture_threshold_myr:75,cooldown_remaining_myr:10}};
    assert.equal(app.riftLoadingInfo(legacy).ratio,.5);assert.match(app.riftLoadingHover(legacy).join('\n'),/50% of threshold/);
    app.state.frame=baseFrame();app.updateLegend();assert.doesNotMatch(get('layer-legend').innerHTML,/Local continental mechanics/);
  });
  await check('missing records and editor remove all stale review data',()=>{
    for(const absent of [undefined,null]){
      show([rift()]);show(absent);assert.equal(get('rift-history').hidden,true);
      assert.equal(get('rift-select').options.length,0);assert.equal(get('rift-detail').children.length,0);
      assert.equal(get('rift-count').textContent,'');assert.equal(get('rift-mechanics-status').textContent,'');
    }
    show([rift()]);app.state.mode='initial';app.state.initial=null;app.updateStats();
    assert.equal(get('rift-history').hidden,true);assert.equal(get('rift-detail').children.length,0);
  });
  console.log(`${checks} progressive rift UI checks passed`);
})().catch(error=>{console.error(error);process.exitCode=1;});
