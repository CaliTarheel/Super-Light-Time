/* Optional crustal structure and persistent local trench history, without a server. */
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
  '  globalThis.app={state,trenchReview,hasGridField,foundationColor,foundationAccounting,updateFoundationLayers,updateFoundationInspection,updateInspection,updateTrenchHistory,trenchMilestones,updateStats,updateEvents,showHover,setVisit(fn){visitTime=fn;}};');
vm.runInContext(source,sandbox);
const app=sandbox.app;
const textOf=node=>[node.textContent,...node.children.map(textOf)].filter(Boolean).join('\n');
const baseFrame=()=>({width:4,height:2,time_myr:40,crust:[0,0,1,2,0,0,1,1],plate:Array(8).fill(0),
  elevation:[-4000,-3000,100,1000,-3000,-4000,300,500],age:Array(8).fill(20),boundary:Array(8).fill(0),
  plates:[{id:0,uid:7,name:'Incoming'},{id:1,uid:12,name:'Overriding'}]});
const basePoint=()=>({index:0,time_myr:40,elevation_m:500,uplift_m:800,extension_m:100,erosion_m:70,
  adjustment_m:0,relief_m:400,lon:0,lat:0,plate_id:0,plate_uid:7,plate_name:'Incoming',crust:1,boundary:0});
const structuralPoint=()=>({...basePoint(),structure_version:1,crustal_thickness_km:36.2,crustal_root_km:7.4,
  rift_thermal_support_m:240,rift_cooling_age_myr:12,foreland_deflection_m:400,denudation_m:500,rebound_m:430,
  thermal_subsidence_m:30,foreland_subsidence_m:20,rift_extension_m:50});
const trench=(id=2,phase='mature')=>({id,phase,episode:2,maturity:1,length_km:1300,active_myr:12,shortening_km:110,
  downgoing_plate_uid:7,overriding_plate_uid:12,history:[
    {time_myr:0,phase:'initiating',reason:'new_convergent_contact',episode:1},
    {time_myr:12,phase:'mature',reason:'mature_slab',episode:1},
    {time_myr:24,phase:'shutdown',reason:'continental_collision',episode:1},
    {time_myr:34,phase:'reactivated',reason:'renewed_convergence',episode:2}]});
function inspect(point){app.state.mode='history';app.state.frame=baseFrame();app.state.frames=[{time_myr:40}];
  app.state.currentIndex=0;app.state.inspection={mode:'material',trace_id:3,points:[point],events:[]};app.updateInspection();}
function showSystems(systems,time=40){app.state.mode='history';app.state.runId='fixture';
  app.state.frame={...baseFrame(),time_myr:time,trench_systems:systems};app.updateTrenchHistory();}
let checks=0;
async function check(name,action){await action();checks++;console.log(`PASS ${name}`);}
(async()=>{
  await check('new maps require actual complete fields, not a version claim',()=>{
    app.state.mode='history';app.state.frame={...baseFrame(),structure_version:1};app.updateFoundationLayers();
    assert.equal(get('layer-thickness').hidden,true);assert.equal(get('layer-basins').disabled,true);
    app.state.frame.crustal_thickness_km=[35];app.updateFoundationLayers();assert.equal(get('layer-thickness').hidden,true);
    app.state.frame.crustal_thickness_km=[7,7,35,42,7,7,36,38];app.updateFoundationLayers();
    assert.equal(get('layer-thickness').hidden,false);assert.equal(get('layer-thickness').disabled,false);
    assert.equal(get('layer-basins').hidden,true);
    app.state.frame.foreland_deflection_m=new Float32Array(8);app.updateFoundationLayers();assert.equal(get('layer-basins').hidden,false);
  });
  await check('legacy navigation clears a formerly selected foundation map',()=>{
    app.state.layer='basins';app.state.frame=baseFrame();app.updateStats();
    assert.equal(app.state.layer,'elevation');assert.equal(get('layer-basins').hidden,true);
    assert.equal(get('layer-thickness').hidden,true);assert.doesNotMatch(get('layer-legend').innerHTML,/FORELAND DEFLECTION/);
    app.state.layer='thickness';app.state.mode='initial';app.state.initial=null;app.updateStats();
    assert.equal(app.state.layer,'crust');assert.equal(get('layer-thickness').disabled,true);
  });
  await check('unavailable layer controls cannot select invented data',async()=>{
    app.state.mode='history';app.state.layer='elevation';app.state.frame=baseFrame();app.updateFoundationLayers();
    await get('layer-thickness').emit('click');assert.equal(app.state.layer,'elevation');
  });
  await check('quantitative palettes distinguish zero, high values and missing data',()=>{
    for(const layer of ['thickness','basins']){
      const zero=Array.from(app.foundationColor(layer,0)),high=Array.from(app.foundationColor(layer,10000));
      assert.notDeepEqual(zero,high);
      for(const invalid of [null,undefined,NaN,-1,'40']){
        const color=Array.from(app.foundationColor(layer,invalid));assert.notDeepEqual(color,zero);assert.notDeepEqual(color,high);
      }
      assert.ok(high.every(value=>value>=0&&value<=255));
    }
  });
  await check('new material accounting keeps rebound out of cumulative uplift',()=>{
    const point=structuralPoint();inspect(point);
    const totals=textOf(get('inspection-processes')),diagnostics=textOf(get('inspection-structure-values'));
    assert.match(totals,/Tectonic lowering\n100 m/);assert.match(totals,/Net erosion lowering\n70 m/);
    assert.match(totals,/Cumulative uplift\n800 m/);assert.doesNotMatch(totals,/rebound/i);
    assert.match(diagnostics,/Cumulative denudation \(gross\): 500 m/);
    assert.match(diagnostics,/Cumulative rebound · included in net erosion: 430 m/);
    assert.match(diagnostics,/Current rift thermal support: 240 m/);assert.match(diagnostics,/Crust thickness: 36.2 km/);
    assert.match(diagnostics,/Cooling subsidence · included in tectonic lowering: 30 m/);
    assert.match(get('inspection-accounting').textContent,/already included in tectonic lowering/);
    assert.match(get('inspection-structure-note').textContent,/must not be added to cumulative uplift/);
    assert.equal(point.uplift_m,800);assert.equal(point.erosion_m,70);
  });
  await check('legacy counters retain their historical interpretation after new traces',()=>{
    inspect(structuralPoint());inspect(basePoint());
    assert.match(textOf(get('inspection-processes')),/Extension loss/);
    assert.match(textOf(get('inspection-processes')),/Relief relaxation loss/);
    assert.doesNotMatch(get('inspection-accounting').textContent,/foreland|rebound/);
    assert.equal(get('inspection-structure').hidden,true);assert.equal(get('inspection-structure-values').children.length,0);
    assert.equal(get('inspection-structure-note').textContent,'');
    assert.equal(app.foundationAccounting(basePoint()),false);
  });
  await check('missing and no-cooling sentinels do not fabricate structural values',()=>{
    inspect({...basePoint(),rift_cooling_age_myr:-1,crustal_thickness_km:NaN});
    assert.equal(get('inspection-structure-values').children.length,1);
    assert.match(textOf(get('inspection-structure-values')),/No recorded cooling episode/);
    inspect({...basePoint(),rift_cooling_age_myr:0});assert.match(textOf(get('inspection-structure-values')),/Rift cooling age: 0 Myr/);
    app.state.mode='initial';app.updateInspection();assert.equal(get('inspection-structure').hidden,true);
    assert.equal(get('inspection-structure-values').children.length,0);
  });
  await check('no current material marker clears the previous structural inspection',()=>{
    inspect(structuralPoint());app.state.inspection=null;app.updateInspection();
    assert.equal(get('inspection-structure').hidden,true);assert.equal(get('inspection-structure-values').children.length,0);
  });
  await check('trench metadata preserves phase identity, owners and lineage without mutation',()=>{
    app.state.record=null;const row={...trench(),parent_trench_id:1,predecessor_trench_id:4};
    const source=JSON.stringify(row);showSystems([row]);const text=textOf(get('trench-detail'));
    assert.equal(get('trench-history').hidden,false);assert.match(text,/Incoming \(UID 7\)/);assert.match(text,/Overriding \(UID 12\)/);
    assert.match(text,/Phase: Mature/);assert.match(text,/Episode: 2/);assert.match(text,/Shortening this episode: 110 km/);
    assert.match(text,/Separated from: trench 1/);assert.match(text,/Preceding polarity: trench 4/);
    assert.equal(JSON.stringify(row),source);
  });
  await check('joined identities display their continuation without claiming shutdown',()=>{
    app.state.record=null;showSystems([{...trench(),phase:'joined',successor_trench_id:8}]);
    const text=textOf(get('trench-detail'));
    assert.match(text,/Phase: Continues in another trench/);
    assert.match(text,/Continues as: trench 8/);
    assert.doesNotMatch(text,/Phase: Shut down/);
  });
  await check('local history and nested global events merge into real chronological milestones',async()=>{
    const row=trench();const events=[
      {type:'trench_reactivated',time_myr:34,description:'Recorded restart',details:{trench_id:2,episode:2,phase:'initiating'}},
      {type:'trench_quiet',time_myr:60,description:'Later quiet',details:{trench_id:2,episode:2}},
      {type:'trench_shutdown',time_myr:66,details:{trench_id:9,episode:1}},
      {type:'collision',time_myr:50,details:{trench_id:2}},
      {type:'trench_quiet',time_myr:NaN,details:{trench_id:2}}];
    const milestones=app.trenchMilestones(row,events);
    assert.deepEqual(Array.from(milestones,event=>event.time_myr),[0,12,24,34,60]);
    assert.equal(milestones[3].type,'trench_reactivated');assert.equal(milestones[3].description,'Recorded restart');
    app.state.record={events};showSystems([row],40);const catalogue=get('trench-detail').children.at(-1);
    assert.equal(catalogue.children.length,5);assert.equal(catalogue.children.at(-1).classList.contains('future-event'),true);
    const visited=[];app.setVisit(time=>visited.push(time));await catalogue.children[0].emit('click');await catalogue.children[3].emit('click');
    assert.deepEqual(visited,[0,34]);
  });
  await check('default active trench and manual selection stay stable in sorted identities',async()=>{
    app.trenchReview.selectedId=null;app.state.record=null;
    const rows=[trench(3,'shutdown'),trench(2,'mature'),trench(1,'initiating')];showSystems(rows);
    assert.deepEqual(get('trench-select').options.map(row=>row.value),['1','2','3']);
    assert.equal(get('trench-select').value,'2');get('trench-select').value='3';await get('trench-select').emit('change');
    showSystems(rows,42);assert.equal(get('trench-select').value,'3');assert.match(textOf(get('trench-detail')),/Phase: Shut down/);
    assert.deepEqual(rows.map(row=>row.id),[3,2,1]);
    app.state.runId='different-run';app.updateTrenchHistory();assert.equal(get('trench-select').value,'2');
  });
  await check('legacy or editor states remove all stale trench review content',()=>{
    for(const value of [undefined,null,'legacy']){
      showSystems([trench()]);showSystems(value);assert.equal(get('trench-history').hidden,true);
      assert.equal(get('trench-detail').children.length,0);assert.equal(get('trench-select').options.length,0);
      assert.equal(get('trench-count').textContent,'');assert.equal(get('trench-description').textContent,'');
    }
    showSystems([]);assert.equal(get('trench-history').hidden,false);assert.equal(get('trench-select').disabled,true);
    assert.match(get('trench-description').textContent,/No local subduction systems recorded/);
    showSystems([trench()]);app.state.mode='initial';app.state.initial=null;app.updateStats();
    assert.equal(get('trench-history').hidden,true);assert.equal(get('trench-detail').children.length,0);
  });
  await check('hover reads mapped trench identity and selected structural quantity only',()=>{
    app.state.mode='history';app.state.layer='thickness';app.state.frame={...baseFrame(),trench:[0,0,2,0,0,0,0,0],
      trench_systems:[trench()],crustal_thickness_km:[7,7,36.2,40,7,7,35,35]};
    app.showHover({x:2,y:0});assert.match(get('map-hover').textContent,/Crust thickness: 36.2 km/);
    assert.match(get('map-hover').textContent,/Trench 2 · mature/);
    app.showHover({x:0,y:0});assert.doesNotMatch(get('map-hover').textContent,/Trench/);
    app.state.layer='elevation';app.state.frame=baseFrame();app.showHover({x:2,y:0});
    assert.doesNotMatch(get('map-hover').textContent,/Trench|Crust thickness|Foreland deflection/);
  });
  console.log(`${checks} foundation UI checks passed`);
})().catch(error=>{console.error(error);process.exitCode=1;});
