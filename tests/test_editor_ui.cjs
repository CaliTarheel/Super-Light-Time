/* Focused editor journeys and independent spherical-area checks; run with node. */
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
    getContext() { return {}; }, getBoundingClientRect() { return {left:0,top:0,width:640,height:320}; } };
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
// This alert is created dynamically by the app and is not in index.html.
nodes.set('simulation-error', element('p'));
nodes.set('source-transition-note', element('p'));
const get = (id) => { assert.ok(nodes.has(id), `Referenced HTML control exists: ${id}`); return nodes.get(id); };
const document = { getElementById:get, createElement:element, createElementNS:(_,tag)=>element(tag),
  querySelector(selector) { if (!outputs.has(selector)) outputs.set(selector, element()); return outputs.get(selector); },
  querySelectorAll:()=>[], addEventListener(){}, body:element(), activeElement:null };
let fetchBody;
const sandbox = {document, URLSearchParams, structuredClone, location:{search:''}, window:{addEventListener(){}}, setTimeout(){}, clearTimeout(){}, setInterval(){},
  requestAnimationFrame(){}, console, Option:function(text,value){return {textContent:text,value};},
  fetch:async (_,options)=>{fetchBody=options?.body;return {ok:true,json:async()=>({})};} };
vm.createContext(sandbox);
const source = fs.readFileSync(path.join(root, 'web/app.js'),'utf8').replace('  initialize();',
  '  globalThis.app={state,mapView,editHistory,designEditor,readConfig,normalizeWorldDesign,updateCollisionHistory,updateAuthoredHistory,copyWorld,initialCoverage,sphericalBrushRows,fillMaterial,regridWorld,rememberEdit,resetEditHistory,restoreEdit,beginEdit,finishEdit,stamp,changeResolution,updateEditorControls,setRunningControls,applyStatus,pauseOrResume,regenerate,mapPoint,nativeApi:api,setApi(fn){api=fn;}};');
vm.runInContext(source,sandbox);
const app = sandbox.app;
let checks = 0;
async function check(name, action) { await action(); checks++; console.log(`PASS ${name}`); }
const world = (width=128,height=width/2) => ({width,height,crust:new Uint8Array(width*height)});
const point = (map,lon,lat) => ({x:(lon+180)/360*map.width-.5,y:(90-lat)/180*map.height-.5});
const paintedCap = (map,lon,lat,radius) => { for(const row of app.sphericalBrushRows(map,point(map,lon,lat),radius)) for(const [a,b] of row.spans) map.crust.fill(1,row.y*map.width+a,row.y*map.width+b); return map; };

(async()=>{
  await check('classic preset remains the explicit default',()=>assert.equal(get('initial-preset').value,'pangaea'));
  await check('brush area follows the spherical-cap formula at several grids and latitudes',()=>{
    const expected=(1-Math.cos(25*Math.PI/180))/2;
    for(const width of [128,512,2048]) for(const latitude of [0,70]) {
      const map=paintedCap(world(width),-177,latitude,25), actual=app.initialCoverage(map).continental;
      assert.ok(Math.abs(actual-expected)<(width===128?.002:.0006), `${width},${latitude}: ${actual} versus ${expected}`);
    }
  });
  await check('polar brush covers all longitudes without crossing into the other pole',()=>{
    const map=world(512), rows=app.sphericalBrushRows(map,point(map,30,89),20);
    assert.ok(rows.some(row=>row.y===0 && row.spans.some(([a,b])=>a===0 && b===512)));
    assert.ok(rows.every(row=>row.y<map.height/2));
  });
  await check('seam brush and preview use wrapped spans; a tiny brush still selects a cell',()=>{
    const map=world(128), rows=app.sphericalBrushRows(map,{x:0,y:32},12);
    assert.ok(rows.some(row=>row.spans.length===2));
    const tiny=app.sphericalBrushRows(world(64),{x:20,y:10},.25);
    assert.equal(tiny.reduce((sum,row)=>sum+row.spans.reduce((n,[a,b])=>n+b-a,0),0),1);
    assert.match(source,/for \(const row of sphericalBrushRows\(data, state.hover, state.brushSize\)\)/);
  });
  await check('coverage weights true globe area and cratons are a subset of crust',()=>{
    const map=world(16,8); map.crust.fill(2,0,16); const coverage=app.initialCoverage(map);
    assert.ok(Math.abs(coverage.continental-(1-Math.cos(Math.PI/8))/2)<1e-12);
    assert.equal(coverage.craton,coverage.continental);
  });
  await check('fill wraps the seam but leaves disconnected material and cratons alone',()=>{
    const map=world(12,6); map.crust[24]=1; map.crust[35]=1; map.crust[29]=1; map.crust[25]=2;
    assert.equal(app.fillMaterial(map,24,2),2); assert.equal(map.crust[35],2); assert.equal(map.crust[29],1);
  });
  await check('fill continues across one polar cap without joining north and south',()=>{
    const map=world(12,6); map.crust[0]=1; map.crust[6]=1; map.crust[60]=1;
    assert.equal(app.fillMaterial(map,0,2),2); assert.equal(map.crust[6],2); assert.equal(map.crust[60],1);
  });
  await check('large connected fill completes with a bounded iterative queue',()=>{
    const map=world(2048); assert.equal(app.fillMaterial(map,0,1),map.crust.length);
    assert.ok(map.crust.every(value=>value===1));
  });
  app.state.status={state:'paused',can_resume:true,run_id:'saved-checkpoint',config:{width:128,height:64}};
  app.state.runId='saved-checkpoint'; app.state.mode='initial'; app.state.recordCount=0; app.state.initial=world();
  await check('a brush stroke can be undone and redone byte-for-byte',async()=>{
    app.resetEditHistory(); app.state.brush=2; app.state.brushSize=12;
    const before=Array.from(app.state.initial.crust);
    await get('world-map').emit('pointerdown',{button:0,pointerId:1,clientX:200,clientY:100});
    await get('world-map').emit('pointerup');
    const after=Array.from(app.state.initial.crust); assert.notDeepEqual(after,before);
    await get('undo-edit').emit('click'); assert.deepEqual(Array.from(app.state.initial.crust),before);
    await get('redo-edit').emit('click'); assert.deepEqual(Array.from(app.state.initial.crust),after);
  });
  await check('resolution selection preserves categorical painting and undo restores dimensions',async()=>{
    const original=app.copyWorld(app.state.initial); get('width').value='512';
    await get('width').emit('change'); assert.equal(app.state.initial.width,512);
    const returned=app.regridWorld(app.state.initial,original.width,original.height);
    assert.deepEqual(Array.from(returned.crust),Array.from(original.crust));
    await get('undo-edit').emit('click'); assert.equal(app.state.initial.width,original.width);
    assert.deepEqual(Array.from(app.state.initial.crust),Array.from(original.crust));
    assert.equal(app.state.status.run_id,'saved-checkpoint');
  });
  await check('fill tool is one undoable action through its actual click handler',async()=>{
    app.state.initial=world(128); app.state.initial.crust.fill(1); app.state.brush=2; app.resetEditHistory();
    await get('fill-tool').emit('click'); await get('world-map').emit('pointerdown',{button:0,clientX:320,clientY:160});
    assert.ok(app.state.initial.crust.every(value=>value===2)); assert.equal(app.editHistory.undo.length,1);
    await get('undo-edit').emit('click'); assert.ok(app.state.initial.crust.every(value=>value===1));
  });
  await check('undo storage remains at most eight maps and 16 MiB',()=>{
    app.resetEditHistory(); const large=world(2048);
    for(let i=0;i<12;i++) app.rememberEdit(app.copyWorld(large));
    assert.equal(app.editHistory.undo.length,8);
    assert.ok([...app.editHistory.undo,...app.editHistory.redo].reduce((sum,map)=>sum+map.crust.byteLength,0)<=16*1024*1024);
  });
  await check('zoomed initial-map clicks map back to the correct source cell',()=>{
    app.state.initial=world(128); app.mapView.zoom=4; app.mapView.cx=.75; app.mapView.cy=.25;
    const selected=app.mapPoint({clientX:320,clientY:160}); assert.equal(selected.x,96); assert.equal(selected.y,16);
    app.mapView.zoom=1; app.mapView.cx=app.mapView.cy=.5;
  });
  await check('typed editor rasters are serialized as JSON arrays',async()=>{
    await app.nativeApi('/unused',{initial:app.state.initial});
    assert.ok(Array.isArray(JSON.parse(fetchBody).initial.crust));
  });
  await check('Highland 65 preset keeps its plate-topology seed and remains undoable',async()=>{
    app.state.initial=world(128); app.resetEditHistory(); const previous=Array.from(app.state.initial.crust);
    get('initial-preset').value='highland65'; let preset;
    const topology={version:1,mode:'passive_margin_aprons',seed:41,passive_coast_fraction:.65,apron_width_km:650,max_attached_ocean_fraction:.45};
    const subduction={enabled:true,initial_slab_depth_km:100,target_margin_fraction:.45,selection_seed:41};
    const lifecycle={version:1,enabled:true,neck_viscosity_pa_s:1e23,neck_thickness_km:80,neck_length_km:100,failure_opening_km:80,support_radius_km:1500,max_source_step_myr:.25,replacement_delay_myr:20,replacement_exclusion_km:600};
    const traction={version:1,enabled:true,reference_strength_n_per_m:1e13,mobility_km_myr:12,reach_km:900,max_equivalent_speed_km_myr:40,boundary_motion_fraction:.25};
    app.setApi(async(route,body)=>{assert.equal(route,'/api/initial');preset=body.preset;const map=world(128);map.crust.fill(2,0,1000);map.initial_plate_topology=topology;map.initial_subduction=subduction;map.continental_lifecycle=lifecycle;map.rift_traction=traction;return map;});
    await get('regenerate').emit('click'); assert.equal(preset,'highland65'); assert.equal(app.state.initial.crust[0],2);
    assert.deepEqual(JSON.parse(JSON.stringify(app.state.initial.initial_plate_topology)),topology);
    assert.deepEqual(JSON.parse(JSON.stringify(app.state.initial.initial_subduction)),subduction);
    assert.deepEqual(JSON.parse(JSON.stringify(app.state.initial.continental_lifecycle)),lifecycle);
    assert.deepEqual(JSON.parse(JSON.stringify(app.state.initial.rift_traction)),traction);
    await get('undo-edit').emit('click'); assert.deepEqual(Array.from(app.state.initial.crust),previous);
    assert.equal(app.state.initial.initial_plate_topology,undefined);
  });
  await check('pause and resume guard transitions and preserve an edited starting world',async()=>{
    const draft=app.copyWorld(app.state.initial), requests=[]; let release;
    let status={state:'running',run_id:'saved-checkpoint',can_resume:false,frames:[],time_myr:14,duration_myr:1000,config:{width:128,height:64}};
    app.state.status=status; app.state.runId=status.run_id; app.state.recordCount=0; app.state.frames=[];
    app.setApi(async(route,body)=>{
      if(route==='/api/pause'||route==='/api/resume'){requests.push({route,body});return await new Promise(resolve=>{release=resolve;});}
      if(route==='/api/status')return status;
      if(route==='/api/runs')return {runs:[]};
      return {events:[]};
    });
    const pausing=app.pauseOrResume(); assert.equal(app.state.status.state,'pausing'); assert.equal(get('pause-button').disabled,true);
    status={...status,state:'paused',can_resume:true}; release(status); await pausing;
    assert.equal(get('pause-button').textContent,'Resume'); assert.equal(get('pause-button').disabled,false);
    await app.applyStatus({...status}); assert.deepEqual(Array.from(app.state.initial.crust),Array.from(draft.crust));
    const resuming=app.pauseOrResume(); assert.equal(app.state.status.state,'resuming'); assert.equal(get('pause-button').disabled,true);
    status={...status,state:'running',can_resume:false}; release(status); await resuming;
    assert.equal(requests.length,2); for(const request of requests) assert.deepEqual(JSON.parse(JSON.stringify(request.body)),{run_id:'saved-checkpoint'});
    assert.deepEqual(Array.from(app.state.initial.crust),Array.from(draft.crust));
  });
  await check('legacy cancelled runs clearly show that no checkpoint can resume',async()=>{
    app.state.status={state:'cancelled',can_resume:false}; app.setRunningControls();
    assert.equal(get('pause-button').disabled,true); assert.match(get('resume-note').textContent,/no saved checkpoint/);
  });
  await check('starting a run selects the new experiment; ordinary refresh preserves deliberate choices',async()=>{
    const oldId='previous-experiment', newId='new-experiment';
    app.state.status={state:'paused',can_resume:true,run_id:oldId}; app.state.runId=oldId;
    app.state.initial=world(128); get('width').value='128'; get('saved-runs').value=oldId;
    const status={state:'running',run_id:newId,can_resume:false,frames:[],time_myr:0,duration_myr:1000};
    app.setApi(async route=>{
      if(route==='/api/run'||route==='/api/status')return status;
      if(route==='/api/runs')return {runs:[{run_id:newId,title:'New experiment',state:'running'},{run_id:oldId,title:'Previous experiment',state:'paused'}]};
      return {events:[],frame_count:0};
    });
    await get('config-form').emit('submit');
    assert.equal(app.state.runId,newId); assert.equal(get('saved-runs').value,newId,get('toast').textContent);
    app.state.status={...status,state:'paused',can_resume:true}; get('saved-runs').value=oldId;
    await get('refresh-runs').emit('click'); assert.equal(get('saved-runs').value,oldId);
  });
  await check('regional preview and map selection do not paint or add an intervention',async()=>{
    app.state.status={state:'paused',can_resume:true}; app.state.mode='initial'; app.state.initial=world(128); app.resetEditHistory();
    const before=Array.from(app.state.initial.crust);
    await get('design-preview').emit('click'); assert.equal(app.designEditor.preview.id,'preview');
    await get('design-pick').emit('click');
    await get('world-map').emit('pointerdown',{button:0,pointerId:1,clientX:635,clientY:8});
    assert.ok(Number(get('design-lon').value)>170); assert.ok(Number(get('design-lat').value)>80);
    assert.deepEqual(Array.from(app.state.initial.crust),before); assert.equal(app.editHistory.undo.length,0);
    assert.equal(app.state.initial.world_design,undefined);
  });
  await check('authored regions add with reason and duration and undo/redo independently of paint',async()=>{
    get('design-reason').value='Keep the northern interior coherent.';
    await get('design-add').emit('click');
    const design=JSON.parse(JSON.stringify(app.state.initial.world_design));
    assert.equal(design.interventions.length,1); assert.equal(design.enabled,true); assert.equal(design.revision,1);
    assert.equal(design.interventions[0].end_myr,350); assert.equal(design.interventions[0].reason,get('design-reason').value);
    await get('undo-edit').emit('click'); assert.equal(app.state.initial.world_design,undefined);
    await get('redo-edit').emit('click'); assert.deepEqual(JSON.parse(JSON.stringify(app.state.initial.world_design)),design);
    const copied=app.copyWorld(app.state.initial); copied.world_design.interventions[0].reason='Independent edit';
    assert.notEqual(app.state.initial.world_design.interventions[0].reason,'Independent edit');
  });
  await check('turning authored regions off preserves portable input and can be undone',async()=>{
    const before=JSON.parse(JSON.stringify(app.state.initial.world_design));
    get('world-design-enabled').checked=false; await get('world-design-enabled').emit('change');
    const config=app.readConfig(); assert.equal(config.world_design.enabled,false);
    assert.deepEqual(JSON.parse(JSON.stringify(config.world_design.interventions)),before.interventions);
    await get('undo-edit').emit('click'); assert.equal(app.state.initial.world_design.enabled,true);
    await get('design-remove').emit('click'); assert.equal(app.state.initial.world_design.interventions.length,0);
    await get('undo-edit').emit('click'); assert.equal(app.state.initial.world_design.interventions.length,1);
  });
  await check('world import round-trips authored controls, rejects invalid controls atomically and preserves undo',async()=>{
    const portable=JSON.parse(JSON.stringify(app.state.initial.world_design));
    const payload={...world(128),crust:Array(8192).fill(1),world_design:portable};
    app.state.initial=world(128); app.resetEditHistory();
    get('initial-file').files=[{size:100,text:async()=>JSON.stringify(payload)}];
    await get('initial-file').emit('change');
    assert.deepEqual(JSON.parse(JSON.stringify(app.state.initial.world_design)),portable);
    assert.deepEqual(JSON.parse(JSON.stringify(app.readConfig().world_design)),portable);
    const invalid=JSON.parse(JSON.stringify(payload)); invalid.world_design.interventions[0].end_myr=-1;
    get('initial-file').files=[{size:100,text:async()=>JSON.stringify(invalid)}];
    await get('initial-file').emit('change');
    assert.deepEqual(JSON.parse(JSON.stringify(app.state.initial.world_design)),portable);
    assert.match(get('toast').textContent,/Could not import/);
    await get('undo-edit').emit('click'); assert.equal(app.state.initial.world_design,undefined);
  });
  await check('regional controls are locked while a simulation is active',()=>{
    app.state.status={state:'running'}; app.setRunningControls();
    for(const id of ['world-design-enabled','design-add','design-pick','design-remove','design-branch']) assert.equal(get(id).disabled,true);
    app.state.status={state:'paused',can_resume:true}; app.setRunningControls();
  });
  await check('branching sends only authored controls and the checkpoint identity, then selects a new paused experiment',async()=>{
    const parent='branch-parent', branch='branch-child';
    app.state.status={state:'paused',can_resume:true,run_id:parent,time_myr:50}; app.state.runId=parent;
    app.state.mode='initial'; app.state.frames=[]; app.state.recordCount=0; app.state.initial=world(128);
    app.state.initial.world_design=app.normalizeWorldDesign({enabled:true,revision:1,interventions:[{id:'future',kind:'weak',lon_deg:179,lat_deg:70,radius_deg:8,strength:1,start_myr:50,end_myr:150,reason:'Encourage a future weak margin.'}]});
    const portable=JSON.parse(JSON.stringify(app.state.initial.world_design)); let request;
    const status={state:'paused',can_resume:true,run_id:branch,time_myr:50,duration_myr:350,frames:[],config:{width:128,height:64,world_design:portable},branch_origin:{run_id:parent,time_myr:50}};
    app.setApi(async(route,body)=>{
      if(route==='/api/branch'){request=JSON.parse(JSON.stringify(body));return status;}
      if(route==='/api/initial')return {...world(128),world_design:portable};
      if(route==='/api/runs')return {runs:[status,{run_id:parent,state:'paused',time_myr:50}]};
      return {events:[],frame_count:0};
    });
    app.updateEditorControls(); assert.equal(get('design-branch').disabled,false);
    await get('design-branch').emit('click');
    assert.deepEqual(request,{run_id:parent,world_design:portable});
    assert.equal(app.state.runId,branch); assert.equal(app.state.status.state,'paused');
    assert.equal(get('saved-runs').value,branch); assert.equal(app.state.loadingRun,false);
    app.state.status.can_resume=false; app.updateEditorControls();
    assert.equal(get('design-branch').disabled,true);
  });
  await check('collision review distinguishes active, quiet and accreted records without inventing legacy data',()=>{
    app.state.mode='history'; app.state.frame={time_myr:100}; app.updateCollisionHistory();
    assert.equal(get('collision-history').hidden,true);
    app.state.frame={time_myr:100,collision_contact_version:1,collision_diagnostics:{buried_area_quadrature_km2:1234,maximum_pair_stack_thickness_km:71},collision_contacts:[
      {id:1,state:'active',top_sheet:12,under_sheet:23,started_myr:10,last_seen_myr:100,overlap_area_km2:1200,suture_strength:.5},
      {id:2,state:'quiet',top_sheet:14,under_sheet:28,started_myr:20,last_seen_myr:70,overlap_area_km2:0,suture_strength:.8},
      {id:3,state:'accreted',top_sheet:16,under_sheet:29,started_myr:30,last_seen_myr:100,overlap_area_km2:300,suture_strength:1}]};
    app.updateCollisionHistory(); assert.equal(get('collision-history').hidden,false);
    assert.equal(get('collision-count').textContent,'1 ACTIVE · 3 RETAINED');
    const lines=get('collision-contacts').children.map(row=>row.textContent).join('\n');
    assert.match(lines,/sheet 12 above 23/); assert.match(lines,/quiet/); assert.match(lines,/accreted/);
    assert.match(lines,/last contact 70 Myr/); assert.match(lines,/suture 50%/);
    assert.match(get('collision-description').textContent,/four samples per contacted triangle/);
  });
  await check('authored history shows saved reasons and distinguishes the end of a timed intervention',()=>{
    app.state.mode='history'; app.state.frame={time_myr:20,world_design:{version:1,enabled:true,revision:2,interventions:[
      {id:'author-one',kind:'weak',lon_deg:10,lat_deg:20,radius_deg:5,strength:.5,start_myr:10,end_myr:20,reason:'Encourage this corridor.'}]}};
    app.updateAuthoredHistory(); assert.equal(get('authored-history').hidden,false);
    assert.match(get('authored-count').textContent,/0 ACTIVE/);
    assert.match(get('authored-regions').children[0].textContent,/inactive now.*Encourage this corridor/);
    app.state.frame.time_myr=19; app.updateAuthoredHistory(); assert.match(get('authored-count').textContent,/1 ACTIVE/);
    app.state.mode='initial'; app.updateAuthoredHistory(); assert.equal(get('authored-history').hidden,true);
  });
  await check('new HTML controls have unique IDs and keep arc recording semantics',()=>{
    const ids=[...html.matchAll(/\bid="([^"]+)"/g)].map(match=>match[1]); assert.equal(new Set(ids).size,ids.length);
    assert.match(html,/max="45"/); assert.match(html,/Highland 65.*six continents/); assert.match(html,/one integration step, not an interval total/);
    assert.doesNotMatch(source,/cancel-button/); assert.match(html,/app\.js\?v=[a-z0-9-]+/);
  });
  console.log(`${checks} editor and pause UI checks passed`);
})().catch(error=>{console.error(error);process.exitCode=1;});
