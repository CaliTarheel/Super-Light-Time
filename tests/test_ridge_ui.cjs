/* Ridge history and spherical overlay checks; no model or service is run. */
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
const sandbox = {performance:{now:()=>clockMs}, document, window:{addEventListener(){},localStorage:{getItem:key=>storage.get(key)??null,setItem:(key,value)=>storage.set(key,value)}}, setTimeout(){}, clearTimeout(){}, setInterval(){},
  requestAnimationFrame(){}, console, Option:function(text,value){return {textContent:text,value};},
  fetch:async (_,options)=>{fetchBody=options?.body;return {ok:true,json:async()=>({})};} };
vm.createContext(sandbox);
const draws=[];
const context={save(){},restore(){},beginPath(){},rect(){},clip(){},setLineDash(){},stroke(){},fill(){},
  moveTo(x,y){draws.push(['M',x,y]);},lineTo(x,y){draws.push(['L',x,y]);},arc(x,y){draws.push(['A',x,y]);},
  measureText(text){return {width:text.length*6};},fillText(text,x,y){draws.push(['T',x,y]);}};
get('world-map').getContext=()=>context;get('world-map').width=640;get('world-map').height=320;
const source=fs.readFileSync(path.join(root,'web/app.js'),'utf8').replace('  initialize();',
  '  globalThis.app={state,mapView,activeRidgeEpisodes,slabWindowProjection,updateSlabWindowNote,drawSlabWindows,updateInspection};');
vm.runInContext(source,sandbox);
const app=sandbox.app;
let checks=0;
function check(name,action){action();checks++;console.log(`PASS ${name}`);}
const xyz=(lon,lat)=>{lon*=Math.PI/180;lat*=Math.PI/180;return [Math.cos(lat)*Math.cos(lon),Math.cos(lat)*Math.sin(lon),Math.sin(lat)];};
const episode=(center=xyz(0,0))=>({id:1,center,radius_km:450,started_myr:0,peak_myr:8,end_myr:32,heat:.75});
const frame=(episodes=[episode()],time=8)=>({time_myr:time,ridge_episodes:episodes});
check('legacy, malformed and expired episodes create no invented footprint',()=>{
  assert.equal(app.activeRidgeEpisodes({time_myr:8}).length,0);
  const valid=episode();
  assert.equal(app.activeRidgeEpisodes(frame([valid,{...valid,center:[0,0,0]},{...valid,center:[NaN,0,1]},
    {...valid,started_myr:10},{...valid,end_myr:8},{...valid,radius_km:0}])).length,1);
  assert.equal(app.activeRidgeEpisodes(frame([valid],0)).length,1);
  assert.equal(app.activeRidgeEpisodes(frame([valid],32)).length,0);
});
check('spherical outlines preserve their angular radius at seam and both poles',()=>{
  for(const center of [xyz(179,0),xyz(-179,60),xyz(28,88),xyz(75,-89),[0,0,1],[0,0,-1]]){
    const original=JSON.stringify(center), ep=episode(center), projected=app.slabWindowProjection(ep);
    for(let i=0;i<projected.ring.length;i++){
      const [x,y]=projected.ring[i];assert.ok(Number.isFinite(x)&&y>=0&&y<=1);
      const p=xyz(x*360-180,90-y*180);
      const dot=p.reduce((sum,v,j)=>sum+v*center[j],0);
      assert.ok(Math.abs(dot-Math.cos(ep.radius_km/6371))<1e-12);
      if(i)assert.ok(Math.abs(x-projected.ring[i-1][0])<=.5,'no map-wide seam chord');
    }
    assert.equal(JSON.stringify(center),original,'source coordinates remain unchanged');
  }
});
check('rotated episode centers move in the same map coordinate convention',()=>{
  const first=app.slabWindowProjection(episode([1,0,0]));
  const yaw90=app.slabWindowProjection(episode([0,1,0]));
  const pitch90=app.slabWindowProjection(episode([0,0,-1]));
  assert.equal(first.center[0],.5);assert.equal(yaw90.center[0],.75);
  assert.equal(yaw90.center[1],.5);assert.equal(pitch90.center[1],1);
});
check('seam and polar drawing uses finite bounded segments at any zoom',()=>{
  for(const zoom of [1,4]){
    app.mapView.zoom=zoom;draws.length=0;
    app.drawSlabWindows(frame([episode(xyz(179,0)),episode([0,0,1])]));
    assert.ok(draws.some(d=>d[0]==='T'));
    for(let i=0;i<draws.length;i++){
      assert.ok(draws[i].slice(1).every(Number.isFinite));
      if(draws[i][0]==='L')assert.ok(Math.abs(draws[i][1]-draws[i-1][1])<=320);
    }
  }
});
check('legacy and editor views hide episode status; toggling does not lose recorded events',()=>{
  app.state.mode='initial';app.updateSlabWindowNote(frame());
  assert.equal(get('slab-window-note').hidden,true);assert.equal(get('show-slab-windows').disabled,true);
  app.state.mode='history';app.updateSlabWindowNote({time_myr:8});assert.equal(get('slab-window-note').hidden,true);
  app.updateSlabWindowNote(frame());assert.equal(get('slab-window-note').hidden,false);
  assert.match(get('slab-window-note').textContent,/1 active slab window/);
  get('show-slab-windows').checked=false;app.updateSlabWindowNote(frame());
  assert.match(get('slab-window-note').textContent,/Turn on Slab windows/);
  app.updateSlabWindowNote(frame([],40));assert.equal(get('slab-window-note').hidden,true);
});
check('inspector distinguishes subset uplift from additional temporary heat and absent history',()=>{
  const point={index:0,time_myr:8,elevation_m:1200,uplift_m:100,relief_m:900,lon:0,lat:0,plate_id:0,plate_name:'A',crust:1,boundary:0};
  app.state.mode='history';app.state.currentIndex=0;app.state.frame=frame();app.state.frames=[{time_myr:8}];
  app.state.inspection={mode:'material',trace_id:3,points:[point],events:[]};app.updateInspection();
  assert.equal(get('inspection-processes').children.length,5);
  assert.doesNotMatch(get('inspection-accounting').textContent,/Ridge-subduction|Slab-window/);
  point.ridge_uplift_m=25;point.ridge_thermal_m=40;app.updateInspection();
  assert.equal(get('inspection-processes').children.length,7);
  assert.match(get('inspection-accounting').textContent,/already included.*do not add it twice/);
  assert.match(get('inspection-accounting').textContent,/temporary offset additional.*not a cumulative/);
  assert.equal(point.uplift_m,100);assert.equal(point.relief_m,900);
});
console.log(`${checks} ridge UI checks passed`);
