/* Continental rift provenance and inversion review; no service or model run. */
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
const source=fs.readFileSync(path.join(root,'web/app.js'),'utf8').replace('  initialize();',
  '  globalThis.app={state,updateInspection,riftEvidenceRows,eventButton,setVisit(fn){visitTime=fn;}};');
vm.runInContext(source,sandbox);
const app=sandbox.app;
let checks=0;
async function check(name,action){await action();checks++;console.log(`PASS ${name}`);}
const base=(index=0)=>({index,time_myr:index*20,elevation_m:1000,uplift_m:100,extension_m:50,relief_m:500,
  lon:0,lat:0,plate_id:0,plate_uid:1,plate_name:'A',crust:1,boundary:0});
const scar=(index=0)=>({...base(index),rift_id:7,rift_birth_myr:0,rift_extension_m:0,inversion_uplift_m:0});
function inspect(points,index=points.length-1){
  app.state.mode='history';app.state.currentIndex=index;app.state.frame={time_myr:points[index].time_myr};
  app.state.frames=points.map(point=>({time_myr:point.time_myr}));
  app.state.inspection={mode:'material',trace_id:3,points,events:[]};app.updateInspection();
}
const processText=()=>get('inspection-processes').children.map(row=>row.children.map(child=>child.textContent).join(':')).join('\n');
(async()=>{
  await check('legacy traces add no rift story or unrecorded process totals',()=>{
    inspect([base()]);assert.equal(get('inspection-processes').children.length,5);
    assert.doesNotMatch(get('inspection-accounting').textContent,/Rift/);
    assert.equal(app.riftEvidenceRows([base()]).length,0);
  });
  await check('seeded time-zero scar does not invent observed stretching',()=>{
    const point=scar();inspect([point]);const rows=app.riftEvidenceRows([point]);
    assert.equal(rows.length,1);assert.match(rows[0].text,/scar created at 0 Myr/);
    assert.match(rows[0].text,/does not record stretching/);
    assert.match(processText(),/Rift origin:Rift 7/);
    assert.match(processText(),/Rift stretching · included above:0 m/);
    assert.match(get('inspection-accounting').textContent,/does not imply earlier stretching/);
  });
  await check('no-origin sentinel and missing birth time remain distinct from time zero',()=>{
    inspect([{...scar(),rift_id:-1,rift_birth_myr:-1}]);
    assert.doesNotMatch(processText(),/Rift origin/);assert.equal(app.riftEvidenceRows([{...scar(),rift_id:-1}]).length,0);
    const rows=app.riftEvidenceRows([{...scar(),rift_birth_myr:-1}]);assert.doesNotMatch(rows[0].text,/created at/);
  });
  await check('observed extension precedes inversion in reviewable saved evidence',()=>{
    const points=[scar(0),{...scar(1),rift_extension_m:30},{...scar(2),rift_extension_m:30,inversion_uplift_m:45},
      {...scar(3),rift_extension_m:30,inversion_uplift_m:60}];
    const rows=app.riftEvidenceRows(points);assert.equal(rows.length,3);
    assert.deepEqual(Array.from(rows,row=>row.time_myr),[0,20,40]);
    assert.match(rows[1].text,/stretching first present/);assert.match(rows[2].text,/inversion first present/);
    inspect(points);assert.match(get('inspection-accounting').textContent,/already included in extension loss/);
    assert.match(get('inspection-accounting').textContent,/already included in cumulative uplift; do not add it twice/);
    assert.equal(points[3].uplift_m,100);assert.equal(points[3].extension_m,50);
  });
  await check('first observation is not backdated to unrecorded extension or inversion onset',()=>{
    const rows=app.riftEvidenceRows([{...scar(3),rift_extension_m:30,inversion_uplift_m:60}]);
    assert.ok(rows.every(row=>row.time_myr===60));
    assert.match(rows[1].text,/saved material history/);assert.match(rows[2].text,/saved material history/);
  });
  await check('evidence links and inversion events navigate to their recorded time',async()=>{
    const visited=[];app.setVisit(time=>visited.push(time));
    inspect([scar(0),{...scar(1),rift_extension_m:10},{...scar(2),rift_extension_m:10,inversion_uplift_m:20}]);
    const link=get('inspection-changes').children.find(row=>row.textContent.includes('inversion first present'));
    await link.emit('click');assert.equal(visited.at(-1),40);
    for(const type of ['rift_inversion','rift_inversion_quiet']){
      const row=app.eventButton({type,time_myr:50,text:'Recorded rift process',plate_uids:[1]});
      await row.emit('click');assert.equal(visited.at(-1),50);
    }
  });
  console.log(`${checks} rift inversion UI checks passed`);
})().catch(error=>{console.error(error);process.exitCode=1;});
