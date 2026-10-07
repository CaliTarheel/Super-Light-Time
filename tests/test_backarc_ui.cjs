/* Saved back-arc basin review, milestones and legacy compatibility. */
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
const sandbox = {performance:{now:()=>clockMs}, document, URLSearchParams, location:{search:''}, window:{addEventListener(){},localStorage:{getItem:key=>storage.get(key)??null,setItem:(key,value)=>storage.set(key,value)}}, setTimeout(){}, clearTimeout(){}, setInterval(){},
  requestAnimationFrame(){}, console, Option:function(text,value){return {textContent:text,value};},
  fetch:async (_,options)=>{fetchBody=options?.body;return {ok:true,json:async()=>({})};} };
vm.createContext(sandbox);
const source=fs.readFileSync(path.join(root,'web/app.js'),'utf8').replace('  initialize();',
  '  globalThis.app={state,backarcReview,updateBackArcHistory,backArcMilestones,setVisit(fn){visitTime=fn;}};');
vm.runInContext(source,sandbox);
const app=sandbox.app;
let checks=0;
async function check(name,action){await action();checks++;console.log(`PASS ${name}`);}
const basin=(phase='loading')=>({id:2,phase,started_myr:20,parent_plate_uid:7,arc_plate_uid:null,downgoing_plate_uid:9,
  center:[1,0,0],opening_km:0,extension_rate_km_myr:0,loading_rate_km_myr:4,loading_km:12,last_active_myr:20});
const textOf=node=>[node.textContent,...node.children.map(textOf)].join('\n');
function show(rows,time=20){app.state.mode='history';app.state.runId='test-world';app.state.frame={time_myr:time,backarc_basins:rows,
  plates:[{id:0,uid:7,name:'Parent'},{id:1,uid:9,name:'Incoming'},{id:2,uid:12,name:'Young arc'}]};app.updateBackArcHistory();}
(async()=>{
  await check('legacy absence, empty saved lists and editor state are distinct',()=>{
    app.state.mode='history';app.state.frame={time_myr:20};app.updateBackArcHistory();assert.equal(get('backarc-history').hidden,true);
    show([]);assert.equal(get('backarc-history').hidden,false);assert.match(get('backarc-description').textContent,/No back-arc basins recorded/);
    app.state.mode='initial';app.updateBackArcHistory();assert.equal(get('backarc-history').hidden,true);
  });
  await check('loading has no invented opening or separated arc; missing figures stay unknown',()=>{
    show([basin()]);let text=textOf(get('backarc-basins'));
    assert.match(text,/Basin 2 · Loading/);assert.match(text,/Accumulated extension: 0 km/);assert.match(text,/Arc: not separated/);
    assert.match(text,/Current opening rate: 0 km\/Myr/);assert.match(text,/Back-arc loading rate: 4 km\/Myr/);
    show([{id:2,phase:'loading',parent_plate_uid:7}]);text=textOf(get('backarc-basins'));
    assert.match(text,/Accumulated extension: Not recorded/);assert.match(text,/Current opening rate: Not recorded/);
    assert.doesNotMatch(text,/Not recorded.*0 km/);
  });
  await check('saved phases and measured rates survive quiet and closed histories',()=>{
    for(const phase of ['rifting','spreading','quiet','closed']){
      show([{...basin(phase),opening_km:84,extension_rate_km_myr:phase==='quiet'||phase==='closed'?0:2.5,arc_plate_uid:12,rupture_myr:28}],50);
      const text=textOf(get('backarc-basins'));assert.match(text,new RegExp(`Basin 2 · ${phase[0].toUpperCase()+phase.slice(1)}`));
      assert.match(text,/Accumulated extension: 84 km/);assert.match(text,/Young arc \(UID 12\)/);assert.match(text,/Rupture: 28 Myr/);
      assert.doesNotMatch(text,/Back-arc loading rate/);
      assert.match(text,/not the present basin width/);
    }
  });
  await check('episode milestones are filtered by basin identity and preserve chronological navigation',async()=>{
    const events=[{type:'backarc_spreading',time_myr:40,details:{backarc_id:2}},{type:'backarc_loading',time_myr:20,details:{backarc_id:2}},
      {type:'backarc_rifting',time_myr:30,details:{backarc_id:3}},{type:'collision',time_myr:35,details:{backarc_id:2}},
      {type:'backarc_quiet',time_myr:70,details:{backarc_id:2}},{type:'backarc_closed',time_myr:80,details:{backarc_id:2}}];
    assert.deepEqual(Array.from(app.backArcMilestones(basin(),events),row=>row.time_myr),[20,40,70,80]);
    app.state.record={events};show([basin('spreading')],40);
    const catalogue=get('backarc-basins').children[0].children.at(-1);assert.equal(catalogue.children.length,4);
    const visited=[];app.setVisit(time=>visited.push(time));await catalogue.children[2].emit('click');assert.equal(visited[0],70);
  });
  await check('nested saved event identity wins over an obsolete flat identity',()=>{
    const events=[{type:'backarc_loading',time_myr:20,details:{backarc_id:3},backarc_id:2},
      {type:'backarc_rifting',time_myr:30,backarc_id:2}];
    assert.deepEqual(Array.from(app.backArcMilestones(basin(),events),row=>row.time_myr),[30]);
  });
  await check('first ruptured basin opens by default while IDs remain in stable order',async()=>{
    app.backarcReview.expanded.clear();app.backarcReview.collapsed.clear();
    const records=[{...basin(),id:3},{...basin('spreading'),id:2,arc_plate_uid:12},{...basin(),id:1}];
    show(records,40);let cards=get('backarc-basins').children;
    assert.deepEqual(cards.map(row=>row.children[0].textContent),['Basin 1 · Loading','Basin 2 · Spreading','Basin 3 · Loading']);
    assert.deepEqual(cards.map(row=>row.open),[false,true,false]);
    assert.deepEqual(records.map(row=>row.id),[3,2,1],'source order is not mutated');
    cards[1].open=false;await cards[1].emit('toggle');show(records,42);
    assert.equal(get('backarc-basins').children[1].open,false,'manual collapse survives refresh');
    app.backarcReview.expanded.clear();app.backarcReview.collapsed.clear();
    show([{...basin(),id:3},{...basin(),id:1}]);cards=get('backarc-basins').children;
    assert.deepEqual(cards.map(row=>row.open),[true,false],'without a rupture the first ID opens');
  });
  await check('rotation changes displayed center while identities and extension remain recorded',()=>{
    show([{...basin('spreading'),center:[0,1,0],opening_km:44}],40);
    const text=textOf(get('backarc-basins'));assert.match(text,/Center: 0.0° N, 90.0° E/);
    assert.match(text,/Parent \(UID 7\)/);assert.match(text,/Accumulated extension: 44 km/);
  });
  await check('initial and old frames remove stale basin cards from a later inspected epoch',()=>{
    show([basin()]);assert.equal(get('backarc-basins').children.length,1);
    app.state.frame={time_myr:0};app.updateBackArcHistory();assert.equal(get('backarc-history').hidden,true);
    assert.equal(get('backarc-basins').children.length,0);
  });
  console.log(`${checks} back-arc UI checks passed`);
})().catch(error=>{console.error(error);process.exitCode=1;});
