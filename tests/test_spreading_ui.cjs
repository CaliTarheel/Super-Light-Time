/* Recorded spreading separates opening, axis drift, and last-step creation. */
const fs = require('node:fs');
const vm = require('node:vm');
const assert = require('node:assert/strict');
const path = require('node:path');
const root = path.resolve(__dirname, '..');
const html = fs.readFileSync(path.join(root, 'web/index.html'), 'utf8');
const nodes = new Map(), outputs = new Map();
function element(tagName = 'div') {
  return {tagName:String(tagName).toUpperCase(), textContent:'', innerHTML:'', hidden:false, checked:false,
    value:'', disabled:false, dataset:{}, style:{}, children:[], options:[], firstChild:{nodeValue:''},
    classList:{toggle(){},add(){},remove(){}},setAttribute(){},removeAttribute(){},setPointerCapture(){},
    addEventListener(){},append(...children){this.children.push(...children);},add(option){this.options.push(option);},
    replaceChildren(...children){this.children=children;if(this.tagName==='SELECT')this.options=children;},
    querySelectorAll(){return [];},reportValidity(){return true;},click(){},remove(){},getContext(){return {};},
    getBoundingClientRect(){return {left:0,top:0,width:640,height:320};}};
}
for(const match of html.matchAll(/<(\w+)\b([^>]*\bid="([^"]+)"[^>]*)>/g)){
  const [,tag,attributes,id]=match,node=element(tag);
  node.value=attributes.match(/\bvalue="([^"]*)"/)?.[1]||'';node.checked=/\bchecked\b/.test(attributes);
  if(tag==='select'){
    const content=html.slice(match.index+match[0].length).split('</select>')[0];
    node.options=[...content.matchAll(/<option\b([^>]*)>([^<]*)/g)].map(option=>({value:option[1].match(/value="([^"]*)"/)?.[1]||'',textContent:option[2],selected:/\bselected\b/.test(option[1])}));
    node.value=(node.options.find(option=>option.selected)||node.options[0])?.value||'';
  }
  nodes.set(id,node);
}
const get=id=>{assert.ok(nodes.has(id),`HTML control exists: ${id}`);return nodes.get(id);};
const document={getElementById:get,createElement:element,createElementNS:(_,tag)=>element(tag),
  querySelector(selector){if(!outputs.has(selector))outputs.set(selector,element());return outputs.get(selector);},
  querySelectorAll:()=>[],addEventListener(){},body:element(),activeElement:null};
const sandbox={document,window:{addEventListener(){},localStorage:{getItem(){return null;},setItem(){}}},
  performance:{now:()=>0},setTimeout(){},clearTimeout(){},setInterval(){},requestAnimationFrame(){},console,
  Option:function(text,value){return {textContent:text,value};},fetch:async()=>({ok:true,json:async()=>({})})};
vm.createContext(sandbox);
const source=fs.readFileSync(path.join(root,'web/app.js'),'utf8').replace('  initialize();',
  '  globalThis.app={state,updateSpreadingHistory,updateStats};');
vm.runInContext(source,sandbox);
const app=sandbox.app;
const recorded={active_segments:12,reconstructed_cells:34,generated_area_km2:8000,
  mean_full_rate_cm_yr:4,mean_axis_speed_cm_yr:0,side_p_area_km2:4000,side_q_area_km2:4000};
const figures=()=>get('spreading-figures').children.map(node=>node.textContent).join('\n');
function show(diagnostic,time=40){app.state.mode='history';app.state.frame={time_myr:time,spreading_diagnostics:diagnostic};app.updateSpreadingHistory();}
let checks=0;
function check(name,action){action();checks++;console.log(`PASS ${name}`);}
check('stationary axis retains real full opening and generated ocean',()=>{
  show(recorded);assert.equal(get('spreading-history').hidden,false);
  assert.match(figures(),/Mean full opening: 4 cm\/yr/);assert.match(figures(),/Mean axis drift: 0 cm\/yr/);
  assert.match(figures(),/Ocean added this step: 8,000 km²/);assert.match(html,/stationary axis can still spread/);
  assert.match(get('spreading-description').textContent,/last integration step/);
  assert.match(get('spreading-description').textContent,/not the whole interval between snapshots/);
});
check('axis drift is independent of opening and does not imply creation',()=>{
  show({...recorded,mean_full_rate_cm_yr:0,mean_axis_speed_cm_yr:3.25,generated_area_km2:0});
  assert.match(figures(),/Mean full opening: 0 cm\/yr/);assert.match(figures(),/Mean axis drift: 3.25 cm\/yr/);
  assert.match(figures(),/Ocean added this step: 0 km²/);
});
check('eligible segments and paired realized areas remain distinct',()=>{
  show(recorded);assert.match(figures(),/Eligible spreading segments: 12/);
  const detail=get('spreading-budget').textContent;
  assert.match(detail,/Reconstructed cells: 34/);assert.match(detail,/Side P added: 4,000 km²/);
  assert.match(detail,/Side Q added: 4,000 km²/);assert.match(detail,/not two global plates/);
  assert.match(detail,/contribute no area/);
});
check('missing and invalid figures do not become invented zero measurements',()=>{
  show({active_segments:0,generated_area_km2:0});
  assert.match(figures(),/Mean full opening: Not recorded/);assert.match(figures(),/Mean axis drift: Not recorded/);
  assert.match(figures(),/Ocean added this step: 0 km²/);assert.match(figures(),/Eligible spreading segments: 0/);
  assert.match(get('spreading-budget').textContent,/Side P added: Not recorded/);
  show({...recorded,active_segments:1.5,mean_full_rate_cm_yr:NaN,mean_axis_speed_cm_yr:'4',generated_area_km2:-1});
  assert.equal((figures().match(/Not recorded/g)||[]).length,4);
});
check('legacy and malformed diagnostics hide and clear stale review content',()=>{
  for(const value of [undefined,null,[],{},'old']){
    show(recorded);show(value);assert.equal(get('spreading-history').hidden,true);
    assert.equal(get('spreading-figures').children.length,0);assert.equal(get('spreading-budget').textContent,'');
    assert.equal(get('spreading-epoch').textContent,'');assert.equal(get('spreading-description').textContent,'');
  }
});
check('normal statistics refresh clears the panel on return to the editor',()=>{
  show(recorded);app.state.mode='initial';app.state.initial=null;app.updateStats();
  assert.equal(get('spreading-history').hidden,true);assert.equal(get('spreading-figures').children.length,0);
});
check('normal saved-frame statistics refresh updates spreading without mutating source',()=>{
  const diagnostic=Object.freeze({...recorded});
  app.state.mode='history';app.state.frame={width:2,height:1,time_myr:62,crust:[0,0],elevation:[-3000,-4000],
    plate:[0,1],age:[0,20],spreading_diagnostics:diagnostic};app.updateStats();
  assert.equal(get('spreading-epoch').textContent,'62 Myr');assert.match(figures(),/Mean full opening: 4 cm\/yr/);
  assert.equal(diagnostic.generated_area_km2,8000);
});
check('initial epoch does not claim an elapsed integration step',()=>{
  show({active_segments:0,generated_area_km2:0},0);
  assert.match(get('spreading-description').textContent,/no elapsed integration step yet/);
  assert.equal(get('spreading-epoch').textContent,'0 Myr');
});
console.log(`${checks} spreading UI checks passed`);
