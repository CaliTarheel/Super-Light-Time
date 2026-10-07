/* The legacy force coefficients are disabled when the force balance is selected. */
const fs = require('node:fs');
const vm = require('node:vm');
const assert = require('node:assert/strict');
const path = require('node:path');
const root = path.resolve(__dirname, '..');
const html = fs.readFileSync(path.join(root, 'web/index.html'), 'utf8');
const nodes = new Map(), outputs = new Map();
function element(tagName = 'div') {
  const classes = new Set();
  return { tagName: String(tagName).toUpperCase(), textContent: '', innerHTML: '', hidden: false, checked: false,
    value: '', disabled: false, dataset: {}, style: {}, children: [], options: [], firstChild: {nodeValue:''},
    classList: { toggle(name, on) { if (on) classes.add(name); else classes.delete(name); },
      add(name) { classes.add(name); }, remove(name) { classes.delete(name); }, contains: (name) => classes.has(name) },
    setAttribute() {}, removeAttribute() {}, setPointerCapture() {}, addEventListener() {},
    append(...children) { this.children.push(...children); }, add(option) { this.options.push(option); },
    replaceChildren(...children) { this.children = children; if (this.tagName === 'SELECT') this.options = children; },
    querySelectorAll() { return []; }, reportValidity() { return true; }, click() {}, remove() {},
    getContext() { return {}; }, getBoundingClientRect() { return {left:0,top:0,width:640,height:320}; } };
}
for (const match of html.matchAll(/<(\w+)\b([^>]*\bid="([^"]+)"[^>]*)>/g)) {
  const [, tag, attributes, id] = match, node = element(tag);
  node.value = attributes.match(/\bvalue="([^"]*)"/)?.[1] || '';
  node.checked = /\bchecked\b/.test(attributes);
  node.hidden = /\bhidden\b/.test(attributes);
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
const storage = new Map();
const sandbox = {performance:{now:()=>0}, document, URLSearchParams, location:{search:''}, window:{addEventListener(){},localStorage:{getItem:key=>storage.get(key)??null,setItem:(key,value)=>storage.set(key,value)}},
  setTimeout(){}, clearTimeout(){}, setInterval(){}, requestAnimationFrame(){}, console,
  Option:function(text,value){return {textContent:text,value};}, fetch:async()=>({ok:true,json:async()=>({})})};
vm.createContext(sandbox);
const source = fs.readFileSync(path.join(root,'web/app.js'),'utf8')
  .replace('  initialize();', '  globalThis.app={updateResolutionNotes,readConfig};');
vm.runInContext(source, sandbox);
const app = sandbox.app;
let checks = 0;

// The force balance derives slab pull and ridge push from the world itself, so
// the legacy coefficients must not look effective.
get('physics_profile').value = 'reviewed_v1';
app.updateResolutionNotes();
assert.equal(get('slab_pull').disabled, true); checks++;
assert.equal(get('ridge_push').disabled, true); checks++;
assert.equal(get('force-coefficient-note').hidden, false); checks++;
assert.ok(get('slab-pull-field').classList.contains('inactive-field')); checks++;
assert.ok(get('ridge-push-field').classList.contains('inactive-field')); checks++;

// The legacy velocity law reads them, so there they stay live.
get('physics_profile').value = 'legacy';
app.updateResolutionNotes();
assert.equal(get('slab_pull').disabled, false); checks++;
assert.equal(get('ridge_push').disabled, false); checks++;
assert.equal(get('force-coefficient-note').hidden, true); checks++;
assert.ok(!get('slab-pull-field').classList.contains('inactive-field')); checks++;

// Erosion and rift strength are read on both paths and stay available.
assert.equal(get('erosion').disabled, false); checks++;
assert.equal(get('rift_strength').disabled, false); checks++;

console.log(`force slider UI: ${checks} checks passed`);
