/* Render the recorded surface through the real viewer; no server or simulation is run. */
const fs = require('node:fs');
const vm = require('node:vm');
const assert = require('node:assert/strict');
const path = require('node:path');

const root = path.resolve(__dirname, '..');
const html = fs.readFileSync(path.join(root, 'web/index.html'), 'utf8');
const nodes = new Map();
const animationFrames = [];
const requests = [];

function canvasContext() {
  return {
    draws: [], picture: null,
    createImageData(width, height) { return { width, height, data: new Uint8ClampedArray(width * height * 4) }; },
    putImageData(picture) { this.picture = picture; },
    drawImage(...args) { this.draws.push(args); },
    setTransform() {}, fillRect() {}, save() {}, restore() {}, translate() {}, scale() {},
  };
}

function element(tagName = 'div') {
  const context = canvasContext();
  return {
    tagName: tagName.toUpperCase(), textContent: '', hidden: false, checked: false,
    value: '', disabled: false, dataset: {}, style: {}, children: [], options: [],
    classList: { toggle() {}, add() {}, remove() {} },
    setAttribute() {}, addEventListener() {}, querySelectorAll() { return []; },
    getContext() { return context; },
    getBoundingClientRect() { return { left: 0, top: 0, width: 1280, height: 640 }; },
  };
}

for (const match of html.matchAll(/<(\w+)\b([^>]*\bid="([^"]+)"[^>]*)>/g)) {
  const [, tag, attributes, id] = match;
  const node = element(tag);
  node.value = attributes.match(/\bvalue="([^"]*)"/)?.[1] || '';
  // Overlays are excluded to isolate the base raster supplied to both projections.
  nodes.set(id, node);
}
const get = id => {
  assert.ok(nodes.has(id), `Referenced HTML control exists: ${id}`);
  return nodes.get(id);
};
const document = {
  getElementById: get, createElement: element, querySelectorAll: () => [], addEventListener() {},
};
const sandbox = {
  URLSearchParams, document, location: { search: '' }, console,
  window: { addEventListener() {}, devicePixelRatio: 1 },
  requestAnimationFrame(callback) { animationFrames.push(callback); },
  setTimeout() {}, clearTimeout() {}, setInterval() {},
  fetch: async url => { requests.push(url); throw new Error(`Unexpected render request: ${url}`); },
};
vm.createContext(sandbox);
const source = fs.readFileSync(path.join(root, 'web/app.js'), 'utf8').replace('  initialize();',
  '  globalThis.app = { state, render, mapView, raster, globeView };');
vm.runInContext(source, sandbox);
const app = sandbox.app;
assert.ok(app, 'Test harness exposes the viewer without starting network polling');

function render() {
  app.render();
  assert.equal(animationFrames.length, 1, 'A render schedules exactly one animation frame');
  animationFrames.shift()();
  assert.equal(app.state.rendering, false);
}

function frame(width, height, index = 0) {
  const count = width * height;
  const result = {
    width, height, index, time_myr: index * 20,
    crust: new Uint8Array(count), plate: new Int32Array(count),
    elevation: new Float32Array(count).fill(1200), age: new Float32Array(count),
  };
  // This marker is far beyond the former 768-pixel source width in the 1024-pixel frame.
  result.marker = Math.floor(height / 2) * width + width - 50;
  result.elevation[result.marker] = 9000;
  result.crust[result.marker] = 3;
  result.plate[result.marker] = 1;
  result.age[result.marker] = 200;
  return result;
}

const pixel = cell => Array.from(app.raster.getContext('2d').picture.data.slice(cell * 4, cell * 4 + 4));
function assertSource(data, projection = 'map', initial = false) {
  assert.equal(app.mapView.rasterSource, data, 'Renderer uses the actual recorded/editor surface object');
  assert.equal(app.mapView.rasterData, data);
  assert.equal(app.raster.width, data.width);
  assert.equal(app.raster.height, data.height);
  const picture = app.raster.getContext('2d').picture;
  assert.equal(picture.width, data.width);
  assert.equal(picture.height, data.height);
  assert.equal(picture.data.length, data.width * data.height * 4);
  assert.equal(picture.data.at(-1), 255, 'All source cells are rasterized, including the last cell');
  const display = sandbox.window.deepTimeDisplayState();
  assert.equal(display.source, initial ? 'initial-grid' : 'recorded-frame');
  assert.equal(display.width, data.width);
  assert.equal(display.height, data.height);
  assert.equal(display.recordedWidth, data.width);
  assert.equal(display.recordedHeight, data.height);
  assert.equal(display.layer, app.state.layer);
  assert.equal(display.projection, projection);
  assert.equal(display.frameIndex, data.index ?? null);
  assert.equal(display.pending, false);
  assert.equal(display.upgradable, false);
}

let checks = 0;
function check(name, action) { action(); checks++; console.log(`PASS ${name}`); }
const recorded = frame(1024, 512);
app.state.mode = 'history';
app.state.runId = 'saved-material-world';
app.state.frame = recorded;

for (const layer of ['elevation', 'plate', 'crust', 'age']) {
  check(`${layer} renders every cell of the recorded 1024 x 512 surface`, () => {
    app.state.layer = layer;
    render();
    assert.deepEqual(requests, [], 'Rendering does not request a replacement mechanics grid');
    assertSource(recorded);
    assert.notDeepEqual(pixel(recorded.marker), pixel(0), 'The high-resolution marker survives rasterization');
    if (layer === 'elevation') assert.deepEqual(pixel(recorded.marker), [247, 245, 234, 255]);
    if (layer === 'crust') assert.deepEqual(pixel(recorded.marker), [175, 128, 104, 255]);
    if (layer === 'age') assert.deepEqual(pixel(recorded.marker), [41, 89, 129, 255]);
  });
}

check('switching to a smaller later frame replaces the source and raster dimensions', () => {
  const later = frame(192, 96, 1);
  later.crust[later.marker] = 2;
  app.state.frame = later;
  app.state.currentIndex = 1;
  app.state.layer = 'crust';
  render();
  assertSource(later);
  assert.deepEqual(pixel(later.marker), [213, 177, 111, 255]);
});

check('another run at the same frame index cannot reuse a stale surface', () => {
  const otherRun = frame(1024, 512);
  otherRun.crust[otherRun.marker] = 1;
  app.state.runId = 'different-world';
  app.state.frame = otherRun;
  app.state.currentIndex = 0;
  render();
  assertSource(otherRun);
  assert.deepEqual(pixel(otherRun.marker), [148, 164, 126, 255]);
});

check('the globe texture is rendered from the same full-resolution recorded surface', () => {
  let texture;
  let drawCount = 0;
  app.globeView.active = true;
  app.globeView.renderer = { setTexture(canvas) { texture = canvas; }, draw() { drawCount++; } };
  app.state.frame = recorded;
  app.state.layer = 'elevation';
  render();
  assertSource(recorded, 'globe');
  assert.equal(app.globeView.textureFrame, recorded);
  assert.equal(texture, get('world-map'));
  assert.equal(drawCount, 1);
  const textureDraw = texture.getContext('2d').draws.at(-1);
  assert.equal(textureDraw[1], recorded.width, 'Texture crop starts at the wrapped source copy');
  assert.equal(textureDraw[3], recorded.width + 2, 'Texture retains the full source width plus seam padding');
  assert.equal(textureDraw[4], recorded.height);
  assert.deepEqual(pixel(recorded.marker), [247, 245, 234, 255]);
});

check('the starting-world editor keeps its own editable cell grid', () => {
  const initial = { width: 16, height: 8, crust: new Uint8Array(128) };
  initial.crust[34] = 2;
  app.globeView.active = false;
  app.state.mode = 'initial';
  app.state.initial = initial;
  app.state.layer = 'crust';
  render();
  assertSource(initial, 'map', true);
  assert.deepEqual(pixel(34), [213, 177, 111, 255].map((value, index) => index === 3 ? value : Math.round(value * .76)));
  assert.deepEqual(requests, [], 'No render has requested /api/display or other network data');
});

check('recorded resolution and physics are identified from the displayed frame', () => {
  app.state.mode = 'history';
  app.state.frame = { ...recorded, mesh_level: 4, physics_profile_version: 1,
    timestep_diagnostics: { rejected_trials: 2 } };
  render();
  const note = get('recorded-physics-note').textContent;
  assert.match(note, /1024 × 512/);
  assert.match(note, /5,120 tectonic control cells/);
  assert.match(note, /Force balance/);
  assert.match(note, /2 rejected trial/);
  app.state.frame = recorded;
  render();
  assert.match(get('recorded-physics-note').textContent, /Legacy or individually configured/);
  assert.doesNotMatch(get('recorded-physics-note').textContent, /rejected trial/);
});

check('inherited primordial slabs are identified only on the recorded experiment', () => {
  app.state.frame = { ...recorded, physics_profile_version: 1,
    primordial_subduction_diagnostics: { enabled: true, initial_slab_depth_km: 100 } };
  render();
  assert.match(get('recorded-physics-note').textContent, /initial ocean subduction, 100 km deep slabs/);
  app.state.frame = recorded;
  render();
  assert.doesNotMatch(get('recorded-physics-note').textContent, /initial ocean subduction/);
});

console.log(`${checks} display resolution UI checks passed.`);
