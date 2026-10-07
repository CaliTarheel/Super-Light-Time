/* Directional geological symbols from saved local polarity, without a server. */
const assert = require('node:assert/strict');
const {build, triangles} = require('../web/subduction.js');

const W = 12, H = 6, DOWN = 2, OVER = 7;
function xyz(x, y, w = W, h = H) {
  const lon = 2 * Math.PI * ((x + .5) / w - .5), lat = Math.PI * (.5 - (y + .5) / h);
  return [Math.cos(lat) * Math.cos(lon), Math.cos(lat) * Math.sin(lon), Math.sin(lat)];
}
function between(a, b, fraction = .5) {
  const p = a.map((v, i) => v * (1 - fraction) + b[i] * fraction), length = Math.hypot(...p);
  return p.map(v => v / length);
}
function fixture(axis = 'vertical', over = 1) {
  const plate = Array.from({length: W * H}, (_, i) => {
    const positive = axis === 'vertical' ? i % W >= W / 2 : Math.floor(i / W) >= H / 2;
    return positive === (over === 1) ? OVER : DOWN;
  });
  const anchor = axis === 'vertical' ? between(xyz(5, 2), xyz(6, 2)) : between(xyz(5, 2), xyz(5, 3));
  return {width: W, height: H, time_myr: 40, plate, boundary: Array(W * H).fill(2),
    crust: Array(W * H).fill(0), trench: Array(W * H).fill(41),
    plates: [{id: DOWN, uid: 101, active: true}, {id: OVER, uid: 203, active: true}],
    trench_systems: [{id: 41, phase: 'mature', last_seen_myr: 40,
      plate_uids: [101, 203], downgoing_plate_uid: 101, overriding_plate_uid: 203,
      geometry_xyz: [anchor], anchor_gap_km: 0, maturity: 1}]};
}
const clone = value => JSON.parse(JSON.stringify(value));
const near = (a, b, tolerance = 1e-8) => Math.abs(a - b) <= tolerance;
let checks = 0;
function check(name, action) { action(); checks++; console.log(`PASS ${name}`); }
function validTriangles(result) {
  assert.ok(Array.isArray(result));
  for (const triangle of result) {
    assert.equal(triangle.length, 3);
    assert.ok(triangle.every(p => p.length === 2 && p.every(Number.isFinite)));
    const [a, b, c] = triangle;
    assert.ok(Math.abs((b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0])) > 0);
  }
}
function extent(triangle) {
  return Math.max(...[0, 1].map(axis => Math.max(...triangle.map(p => p[axis])) - Math.min(...triangle.map(p => p[axis]))));
}

for (const axis of ['vertical', 'horizontal']) for (const over of [-1, 1]) {
  check(`${axis} teeth use saved overriding UID, on the ${over === 1 ? 'positive' : 'negative'} side`, () => {
    const result = build(fixture(axis, over));
    assert.equal(result.available, true);
    assert.ok(result.faces.length > 0);
    assert.ok(result.faces.every(f => f.axis === axis && f.over === over && f.trenchId === 41));
    const symbol = triangles(result.faces, {width: 720, height: 360});
    validTriangles(symbol); assert.ok(symbol.length > 0);
    const coordinate = axis === 'vertical' ? 0 : 1, base = coordinate === 0 ? 360 : 180;
    for (const t of symbol) {
      assert.ok(t.every(p => over * (p[coordinate] - base) >= -1e-7));
      assert.ok(t.some(p => over * (p[coordinate] - base) > 0));
    }
  });
}

check('building and drawing never modify saved history or caller options', () => {
  const data = fixture(), before = clone(data), options = {width: 720, height: 360, zoom: 2, pixelRatio: 2};
  const optionCopy = clone(options), result = build(data), faceCopy = clone(result.faces);
  triangles(result.faces, options);
  assert.deepEqual(data, before); assert.deepEqual(options, optionCopy); assert.deepEqual(result.faces, faceCopy);
});

check('legacy maps retain no invented polarity', () => {
  const data = fixture(); delete data.trench_systems; delete data.trench;
  const result = build(data); assert.equal(result.available, false); assert.deepEqual(result.faces, []);
});

check('quiet, shut down, and joined traces do not imply current subduction', () => {
  for (const phase of ['quiet', 'shutdown', 'joined']) {
    const data = fixture(); data.trench_systems[0].phase = phase;
    assert.deepEqual(build(data).faces, []);
  }
});

check('stale and future geometry is not presented as current activity', () => {
  for (const time of [38, 42]) {
    const data = fixture(); data.trench_systems[0].last_seen_myr = time;
    assert.deepEqual(build(data).faces, []);
  }
});

check('unknown or self-identical owners cannot produce directional symbols', () => {
  for (const field of ['downgoing_plate_uid', 'overriding_plate_uid']) {
    const data = fixture(); data.trench_systems[0][field] = 999;
    assert.deepEqual(build(data).faces, []);
  }
  const data = fixture(); data.trench_systems[0].overriding_plate_uid = 101;
  assert.deepEqual(build(data).faces, []);
});

check('an initiating trench receives teeth before full maturity', () => {
  const data = fixture(); data.trench_systems[0].phase = 'initiating'; data.trench_systems[0].maturity = .1;
  assert.ok(build(data).faces.length > 0);
});

check('unordered and repeated anchors do not join distant trench pieces', () => {
  const data = fixture();
  const first = between(xyz(5, 1), xyz(6, 1)), second = between(xyz(5, 4), xyz(6, 4));
  data.trench_systems[0].geometry_xyz = [second, first, second];
  const result = build(data);
  const reversed = clone(data); reversed.trench_systems[0].geometry_xyz.reverse();
  const key = f => `${f.axis}/${f.x}/${f.y}/${f.over}/${f.trenchId}`;
  assert.deepEqual(result.faces.map(key).sort(), build(reversed).faces.map(key).sort());
  assert.equal(new Set(result.faces.map(key)).size, result.faces.length);
  assert.ok(result.faces.length >= 2);
  assert.ok(result.faces.every(f => f.y < 2 / H || f.y > 4 / H), 'No artificial connecting face through the gap');
});

check('dateline polarity is resolved locally and symbols wrap with the map', () => {
  const data = fixture();
  data.trench_systems[0].geometry_xyz = [between(xyz(11, 2), xyz(0, 2))];
  const result = build(data);
  assert.ok(result.faces.length > 0);
  assert.ok(result.faces.every(f => f.axis === 'vertical' && f.over === -1 && (near(f.x, 0) || near(f.x, 1))));
  const symbols = triangles(result.faces, {width: 720, height: 360});
  validTriangles(symbols); assert.ok(symbols.length > 0);
  assert.ok(symbols.some(t => t.some(p => p[0] > 700)), 'Visible teeth appear on the overriding side at the right map edge');
  assert.ok(symbols.every(t => Math.max(...t.map(p => p[0])) - Math.min(...t.map(p => p[0])) < 30), 'No symbol spans the longitude seam');
});

check('subcell anchor positions still use the neighboring recorded plate pair', () => {
  const data = fixture(); data.trench_systems[0].geometry_xyz = [between(xyz(5, 2), xyz(6, 2), .75)];
  const result = build(data); assert.ok(result.faces.length > 0);
  assert.ok(result.faces.every(f => f.axis === 'vertical' && f.over === 1));
});

check('longitude rotation carries both the saved polarity and seam placement', () => {
  const data = fixture(), before = data.plate.slice();
  for (let y = 0; y < H; y++) for (let x = 0; x < W; x++) data.plate[y * W + (x + W / 2) % W] = before[y * W + x];
  data.trench_systems[0].geometry_xyz = data.trench_systems[0].geometry_xyz.map(([x, y, z]) => [-x, -y, z]);
  const result = build(data); assert.ok(result.faces.length > 0);
  assert.ok(result.faces.every(f => f.axis === 'vertical' && f.over === 1 && (near(f.x, 0) || near(f.x, 1))));
  validTriangles(triangles(result.faces, {width: 720, height: 360}));
});

check('rotation over a pole reverses the map direction while retaining the overriding owner', () => {
  const data = fixture('horizontal'), before = data.plate.slice();
  for (let y = 0; y < H; y++) for (let x = 0; x < W; x++) data.plate[(H - 1 - y) * W + W - 1 - x] = before[y * W + x];
  data.trench_systems[0].geometry_xyz = data.trench_systems[0].geometry_xyz.map(([x, y, z]) => [x, -y, -z]);
  const result = build(data); assert.ok(result.faces.length > 0);
  assert.ok(result.faces.every(f => f.axis === 'horizontal' && f.over === -1));
  const symbols = triangles(result.faces, {width: 720, height: 360});
  validTriangles(symbols); assert.ok(symbols.length > 0);
  assert.ok(symbols.every(t => t.every(p => p[1] <= 180 + 1e-8)));
});

check('polar anchors stay finite and do not connect across the whole map', () => {
  for (const row of [0, H - 1]) {
    const data = fixture(); data.trench_systems[0].geometry_xyz = [between(xyz(5, row), xyz(6, row))];
    const result = build(data); assert.ok(result.faces.length > 0);
    assert.ok(result.faces.every(f => f.over === 1));
    const symbols = triangles(result.faces, {width: 720, height: 360});
    validTriangles(symbols);
    assert.ok(symbols.every(t => extent(t) < 30));
  }
});

check('symbol size remains constant in CSS pixels through zoom and DPI changes', () => {
  const faces = build(fixture()).faces;
  const a = triangles(faces, {width: 720, height: 360, zoom: 1, pixelRatio: 1});
  const b = triangles(faces, {width: 1440, height: 720, zoom: 2, pixelRatio: 2});
  const c = triangles(faces, {width: 720, height: 360, zoom: 4, pixelRatio: 1});
  for (const result of [a, b, c]) { validTriangles(result); assert.ok(result.length > 0); }
  assert.ok(near(extent(a[0]), extent(b[0]) * 2 / 2, .01));
  assert.ok(near(extent(a[0]), extent(c[0]) * 4, .01));
});

check('coarse source maps produce bounded, non-overlapping screen density', () => {
  const data = fixture(); data.trench_systems[0].geometry_xyz = Array.from({length: H}, (_, row) => between(xyz(5, row), xyz(6, row)));
  const faces = build(data).faces;
  for (const zoom of [1, 4, 16]) {
    const result = triangles(faces, {width: 720, height: 360, zoom, pixelRatio: 1});
    validTriangles(result); assert.ok(result.length > 0); assert.ok(result.length < 100, 'Screen symbols are thinned as a line, not emitted for every source pixel');
    const centers = result.map(t => t.reduce((p, q) => [p[0] + q[0] / 3, p[1] + q[1] / 3], [0, 0]));
    for (let i = 0; i < centers.length; i++) for (let j = i + 1; j < centers.length; j++) {
      assert.ok(Math.hypot(centers[i][0] - centers[j][0], centers[i][1] - centers[j][1]) * zoom >= 3, 'Avoid coincident/overlapping tooth emissions');
    }
  }
});

console.log(`${checks} subduction-symbol checks passed.`);
