/* Cartographic teeth from recorded trench polarity; no tectonic state changes. */
(function (root) {
  "use strict";
  const clamp = (x, lo, hi) => Math.max(lo, Math.min(hi, x));
  const wrap = (x, width) => ((x % width) + width) % width;
  const unit = (p) => { const length = Math.hypot(...p); return p.map(v => v / length); };
  const distance2 = (a, b) => a.reduce((sum, value, i) => sum + (value - b[i]) ** 2, 0);

  function build(data) {
    const w = data?.width, h = data?.height;
    const available = Array.isArray(data?.trench_systems);
    const empty = { available, faces: [], activeSystems: 0, omittedAnchors: 0 };
    if (!available || !Number.isInteger(w) || !Number.isInteger(h) || w < 2 || h < 2 || data.plate?.length !== w * h) return empty;
    const slots = new Map((data.plates || []).map(row => [row.uid, row.id]));
    const active = data.trench_systems.filter(row => ["initiating", "mature"].includes(row.phase) &&
      Number.isFinite(row.last_seen_myr) && Number.isFinite(data.time_myr) &&
      Math.abs(row.last_seen_myr - data.time_myr) < 1e-7 &&
      slots.has(row.downgoing_plate_uid) && slots.has(row.overriding_plate_uid) &&
      row.downgoing_plate_uid !== row.overriding_plate_uid);
    const centers = new Map(), faces = new Map(), conflicts = new Set();
    const center = (x, y) => {
      const index = y * w + x;
      if (!centers.has(index)) {
        const lon = ((x + .5) / w * 2 - 1) * Math.PI, lat = (.5 - (y + .5) / h) * Math.PI;
        centers.set(index, [Math.cos(lat) * Math.cos(lon), Math.cos(lat) * Math.sin(lon), Math.sin(lat)]);
      }
      return centers.get(index);
    };
    let omittedAnchors = 0;
    for (const row of active) {
      const down = slots.get(row.downgoing_plate_uid), over = slots.get(row.overriding_plate_uid);
      for (const raw of row.geometry_xyz || []) {
        if (!Array.isArray(raw) || raw.length !== 3 || !raw.every(Number.isFinite) || Math.hypot(...raw) < .5) { omittedAnchors++; continue; }
        const point = unit(raw);
        const u = (Math.atan2(point[1], point[0]) / Math.PI + 1) / 2, v = .5 - Math.asin(clamp(point[2], -1, 1)) / Math.PI;
        const col = Math.floor(u * w), line = clamp(Math.floor(v * h), 0, h - 1);
        const cellDiagonal = Math.hypot(Math.PI / h, 2 * Math.PI / w * Math.hypot(point[0], point[1]));
        const candidates = new Map();
        const consider = (x, y, axis) => {
          const x2 = axis === "vertical" ? wrap(x + 1, w) : x, y2 = axis === "horizontal" ? y + 1 : y;
          if (y2 >= h) return;
          const a = y * w + x, b = y2 * w + x2, pa = data.plate[a], pb = data.plate[b];
          if (!((pa === down && pb === over) || (pa === over && pb === down))) return;
          // Buoyant arrivals can end or change a local trench. Do not extend
          // old polarity onto a newly continental downgoing raster cell.
          if (data.crust?.length === w * h && data.crust[pa === down ? a : b] !== 0) return;
          const first = center(x, y), second = center(x2, y2);
          const midpoint = unit(first.map((value, i) => value + second[i]));
          const score = distance2(point, midpoint);
          if (score > (.75 * cellDiagonal) ** 2) return;
          const key = `${axis}:${x}:${y}`;
          candidates.set(key, { key, score, x: (x + (axis === "vertical" ? 1 : .5)) / w,
            y: (y + (axis === "horizontal" ? 1 : .5)) / h, axis, over: pb === over ? 1 : -1,
            trenchId: row.id, a, b, span: axis === "vertical" ? 1 / h : 1 / w });
        };
        // Match unordered spherical anchors to local same-pair raster faces.
        // Longitude wraps; neither polar edge is joined to the opposite pole.
        for (let y = Math.max(0, line - 2); y <= Math.min(h - 1, line + 2); y++) {
          for (let dx = -2; dx <= 2; dx++) {
            const x = wrap(col + dx, w);
            consider(x, y, "vertical"); consider(x, y, "horizontal");
          }
        }
        const ordered = [...candidates.values()].sort((a, b) => a.score - b.score || a.key.localeCompare(b.key));
        const best = ordered[0], next = ordered[1];
        // Near an unresolved junction, omit an ambiguous symbol instead of
        // assigning it to a different face of the same plate pair.
        if (!best || (next && next.score - best.score < (.04 * cellDiagonal) ** 2)) { omittedAnchors++; continue; }
        const previous = faces.get(best.key);
        if (previous && previous.over !== best.over) { conflicts.add(best.key); omittedAnchors++; continue; }
        if (!previous || best.score < previous.score) faces.set(best.key, best);
      }
    }
    return { available, activeSystems: active.length, omittedAnchors,
      faces: [...faces.values()].filter(face => !conflicts.has(face.key)) };
  }

  function triangles(faces, { width, height, zoom = 1, cx = .5, cy = .5, pixelRatio = 1 }) {
    if (!(width > 0 && height > 0 && zoom > 0 && pixelRatio > 0)) return [];
    const size = 8.5 * pixelRatio / zoom, halfBase = 5 * pixelRatio / zoom;
    const spacing = 22 * pixelRatio / zoom, bins = new Map(), result = [];
    const left = width * (cx - .5 / zoom), right = width * (cx + .5 / zoom);
    const top = height * (cy - .5 / zoom), bottom = height * (cy + .5 / zoom);
    const accepted = (x, y) => {
      const bx = Math.floor(x / spacing), by = Math.floor(y / spacing);
      for (let yy = by - 1; yy <= by + 1; yy++) for (let xx = bx - 1; xx <= bx + 1; xx++) {
        for (const other of bins.get(`${xx}:${yy}`) || []) {
          const dx = Math.abs(x - other[0]);
          if (dx ** 2 + (y - other[1]) ** 2 < spacing ** 2) return false;
        }
      }
      const key = `${bx}:${by}`;
      if (!bins.has(key)) bins.set(key, []);
      bins.get(key).push([x, y]); return true;
    };
    for (const face of faces) {
      const along = face.axis === "vertical" ? height : width;
      const length = face.span * along;
      const count = Math.max(1, Math.ceil(length / spacing));
      for (let j = 0; j < count; j++) {
        const offset = length * ((j + .5) / count - .5);
        const baseX = face.x * width + (face.axis === "horizontal" ? offset : 0);
        const y = face.y * height + (face.axis === "vertical" ? offset : 0);
        // The same dateline face is visible on both map edges; never draw a
        // triangle whose vertices run across the entire world.
        const copies = face.axis === "vertical" && Math.abs(face.x - 1) < 1e-10 ? [baseX, baseX - width] : [baseX];
        for (const x of copies) {
          if (x < left - size || x > right + size || y < top - size || y > bottom + size || !accepted(x, y)) continue;
          result.push(face.axis === "vertical" ?
            [[x, y - halfBase], [x, y + halfBase], [x + face.over * size, y]] :
            [[x - halfBase, y], [x + halfBase, y], [x, y + face.over * size]]);
        }
      }
    }
    return result;
  }
  const api = { build, triangles };
  if (typeof module !== "undefined" && module.exports) module.exports = api;
  else root.DeepTimeSubduction = api;
})(globalThis);
