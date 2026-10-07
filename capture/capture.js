/**
 * Capture the Deep Time app's OWN canvas render for archival frames.
 *
 * The app draws elevation, plate boundaries, subduction teeth and arc islands in
 * the browser, from app.js. None of that exists server-side -- /api/preview is
 * only color_image(elevation) at the raw 192x96 grid -- so the only way to
 * archive what the app actually shows is to drive the app and read its canvas.
 *
 * Doing it this way means there is exactly ONE renderer. A separate Python
 * reimplementation would drift from the app the moment either changed, and had
 * already produced a picture that disagreed with the physics.
 *
 *   node capture.js --out <dir> [--from N] [--to N] [--latest] [--scale 2]
 *                   [--url http://127.0.0.1:8766/] [--run-id ID] [--watch] [--interval 60]
 *
 * --scale enlarges the canvas backing store before rendering. The app's draw
 * code reads canvas.width/height dynamically, so this yields a genuinely higher
 * resolution render rather than an upscale. Verify output before trusting a
 * large value; the app was authored against 1152x576.
 */
'use strict';

const fs = require('fs');
const path = require('path');
const puppeteer = require('puppeteer');

function arg(name, fallback = null) {
  const i = process.argv.indexOf('--' + name);
  if (i < 0) return fallback;
  const next = process.argv[i + 1];
  return next && !next.startsWith('--') ? next : true;
}

const OUT = arg('out');
const URL = arg('url', 'http://127.0.0.1:8766/');
const EXPECTED_RUN_ID = arg('run-id');
const FROM = Number(arg('from', 0));
// The app sizes its canvas to the page layout, so the VIEWPORT is the render
// resolution. Enlarging it makes the app draw more detail, rather than upscaling
// a small draw. Setting canvas.width directly does not work -- the assignment
// clears the backing store and the app resets it on the next render.
const VIEWPORT_W = Number(arg('vw', 3400)) || 3400;
const VIEWPORT_H = Number(arg('vh', 2000)) || 2000;
const WATCH = !!arg('watch');
const REPLACE = !!arg('replace');
const INTERVAL = Number(arg('interval', 60)) * 1000;
if (!OUT) {
  console.error('usage: node capture.js --out <dir> [--from N] [--to N] [--latest] [--scale S] [--watch]');
  process.exit(2);
}
fs.mkdirSync(OUT, { recursive: true });

const sleep = (ms) => new Promise((r) => setTimeout(r, ms));

// Frames are named by AGE BEFORE PRESENT, counting down from the run's duration: the frame at
// simulation time t is (duration - t) Ma, zero-padded to four digits. Ages come from the run's
// own frame list rather than index arithmetic, because a branch or an authored transition can
// put a frame at a fractional epoch that "index * 2" would mislabel.
let frameAges = [];

async function loadAges() {
  // `URL` here is this script's --url string, which shadows the global URL class.
  const status = await (await fetch(URL.replace(/\/?$/, '/') + 'api/status')).json();
  if (EXPECTED_RUN_ID && status?.run_id !== EXPECTED_RUN_ID)
    throw new Error(`refusing capture: selected run ${status?.run_id ?? '(none)'} does not match --run-id ${EXPECTED_RUN_ID}`);
  // The present is simulation time TOTAL_MYR (1000 by convention), NOT the run's own
  // config.duration_myr: the parent run was configured for 300 Myr before the branch
  // extended it, and reading its duration labelled its 12 Myr frame "0288Ma".
  const total = Number(arg('total', 1000));
  const frames = Array.isArray(status?.frames) ? status.frames : [];
  if (!Number.isFinite(total) || !frames.length) throw new Error('could not read the frame list');
  frameAges = frames.map((f) => total - Number(f.time_myr));
  return frameAges.length;
}

function nameFor(index) {
  const age = frameAges[index];
  if (!Number.isFinite(age)) throw new Error(`no recorded epoch for frame ${index}`);
  const rounded = Math.round(age);
  const text = Math.abs(age - rounded) < 1e-6
    ? String(rounded).padStart(4, '0')
    : age.toFixed(2).padStart(7, '0');
  return `${text}Ma.png`;
}

async function ready(page, timeoutMs = 180000) {
  // The app fetches /api/initial and the newest frame on load. While the engine
  // is mid-step those requests queue behind it, so this can legitimately take
  // minutes -- every coordinate readout is NaN until it lands.
  const t0 = Date.now();
  while (Date.now() - t0 < timeoutMs) {
    const state = await page.evaluate(() => ({
      empty: document.getElementById('map-empty')?.hidden,
      frames: Number(document.getElementById('timeline')?.max || 0),
      displayIndex: window.deepTimeDisplayState?.()?.frameIndex ?? null,
    }));
    // One saved initial epoch is ready even though its timeline is disabled.
    // Check the loaded frame identity so an editor preview cannot qualify.
    if (state.empty && Number.isInteger(state.displayIndex) &&
        state.displayIndex >= 0 && state.displayIndex <= state.frames) return state.frames;
    await sleep(1000);
  }
  throw new Error('app did not finish loading');
}

async function maximise(page) {
  // The map is one panel in a page of controls, so it only ever gets a fraction
  // of the viewport. Promote it to fill the window and let the app's own resize
  // handler (app.js:2730 -> render()) redraw at the new size. This is the app's
  // real renderer at a larger canvas, not an upscale, and it avoids the headless
  // fullscreen API entirely.
  await page.addStyleTag({
    content: `#map-shell { position: fixed !important; inset: 0 !important;
                           width: 100vw !important; height: 100vh !important;
                           z-index: 2147483647 !important; margin: 0 !important; }
              #map-shell canvas { width: 100% !important; height: 100% !important; }
              #exit-map-fullscreen, .map-scale, .map-coordinate { display: none !important; }`,
  });
  await page.evaluate(() => window.dispatchEvent(new Event('resize')));
  await sleep(1500);
}

async function showFrame(page, index) {
  // The displayed epoch before we ask for anything. A frame load takes seconds
  // (measured ~8 s against a busy engine) and the OLD value is present the whole
  // time, so settling on "stable" alone captures the previous frame. Wait for it
  // to actually change, then settle.
  const before = await page.evaluate(() =>
    document.getElementById('time-value')?.textContent);

  await page.evaluate((i) => {
    const t = document.getElementById('timeline');
    t.value = String(i);
    t.dispatchEvent(new Event('input', { bubbles: true }));
  }, index);

  let last = null, stable = 0, changed = false;
  const t0 = Date.now();
  for (let k = 0; k < 480; k++) {
    await sleep(250);
    const now = await page.evaluate(() => ({
      time: document.getElementById('time-value')?.textContent,
      index: Number(document.getElementById('timeline')?.value),
      rendering: !document.getElementById('map-empty')?.hidden,
    }));
    if (now.index !== index || now.rendering) { stable = 0; continue; }
    if (now.time !== before) changed = true;
    if (now.time === last) stable++; else { last = now.time; stable = 0; }
    // Require a real change (or 20 s, for a frame whose epoch genuinely matches
    // the previous display) plus a settled second.
    if (stable >= 4 && (changed || Date.now() - t0 > 20000)) return { time: now.time, ...(await waitForDisplay(page, index)) };
  }
  return { time: last, ...(await waitForDisplay(page, index)) };
}

async function waitForDisplay(page, index) {
  // The app draws the stored 192x96 raster first and re-draws when the finer mesh
  // sampling arrives (/api/display; first request per view also builds a 1.7 s lookup).
  // Settling on the epoch label alone can therefore grab the coarse picture. Wait until
  // the app reports the fine grid is on screen, or that this layer has no finer grid.
  for (let k = 0; k < 80; k++) {
    const s = await page.evaluate(() => window.deepTimeDisplayState ? window.deepTimeDisplayState() : null);
    if (s === null) return { width: null, hires: false, frameIndex: null };
    if (s.frameIndex === index && !s.pending && (!s.upgradable || (s.width || 0) >= 768))
      return { width: s.width, hires: !s.upgradable || s.width >= 768, frameIndex: s.frameIndex };
    await sleep(250);
  }
  const s = await page.evaluate(() => window.deepTimeDisplayState());
  return { width: s?.width ?? null, hires: false, frameIndex: s?.frameIndex ?? null };
}

async function grab(page, file) {
  const data = await page.evaluate(() => {
    const c = document.getElementById('world-map');
    return c.toDataURL('image/png');
  });
  const buf = Buffer.from(data.split(',')[1], 'base64');
  const tmp = file + '.tmp';
  fs.writeFileSync(tmp, buf);
  fs.renameSync(tmp, file);          // atomic, so a sync client never sees a partial
  return buf.length;
}

(async () => {
  const browser = await puppeteer.launch({
    headless: 'new',
    args: ['--no-sandbox', '--disable-dev-shm-usage'],
  });
  try {
    const page = await browser.newPage();
    await page.setViewport({ width: VIEWPORT_W, height: VIEWPORT_H, deviceScaleFactor: 1 });
    page.on('pageerror', (e) => console.error('PAGE ERROR', e.message));
    await page.goto(URL, { waitUntil: 'domcontentloaded', timeout: 120000 });

    let maxIndex = await ready(page);
    await maximise(page);
    const size = await page.evaluate(() => {
      const c = document.getElementById('world-map');
      return { w: c.width, h: c.height };
    });
    await loadAges();
    console.log(`app ready, ${maxIndex + 1} frames, canvas ${size.w}x${size.h}; naming by age, newest ${nameFor(maxIndex)}`);

    const doRange = async (from, to) => {
      for (let i = from; i <= to; i++) {
        const name = nameFor(i);
        const file = path.join(OUT, name);
        if (fs.existsSync(file) && !REPLACE) continue;
        const started = Date.now();
        let shown = await showFrame(page, i);
        if (shown.hires !== true || shown.frameIndex !== i) {
          // One retry: a transient /api/display failure falls back to the coarse raster.
          await page.reload({ waitUntil: 'domcontentloaded', timeout: 120000 });
          await ready(page); await maximise(page);
          shown = await showFrame(page, i);
        }
        if (shown.hires !== true || shown.frameIndex !== i)
          throw new Error(`refusing ${name}: frame ${shown.frameIndex}, display width ${shown.width}, high-res ${shown.hires}`);
        const bytes = await grab(page, file);
        const grid = shown.width ? `${shown.width}px${shown.hires === false ? ' COARSE' : ''}` : 'n/a';
        console.log(`${new Date().toTimeString().slice(0, 8)}  ${name.padEnd(12)}  ` +
          `${shown.time} Myr  ${grid}  ${(bytes / 1e6).toFixed(2)} MB  ${((Date.now() - started) / 1000).toFixed(0)}s`);
      }
    };

    if (arg('latest')) {
      await doRange(maxIndex, maxIndex);
    } else {
      await doRange(FROM, Number(arg('to', maxIndex)));
    }

    while (WATCH) {
      await sleep(INTERVAL);
      // Reload rather than poll in place: the run appends frames, and a long-lived
      // page holds a stale frame list.
      await page.reload({ waitUntil: 'domcontentloaded', timeout: 120000 });
      try {
        const found = await ready(page); await maximise(page);
        await loadAges();   // new frames appended since the last pass
        if (found > maxIndex) maxIndex = found;
        await doRange(FROM, maxIndex);
      } catch (e) {
        console.error(`${new Date().toTimeString().slice(0, 8)}  reload failed: ${e.message}`);
      }
    }
  } finally {
    await browser.close();
  }
})().catch((e) => { console.error(e); process.exit(1); });
