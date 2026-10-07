/* Local orthographic camera for an existing equirectangular review texture. */
(function (root) {
  "use strict";
  const RAD = Math.PI / 180, TAU = 2 * Math.PI;
  const clamp = (x, a, b) => Math.max(a, Math.min(b, x));
  const wrap = x => ((x + 180) % 360 + 360) % 360 - 180;
  const finite = (x, fallback) => Number.isFinite(Number(x)) ? Number(x) : fallback;
  const dot = (a, b) => a[0]*b[0] + a[1]*b[1] + a[2]*b[2];
  function pose(value = {}) {
    return {lon: wrap(finite(value.lon, 0)), lat: clamp(finite(value.lat, 0), -90, 90), zoom: clamp(finite(value.zoom, 1), 1, 4)};
  }
  function basis(lon = 0, lat = 0) {
    const p = pose({lon, lat}), l = p.lon * RAD, a = p.lat * RAD;
    return {center: [Math.cos(a)*Math.cos(l), Math.cos(a)*Math.sin(l), Math.sin(a)],
      east: [-Math.sin(l), Math.cos(l), 0],
      north: [-Math.sin(a)*Math.cos(l), -Math.sin(a)*Math.sin(l), Math.cos(a)]};
  }
  function xyz(lon, lat) {
    const l = lon * RAD, a = lat * RAD;
    return [Math.cos(a)*Math.cos(l), Math.cos(a)*Math.sin(l), Math.sin(a)];
  }
  // Normalized disk coordinates: x right, y up. Hidden points remain marked.
  function project(lon, lat, camera = {}) {
    if (!Number.isFinite(lon) || !Number.isFinite(lat) || Math.abs(lat) > 90) return null;
    const p = pose(camera), b = basis(p.lon, p.lat), point = xyz(lon, lat), z = dot(point, b.center);
    return {x: dot(point, b.east), y: dot(point, b.north), z, visible: z >= -1e-12};
  }
  function coordinates(point) {
    const lon = Math.hypot(point[0], point[1]) < 1e-14 ? 0 : wrap(Math.atan2(point[1], point[0]) / RAD);
    const lat = Math.asin(clamp(point[2], -1, 1)) / RAD;
    return {u: (lon + 180) / 360, v: clamp(.5 - lat / 180, 0, 1), lon, lat};
  }
  function unproject(x, y, camera = {}) {
    if (!Number.isFinite(x) || !Number.isFinite(y)) return null;
    const r2 = x*x + y*y;
    if (r2 > 1 + 1e-12) return null;
    // Roundoff just beyond the limb is projected back to the limb itself.
    if (r2 > 1) { const scale = 1 / Math.sqrt(r2); x *= scale; y *= scale; }
    const p = pose(camera), b = basis(p.lon, p.lat), z = Math.sqrt(Math.max(0, 1-x*x-y*y));
    return coordinates(b.center.map((v, i) => v*z + b.east[i]*x + b.north[i]*y));
  }
  const vertexSource = `attribute vec2 position;
    void main() { gl_Position = vec4(position, 0.0, 1.0); }`;
  const fragmentSource = `precision highp float;
    uniform vec2 resolution;
    uniform float radius;
    uniform vec3 center, east, north;
    uniform sampler2D surface;
    uniform float sourceWidth;
    void main() {
      vec2 q = (gl_FragCoord.xy - resolution * 0.5) / radius;
      float r2 = dot(q,q);
      if (r2 > 1.0) { gl_FragColor = vec4(12.0/255.0,24.0/255.0,33.0/255.0,1.0); return; }
      float depth = sqrt(max(0.0,1.0-r2));
      vec3 p = center*depth + east*q.x + north*q.y;
      float u = fract(atan(p.y,p.x)/6.283185307179586 + 0.5);
      float v = 0.5 - asin(clamp(p.z,-1.0,1.0))/3.141592653589793;
      vec3 color = texture2D(surface,vec2((u*sourceWidth+1.0)/(sourceWidth+2.0),v)).rgb;
      gl_FragColor = vec4(color*(0.86+0.14*depth),1.0);
    }`;

  function create(canvas) {
    const doc = canvas.ownerDocument || root.document;
    if (!doc?.createElement) throw new Error("The globe viewer needs a canvas document.");
    const ctx = canvas.getContext("2d", {alpha: false});
    if (!ctx) throw new Error("The globe viewer cannot create its display canvas.");
    // A separate GPU canvas allows a genuine CPU fallback on context loss.
    const gpuCanvas = doc.createElement("canvas");
    let gl = null, gpu = null, image = null, current = pose(), disposed = false, lost = false, backend = "cpu";
    let cpuPixels = null;
    try { gl = gpuCanvas.getContext("webgl", {alpha: false, antialias: false, depth: false, stencil: false, premultipliedAlpha: false}); } catch (_) {}

    function disposeGPU() {
      if (!gpu || !gl || lost) { gpu = null; return; }
      gl.deleteTexture(gpu.texture); gl.deleteBuffer(gpu.buffer); gl.deleteProgram(gpu.program); gpu = null;
    }
    function initializeGPU() {
      if (!gl || lost || disposed) return false;
      let vs = null, fs = null, program = null, buffer = null, texture = null;
      try {
        function shader(type, source) {
          const s = gl.createShader(type); gl.shaderSource(s, source); gl.compileShader(s);
          if (!gl.getShaderParameter(s, gl.COMPILE_STATUS)) { gl.deleteShader(s); throw new Error("Globe shader unavailable"); }
          return s;
        }
        vs = shader(gl.VERTEX_SHADER, vertexSource);
        fs = shader(gl.FRAGMENT_SHADER, fragmentSource);
        program = gl.createProgram(); gl.attachShader(program, vs); gl.attachShader(program, fs); gl.linkProgram(program);
        if (!gl.getProgramParameter(program, gl.LINK_STATUS)) throw new Error("Globe shader link unavailable");
        buffer = gl.createBuffer(); gl.bindBuffer(gl.ARRAY_BUFFER, buffer);
        gl.bufferData(gl.ARRAY_BUFFER, new Float32Array([-1,-1, 1,-1, -1,1, -1,1, 1,-1, 1,1]), gl.STATIC_DRAW);
        texture = gl.createTexture();
        gpu = {program, buffer, texture, position: gl.getAttribLocation(program,"position"), uniforms: {}};
        for (const name of ["resolution","radius","center","east","north","surface","sourceWidth"])
          gpu.uniforms[name] = gl.getUniformLocation(program, name);
        backend = "webgl";
        return true;
      } catch (_) {
        if (texture) gl.deleteTexture(texture);
        if (buffer) gl.deleteBuffer(buffer);
        if (program) gl.deleteProgram(program);
        gpu = null; backend = "cpu"; return false;
      } finally { if (vs) gl.deleteShader(vs); if (fs) gl.deleteShader(fs); }
    }
    function upload() {
      if (!gpu || !image) return;
      const w = image.width, h = image.height;
      if (w + 2 > gl.getParameter(gl.MAX_TEXTURE_SIZE) || h > gl.getParameter(gl.MAX_TEXTURE_SIZE)) {
        disposeGPU(); backend = "cpu"; return;
      }
      // Neighbour columns make linear filtering periodic with NPOT textures.
      const padded = new Uint8Array((w+2)*h*4);
      for (let y = 0; y < h; y++) {
        const from = y*w*4, to = y*(w+2)*4;
        padded.set(image.data.subarray(from+(w-1)*4,from+w*4), to);
        padded.set(image.data.subarray(from,from+w*4), to+4);
        padded.set(image.data.subarray(from,from+4), to+(w+1)*4);
      }
      gl.bindTexture(gl.TEXTURE_2D, gpu.texture);
      gl.pixelStorei(gl.UNPACK_ALIGNMENT, 1);
      gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE);
      gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE);
      gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.LINEAR);
      gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.LINEAR);
      gl.texImage2D(gl.TEXTURE_2D,0,gl.RGBA,w+2,h,0,gl.RGBA,gl.UNSIGNED_BYTE,padded);
      if (gl.getError() !== gl.NO_ERROR) { disposeGPU(); backend = "cpu"; }
    }
    function resize() {
      const rect = canvas.getBoundingClientRect(), ratio = clamp(finite(root.devicePixelRatio, 1), 1, 1.5);
      const w = Math.max(1, Math.round(rect.width*ratio)), h = Math.max(1, Math.round(rect.height*ratio));
      if (canvas.width !== w || canvas.height !== h) { canvas.width = w; canvas.height = h; cpuPixels = null; }
      if (gpuCanvas.width !== w || gpuCanvas.height !== h) { gpuCanvas.width = w; gpuCanvas.height = h; }
      return {w, h, radius: .44*Math.min(w,h)*current.zoom};
    }
    function drawCPU(w, h, radius, b) {
      if (!cpuPixels || cpuPixels.width !== w || cpuPixels.height !== h) cpuPixels = ctx.createImageData(w,h);
      const out = cpuPixels.data, sw = image.width, sh = image.height, source = image.data;
      for (let y = 0; y < h; y++) for (let x = 0; x < w; x++) {
        const at = (y*w+x)*4, nx = (x+.5-w*.5)/radius, ny = (h*.5-y-.5)/radius, r2 = nx*nx+ny*ny;
        out[at+3] = 255;
        if (r2 > 1) { out[at] = 12; out[at+1] = 24; out[at+2] = 33; continue; }
        const z = Math.sqrt(Math.max(0,1-r2)), px = b.center[0]*z+b.east[0]*nx+b.north[0]*ny;
        const py = b.center[1]*z+b.east[1]*nx+b.north[1]*ny, pz = b.center[2]*z+b.east[2]*nx+b.north[2]*ny;
        const u = ((Math.atan2(py,px)/TAU+.5)%1+1)%1, v = .5-Math.asin(clamp(pz,-1,1))/Math.PI;
        const sx = u*sw-.5, sy = clamp(v*sh-.5,0,sh-1), ix = Math.floor(sx), iy = Math.floor(sy);
        const fx = sx-ix, fy = sy-iy, x0 = (ix+sw)%sw, x1 = (x0+1)%sw, y1 = Math.min(sh-1,iy+1);
        const a = (iy*sw+x0)*4, c = (iy*sw+x1)*4, d = (y1*sw+x0)*4, e = (y1*sw+x1)*4;
        for (let k = 0; k < 3; k++) out[at+k] = ((source[a+k]*(1-fx)+source[c+k]*fx)*(1-fy)+(source[d+k]*(1-fx)+source[e+k]*fx)*fy)*(.86+.14*z);
      }
      ctx.putImageData(cpuPixels,0,0);
    }
    function draw(camera = current) {
      if (disposed) return;
      current = pose(camera);
      const {w,h,radius} = resize(), b = basis(current.lon,current.lat);
      if (!image) { ctx.fillStyle = "#0c1821"; ctx.fillRect(0,0,w,h); return; }
      if (!gpu || lost) { backend = "cpu"; drawCPU(w,h,radius,b); return; }
      gl.viewport(0,0,w,h); gl.useProgram(gpu.program);
      gl.bindBuffer(gl.ARRAY_BUFFER,gpu.buffer); gl.enableVertexAttribArray(gpu.position); gl.vertexAttribPointer(gpu.position,2,gl.FLOAT,false,0,0);
      gl.activeTexture(gl.TEXTURE0); gl.bindTexture(gl.TEXTURE_2D,gpu.texture);
      gl.uniform1i(gpu.uniforms.surface,0); gl.uniform1f(gpu.uniforms.sourceWidth,image.width);
      gl.uniform2f(gpu.uniforms.resolution,w,h); gl.uniform1f(gpu.uniforms.radius,radius);
      for (const name of ["center","east","north"]) gl.uniform3fv(gpu.uniforms[name],b[name]);
      gl.drawArrays(gl.TRIANGLES,0,6);
      ctx.drawImage(gpuCanvas,0,0,w,h);
    }
    function setTexture(sourceCanvas) {
      if (disposed) throw new Error("The globe viewer has been destroyed.");
      const w = sourceCanvas?.width, h = sourceCanvas?.height;
      if (!Number.isInteger(w) || !Number.isInteger(h) || w < 1 || h < 1) throw new Error("The globe texture needs positive canvas dimensions.");
      image = sourceCanvas.getContext("2d").getImageData(0,0,w,h);
      if (!gpu && gl && !lost) initializeGPU();
      upload();
    }
    function pick(clientX, clientY, camera = current) {
      if (disposed || !Number.isFinite(clientX) || !Number.isFinite(clientY)) return null;
      const rect = canvas.getBoundingClientRect();
      if (!(rect.width > 0 && rect.height > 0) || clientX < rect.left || clientY < rect.top || clientX > rect.left+rect.width || clientY > rect.top+rect.height) return null;
      const p = pose(camera), radius = .44*Math.min(canvas.width,canvas.height)*p.zoom;
      const x = ((clientX-rect.left)/rect.width-.5)*canvas.width/radius;
      const y = (.5-(clientY-rect.top)/rect.height)*canvas.height/radius;
      return unproject(x,y,p);
    }
    function contextLost(event) { event.preventDefault(); lost = true; gpu = null; backend = "cpu"; draw(); }
    function contextRestored() { if (disposed) return; lost = false; initializeGPU(); upload(); draw(); }
    gpuCanvas.addEventListener("webglcontextlost",contextLost);
    gpuCanvas.addEventListener("webglcontextrestored",contextRestored);
    initializeGPU();
    return {setTexture,draw,pick,
      get backend() { return backend; },
      reset() { draw({lon:0,lat:0,zoom:1}); },
      destroy() { if (disposed) return; disposeGPU(); disposed = true; image = null; cpuPixels = null;
        gpuCanvas.removeEventListener("webglcontextlost",contextLost); gpuCanvas.removeEventListener("webglcontextrestored",contextRestored); }
    };
  }
  const api = {create,basis,project,unproject};
  if (typeof module !== "undefined" && module.exports) module.exports = api;
  else root.DeepTimeGlobe = api;
})(globalThis);
