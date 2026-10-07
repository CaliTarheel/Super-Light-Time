/* Camera and fallback rendering tests; no engine, network, or dependencies. */
const assert = require('node:assert/strict');
const {basis,project,unproject,create} = require('../web/globe.js');
const near = (a,b,t=1e-11) => assert.ok(Math.abs(a-b)<=t, `${a} differs from ${b}`);
const xyz = (p) => { const l=p.lon*Math.PI/180,a=p.lat*Math.PI/180; return [Math.cos(a)*Math.cos(l),Math.cos(a)*Math.sin(l),Math.sin(a)]; };
const dot = (a,b) => a.reduce((s,v,i)=>s+v*b[i],0);
let checks=0;
function check(name,fn) { fn(); checks++; console.log('PASS '+name); }

check('camera axes are orthonormal at arbitrary centers and poles',()=>{
  for(const [lon,lat] of [[0,0],[-98,20],[179,75],[-142,-90],[0,90]]) {
    const b=basis(lon,lat);
    for(const v of Object.values(b))near(dot(v,v),1);
    near(dot(b.center,b.east),0);near(dot(b.center,b.north),0);near(dot(b.east,b.north),0);
  }
});
check('cardinal directions and the far hemisphere have the expected projection',()=>{
  const center=project(0,0);near(center.x,0);near(center.y,0);near(center.z,1);assert.ok(center.visible);
  near(project(90,0).x,1);near(project(-90,0).x,-1);near(project(0,90).y,1);near(project(0,-90).y,-1);
  assert.equal(project(180,0).visible,false);
  near(unproject(0,0).lon,0);near(unproject(0,0).lat,0);
});
check('visible points round-trip across the seam and near both poles',()=>{
  for(const pose of [{lon:180,lat:0},{lon:-98,lat:20},{lon:30,lat:89},{lon:110,lat:-89}]) {
    for(const [x,y] of [[0,0],[.8,0],[-.8,0],[0,.999999],[.3,-.7]]) {
      const p=unproject(x,y,pose),q=project(p.lon,p.lat,pose);assert.ok(q.visible);near(q.x,x,2e-10);near(q.y,y,2e-10);
      assert.ok(p.u>=0&&p.u<1&&p.v>=0&&p.v<=1);
    }
  }
  const a=project(179.9,0,{lon:180}),b=project(-179.9,0,{lon:180});
  assert.ok(a.visible&&b.visible&&a.x<0&&b.x>0);near(Math.abs(a.x),Math.abs(b.x));
});
check('exact poles and longitude aliases represent the same spherical point',()=>{
  for(const lat of [-90,90]) {
    const p=unproject(0,0,{lon:123,lat});near(p.lat,lat);assert.ok(Number.isFinite(p.lon));
    near(dot(xyz(p),xyz({lon:-42,lat})),1);
  }
  assert.deepEqual(unproject(0,0,{lon:180}),unproject(0,0,{lon:-180}));
});
check('outside disk and invalid coordinates cannot pick the hidden hemisphere',()=>{
  for(const [x,y] of [[1.00001,0],[1,1],[NaN,0],[0,Infinity]])assert.equal(unproject(x,y),null);
  assert.equal(project(NaN,0),null);assert.equal(project(0,91),null);
  assert.ok(unproject(1,0));assert.ok(unproject(1+1e-14,0));
  near(unproject(0,0,{lon:180}).lon,-180);
});

function fakeCanvas(w=101,h=101) {
  const handlers=new Map();
  const ctx={pixels:null,drawImage(){},fillRect(){},
    createImageData(width,height){return {width,height,data:new Uint8ClampedArray(width*height*4)};},
    putImageData(p){this.pixels=p;}};
  const doc={createElement(){return {width:1,height:1,getContext(){return null;},addEventListener(k,v){handlers.set(k,v);},removeEventListener(k){handlers.delete(k);}};}};
  return {canvas:{width:w,height:h,ownerDocument:doc,getContext(type){assert.equal(type,'2d');return ctx;},
    getBoundingClientRect(){return {left:10,top:20,width:w,height:h};}},ctx,handlers};
}
const source=(width,height,pixel)=>{
  const data=new Uint8ClampedArray(width*height*4);
  for(let y=0;y<height;y++)for(let x=0;x<width;x++)data.set(pixel(x,y),4*(y*width+x));
  return {width,height,pixels:data,getContext(){return {getImageData(){return {width,height,data:data.slice()};}};}};
};
check('CPU fallback samples north at the top and wraps the texture seam',()=>{
  const {canvas,ctx}=fakeCanvas(),v=create(canvas);
  assert.equal(v.backend,'cpu');
  v.setTexture(source(4,2,(x,y)=>y?[0,0,200,255]:[200,0,0,255]));
  v.draw({lon:0,lat:90,zoom:1});
  const center=(50*101+50)*4;assert.deepEqual(Array.from(ctx.pixels.data.slice(center,center+4)),[200,0,0,255]);
  v.draw({lon:0,lat:-90,zoom:1});assert.deepEqual(Array.from(ctx.pixels.data.slice(center,center+4)),[0,0,200,255]);
  v.setTexture(source(4,2,(x)=>[x===0?200:0,x===3?200:0,0,255]));
  v.draw({lon:180,lat:0});assert.deepEqual(Array.from(ctx.pixels.data.slice(center,center+4)),[100,100,0,255]);
  assert.deepEqual(Array.from(ctx.pixels.data.slice(0,4)),[12,24,33,255]);v.destroy();
});
check('CSS picks share backing-store geometry, obey zoom, and reject the canvas exterior',()=>{
  const {canvas}=fakeCanvas(200,100),v=create(canvas),old=globalThis.devicePixelRatio;
  globalThis.devicePixelRatio=3;
  try {
    v.draw({lon:-98,lat:20,zoom:1});assert.equal(canvas.width,300);assert.equal(canvas.height,150);
    const p=v.pick(110,70);near(p.lon,-98);near(p.lat,20);
    assert.equal(v.pick(10,20),null);assert.equal(v.pick(9,70,{zoom:4}),null);
    assert.equal(v.pick(165,70,{zoom:1}),null);assert.ok(v.pick(165,70,{zoom:2}));
  } finally { globalThis.devicePixelRatio=old;v.destroy(); }
});
check('texture capture is immutable and destruction releases event handlers',()=>{
  const {canvas,ctx,handlers}=fakeCanvas(),v=create(canvas);
  const s=source(2,1,()=>[100,120,140,255]);v.setTexture(s);s.pixels.fill(0);v.draw();
  const center=(50*101+50)*4;assert.deepEqual(Array.from(ctx.pixels.data.slice(center,center+4)),[100,120,140,255]);
  let prevented=false;handlers.get('webglcontextlost')({preventDefault(){prevented=true;}});
  assert.ok(prevented);assert.equal(v.backend,'cpu');handlers.get('webglcontextrestored')();
  assert.deepEqual(Array.from(ctx.pixels.data.slice(center,center+4)),[100,120,140,255]);
  const pixels=ctx.pixels;v.destroy();v.destroy();assert.equal(handlers.size,0);assert.equal(v.pick(50,50),null);
  v.draw();assert.equal(ctx.pixels,pixels);assert.throws(()=>v.setTexture(s),/destroyed/);
});
console.log(`${checks} globe checks passed.`);
