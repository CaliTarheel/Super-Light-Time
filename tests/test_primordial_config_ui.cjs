/* Exercise the actual form serialization, including old histories and partial grid updates. */
const fs = require('node:fs');
const vm = require('node:vm');
const assert = require('node:assert/strict');
const path = require('node:path');
const root = path.resolve(__dirname, '..');
const source = fs.readFileSync(path.join(root, 'web/app.js'), 'utf8');
const html = fs.readFileSync(path.join(root, 'web/index.html'), 'utf8');
const nodes = new Map();
for (const match of html.matchAll(/\bid="([^"]+)"/g)) nodes.set(match[1], {
  value: '1', checked: false, options: [{value: '1024'}], add(option) { this.options.push(option); },
  // updateResolutionNotes toggles the legacy force-coefficient fields (9c25816).
  classList: {toggle() {}, add() {}, remove() {}},
});
const get = id => { assert.ok(nodes.has(id), `Control exists: ${id}`); return nodes.get(id); };
function actualFunction(name) {
  const start = source.indexOf(`  function ${name}(`);
  assert.ok(start >= 0);
  return source.slice(start, source.indexOf('\n  function ', start + 1));
}
const sandbox = { $: get, configKeys: ['seed', 'width'], state: {initial:{}}, lipDraft:{events:[]},
  oceanAttachmentDraft: {entries:[],plateCount:null}, isBusy:()=>false, toast() {},
  document:{querySelector:()=>({textContent:''})},
  normalizeWorldDesign: () => ({version:1, enabled:false}), structuredClone,
  Option: function(text,value) { this.text=text; this.value=value; },
  updateLipDraft() {}, updateRangeOutputs() {}, updateResolutionNotes() {},
};
vm.createContext(sandbox);
vm.runInContext(actualFunction('readConfig') + actualFunction('fillConfig') + actualFunction('updateOceanAttachmentNote') + actualFunction('removeOceanAttachments') + actualFunction('updateResolutionNotes'), sandbox);

sandbox.fillConfig({seed:12,width:1024,physics_profile:'reviewed_v1',
  primordial_subduction:{enabled:true,initial_slab_depth_km:100},
  primordial_ocean:{enabled:true,plate_count:3,half_spreading_rate_cm_yr:1.5,maximum_age_myr:200}});
let result = sandbox.readConfig();
assert.equal(result.primordial_subduction.enabled, true);
assert.equal(result.primordial_subduction.initial_slab_depth_km,100);
assert.deepEqual(JSON.parse(JSON.stringify(result.primordial_ocean)), {
  enabled:true,plate_count:3,half_spreading_rate_cm_yr:1.5,maximum_age_myr:200,
});
assert.equal(result.subduction_response,'fixed_trench');
assert.equal(result.retained_phases,'disabled');
// A complete run saved before the enhanced-rifting control ran with it off;
// off sends only the flag and the server fills the unused parameters.
assert.deepEqual(JSON.parse(JSON.stringify(result.enhanced_rifting)), {version:1,enabled:false});
assert.equal(get('enhanced-rifting-seed').value,37);

// Saved advanced physics survives reload and a partial display-resolution update.
const modern = {seed:37,width:1024,physics_profile:'reviewed_v1',
  subduction_response:'moving_hinge_v1',retained_phases:'thermal_v1',
  enhanced_rifting:{version:1,enabled:true,seed:41,amplitude:0.3,correlation_km:500,craton_margin_km:120},
  primordial_subduction:{enabled:true,initial_slab_depth_km:100},
  primordial_ocean:{enabled:true,plate_count:3,half_spreading_rate_cm_yr:1.5,maximum_age_myr:200,
    continental_attachments:[{region_index:1,continental_plate_uid:3}]}};
sandbox.fillConfig(modern);
result=sandbox.readConfig();
assert.equal(result.subduction_response,'moving_hinge_v1');
assert.equal(result.retained_phases,'thermal_v1');
assert.equal(get('advanced-physics-controls').hidden,false);
assert.match(get('physics-profile-note').textContent,/Moving-trench slab gravity/);
assert.match(get('physics-profile-note').textContent,/temperature-dependent phase evolution/);
assert.match(get('physics-profile-note').textContent,/Enhanced rifting: continental crust starts up to 30% weaker \(about 15% on average\) in a seeded pattern that becomes unrelated over about 1600 km \(800 px/);
assert.match(get('physics-profile-note').textContent,/durable scar up to 50% weaker/);
assert.deepEqual(JSON.parse(JSON.stringify(result.enhanced_rifting)),modern.enhanced_rifting);
assert.doesNotMatch(get('physics-profile-note').textContent,/3 independent plates/);
assert.deepEqual(JSON.parse(JSON.stringify(result.primordial_ocean)),modern.primordial_ocean);
// Neither later mutation of imported JSON nor a previous export mutates the form draft.
modern.primordial_ocean.continental_attachments[0].continental_plate_uid=99;
result.primordial_ocean.continental_attachments[0].continental_plate_uid=88;
assert.equal(sandbox.readConfig().primordial_ocean.continental_attachments[0].continental_plate_uid,3);

sandbox.fillConfig({width:1024});
assert.equal(sandbox.readConfig().primordial_subduction.enabled,true);
assert.equal(sandbox.readConfig().physics_profile,'reviewed_v1');
assert.equal(sandbox.readConfig().primordial_ocean.enabled,true);
assert.equal(sandbox.readConfig().primordial_ocean.half_spreading_rate_cm_yr,1.5);
assert.equal(sandbox.readConfig().subduction_response,'moving_hinge_v1');
assert.equal(sandbox.readConfig().retained_phases,'thermal_v1');
assert.equal(sandbox.readConfig().primordial_ocean.continental_attachments[0].continental_plate_uid,3);
assert.deepEqual(JSON.parse(JSON.stringify(sandbox.readConfig().enhanced_rifting)),modern.enhanced_rifting);
sandbox.updateOceanAttachmentNote();
assert.equal(get('ocean-attachment-controls').hidden,false);
assert.match(get('ocean-attachment-summary').textContent,/water region 2 → continental plate 3/);

// Count changes must not silently attach a different partition to the same plate.
get('primordial-ocean-count').value='4';
assert.throws(()=>sandbox.readConfig(),/Restore that count or remove/);
sandbox.updateOceanAttachmentNote();
assert.match(get('ocean-attachment-summary').textContent,/restore 3 ocean regions/);
get('primordial-ocean-count').value='3';
get('primordial-ocean-enabled').checked=false;
assert.throws(()=>sandbox.readConfig(),/Enable it again or remove/);
sandbox.updateOceanAttachmentNote();
assert.match(get('ocean-attachment-summary').textContent,/enable ocean splitting again/);
get('primordial-ocean-enabled').checked=true;

get('physics_profile').value='legacy';
sandbox.updateResolutionNotes();
assert.equal(get('advanced-physics-controls').hidden,true);
assert.equal(sandbox.readConfig().primordial_subduction.enabled,false);
assert.equal(sandbox.readConfig().primordial_ocean.enabled,false);
assert.equal(sandbox.readConfig().subduction_response,'fixed_trench');
assert.equal(sandbox.readConfig().retained_phases,'disabled');
assert.equal(sandbox.readConfig().primordial_ocean.continental_attachments,undefined);
// Legacy always keeps the historical rupture gate and sends no hidden parameters.
assert.deepEqual(JSON.parse(JSON.stringify(sandbox.readConfig().enhanced_rifting)),{version:1,enabled:false});
sandbox.updateOceanAttachmentNote();
assert.match(get('ocean-attachment-summary').textContent,/Inactive in Legacy/);
get('physics_profile').value='reviewed_v1';
assert.equal(sandbox.readConfig().enhanced_rifting.enabled,true);
assert.equal(sandbox.readConfig().primordial_ocean.continental_attachments[0].continental_plate_uid,3);
assert.equal(sandbox.readConfig().subduction_response,'moving_hinge_v1');

// Explicit removal permits changing the count; changing a running form does not.
sandbox.isBusy=()=>true;
sandbox.removeOceanAttachments();
assert.equal(sandbox.readConfig().primordial_ocean.continental_attachments.length,1);
sandbox.isBusy=()=>false;
sandbox.removeOceanAttachments();
get('primordial-ocean-count').value='4';
assert.equal(sandbox.readConfig().primordial_ocean.continental_attachments,undefined);
assert.equal(sandbox.readConfig().primordial_ocean.plate_count,4);
sandbox.updateOceanAttachmentNote();
assert.equal(get('ocean-attachment-controls').hidden,true);

// Loading a complete old run clears every newer setting left by another run.
sandbox.fillConfig(modern);

sandbox.fillConfig({seed:12,width:1024,physics_profile:'reviewed_v1'});
result=sandbox.readConfig();
assert.equal(result.primordial_subduction.enabled,false);
assert.equal(result.primordial_subduction.initial_slab_depth_km,100);
assert.equal(result.subduction_response,'fixed_trench');
assert.equal(result.retained_phases,'disabled');
assert.equal(result.enhanced_rifting.enabled,false);
assert.equal(get('enhanced-rifting-seed').value,37);
assert.deepEqual(JSON.parse(JSON.stringify(result.primordial_ocean)), {
  enabled:false,plate_count:4,half_spreading_rate_cm_yr:2,maximum_age_myr:180,
});
sandbox.updateOceanAttachmentNote();
assert.equal(get('ocean-attachment-controls').hidden,true);
sandbox.fillConfig({seed:12,width:1024});
assert.equal(sandbox.readConfig().physics_profile,'legacy');
assert.equal(sandbox.readConfig().subduction_response,'fixed_trench');
assert.equal(sandbox.readConfig().retained_phases,'disabled');
assert.equal(sandbox.readConfig().enhanced_rifting.enabled,false);

// The server's fresh-world defaults (server.fresh_world_defaults) select the reviewed laws;
// the moving hinge waits for the N1 back-arc repair.
sandbox.fillConfig({seed:12,width:1024,physics_profile:'reviewed_v1',
  subduction_response:'fixed_trench',retained_phases:'thermal_v1',
  enhanced_rifting:{version:1,enabled:true,seed:37,amplitude:0.25,correlation_km:350,craton_margin_km:150},
  primordial_subduction:{enabled:false,initial_slab_depth_km:100},
  primordial_ocean:{enabled:false,plate_count:4,half_spreading_rate_cm_yr:2,maximum_age_myr:180},
  lip_events:{version:1,enabled:false,seed:37001,generated_count:0,events:[]}});
result=sandbox.readConfig();
assert.equal(result.physics_profile,'reviewed_v1');
assert.equal(result.subduction_response,'fixed_trench');
assert.equal(result.retained_phases,'thermal_v1');
assert.deepEqual(JSON.parse(JSON.stringify(result.enhanced_rifting)),
  {version:1,enabled:true,seed:37,amplitude:0.25,correlation_km:350,craton_margin_km:150});
assert.match(get('physics-profile-note').textContent,/up to 25% weaker \(about 12\.5% on average\) in a seeded pattern that becomes unrelated over about 1100 km \(550 px/);
// Unticking the box under reviewed sends only the flag.
get('enhanced-rifting-enabled').checked=false;
assert.deepEqual(JSON.parse(JSON.stringify(sandbox.readConfig().enhanced_rifting)),{version:1,enabled:false});
get('enhanced-rifting-enabled').checked=true;

vm.runInContext(actualFunction('updateRecordedPhysicsNote'), sandbox);
sandbox.updateRecordedPhysicsNote({width:1024,height:512,physics_profile_version:1,
  primordial_ocean_diagnostics:{enabled:true,plate_count:3}}, false);
assert.match(get('recorded-physics-note').textContent, /initially 3 ocean plates/);
// Recorded evidence is independent of the next experiment's selected settings.
sandbox.updateRecordedPhysicsNote({width:1024,height:512,physics_profile_version:1,
  subduction_response_version:1,retained_dense_crust_version:1,
  primordial_ocean_diagnostics:{enabled:true,plate_count:2,water_region_count:4,realized_plate_count:4,
    independent_ocean_plate_count:2,
    continental_attachments:[{region_index:1,continental_plate_uid:3},{region_index:3,continental_plate_uid:5}]}},false);
assert.match(get('recorded-physics-note').textContent,/moving-trench overriding-plate response/);
assert.match(get('recorded-physics-note').textContent,/retained dense crust with thermal phase evolution/);
assert.doesNotMatch(get('recorded-physics-note').textContent,/enhanced rifting/);
assert.match(get('recorded-physics-note').textContent,/initially 4 ocean regions: 2 attached to continental plates, 2 independent/);
assert.doesNotMatch(get('recorded-physics-note').textContent,/initially 4 ocean plates/);
// Earlier diagnostic shapes may omit the explicit region or independent count.
sandbox.updateRecordedPhysicsNote({width:1024,height:512,physics_profile_version:1,
  primordial_ocean_diagnostics:{enabled:true,plate_count:2,realized_plate_count:4,
    continental_attachments:[{region_index:1,continental_plate_uid:3},{region_index:3,continental_plate_uid:5}]}},false);
assert.match(get('recorded-physics-note').textContent,/initially 4 ocean regions: 2 attached to continental plates, 2 independent/);
sandbox.updateRecordedPhysicsNote({width:1024,height:512,physics_profile_version:1,
  primordial_ocean_diagnostics:{enabled:true,plate_count:4,
    continental_attachments:[{region_index:1,continental_plate_uid:3}]}},false);
assert.match(get('recorded-physics-note').textContent,/initially 4 ocean regions: 1 attached to continental plates, 3 independent/);
sandbox.fillConfig(modern);
sandbox.updateRecordedPhysicsNote({width:1024,height:512,physics_profile_version:1}, false);
assert.doesNotMatch(get('recorded-physics-note').textContent, /ocean plates/);
assert.match(get('recorded-physics-note').textContent,/historical fixed-trench response/);
assert.match(get('recorded-physics-note').textContent,/historical immediate dense-crust foundering/);
assert.doesNotMatch(get('recorded-physics-note').textContent,/moving-trench|retained dense crust with/);
sandbox.updateRecordedPhysicsNote({width:1024,height:512,physics_profile_version:1,
  physics_profile:{subduction_response_version:1,retained_dense_crust_version:1}}, false);
assert.match(get('recorded-physics-note').textContent,/moving-trench overriding-plate response/);
assert.match(get('recorded-physics-note').textContent,/retained dense crust with/);
sandbox.updateRecordedPhysicsNote({width:1024,height:512,physics_profile_version:1,
  subduction_response_version:1,retained_dense_crust_version:1,enhanced_rifting_version:1}, false);
assert.match(get('recorded-physics-note').textContent,/enhanced rifting: seeded inheritance, realized-strain rupture/);
assert.doesNotMatch(get('recorded-physics-note').textContent,/numerical gate/);
sandbox.updateRecordedPhysicsNote({width:1024,height:512,physics_profile_version:1,enhanced_rifting_version:1,
  rift_mechanics:{rupture_strain_threshold:0.35}}, false);
assert.match(get('recorded-physics-note').textContent,/realized-strain rupture \(numerical gate 0\.35 logarithmic strain\)/);
sandbox.updateRecordedPhysicsNote({width:1024,height:512,subduction_response_version:1},true);
assert.match(get('recorded-physics-note').textContent,/Starting artwork/);
assert.doesNotMatch(get('recorded-physics-note').textContent,/moving-trench/);
console.log('Advanced physics round trips, attachment guards, legacy defaults and recorded-history checks passed.');
