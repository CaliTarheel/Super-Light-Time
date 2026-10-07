"""Native root-local welding and retained legacy accretion contracts."""
from pathlib import Path
import sys
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from copy import deepcopy
import io
import json
import math
import unittest
import numpy as np
import adaptive_material
import checkpoint
import collision_contacts
import local_accretion as local
import localized_accretion as welding
import material_surface
from ridge_geometry import rotate
from tests.test_collision_contacts import rectangle
from tests.test_persistent_contact_accretion import world, mature


def observations(s):
    exact = local._persistent_contact_geometry(s)
    occupancy = local._native_occupancy(s)
    owners = set()
    for row in exact: owners.update((row['p'], row['q']))
    geometry = {p: local._material_components(s, p, occupancy) for p in owners}
    return local._persistent_observations(s, exact, geometry), geometry


def extended_world():
    s = world()
    surface = material_surface.initialize_surface(*rectangle(-2400, 80, 0, width=1300, nx=10, ny=11))
    material_surface.append_surface(surface, *rectangle(-80, 3400, 1, width=1500, nx=12, ny=13))
    s.material_surface = surface
    count = len(surface['faces'])
    for name, value in dict(parcel_patch=surface['face_id'], parcel_plate=surface['face_owner'],
                            kind=surface['face_kind'], mass=surface['reference_area_km2'],
                            pos=material_surface.face_centres(surface)).items():
        setattr(s, name, value.copy())
    for name, value in (('parcel_craton', -1), ('relief', 500.), ('suture', 0.)):
        setattr(s, name, np.full(count, value))
    s.material_lineage = {'root_id':s.parcel_patch.copy()}
    s.structure = {'thickness_km':np.full(count, 35.)}
    for name in collision_contacts.PARCEL_FIELDS:
        if hasattr(s, name): delattr(s, name)
    for name in ('collision_contacts', '_collision_overlap_cache'):
        if hasattr(s, name): delattr(s, name)
    collision_contacts.refresh(s)
    return s


def dose(s, obs, geometry, total, step):
    for _ in range(round(total/step)):
        s.t += step
        welding.update(s, obs, geometry, step)


class LocalizedAccretionTests(unittest.TestCase):
    def test_finite_arc_distance_has_independent_endpoint_and_width_limits(self):
        obs = dict(center=np.array([1.,0.,0.]),normal=np.array([0.,1.,0.]),length=200.)
        # Points lie on the equator transverse to the front, or on its own
        # great circle beyond an endpoint. Their exact scalar distances are known.
        a=120./6371.; b=175./6371.
        points=np.array([[math.cos(a),math.sin(a),0.],[math.cos(b),0.,math.sin(b)]])
        np.testing.assert_allclose(welding._distance(points,obs,6371.),[120.,75.],rtol=0,atol=2e-12)
        s=world(width=180.); rows,geo=observations(s)
        component=geo[0]['components'][rows[0]['cp']]
        s.config={'deformation_width_km':100.}
        narrow,_=welding.process_weights(s,rows,component,0)
        s.config={'deformation_width_km':400.}
        wide,_=welding.process_weights(s,rows,component,0)
        self.assertTrue(np.all(wide>=narrow))
        self.assertTrue(np.any(wide>narrow))

    def test_huge_connected_body_does_not_borrow_a_mature_front(self):
        legacy = extended_world()
        candidate = deepcopy(legacy)
        welding.initialize(candidate)
        plan = mature(legacy, steps=100)
        self.assertIsNotNone(plan)
        self.assertGreater(plan['area_km2'], 5e6)
        self.assertIsNone(mature(candidate, steps=100))
        checks = candidate.native_accretion_diagnostics['qualification_checks']
        self.assertTrue(checks)
        self.assertGreater(checks[0]['component_roots'], checks[0]['engaged_roots'])
        self.assertEqual(checks[0]['minimum_process_weight'], 0.)
        self.assertEqual(candidate.process_totals['accreted_km2'], 0.)

    def test_real_small_terrane_transfer_preserves_rock_and_owner_motion_proxy(self):
        s = world(width=180.)
        welding.initialize(s)
        plan = mature(s)
        self.assertIsNotNone(plan)
        self.assertTrue(plan['local_welding_qualification']['eligible'])
        before = {name:getattr(s,name).copy() for name in ('mass','pos','parcel_patch','relief','parcel_plate','trace_xyz')}
        vertices = s.material_surface['vertices'].copy()
        faces = s.material_surface['faces'].copy()
        thickness = s.structure['thickness_km'].copy()
        total_support = s.support.sum(axis=0).copy()
        momentum = np.sum(s.mass[:,None]*s.omega[s.parcel_plate], axis=0)
        self.assertTrue(local.apply_accretion(s, plan))
        for name in ('mass','pos','parcel_patch','relief','trace_xyz'):
            np.testing.assert_array_equal(getattr(s,name), before[name])
        np.testing.assert_array_equal(s.material_surface['vertices'], vertices)
        np.testing.assert_array_equal(s.material_surface['faces'], faces)
        np.testing.assert_array_equal(s.structure['thickness_km'], thickness)
        np.testing.assert_array_equal(s.parcel_plate[16:], before['parcel_plate'][16:])
        np.testing.assert_array_equal(s.support.sum(axis=0), total_support)
        np.testing.assert_allclose(np.sum(s.mass[:,None]*s.omega[s.parcel_plate],axis=0),momentum,rtol=0,atol=1e-10)
        self.assertEqual(s.process_totals['accreted_km2'], before['mass'][:8].sum())
        self.assertFalse(local.apply_accretion(s, plan))

    def test_connected_process_zone_cannot_jump_a_protected_craton_gap(self):
        s = world(width=180.)
        s.parcel_craton[s.parcel_plate == 0] = 17
        welding.initialize(s)
        obs, geo = observations(s)
        source = geo[0]['components'][obs[0]['cp']]
        self.assertEqual(len(source['parcel_indices']), 16)
        weight, data = welding.process_weights(s,obs,source,0)
        self.assertTrue(np.any(weight > 0))
        self.assertTrue(np.all(weight[data['material'] >= 16] == 0))
        dose(s, obs, geo, 100., 2.)
        self.assertFalse(welding.qualify(s,source,0,1)['eligible'])

    def test_disconnected_part_of_same_birth_root_cannot_lend_engagement(self):
        s=world(width=180.)
        s.material_lineage['root_id'][s.parcel_plate==0]=555
        welding.initialize(s); obs,geo=observations(s)
        dose(s,obs,geo,100.,2.)
        source=geo[0]['components'][obs[0]['cp']]
        self.assertEqual(len(source['parcel_indices']),8)
        self.assertFalse(welding.qualify(s,source,0,1)['eligible'])
        use=s.accretion_welding_state['root_id']==555
        np.testing.assert_array_equal(s.accretion_welding_state['active_weight'][use],0.)

    def test_current_opening_or_young_plate_cannot_use_mature_exposure(self):
        for mode in ('opening','young'):
            s=world(width=180.); welding.initialize(s)
            self.assertIsNotNone(mature(s))
            before=s.accretion_welding_state['loading'].copy()
            if mode=='opening': s.omega=-s.omega
            else: s.born[:]=s.t
            s.t+=2.; collision_contacts.refresh(s,2.)
            self.assertEqual(local.plan_accretions(s,2.),[])
            if mode=='opening':
                np.testing.assert_allclose(s.accretion_welding_state['loading'],before*np.exp(-2/35),rtol=2e-14,atol=1e-12)

    def test_age_policy_lets_a_young_plate_dock_once_its_own_contacts_mature(self):
        # A plate born mid-run (rift or back-arc daughter) used to wait 70 Myr
        # whatever its contacts did. Under age policy 1 it waits only as long as
        # its own roots need to mature from zero.
        base = world(width=180.)
        base.born[:] = base.t
        welding.initialize(base)
        legacy, reviewed = deepcopy(base), deepcopy(base)
        reviewed.accretion_age_policy_version = local.AGE_POLICY_VERSION
        self.assertEqual(local.minimum_plate_age_myr(legacy), local.MIN_PLATE_AGE_MYR)
        self.assertEqual(local.minimum_plate_age_myr(reviewed), welding.THRESHOLD)
        self.assertIsNone(mature(legacy, steps=25))
        plan = mature(reviewed, steps=25)
        self.assertIsNotNone(plan)
        self.assertTrue(plan['local_welding_qualification']['eligible'])
        # The docking time is set by root maturation, exactly as for an old plate.
        old = world(width=180.)
        welding.initialize(old)
        start = old.t
        self.assertIsNotNone(mature(old, steps=25))
        self.assertEqual(reviewed.t-base.t, old.t-start)
        self.assertLess(reviewed.t-max(reviewed.born[:2]), local.MIN_PLATE_AGE_MYR)
        # A plate reborn with mature exposure still cannot use it.
        young = deepcopy(base)
        young.accretion_age_policy_version = local.AGE_POLICY_VERSION
        for _ in range(17):
            young.t += 2.; collision_contacts.refresh(young, 2.)
            self.assertEqual(local.plan_accretions(young, 2.), [])
        self.assertGreater(float(young.accretion_welding_state['loading'].max()), welding.THRESHOLD)
        young.born[:] = young.t
        young.t += 2.; collision_contacts.refresh(young, 2.)
        self.assertEqual(local.plan_accretions(young, 2.), [])

    def test_age_policy_requires_root_local_welding(self):
        s = world(width=180.)
        s.accretion_age_policy_version = local.AGE_POLICY_VERSION
        with self.assertRaises(ValueError):
            local.minimum_plate_age_myr(s)
        s.accretion_age_policy_version = True
        with self.assertRaises(ValueError):
            local.minimum_plate_age_myr(s)

    def test_exact_elapsed_dose_decay_and_duplicate_epoch(self):
        base = world(width=180.)
        welding.initialize(base)
        obs, geo = observations(base)
        variants = []
        for dt in (2.,1.,.5):
            s = deepcopy(base)
            dose(s,obs,geo,40.,dt)
            loaded = s.accretion_welding_state['loading'].copy()
            dose(s,[],{},12.,dt)
            np.testing.assert_allclose(s.accretion_welding_state['loading'],loaded*np.exp(-12/35),rtol=2e-14,atol=1e-13)
            before = s.accretion_welding_state['loading'].copy()
            welding.update(s,[],{},dt)
            np.testing.assert_array_equal(s.accretion_welding_state['loading'],before)
            variants.append(s)
        for s in variants[1:]:
            np.testing.assert_allclose(s.accretion_welding_state['loading'], variants[0].accretion_welding_state['loading'],rtol=2e-14,atol=1e-13)

    def test_front_order_duplicates_and_exact_geometric_subdivision(self):
        base = world(width=180.)
        welding.initialize(base)
        obs, geo = observations(base)
        self.assertEqual(len(obs),1)
        original = obs[0]
        center, normal, length = original['center'], original['normal'], original['length']
        tangent = np.cross(normal,center)
        pieces = []
        for offset in (-.25,.25):
            angle = offset*length/6371.
            row = dict(original, length=length/2, center=np.cos(angle)*center+np.sin(angle)*tangent)
            pieces.append(row)
        loads = []
        for rows in (obs,obs+obs,pieces,pieces[::-1]):
            s = deepcopy(base)
            dose(s,rows,geo,40.,2.)
            loads.append(s.accretion_welding_state['loading'])
        for value in loads[1:]: np.testing.assert_allclose(value,loads[0],rtol=2e-13,atol=1e-12)

    def test_rotation_keeps_profiles_and_maturity(self):
        original = world(width=180.)
        rotated = deepcopy(original)
        vector = np.array([.38,-.22,.51])
        rotated.material_surface['vertices'] = rotate(rotated.material_surface['vertices'],vector)
        rotated.pos = rotate(rotated.pos,vector)
        obs, geo = observations(original)
        robs = [dict(row,center=rotate(row['center'][None],vector)[0],normal=rotate(row['normal'][None],vector)[0]) for row in obs]
        for s, rows in ((original,obs),(rotated,robs)):
            welding.initialize(s)
            dose(s,rows,geo,40.,2.)
        np.testing.assert_allclose(original.accretion_welding_state['loading'],rotated.accretion_welding_state['loading'],rtol=2e-12,atol=1e-12)

    def test_refinement_preserves_birth_root_exposure(self):
        base = world(width=180.)
        refined = deepcopy(base)
        old = refined.material_surface
        result = adaptive_material.refine(old['vertices'],old['faces'],refined.parcel_plate,refined.kind,desired_edge_km=80.,face_ids=refined.parcel_patch)
        source = result['source_face']
        self.assertGreater(len(source),len(refined.mass))
        for name in ('parcel_plate','kind','parcel_craton','relief','suture',*collision_contacts.PARCEL_FIELDS):
            setattr(refined,name,getattr(refined,name)[source])
        refined.material_lineage = {'root_id':refined.material_lineage['root_id'][source]}
        refined.material_surface = material_surface.initialize_surface(result['vertices'],result['faces'],refined.parcel_plate,refined.kind,face_id=np.arange(len(source))+1000)
        refined.parcel_patch = refined.material_surface['face_id'].copy()
        refined.mass = refined.material_surface['reference_area_km2'].copy()
        refined.pos = material_surface.face_centres(refined.material_surface)
        refined.structure = {'thickness_km':refined.structure['thickness_km'][source]}
        collision_contacts.refresh(refined)
        for s in (base,refined):
            welding.initialize(s)
            obs,geo = observations(s)
            dose(s,obs,geo,40.,2.)
        for name in welding.TABLE[:3]: np.testing.assert_array_equal(base.accretion_welding_state[name],refined.accretion_welding_state[name])
        np.testing.assert_allclose(base.accretion_welding_state['loading'],refined.accretion_welding_state['loading'],rtol=1e-11,atol=1e-10)

    def test_checkpoint_codec_retains_exact_maturity_continuation(self):
        s = world(width=180.)
        welding.initialize(s)
        obs,geo = observations(s)
        dose(s,obs,geo,12.,2.)
        restored = deepcopy(s)
        arrays = {}
        tree = checkpoint._encode(s.accretion_welding_state,arrays)
        storage = io.BytesIO()
        np.savez(storage,**arrays); storage.seek(0)
        with np.load(storage,allow_pickle=False) as archive:
            restored.accretion_welding_state = checkpoint._decode(json.loads(json.dumps(tree)),archive)
        for x in (s,restored): dose(x,obs,geo,8.,2.)
        for name in welding.TABLE: np.testing.assert_array_equal(s.accretion_welding_state[name],restored.accretion_welding_state[name])

    def test_stale_plan_cannot_reassign_material(self):
        for change in ('geometry','time','owner'):
            s = world(width=180.); welding.initialize(s); plan = mature(s)
            self.assertIsNotNone(plan)
            if change == 'geometry': s.material_surface['vertices'] = rotate(s.material_surface['vertices'],[0,0,.001])
            elif change == 'time': s.t += .1
            else: s.parcel_plate[0] = 2
            before = s.parcel_plate.copy()
            self.assertFalse(local.apply_accretion(s,plan))
            np.testing.assert_array_equal(s.parcel_plate,before)

    def test_optional_saved_schema_rejects_stripped_or_malformed_policy(self):
        s = world(); welding.initialize(s); obs,geo = observations(s); dose(s,obs,geo,2.,2.)
        frame = dict(welding.snapshot_fields(s),time_myr=s.t)
        welding.validate_frame(frame); welding.validate_frame({})
        mutants = []
        row = deepcopy(frame); del row['native_accretion_version']; mutants.append(row)
        row = deepcopy(frame); row['accretion_welding_loading'][0] = np.nan; mutants.append(row)
        row = deepcopy(frame); row['accretion_welding_root_id'] = np.array(['bad']); mutants.append(row)
        row = deepcopy(frame); row['accretion_welding_active_weight'] = row['accretion_welding_active_weight'][:-1]; mutants.append(row)
        row = deepcopy(frame); row['native_accretion_welding_time_myr'] = s.t+1; mutants.append(row)
        row = deepcopy(frame); row['accretion_welding_source_uid'] = row['accretion_welding_target_uid'].copy(); mutants.append(row)
        for row in mutants:
            with self.assertRaises(ValueError): welding.validate_frame(row)


if __name__ == '__main__': unittest.main(verbosity=2)
