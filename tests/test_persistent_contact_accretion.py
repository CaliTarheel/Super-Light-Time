"""Accretion admits actual hidden material contacts without remote clock transfer."""
from copy import deepcopy
from types import MethodType, SimpleNamespace
import json
import unittest

import numpy as np

import collision_contacts
import adaptive_material
import local_accretion as local
import material_surface
import mesh_geometry
from ridge_geometry import rotate
from tests.test_collision_contacts import rectangle
from tests.test_native_local_accretion import world as visible_world


def world(width=120.):
    mesh = mesh_geometry.icosphere(2)
    surface = material_surface.initialize_surface(*rectangle(-200., 80., 0, width=width, nx=3, ny=3))
    material_surface.append_surface(surface, *rectangle(-80., 300., 1, width=width, nx=3, ny=3))
    remote = list(rectangle(-200., 80., 0, width=width, nx=3, ny=3))
    remote[0] = rotate(remote[0], [0., 0., .7])
    material_surface.append_surface(surface, *remote)
    count, n = len(surface['faces']), len(mesh['faces'])
    owner, kind = surface['face_owner'].copy(), surface['face_kind'].copy()
    pos, patches = material_surface.face_centres(surface), surface['face_id'].copy()
    locator = mesh_geometry.build_locator(mesh['vertices'], mesh['faces'])
    def indices(points):
        return mesh_geometry.locate_points(points, locator)[0]
    simulation = SimpleNamespace(native_mesh=mesh, material_surface=surface, n=n, xyz=mesh['xyz'].copy(),
        cell_area=mesh['area_km2'].copy(), pos=pos, parcel_patch=patches,
        parcel_plate=owner, mass=surface['reference_area_km2'].copy(), kind=kind,
        parcel_craton=np.full(count, -1, int), material_lineage={'root_id':patches.copy()},
        relief=np.full(count, 500.), suture=np.zeros(count),
        trace_patch=patches.copy(), trace_plate=owner.copy(), trace_xyz=pos.copy(),
        trace_kind=kind.copy(), trace_origin_kind=kind.copy(), trace_relief_m=np.full(count, 500.),
        trace_suture=np.zeros(count), trace_adjustment_m=np.zeros(count),
        structure={'thickness_km':np.full(count, 35.)}, trace_structure={'thickness_km':np.full(count, 35.)},
        plate=np.full(n, 2), crust=np.zeros(n, np.uint8), _indices=indices,
        active=np.ones(3, bool), plate_uid=np.array([41, 52, 63]), born=np.zeros(3), t=100.,
        omega=np.array([[0.,0.,.003], [0.,0.,-.003], [0.,0.,0.]]),
        mantle=np.zeros((3, 3)), names=['Incoming', 'Receiver', 'Ocean'],
        events=[], ridge_episodes=[], backarc_basins=[], process_totals={'accreted_km2':0.},
        ba=np.empty(0, int), bb=np.empty(0, int), bp=np.empty(0, int), bq=np.empty(0, int),
        bmid=np.empty((0,3)), bn=np.empty((0,3)), bcode=np.empty(0, np.uint8), bl=np.empty(0))
    simulation.support=np.zeros((3,n)); simulation.support[2]=1.
    def record(self, kind, description, key=None, *, plates=(), xyz=None, details=None):
        self.events.append(dict(type=kind, details=deepcopy(details or {})))
    simulation._record = MethodType(record, simulation)
    collision_contacts.refresh(simulation)
    hits = material_surface.sample_surface(surface, simulation.xyz)
    simulation.plate[hits['query_index']] = owner[hits['face_index']]
    simulation.crust[hits['query_index']] = kind[hits['face_index']]
    simulation.support[:] = 0.
    simulation.support[simulation.plate, np.arange(n)] = 1.
    return simulation


def mature(simulation, steps=70):
    for _ in range(steps):
        simulation.t += 2.
        collision_contacts.refresh(simulation, 2.)
        plans = local.plan_accretions(simulation, 2.)
        if plans:
            return plans[0]
    return None


class PersistentContactAccretionTests(unittest.TestCase):
    def test_hidden_subcontrol_contact_builds_local_loading_and_matures(self):
        simulation = world(width=180.)
        hits = material_surface.sample_surface(simulation.material_surface, simulation.xyz)
        self.assertEqual(np.count_nonzero(hits['face_index'] < 16), 0)
        self.assertEqual(len(simulation.bcode), 0)
        self.assertEqual(len(simulation.collision_contacts), 1)
        self.assertGreater(simulation.collision_contacts[0]['overlap_area_km2'], 30000.)
        self.assertLess(simulation.collision_contacts[0]['normal_speed_km_myr'], -2.)
        plan = mature(simulation)
        self.assertIsNotNone(plan)
        self.assertEqual((plan['source'], plan['target']), (0, 1))
        np.testing.assert_array_equal(plan['parcel_indices'], np.arange(8))
        self.assertEqual(len(plan['surface_cells']), 0)
        self.assertEqual(plan['collision_contact_id'], simulation.collision_contacts[0]['id'])

    def test_age_loading_and_current_convergence_are_all_required(self):
        simulation = world(width=180.)
        simulation.born[:] = simulation.t
        self.assertIsNone(mature(simulation, steps=25))
        self.assertGreater(simulation.local_accretion_contacts[0]['loading'], local.THRESHOLD)
        simulation.omega[:] = 0.
        self.assertIsNone(mature(simulation, steps=35))
        self.assertGreater(simulation.t-max(simulation.born[:2]), local.MIN_PLATE_AGE_MYR)
        self.assertEqual(simulation.process_totals['accreted_km2'], 0.)

    def test_separation_and_shear_do_not_borrow_persistent_contact_age(self):
        for mode in ('separating', 'shearing', 'inactive', 'geometric_gap'):
            simulation = world()
            mature(simulation, steps=4)
            loading = simulation.local_accretion_contacts[0]['loading']
            if mode == 'separating': simulation.omega = -simulation.omega
            elif mode == 'shearing': simulation.omega[1, 1] = .1
            elif mode == 'inactive': simulation.active[1] = False
            else:
                selected = simulation.material_surface['vertex_owner'] == 1
                simulation.material_surface['vertices'][selected] = rotate(
                    simulation.material_surface['vertices'][selected], [0.,0.,.3])
                material_surface.refresh_geometry(simulation.material_surface)
                simulation.pos = material_surface.face_centres(simulation.material_surface)
            with self.subTest(mode=mode):
                self.assertIsNone(mature(simulation, steps=15))
                self.assertLess(simulation.local_accretion_contacts[0]['loading'], loading)

    def test_projected_duplicate_cannot_inflate_exact_contact_loading(self):
        original, projected = world(), world()
        # Resolve the same two sheets on a finer actual control mesh, producing
        # real visible collision edges in addition to the unchanged overlaps.
        mesh = mesh_geometry.icosphere(5)
        projected.native_mesh, projected.xyz, projected.n = mesh, mesh['xyz'], len(mesh['faces'])
        projected.cell_area = mesh['area_km2'].copy()
        locator = mesh_geometry.build_locator(mesh['vertices'],mesh['faces'])
        projected._indices = lambda points: mesh_geometry.locate_points(points,locator)[0]
        projected.plate = np.full(projected.n,2)
        projected.crust = np.zeros(projected.n,np.uint8)
        hits = material_surface.sample_surface(projected.material_surface,projected.xyz)
        eligible = collision_contacts.eligible_hits(hits['query_index'],hits['face_index'],
            projected.parcel_collision_sheet,projected.collision_contacts)
        cells, faces = hits['query_index'][eligible], hits['face_index'][eligible]
        projected.plate[cells], projected.crust[cells] = projected.parcel_plate[faces], projected.kind[faces]
        a = np.flatnonzero(projected.plate == 0)
        p, edge = np.nonzero(projected.plate[mesh['face_neighbors'][a]] == 1)
        projected.ba, projected.bb = a[p], mesh['face_neighbors'][a[p],edge]
        self.assertGreater(len(projected.ba),0)
        projected.bp, projected.bq = np.zeros(len(p),int), np.ones(len(p),int)
        middle = projected.xyz[projected.ba]+projected.xyz[projected.bb]
        middle /= np.linalg.norm(middle,axis=1,keepdims=True)
        normal = projected.xyz[projected.bb]-projected.xyz[projected.ba]
        normal -= np.sum(normal*middle,axis=1)[:,None]*middle
        normal /= np.linalg.norm(normal,axis=1,keepdims=True)
        projected.bmid, projected.bn = middle, normal
        projected.bcode, projected.bl = np.full(len(p),4,np.uint8), np.full(len(p),1.e9)
        for simulation in (original, projected): mature(simulation, steps=4)
        self.assertEqual(projected.local_accretion_contacts, original.local_accretion_contacts)
        self.assertEqual(len(projected.local_accretion_contacts), 1)

    def test_distant_valid_touching_front_keeps_independent_loading(self):
        simulation = visible_world()
        before = len(simulation.mass)
        for lower, upper, owner in ((-200.,80.,0), (-80.,300.,1)):
            material_surface.append_surface(simulation.material_surface,
                *rectangle(lower,upper,owner,width=120.,nx=3,ny=3))
        surface = simulation.material_surface
        extra = len(surface['faces'])-before
        simulation.pos = material_surface.face_centres(surface)
        simulation.mass = surface['reference_area_km2'].copy()
        simulation.parcel_patch = surface['face_id'].copy()
        simulation.parcel_plate = surface['face_owner'].copy()
        simulation.kind = surface['face_kind'].copy()
        for name, value in (('parcel_craton',-1), ('relief',500.), ('suture',0.)):
            setattr(simulation,name,np.r_[getattr(simulation,name),np.full(extra,value)])
        simulation.structure['thickness_km'] = np.r_[simulation.structure['thickness_km'],np.full(extra,35.)]
        simulation.material_lineage = {'root_id':simulation.parcel_patch.copy()}
        collision_contacts.refresh(simulation)
        self.assertEqual(len(simulation.collision_contacts),1)
        self.assertIsNone(mature(simulation,steps=1))
        records = simulation.local_accretion_contacts
        self.assertEqual(len(records),2)
        exact = next(row for row in records if 'collision_contact_id' in row)
        visible = next(row for row in records if 'collision_contact_id' not in row)
        self.assertGreater(exact['loading'],0.)
        self.assertEqual(visible['loading'],4.)
        self.assertTrue(set(visible['p_patches']) & simulation.source_patches[0])
        self.assertFalse(set(visible['p_patches']) & simulation.source_patches[1])
        plan = mature(simulation)
        self.assertEqual(set(plan['patch_ids']),simulation.source_patches[0])
        self.assertNotEqual(plan['contact_id'],exact['id'])

    def test_same_epoch_calls_do_not_charge_or_decay_exact_loading_twice(self):
        simulation = world()
        mature(simulation, steps=3)
        before = deepcopy(simulation.local_accretion_contacts)
        self.assertEqual(local.plan_accretions(simulation, 2.), [])
        self.assertEqual(simulation.local_accretion_contacts, before)
        simulation.omega[:] = 0.
        mature(simulation, steps=1)
        after = deepcopy(simulation.local_accretion_contacts)
        self.assertEqual(local.plan_accretions(simulation, 2.), [])
        self.assertEqual(simulation.local_accretion_contacts, after)

    def test_refinement_keeps_local_contact_clock_through_birth_root_identity(self):
        simulation = world()
        mature(simulation, steps=4)
        baseline = deepcopy(simulation)
        identity = simulation.local_accretion_contacts[0]['id']
        old = simulation.material_surface
        result = adaptive_material.refine(old['vertices'], old['faces'], simulation.parcel_plate,
            simulation.kind, desired_edge_km=80., face_ids=simulation.parcel_patch)
        source = result['source_face']
        self.assertGreater(len(source), len(simulation.mass))
        for name in ('parcel_plate','kind','parcel_craton','relief','suture', *collision_contacts.PARCEL_FIELDS):
            setattr(simulation, name, getattr(simulation, name)[source])
        simulation.material_lineage = {'root_id':simulation.material_lineage['root_id'][source]}
        simulation.material_surface = material_surface.initialize_surface(result['vertices'], result['faces'],
            simulation.parcel_plate, simulation.kind, face_id=np.arange(len(source))+1000)
        simulation.parcel_patch = simulation.material_surface['face_id'].copy()
        simulation.mass = simulation.material_surface['reference_area_km2'].copy()
        simulation.pos = material_surface.face_centres(simulation.material_surface)
        simulation.structure = {'thickness_km':simulation.structure['thickness_km'][source]}
        for world_state in (simulation, baseline): mature(world_state, steps=1)
        self.assertEqual(simulation.local_accretion_contacts[0]['id'], identity)
        self.assertEqual(len(simulation.local_accretion_contacts), 1)
        self.assertAlmostEqual(simulation.local_accretion_contacts[0]['loading'],
            baseline.local_accretion_contacts[0]['loading'], places=10)
        self.assertEqual(simulation.local_accretion_contacts[0]['collision_contact_id'],
            baseline.local_accretion_contacts[0]['collision_contact_id'])

    def test_cross_owner_craton_is_rejected_and_real_transfer_preserves_remote_rock(self):
        malformed = world(width=180.)
        malformed.parcel_craton[:9] = 17
        self.assertIsNone(mature(malformed))
        simulation = world(width=180.)
        plan = mature(simulation)
        original = {name:getattr(simulation,name).copy() for name in ('mass','pos','parcel_patch','relief','parcel_plate')}
        vertices = simulation.material_surface['vertices'].copy()
        columns = simulation.structure['thickness_km'].copy()
        self.assertTrue(local.apply_accretion(simulation, plan))
        for name in ('mass','pos','parcel_patch','relief'):
            np.testing.assert_array_equal(getattr(simulation,name), original[name])
        np.testing.assert_array_equal(simulation.parcel_plate[16:], original['parcel_plate'][16:])
        np.testing.assert_array_equal(simulation.material_surface['vertices'], vertices)
        np.testing.assert_array_equal(simulation.structure['thickness_km'], columns)
        np.testing.assert_array_equal(simulation.parcel_plate[:8], 1)
        self.assertFalse(local.apply_accretion(simulation, plan))
        self.assertEqual(simulation.process_totals['accreted_km2'], original['mass'][:8].sum())
        self.assertEqual(simulation.events[-1]['details']['collision_contact_id'],plan['collision_contact_id'])

    def test_serialized_shared_contact_history_keeps_the_next_local_loading(self):
        original = world()
        mature(original, steps=4)
        restored = deepcopy(original)
        restored.local_accretion_contacts = json.loads(json.dumps(original.local_accretion_contacts))
        restored.collision_contacts = json.loads(json.dumps(original.collision_contacts))
        for simulation in (original, restored): mature(simulation, steps=1)
        self.assertEqual(original.local_accretion_contacts, restored.local_accretion_contacts)
        self.assertEqual(original.collision_contacts, restored.collision_contacts)
        self.assertEqual(original.collision_contacts[0]['local_accretion_contact_ids'],
            [original.local_accretion_contacts[0]['id']])

    def test_matching_broad_roots_do_not_lend_a_distant_front_their_maturity(self):
        simulation = world()
        mature(simulation,steps=3)
        previous = simulation.local_accretion_contacts[0]
        identity = previous['id']
        # Root ancestry and leaf membership alone are deliberately identical.
        # The prior clipped footprint belongs to another distant local front.
        previous['center'] = [0.,1.,0.]
        previous['contact_radius_km'] = 10.
        previous['loading'] = local.THRESHOLD+5.
        self.assertIsNone(mature(simulation,steps=1))
        self.assertEqual(len(simulation.local_accretion_contacts),2)
        current = simulation.local_accretion_contacts[-1]
        self.assertNotEqual(current['id'],identity)
        self.assertEqual(current['collision_contact_id'],previous['collision_contact_id'])
        self.assertEqual(current['p_roots'],previous['p_roots'])
        self.assertEqual(current['q_roots'],previous['q_roots'])
        self.assertLess(current['loading'],2.)

    def test_disjoint_fronts_sharing_coarse_face_and_roots_keep_local_clocks(self):
        from tests.test_collision_local_fronts import fixture
        simulation = world()
        simulation.__dict__.update(vars(fixture(coarse=True)))
        simulation.t = 100.
        simulation.parcel_craton = np.full(len(simulation.mass),-1)
        # Deliberately broad ancestry on each connected sheet; the two actual
        # clipped patches still have distinct physical footprints.
        simulation.material_lineage = {'root_id':simulation.parcel_plate.copy()}
        self.assertIsNone(mature(simulation,steps=1))
        self.assertEqual(len(simulation.local_accretion_contacts),1)
        previous = simulation.local_accretion_contacts[0]
        self.assertLess(previous['center'][1],-.1)
        previous['loading'] = local.THRESHOLD+5.
        simulation.omega = -simulation.omega
        self.assertIsNone(mature(simulation,steps=1))
        self.assertEqual(len(simulation.local_accretion_contacts),2)
        current = simulation.local_accretion_contacts[-1]
        self.assertGreater(current['center'][1],.1)
        self.assertEqual(current['collision_contact_id'],previous['collision_contact_id'])
        self.assertEqual(current['p_roots'],previous['p_roots'])
        self.assertEqual(current['q_roots'],previous['q_roots'])
        self.assertLess(current['loading'],2.)


if __name__ == '__main__':
    unittest.main()
