"""Adversarial contact, provenance, rock-budget and saved-surface contracts."""
from copy import deepcopy
from types import SimpleNamespace
from unittest.mock import patch
import tempfile
from pathlib import Path
import unittest
import numpy as np

import collision_contacts as contacts
import material_surface
import mesh_geometry
import native_frame_sampling as sampling
import native_material_evolution as evolution
import structure_engine
import crustal_structure as columns
from ridge_geometry import rotate
from orientation import orient_frame, rotation_matrix
from checkpoint import write_checkpoint, read_checkpoint
from tests.test_native_gospl_sampling import frame as ocean_frame

RADIUS = 6371.


def unit(value):
    value = np.asarray(value, float)
    return value/np.linalg.norm(value, axis=-1, keepdims=True)


def rectangle(left, right, owner, *, width=400., nx=7, ny=5, kind=1):
    x, y = np.meshgrid(np.linspace(left, right, nx), np.linspace(-width, width, ny))
    vertices = unit(np.column_stack((np.ones(x.size), x.ravel()/RADIUS, y.ravel()/RADIUS)))
    faces = []
    for row in range(ny-1):
        for col in range(nx-1):
            a = row*nx+col
            faces.extend(((a, a+1, a+1+nx), (a, a+1+nx, a+nx)))
    faces = np.array(faces)
    return vertices, faces, np.full(len(faces), owner), np.full(len(faces), kind)


def fixture(*, separated=False, arc=False):
    first = rectangle(-900., 200., 0)
    mesh = material_surface.initialize_surface(*first)
    second = rectangle(350. if separated else -200., 900., 1, kind=3 if arc else 1)
    material_surface.append_surface(mesh, *second)
    n = len(mesh['faces'])
    s = SimpleNamespace(material_surface=mesh, parcel_patch=mesh['face_id'].copy(),
        parcel_plate=mesh['face_owner'].copy(), kind=mesh['face_kind'].copy(),
        mass=mesh['reference_area_km2'].copy(), pos=material_surface.face_centres(mesh),
        relief=np.where(mesh['face_owner'] == 0, 780., 480.), suture=np.zeros(n),
        plate_uid=np.array([101, 202], np.int64), t=0.,
        omega=np.array([[0., 0., .003], [0., 0., -.003]]),
        config=dict(deforming_regions=1, deformation_width_km=450.),
        bmid=np.empty((0, 3)), bn=np.empty((0, 3)), bl=np.empty(0), bp=np.empty(0, int), bq=np.empty(0, int),
        trace_xyz=np.empty((0, 3)), trace_patch=np.empty(0, np.int64),
        trace_rift_tangent=np.empty((0, 3)), rift_tangent=np.tile([0., 0., 1.], (n, 1)))
    structure_engine.initialize_parcels(s)
    contacts.refresh(s)
    return s


def saved(s, *, margins=False):
    f = ocean_frame(centers=[], time=s.t)
    n = len(s.mass)
    f.update(material_vertices=s.material_surface['vertices'].copy(), material_faces=s.material_surface['faces'].copy(),
        material_face_id=s.parcel_patch.copy(), material_owner=s.parcel_plate.copy(), material_kind=s.kind.copy(),
        material_height_m=structure_engine.material_height(s.kind, s.relief),
        material_erosion_rate_m_myr=np.zeros(n), surface_reconstruction_version=1,
        arc_material_version=1, arc_surface_version=2,
        material_arc_id=np.zeros(n, np.int64), material_arc_basal_m=np.zeros(n),
        **contacts.snapshot_fields(s))
    if margins:
        f.update(continental_margin_version=1, continental_margin_parameters=dict(width_km=150., shelf_depth_m=180.))
    return f


class CollisionContactsTests(unittest.TestCase):
    def test_overlapping_continents_retain_every_face_and_volume(self):
        s = fixture()
        self.assertEqual(len(s.collision_contacts), 1)
        old = deepcopy(s)
        s.t = 2.; contacts.refresh(s, 2.)
        np.testing.assert_array_equal(s.mass, old.mass)
        np.testing.assert_array_equal(s.parcel_patch, old.parcel_patch)
        np.testing.assert_array_equal(s.structure['thickness_km'], old.structure['thickness_km'])
        self.assertGreater(s.collision_diagnostics['pair_overlap_area_km2'], 1000.)
        self.assertGreater(s.collision_diagnostics['buried_area_quadrature_km2'], 1000.)
        self.assertGreater(s.parcel_burial_myr.max(), 0.)
        self.assertGreater(s.parcel_collision_suture.max(), 0.)
        # Refreshing a cached epoch cannot charge the same burial time twice.
        age = s.parcel_burial_myr.copy(); contacts.refresh(s, 2.)
        np.testing.assert_array_equal(s.parcel_burial_myr, age)

    def test_arc_continent_first_contact_polarity_persists_after_height_crossing(self):
        s = fixture(arc=True)
        relation = deepcopy(s.collision_contacts[0])
        s.relief[s.parcel_plate == 1] = 4000.
        s.t = 2.; contacts.refresh(s, 2.)
        self.assertEqual(s.collision_contacts[0]['top_sheet'], relation['top_sheet'])
        self.assertEqual(s.collision_contacts[0]['under_sheet'], relation['under_sheet'])
        result = sampling.sample_frame(saved(s), np.array([[1., 0., 0.]]))
        self.assertEqual(int(s.parcel_plate[result['material_face'][0]]), 0)
        self.assertGreater(result['elevation'][0], 3000.)
        self.assertGreater(result['collision_surface_offset_m'][0], 2000.)
        selected_height = result['elevation'][0]-result['collision_surface_offset_m'][0]
        self.assertAlmostEqual(selected_height, 1000., places=7)
        self.assertTrue(np.all(s.structure['thickness_km'] <= columns.MAX_THICKNESS_KM))

    def test_real_gap_never_has_contact_or_blanket_hidden_sheet(self):
        s = fixture(separated=True)
        self.assertEqual(s.collision_contacts, [])
        np.testing.assert_array_equal(s.parcel_exposed_fraction, 1.)
        point = unit([1., 275./RADIUS, 0.])[None]
        result = sampling.sample_frame(saved(s), point)
        self.assertEqual(result['material_face'][0], -1)
        self.assertEqual(result['crust'][0], 0)

    def test_separation_exposes_formerly_buried_material_and_retains_suture(self):
        s = fixture(); s.t = 2.; contacts.refresh(s, 2.)
        sutures = s.parcel_collision_suture.copy()
        selected = s.material_surface['vertex_owner'] == 1
        s.material_surface['vertices'][selected] = rotate(s.material_surface['vertices'][selected], [0., 0., .3])
        material_surface.refresh_geometry(s.material_surface)
        s.pos = material_surface.face_centres(s.material_surface)
        s.t = 4.; s.omega[:] = 0.; contacts.refresh(s, 2.)
        np.testing.assert_array_equal(s.parcel_exposed_fraction, 1.)
        np.testing.assert_array_equal(s.parcel_collision_suture, sutures)
        self.assertEqual(s.collision_contacts[0]['state'], 'quiet')

    def test_retired_motion_adjustment_preserves_velocities_and_reports_contacts(self):
        s = fixture()
        area = np.bincount(s.parcel_plate, weights=s.material_surface['area_km2'])
        before = s.omega.copy(); momentum = area@before
        energy = float(np.sum(area[:, None]*before**2))
        contacts.resist_motion(s, 2.)
        np.testing.assert_allclose(area@s.omega, momentum, rtol=0, atol=1e-12)
        np.testing.assert_array_equal(s.omega,before)
        self.assertEqual(float(np.sum(area[:, None]*s.omega**2)), energy)
        self.assertGreater(s.collision_resistance_diagnostics['input_active_contacts'],0)
        self.assertEqual(s.collision_resistance_diagnostics['rotational_energy_proxy_removed'],0.)
        s.omega = -before; opening = s.omega.copy(); contacts.resist_motion(s, 2.)
        np.testing.assert_array_equal(s.omega, opening)
        s.omega[:] = [.003, -.002, .005]; common = s.omega.copy(); contacts.resist_motion(s, 2.)
        np.testing.assert_array_equal(s.omega, common)

    def test_subcell_contact_front_drives_real_bounded_area_shortening_once(self):
        s = fixture()
        original = s.material_surface['area_km2'].copy()
        thickness = s.structure['thickness_km'].copy()
        # There is no native control front in this fixture. Actual material
        # intersection must supply the admitted loading itself.
        evolution.advect(s, 2.)
        area = s.material_surface['area_km2']
        self.assertLess(float(area.sum()), float(original.sum()))
        changed, _ = columns.evolve_structure(s.structure, 2., geometric_log_area=s.geometric_log_area)
        np.testing.assert_allclose(area*changed['thickness_km'], original*thickness, rtol=2e-12)
        self.assertTrue(np.all(changed['thickness_km'] <= columns.MAX_THICKNESS_KM))
        self.assertAlmostEqual(s.material_geometry_volume_residual_km3, 0., places=5)

    def test_pole_and_dateline_rotation_preserve_contact_area_polarity_and_sampling(self):
        s = fixture(); source = saved(s)
        points = unit(np.array([[1., 0., 0.], [1., -.07, 0.], [1., .07, 0.], [1., 0., .2]]))
        original = sampling.sample_frame(source, points)
        for angles in (dict(yaw=180.), dict(pitch=90.), dict(yaw=47., pitch=-63., roll=18.)):
            matrix = rotation_matrix(angles); turned = deepcopy(s)
            turned.material_surface['vertices'] = s.material_surface['vertices']@matrix
            turned.pos = s.pos@matrix; turned.omega = s.omega@matrix
            material_surface.refresh_geometry(turned.material_surface)
            contacts.refresh(turned)
            self.assertAlmostEqual(turned.collision_diagnostics['pair_overlap_area_km2']/s.collision_diagnostics['pair_overlap_area_km2'], 1., places=10)
            self.assertEqual(turned.collision_contacts[0]['top_sheet'], s.collision_contacts[0]['top_sheet'])
            oriented = orient_frame(source, angles)
            result = sampling.sample_frame(oriented, points@matrix)
            np.testing.assert_array_equal(result['material_face'], original['material_face'])
            np.testing.assert_allclose(result['elevation'], original['elevation'], atol=1e-7)
            np.testing.assert_allclose(oriented['collision_contacts'][0]['center'], np.array(source['collision_contacts'][0]['center'])@matrix, atol=1e-14)

    def test_persistent_identity_does_not_introduce_a_height_jump_at_overlap_edge(self):
        s = fixture(); s.relief[s.parcel_plate == 1] = 1780.
        f = saved(s, margins=True)
        differences = []
        for distance in (.01, .001):
            points = unit(np.array([[1., (200.-distance)/RADIUS, 0.], [1., (200.+distance)/RADIUS, 0.]]))
            value = sampling.sample_frame(f, points)
            differences.append(abs(np.diff(value['elevation'])[0]))
        self.assertLess(differences[1], max(1e-8, differences[0]*.11))

    def test_cycle_and_unversioned_collision_data_are_rejected(self):
        s = fixture(); f = saved(s)
        del f['collision_contact_version']
        with self.assertRaises(ValueError): sampling.prepare(f)
        f = saved(s)
        f['collision_contacts'] = [dict(top_sheet=1, under_sheet=2), dict(top_sheet=2, under_sheet=3), dict(top_sheet=3, under_sheet=1)]
        with self.assertRaises(ValueError): sampling.prepare(f)

    def test_remapping_refinement_preserves_sheet_burial_and_suture(self):
        from tests.test_native_material_adaptivity import world
        from native_material_adaptivity import adapt
        s = world(); contacts.ensure_fields(s)
        s.parcel_burial_myr[:] = 12.; s.parcel_collision_suture[:] = .6
        s.parcel_exposed_fraction[:] = .75
        before = s.mass.sum(); volume = float(s.material_surface['area_km2']@s.structure['thickness_km'])
        sheet = np.unique(s.parcel_collision_sheet)
        self.assertTrue(adapt(s, 2.)); contacts.ensure_fields(s)
        np.testing.assert_array_equal(np.unique(s.parcel_collision_sheet), sheet)
        np.testing.assert_allclose(s.parcel_burial_myr, 12.)
        np.testing.assert_allclose(s.parcel_collision_suture, .6)
        np.testing.assert_allclose(s.parcel_exposed_fraction, .75)
        self.assertAlmostEqual(s.mass.sum()/before, 1., places=12)
        self.assertAlmostEqual(float(s.material_surface['area_km2']@s.structure['thickness_km'])/volume, 1., places=12)

    def test_explicit_owner_split_inherits_underthrust_relationship(self):
        s = fixture(); original = deepcopy(s.collision_contacts[0])
        s.parcel_plate[(s.parcel_plate == 1) & (s.pos[:, 2] > 0)] = 2
        contacts.ensure_fields(s)
        self.assertEqual(len(np.unique(s.parcel_collision_sheet)), 3)
        self.assertEqual(len(s.collision_contacts), 2)
        self.assertEqual({r['top_sheet'] for r in s.collision_contacts}, {original['top_sheet']})
        self.assertEqual(len({r['under_sheet'] for r in s.collision_contacts}), 2)

    def test_arc_birth_before_refinement_preserves_every_original_sheet_identity(self):
        from native_engine import Simulation
        import native_arc_material
        from native_material_adaptivity import adapt
        s = Simulation(dict(mesh_level=2, coast_geometry_level=2, adaptive_refinement=1,
                            width=64, height=32, plate_count=4, seed=37))
        # Historical tiny-patch birth/refinement lifecycle fixture; physical
        # profile-one source formation and native evolution have separate tests.
        s.native_arc_birth_profile_version = 0
        initial_sheets = s.parcel_collision_sheet.copy()
        original_ids = s.parcel_patch.copy()
        cell = int(np.flatnonzero(s.crust == 0)[0])
        native_arc_material.add_arc_crust(s, np.array([cell]), np.array([1000.]))
        self.assertEqual(len(s.parcel_collision_sheet), len(s.mass))
        np.testing.assert_array_equal(s.parcel_collision_sheet[:len(initial_sheets)], initial_sheets)
        self.assertEqual(len(np.unique(s.parcel_collision_sheet)), len(np.unique(initial_sheets))+1)
        expected = dict(zip(s.parcel_patch.tolist(), s.parcel_collision_sheet.tolist()))
        s.material_deformation = dict(face_weight=np.ones(len(s.mass)), face_rigid=np.zeros(len(s.mass), bool),
                                     face_strain=np.zeros(len(s.mass)))
        s.t = 2.
        self.assertTrue(adapt(s, 2.))
        expected_sheets = np.array([expected[int(root)] for root in s.material_lineage['root_id']])
        np.testing.assert_array_equal(s.parcel_collision_sheet, expected_sheets)
        contacts.ensure_fields(s)
        np.testing.assert_array_equal(s.parcel_collision_sheet, expected_sheets)
        self.assertEqual(len(np.unique(s.parcel_collision_sheet)), len(np.unique(initial_sheets))+1)

    def test_adaptation_refuses_unaligned_collision_arrays_before_any_mutation(self):
        from tests.test_native_material_adaptivity import world
        from native_material_adaptivity import adapt
        s = world(); contacts.ensure_fields(s)
        before = s.material_surface['vertices'].copy()
        s.parcel_collision_sheet = s.parcel_collision_sheet[:-1]
        with self.assertRaisesRegex(ValueError, 'align.*before adaptation'):
            adapt(s, 2.)
        np.testing.assert_array_equal(s.material_surface['vertices'], before)

    def test_buried_denudation_is_zero_while_exposed_rock_keeps_existing_erosion(self):
        from tests.test_structure_engine import world
        buried = world(erosion=1.); exposed = deepcopy(buried)
        buried.parcel_exposed_fraction = np.zeros(len(buried.mass))
        before = buried.structure['denudation_m'].copy()
        zero = np.zeros(buried.n)
        with patch.object(contacts, 'erosion_fraction', side_effect=lambda s, trace=False:
                np.zeros(len(s.trace_patch) if trace else len(s.mass))):
            structure_engine.deform(buried, zero, zero, zero, 2.)
        structure_engine.deform(exposed, zero, zero, zero, 2.)
        np.testing.assert_array_equal(buried.structure['denudation_m'], before)
        self.assertGreater(exposed.structure['denudation_m'].sum(), before.sum())
        np.testing.assert_array_equal(buried.trace_structure['denudation_m'], 0.)

    def test_real_checkpoint_preserves_contacts_and_identical_next_step(self):
        from native_engine import Simulation
        s = Simulation(dict(mesh_level=2, coast_geometry_level=2, adaptive_refinement=0,
                            width=64, height=32, plate_count=4, seed=37))
        s.step(2.)
        self.assertTrue(s.collision_contacts)
        compatibility = dict(engine_sha256='collision-test', auxiliary_sources_sha256={}, numpy_version=np.__version__)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)/'checkpoint.npz'
            write_checkpoint(path, s, dict(config=s.config), compatibility)
            restored, _ = read_checkpoint(path, compatibility, Simulation)
        self.assertEqual(restored.collision_contacts, s.collision_contacts)
        s.step(2.); restored.step(2.)
        for name in ('mass', 'parcel_patch', 'parcel_collision_sheet', 'parcel_burial_myr',
                     'parcel_exposed_fraction', 'parcel_collision_suture', 'omega'):
            np.testing.assert_array_equal(getattr(restored, name), getattr(s, name), err_msg=name)
        np.testing.assert_array_equal(restored.material_surface['vertices'], s.material_surface['vertices'])
        self.assertEqual(restored.collision_contacts, s.collision_contacts)
        self.assertLess(abs(s.material_column_budget['residual_km3']),
                        s.material_column_budget['before_motion_volume_km3']*1e-10)


if __name__ == '__main__': unittest.main()
