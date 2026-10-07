"""First-contact decisions must depend on overlapping rock, not triangulation."""
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest

import numpy as np

import collision_contacts as contacts
import material_surface
import structure_engine
from checkpoint import read_checkpoint, write_checkpoint
from orientation import rotation_matrix


def fixture(refine_a=False, refine_b=False, *, height_b=60., orientation=None, reverse=False):
    """Two coincident patches with identical piecewise-constant height fields.

    Refinement splits the high half four ways and conformingly bisects the
    neighboring low half. Great-circle exterior edges and the height boundary
    stay unchanged, independently on either sheet.
    """
    lon = np.radians([-2., 2., 2., -2.])
    lat = np.radians([-2., -2., 2., 2.])
    corners = np.column_stack((np.cos(lat)*np.cos(lon), np.cos(lat)*np.sin(lon), np.sin(lat)))
    midpoints = np.array([corners[0]+corners[2], corners[2]+corners[3], corners[3]+corners[0]])
    midpoints /= np.linalg.norm(midpoints, axis=1, keepdims=True)
    vertices = np.vstack((corners, midpoints)) @ rotation_matrix(orientation)
    coarse = [[0, 1, 2], [0, 2, 3]]
    fine = [[0, 1, 4], [4, 1, 2], [0, 4, 6], [4, 2, 5], [6, 5, 3], [4, 5, 6]]
    first, second = fine if refine_a else coarse, fine if refine_b else coarse
    faces = np.array(first+second, np.int64)
    owners = np.r_[np.zeros(len(first), np.int64), np.ones(len(second), np.int64)]
    heights = np.r_[[0., 0., 100., 100., 100., 100.] if refine_a else [0., 100.],
                    np.full(len(second), height_b)]
    if reverse:
        faces, owners, heights = faces[::-1].copy(), owners[::-1].copy(), heights[::-1].copy()
    kind = np.ones(len(faces), np.uint8)
    surface = material_surface.initialize_surface(vertices, faces, owners, kind)
    return SimpleNamespace(material_surface=surface, parcel_plate=owners,
        parcel_patch=surface['face_id'].copy(), parcel_collision_sheet=owners+1,
        mass=surface['reference_area_km2'].copy(), kind=kind, relief=heights-220.,
        suture=np.zeros(len(faces)), pos=material_surface.face_centres(surface),
        omega=np.zeros((2, 3)), plate_uid=np.array([1, 2]), t=0.,
        structure=dict(thickness_km=np.full(len(faces), 35.)),
        collision_contacts=[], next_collision_contact_id=1, next_collision_sheet_id=3,
        config={}, rng=np.random.default_rng(1))


def inspect(world):
    """Exercise real clipping, decision, and exposure without a simulation run."""
    contacts.refresh(world)
    row = world.collision_contacts[0]
    sheets = world.parcel_collision_sheet
    return dict(material_faces=len(world.mass),
        exact_contact_pairs=len(world._collision_overlap['first']),
        sheet_area_km2={str(sheet):float(world.mass[sheets == sheet].sum()) for sheet in (1, 2)},
        overlap_area_km2=row['overlap_area_km2'],
        polarity_policy_version=row['polarity_policy_version'],
        height_scores_m=deepcopy(row['polarity_height_scores_m']),
        top_sheet=row['top_sheet'], under_sheet=row['under_sheet'],
        sheet_mean_exposed_fraction={str(sheet):float(np.average(world.parcel_exposed_fraction[sheets == sheet],
            weights=world.mass[sheets == sheet])) for sheet in (1, 2)})


class CollisionPolarityTests(unittest.TestCase):
    def test_either_or_both_sheet_refinement_preserves_contact_decision_and_exposure(self):
        baseline = inspect(fixture())
        for a in (False, True):
            for b in (False, True):
                with self.subTest(refine_a=a, refine_b=b):
                    result = inspect(fixture(a, b))
                    self.assertEqual(result['top_sheet'], 2)
                    self.assertEqual(result['under_sheet'], 1)
                    self.assertEqual(result['polarity_policy_version'], 2)
                    self.assertAlmostEqual(result['height_scores_m']['1'], 50., places=9)
                    self.assertAlmostEqual(result['height_scores_m']['2'], 60., places=9)
                    for sheet in ('1', '2'):
                        np.testing.assert_allclose(result['sheet_area_km2'][sheet],
                            baseline['sheet_area_km2'][sheet], rtol=1e-12, atol=1e-8)
                    np.testing.assert_allclose(result['overlap_area_km2'], baseline['overlap_area_km2'], rtol=1e-12)
                    self.assertEqual(result['sheet_mean_exposed_fraction'], {'1':0., '2':1.})

    def test_pole_dateline_and_face_order_keep_refinement_invariant_scores(self):
        for orientation in ({'yaw':179.7}, {'yaw':41., 'pitch':-89.8, 'roll':25.}):
            for a in (False, True):
                for b in (False, True):
                    for reverse in (False, True):
                        with self.subTest(orientation=orientation, a=a, b=b, reverse=reverse):
                            result = inspect(fixture(a, b, orientation=orientation, reverse=reverse))
                            self.assertEqual(result['top_sheet'], 2)
                            self.assertAlmostEqual(result['height_scores_m']['1'], 50., places=8)
                            self.assertAlmostEqual(result['height_scores_m']['2'], 60., places=8)

    def test_ties_use_stable_sheet_id_but_resolved_height_difference_still_wins(self):
        tolerance = contacts.POLARITY_TIE_TOLERANCE_M
        for delta, expected in ((0., 1), (.25*tolerance, 1), (2.*tolerance, 2)):
            for a in (False, True):
                for b in (False, True):
                    for reverse in (False, True):
                        with self.subTest(delta=delta, a=a, b=b, reverse=reverse):
                            world = fixture(a, b, height_b=50.+delta,
                                orientation={'yaw':41., 'pitch':-89.8, 'roll':25.}, reverse=reverse)
                            self.assertEqual(inspect(world)['top_sheet'], expected)

    def test_new_contact_never_redecides_after_heights_cross(self):
        world = fixture(False, True)
        inspect(world)
        original = deepcopy(world.collision_contacts[0])
        world.relief[world.parcel_collision_sheet == 1] += 5000.
        world.t = 1.
        contacts.refresh(world, 1.)
        for key in ('id', 'top_sheet', 'under_sheet', 'started_myr', 'polarity_policy_version',
                    'polarity_height_scores_m', 'polarity_tie_tolerance_m', 'polarity_cycle_override'):
            self.assertEqual(world.collision_contacts[0][key], original[key], key)

    def test_policy_two_decision_metadata_has_exact_checkpoint_roundtrip(self):
        world = fixture(True, True)
        inspect(world)
        compatible = dict(engine_sha256='fixture', auxiliary_sources_sha256={}, numpy_version=np.__version__)
        with tempfile.TemporaryDirectory() as temporary:
            target = Path(temporary)/'checkpoint.npz'
            write_checkpoint(target, world, dict(config={}), compatible)
            restored, _ = read_checkpoint(target, compatible, SimpleNamespace)
        self.assertEqual(restored.collision_contacts, world.collision_contacts)
        for simulation in (world, restored):
            simulation.t = 1.
            contacts.refresh(simulation, 1.)
        self.assertEqual(restored.collision_contacts, world.collision_contacts)
        for field in contacts.PARCEL_FIELDS:
            np.testing.assert_array_equal(getattr(restored, field), getattr(world, field))

    def test_legacy_contact_survives_checkpoint_and_refresh_without_reselection(self):
        world = fixture(False, True)
        # A contact made by policy 1 can have this now-disfavored polarity.
        # It must keep its historical identity, with no retroactive relabeling.
        legacy = dict(id=17, top_sheet=1, under_sheet=2, started_myr=-2., last_seen_myr=0.,
            cumulative_convergence_km=10., convergent_myr=2., suture_strength=.1,
            state='active', overlap_area_km2=0.)
        world.collision_contacts = [deepcopy(legacy)]
        world.next_collision_contact_id = 18
        contacts.refresh(world)
        self.assertEqual(world.collision_contacts[0]['top_sheet'], 1)
        self.assertNotIn('polarity_policy_version', world.collision_contacts[0])
        compatible = dict(engine_sha256='fixture', auxiliary_sources_sha256={}, numpy_version=np.__version__)
        with tempfile.TemporaryDirectory() as temporary:
            target = Path(temporary)/'checkpoint.npz'
            write_checkpoint(target, world, dict(config={}), compatible)
            restored, _ = read_checkpoint(target, compatible, SimpleNamespace)
        self.assertEqual(restored.collision_contacts, world.collision_contacts)
        for simulation in (world, restored):
            simulation.t = 1.
            contacts.refresh(simulation, 1.)
        self.assertEqual(restored.collision_contacts, world.collision_contacts)
        for field in contacts.PARCEL_FIELDS:
            np.testing.assert_array_equal(getattr(restored, field), getattr(world, field))
        row = restored.collision_contacts[0]
        self.assertEqual((row['id'], row['top_sheet'], row['under_sheet'], row['started_myr']), (17, 1, 2, -2.))
        self.assertNotIn('polarity_policy_version', row)


if __name__ == '__main__':
    unittest.main()
