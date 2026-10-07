"""Finite inherited order at actual native first contact; no new entry work."""
from copy import deepcopy
from fractions import Fraction
from pathlib import Path
import pickle
import tempfile
import unittest

import numpy as np

import checkpoint
import collision_contacts as contacts
import continental_entry
import entry_regions
import inherited_collision_polarity as policy
import material_surface
from ridge_geometry import rotate
from tests.test_continental_entry import unit
from tests.test_slab_tether_forces import fixture as loaded_source


def _piece(triangle, refine):
    if not refine:
        return triangle, np.array([[0, 1, 2]])
    vertices = np.vstack((triangle, unit(triangle+np.roll(triangle, -1, axis=0))))
    return vertices, np.array([[0, 3, 5], [3, 1, 4], [5, 4, 2], [3, 4, 5]])


def world(*, half_length=500., finite=True, opposite=False,
          orientation=(0., 0., 0.), refine_source=False, refine_upper=False):
    s = loaded_source()
    lower, lf = _piece(unit([[1., -.04, -.02], [1., -.02, -.01], [1., -1e-7, .04]]), refine_source)
    upper, uf = _piece(unit([[1., .02, -.02], [1., .07, -.01], [1., .05, .04]]), refine_upper)
    faces = np.vstack((lf, uf+len(lower)))
    owners = np.r_[np.zeros(len(lf), int), np.ones(len(uf), int)]
    vertices = rotate(np.vstack((lower, upper)), orientation)
    ids = np.arange(len(faces))+917
    surface = material_surface.initialize_surface(vertices, faces, owners,
        np.ones(len(faces), np.uint8), face_id=ids)
    s.material_surface = surface
    s.parcel_patch = ids.copy(); s.parcel_plate = owners.copy()
    s.kind = surface['face_kind'].copy(); s.mass = surface['reference_area_km2'].copy()
    s.pos = material_surface.face_centres(surface)
    s.relief = np.where(owners == 0, 5000., 100.); s.suture = np.zeros(len(faces))
    s.structure = dict(thickness_km=np.full(len(faces), 35.), area_factor=np.ones(len(faces)))
    s.trace_patch = np.empty(0, np.int64); s.trace_xyz = np.empty((0, 3))
    s.t = 0.; s.config['deforming_regions'] = 1
    s.rng = np.random.default_rng(329)
    s.plate_balance_version = s.plate_resistance_version = s.material_mechanics_version = 1
    s.bmid = rotate(s.bmid, orientation); s.bn = rotate(s.bn, orientation)
    contacts.refresh(s)
    assert not s.collision_contacts
    row = s.trench_systems[0]
    spec = dict(face_ids=ids[owners == 0], trench_id=row['id'], overriding_plate_uid=22,
        hinge_normal=rotate(np.array([[0., 1., 0.]]), orientation)[0], dip_degrees=50.)
    if finite:
        spec.update(finite_midpoint=rotate(np.array([[1., 0., 0.]]), orientation)[0],
                    finite_half_length_km=half_length)
    specs = [spec]
    if opposite:
        # Two independently authored opposed histories are a negative input
        # for this one-order-per-sheet-pair representation. The policy creates
        # neither history and must leave both inventories unchanged on refusal.
        other = deepcopy(loaded_source().trench_systems[0])
        other.update(id=2, downgoing_plate_uid=22, overriding_plate_uid=11)
        s.trench_systems.append(other)
        specs.append(dict(spec, face_ids=ids[owners == 1], trench_id=2, overriding_plate_uid=11))
    entry_regions.initialize(s, specs)
    if not opposite:
        assert entry_regions.energy_j(s) == 0.
    return s


def move_to_contact(s, orientation=(0., 0., 0.)):
    axis = rotate(np.array([[0., 0., .055]]), orientation)[0]
    selected = s.material_surface['vertex_owner'] == 0
    s.material_surface['vertices'][selected] = rotate(s.material_surface['vertices'][selected], axis)
    material_surface.refresh_geometry(s.material_surface)
    s.pos = material_surface.face_centres(s.material_surface)


def state_bytes(s):
    # The miniature contains only native checkpoint-safe data and RNG. Pickle
    # is a test-only full-state comparison, never a checkpoint input reader.
    return pickle.dumps(vars(s), protocol=5)


class InheritedCollisionPolarityTests(unittest.TestCase):
    def test_real_contact_keeps_actual_finite_trench_order_despite_reversed_height_vote(self):
        s = world(); before = deepcopy(s.trench_systems)
        mass = s.mass.copy(); columns = deepcopy(s.structure)
        move_to_contact(s); contacts.refresh(s)
        row, = s.collision_contacts
        self.assertEqual((row['top_plate_uid'], row['under_plate_uid']), (22, 11))
        self.assertEqual(row['polarity_policy_version'], 3)
        self.assertGreater(row['polarity_height_scores_m']['1'], row['polarity_height_scores_m']['2'])
        self.assertAlmostEqual(row['overlap_area_km2'], 13371.202355900572, places=6)
        self.assertTrue(all(w['covered_fraction'] == ['1', '1'] for w in row['polarity_entry_witnesses']))
        self.assertEqual(s.trench_systems, before)
        np.testing.assert_array_equal(s.mass, mass)
        for name in columns: np.testing.assert_array_equal(s.structure[name], columns[name])
        # Correct identity does NOT install the missing combined energy law.
        with self.assertRaisesRegex(continental_entry.EntryGeometryError, 'resolve oceanic subregions'):
            entry_regions.energy_j(s)

    def test_no_registry_preserves_the_original_height_choice(self):
        s = world(); del s.continental_entry_regions; del s.parcel_entry_region
        move_to_contact(s); contacts.refresh(s)
        row, = s.collision_contacts
        self.assertEqual((row['top_plate_uid'], row['under_plate_uid']), (11, 22))
        self.assertEqual(row['polarity_policy_version'], 2)

    def test_remote_finite_arc_does_not_order_unrelated_overlap(self):
        s = world()
        s.continental_entry_regions['regions'][0]['finite_midpoint'] = unit([1., 0., 1.])
        move_to_contact(s); contacts.refresh(s)
        row, = s.collision_contacts
        self.assertEqual(row['top_plate_uid'], 11)
        self.assertEqual(row['polarity_policy_version'], 2)

    def test_partial_infinite_conflicting_and_stale_witnesses_reject_full_refresh_without_mutation(self):
        for kind in ('partial', 'infinite', 'opposed', 'source_missing', 'source_owner', 'shutdown', 'unknown_phase'):
            with self.subTest(kind=kind):
                s = world(half_length=50. if kind == 'partial' else 500.,
                          finite=kind != 'infinite', opposite=kind == 'opposed')
                if kind == 'source_missing': s.trench_systems.clear()
                if kind == 'source_owner': s.trench_systems[0]['downgoing_plate_uid'] = 22
                if kind == 'shutdown': s.trench_systems[0]['phase'] = 'shutdown'
                if kind == 'unknown_phase': s.trench_systems[0]['phase'] = 'unsupported'
                move_to_contact(s); before = state_bytes(s)
                aliases = (s.suture, s.parcel_burial_myr, s.collision_contacts, s._collision_overlap)
                alias_before = pickle.dumps(aliases, protocol=5)
                with self.assertRaises(ValueError): contacts.refresh(s, 2.)
                self.assertEqual(state_bytes(s), before)
                self.assertEqual(pickle.dumps(aliases, protocol=5), alias_before)

    def test_first_use_and_malformed_collision_fields_remain_absent_or_unchanged_on_rejection(self):
        for malformed in (False, True):
            s = world(half_length=50.)
            move_to_contact(s)
            for name in contacts.PARCEL_FIELDS+('collision_contacts', 'next_collision_contact_id',
                    'next_collision_sheet_id', '_collision_signature', '_collision_overlap',
                    'collision_diagnostics'):
                if hasattr(s, name): delattr(s, name)
            if malformed: s.parcel_collision_sheet = np.ones(3, np.int64)
            before = state_bytes(s)
            with self.assertRaises(ValueError): contacts.refresh(s, 2.)
            self.assertEqual(state_bytes(s), before)

    def test_direct_policy_rejects_pair_shape_truncation_and_invalid_indices(self):
        s = world(); move_to_contact(s); before = state_bytes(s)
        for first, second in (([0], []), ([0, 0], [1]), ([[0]], [[1]]),
                ([0.], [1]), ([-1], [1]), ([0], [2]), ([0], [0])):
            with self.subTest(first=first, second=second):
                with self.assertRaises(ValueError): policy.choose(s, 1, 2, first, second)
                self.assertEqual(state_bytes(s), before)

    def test_direct_policy_rejects_nonunit_and_misaligned_original_geometry_without_mutation(self):
        s = world(); move_to_contact(s)
        mutations = {
            'scaled_vertices': lambda x: x.material_surface.update(vertices=x.material_surface['vertices']*2.),
            'nonfinite_vertices': lambda x: x.material_surface['vertices'].__setitem__((0, 0), np.nan),
            'complex_vertices': lambda x: x.material_surface.update(vertices=x.material_surface['vertices'].astype(complex)),
            'object_vertices': lambda x: x.material_surface.update(vertices=x.material_surface['vertices'].astype(object)),
            'short_vertices': lambda x: x.material_surface.update(vertices=x.material_surface['vertices'][:, :2]),
            'float_faces': lambda x: x.material_surface.update(faces=x.material_surface['faces'].astype(float)),
            'negative_vertex_index': lambda x: x.material_surface['faces'].__setitem__((0, 0), -1),
            'large_vertex_index': lambda x: x.material_surface['faces'].__setitem__((0, 0), len(x.material_surface['vertices'])),
            'repeated_vertex_index': lambda x: x.material_surface['faces'].__setitem__((0, 0), x.material_surface['faces'][0, 1]),
            'missing_face': lambda x: x.material_surface.update(faces=x.material_surface['faces'][:1]),
            'float_radius': lambda x: x.material_surface.update(radius_km=np.nan),
            'bool_radius': lambda x: x.material_surface.update(radius_km=True),
            'unaligned_face_owner': lambda x: x.material_surface['face_owner'].__setitem__(0, 1),
        }
        for name, mutate in mutations.items():
            with self.subTest(case=name):
                trial = deepcopy(s); mutate(trial); before = state_bytes(trial)
                with self.assertRaises(ValueError): policy.choose(trial, 1, 2, [0], [1])
                self.assertEqual(state_bytes(trial), before)

    def test_direct_policy_rejects_coerced_or_wrapped_original_identities_without_mutation(self):
        s = world(); move_to_contact(s)
        mutations = {
            # -2 formerly wrapped to slot 0 and fabricated a matching UID 11.
            'negative_owner_wrap': lambda x: x.parcel_plate.__setitem__(0, -2),
            'large_owner': lambda x: x.parcel_plate.__setitem__(0, len(x.plate_uid)),
            'float_owner': lambda x: setattr(x, 'parcel_plate', x.parcel_plate.astype(float)),
            'fractional_trench_id': lambda x: x.trench_systems[0].update(id=1.75),
            'float_trench_id': lambda x: x.trench_systems[0].update(id=1.),
            'bool_trench_id': lambda x: x.trench_systems[0].update(id=True),
            'fractional_source_id': lambda x: x.continental_entry_regions['regions'][0].update(source_trench_id=1.75),
            'fractional_region_id': lambda x: x.continental_entry_regions['regions'][0].update(id=1.75),
            'fractional_source_down_uid': lambda x: x.trench_systems[0].update(downgoing_plate_uid=11.75),
            'float_source_over_uid': lambda x: x.trench_systems[0].update(overriding_plate_uid=22.),
            'float_region_over_uid': lambda x: x.continental_entry_regions['regions'][0].update(overriding_plate_uid=22.),
            'float_plate_uid': lambda x: setattr(x, 'plate_uid', x.plate_uid.astype(float)),
            'duplicate_plate_uid': lambda x: x.plate_uid.__setitem__(1, x.plate_uid[0]),
            'float_material_ids': lambda x: setattr(x, 'parcel_patch', x.parcel_patch.astype(float)),
            'duplicate_material_id': lambda x: x.parcel_patch.__setitem__(1, x.parcel_patch[0]),
            'overflow_material_id': lambda x: setattr(x, 'parcel_patch', np.array([2**63, 918], np.uint64)),
            'float_surface_id': lambda x: x.material_surface.update(face_id=x.material_surface['face_id'].astype(float)),
            'float_registered_ids': lambda x: x.continental_entry_regions.update(face_ids=x.parcel_patch.astype(float)),
            'float_membership': lambda x: setattr(x, 'parcel_entry_region', x.parcel_entry_region.astype(float)),
            'negative_membership': lambda x: x.parcel_entry_region.__setitem__(0, -1),
            'float_sheet': lambda x: setattr(x, 'parcel_collision_sheet', x.parcel_collision_sheet.astype(float)),
            'zero_sheet': lambda x: x.parcel_collision_sheet.__setitem__(0, 0),
            'unaligned_sheets': lambda x: setattr(x, 'parcel_collision_sheet', x.parcel_collision_sheet[:1]),
        }
        for name, mutate in mutations.items():
            with self.subTest(case=name):
                trial = deepcopy(s); mutate(trial); before = state_bytes(trial)
                with self.assertRaises(ValueError): policy.choose(trial, 1, 2, [0], [1])
                self.assertEqual(state_bytes(trial), before)
        before = state_bytes(s)
        for one, two in ((1., 2), (True, 2), (1, 2.75), (1, 1), (-1, 2)):
            with self.subTest(one=one, two=two):
                with self.assertRaises(ValueError): policy.choose(s, one, two, [0], [1])
                self.assertEqual(state_bytes(s), before)

    def test_existing_contact_is_never_rescored_or_migrated(self):
        s = world(); registry = s.continental_entry_regions; membership = s.parcel_entry_region
        del s.continental_entry_regions; del s.parcel_entry_region
        move_to_contact(s); contacts.refresh(s)
        original = deepcopy(s.collision_contacts[0])
        s.continental_entry_regions = registry; s.parcel_entry_region = membership
        s.relief = s.relief[::-1].copy()
        contacts.refresh(s)
        row, = s.collision_contacts
        for key in ('id', 'top_sheet', 'under_sheet', 'polarity_policy_version', 'polarity_height_scores_m'):
            self.assertEqual(row[key], original[key])

    def test_inherited_cycle_conflict_rejects_instead_of_flipping_the_source(self):
        s = world()
        s.collision_contacts = [dict(id=17, top_sheet=1, under_sheet=3, state='quiet'),
                                dict(id=18, top_sheet=3, under_sheet=2, state='quiet')]
        s.next_collision_contact_id = 19
        move_to_contact(s); before = state_bytes(s)
        with self.assertRaisesRegex(ValueError, 'conflicts with existing stack order'):
            contacts.refresh(s)
        self.assertEqual(state_bytes(s), before)

    def test_exact_positive_uncovered_sliver_cannot_disappear_in_area_roundoff(self):
        whole = [tuple(Fraction(int(i == j)) for i in range(3)) for j in range(3)]
        tiny = Fraction(1, 10**30)
        inside = policy._clip(whole, (-tiny, Fraction(1), Fraction(1)))
        area = policy._area(inside)
        self.assertEqual(float(area), 1.)  # Float subtraction would erase loss.
        self.assertLess(area, 1)
        self.assertEqual(1-area, tiny**2/(1+tiny)**2)

    def test_rotated_and_independently_refined_sheets_preserve_finite_order(self):
        for orientation in ((0., 0., 0.), (.4, -.6, .7), (0., 1.5, 3.)):
            for a, b in ((True, False), (False, True), (True, True)):
                with self.subTest(orientation=orientation, lower=a, upper=b):
                    s = world(orientation=orientation, refine_source=a, refine_upper=b)
                    move_to_contact(s, orientation); contacts.refresh(s)
                    row, = s.collision_contacts
                    self.assertEqual((row['top_plate_uid'], row['under_plate_uid']), (22, 11))
                    self.assertTrue(all(w['covered_fraction'] == ['1', '1'] for w in row['polarity_entry_witnesses']))
                    self.assertAlmostEqual(row['overlap_area_km2'], 13371.202355900572, places=5)

    def test_typed_checkpoint_preserves_order_and_source_witnesses(self):
        s = world(); move_to_contact(s); contacts.refresh(s)
        with tempfile.TemporaryDirectory() as directory:
            target = Path(directory)/'finite-polarity.npz'
            checkpoint.write_checkpoint(target, s, dict(config=s.config), {})
            restored, _ = checkpoint.read_checkpoint(target, None, type(s))
        self.assertEqual(restored.collision_contacts, s.collision_contacts)
        s.relief *= -1.; restored.relief *= -1.
        contacts.refresh(s); contacts.refresh(restored)
        self.assertEqual(restored.collision_contacts, s.collision_contacts)


if __name__ == '__main__': unittest.main(verbosity=2)
