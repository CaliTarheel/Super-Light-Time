"""The plastic suture weld's contact state and its material consolidation.

Three things are under test. (1) resist_motion no longer moves a plate: the
weld is a force in plate_balance now, and this module only measures. (2) The
per-contact weld state -- when the suture last shortened, and how much cohesion
an idle one has left -- is created, clocked, saved and validated. (3)
consolidate re-parents welded, buried under-sheet material without changing any
column, any control cell, or the connectedness of the sheet it takes it from.

Where a number is a reproduction of the 396 Myr checkpoint copy it is asserted
against that copy, not against a fixture that could be tuned to agree with it.
"""
from copy import deepcopy
from types import SimpleNamespace
import math
import os
import unittest
from unittest.mock import patch
import numpy as np

import collision_contacts as contacts
import collision_surface
import material_surface
import structure_engine
from tests.test_collision_contacts import fixture as small_fixture, saved

RADIUS = 6371.
CHECKPOINT = os.environ.get('DEEP_TIME_SUTURE_CHECKPOINT', '')


def unit(value):
    value = np.asarray(value, float)
    return value/np.linalg.norm(value, axis=-1, keepdims=True)


def sheet(left, right, owner, *, half_width=1400., nx=17, ny=13, kind=1):
    """A continental patch large enough for the shipped area guards to bite."""
    x, y = np.meshgrid(np.linspace(left, right, nx), np.linspace(-half_width, half_width, ny))
    vertices = unit(np.column_stack((np.ones(x.size), x.ravel()/RADIUS, y.ravel()/RADIUS)))
    faces = []
    for row in range(ny-1):
        for col in range(nx-1):
            a = row*nx+col
            faces.extend(((a, a+1, a+1+nx), (a, a+1+nx, a+nx)))
    return vertices, np.array(faces), np.full(len(faces), owner), np.full(len(faces), kind, np.uint8)


def collided(upper, lower, *, weld=True):
    """Two overlapping sheets on two plates, the upper one standing higher."""
    mesh = material_surface.initialize_surface(*upper)
    material_surface.append_surface(mesh, *lower)
    n = len(mesh['faces'])
    s = SimpleNamespace(material_surface=mesh, parcel_patch=mesh['face_id'].copy(),
        parcel_plate=mesh['face_owner'].copy(), kind=mesh['face_kind'].copy(),
        mass=mesh['reference_area_km2'].copy(), pos=material_surface.face_centres(mesh),
        relief=np.where(mesh['face_owner'] == 0, 980., 480.), suture=np.zeros(n),
        plate_uid=np.array([101, 202], np.int64), t=0., active=np.array([True, True]),
        omega=np.array([[0., 0., .003], [0., 0., -.003]]),
        config=dict(deforming_regions=1, deformation_width_km=450.),
        plate=np.zeros(64, np.int16), trace_xyz=np.empty((0, 3)),
        trace_patch=np.empty(0, np.int64), trace_plate=np.empty(0, np.int64),
        suture_weld_version=contacts.SUTURE_WELD_VERSION)
    structure_engine.initialize_parcels(s)
    contacts.refresh(s)
    if weld:
        s.parcel_collision_suture[:] = 1.
    return s


def ledger(s):
    """The ordered overlap ledger every downstream mass law reads.

    Pairs, their areas and the gravitational cross term sum w h_upper h_lower.
    Consolidation must leave all of it alone: it moves ownership, not geometry.
    """
    overlap = s._collision_overlap
    graph = collision_surface.descendants(s.collision_contacts)
    sheets = s.parcel_collision_sheet
    thickness = s.structure['thickness_km']
    pairs, cross, load = [], 0., 0.
    for a, b, weight in zip(overlap['first'], overlap['second'], overlap['area_km2']):
        one, two = int(sheets[a]), int(sheets[b])
        upper, lower = (int(a), int(b)) if two in graph.get(one, ()) else (int(b), int(a))
        if one not in graph.get(two, ()) and two not in graph.get(one, ()):
            raise ValueError('An overlapping pair lost its vertical order.')
        pairs.append((upper, lower)); cross += float(weight)*float(thickness[a])*float(thickness[b])
        load += float(weight)*float(thickness[lower])
    return sorted(pairs), cross, load, float(overlap['area_km2'].sum())


def strip(count, base=0):
    """A one-quad-wide ribbon: 2*count triangles in a path of shared edges."""
    faces = []
    for i in range(count):
        a = base+2*i
        faces.extend(((a, a+1, a+3), (a, a+3, a+2)))
    return np.array(faces)


class ResistMotionRetirementTests(unittest.TestCase):
    def test_the_arrest_width_and_its_step_cap_are_gone(self):
        self.assertFalse(hasattr(contacts, 'COLLISION_ARREST_WIDTH_KM'))
        source = open(contacts.__file__, encoding='utf-8').read()
        self.assertNotIn('COLLISION_ARREST_WIDTH_KM', source)
        self.assertNotIn('min(.2,', source)

    def test_resist_motion_leaves_motion_alone_and_keeps_its_diagnostic_keys(self):
        s = small_fixture()
        before = s.omega.copy()
        contacts.resist_motion(s, 2.)
        np.testing.assert_array_equal(s.omega, before)
        diagnostics = s.collision_resistance_diagnostics
        for key in ('contacts', 'input_active_contacts', 'distinct_motion_fronts',
                    'rotational_energy_proxy_before', 'rotational_energy_proxy_after',
                    'rotational_energy_proxy_removed', 'angular_motion_proxy_before',
                    'angular_motion_proxy_after', 'angular_motion_proxy_residual_norm',
                    'angular_motion_proxy_contribution_scale', 'angular_motion_proxy_tolerance',
                    'stationarity_residual_norm', 'stationarity_tolerance', 'newton_iterations',
                    'line_search_backtracks', 'maximum_damping_per_contact',
                    'maximum_isolated_front_fraction', 'relaxation_myr',
                    'multiple_contacts_apply_sequentially', 'area_weighted_angular_momentum_conserved',
                    'model', 'proxy_limit', 'cap_scope'):
            self.assertIn(key, diagnostics)
        self.assertEqual(diagnostics['rotational_energy_proxy_removed'], 0.)
        self.assertEqual(diagnostics['maximum_damping_per_contact'], 0.)
        self.assertGreater(diagnostics['input_active_contacts'], 0)
        self.assertEqual(diagnostics['distinct_motion_fronts'], 1)
        # The recorded proxy vectors stay real three-vectors for orient_frame.
        for key in ('angular_motion_proxy_before', 'angular_motion_proxy_after'):
            self.assertEqual(np.asarray(diagnostics[key], float).shape, (3,))

    def test_a_bad_overlap_area_is_still_refused(self):
        s = small_fixture()
        s.collision_contacts[0]['overlap_area_km2'] = np.nan
        s.collision_contacts[0].pop('local_fronts', None)
        with self.assertRaises(ValueError):
            contacts.resist_motion(s, 2.)
        with self.assertRaises(ValueError):
            contacts.resist_motion(small_fixture(), 0.)


class WeldStateTests(unittest.TestCase):
    def test_a_new_contact_is_born_converging_and_fully_cohesive(self):
        s = collided(sheet(-3000., 1200., 0), sheet(-1200., 3000., 1))
        row = s.collision_contacts[0]
        self.assertEqual(row['last_convergent_myr'], s.t)
        self.assertEqual(row['weld_cohesion'], 1.)

    def test_a_migrated_row_dates_its_clock_from_its_own_record(self):
        s = collided(sheet(-3000., 1200., 0), sheet(-1200., 3000., 1))
        closing, idle = deepcopy(s.collision_contacts[0]), deepcopy(s.collision_contacts[0])
        for row in (closing, idle):
            row.pop('last_convergent_myr'); row.pop('weld_cohesion')
            row.update(started_myr=100., last_seen_myr=400., convergent_myr=120.)
        closing['normal_speed_km_myr'] = -4.7
        idle['normal_speed_km_myr'] = 21.
        s.collision_contacts = [closing, idle]
        contacts.ensure_fields(s)
        # Still closing: converging now. Not closing: its shortening is placed
        # as early as its own history allows, so it is credited with the most
        # cooling that record can support and no more.
        self.assertEqual(closing['last_convergent_myr'], 400.)
        self.assertEqual(idle['last_convergent_myr'], 220.)
        self.assertEqual(idle['weld_cohesion'], 1.)

    def test_converging_stamps_the_clock_and_idling_anneals_monotonically(self):
        s = collided(sheet(-3000., 1200., 0), sheet(-1200., 3000., 1))
        row = s.collision_contacts[0]
        s.t = 2.; contacts.refresh(s, 2.)
        self.assertEqual(row['last_convergent_myr'], 2.)
        self.assertEqual(row['weld_cohesion'], 1.)
        # Stop the plates. The suture is idle from here on.
        s.omega[:] = 0.
        cohesion = [row['weld_cohesion']]
        for step in range(75):
            s.t = 2.+2.*(step+1); contacts.refresh(s, 2.)
            cohesion.append(row['weld_cohesion'])
        self.assertEqual(row['last_convergent_myr'], 2.)
        self.assertTrue(all(b <= a for a, b in zip(cohesion, cohesion[1:])))
        self.assertGreater(cohesion[-1], contacts.WELD_COHESION_FLOOR)
        expected = (contacts.WELD_COHESION_FLOOR
                    +(1.-contacts.WELD_COHESION_FLOOR)*np.exp(-150./contacts.WELD_ANNEAL_TAU_MYR))
        self.assertAlmostEqual(cohesion[-1], expected, places=12)

    def test_a_second_refresh_in_one_epoch_cannot_anneal_twice(self):
        s = collided(sheet(-3000., 1200., 0), sheet(-1200., 3000., 1))
        s.omega[:] = 0.
        s.t = 2.; contacts.refresh(s, 2.)
        once = s.collision_contacts[0]['weld_cohesion']
        contacts.refresh(s, 2.)
        self.assertEqual(s.collision_contacts[0]['weld_cohesion'], once)
        contacts.refresh(s, 0.)
        self.assertEqual(s.collision_contacts[0]['weld_cohesion'], once)

    def test_capacity_strengthens_with_cooling_and_falls_with_lost_cohesion(self):
        row = dict(last_convergent_myr=0., weld_cohesion=1.)
        hot = contacts.weld_capacity_factor(row, 0.)
        self.assertAlmostEqual(hot, contacts.WELD_HOT_N_PER_M/contacts.WELD_COLD_N_PER_M, places=12)
        cooled = [contacts.weld_capacity_factor(row, t) for t in (10., 50., 200.)]
        self.assertTrue(all(b > a for a, b in zip(cooled, cooled[1:])))
        # Conductive strengthening is an exponential approach, so it never reaches the cold
        # capacity: at 200 Myr = 4 tau it is 1-e^-4 of the way there. Assert the approach and
        # the e-folding time itself rather than equality with the asymptote, which no finite
        # idle time can satisfy.
        self.assertLess(cooled[-1], 1.)
        self.assertGreater(cooled[-1], .98)
        at_tau = contacts.weld_capacity_factor(row, contacts.WELD_THERMAL_TAU_MYR)
        self.assertAlmostEqual((at_tau-hot)/(1.-hot), 1.-math.exp(-1.), places=9)
        row['weld_cohesion'] = contacts.WELD_COHESION_FLOOR
        self.assertAlmostEqual(contacts.weld_capacity_factor(row, 200.),
                               cooled[-1]*contacts.WELD_COHESION_FLOOR, places=12)

    def test_weld_rows_are_live_two_plate_sutures_only(self):
        s = collided(sheet(-3000., 1200., 0), sheet(-1200., 3000., 1))
        rows = contacts.weld_rows(s)
        self.assertEqual(len(rows), 1)
        row, top, under = rows[0]
        self.assertEqual((top, under), (int(row['top_owner']), int(row['under_owner'])))
        self.assertNotEqual(top, under)
        for state in ('accreted', 'quiet'):
            row['state'] = state
            self.assertEqual(contacts.weld_rows(s), [])
        row['state'] = 'active'; row['overlap_area_km2'] = 0.
        self.assertEqual(contacts.weld_rows(s), [])

    def test_version_zero_worlds_carry_no_weld_state_at_all(self):
        s = collided(sheet(-3000., 1200., 0), sheet(-1200., 3000., 1))
        s.suture_weld_version = 0
        s.collision_contacts[0].pop('last_convergent_myr')
        s.collision_contacts[0].pop('weld_cohesion')
        s.t = 2.; contacts.refresh(s, 2.)
        for field in contacts.WELD_ROW_FIELDS:
            self.assertNotIn(field, s.collision_contacts[0])
        self.assertEqual(contacts.weld_rows(s), [])
        self.assertFalse(contacts.consolidate(s)['enabled'])
        self.assertNotIn('suture_weld_version', contacts.snapshot_fields(s))
        contacts.validate_alignment(s)


class WeldSchemaTests(unittest.TestCase):
    def frame(self, **overrides):
        s = small_fixture()
        s.suture_weld_version = contacts.SUTURE_WELD_VERSION
        contacts.ensure_fields(s)
        f = saved(s)
        f.update(overrides)
        return f

    def test_a_version_one_frame_round_trips_and_carries_its_state(self):
        f = self.frame()
        self.assertEqual(f['suture_weld_version'], contacts.SUTURE_WELD_VERSION)
        self.assertIn('collision_consolidation_diagnostics', f)
        for row in f['collision_contacts']:
            for field in contacts.WELD_ROW_FIELDS:
                self.assertIn(field, row)
        contacts.validate_frame(f)

    def test_weld_state_without_its_version_is_rejected(self):
        f = self.frame()
        del f['suture_weld_version']
        with self.assertRaises(ValueError):
            contacts.validate_frame(f)

    def test_a_legacy_frame_validates_exactly_as_before(self):
        s = small_fixture()
        f = saved(s)
        self.assertNotIn('suture_weld_version', f)
        for row in f['collision_contacts']:
            for field in contacts.WELD_ROW_FIELDS:
                self.assertNotIn(field, row)
        contacts.validate_frame(f)
        contacts.validate_frame(dict(collision_contact_version=0, material_faces=f['material_faces']))
        with self.assertRaises(ValueError):
            contacts.validate_frame(dict(collision_contact_version=0, suture_weld_version=1,
                                         material_faces=f['material_faces']))

    def test_impossible_weld_state_is_refused(self):
        for field, value in (('weld_cohesion', 1.2), ('weld_cohesion', 0.),
                             ('last_convergent_myr', 1e6), ('last_convergent_myr', np.nan),
                             ('weld_cohesion', 'welded')):
            f = self.frame()
            f['collision_contacts'][0][field] = value
            with self.assertRaises(ValueError, msg=f'{field}={value}'):
                contacts.validate_frame(f)


class ConsolidationGuardTests(unittest.TestCase):
    def test_would_disconnect_sees_a_cut_and_an_emptied_body(self):
        faces = strip(6)
        donor = np.arange(12)
        self.assertFalse(contacts._would_disconnect(faces, donor, np.array([0])))
        self.assertFalse(contacts._would_disconnect(faces, donor, np.array([0, 1])))
        self.assertTrue(contacts._would_disconnect(faces, donor, np.array([6, 7])))
        self.assertTrue(contacts._would_disconnect(faces, donor, np.array([6])))
        self.assertTrue(contacts._would_disconnect(faces, donor, donor))

    def test_restore_connectivity_releases_a_minimal_bridge(self):
        faces = strip(6)
        donor = np.arange(12)
        kept, released = contacts._restore_connectivity(faces, donor, np.array([6, 7]))
        self.assertEqual(released, 2)
        self.assertEqual(len(kept), 0)
        kept, released = contacts._restore_connectivity(faces, donor, np.array([0, 1, 6, 7]))
        self.assertEqual(released, 2)
        np.testing.assert_array_equal(kept, np.array([0, 1]))
        self.assertFalse(contacts._would_disconnect(faces, donor, kept))
        kept, released = contacts._restore_connectivity(faces, donor, np.array([0, 1]))
        self.assertEqual(released, 0)
        np.testing.assert_array_equal(kept, np.array([0, 1]))

    def test_whole_bodies_blocks_a_body_that_would_move_entire(self):
        faces = np.r_[strip(4), strip(3, base=100)]
        donor = np.arange(len(faces))
        second = np.arange(8, len(faces))
        np.testing.assert_array_equal(contacts._whole_bodies(faces, donor, second), second)
        self.assertEqual(len(contacts._whole_bodies(faces, donor, second[:-1])), 0)


class ConsolidationTests(unittest.TestCase):
    def wide(self):
        return collided(sheet(-3000., 1200., 0), sheet(-1200., 3000., 1))

    def narrow(self):
        return collided(sheet(-3000., 1400., 0), sheet(-1200., 1400., 1, half_width=1000., ny=9))

    def qualifying(self, s):
        row = s.collision_contacts[0]
        under = int(row['under_sheet'])
        overlap = s._collision_overlap
        first, second, sheets = overlap['first'], overlap['second'], s.parcel_collision_sheet
        take = np.flatnonzero((sheets[first] == under) | (sheets[second] == under))
        touched = np.unique(np.r_[first[take][sheets[first[take]] == under],
                                  second[take][sheets[second[take]] == under]])
        return touched[(s.parcel_exposed_fraction[touched] < contacts.CONSOLIDATION_EXPOSED_FRACTION)
                       & (s.parcel_collision_suture[touched] >= contacts.CONSOLIDATION_SUTURE)]

    def test_only_welded_buried_under_faces_move_and_control_cells_do_not(self):
        s = self.wide()
        row = s.collision_contacts[0]
        expected = self.qualifying(s)
        self.assertGreater(len(expected), 0)
        before_plate = s.plate.copy(); before_parcel = s.parcel_plate.copy()
        report = contacts.consolidate(s)
        self.assertTrue(report['moved'])
        self.assertEqual(report['qualifying_faces'], len(expected))
        moved = np.flatnonzero(s.parcel_plate != before_parcel)
        self.assertTrue(np.all(np.isin(moved, expected)))
        self.assertTrue(np.all(s.parcel_plate[moved] == int(row['top_owner'])))
        # Nothing exposed, nothing unwelded and nothing of the upper sheet.
        self.assertTrue(np.all(s.parcel_exposed_fraction[moved] < .5))
        self.assertTrue(np.all(s.parcel_collision_sheet[moved] == int(row['under_sheet'])))
        np.testing.assert_array_equal(s.plate, before_plate)
        self.assertFalse(report['control_ownership_changed'])

    def test_plan_predicts_the_commit_without_changing_material_ownership(self):
        s = self.wide()
        before = s.parcel_plate.copy()
        surface = s.material_surface
        surface_owners = surface['face_owner'].copy()
        moving, owners, planned = contacts._plan_consolidation(s)
        self.assertGreater(len(moving), 0)
        self.assertEqual(len(moving), planned['faces'])
        np.testing.assert_array_equal(s.parcel_plate, before)
        np.testing.assert_array_equal(surface['face_owner'], surface_owners)
        report = contacts.consolidate(s)
        np.testing.assert_array_equal(np.flatnonzero(s.parcel_plate != before), np.sort(moving))
        np.testing.assert_array_equal(s.parcel_plate[moving], owners)
        self.assertEqual(report['contacts'], planned['contacts'])
        self.assertIs(s.material_surface, surface)

    def test_failed_surface_preparation_leaves_all_owners_and_markers_unchanged(self):
        s = self.wide()
        s.trace_patch = s.parcel_patch.copy()
        s.trace_plate = s.parcel_plate.copy()
        owners = s.parcel_plate.copy()
        trace = s.trace_plate.copy()
        surface = s.material_surface
        surface_owners = surface['face_owner'].copy()
        vertices = surface['vertices'].copy()
        with patch.object(material_surface, 'reassign_owners', side_effect=RuntimeError('surface preparation failed')):
            with self.assertRaisesRegex(RuntimeError, 'surface preparation failed'):
                contacts.consolidate(s)
        np.testing.assert_array_equal(s.parcel_plate, owners)
        np.testing.assert_array_equal(s.trace_plate, trace)
        np.testing.assert_array_equal(surface['face_owner'], surface_owners)
        np.testing.assert_array_equal(surface['vertices'], vertices)
        self.assertIs(s.material_surface, surface)

    def test_entered_face_cannot_change_owner_without_an_energy_handoff(self):
        s = self.wide()
        ordinary = deepcopy(s)
        contacts.consolidate(ordinary)
        moving = np.flatnonzero(ordinary.parcel_plate != s.parcel_plate)
        self.assertGreater(len(moving), 0)
        s.continental_entry_regions = dict(version=1, face_ids=s.parcel_patch.copy())
        s.parcel_entry_region = np.zeros(len(s.parcel_patch), np.int64)
        s.parcel_entry_region[moving[0]] = 1
        owners = s.parcel_plate.copy()
        surface_owners = s.material_surface['face_owner'].copy()
        vertices = s.material_surface['vertices'].copy()
        with self.assertRaisesRegex(ValueError, 'explicit energy and owner handoff'):
            contacts.consolidate(s)
        np.testing.assert_array_equal(s.parcel_plate, owners)
        np.testing.assert_array_equal(s.material_surface['face_owner'], surface_owners)
        np.testing.assert_array_equal(s.material_surface['vertices'], vertices)

    def test_an_unwelded_or_exposed_sheet_consolidates_nothing(self):
        s = self.wide()
        s.parcel_collision_suture[:] = .94
        self.assertFalse(contacts.consolidate(s)['moved'])
        s.parcel_collision_suture[:] = 1.
        s.parcel_exposed_fraction[:] = .5
        report = contacts.consolidate(s)
        self.assertFalse(report['moved'])
        self.assertEqual(report['qualifying_faces'], 0)

    def test_the_step_cap_binds_and_successive_steps_drain_the_backlog(self):
        s = self.wide()
        backlog = len(self.qualifying(s))
        first = contacts.consolidate(s)
        self.assertLessEqual(first['area_km2'], contacts.CONSOLIDATION_STEP_AREA_KM2)
        self.assertGreater(first['cap_withheld_area_km2'], 0.)
        self.assertLess(first['faces'], backlog)
        moved, steps = first['faces'], 1
        for _ in range(12):
            contacts.ensure_fields(s); contacts.refresh(s)
            s.parcel_collision_suture[:] = 1.
            report = contacts.consolidate(s)
            if not report['moved']: break
            self.assertLessEqual(report['area_km2'], contacts.CONSOLIDATION_STEP_AREA_KM2)
            moved += report['faces']; steps += 1
        self.assertGreater(steps, 1)
        self.assertEqual(moved, backlog)

    def test_a_rim_of_the_under_sheet_always_stays(self):
        s = self.narrow()
        row = s.collision_contacts[0]
        under = int(row['under_sheet'])
        area = s.material_surface['area_km2']
        donor = float(area[s.parcel_collision_sheet == under].sum())
        self.assertLess(donor, contacts.CONSOLIDATION_RIM_AREA_KM2+contacts.CONSOLIDATION_STEP_AREA_KM2)
        # Leave a few donor faces unwelded. In this fixture every face of the sheet otherwise
        # qualifies, so the set IS the whole connected component and the first guard blocks it
        # outright -- the rim never becomes the binding constraint, and this test would pass
        # or fail on a guarantee it is not about. Excluding four faces makes the rim the guard
        # under test, which is the point.
        donor_faces = np.flatnonzero((s.parcel_collision_sheet == under)
                                     & (s.parcel_plate == int(row['under_owner'])))
        s.parcel_collision_suture[donor_faces[:4]] = 0.
        report = contacts.consolidate(s)
        self.assertEqual(report['whole_component_faces'], 0)
        self.assertGreater(report['rim_withheld_area_km2'], 0.)
        remaining = float(area[(s.parcel_collision_sheet == under)
                               & (s.parcel_plate == int(row['under_owner']))].sum())
        self.assertGreaterEqual(remaining, contacts.CONSOLIDATION_RIM_AREA_KM2)
        self.assertAlmostEqual(remaining+report['area_km2'], donor, places=6)

    def test_the_donor_sheet_stays_one_connected_body(self):
        s = self.wide()
        row = s.collision_contacts[0]
        under = int(row['under_sheet'])
        faces = s.material_surface['faces']
        donor = np.flatnonzero(s.parcel_collision_sheet == under)
        self.assertEqual(len(np.unique(contacts._components(faces[donor]))), 1)
        contacts.consolidate(s)
        kept = np.flatnonzero((s.parcel_collision_sheet == under)
                              & (s.parcel_plate == int(row['under_owner'])))
        self.assertGreater(len(kept), 0)
        self.assertEqual(len(np.unique(contacts._components(faces[kept]))), 1)

    def test_the_transfer_is_volume_and_cross_term_neutral_and_keeps_provenance(self):
        s = self.wide()
        area = s.material_surface['area_km2']
        volume = float(area@s.structure['thickness_km'])
        pairs, cross, load, total = ledger(s)
        parent = {int(row['id']) for row in s.collision_contacts}
        report = contacts.consolidate(s)
        self.assertTrue(report['moved'])
        contacts.ensure_fields(s)
        contacts.refresh(s)
        after_pairs, after_cross, after_load, after_total = ledger(s)
        self.assertEqual(pairs, after_pairs)
        self.assertLessEqual(abs(after_cross-cross), abs(cross)*1e-12)
        self.assertLessEqual(abs(after_load-load), abs(load)*1e-12)
        self.assertLessEqual(abs(after_total-total), abs(total)*1e-12)
        self.assertEqual(float(area@s.structure['thickness_km']), volume)
        # The daughter sheet is new, its rows are clones, and every clone names
        # the contact it descends from, so a by-sheet or by-contact ledger can
        # still be followed back across the split.
        daughters = [row for row in s.collision_contacts if 'inherited_from_contact' in row]
        self.assertTrue(daughters)
        for row in daughters:
            self.assertIn(int(row['inherited_from_contact']), parent)
            for field in contacts.WELD_ROW_FIELDS:
                self.assertIn(field, row)

    def test_a_consolidated_pair_is_accreted_and_never_consolidates_twice(self):
        s = self.wide()
        first = contacts.consolidate(s)
        contacts.ensure_fields(s); contacts.refresh(s)
        s.parcel_collision_suture[:] = 1.
        owners = s.parcel_plate.copy()
        for row in s.collision_contacts:
            if row['state'] == 'active': continue
            self.assertIn(row['state'], ('accreted', 'quiet'))
        second = contacts.consolidate(s)
        self.assertLessEqual(second['faces'], first['qualifying_faces']-first['faces'])
        daughter = np.flatnonzero(owners != s.parcel_plate)
        self.assertTrue(np.all(s.parcel_plate[daughter] == int(s.collision_contacts[0]['top_owner'])))

    def test_history_markers_follow_their_face(self):
        s = self.wide()
        expected = self.qualifying(s)
        centres = material_surface.face_centres(s.material_surface)
        s.trace_xyz = centres[expected].copy()
        s.trace_patch = s.parcel_patch[expected].copy()
        s.trace_plate = s.parcel_plate[expected].copy()
        contacts.consolidate(s)
        order = np.argsort(s.parcel_patch)
        at = np.searchsorted(s.parcel_patch[order], s.trace_patch)
        np.testing.assert_array_equal(s.trace_plate, s.parcel_plate[order[at]])


@unittest.skipUnless(os.path.exists(CHECKPOINT), 'checkpoint copy unavailable')
class CheckpointInventoryTests(unittest.TestCase):
    """Standing reproduction of the 396 Myr copy; never written back."""

    @classmethod
    def setUpClass(cls):
        import checkpoint
        from tectonics import Simulation
        cls.state, _ = checkpoint.read_checkpoint(CHECKPOINT, None, Simulation)
        cls.state.suture_weld_version = contacts.SUTURE_WELD_VERSION
        contacts.refresh(cls.state, 0.)

    def test_inventory_rim_connectivity_and_neutrality(self):
        s = deepcopy(self.state)
        area = s.material_surface['area_km2']
        volume = float(area@s.structure['thickness_km'])
        pairs, cross, load, total = ledger(s)
        self.assertEqual(len(pairs), 7634)
        self.assertAlmostEqual(cross/1e10, 1.959958, places=5)
        self.assertAlmostEqual(load/1e8, 5.319447, places=5)
        control = s.plate.copy()
        report = contacts.consolidate(s)
        self.assertAlmostEqual(report['qualifying_faces'], 481, delta=10)
        self.assertAlmostEqual(report['qualifying_area_km2'], 8.12e6, delta=1e5)
        self.assertTrue(report['moved'])
        self.assertLessEqual(report['area_km2'], contacts.CONSOLIDATION_STEP_AREA_KM2)
        np.testing.assert_array_equal(s.plate, control)
        row = next(r for r in report['contacts'] if r['qualifying_faces'])
        under, owner = row['under_sheet'], row['under_owner']
        faces = s.material_surface['faces']
        kept = np.flatnonzero((s.parcel_collision_sheet == under) & (s.parcel_plate == owner))
        self.assertEqual(len(np.unique(contacts._components(faces[kept]))), 1)
        self.assertGreaterEqual(float(area[kept].sum()), contacts.CONSOLIDATION_RIM_AREA_KM2)
        contacts.ensure_fields(s); contacts.refresh(s, 0.)
        after_pairs, after_cross, after_load, after_total = ledger(s)
        self.assertEqual(pairs, after_pairs)
        self.assertLessEqual(abs(after_cross-cross), abs(cross)*1e-12)
        self.assertLessEqual(abs(after_load-load), abs(load)*1e-12)
        self.assertLessEqual(abs(after_total-total), abs(total)*1e-12)
        self.assertEqual(float(area@s.structure['thickness_km']), volume)


if __name__ == '__main__':
    unittest.main()
