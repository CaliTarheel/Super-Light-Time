"""Independent oracles for the eclogitic mass sink on real stacked sheets.

Expected eligibility, removal, subsidence and ledger shares are restated here
from the documented law and the fixture's own spherical areas; nothing is read
back from the module's diagnostics. The fixtures build actual overlapping
material sheets, so the overlap ledger, the contact order graph and the trace
mapping are the production ones.
"""
from copy import deepcopy
from types import SimpleNamespace
import unittest

import numpy as np

import collision_contacts
import collision_surface
import crustal_structure as columns
import eclogite_sink
import mesh_coverage
import structure_engine
from tests.test_collision_surface_acceptance import world, triangle, unit


# Independent statement of the law's constants, not the module's imports. A
# changed constitutive choice needs a new oracle, not a silently updated test.
DEPTH_KM = 50.
TAU_MYR = 20.
HEAT_MYR = 10.
PHI = .6
FLOOR_KM = 10.
AIRY = 1000.*(3300.-2800.)/3300.


def stack(*, lower_km=60., upper_km=40., alone_km=70., thin_km=30., burial_myr=42.,
          erosion=0., cover=.45):
    """Upper sheet over a thicker lower sheet, one unburied thick root, one thin.

    Face 0 is the upper sheet, 1 the partly covered lower sheet, 2 an isolated
    70 km root that converts without any burial, and 3 an isolated 30 km column
    that must stay completely untouched, including its support and its erosion.
    """
    lower = triangle()
    upper = lower.copy(); upper[:, 1:] *= cover; upper = unit(upper)
    alone = unit([[-1., -.08, -.05], [-1., .08, -.05], [-1., 0., .08]])
    thin = unit([[-.05, -1., -.08], [-.05, -1., .08], [.08, -1., 0.]])
    s = world([(upper, upper_km, 1000., False), (lower, lower_km, 4000., False),
               (alone, alone_km, 5000., False), (thin, thin_km, 300., False)],
              relations=[(1, 2)])
    count = len(s.mass)
    s.n = count; s.parcel_cell = np.arange(count)
    s.config = dict(erosion=erosion); s.ridge_episodes = []
    s.bcode = np.empty(0, int); s.normal_speed = np.empty(0)
    s.geometric_log_area = np.zeros(count); s.inversion_uplift_m = np.zeros(count)
    s.active = np.ones(len(s.plate_uid), bool)
    s.process_totals = {}
    s.foundering_version = 1  # frozen legacy-law oracles; version 2 has separate volume tests
    # One marker on each column, so the trace pass exercises the real patch map.
    s.trace_xyz = s.pos.copy(); s.trace_patch = s.parcel_patch.copy()
    s.trace_kind = s.kind.copy(); s.trace_plate = s.parcel_plate.copy()
    s.trace_relief_m = s.relief.copy(); s.trace_suture = np.zeros(count)
    s.trace_geometric_log_area = np.zeros(count)
    for name in ('inversion_uplift_m', 'uplift_m', 'extension_m', 'erosion_m',
                 'adjustment_m', 'ridge_uplift_m'):
        setattr(s, 'trace_'+name, np.zeros(count))
    structure_engine.initialize_traces(s, np.arange(count))
    s._indices = lambda points: np.zeros(len(points), int)
    s._prepare_rift_memory = lambda: None
    s._remember_rift_extension = lambda prefix, loss: None
    s._rift_inversion_gain = lambda prefix, stretch, dt: np.zeros(
        len(s.trace_xyz) if prefix else len(s.mass))
    s._record_rift_inversion = lambda gain, dt: None
    overlap = mesh_coverage.material_overlaps(s.material_surface['vertices'],
                                              s.material_surface['faces'],
                                              s.parcel_collision_sheet)
    s._collision_overlap = overlap
    s.parcel_exposed_fraction = collision_contacts._exposure(s, overlap)
    # A face carries burial where something actually covers part of it; the
    # partly covered lower sheet is still more than half exposed here.
    s.parcel_burial_myr = np.where(s.parcel_exposed_fraction < 1., burial_myr, 0.)
    collision_surface.refresh(s, overlap)
    eclogite_sink.ensure_fields(s)
    return s


def expected_eligible(s):
    """Restate the overburden rule from the fixture's own geometry."""
    area = np.asarray(s.material_surface['area_km2'], float)
    thickness = np.asarray(s.structure['thickness_km'], float)
    sheets = np.asarray(s.parcel_collision_sheet)
    graph = collision_surface.descendants(s.collision_contacts)
    overlap = s._collision_overlap
    covered_area = np.zeros(len(area)); overburden = np.zeros(len(area))
    for a, b, weight in zip(overlap['first'], overlap['second'], overlap['area_km2']):
        top, low = ((a, b) if int(sheets[b]) in graph.get(int(sheets[a]), ()) else (b, a))
        covered_area[low] += weight; overburden[low] += weight*thickness[top]
    fraction = np.minimum(covered_area/area, 1.)
    depth = np.divide(overburden, area*fraction, out=np.zeros(len(area)),
                      where=fraction > 0)+thickness
    heated = np.asarray(s.parcel_burial_myr) >= HEAT_MYR
    rooted = np.asarray(s.parcel_root_age_myr) >= HEAT_MYR
    covered = fraction*np.clip(depth-DEPTH_KM, 0., thickness)*heated
    own = (1.-fraction)*np.clip(thickness-DEPTH_KM, 0., thickness)*rooted
    return np.minimum(covered+own, thickness), fraction


def expected_removal(state, eligible, dt):
    thickness = np.asarray(state['thickness_km'], float)
    foundered = np.asarray(state['foundered_m'], float)/1000.
    rate = eligible*(1.-np.exp(-dt/TAU_MYR))
    cap = np.maximum(PHI*thickness-(1.-PHI)*foundered, 0.)
    return np.minimum(np.minimum(rate, cap), np.maximum(thickness-FLOOR_KM, 0.))


class EligibilityTests(unittest.TestCase):
    def test_overburden_depth_uses_the_whole_cover_not_one_pair(self):
        s = stack()
        eligible, scope, attribution = eclogite_sink.eligible_thickness_km(
            s, s.structure['thickness_km'])
        oracle, fraction = expected_eligible(s)
        np.testing.assert_allclose(eligible, oracle, rtol=0, atol=1e-12)
        # Upper sheet: 40 km, uncovered and shallower than the eclogite depth.
        self.assertEqual(eligible[0], 0.)
        # Lower sheet under 40 km of cover: the covered part converts from 50 km
        # down, the uncovered remainder only below its own 50 km.
        self.assertAlmostEqual(eligible[1], fraction[1]*50.+(1.-fraction[1])*10., places=12)
        self.assertGreater(fraction[1], 0.)
        self.assertLess(fraction[1], 1.)
        # The isolated 70 km column converts its own root with no burial at all.
        self.assertAlmostEqual(eligible[2], 20., places=12)
        self.assertEqual(eligible[3], 0.)    # a 30 km column has no root to convert
        self.assertEqual(scope['overlapping_face_pairs'], len(s._collision_overlap['first']))
        self.assertEqual(len(attribution['contact']), len(s._collision_overlap['first']))

    def test_the_two_heating_clocks_gate_their_own_branch(self):
        cold = stack(burial_myr=HEAT_MYR-1.)
        cold.parcel_root_age_myr[:] = 0.
        eligible, _, _ = eclogite_sink.eligible_thickness_km(cold, cold.structure['thickness_km'])
        np.testing.assert_array_equal(eligible, 0.)
        cold.parcel_burial_myr[:] = HEAT_MYR
        covered, _, _ = eclogite_sink.eligible_thickness_km(cold, cold.structure['thickness_km'])
        self.assertGreater(covered[1], 0.)
        self.assertEqual(covered[2], 0.)     # own root still unheated
        cold.parcel_root_age_myr[:] = HEAT_MYR
        both, _, _ = eclogite_sink.eligible_thickness_km(cold, cold.structure['thickness_km'])
        self.assertAlmostEqual(both[2], 20., places=12)

    def test_the_root_clock_advances_only_on_deep_columns_and_once_a_step(self):
        s = stack()
        s.parcel_root_age_myr[:] = 0.
        s.t = 2.
        eclogite_sink.advance_clocks(s, 2.)
        eclogite_sink.advance_clocks(s, 2.)      # same model time: no second advance
        np.testing.assert_array_equal(s.parcel_root_age_myr, [0., 2., 2., 0.])
        s.t = 4.
        eclogite_sink.advance_clocks(s, 2.)
        np.testing.assert_array_equal(s.parcel_root_age_myr, [0., 4., 4., 0.])


class TransactionTests(unittest.TestCase):
    """B2 bounds, B3 elevation identity and B4 the cumulative cap."""

    def test_b2_bounds_hold_and_reference_area_never_moves(self):
        s = stack()
        area_factor = s.structure['area_factor'].copy()
        mass = s.mass.copy()
        for step in range(40):
            s.t = 2.*(step+1)
            eclogite_sink.advance_clocks(s, 2.)
            eligible, _, _ = eclogite_sink.eligible_thickness_km(s, s.structure['thickness_km'])
            removed, _ = eclogite_sink.apply(s.structure, eligible, 2.)
            s.parcel_burial_myr += (1.-s.parcel_exposed_fraction)*2.
            self.assertTrue(np.all(s.structure['thickness_km'][removed > 0] >= FLOOR_KM-1e-12))
            self.assertTrue(np.all(s.structure['thickness_km'] >= columns.MIN_THICKNESS_KM))
            self.assertTrue(np.all(s.structure['reference_thickness_km'] >= columns.MIN_THICKNESS_KM))
            columns.elevation(s.structure)       # crustal_structure._state must pass
        np.testing.assert_array_equal(s.structure['area_factor'], area_factor)
        np.testing.assert_array_equal(s.mass, mass)
        self.assertGreater(s.structure['foundered_m'][1], 0.)

    def test_b3_the_surface_falls_by_exactly_the_airy_amount(self):
        s = stack()
        before = {name: value.copy() for name, value in s.structure.items()}
        eligible, _, _ = eclogite_sink.eligible_thickness_km(s, s.structure['thickness_km'])
        removed, lowering = eclogite_sink.apply(s.structure, eligible, 2.)
        target = columns.elevation(dict(before, thickness_km=before['thickness_km']-removed))
        np.testing.assert_allclose(columns.elevation(s.structure), target, rtol=0, atol=1e-9)
        np.testing.assert_allclose(lowering, columns.elevation(before)-target, rtol=0, atol=1e-9)
        # Air-loaded columns: the drop is the documented coefficient per km.
        np.testing.assert_allclose(lowering, AIRY*removed, rtol=1e-9, atol=1e-9)
        # The neutral reference follows the column, so the relaxation self term
        # sees no new contraction drive on a foundered root.
        bound = before['reference_thickness_km']-removed < columns.MIN_THICKNESS_KM
        np.testing.assert_allclose(
            (s.structure['thickness_km']-s.structure['reference_thickness_km'])[~bound],
            (before['thickness_km']-before['reference_thickness_km'])[~bound], rtol=0, atol=1e-12)

    def test_b3_reference_floor_binds_without_breaking_the_surface_identity(self):
        s = stack()
        s.structure['reference_thickness_km'][:] = columns.MIN_THICKNESS_KM+.5
        before = {name: value.copy() for name, value in s.structure.items()}
        eligible, _, _ = eclogite_sink.eligible_thickness_km(s, s.structure['thickness_km'])
        removed, _ = eclogite_sink.apply(s.structure, eligible, 2.)
        target = columns.elevation(dict(before, thickness_km=before['thickness_km']-removed))
        np.testing.assert_allclose(columns.elevation(s.structure), target, rtol=0, atol=1e-9)
        self.assertTrue(np.all(s.structure['reference_thickness_km'] >= columns.MIN_THICKNESS_KM))
        self.assertTrue(np.any(removed > .5))    # the floor really did bind somewhere

    def test_b4_the_compositional_cap_not_the_floor_limits_the_loss(self):
        s = stack()
        original = s.structure['thickness_km'].copy()
        for step in range(60):
            s.t = 2.*(step+1)
            eclogite_sink.advance_clocks(s, 2.)
            eligible, _, _ = eclogite_sink.eligible_thickness_km(s, s.structure['thickness_km'])
            eclogite_sink.apply(s.structure, eligible, 2.)
            s.parcel_burial_myr += (1.-s.parcel_exposed_fraction)*2.
            foundered = s.structure['foundered_m']/1000.
            self.assertTrue(np.all(foundered <= PHI*(s.structure['thickness_km']+foundered)+1e-12))
        # Every touched column keeps at least 1 - phi of what it started with,
        # and none of them is anywhere near the hard minimum.
        kept = s.structure['thickness_km']/original
        touched = s.structure['foundered_m'] > 0
        self.assertTrue(np.all(kept[touched] >= 1.-PHI-1e-12))
        self.assertTrue(np.all(s.structure['thickness_km'] > columns.MIN_THICKNESS_KM+1.))

    def test_b4_without_the_cap_only_the_floor_stands_between_column_and_bound(self):
        def integrate(fraction, steps=60):
            s = stack(cover=.97)      # a nearly fully covered sheet, as a suture is
            original = s.structure['thickness_km'].copy()
            saved = eclogite_sink.RETURNED_FRACTION
            try:
                eclogite_sink.RETURNED_FRACTION = fraction
                for step in range(steps):
                    s.t = 2.*(step+1)
                    eclogite_sink.advance_clocks(s, 2.)
                    eligible, _, _ = eclogite_sink.eligible_thickness_km(
                        s, s.structure['thickness_km'])
                    eclogite_sink.apply(s.structure, eligible, 2.)
                    s.parcel_burial_myr += (1.-s.parcel_exposed_fraction)*2.
            finally:
                eclogite_sink.RETURNED_FRACTION = saved
            return s.structure['thickness_km'], original
        capped, original = integrate(PHI)
        uncapped, _ = integrate(1.)
        # This is the measurement behind choosing the compositional cap rather
        # than the engineering floor as the limiter: with no cap the columns run
        # down toward the residual floor, which is the only thing then keeping
        # them off the hard minimum the deformation solver enforces.
        self.assertTrue(np.all(uncapped <= capped+1e-12))
        self.assertTrue(np.any(uncapped < capped-1.))
        self.assertTrue(np.all(uncapped >= FLOOR_KM-1e-12))
        lost = 1.-uncapped/original
        self.assertTrue(np.any(lost > PHI+.05))
        self.assertTrue(np.all(1.-capped/original <= PHI+1e-12))


class AutonomyTests(unittest.TestCase):
    """B5: the sink reads no label, no weld state and no motion."""

    def _removal(self, s):
        eligible, _, attribution = eclogite_sink.eligible_thickness_km(
            s, s.structure['thickness_km'])
        state = {name: value.copy() for name, value in s.structure.items()}
        removed, _ = eclogite_sink.apply(state, eligible, 2.)
        return removed, attribution

    def test_b5_labels_weld_and_motion_do_not_change_one_gram(self):
        s = stack()
        base, base_attribution = self._removal(s)
        scrambled = stack()
        scrambled.omega = np.zeros((len(scrambled.plate_uid), 3))
        for index, row in enumerate(scrambled.collision_contacts):
            row['state'] = ('quiet', 'accreted', 'active')[index % 3]
            row['suture_strength'] = .123
        scrambled.parcel_collision_suture[:] = np.linspace(0., 1., len(scrambled.mass))
        scrambled.suture[:] = 1.
        # Emulate consolidation: the buried under sheet moves to the top owner.
        buried = scrambled.parcel_exposed_fraction < .5
        scrambled.parcel_plate[buried] = scrambled.parcel_plate[0]
        moved, moved_attribution = self._removal(scrambled)
        np.testing.assert_array_equal(moved, base)
        # Provenance by sheet and contact survives the ownership move.
        np.testing.assert_array_equal(moved_attribution['contact'], base_attribution['contact'])

    def test_b5_the_sink_writes_nothing_it_does_not_own(self):
        s = stack()
        watched = {name: getattr(s, name).copy() for name in
                   ('parcel_plate', 'parcel_collision_sheet', 'parcel_exposed_fraction',
                    'parcel_burial_myr', 'parcel_collision_suture', 'mass', 'suture')}
        eligible, _, _ = eclogite_sink.eligible_thickness_km(s, s.structure['thickness_km'])
        eclogite_sink.apply(s.structure, eligible, 2.)
        for name, value in watched.items():
            np.testing.assert_array_equal(getattr(s, name), value, err_msg=name)

    def test_b5_two_applications_of_the_same_state_are_bitwise_identical(self):
        s = stack()
        first, _ = self._removal(s)
        second, _ = self._removal(s)
        np.testing.assert_array_equal(first, second)
        self.assertEqual(first.tobytes(), second.tobytes())


class LedgerTests(unittest.TestCase):
    def test_every_key_of_the_reservoir_sums_to_the_total(self):
        s = stack()
        area = np.asarray(s.material_surface['area_km2'], float)
        for step in range(5):
            s.t = 2.*(step+1)
            eclogite_sink.advance_clocks(s, 2.)
            eligible, _, attribution = eclogite_sink.eligible_thickness_km(
                s, s.structure['thickness_km'])
            removed, _ = eclogite_sink.apply(s.structure, eligible, 2.)
            eclogite_sink.record(s, removed, attribution)
            s.parcel_burial_myr += (1.-s.parcel_exposed_fraction)*2.
        ledger = s.mantle_return_km3
        self.assertAlmostEqual(float(area@s.structure['foundered_m'])/1000., ledger['total'], places=6)
        for key in ('by_plate_uid', 'by_sheet', 'by_contact'):
            self.assertAlmostEqual(sum(ledger[key].values()), ledger['total'], places=6)
        self.assertAlmostEqual(s.process_totals['crust_returned_to_mantle_km3'],
                               ledger['total'], places=9)
        # The isolated root lay under no suture and is booked to contact zero;
        # the stacked sheet is booked to the contact that buried it.
        self.assertGreater(ledger['by_contact']['0'], 0.)
        self.assertGreater(ledger['by_contact']['1'], 0.)

    def test_the_reservoir_is_created_once_and_ensure_fields_is_idempotent(self):
        s = stack()
        eclogite_sink.ensure_fields(s)
        ledger = s.mantle_return_km3
        clock = s.parcel_root_age_myr
        eclogite_sink.ensure_fields(s)
        self.assertIs(s.mantle_return_km3, ledger)
        np.testing.assert_array_equal(s.parcel_root_age_myr, clock)
        self.assertEqual(set(ledger), {'total', 'by_plate_uid', 'by_sheet', 'by_contact'})

    def test_the_root_clock_starts_heated_only_where_the_column_is_already_deep(self):
        s = stack()
        del s.parcel_root_age_myr
        eclogite_sink.ensure_fields(s)
        np.testing.assert_array_equal(s.parcel_root_age_myr > 0., [False, True, True, False])


class DeformTests(unittest.TestCase):
    """C4: the marker ledger closes and the erosion accounts are untouched."""

    def _run(self, s, steps=3):
        for step in range(steps):
            s.t = 2.*(step+1)
            structure_engine.deform(s, np.zeros(s.n), np.zeros(s.n), np.zeros(s.n), 2.)
            s.parcel_burial_myr += (1.-s.parcel_exposed_fraction)*2.
        return s

    def test_c4_the_trace_ledger_closes_with_foundering_as_tectonic_lowering(self):
        s = self._run(stack(erosion=1.))
        start = np.array([1000., 4000., 5000., 300.])-220.
        closing = (s.trace_uplift_m-s.trace_extension_m-s.trace_erosion_m+s.trace_adjustment_m)
        np.testing.assert_allclose(s.trace_relief_m-start, closing, rtol=0, atol=2e-10)
        self.assertGreater(s.trace_structure['foundered_m'][1], 0.)
        # The subsidence went to tectonic lowering, never to the erosion counter
        # that the surface-process export reads as sediment.
        self.assertGreater(s.trace_extension_m[1], AIRY*s.trace_structure['foundered_m'][1]/1000.*.99)
        self.assertLess(s.trace_erosion_m[1], 1.)

    def test_c4_markers_carry_their_source_face_loss(self):
        s = self._run(stack())
        np.testing.assert_allclose(s.trace_structure['foundered_m'],
                                   s.structure['foundered_m'], rtol=1e-12, atol=1e-9)

    def test_c4_the_erosion_accounts_are_bitwise_unchanged_where_nothing_founders(self):
        with_sink = self._run(stack(erosion=1.))
        without = stack(erosion=1.)
        without.foundering_version = 0
        without = self._run(without)
        # Face 3 neither founders nor shares a support group with a thinned
        # root. The upper sheet DOES change, because it is carried by a lower
        # column the sink has thinned and therefore requests less erosion: that
        # coupling is physical and must not be tested away.
        quiet = np.array([False, False, False, True])
        np.testing.assert_array_equal(with_sink.structure['foundered_m'][quiet], 0.)
        for name in ('denudation_m', 'rebound_m', 'erosion_rate_m_myr'):
            np.testing.assert_array_equal(with_sink.structure[name][quiet],
                                          without.structure[name][quiet], err_msg=name)
        for name in ('trace_erosion_m', 'trace_uplift_m', 'trace_extension_m', 'trace_relief_m'):
            np.testing.assert_array_equal(getattr(with_sink, name)[quiet],
                                          getattr(without, name)[quiet], err_msg=name)
        self.assertGreater(with_sink.structure['denudation_m'][3], 0.)

    def test_the_step_diagnostics_and_the_column_delta_agree(self):
        s = stack()
        area = np.asarray(s.material_surface['area_km2'], float)
        before = s.structure['foundered_m'].copy()
        s.t = 2.
        structure_engine.deform(s, np.zeros(s.n), np.zeros(s.n), np.zeros(s.n), 2.)
        report = s.foundering_diagnostics
        moved = float(area@(s.structure['foundered_m']-before))/1000.
        self.assertAlmostEqual(report['step_returned_km3'], moved, places=9)
        self.assertAlmostEqual(report['cumulative_returned_km3'], s.mantle_return_km3['total'], places=9)
        self.assertEqual(report['faces_touched'], 2)
        self.assertFalse(report['density_or_temperature_resolved'])
        self.assertFalse(report['lateral_escape_resolved'])
        self.assertEqual(report['continental_slab_pull_coefficient'], 0.)
        self.assertTrue(all(row['continental_slab_pull_n_per_m'] == 0. for row in report['contacts']))
        self.assertGreater(report['maximum_surface_collapse_m_per_myr'], 0.)

    def test_a_version_zero_world_founders_nothing_and_saves_nothing(self):
        s = stack()
        s.foundering_version = 0
        self._run(s, steps=2)
        np.testing.assert_array_equal(s.structure['foundered_m'], 0.)
        np.testing.assert_array_equal(s.trace_structure['foundered_m'], 0.)
        self.assertEqual(eclogite_sink.snapshot_fields(s), {})
        self.assertFalse(hasattr(s, 'foundering_diagnostics'))


class FrameTests(unittest.TestCase):
    def frame(self, count=3, version=1):
        frame = dict(material_faces=np.zeros((count, 3), int))
        if version:
            frame.update(foundering_version=version,
                         material_foundered_m=np.arange(count, dtype=float),
                         mantle_return_km3=dict(total=1., by_plate_uid={'101': 1.},
                                                by_sheet={'2': 1.}, by_contact={'1': 1.}))
        return frame

    def test_a_version_zero_frame_still_validates(self):
        eclogite_sink.validate_frame(self.frame(version=0))
        eclogite_sink.validate_frame(dict(material_faces=np.zeros((3, 3), int)))

    def test_fields_without_their_version_are_rejected(self):
        frame = self.frame(); frame['foundering_version'] = 0
        with self.assertRaises(ValueError):
            eclogite_sink.validate_frame(frame)

    def test_an_unsupported_version_is_rejected(self):
        for value in (2, True, -1):
            with self.assertRaises(ValueError):
                eclogite_sink.validate_frame(dict(self.frame(), foundering_version=value))

    def test_a_damaged_ledger_or_counter_is_rejected(self):
        eclogite_sink.validate_frame(self.frame())
        for broken in (dict(self.frame(), material_foundered_m=np.full(3, -1.)),
                       dict(self.frame(), material_foundered_m=np.zeros(2)),
                       dict(self.frame(), material_foundered_m=np.full(3, np.nan))):
            with self.assertRaises(ValueError):
                eclogite_sink.validate_frame(broken)
        for ledger in (dict(total=-1., by_plate_uid={}, by_sheet={}, by_contact={}),
                       dict(total=1., by_plate_uid={}, by_sheet={}),
                       dict(total=1., by_plate_uid={'1': -1.}, by_sheet={}, by_contact={})):
            with self.assertRaises(ValueError):
                eclogite_sink.validate_frame(dict(self.frame(), mantle_return_km3=ledger))

    def attribution(self, share):
        return dict(lower=np.arange(3), weight=np.ones(3), covered_share=np.full(3, share),
                    covered_area=np.ones(3), contact=np.full(3, 4))

    def test_cancellation_never_books_a_negative_uncovered_remainder(self):
        # Contact key 0 is the remainder that lay under no suture. It is a
        # difference of two sums of the same terms, so summation order alone can
        # make it a tiny negative, and the ledger refuses negative volumes with
        # no tolerance: a run stopped on -1.6e-27 km3.
        volume = np.array([3., 5., 7.])
        shares = eclogite_sink._contact_shares(None, volume, self.attribution(1.+2.220446049250313e-16))
        self.assertGreaterEqual(shares[0], 0.)
        eclogite_sink.validate_frame(dict(self.frame(), mantle_return_km3=dict(
            total=float(volume.sum()), by_plate_uid={}, by_sheet={},
            by_contact={str(key): float(value) for key, value in shares.items()})))

    def test_a_real_covered_imbalance_still_raises(self):
        # Absorbing roundoff must not hide a covered share that exceeds one.
        with self.assertRaisesRegex(ValueError, 'exceeds the volume returned'):
            eclogite_sink._contact_shares(None, np.array([3., 5., 7.]), self.attribution(1.5))

    def test_the_snapshot_carries_the_counter_the_ledger_and_the_report(self):
        s = stack()
        s.t = 2.
        structure_engine.deform(s, np.zeros(s.n), np.zeros(s.n), np.zeros(s.n), 2.)
        fields = eclogite_sink.snapshot_fields(s)
        self.assertEqual(fields['foundering_version'], 1)
        np.testing.assert_array_equal(fields['material_foundered_m'], s.structure['foundered_m'])
        self.assertIsNot(fields['mantle_return_km3'], s.mantle_return_km3)
        self.assertEqual(fields['mantle_return_km3'], s.mantle_return_km3)
        self.assertGreater(fields['foundering_diagnostics']['step_returned_km3'], 0.)
        eclogite_sink.validate_frame(dict(fields, material_faces=s.material_surface['faces']))


class AlignmentTests(unittest.TestCase):
    def test_a_short_counter_is_padded_and_a_long_one_is_refused(self):
        s = stack()
        s.structure['foundered_m'] = np.zeros(len(s.mass)-1)
        eclogite_sink.ensure_fields(s)
        self.assertEqual(len(s.structure['foundered_m']), len(s.mass))
        s.structure['foundered_m'] = np.zeros(len(s.mass)+1)
        with self.assertRaises(ValueError):
            eclogite_sink.ensure_fields(s)

    def test_validate_alignment_rejects_a_stale_ledger(self):
        s = stack()
        eclogite_sink.validate_alignment(s)
        s.structure['foundered_m'] = np.zeros(len(s.mass)+1)
        with self.assertRaises(ValueError):
            eclogite_sink.validate_alignment(s)
        s.structure['foundered_m'] = np.zeros(len(s.mass))
        s.parcel_root_age_myr = np.full(len(s.mass), -1.)
        with self.assertRaises(ValueError):
            eclogite_sink.validate_alignment(s)

    def test_a_trace_without_a_material_face_is_refused(self):
        s = stack()
        s.trace_patch = s.trace_patch+10_000
        with self.assertRaises(ValueError):
            eclogite_sink.trace_values(s, np.zeros(len(s.mass)))
        with self.assertRaises(ValueError):
            eclogite_sink.trace_values(s, None)

    def test_an_overlap_without_a_recorded_order_is_refused(self):
        s = stack()
        s.collision_contacts = []
        with self.assertRaises(ValueError):
            eclogite_sink.eligible_thickness_km(s, s.structure['thickness_km'])


if __name__ == '__main__':
    unittest.main()
