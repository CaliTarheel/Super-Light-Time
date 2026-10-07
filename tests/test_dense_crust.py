"""Mass, heat and loading oracles for the retained-phase column transaction."""
from copy import deepcopy
import unittest
import numpy as np
import dense_crust as phase
import crust_inventory as inventory
import crustal_structure as columns


def advance(*args, **kwargs):
    # These analytical drainage scenarios prescribe an admitted weak root;
    # the production-independent kernel itself defaults to NO detachment.
    kwargs.setdefault('detachment_fraction', 1.)
    return phase.advance(*args, **kwargs)


def state(thickness=40., temperature=700.):
    s = dict(thickness_km=np.atleast_1d(float(thickness)), area_factor=np.ones(1), foundered_m=np.zeros(1))
    inventory.initialize(s)
    phase.initialize(s, temperature)
    return s


def elevation(s):
    return columns.AIRY_M_PER_KM*s['thickness_km']+phase.elevation_correction(s)


class DenseCrustTests(unittest.TestCase):
    def sparse_dense_column(self):
        s = state(thickness=150., temperature=200.)
        dense_eq = .002348043038260685
        s[phase.DENSE][:] = dense_eq*phase.RATIO
        s[phase.CONVERTED][:] = s[phase.DENSE]
        s[inventory.REMAINING][:] = 85.5265935134907+dense_eq
        s[inventory.BASELINE][:] = s[inventory.REMAINING]
        s[phase.HEAT][:] = phase.mass_volume(s)*200.
        s[phase.HEAT_BASELINE][:] = s[phase.HEAT]
        phase.validate(s)
        return s

    def test_inactive_dense_root_never_accumulates_fictitious_return(self):
        s = self.sparse_dense_column()
        before = deepcopy(s)
        for _ in range(256):
            report = phase.advance(s, 0., 200., 10., .25, drainage_rate_myr=0.)
            np.testing.assert_array_equal(report['returned_equivalent_km'], 0.)
        for key in phase.FIELDS+inventory.FIELDS+('thickness_km', 'foundered_m'):
            np.testing.assert_array_equal(s[key], before[key])

    def test_weak_drainage_is_resolved_on_dense_inventory_scale(self):
        s = self.sparse_dense_column()
        dense_eq = s[phase.DENSE]/phase.RATIO
        expected_return = dense_eq*(-np.expm1(-1e-9*.25))
        report = phase.advance(s, 0., 200., 10., .25, drainage_rate_myr=1e-9)
        np.testing.assert_allclose(report['returned_equivalent_km'], expected_return,
                                   rtol=0., atol=4*float(np.spacing(dense_eq[0])))
        self.assertGreater(float(report['returned_equivalent_km'][0]), 0.)
        phase.validate(s)

    def close(self, one, two):
        np.testing.assert_allclose(one, two, rtol=3e-12, atol=1e-10)

    def test_sequential_solution_retains_phase_before_return_and_conserves_mass(self):
        s = state()
        original = phase.mass_volume(s).copy()
        report = advance(s, 1., 700., 10., 5., reaction_tau_myr=10., detachment_tau_myr=20.)
        # Independent textbook solution U=24e^-at, D=24a/(b-a)(e^-at-e^-bt).
        u = 24.*np.exp(-.5)
        d = 24.*.1/(.05-.1)*(np.exp(-.5)-np.exp(-.25))
        r = 24.-u-d
        self.close(s[phase.DENSE], d*2800./3450.)
        self.close(s[inventory.RETURNED], r)
        self.close(phase.mass_volume(s)+s[inventory.RETURNED], original)
        self.close(phase.restored_thickness(s), original-s[inventory.RETURNED])
        self.close(original-phase.restored_thickness(s), report['returned_equivalent_km'])
        self.assertGreater(float(s[phase.DENSE][0]), 0.)
        self.assertGreater(float(report['contraction_km'][0]), 0.)
        self.close(s[phase.CONVERTED], s[phase.DENSE]+s[phase.RETURNED])
        self.close(phase.temperature(s), 700.)

    def test_density_alone_does_not_authorize_detachment(self):
        s = state()
        phase.advance(s, 1., 700., 10., 100.)
        self.assertGreater(float(s[phase.DENSE][0]), 19.)
        self.close(s[phase.RETURNED], 0.)
        self.close(s[inventory.RETURNED], 0.)
        self.close(phase.mass_volume(s), 40.)

    def test_equal_rates_and_long_steps_have_stable_exact_limits(self):
        u, d, r = phase._chain(np.array([24.]), np.array([0.]), np.array([.1]), np.array([5.]), .1)
        self.close(u, 24.*np.exp(-.5))
        self.close(d, 24.*.5*np.exp(-.5))
        self.close(u+d+r, 24.)
        s = state()
        advance(s, 1., 700., 10., 100000.)
        self.close(phase.mass_volume(s)+s[inventory.RETURNED], 40.)
        self.close(s[phase.DENSE], 0.)
        self.close(s[inventory.RETURNED], 24.)

    def test_constant_conditions_are_timestep_partition_invariant(self):
        whole, pieces = state(), state()
        advance(whole, .7, 700., 10., 20.)
        for _ in range(20):
            advance(pieces, .7, 700., 10., 1.)
        for key in phase.FIELDS+inventory.FIELDS+('thickness_km',):
            self.close(whole[key], pieces[key])

    def test_cold_root_does_not_react_until_thermal_threshold_is_crossed(self):
        s = state(temperature=200.)
        before = deepcopy(s)
        first = advance(s, 1., 800., 10., 5.)
        self.close(s[phase.CONVERTED], 0.)
        self.close(first['hot_duration_myr'], 0.)
        report = advance(s, 1., 800., 10., 15.)
        crossing = 10.*np.log(3.)
        self.close(report['hot_duration_myr'], 20.-crossing)
        self.close(s[phase.CONVERTED]/phase.RATIO, 24.*(1.-np.exp(-(20.-crossing)/10.)))
        self.close(phase.temperature(s), 800.-600.*np.exp(-2.))
        self.assertTrue(np.array_equal(before['foundered_m'], [0.]))

    def test_cooling_stops_new_reaction_but_keeps_and_drains_existing_phase(self):
        s = state(temperature=800.)
        advance(s, 1., 800., 10., 5.)
        prior_converted = s[phase.CONVERTED].copy()
        report = advance(s, 1., 200., 1., 10.)
        self.close(report['hot_duration_myr'], np.log(1.5))
        after = s[phase.CONVERTED].copy()
        self.assertTrue(np.all(after > prior_converted))
        dense = s[phase.DENSE].copy()
        advance(s, 1., 200., 1., 10.)
        self.close(s[phase.CONVERTED], after)
        self.close(s[phase.DENSE], dense*np.exp(-.5))
        self.assertLess(float(phase.temperature(s)[0]), 201.)

    def test_retained_conversion_subsides_and_detachment_rebounds(self):
        s = state()
        initial = elevation(s)
        converted = advance(s, 1., 700., 10., 10., detachment_tau_myr=1e100)
        self.assertLess(float(elevation(s)[0]), float(initial[0]))
        self.close(initial-elevation(s), converted['converted_equivalent_km']*(1.-2800./3450.)*1000.)
        before = elevation(s)
        detached = advance(s, 0., 700., 10., 10.)
        self.close(elevation(s)-before, detached['returned_km']*(3450.-3300.)/3300.*1000.)
        self.assertGreater(float(elevation(s)[0]), float(before[0]))

    def test_erosion_reaches_dense_material_only_after_removing_upper_crust(self):
        s = state()
        advance(s, 1., 700., 10., 10., detachment_tau_myr=1e100)
        initial_mass = phase.mass_volume(s).copy()
        dense = s[phase.DENSE].copy()
        phase.erosion(s, np.array([1.]))
        s['thickness_km'] -= 1.
        self.close(s[phase.DENSE], dense)
        ordinary = s['thickness_km']-dense
        phase.erosion(s, ordinary+1.)
        s['thickness_km'] -= ordinary+1.
        self.close(s[phase.DENSE], dense-1.)
        self.close(phase.mass_volume(s)+(1.+ordinary)+3450./2800., initial_mass)
        phase.validate(s)
        inventory.validate(s)
        self.close(phase.temperature(s), 700.)

    def test_heat_ledger_closes_under_bath_erosion_magma_and_drainage(self):
        s = state(temperature=400.)
        advance(s, .5, 900., 5., 15.)
        phase.erosion(s, np.array([2.]))
        s['thickness_km'] -= 2.
        s['thickness_km'] += 3.
        inventory.add(s, np.array([3.]))
        phase.validate(s)
        advance(s, .2, 200., 10., 20.)
        phase.validate(s)
        self.close(s[phase.HEAT_BASELINE]+s[phase.HEAT_ADDED]+s[phase.HEAT_BATH],
                   s[phase.HEAT]+s[phase.HEAT_ERODED]+s[phase.HEAT_RETURNED])

    def test_invalid_state_and_infeasible_floor_are_rejected_atomically(self):
        s = state()
        before = deepcopy(s)
        for kwargs in (dict(eligible_fraction=2.), dict(thermal_tau_myr=0.),
                       dict(dt=-1.), dict(minimum_thickness_km=41.)):
            controls = dict(eligible_fraction=1., bath_temperature_c=700., thermal_tau_myr=10., dt=20.)
            controls.update(kwargs)
            with self.assertRaises(ValueError): advance(s, **controls)
            for key in s: np.testing.assert_array_equal(s[key], before[key])
        corrupt = deepcopy(s)
        corrupt[phase.CONVERTED] += 1.
        with self.assertRaises(ValueError): phase.validate(corrupt)
        with self.assertRaises(ValueError): phase.initialize(s, 700.)

    def test_phase_adjusted_floor_preserves_conversion_but_return_spends_reserve(self):
        converted = state(8.01, 900.)
        initial_mass = phase.mass_volume(converted).copy()
        initial_restored = phase.restored_thickness(converted).copy()

        phase.advance(converted, 1., 900., 10., 2., minimum_thickness_km=8.)
        self.assertLess(float(converted['thickness_km'][0]), 8.)
        self.close(phase.restored_thickness(converted), initial_restored)
        self.close(phase.mass_volume(converted), initial_mass)
        self.close(converted[phase.RETURNED], 0.)
        phase.validate(converted)

        # Return spends the remaining 0.01 km of reserve and no more: the
        # transaction completes on the floor instead of being rejected, and the
        # dense phase that the reserve cannot release stays in the column.
        before = deepcopy(converted)
        report = phase.advance(converted, 0., 900., 10., 2., minimum_thickness_km=8.,
                               drainage_rate_myr=1.)
        self.assertTrue(bool(report['reserve_limited'][0]))
        self.close(report['returned_equivalent_km'], initial_restored-8.)
        self.assertGreater(float(report['withheld_equivalent_km'][0]), 0.)
        self.close(phase.restored_thickness(converted), 8.)
        self.close(phase.mass_volume(converted)+converted[inventory.RETURNED], initial_mass)
        self.assertGreater(float(converted[phase.DENSE][0]), 0.)
        self.close(converted[phase.CONVERTED], before[phase.CONVERTED])
        phase.validate(converted)
        # A column with nothing above the floor cannot return anything.
        for _ in range(3):
            report = phase.advance(converted, 0., 900., 10., 2., minimum_thickness_km=8.,
                                   drainage_rate_myr=1.)
            self.close(report['returned_equivalent_km'], 0.)
            self.close(phase.restored_thickness(converted), 8.)

    def test_returned_material_does_not_create_virtual_floor_capacity(self):
        s = state(40., 900.)
        advance(s, 1., 900., 10., 1000., minimum_thickness_km=8.)
        self.close(s[phase.DENSE], 0.)
        self.close(s[inventory.RETURNED], 24.)
        self.close(phase.restored_thickness(s), [16.])
        self.close(columns.fixed_area_removal_capacity(s), [8.])

    def test_restored_thickness_is_current_mass_equivalent_after_all_sinks(self):
        s = state(40., 900.)
        advance(s, .7, 900., 10., 5.)
        phase.erosion(s, np.array([1.]))
        s['thickness_km'] -= 1.
        self.close(phase.restored_thickness(s),
                   phase.mass_volume(s)/s['area_factor'])

    def test_near_floor_conversion_is_timestep_partition_invariant(self):
        whole, pieces = state(8.01, 900.), state(8.01, 900.)
        phase.advance(whole, 1., 900., 10., 2., minimum_thickness_km=8.)
        for _ in range(8):
            phase.advance(pieces, 1., 900., 10., .25, minimum_thickness_km=8.)
        for key in phase.FIELDS+inventory.FIELDS+('thickness_km',):
            self.close(whole[key], pieces[key])
        self.close(phase.restored_thickness(whole), [8.01])

    def test_unledgered_subfloor_column_is_rejected(self):
        s = state(8.01, 900.)
        s['thickness_km'][:] = 7.99
        with self.assertRaisesRegex(ValueError, 'phase-adjusted mechanical floor'):
            phase.validate(s)

    def test_dense_erosion_spends_ordinary_equivalent_floor_capacity(self):
        s = state(40., 900.)
        phase.advance(s, 1., 900., 10., 1000., minimum_thickness_km=8.,
                      detachment_fraction=0.)
        self.close(phase.restored_thickness(s), [40.])
        capacity = columns.fixed_area_removal_capacity(s)
        ordinary = s['thickness_km']-s[phase.DENSE]
        expected = ordinary+(32.-ordinary)*phase.RATIO
        self.close(capacity, expected)
        phase.erosion(s, capacity)
        s['thickness_km'] -= capacity
        self.close(phase.restored_thickness(s), [8.])
        self.assertGreater(float(s[phase.ERODED][0]), 0.)
        phase.validate(s)


class DenseCrustLifecycleTests(unittest.TestCase):
    def full(self, n=2):
        s = columns.initialize_structure(np.ones(n, int), np.full(n, 2000.))
        s['foundered_m'] = np.zeros(n)
        inventory.initialize(s)
        phase.initialize(s, 700.)
        return s

    def test_surface_change_uses_density_and_preserves_geometric_strain_reference(self):
        s = self.full()
        original = columns.elevation(s)
        strain_ratio = s['thickness_km']/s['reference_thickness_km']
        report = advance(s, 1., 700., 10., 10., detachment_tau_myr=1e100)
        np.testing.assert_allclose(s['thickness_km']/s['reference_thickness_km'], strain_ratio)
        np.testing.assert_allclose(columns._air_height(original)-columns._air_height(columns.elevation(s)),
                                   report['contraction_km']*1000., rtol=2e-12)
        before = columns._air_height(columns.elevation(s))
        report = advance(s, 0., 700., 10., 10.)
        np.testing.assert_allclose(columns._air_height(columns.elevation(s))-before,
                                   report['returned_km']*150./3300.*1000., rtol=2e-12)

    def test_geometric_strain_and_coalescence_conserve_dense_mass_and_heat(self):
        s = self.full()
        advance(s, np.array([.2, .8]), 700., 10., 5.)
        before = {key: s[key].copy() for key in phase.FIELDS}
        moved, _ = columns.evolve_structure(s, 2., geometric_log_area=np.array([.1, -.1]))
        for key in phase.FIELDS: np.testing.assert_array_equal(moved[key], before[key])
        weights = np.array([2., 3.])
        merged = columns.coalesce_structure(moved, weights, np.zeros(2, int))
        for key in phase.FIELDS+inventory.FIELDS:
            np.testing.assert_allclose(weights@moved[key], weights.sum()*merged[key][0], rtol=3e-13, atol=1e-9)
        np.testing.assert_allclose(columns.elevation(merged), np.average(columns.elevation(moved), weights=weights), atol=1e-9)
        phase.validate(merged)

    def test_phase_adjusted_floor_bounds_extension_and_surface_erosion(self):
        s = columns.initialize_structure(np.array([3]), np.array([0.]))
        s['thickness_km'][:] = 8.01
        s['reference_thickness_km'][:] = 8.01
        s['foundered_m'] = np.zeros(1)
        inventory.initialize(s)
        phase.initialize(s, 900.)
        phase.advance(s, 1., 900., 10., 2., minimum_thickness_km=8.)
        self.assertLess(float(s['thickness_km'][0]), 8.)
        np.testing.assert_allclose(columns.restored_thickness(s), [8.01], atol=2e-12)

        eroded, budget = columns.evolve_structure(s, 1., denudation_m=1e9)
        np.testing.assert_allclose(columns.restored_thickness(eroded), [8.], atol=2e-12)
        np.testing.assert_allclose(budget['denudation_m'], [10.], atol=2e-9)
        phase.validate(eroded)

        stretched, _ = columns.evolve_structure(s, 100., extension_km_myr=1e9)
        np.testing.assert_allclose(columns.restored_thickness(stretched), [8.], atol=2e-12)
        self.assertLess(float(stretched['thickness_km'][0]), 8.)
        np.testing.assert_allclose(stretched['thickness_km']*stretched['area_factor'],
                                   s['thickness_km']*s['area_factor'], atol=2e-12)
        phase.validate(stretched)

    def test_reference_growth_dilutes_every_extensive_phase_and_heat_inventory(self):
        s = self.full()
        advance(s, 1., 700., 10., 5.)
        old = deepcopy(s)
        selected = np.array([0, 1])
        old_area, new_area, addition = np.array([2., 3.]), np.array([4., 6.]), np.array([50., 75.])
        s['thickness_km'] = (old_area*s['thickness_km']+addition)/new_area
        inventory.grow_reference(s, selected, old_area, new_area, addition)
        for key in (phase.DENSE, phase.CONVERTED, phase.ERODED, phase.RETURNED, phase.HEAT_BASELINE):
            np.testing.assert_allclose(new_area*s[key], old_area*old[key], rtol=2e-13)
        np.testing.assert_allclose(new_area*s[phase.HEAT], old_area*old[phase.HEAT]+addition*1200., rtol=2e-13)
        phase.validate(s)

    def test_actual_arc_birth_and_growth_initialize_and_preserve_phase_state(self):
        from tests.test_native_processes import ocean_fixture
        from tests.test_arc_source_geometry import add, positions_in_cell
        import eclogite_sink
        s = ocean_fixture()
        eclogite_sink.upgrade_inventory(s)
        for column in (s.structure, s.trace_structure): phase.initialize(column, 700.)
        point = positions_in_cell(s)[:1]
        first = add(s, point, [500.])
        np.testing.assert_allclose(phase.temperature(s.structure), 1200.)
        baseline = float(s.mass@s.structure[phase.HEAT_BASELINE])
        self.assertAlmostEqual(baseline, first['added_volume_km3']*1200., places=4)
        advance(s.structure, 1., 1200., 10., 2.)
        dense = float(s.mass@s.structure[phase.DENSE])
        heat = float(s.mass@s.structure[phase.HEAT])
        second = add(s, point, [500.])
        self.assertGreater(second['grown_patches'], 0)
        self.assertAlmostEqual(float(s.mass@s.structure[phase.DENSE]), dense, places=7)
        self.assertAlmostEqual(float(s.mass@s.structure[phase.HEAT])-heat, second['added_volume_km3']*1200., places=4)
        phase.validate(s.structure)
        phase.validate(s.trace_structure)

    def test_checkpoint_preserves_phase_and_thermal_continuation_exactly(self):
        from pathlib import Path
        from types import SimpleNamespace
        import tempfile
        import checkpoint
        s = SimpleNamespace(structure=self.full(), t=12., config={}, rng=np.random.default_rng(7))
        advance(s.structure, .5, 900., 10., 5.)
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder)/'state.npz'
            checkpoint.write_checkpoint(path, s, dict(config={}), {})
            restored, _ = checkpoint.read_checkpoint(path, None, SimpleNamespace)
        for candidate in (s, restored): advance(candidate.structure, .6, 300., 10., 3.)
        for key in s.structure:
            np.testing.assert_array_equal(s.structure[key], restored.structure[key])

    def test_refinement_and_coarsening_conserve_every_phase_and_heat_reservoir(self):
        from tests.test_native_material_adaptivity import world
        from native_material_adaptivity import adapt
        import eclogite_sink
        s = world()
        eclogite_sink.upgrade_inventory(s)
        for column in (s.structure, s.trace_structure):
            phase.initialize(column, 700.)
            advance(column, .5, 800., 10., 5.)
        expected = {key: float(s.mass@s.structure[key]) for key in phase.FIELDS}
        original_count = len(s.mass)
        self.assertTrue(adapt(s, 2.))
        self.assertGreater(len(s.mass), original_count)
        for key in phase.FIELDS:
            np.testing.assert_allclose(s.mass@s.structure[key], expected[key], rtol=3e-13, atol=1e-6)
        for t in (12., 22., 32., 42.):
            s.t = t
            s.material_deformation = dict(face_weight=np.zeros(len(s.mass)),
                face_rigid=np.ones(len(s.mass), bool), face_strain=np.zeros(len(s.mass)))
            adapt(s, 2.)
        self.assertEqual(len(s.mass), original_count)
        for key in phase.FIELDS:
            np.testing.assert_allclose(s.mass@s.structure[key], expected[key], rtol=3e-13, atol=1e-6)
        phase.validate(s.structure)


def floor_column(restored_km=8., area=3.125, temperature=1200.):
    """A juvenile-arc-like column: 25 km source spread to the 8 km mass floor."""
    s = dict(thickness_km=np.atleast_1d(float(restored_km)), area_factor=np.full(1, float(area)),
             foundered_m=np.zeros(1))
    inventory.initialize(s)
    phase.initialize(s, temperature)
    return s


def ledgers(s):
    return dict(mass=phase.mass_volume(s)+s[inventory.RETURNED],
                heat_sources=s[phase.HEAT_BASELINE]+s[phase.HEAT_ADDED]+s[phase.HEAT_BATH],
                heat_sinks=s[phase.HEAT]+s[phase.HEAT_ERODED]+s[phase.HEAT_RETURNED],
                mafic=s[inventory.BASELINE]+s[inventory.ADDED],
                converted=s[phase.CONVERTED], accounted=s[phase.DENSE]+s[phase.ERODED]+s[phase.RETURNED])


class ReserveLimitedDrainageTests(unittest.TestCase):
    """Mantle return may spend only the reserve above the phase-adjusted floor.

    The 112 -> 114 Myr failure of run 20260917-0012-nonpenetration: buried
    juvenile-arc columns born exactly at the 8 km mass-equivalent floor converted
    to dense phase and then requested drainage that no reserve could pay for.
    """
    FLOOR = phase.MECHANICAL_FLOOR_KM

    def close(self, one, two, **kwargs):
        np.testing.assert_allclose(one, two, rtol=3e-12, atol=1e-10, **kwargs)

    def converted(self, s, myr=40.):
        # Grow a dense root well above the 3.4 km overstress threshold without
        # any drainage: conversion itself is floor-neutral.
        for _ in range(int(myr/5.)):
            phase.advance(s, 1., 1200., 10., 5., minimum_thickness_km=self.FLOOR)
        self.assertGreater(float(s[phase.DENSE][0]/s['area_factor'][0]), 3.5)
        return s

    def test_zero_reserve_column_withholds_all_return_and_holds_the_floor_exactly(self):
        s = self.converted(floor_column())
        self.close(phase.restored_thickness(s), self.FLOOR)
        before = deepcopy(s)
        report = phase.advance(s, 1., 1200., 10., .25, drainage_rate_myr=.9,
                               minimum_thickness_km=self.FLOOR)
        self.assertTrue(bool(report['reserve_limited'][0]))
        self.assertGreater(float(report['withheld_equivalent_km'][0]), .5)
        self.close(report['returned_km'], 0.)
        self.close(report['returned_equivalent_km'], 0.)
        self.assertGreaterEqual(float(phase.restored_thickness(s)[0]), self.FLOOR-phase.FLOOR_TOLERANCE_KM)
        self.close(phase.restored_thickness(s), self.FLOOR)
        # Withheld material stays in the column as dense phase; conversion continued.
        self.assertGreater(float(s[phase.DENSE][0]), float(before[phase.DENSE][0]))
        self.close(s[phase.RETURNED], before[phase.RETURNED])
        self.close(s[inventory.RETURNED], before[inventory.RETURNED])
        self.close(s['foundered_m'], before['foundered_m'])
        self.close(s[phase.HEAT_RETURNED], before[phase.HEAT_RETURNED])
        old, new = ledgers(before), ledgers(s)
        self.close(new['mass'], old['mass'])
        self.close(new['heat_sources'], new['heat_sinks'])
        self.close(new['converted'], new['accounted'])
        phase.validate(s)
        inventory.validate(s)

    def test_partial_reserve_is_spent_exactly_and_the_remainder_is_retained(self):
        s = self.converted(floor_column(restored_km=8.05))
        reserve = float(phase.restored_thickness(s)[0])-self.FLOOR
        self.assertGreater(reserve, .04)
        # The same column with a large reserve shows what the unconstrained law
        # would have returned in this transaction.
        unconstrained = deepcopy(s)
        unconstrained['thickness_km'] += 10.; unconstrained['area_factor'] *= 1.
        unconstrained[inventory.BASELINE] += 10.*unconstrained['area_factor']
        unconstrained[inventory.REMAINING] += 10.*unconstrained['area_factor']
        unconstrained[phase.HEAT] = phase.mass_volume(unconstrained)*phase.temperature(s)
        unconstrained[phase.HEAT_BASELINE] += unconstrained[phase.HEAT]-s[phase.HEAT]
        phase.validate(unconstrained)
        free = phase.advance(unconstrained, 1., 1200., 10., .25, drainage_rate_myr=.9,
                             minimum_thickness_km=self.FLOOR)
        self.assertFalse(bool(free['reserve_limited'][0]))
        self.assertGreater(float(free['returned_equivalent_km'][0]), reserve)
        dense_before = s[phase.DENSE].copy()
        mass_before = ledgers(s)['mass'].copy()
        report = phase.advance(s, 1., 1200., 10., .25, drainage_rate_myr=.9,
                               minimum_thickness_km=self.FLOOR)
        self.assertTrue(bool(report['reserve_limited'][0]))
        self.close(report['returned_equivalent_km'], reserve)
        self.close(phase.restored_thickness(s), self.FLOOR)
        self.assertGreater(float(report['withheld_equivalent_km'][0]), 0.)
        # Withheld return stays in the column as dense phase, mass for mass:
        # D_after = D_before + RATIO * (converted - returned) per reference area.
        area = s['area_factor']
        self.close(s[phase.DENSE], dense_before+phase.RATIO*area*(
            report['converted_equivalent_km']-report['returned_equivalent_km']))
        self.assertGreater(float(s[phase.DENSE][0]), float(dense_before[0])-phase.RATIO*reserve*float(area[0]))
        new = ledgers(s)
        self.close(new['mass'], mass_before)
        self.close(new['heat_sources'], new['heat_sinks'])
        self.close(new['converted'], new['accounted'])
        phase.validate(s)
        inventory.validate(s)

    def test_columns_with_reserve_are_unchanged_by_the_limit(self):
        for restored in (12., 40.):
            limited, free = floor_column(restored_km=restored), floor_column(restored_km=restored)
            for _ in range(4):
                a = phase.advance(limited, 1., 1200., 10., 5., drainage_rate_myr=.05,
                                  minimum_thickness_km=self.FLOOR)
                b = phase.advance(free, 1., 1200., 10., 5., drainage_rate_myr=.05)
                self.assertFalse(bool(a['reserve_limited'][0]))
                self.close(a['withheld_equivalent_km'], 0.)
                np.testing.assert_array_equal(a['returned_km'], b['returned_km'])
            for key in phase.FIELDS+inventory.FIELDS+('thickness_km', 'foundered_m'):
                np.testing.assert_array_equal(limited[key], free[key])
            self.assertGreater(float(limited[inventory.RETURNED][0]), 0.)
            self.assertGreater(float(phase.restored_thickness(limited)[0]), self.FLOOR)

    def test_reserve_limited_drainage_is_timestep_partition_invariant(self):
        whole = self.converted(floor_column(restored_km=8.02))
        pieces = deepcopy(whole)
        total = phase.advance(whole, 1., 1200., 10., 2., drainage_rate_myr=.9,
                              minimum_thickness_km=self.FLOOR)
        returned = np.zeros(1); limited = 0
        for _ in range(8):
            part = phase.advance(pieces, 1., 1200., 10., .25, drainage_rate_myr=.9,
                                 minimum_thickness_km=self.FLOOR)
            returned += part['returned_equivalent_km']; limited += int(part['reserve_limited'][0])
        for key in phase.FIELDS+inventory.FIELDS+('thickness_km', 'foundered_m'):
            self.close(whole[key], pieces[key])
        # Actual return is additive; the withheld request is a per-transaction
        # demand measure (retained material is requested again each substep).
        self.close(returned, total['returned_equivalent_km'])
        self.assertEqual(limited, 8)
        self.close(phase.restored_thickness(whole), self.FLOOR)

    def test_reserve_limit_never_creates_capacity_from_departed_mass(self):
        # Erosion and return both consume reserve; nothing below the floor may drain.
        s = self.converted(floor_column(restored_km=9.))
        removable = float(phase.restored_thickness(s)[0])-self.FLOOR
        phase.erosion(s, removable*.5)
        s['thickness_km'] -= removable*.5
        phase.advance(s, 1., 1200., 10., 1., drainage_rate_myr=.9, minimum_thickness_km=self.FLOOR)
        self.close(phase.restored_thickness(s), self.FLOOR)
        for _ in range(4):
            report = phase.advance(s, 1., 1200., 10., 1., drainage_rate_myr=.9,
                                   minimum_thickness_km=self.FLOOR)
            self.close(report['returned_equivalent_km'], 0.)
            self.assertGreaterEqual(float(phase.restored_thickness(s)[0]),
                                    self.FLOOR-phase.FLOOR_TOLERANCE_KM)
        phase.validate(s)


if __name__ == '__main__': unittest.main()
