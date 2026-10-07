"""Resolved interior strain changes columns once and closes height ledgers."""
from copy import deepcopy
from types import SimpleNamespace
import unittest
from unittest.mock import Mock

import numpy as np

import crustal_structure as columns
import structure_engine as structure


def material():
    s = SimpleNamespace(kind=np.array([1, 2, 3, 1]),
                        relief=np.array([100., 500., 300., -2500.]), t=42.)
    s.trace_kind = s.kind.copy()
    s.trace_relief_m = s.relief.copy()
    structure.initialize_parcels(s)
    structure.initialize_traces(s, np.arange(len(s.kind)))
    for field in ('uplift', 'extension', 'erosion', 'adjustment', 'ridge_uplift'):
        setattr(s, 'trace_'+field+'_m', np.zeros(len(s.kind)))
    s._remember_rift_extension = Mock()
    s.mass = np.array([200., 500., 100., 900.])
    return s


class InteriorStructureTests(unittest.TestCase):
    def assert_height_and_ledger(self, s, birth):
        for trace in (False, True):
            state = s.trace_structure if trace else s.structure
            kinds = s.trace_kind if trace else s.kind
            relief = s.trace_relief_m if trace else s.relief
            np.testing.assert_allclose(columns.elevation(state), structure.material_height(kinds, relief), atol=1e-10)
        expected = birth+s.trace_uplift_m-s.trace_extension_m-s.trace_erosion_m+s.trace_adjustment_m
        np.testing.assert_allclose(s.trace_relief_m, expected, atol=1e-10)

    def test_logarithmic_extension_conserves_column_volume_and_closes_height(self):
        s = material()
        birth = s.trace_relief_m.copy()
        before_t = s.structure['thickness_km'].copy()
        before_area = s.structure['area_factor'].copy()
        mass = s.mass.copy()
        strain = np.array([.1, .05, .03, .2])
        budget = structure.extend_interior(s, strain, strain)
        np.testing.assert_allclose(s.structure['thickness_km'], before_t*np.exp(-strain), atol=1e-12)
        np.testing.assert_allclose(s.structure['area_factor'], before_area*np.exp(strain), atol=1e-12)
        np.testing.assert_allclose(s.structure['thickness_km']*s.structure['area_factor'], before_t*before_area, atol=1e-12)
        np.testing.assert_allclose(s.structure['rift_heat_m'], columns.MAX_RIFT_HEAT_M*(-np.expm1(-1.4*strain)), atol=1e-12)
        np.testing.assert_allclose(budget['parcel']['actual_strain'], strain, atol=1e-14)
        self.assertTrue(np.all(budget['trace']['extension_loss_m'] > 0))
        self.assertTrue(np.all(budget['trace']['thermal_uplift_m'] > 0))
        np.testing.assert_array_equal(s.mass, mass)
        self.assertEqual(s.t, 42.)
        self.assert_height_and_ledger(s, birth)

    def test_calls_existing_rift_memory_with_only_mechanical_lowering(self):
        s = material()
        p = np.array([.04, 0., 0., .09])
        t = np.array([0., .08, .12, .01])
        budget = structure.extend_interior(s, p, t)
        self.assertEqual(s._remember_rift_extension.call_count, 2)
        first, second = s._remember_rift_extension.call_args_list
        self.assertEqual(first.args[0], '')
        self.assertEqual(second.args[0], 'trace_')
        np.testing.assert_array_equal(first.args[1], budget['parcel']['extension_loss_m'])
        np.testing.assert_array_equal(second.args[1], budget['trace']['extension_loss_m'])
        np.testing.assert_array_equal(s.trace_extension_m, budget['trace']['extension_loss_m'])
        np.testing.assert_array_equal(s.trace_uplift_m, budget['trace']['thermal_uplift_m'])

    def test_no_second_cooling_erosion_foreland_or_clock_update(self):
        s = material()
        for state in (s.structure, s.trace_structure):
            before = columns.elevation(state)
            state['rift_heat_m'][:] = [0., 800., 300., 100.]
            state['rift_age_myr'][:] = [-1., 75., 12., 400.]
            state['foreland_m'][:] = [0., 150., 400., 900.]
            state['denudation_m'][:] = 200.
            state['rebound_m'][:] = 160.
            state['erosion_rate_m_myr'][:] = 3.
            relief = s.relief if state is s.structure else s.trace_relief_m
            relief += columns.elevation(state)-before
        old = deepcopy(s)
        birth = s.trace_relief_m.copy()
        structure.extend_interior(s, np.array([.1, 0., 0., 0.]), np.array([.1, 0., 0., 0.]))
        for state_name in ('structure', 'trace_structure'):
            state, prior = getattr(s, state_name), getattr(old, state_name)
            for key in ('foreland_m', 'denudation_m', 'rebound_m', 'erosion_rate_m_myr',
                        'thermal_subsidence_m', 'foreland_subsidence_m', 'foreland_rebound_m',
                        'magmatic_uplift_m', 'added_volume_km_per_reference_km2'):
                np.testing.assert_array_equal(state[key], prior[key], err_msg=key)
            for key in structure.FIELDS:
                np.testing.assert_array_equal(state[key][1:], prior[key][1:], err_msg=key)
            np.testing.assert_array_equal(state['rift_age_myr'], [0., 75., 12., 400.])
        self.assertEqual(s.t, old.t)
        np.testing.assert_array_equal(s.trace_erosion_m, old.trace_erosion_m)
        np.testing.assert_array_equal(s.trace_adjustment_m, old.trace_adjustment_m)
        self.assert_height_and_ledger(s, birth)

    def test_realized_thinning_bounds_heat_and_no_repeated_loading_at_floor(self):
        s = material()
        birth = s.trace_relief_m.copy()
        original_volume = s.structure['thickness_km']*s.structure['area_factor']
        budget = structure.extend_interior(s, np.full(4, 1e300), np.full(4, 1e300))
        np.testing.assert_array_equal(s.structure['thickness_km'], columns.MIN_THICKNESS_KM)
        np.testing.assert_allclose(s.structure['thickness_km']*s.structure['area_factor'], original_volume, atol=1e-12)
        self.assertTrue(np.all(s.structure['rift_heat_m'] < columns.MAX_RIFT_HEAT_M))
        self.assertTrue(np.all(np.isfinite(budget['parcel']['actual_strain'])))
        for state in (s.structure, s.trace_structure):
            state['rift_age_myr'][:] = 35.
        previous = deepcopy(s)
        second = structure.extend_interior(s, np.ones(4), np.ones(4))
        for name in ('structure', 'trace_structure'):
            for key in structure.FIELDS:
                np.testing.assert_array_equal(getattr(s, name)[key], getattr(previous, name)[key], err_msg=key)
        np.testing.assert_array_equal(second['trace']['actual_strain'], 0.)
        np.testing.assert_array_equal(second['trace']['extension_loss_m'], 0.)
        np.testing.assert_array_equal(second['trace']['thermal_uplift_m'], 0.)
        self.assert_height_and_ledger(s, birth)

    def test_partitioning_strain_increment_preserves_final_columns_and_height(self):
        once, split = material(), material()
        birth = once.trace_relief_m.copy()
        strain = np.array([.3, .15, .04, .25])
        structure.extend_interior(once, strain, strain)
        structure.extend_interior(split, strain*.4, strain*.4)
        structure.extend_interior(split, strain*.6, strain*.6)
        for name in ('structure', 'trace_structure'):
            for key in columns.STATE_FIELDS:
                np.testing.assert_allclose(getattr(once, name)[key], getattr(split, name)[key], atol=2e-12, err_msg=key)
        np.testing.assert_allclose(once.relief, split.relief, atol=2e-12)
        self.assert_height_and_ledger(once, birth)
        self.assert_height_and_ledger(split, birth)

    def test_invalid_trace_input_does_not_partly_mutate_parcels(self):
        for invalid in (np.ones(3), np.array([0., 0., -1., 0.]), np.full(4, np.nan)):
            s = material()
            old = deepcopy(s)
            with self.subTest(invalid=invalid), self.assertRaises(ValueError):
                structure.extend_interior(s, np.full(4, .1), invalid)
            np.testing.assert_array_equal(s.relief, old.relief)
            for key in structure.FIELDS:
                np.testing.assert_array_equal(s.structure[key], old.structure[key], err_msg=key)
            s._remember_rift_extension.assert_not_called()

    def test_zero_increment_preserves_every_existing_column_and_process_counter(self):
        s = material()
        old = deepcopy(s)
        result = structure.extend_interior(s, np.zeros(4), np.zeros(4))
        for name in ('structure', 'trace_structure'):
            for key in structure.FIELDS:
                np.testing.assert_array_equal(getattr(s, name)[key], getattr(old, name)[key], err_msg=key)
        np.testing.assert_array_equal(s.trace_relief_m, old.trace_relief_m)
        for side in ('parcel', 'trace'):
            for value in result[side].values():
                np.testing.assert_array_equal(value, 0.)


if __name__ == '__main__':
    unittest.main()
