"""Independent relief-decay oracles and explicit fresh-profile safeguards."""
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

import collision_surface
import crustal_structure as columns
import crust_inventory
import dense_crust
import structure_engine as structure
from checkpoint import read_checkpoint, write_checkpoint
from tests.test_collision_surface_acceptance import triangle, triangle_area, world as stacked_world
from tests.test_structure_engine import world


def evolve(state, kinds, dt, *, exposure=1., support=0., version=1):
    s = SimpleNamespace(config=dict(erosion=1.), erosion_relief_version=version)
    height = columns.elevation(state)+support
    request = structure._erosion_request(s, state, np.asarray(kinds), height, dt,
                                         np.broadcast_to(exposure, height.shape))
    return columns.evolve_structure(state, dt, denudation_m=request)


class ReviewedReliefTests(unittest.TestCase):
    def test_mobile_and_craton_timescales_are_net_surface_decay(self):
        state = columns.initialize_structure([1, 2], [1000., 1000.])
        result, budget = evolve(state, [1, 2], 180.)
        expected = 1000.*np.exp(-180./np.array([180., 350.]))
        np.testing.assert_allclose(columns.elevation(result), expected, atol=1e-11)
        # Independent Airy density ratio: gross removal includes rebound.
        np.testing.assert_allclose(budget['denudation_m'], (1000.-expected)*3300./500., atol=1e-10)
        np.testing.assert_allclose(budget['denudation_m']-budget['rebound_m'], 1000.-expected, atol=1e-10)

    def test_constant_partial_exposure_is_timestep_partition_invariant(self):
        state = columns.initialize_structure([1, 2], [1000., 1000.])
        whole, _ = evolve(state, [1, 2], 100., exposure=.3)
        pieces = deepcopy(state)
        for _ in range(50):
            pieces, _ = evolve(pieces, [1, 2], 2., exposure=.3)
        expected = 1000.*np.exp(-100.*.3/np.array([180., 350.]))
        np.testing.assert_allclose(columns.elevation(whole), expected, atol=2e-11)
        np.testing.assert_allclose(columns.elevation(pieces), expected, atol=2e-11)

    def test_water_loaded_inverse_handles_crossing_the_sea_datum(self):
        state = columns.initialize_structure([1], [20.])
        removal = columns.denudation_for_net_loss(state, [50.])
        # 20 m above sea level and 30 m water-loaded below: two slopes.
        expected = (20.+30.*2270./3300.)*3300./500.
        np.testing.assert_allclose(removal, [expected], atol=1e-11)
        result, budget = columns.evolve_structure(state, 1., denudation_m=removal)
        np.testing.assert_allclose(columns.elevation(result), [-30.], atol=1e-11)
        np.testing.assert_allclose(budget['net_erosion_loss_m'], [50.], atol=1e-11)

    def test_supported_submerged_column_uses_surface_height_and_correct_rebound(self):
        state = columns.initialize_structure([1], [-1000.])
        result, budget = evolve(state, [1], 180., support=2000.)
        expected_loss = 1000.*(1.-np.exp(-1.))
        np.testing.assert_allclose(columns.elevation(result)+2000., [1000.-expected_loss], atol=1e-10)
        np.testing.assert_allclose(budget['denudation_m'], [expected_loss*2270./500.], atol=1e-10)

    def test_foreland_offset_does_not_change_the_water_loading_datum(self):
        state = columns.initialize_structure([1], [200.])
        state['foreland_m'][:] = 300.
        removal = columns.denudation_for_net_loss(state, [50.])
        # Actual height is -100 m, but the pre-foreland Airy column is emerged.
        np.testing.assert_allclose(removal, [50.*3300./500.], atol=1e-11)

    def test_submerged_surface_and_buried_columns_request_no_erosion(self):
        for height, exposure in ((-500., 1.), (1000., 0.)):
            state = columns.initialize_structure([1], [height])
            result, budget = evolve(state, [1], 180., exposure=exposure)
            np.testing.assert_array_equal(result['thickness_km'], state['thickness_km'])
            np.testing.assert_array_equal(budget['denudation_m'], [0.])

    def test_floor_limits_real_rock_and_closes_volume_inventory_and_height(self):
        state = columns.initialize_structure([1], [10000.])
        state['thickness_km'][:] = 8.2
        state['reference_thickness_km'][:] = 8.2
        state['area_factor'][:] = 3.
        crust_inventory.initialize(state)
        result, budget = evolve(state, [1], 180.)
        np.testing.assert_allclose(result['thickness_km'], [8.], atol=1e-14)
        np.testing.assert_allclose(budget['denudation_m'], [200.], atol=1e-11)
        np.testing.assert_allclose(budget['removed_volume_km_per_reference_km2'], [.6], atol=1e-13)
        crust_inventory.validate(result)
        np.testing.assert_allclose(state[crust_inventory.REMAINING]-result[crust_inventory.REMAINING],
                                   result[crust_inventory.ERODED], atol=1e-13)
        np.testing.assert_allclose(budget['denudation_m']-budget['rebound_m'],
                                   10000.-columns.elevation(result), atol=1e-10)

    def test_unversioned_and_version_zero_preserve_old_gross_removal(self):
        kinds = np.array([1, 2])
        state = columns.initialize_structure(kinds, [1000., 2000.])
        exposure = np.array([.25, .75])
        expected = np.array([1000., 2000.])*(-np.expm1(-2./np.array([180., 350.])))*exposure
        for s in (SimpleNamespace(config=dict(erosion=1.)),
                  SimpleNamespace(config=dict(erosion=1.), erosion_relief_version=0)):
            result = structure._erosion_request(s, state, kinds, columns.elevation(state), 2., exposure)
            np.testing.assert_array_equal(result, expected)

    def test_invalid_inverse_and_unknown_versions_reject(self):
        state = columns.initialize_structure([1], [1000.])
        for loss in (-1., np.inf):
            with self.assertRaises(ValueError):
                columns.denudation_for_net_loss(state, loss)
        for version in (True, 2):
            with self.assertRaises(ValueError):
                evolve(state, [1], 2., version=version)

    def test_retained_dense_phase_cannot_silently_use_ordinary_erosion_inverse(self):
        state = columns.initialize_structure([1], [1000.])
        crust_inventory.initialize(state)
        dense_crust.initialize(state, 700.)
        # Merely having the zero phase inventory is harmless.
        np.testing.assert_allclose(columns.denudation_for_net_loss(state, [100.]), [660.], atol=1e-10)
        dense_crust.advance(state, 1., 700., 10., 2.)
        before = deepcopy(state)
        with self.assertRaisesRegex(ValueError, 'retained dense crust'):
            columns.denudation_for_net_loss(state, [100.])
        for name in before:
            np.testing.assert_array_equal(state[name], before[name])

    def test_explicitly_disabled_foreland_cannot_fall_back_to_legacy_trough(self):
        for version in (0, 1):
            s = SimpleNamespace(foreland_loading_version=version, foreland_loading_enabled=False)
            with patch('foreland_loading.prepare_native', side_effect=AssertionError('regional called')):
                context = structure._foreland_inputs(s)
            self.assertIsNone(context)
            self.assertEqual(s.foreland_loading_diagnostics['state'], 'disabled')
            np.testing.assert_array_equal(structure._target(np.zeros((2, 3)), [0, 0],
                np.array([1, 2]), context), [0., 0.])

    def test_missing_foreland_switch_preserves_legacy_trough(self):
        s = SimpleNamespace(bcode=np.array([4]), normal_speed=np.array([-1.]),
            crust=np.array([1, 1]), land_relief=np.array([3000., 2000.]),
            ba=np.array([0]), bb=np.array([1]), bmid=np.array([[1., 0., 0.]]),
            bn=np.array([[0., 1., 0.]]), bl=np.array([100.]), bp=np.array([0]), bq=np.array([1]))
        context = structure._foreland_inputs(s)
        self.assertIsInstance(context, tuple)
        np.testing.assert_array_equal(context[-1], [3220.])

    def test_real_engine_parcel_and_trace_ledgers_close_with_reviewed_law(self):
        s = world(erosion=1.)
        s.erosion_relief_version = 1
        s.foreland_loading_enabled = False
        start = s.trace_relief_m.copy()
        old_height = columns.elevation(s.trace_structure)
        mass = s.mass.copy()
        zero = np.zeros(s.n)
        structure.deform(s, zero, zero, zero, 2.)
        expected_loss = np.maximum(old_height, 0.)*(-np.expm1(-2./np.where(s.trace_kind == 2, 350., 180.)))
        np.testing.assert_allclose(s.trace_erosion_m, expected_loss, atol=2e-10)
        np.testing.assert_allclose(s.trace_relief_m-start,
            s.trace_uplift_m-s.trace_extension_m-s.trace_erosion_m+s.trace_adjustment_m, atol=2e-10)
        np.testing.assert_allclose(s.trace_structure['denudation_m']-s.trace_structure['rebound_m'],
                                   s.trace_erosion_m, atol=2e-10)
        np.testing.assert_array_equal(s.mass, mass)
        np.testing.assert_array_equal(s.structure['foreland_m'], 0.)
        np.testing.assert_array_equal(s.trace_structure['foreland_m'], 0.)

    def test_combined_column_diagnostic_reports_stack_without_clipping_material(self):
        tri = triangle()
        s = stacked_world([(tri, 50., 500., False), (tri, 40., 1000., False)])
        before = deepcopy(s.structure)
        report = collision_surface.refresh(s)
        self.assertAlmostEqual(report['maximum_root_averaged_combined_column_km'], 90., places=9)
        self.assertIn('not a pointwise maximum', report['combined_column_interpretation'])
        self.assertFalse(report['stack_collapse_resolved'])
        for name in before:
            np.testing.assert_array_equal(s.structure[name], before[name])

    def test_checkpoint_preserves_explicit_choices_and_continues_exactly(self):
        s = world(erosion=1.)
        s.erosion_relief_version = 1
        s.foreland_loading_enabled = False
        s.foreland_loading_version = 1
        zero = np.zeros(s.n)
        structure.deform(s, zero, zero, zero, 2.)
        compatibility = dict(engine_sha256='reviewed-relief-fixture',
                             auxiliary_sources_sha256={}, numpy_version=np.__version__)
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder)/'state.npz'
            write_checkpoint(path, s, dict(config=s.config, time_myr=s.t), compatibility)
            restored, _ = read_checkpoint(path, compatibility, type(s))
        self.assertEqual(restored.erosion_relief_version, 1)
        self.assertFalse(restored.foreland_loading_enabled)
        self.assertEqual(restored.foreland_loading_version, 1)
        for candidate in (s, restored):
            structure.deform(candidate, zero, zero, zero, 2.)
        for name in ('structure', 'trace_structure'):
            for field in getattr(s, name):
                np.testing.assert_array_equal(getattr(s, name)[field], getattr(restored, name)[field])

    def test_combined_diagnostic_averages_own_column_as_well_as_buried_load(self):
        tri = triangle()
        s = stacked_world([(tri, 35., 500., True), (tri, 20., 1000., False)])
        top = s.parcel_plate == 0
        s.structure['thickness_km'][top] = [20., 40., 60., 70.]
        surface = s.material_surface
        areas = np.array([triangle_area(surface['vertices'][f]) for f in surface['faces'][top]])
        expected = np.average([20., 40., 60., 70.], weights=areas)+20.
        report = collision_surface.refresh(s)
        self.assertAlmostEqual(report['maximum_root_averaged_combined_column_km'], expected, places=9)
        self.assertLess(expected, 90.)


if __name__ == '__main__':
    unittest.main()
