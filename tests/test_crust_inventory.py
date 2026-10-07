"""Physical-volume oracles for the removable-crust budget and its writers."""
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest

import numpy as np

import crust_inventory as inventory
import crustal_structure as columns
import eclogite_sink as sink
import structure_engine as structure


def column(thickness=40., area=1.):
    state = structure._new(np.ones(1, np.uint8), np.zeros(1))
    state['thickness_km'][:] = thickness
    state['area_factor'][:] = area
    inventory.initialize(state)
    return state


def remove(state, thickness):
    # Large duration consumes the requested eligible thickness exactly.
    return sink.apply(state, np.array([thickness]), 20000.)[0][0]


class InventoryTests(unittest.TestCase):
    def test_reported_compression_case_removes_60_percent_of_original_volume(self):
        state = column()
        self.assertEqual(remove(state, 10.), 10.)
        state, _ = columns.evolve_structure(state, 1., geometric_log_area=-np.log(2.))
        state['foundered_m'] = np.array([10000.])  # historical metres, not a volume allowance
        self.assertAlmostEqual(state['thickness_km'][0], 60.)
        second = remove(state, 60.)*state['area_factor'][0]
        self.assertAlmostEqual(10.+second, 24.)
        self.assertAlmostEqual(state['thickness_km'][0]*state['area_factor'][0], 16.)
        self.assertEqual(state[inventory.REMAINING][0], 0.)
        inventory.validate(state)

    def test_same_volume_budget_under_arbitrary_area_history(self):
        state = column()
        volume_removed = 0.
        for area in (1.2, .8, .55, .9, .7, .5, .4):
            retained = state['thickness_km'][0]*state['area_factor'][0]
            state['area_factor'][:] = area
            state['thickness_km'][:] = retained/area
            volume_removed += remove(state, 3.)*area
        volume_removed += remove(state, 100.)*state['area_factor'][0]
        self.assertAlmostEqual(volume_removed, 24.)
        self.assertAlmostEqual(state[inventory.RETURNED][0], 24.)
        inventory.validate(state)

    def test_erosion_exports_the_current_mixture_and_cannot_replenish_cap(self):
        state = column()
        remove(state, 10.)  # 14 removable among 30 retained cubic km
        evolved, _ = columns.evolve_structure(state, 1., denudation_m=6000.)
        evolved['foundered_m'] = state['foundered_m'].copy()
        self.assertAlmostEqual(evolved[inventory.ERODED][0], 2.8)
        self.assertAlmostEqual(evolved[inventory.REMAINING][0], 11.2)
        remove(evolved, 100.)
        self.assertAlmostEqual(evolved[inventory.RETURNED][0], 21.2)
        inventory.validate(evolved)

    def test_surface_magma_source_adds_only_actual_admitted_volume(self):
        state = column(74., .5)
        baseline = state[inventory.REMAINING].copy()
        _, added = structure._surface_change(state, 10000., conserve_volume=False)
        self.assertEqual(added[0], .5)  # thickness bound admits one km, half km2
        self.assertAlmostEqual(state[inventory.REMAINING][0]-baseline[0], .3)
        self.assertAlmostEqual(state[inventory.ADDED][0], .3)
        before = deepcopy(state)
        structure._surface_change(state, -100., conserve_volume=True)
        for name in inventory.FIELDS:
            np.testing.assert_array_equal(state[name], before[name])

    def test_coalescence_preserves_every_extensive_ledger_with_unequal_weights(self):
        one, two = column(40., .7), column(35., 1.3)
        remove(one, 10.); remove(two, 5.)
        state = {name: np.r_[one[name], two[name]] for name in one}
        weights = np.array([3., 11.])
        merged = columns.coalesce_structure(state, weights, np.array([0, 0]))
        for name in inventory.FIELDS:
            self.assertAlmostEqual(float(weights@state[name]), 14.*merged[name][0])
        inventory.validate(merged)

    def test_partial_corrupt_and_overfull_ledgers_are_rejected(self):
        state = column()
        for name in inventory.FIELDS:
            broken = deepcopy(state); del broken[name]
            with self.assertRaises(ValueError): inventory.validate(broken)
            broken = deepcopy(state); broken[name][0] = np.nan
            with self.assertRaises(ValueError): inventory.validate(broken)
        broken = deepcopy(state); broken[inventory.REMAINING] += 1.
        with self.assertRaises(ValueError): inventory.validate(broken)
        broken = deepcopy(state)
        broken[inventory.BASELINE][:] = broken[inventory.REMAINING][:] = 41.
        with self.assertRaises(ValueError): inventory.validate(broken)


class LifecycleTests(unittest.TestCase):
    def test_explicit_migration_is_atomic_idempotent_and_preserves_old_allowance(self):
        from tests.test_eclogite_sink import stack
        s = stack(); s.structure['foundered_m'][0] = 10000.
        old = deepcopy(s.structure)
        clock = s.parcel_root_age_myr.copy()
        report = sink.upgrade_inventory(s)
        expected = np.maximum(.6*old['thickness_km']-.4*old['foundered_m']/1000., 0.)*old['area_factor']
        np.testing.assert_array_equal(s.structure[inventory.REMAINING], expected)
        for name in old: np.testing.assert_array_equal(s.structure[name], old[name])
        np.testing.assert_array_equal(s.parcel_root_age_myr, clock)
        self.assertFalse(report['historical_volume_reconstructed'])
        self.assertEqual(sink.upgrade_inventory(s), report)
        broken = stack(); broken.trace_structure['foundered_m'][0] = -1.
        with self.assertRaises(ValueError): sink.upgrade_inventory(broken)
        self.assertEqual(broken.foundering_version, 1)
        self.assertFalse(inventory.present(broken.structure))

    def test_no_silent_initialization_of_loaded_v2_and_legacy_law_remains(self):
        from tests.test_eclogite_sink import stack
        s = stack(); sink.ensure_fields(s)
        self.assertFalse(inventory.present(s.structure))
        s.foundering_version = 2
        with self.assertRaises(ValueError): sink.ensure_fields(s)
        state = column()
        for name in inventory.FIELDS: del state[name]
        remove(state, 10.)
        state['thickness_km'] *= 2.; state['area_factor'] /= 2.
        self.assertAlmostEqual(10.+remove(state, 100.)*.5, 26.)

    def test_real_deform_closes_mantle_budget_and_keeps_marker_inventory(self):
        from tests.test_eclogite_sink import stack
        s = stack(); sink.upgrade_inventory(s)
        initial = s.mass@s.structure[inventory.BASELINE]
        for step in range(4):
            s.t = 2.*(step+1)
            structure.deform(s, np.zeros(s.n), np.zeros(s.n), np.zeros(s.n), 2.)
        state = s.structure
        self.assertAlmostEqual(float(s.mass@state[inventory.RETURNED])/s.mantle_return_km3['total'], 1., places=12)
        self.assertAlmostEqual(float(s.mass@(state[inventory.ERODED]+state[inventory.RETURNED]+state[inventory.REMAINING]))/initial, 1., places=12)
        for name in inventory.FIELDS:
            np.testing.assert_allclose(s.trace_structure[name], state[name], rtol=1e-12)
        inventory.validate(state)

    def test_real_adaptivity_refines_and_coarsens_without_inventory_drift(self):
        from tests.test_native_material_adaptivity import world
        from native_material_adaptivity import adapt
        s = world(); sink.upgrade_inventory(s)
        original = {name: float(s.mass@s.structure[name]) for name in inventory.FIELDS}
        self.assertTrue(adapt(s, 2.))
        for t in (12., 22., 32., 42.):
            s.t = t
            s.material_deformation = dict(face_weight=np.zeros(len(s.mass)),
                face_rigid=np.ones(len(s.mass), bool), face_strain=np.zeros(len(s.mass)))
            changed = adapt(s, 2.)
        self.assertTrue(changed)
        for name in inventory.FIELDS:
            self.assertAlmostEqual(float(s.mass@s.structure[name])/max(original[name], 1.), original[name]/max(original[name], 1.), places=12)
        inventory.validate(s.structure)

    def test_deform_erosion_exports_each_column_and_marker_mixture(self):
        from tests.test_eclogite_sink import stack
        s = stack(erosion=1.); sink.upgrade_inventory(s)
        s.t = 2.
        before = deepcopy(s.structure)
        before_trace = deepcopy(s.trace_structure)
        structure.deform(s, np.zeros(s.n), np.zeros(s.n), np.zeros(s.n), 2.)
        for state, old in ((s.structure, before), (s.trace_structure, before_trace)):
            # Markers are points and faces integrate exposed fractions, so their
            # erosion can differ. Each must export its own pre-erosion mixture.
            expected = old[inventory.REMAINING]*(state['denudation_m']/1000.)/old['thickness_km']
            np.testing.assert_allclose(state[inventory.ERODED], expected, atol=1e-12)
            inventory.validate(state)

    def test_selected_replenishment_preserves_and_updates_optional_inventory(self):
        from tests.test_native_material_adaptivity import world
        s = world(); sink.upgrade_inventory(s)
        s.trace_adjustment_m = np.zeros(len(s.trace_patch))
        for state in (s.structure, s.trace_structure):
            state['lip_heat_m'] = np.full(len(s.mass), 25.)
        before = deepcopy(s.structure)
        structure.replenish(s, np.array([1, 4]), np.array([1, 4]), 1000.)
        added = (s.structure['thickness_km']-before['thickness_km'])*before['area_factor']
        np.testing.assert_allclose(s.structure[inventory.ADDED], .6*added)
        np.testing.assert_array_equal(s.structure['lip_heat_m'], before['lip_heat_m'])
        self.assertGreater(s.structure[inventory.ADDED][1], 0.)
        inventory.validate(s.structure)

    def test_lip_addition_supplies_actual_admitted_removable_volume(self):
        from tests.test_lip_events import world, advance
        s = world(); sink.upgrade_inventory(s)
        baseline = s.mass@s.structure[inventory.REMAINING]
        advance(s, 2.)
        self.assertAlmostEqual(float(s.mass@s.structure[inventory.ADDED]), 60000., places=6)
        self.assertAlmostEqual(float(s.mass@s.structure[inventory.REMAINING])-baseline, 60000., places=5)
        np.testing.assert_array_equal(s.structure[inventory.REMAINING], s.trace_structure[inventory.REMAINING])

    def test_actual_arc_birth_and_growth_conserve_source_inventory(self):
        from test_native_processes import ocean_fixture
        from tests.test_arc_source_geometry import add, positions_in_cell
        s = ocean_fixture(); sink.upgrade_inventory(s)
        point = positions_in_cell(s)[:1]
        first = add(s, point, [500.])
        baseline = float(s.mass@s.structure[inventory.BASELINE])
        self.assertAlmostEqual(baseline, .6*first['added_volume_km3'], places=6)
        second = add(s, point, [500.])
        self.assertGreater(second['grown_patches'], 0)
        self.assertAlmostEqual(float(s.mass@s.structure[inventory.BASELINE]), baseline, places=6)
        self.assertAlmostEqual(float(s.mass@s.structure[inventory.ADDED]), .6*second['added_volume_km3'], places=6)
        self.assertAlmostEqual(float(s.mass@s.structure[inventory.REMAINING]), 15000., places=6)
        sink.ensure_fields(s)  # runtime pads the newborn root clocks before deformation
        sink.validate_alignment(s)

    def test_checkpoint_and_frame_preserve_inventory_and_reject_corruption(self):
        import checkpoint
        s = SimpleNamespace(structure=column(), trace_structure=column(),
                            mass=np.ones(1), trace_patch=np.array([1]), t=0., config={},
                            rng=np.random.default_rng(21))
        for state in (s.structure, s.trace_structure):
            for name in inventory.FIELDS: del state[name]
        sink.upgrade_inventory(s); remove(s.structure, 10.)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)/'state.npz'
            checkpoint.write_checkpoint(path, s, dict(config=s.config), {})
            restored, _ = checkpoint.read_checkpoint(path, {}, SimpleNamespace)
        remove(s.structure, 3.); remove(restored.structure, 3.)
        for name in inventory.FIELDS:
            np.testing.assert_array_equal(restored.structure[name], s.structure[name])
        frame = dict(material_faces=np.zeros((1, 3), int), **sink.snapshot_fields(s))
        sink.validate_frame(frame)
        frame['material_'+inventory.REMAINING][0] += 1.
        with self.assertRaises(ValueError): sink.validate_frame(frame)


if __name__ == '__main__':
    unittest.main()
