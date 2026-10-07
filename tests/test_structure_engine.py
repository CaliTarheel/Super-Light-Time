"""Real material arrays keep compensated height, process and restart budgets."""
from copy import deepcopy
from pathlib import Path
import tempfile
import unittest

import numpy as np

from checkpoint import read_checkpoint, write_checkpoint
import crustal_structure as columns
import structure_engine as structure
from raster_engine import Simulation
from tests.test_rift_inversion_engine import compressional_world
from tests.test_ridge_engine import prescribed_world


def attach(s):
    """Also works before the core calls the new lifecycle hooks itself."""
    if not hasattr(s, 'structure'):
        structure.initialize_parcels(s)
    if not hasattr(s, 'trace_structure'):
        initial_count = np.count_nonzero(s.initial_crust > 0)
        chosen = np.searchsorted(s.parcel_patch[:initial_count], s.trace_patch)
        structure.initialize_traces(s, chosen)
    return s


def world(erosion=0.):
    s = attach(Simulation(dict(width=64, height=32, plate_count=4, seed=12, erosion=erosion)))
    s.bcode[:] = 3
    s.ridge_episodes = []
    return s


def ledger(s):
    return s.trace_uplift_m-s.trace_extension_m-s.trace_erosion_m+s.trace_adjustment_m


class StructureEngineTests(unittest.TestCase):
    def aligned(self, s):
        for trace in (False, True):
            state = s.trace_structure if trace else s.structure
            kind = s.trace_kind if trace else s.kind
            relief = s.trace_relief_m if trace else s.relief
            self.assertEqual(set(state), set(structure.FIELDS))
            for field in state.values():
                self.assertEqual(field.shape, relief.shape)
                self.assertTrue(np.isfinite(field).all())
            np.testing.assert_allclose(columns.elevation(state), structure.material_height(kind, relief), atol=2e-10)

    def test_initial_columns_match_actual_parcels_and_selected_traces(self):
        s = world()
        self.aligned(s)
        self.assertTrue(np.any(s.kind == 2))
        np.testing.assert_array_equal(s.structure['thickness_km'][s.kind == 2], 42.)
        for name in structure.DIAGNOSTIC_FIELDS:
            self.assertFalse(np.any(s.structure[name]))
            self.assertFalse(np.any(s.trace_structure[name]))
        np.testing.assert_array_equal(s.structure['rift_age_myr'], -1.)

    def test_strain_cooling_erosion_and_trace_budgets_close_without_area_loss(self):
        s = world(erosion=1.)
        mass, patch, initial = s.mass.copy(), s.parcel_patch.copy(), s.trace_relief_m.copy()
        for index, (short, extension, volcanic) in enumerate(((0., 30., 0.), (0., 0., 0.), (18., 0., 12.))):
            s.t += 2.
            structure.deform(s, np.full(s.n, short), np.full(s.n, extension), np.full(s.n, volcanic), 2.)
            self.aligned(s)
            np.testing.assert_allclose(s.trace_relief_m-initial, ledger(s), atol=2e-10)
        self.assertTrue(np.any(s.trace_structure['thermal_subsidence_m'] > 0))
        self.assertTrue(np.any(s.trace_structure['denudation_m'] > s.trace_erosion_m))
        np.testing.assert_allclose(s.trace_structure['denudation_m']-s.trace_structure['rebound_m'],
                                   s.trace_erosion_m, atol=2e-10)
        self.assertTrue(np.all(s.trace_rift_extension_m <= s.trace_extension_m+1e-10))
        np.testing.assert_array_equal(s.mass, mass)
        np.testing.assert_array_equal(s.parcel_patch, patch)

    def test_net_erosion_fallback_rate_removes_erosion_and_rebound_together(self):
        eroded = world(erosion=1.)
        uneroded = deepcopy(eroded)
        uneroded.config['erosion'] = 0.
        start = eroded.trace_relief_m.copy()
        zero = np.zeros(eroded.n)
        for s in (eroded, uneroded):
            s.t += 2.
            structure.deform(s, zero, zero, np.full(s.n, 25.), 2.)
        # goSPL's geometric delta plus NET erosion recovers the same forcing as
        # the no-erosion twin; rebound is not counted again as tectonic uplift.
        np.testing.assert_allclose(eroded.trace_relief_m-start+eroded.trace_erosion_m,
                                   uneroded.trace_relief_m-start, atol=2e-10)
        np.testing.assert_allclose(eroded.trace_structure['erosion_rate_m_myr']*2.,
                                   eroded.trace_erosion_m, atol=2e-10)
        np.testing.assert_allclose(eroded.trace_uplift_m, uneroded.trace_uplift_m, atol=2e-10)
        self.assertTrue(np.all(eroded.trace_structure['rebound_m'] > 0))

    def test_inversion_consumes_only_realized_bounded_thickening(self):
        s = attach(compressional_world())
        baseline = deepcopy(s)
        baseline.rift_extension_m[:] = 0
        baseline.trace_rift_extension_m[:] = 0
        initial = s.trace_relief_m.copy()
        volume = s.structure['thickness_km']*s.structure['area_factor']
        for candidate in (s, baseline):
            candidate.t += 2.
            structure.deform(candidate, *(np.zeros(candidate.n) for _ in range(3)), 2.)
        self.assertTrue(np.any(s.trace_inversion_uplift_m > 0))
        np.testing.assert_allclose(s.trace_uplift_m-baseline.trace_uplift_m,
                                   s.trace_inversion_uplift_m, atol=2e-10)
        np.testing.assert_allclose(s.structure['thickness_km']*s.structure['area_factor'], volume, atol=1e-12)
        np.testing.assert_allclose(s.trace_relief_m-initial, ledger(s), atol=2e-10)
        # At the physical thickness cap there is no fictitious reservoir use.
        for state, kinds, relief in ((s.structure, s.kind, s.relief),
                                     (s.trace_structure, s.trace_kind, s.trace_relief_m)):
            before = columns.elevation(state)
            state['thickness_km'][:] = columns.MAX_THICKNESS_KM
            relief[:] += columns.elevation(state)-before
        previous = s.trace_inversion_uplift_m.copy()
        s.t += 2.
        structure.deform(s, *(np.zeros(s.n) for _ in range(3)), 2.)
        np.testing.assert_array_equal(s.trace_inversion_uplift_m, previous)
        self.aligned(s)

    def test_actual_ridge_episode_adds_column_once_and_preserves_area(self):
        s = attach(prescribed_world())
        s._update_ridge_windows(0.)
        self.assertEqual(len(s.ridge_episodes), 1)
        s.bcode[:] = 3
        old_area = s.structure['area_factor'].copy()
        old_volume = s.structure['thickness_km']*old_area
        initial = s.trace_relief_m.copy()
        for epoch in range(2, 35, 2):
            s.t = float(epoch)
            structure.deform(s, *(np.zeros(s.n) for _ in range(3)), 2.)
        self.assertGreater(s.trace_ridge_uplift_m.max(), 100.)
        self.assertLessEqual(s.trace_ridge_uplift_m.max(), 200.)
        np.testing.assert_allclose(s.trace_uplift_m, s.trace_ridge_uplift_m, atol=2e-10)
        np.testing.assert_allclose(s.trace_relief_m-initial, s.trace_ridge_uplift_m, atol=2e-10)
        np.testing.assert_array_equal(s.structure['area_factor'], old_area)
        np.testing.assert_allclose(s.structure['thickness_km']*old_area-old_volume,
                                   s.structure['added_volume_km_per_reference_km2'], atol=1e-12)
        self.aligned(s)

    def test_direct_rift_cut_records_realized_loss_once_and_keeps_volume(self):
        s = world()
        p = (s.rift_id > 0) & (s.kind == 1)
        t = (s.trace_rift_id > 0) & (s.trace_kind == 1)
        self.assertTrue(np.any(p) and np.any(t))
        volume = s.structure['thickness_km']*s.structure['area_factor']
        initial = s.trace_relief_m.copy()
        result = structure.cut_rift(s, p, t)
        np.testing.assert_allclose(result['parcel'], 250., atol=2e-10)
        np.testing.assert_allclose(result['trace'], 250., atol=2e-10)
        np.testing.assert_allclose(s.trace_relief_m-initial, -s.trace_extension_m, atol=2e-10)
        np.testing.assert_allclose(s.trace_rift_extension_m, s.trace_extension_m, atol=2e-10)
        np.testing.assert_allclose(s.structure['thickness_km']*s.structure['area_factor'], volume, atol=1e-12)
        self.assertFalse(np.any(s.structure['rift_heat_m']))
        self.aligned(s)

    def test_arc_append_coalesce_and_replenishment_preserve_columns(self):
        s = world()
        old_count, old_trace = len(s.kind), len(s.trace_kind)
        # Append real new arc material through the engine. Until root wires the
        # hook, attach exactly the missing columns after the actual operation.
        cells = np.flatnonzero(s.crust == 0)[:2]
        s.steps = 1
        s._add_arc_crust(cells, np.full(len(cells), 40.))
        if len(s.structure['thickness_km']) < len(s.kind):
            structure.append_parcels(s, len(s.kind)-old_count)
        if len(s.trace_structure['thickness_km']) < len(s.trace_kind):
            structure.append_traces(s, len(s.trace_kind)-old_trace)
        self.assertGreater(len(s.kind), old_count)
        self.aligned(s)
        p = np.flatnonzero(s.kind == 3)
        t = np.flatnonzero(s.trace_kind == 3)
        # A real mechanically lowered arc receives bounded magma; its marker
        # floor gain is explicit adjustment and never an extra uplift entry.
        structure.cut_rift(s, p, t, 500.)
        previous = s.trace_relief_m.copy()
        prior_adjustment = s.trace_adjustment_m.copy()
        structure.replenish(s, p, t, 700.)
        np.testing.assert_allclose(s.trace_relief_m-previous,
                                   s.trace_adjustment_m-prior_adjustment, atol=2e-10)
        self.assertFalse(np.any(s.trace_rift_extension_m[t]))
        self.aligned(s)
        # The pure helper preserves height when unresolved juvenile columns
        # are merged. Exercise the wrapper against current real arc arrays.
        old_mass = s.mass[p].copy()
        mean_height = np.average(columns.elevation(s.structure)[p], weights=old_mass)
        mean_volume = np.average((s.structure['thickness_km']*s.structure['area_factor'])[p], weights=old_mass)
        keep = s.kind != 3
        trace_copy = deepcopy(s.trace_structure)
        structure.coalesce(s, p, keep, np.zeros(len(p), dtype=int))
        self.assertEqual(len(s.structure['thickness_km']), np.count_nonzero(keep)+1)
        self.assertAlmostEqual(columns.elevation(s.structure)[-1], mean_height, places=10)
        self.assertAlmostEqual((s.structure['thickness_km']*s.structure['area_factor'])[-1], mean_volume, places=10)
        for field in structure.FIELDS:
            np.testing.assert_array_equal(s.trace_structure[field], trace_copy[field])

    def test_kind_change_preserves_full_surface_and_structural_state(self):
        s = world()
        # Only the label and old relief coordinate change on arc accretion.
        s.kind[:4] = 3
        s.trace_kind[:2] = 3
        structure.initialize_parcels(s)
        initial_count = np.count_nonzero(s.initial_crust > 0)
        chosen = np.searchsorted(s.parcel_patch[:initial_count], s.trace_patch)
        # New labels must correspond at source markers for this fixture.
        s.trace_kind[:] = s.kind[chosen]
        s.trace_relief_m[:] = s.relief[chosen]
        structure.initialize_traces(s, chosen)
        old = columns.elevation(s.structure).copy()
        p, t = s.kind == 3, s.trace_kind == 3
        s.kind[p] = 1
        s.relief[p] -= 100.
        s.trace_kind[t] = 1
        s.trace_relief_m[t] -= 100.
        s.trace_adjustment_m[t] -= 100.
        self.aligned(s)
        np.testing.assert_array_equal(columns.elevation(s.structure), old)

    def test_checkpoint_roundtrip_continues_both_column_and_trace_ledgers_exactly(self):
        s = world(erosion=1.)
        zero = np.zeros(s.n)
        s.t += 2.
        structure.deform(s, zero, np.full(s.n, 30.), zero, 2.)
        compatibility = dict(engine_sha256='structure-fixture', auxiliary_sources_sha256={}, numpy_version=np.__version__)
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder)/'state.npz'
            write_checkpoint(path, s, dict(config=s.config, time_myr=s.t), compatibility)
            loaded, _ = read_checkpoint(path, compatibility, Simulation)
        for candidate in (s, loaded):
            candidate.t += 2.
            structure.deform(candidate, np.full(s.n, 15.), zero, np.full(s.n, 8.), 2.)
        for name in ('structure', 'trace_structure'):
            for field in structure.FIELDS:
                np.testing.assert_array_equal(getattr(s, name)[field], getattr(loaded, name)[field], err_msg=name+'.'+field)
        for name in ('relief', 'trace_relief_m', 'trace_uplift_m', 'trace_extension_m', 'trace_erosion_m', 'trace_adjustment_m'):
            np.testing.assert_array_equal(getattr(s, name), getattr(loaded, name), err_msg=name)
        self.aligned(loaded)


if __name__ == '__main__':
    unittest.main()
