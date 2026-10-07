"""Small native integration fixture; authored strengths are not production tuning.

The weak ocean belt is a supplied test yield field, applied only at the
cell-strength input boundary. Forces, mode solving, accumulated cut work,
daughter viability, topology transactions, capture and checkpoint code remain
real. Held-candidate cut work is not a full-world finite-time energy certificate.
"""
from dataclasses import asdict
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import numpy as np

import checkpoint
import force_rifting
import native_engine
import native_spreading
import plate_balance as pb
import plate_limit_analysis as limit
import slab_memory as slab
import slab_tether_history as history
import slab_tether_native
from tests.test_slab_tether import neck
from tests.test_slab_tether_native import world as attached_world


def authored_world():
    s = attached_world()
    # A fresh, attached initial inventory, rather than the rapid slab-rupture
    # fixture's nearly failed10m neck. This is explicit fixture rheology.
    for row in s.trench_systems:
        for channel in row[history.FIELD]:
            channel['neck'] = asdict(neck(damage=.3))
        slab.validate_row(row, require_mass=True)
    s.trench_shutdown_version = 1
    s.subduction_response_version = 1
    # The reused hemispheric fixture manually assigned UID2; its allocator
    # must start after those supplied identities before a real split.
    s.next_plate_uid = int(np.max(s.plate_uid[s.active]))+1
    s.config['force_limit_rifting'] = dict(enabled=True, ocean_breakup_opening_km=1.,
        rift_width_km=10., breakup_stretch=2., rescan_interval_myr=10.)
    s.coupled_rift_fixture = dict(interpretation='authored test-only ocean yield belt',
                                belt_axis=[0., 0., 1.], half_width=.2, strength_factor=1e-4)
    force_rifting.initialize(s)
    return s


def authored_strengths(s, original):
    def evaluate(crust, age, heat_flow=None):
        strength = original(crust, age, heat_flow)
        settings = s.coupled_rift_fixture
        belt = np.abs(s.xyz@settings['belt_axis']) < settings['half_width']
        factor = np.where(belt & (np.asarray(crust) == 0), settings['strength_factor'], 1.)
        return tuple(value*factor for value in strength)
    return evaluate


def assert_checkpoint_equal(test, state, decoded):
    """Every encoded state array and JSON scalar/list field must round-trip."""
    with tempfile.TemporaryDirectory() as directory:
        left, right = Path(directory)/'left.npz', Path(directory)/'right.npz'
        checkpoint.write_checkpoint(left, state, dict(config=state.config), {})
        checkpoint.write_checkpoint(right, decoded, dict(config=decoded.config), {})
        with np.load(left, allow_pickle=False) as a, np.load(right, allow_pickle=False) as b:
            test.assertEqual(a.files, b.files)
            for key in a.files:
                np.testing.assert_array_equal(a[key], b[key], err_msg=key)
                test.assertEqual(a[key].dtype, b[key].dtype, key)
                test.assertEqual(a[key].shape, b[key].shape, key)
                test.assertEqual(a[key].tobytes(), b[key].tobytes(), key)


def evolve(s, duration, source_step):
    original = limit.cell_strengths
    with patch.object(limit, 'cell_strengths', authored_strengths(s, original)):
        return slab_tether_native.advance(s, duration, new_neck=neck(), support_radius_km=15000.,
            max_source_step_myr=source_step, absolute_tolerance=1e-6, relative_tolerance=1e-5)


def assert_budgets(test, s):
    for row in s.trench_systems:
        slab.validate_row(row, require_mass=True)
    area_source = sum(r.get(slab.INITIAL_AREA_FIELD, 0.)+r['slab_fed_area_km2'] for r in s.trench_systems)
    area_retained = sum(r['slab_retained_area_km2']+r['slab_retired_area_km2'] for r in s.trench_systems)
    mass_source = sum(r['slab_initial_excess_mass_kg']+r['slab_fed_excess_mass_kg'] for r in s.trench_systems)
    mass_retained = sum(r['slab_retained_excess_mass_kg']+r['slab_retired_excess_mass_kg'] for r in s.trench_systems)
    test.assertAlmostEqual(area_retained/area_source, 1., places=11)
    test.assertAlmostEqual(mass_retained/mass_source, 1., places=11)
    test.assertTrue(np.isfinite(s.support).all())
    test.assertTrue(np.all(s.support >= 0.))
    np.testing.assert_allclose(s.support.sum(axis=0), 1., rtol=0., atol=1e-9)
    diagnostic = s.spreading_diagnostics
    test.assertAlmostEqual(diagnostic['side_p_area_km2'], diagnostic['side_q_area_km2'],
        delta=1e-8*max(diagnostic['generated_area_km2'], 1.))
    test.assertAlmostEqual(diagnostic['generated_area_km2'],
        diagnostic['side_p_area_km2']+diagnostic['side_q_area_km2'],
        delta=1e-8*max(diagnostic['generated_area_km2'], 1.))
    test.assertEqual(s.plate_balance_diagnostics['negative_resisting_work_elements'], 0)
    test.assertLessEqual(s.plate_balance_diagnostics['scaled_force_residual'], pb.FORCE_RELATIVE_TOLERANCE)


def checkpoint_copy(test, s):
    with tempfile.TemporaryDirectory() as directory:
        path = Path(directory)/'candidate.npz'
        checkpoint.write_checkpoint(path, s, dict(config=s.config), {})
        decoded, _ = checkpoint.read_checkpoint(path, None, native_engine.Simulation)
    assert_checkpoint_equal(test, s, decoded)
    return decoded


class CoupledRiftEvolutionTests(unittest.TestCase):
    def test_authored_native_fixture_update_commit_and_exact_restart(self):
        s = authored_world()
        first = evolve(s, .5, .5)
        self.assertEqual(len(first['source_intervals']), 1)
        self.assertGreater(s.force_rifting_diagnostics['checks'], 0)
        self.assertEqual(s.force_rifting_diagnostics['commits'], 0)
        self.assertTrue(s.force_rifting_state)
        self.assertTrue(all(row['force_accounting'] == 'active attached-slab ledger'
                            for row in s.force_rifting_state.values()))
        self.assertGreater(s.force_rifting_diagnostics['accepted_cut_work_j'], 0.)
        assert_budgets(self, s)
        decoded = checkpoint_copy(self, s)
        left, right = evolve(s, .5, .5), evolve(decoded, .5, .5)
        self.assertEqual(left, right)
        assert_checkpoint_equal(self, s, decoded)
        self.assertEqual(s.force_rifting_diagnostics['commits'], 1)
        self.assertEqual(len(np.unique(s.plate_uid[s.active])), int(s.active.sum()))
        event = [e for e in s.events if e['type'] == 'rift'][-1]
        details = event['details']
        self.assertEqual(details['setting'], 'oceanic')
        self.assertTrue(details['daughter_viability']['viable'])
        self.assertGreater(details['daughter_viability']['min_interior_cells'], 0)
        loading = details['loading']
        self.assertEqual(loading['cause'], 'force_limit')
        self.assertGreaterEqual(loading['accumulated_opening_km'], loading['breakup_opening_km'])
        self.assertGreater(loading['paid_cut_work_j'], 0.)
        self.assertLessEqual(loading['last_opening_work']['power_closure_relative_to_cut'], 1e-7)
        self.assertLess(loading['opening_mode']['common_restriction_relative_error'], 1e-10)
        self.assertLess(loading['opening_mode']['differential_virtual_work_relative_error'], 1e-10)
        retired = [row for row in s.force_rifting_diagnostics['retired_candidate_paths']
                   if row['plate_uid'] == details['parent_plate_uid'] and row['reason'] == 'committed breakup']
        self.assertEqual(len(retired), 1)
        self.assertEqual(retired[0]['paid_cut_work_j'], loading['paid_cut_work_j'])
        self.assertEqual(retired[0]['last_opening_work'], loading['last_opening_work'])
        self.assertGreater(retired[0]['cut_edge_count'], 0)
        assert_budgets(self, s)

        # Re-solving the current daughter geometry gives real divergent ocean
        # pieces on the new pair. The next source interval runs ordinary native
        # paired spreading/capture, not a prescribed ridge velocity.
        pair = {details['parent_plate_uid'], details['new_plate_uid']}
        pieces = native_spreading.prepare(s)['ocean']
        chosen = np.array([set((int(s.plate_uid[a]), int(s.plate_uid[b]))) == pair
            for a, b in zip(pieces['owner_a'], pieces['owner_b'])])
        self.assertTrue(np.any(chosen & pieces['active'] & pieces['divergent']))
        post_split = checkpoint_copy(self, s)
        created = s.process_totals['ocean_created_km2']
        left, right = evolve(s, .01, .01), evolve(post_split, .01, .01)
        self.assertEqual(left, right)
        self.assertGreater(s.process_totals['ocean_created_km2'], created)
        assert_budgets(self, s)
        assert_checkpoint_equal(self, s, post_split)


if __name__ == '__main__':
    unittest.main()
