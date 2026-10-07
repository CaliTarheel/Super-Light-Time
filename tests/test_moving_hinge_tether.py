"""Independent work oracle for a moving hinge with a deformable slab neck."""
import unittest
from pathlib import Path
import tempfile

import numpy as np

import plate_balance as pb
import slab_tether_forces
import slab_memory
import checkpoint
from native_engine import Simulation
from tests.test_entry_regions import world as entry_world, advance as native_advance
from tests.test_slab_tether_forces import fixture


def moving_fixture(edges=(400., 600.)):
    state = fixture(edges=edges)
    state.plate_resistance_version = 1
    state.subduction_response_version = 1
    return state


class MovingHingeTetherTests(unittest.TestCase):
    def test_reversing_boundary_storage_preserves_plate_operator(self):
        original = pb.Balance(moving_fixture(), 1., slab_tethers=True)
        single_edge = pb.Balance(moving_fixture(edges=(1000.,)), 1., slab_tethers=True)
        np.testing.assert_allclose(original.stiffness, single_edge.stiffness,
                                   rtol=2e-14, atol=1e-5)
        np.testing.assert_allclose(original.torque, single_edge.torque,
                                   rtol=2e-14, atol=1e-5)
        state = moving_fixture()
        state.bp[::2], state.bq[::2] = state.bq[::2].copy(), state.bp[::2].copy()
        state.ba[::2], state.bb[::2] = state.bb[::2].copy(), state.ba[::2].copy()
        state.bn[::2] *= -1.
        reversed_model = pb.Balance(state, 1., slab_tethers=True)
        np.testing.assert_allclose(reversed_model.stiffness, original.stiffness,
                                   rtol=2e-14, atol=1e-5)
        np.testing.assert_allclose(reversed_model.torque, original.torque,
                                   rtol=2e-14, atol=1e-5)

    def test_reduced_balance_matches_independent_slab_speed_system(self):
        model = pb.Balance(moving_fixture(), 1., slab_tethers=True)
        assembly = model.tether_assembly
        channels = [(record, edge) for record in assembly['channels']
                    for edge in range(len(record['edge_shares']))]
        size = model.size
        matrix = np.zeros((size+len(channels), size+len(channels)))
        force = np.zeros(size+len(channels))
        matrix[:size, :size] = model.stiffness-assembly['stiffness']
        force[:size] = model.torque-assembly['drive']
        for offset, (record, edge) in enumerate(channels):
            q = record['inlet_rows'][edge]
            h = record['hinge_rows'][edge]
            share = record['edge_shares'][edge]
            cs, cn = record['mantle_drag_n_s_m'], record['neck_drag_n_s_m']
            cosine = record['dip_cosine']
            slot = size+offset
            matrix[:size, :size] += share*pb.CM_YR_M_S**2*(cs*np.outer(h, h)+cn*np.outer(q, q))
            cross = share*pb.CM_YR_M_S**2*(cs*cosine*h-cn*q)
            matrix[:size, slot] += cross
            matrix[slot, :size] += cross
            matrix[slot, slot] = share*pb.CM_YR_M_S**2*(cs+cn)
            force[slot] = share*pb.CM_YR_M_S*record['weight_n']
        expanded = np.linalg.solve(matrix, force)
        condensed = np.linalg.solve(model.stiffness, model.torque)
        np.testing.assert_allclose(condensed, expanded[:size], rtol=2e-12, atol=1e-11)
        self.assertGreater(np.linalg.eigvalsh(assembly['stiffness']).min(), -1e-5)
        self.assertGreater(np.linalg.norm(assembly['over_drive']), 0.)
        np.testing.assert_allclose(model.drivers['slab'],
                                   assembly['down_drive']+assembly['over_drive'])

    def test_slab_power_closes_for_arbitrary_plate_motion(self):
        model = pb.Balance(moving_fixture(), 1., slab_tethers=True)
        assembly = model.tether_assembly
        for x in np.random.default_rng(24931).normal(size=(20, model.size))*10.:
            report = slab_tether_forces.audit(assembly, x)
            expected = float(x@(assembly['drive']-assembly['stiffness']@x))
            scale = max(abs(report['weight_power_w']), abs(expected), 1.)
            self.assertLess(abs(report['plate_power_w']-expected)/scale, 2e-13)
            self.assertLess(abs(report['power_residual_w'])/scale, 2e-13)
            self.assertGreaterEqual(report['mantle_dissipation_w'], 0.)
            self.assertGreaterEqual(report['neck_dissipation_w'], 0.)

    def test_entry_and_attachment_replay_after_native_checkpoint(self):
        state = entry_world()
        state.subduction_response_version = 1
        first = native_advance(state, .000005, max_source_step_myr=.000005)
        self.assertEqual(state.continental_entry_regions['epoch_myr'], state.t)
        self.assertTrue(first['persistent_entry_regions'])
        self.assertIn('independently eliminated',
                      state.plate_balance_diagnostics['subduction_response'])
        balance = state.plate_balance_diagnostics
        self.assertLess(abs(balance['power_balance_error_w']),
                        1e-9*max(abs(balance['driver_work_w']), 1.))
        tether = balance['slab_tether_power']
        self.assertLess(abs(tether['power_residual_w']),
                        1e-12*max(abs(tether['weight_power_w']), 1.))
        for row in state.trench_systems:
            slab_memory.validate_row(row, require_mass=True)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)/'coupled-entry.npz'
            checkpoint.write_checkpoint(path, state, dict(config=state.config), {})
            resumed, _ = checkpoint.read_checkpoint(path, None, Simulation)
        left = native_advance(state, .000005, max_source_step_myr=.000005)
        right = native_advance(resumed, .000005, max_source_step_myr=.000005)
        self.assertEqual(left, right)
        np.testing.assert_array_equal(state.omega, resumed.omega)
        np.testing.assert_array_equal(state.parcel_entry_region, resumed.parcel_entry_region)
        np.testing.assert_array_equal(state.material_surface['vertices'],
                                      resumed.material_surface['vertices'])


if __name__ == '__main__':
    unittest.main()
