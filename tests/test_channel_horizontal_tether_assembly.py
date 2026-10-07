"""Independent explicit-speed and power checks for horizontal-intake slab forces."""

import unittest
import math

import numpy as np

import plate_balance as pb
import slab_tether_forces
from channel_horizontal_slab_oracle import horizontal_intake_attachment
from tests.test_moving_hinge_tether import moving_fixture


class HorizontalTetherAssemblyTests(unittest.TestCase):
    def test_candidate_matches_explicit_slab_speed_system(self):
        balance = pb.Balance(moving_fixture(), 1., slab_tethers=True)
        legacy = balance.tether_assembly
        before = legacy['stiffness'].copy(), legacy['drive'].copy()
        candidate = slab_tether_forces.prepare_horizontal(balance)
        np.testing.assert_array_equal(legacy['stiffness'], before[0])
        np.testing.assert_array_equal(legacy['drive'], before[1])
        size = balance.size
        kpp = np.zeros((size, size))
        cross = []
        slab_diagonal = []
        slab_force = []
        for record in candidate['channels']:
            q = record['inlet_rows']
            h = record['hinge_rows']
            cs, cn, weight = (record[key] for key in
                              ('mantle_drag_n_s_m', 'neck_drag_n_s_m', 'weight_n'))
            cosine = record['dip_cosine']
            for share, qr, hr in zip(record['edge_shares'], q, h):
                if share <= 0.:
                    continue
                kpp += share * pb.CM_YR_M_S**2 * (
                    cs * np.outer(hr, hr) + cn * np.outer(qr, qr))
                cross.append(share * pb.CM_YR_M_S *
                             (cs * cosine * hr - cn * cosine * qr))
                slab_diagonal.append(share * (cs + cn * cosine**2))
                slab_force.append(share * weight)
        cross = np.column_stack(cross)
        inverse = 1. / np.asarray(slab_diagonal)
        reduced_k = kpp - (cross * inverse) @ cross.T
        reduced_f = -cross @ (inverse * slab_force)
        np.testing.assert_allclose(candidate['stiffness'], reduced_k,
                                   rtol=3e-12, atol=1e-5)
        np.testing.assert_allclose(candidate['drive'], reduced_f,
                                   rtol=3e-12, atol=1e-5)
        np.testing.assert_allclose(candidate['drive'],
                                   candidate['down_drive'] + candidate['over_drive'])
        self.assertGreaterEqual(np.linalg.eigvalsh(candidate['stiffness']).min(),
                                -1e-12 * np.linalg.norm(candidate['stiffness']))

    def test_candidate_power_closes_without_changing_legacy_assembly(self):
        balance = pb.Balance(moving_fixture(), 1., slab_tethers=True)
        candidate = slab_tether_forces.prepare_horizontal(balance)
        for x in np.random.default_rng(8622).normal(size=(30, balance.size)) * 10.:
            report = slab_tether_forces.audit(candidate, x)
            expected = float(x @ (candidate['drive'] - candidate['stiffness'] @ x))
            scale = max(abs(report['weight_power_w']), abs(expected), 1.)
            self.assertLess(abs(report['plate_power_w'] - expected) / scale, 3e-13)
            self.assertLess(abs(report['power_residual_w']) / scale, 3e-13)
            self.assertGreaterEqual(report['mantle_dissipation_w'], 0.)
            self.assertGreaterEqual(report['neck_dissipation_w'], 0.)
        np.testing.assert_array_equal(balance.tether_assembly['stiffness'],
                                      slab_tether_forces.prepare(balance)['stiffness'])

    def test_candidate_agrees_with_independent_scalar_work_oracle(self):
        balance = pb.Balance(moving_fixture(), 1., slab_tethers=True)
        candidate = slab_tether_forces.prepare_horizontal(balance)
        x = np.random.default_rng(29).normal(size=balance.size)
        expected = np.zeros(4)
        for record in candidate['channels']:
            dip = math.degrees(math.acos(record['dip_cosine']))
            for share, q, h in zip(record['edge_shares'],
                                   record['inlet_rows'], record['hinge_rows']):
                if share <= 0.:
                    continue
                hinge = float(h @ x * pb.CM_YR_M_S)
                down = float((q + h) @ x * pb.CM_YR_M_S)
                row = horizontal_intake_attachment(
                    down, hinge, dip, record['mantle_drag_n_s_m'],
                    record['neck_drag_n_s_m'],
                    record['weight_n'] / record['dip_sine'])
                expected += share * np.array((row['gravity_power_w'],
                    row['plate_power_w'], row['mantle_dissipation_w'],
                    row['neck_dissipation_w']))
        actual = slab_tether_forces.audit(candidate, x)
        np.testing.assert_allclose(expected, [actual[key] for key in
            ('weight_power_w', 'plate_power_w', 'mantle_dissipation_w',
             'neck_dissipation_w')], rtol=3e-13)

    def test_fixed_hinge_state_does_not_select_horizontal_candidate(self):
        balance = pb.Balance(moving_fixture(), 1., slab_tethers=True)
        balance.subduction_response_version = 0
        with self.assertRaisesRegex(ValueError, 'moving-hinge'):
            slab_tether_forces.prepare_horizontal(balance)


if __name__ == '__main__':
    unittest.main()
