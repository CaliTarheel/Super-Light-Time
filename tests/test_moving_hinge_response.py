"""Work and symmetry oracles for the opt-in, trench-relative slab closure.

These fixtures use the complete Balance assembly and nonlinear solver. Only
the inventory lookup is replaced by prescribed line loads; the force, basal
drag, slab drag, hinge and passive interface laws are production code.
"""
from copy import deepcopy
import math
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np

import plate_balance as balance
import slab_memory


def fixture(*, version=1, opposed=False, pieces=1, load=5.e11):
    count = 3 if opposed else 2
    # Three non-collinear equal-area samples give each plate an isotropic,
    # strictly positive basal resistance without prescribing its velocity.
    xyz = np.tile(np.eye(3), (count, 1))
    angle = math.radians(20.)
    radial = [[math.cos(angle), math.sin(angle), 0.]]
    normal = [[-math.sin(angle), math.cos(angle), 0.]]
    if opposed:
        radial.append([math.cos(angle), -math.sin(angle), 0.])
        normal.append([-math.sin(angle), -math.cos(angle), 0.])
    s = SimpleNamespace(
        active=np.ones(count, bool), omega=np.zeros((count, 3)),
        names=[f'Plate {i}' for i in range(count)],
        plate=np.repeat(np.arange(count), 3), xyz=xyz,
        cell_area=np.full(3*count, 1.e6),
        crust=np.zeros(3*count, int), age=np.full(3*count, 100.),
        bmid=np.repeat(np.asarray(radial), pieces, axis=0),
        bn=np.repeat(np.asarray(normal), pieces, axis=0),
        bp=np.zeros((count-1)*pieces, int),
        bq=np.repeat(np.arange(1, count), pieces),
        ba=np.zeros((count-1)*pieces, int),
        bb=np.repeat(3*np.arange(1, count), pieces),
        bl=np.full((count-1)*pieces, 1000./pieces),
        plate_resistance_version=1,
        subduction_response_version=version,
        test_slab_owner=np.zeros((count-1)*pieces, int),
        test_slab_load=np.full((count-1)*pieces, load),
        test_slab_length=np.full((count-1)*pieces, 200.))
    return s


def assembled(s):
    inventory = (s.test_slab_owner, s.test_slab_load, s.test_slab_length)
    with patch.object(slab_memory, 'line_load', return_value=inventory):
        return balance.Balance(s, 2.)


def gravitational_power(s, x):
    """Independent physical velocities and negative-buoyancy work in watts."""
    rotation = np.asarray(x).reshape(-1, 3)
    down = s.test_slab_owner
    over = np.where(down == s.bp, s.bq, s.bp)
    toward = np.where((down == s.bp)[:, None], s.bn, -s.bn)
    incoming = np.cross(rotation[down], s.bmid)*balance.CM_YR_M_S
    hinge = np.cross(rotation[over], s.bmid)*balance.CM_YR_M_S
    intake = np.sum((incoming-hinge)*toward, axis=1)
    vertical = intake*math.sin(math.radians(slab_memory.SUBDUCTION_DIP_DEG))
    return float(np.sum(9.81*s.test_slab_load*s.bl*1000.*vertical))


def slab_drag_matrix(model):
    """Extract only the actual assembled slab-mantle resistance matrix."""
    matrix = model.stiffness.copy()-model.interface_stiffness
    for plate, block in model.basal.items():
        start = 3*model.slot[plate]
        matrix[start:start+3, start:start+3] -= block
    matrix -= np.einsum('e,ei,ej->ij', model.slab_anchor_coefficient,
                        model.slab_anchor_rows, model.slab_anchor_rows)
    return matrix


class MovingHingeWorkTests(unittest.TestCase):
    def test_gravity_is_work_conjugate_to_intake_and_counted_once(self):
        s = fixture(opposed=True)
        model = assembled(s)
        x = np.array([.2, -.7, .4, 1.1, -.6, -.8, -.3, .5, 1.7])
        expected = gravitational_power(s, x)
        self.assertAlmostEqual(float(model.drivers['slab']@x)/expected, 1., places=13)
        np.testing.assert_allclose(model.drivers['slab'],
                                   model.slab_down_drive+model.slab_over_drive)
        np.testing.assert_allclose(model.torque,
                                   sum(model.drivers.values()), rtol=1.e-14)
        # Generalized force is the derivative of released gravitational work.
        numeric = []
        for axis in np.eye(model.size):
            numeric.append((gravitational_power(s, x+axis*1.e-5)
                            -gravitational_power(s, x-axis*1.e-5))/(2.e-5))
        np.testing.assert_allclose(model.drivers['slab'], numeric,
                                   rtol=2.e-10, atol=2.e-2)

    def test_common_rotation_cannot_release_gravitational_energy(self):
        for opposed in (False, True):
            s = fixture(opposed=opposed)
            model = assembled(s)
            for spin in ([.3, -.2, .7], [-1., .5, -3.]):
                x = np.tile(spin, len(s.active))
                self.assertEqual(gravitational_power(s, x), 0.)
                self.assertLess(abs(float(model.drivers['slab']@x)),
                                1.e-14*np.linalg.norm(model.drivers['slab'])*np.linalg.norm(x))
            np.testing.assert_allclose(model.drivers['slab'].reshape(-1, 3).sum(axis=0),
                                       0., atol=1.e-6)

    def test_stationary_incoming_plate_can_feed_a_moving_hinge(self):
        s = fixture()
        model = assembled(s)
        x = np.array([0., 0., 0., 0., 0., -2.])
        expected = gravitational_power(s, x)
        self.assertGreater(expected, 0.)
        self.assertEqual(float(model.slab_down_drive@x), 0.)
        self.assertAlmostEqual(float(model.slab_over_drive@x)/expected, 1., places=13)
        self.assertGreater(float((model.slab_vertical_rows@x)[0]), 0.)

    def test_opening_keeps_signed_gravity_work_and_has_no_active_resistance_tail(self):
        s = fixture()
        model = assembled(s)
        # Opening implies lifting under this constrained geometry. Its signed
        # gravitational work must not be clipped to zero or treated as another
        # positive source; finite capture does not reverse the slab inventory.
        x = np.array([0., 0., 0., 0., 0., 2.])
        expected = gravitational_power(s, x)
        self.assertLess(expected, 0.)
        self.assertAlmostEqual(float(model.drivers['slab']@x)/expected, 1., places=13)
        delta = balance.HUBER_CONTINUATION_KM_MYR[-1]*balance.KM_MYR_CM_YR
        work, checks = model._resistance_work(x, delta)
        self.assertEqual(work['hinge'], 0.)
        self.assertEqual(work['megathrust'], 0.)
        self.assertEqual(work['strike_slip'], 0.)
        self.assertEqual(checks['negative_resisting_work_elements'], 0)
        self.assertGreater(float(x@model.stiffness@x), 0.)
        _, gradient, _, _, _ = model._evaluate(x, delta)
        np.testing.assert_allclose(gradient, model.stiffness@x-model.torque)

    def test_overriding_power_can_be_negative_even_while_total_slab_work_is_positive(self):
        model = assembled(fixture())
        x = np.array([0., 0., 2., 0., 0., 1.])
        self.assertGreater(float(model.slab_down_drive@x), 0.)
        self.assertLess(float(model.slab_over_drive@x), 0.)
        self.assertGreater(float(model.drivers['slab']@x), 0.)
        self.assertAlmostEqual(float(model.drivers['slab']@x),
                               float(model.slab_down_drive@x+model.slab_over_drive@x))

    def test_slab_drag_is_passive_and_uses_actual_absolute_slab_velocity(self):
        s = fixture(opposed=True)
        model = assembled(s)
        matrix = slab_drag_matrix(model)
        np.testing.assert_allclose(matrix, matrix.T, atol=1.e-6)
        self.assertGreaterEqual(np.linalg.eigvalsh(matrix).min(),
                                -1.e-13*np.linalg.norm(matrix))
        down = s.test_slab_owner
        over = np.where(down == s.bp, s.bq, s.bp)
        dip = math.radians(slab_memory.SUBDUCTION_DIP_DEG)
        coefficient = (balance.SLAB_STOKES_PA_S
                       *np.minimum(s.test_slab_length/balance.SLAB_REFERENCE_LENGTH_KM, 1.)
                       *s.bl*1000.)
        for x in np.random.default_rng(81312).normal(size=(30, model.size)):
            rotation = x.reshape(-1, 3)
            down_speed = np.sum(np.cross(rotation[down], s.bmid)*s.bn, axis=1)*balance.CM_YR_M_S
            over_speed = np.sum(np.cross(rotation[over], s.bmid)*s.bn, axis=1)*balance.CM_YR_M_S
            horizontal = over_speed+(down_speed-over_speed)*math.cos(dip)
            vertical = (down_speed-over_speed)*math.sin(dip)
            expected = float(coefficient@(horizontal**2+vertical**2))
            self.assertGreaterEqual(float(x@matrix@x), 0.)
            self.assertAlmostEqual(float(x@matrix@x)/expected, 1., places=12)

    def test_fixed_upper_recovers_original_stokes_drag_and_common_motion_is_resisted(self):
        moving = assembled(fixture())
        fixed = assembled(fixture(version=0))
        matrix = slab_drag_matrix(moving)
        old_matrix = slab_drag_matrix(fixed)
        np.testing.assert_allclose(matrix[:3, :3], old_matrix[:3, :3], rtol=1.e-13, atol=1.e-6)
        common = np.array([0., 0., 2., 0., 0., 2.])
        self.assertGreater(float(common@matrix@common), 0.)
        np.testing.assert_allclose(moving.slab_vertical_rows@common, 0., atol=1.e-14)
        np.testing.assert_allclose(moving.slab_horizontal_rows@common, [2.], atol=1.e-14)

    def test_exactly_opposed_slabs_close_both_margins_without_translating_central_plate(self):
        model = assembled(fixture(opposed=True))
        model.solve()
        np.testing.assert_allclose(model.drivers['slab'][:3], 0., atol=1.e-6)
        np.testing.assert_allclose(model.x[:3], 0., atol=1.e-12)
        self.assertLess(model.x[5], 0.)
        self.assertGreater(model.x[8], 0.)
        self.assertAlmostEqual(model.x[5], -model.x[8], places=11)
        normal, _ = model._relative_rows(model.trench_edges)
        self.assertTrue(np.all(normal@model.x < 0.))
        diag = model.diagnostics(model.rotation())
        self.assertLessEqual(diag['scaled_force_residual'], balance.FORCE_RELATIVE_TOLERANCE)
        self.assertEqual(diag['negative_resisting_work_elements'], 0)
        self.assertLess(abs(diag['power_balance_error_w']), 1.e-7*diag['driver_work_w'])

    def test_no_retained_slab_or_other_driver_has_no_motion(self):
        model = assembled(fixture(opposed=True, load=0.))
        model.solve()
        np.testing.assert_array_equal(model.torque, 0.)
        np.testing.assert_array_equal(model.x, 0.)
        self.assertEqual(model.diagnostics(model.rotation())['driver_work_w'], 0.)

    def test_diagnostics_partition_actual_plate_work_without_a_second_slab_driver(self):
        for version in (0, 1):
            model = assembled(fixture(version=version))
            model.solve()
            diag = model.diagnostics(model.rotation())
            self.assertEqual(diag['subduction_response_version'], version)
            self.assertEqual([key for key in model.drivers if key.startswith('slab')], ['slab'])
            for component, force in (('downgoing', model.slab_down_drive),
                                     ('overriding', model.slab_over_drive)):
                expected = float(force@model.x)
                self.assertAlmostEqual(diag['slab_power_w'][component], expected,
                                       delta=1.e-12*max(abs(expected), 1.))
                per_plate = diag['slab_power_by_plate_w']
                self.assertAlmostEqual(sum(row[component] for row in per_plate.values()), expected,
                                       delta=1.e-12*max(abs(expected), 1.))
                for plate, index in model.slot.items():
                    part = slice(3*index, 3*index+3)
                    self.assertEqual(per_plate[model.s.names[plate]][component],
                                     float(force[part]@model.x[part]))
            power = diag['slab_power_w']
            self.assertAlmostEqual(power['total'], power['downgoing']+power['overriding'],
                                   delta=1.e-12*max(abs(power['total']), 1.))
            # This fixture has no ridge-age gradient or collision overlap, so
            # the complete source budget must equal this one slab total.
            self.assertEqual(diag['driver_work_w'], power['total'])
            self.assertAlmostEqual(diag['slab_stokes_dissipation_w'],
                                   float(model.x@slab_drag_matrix(model)@model.x),
                                   delta=1.e-12*diag['viscous_dissipation_w'])
            parts = (diag['slab_stokes_dissipation_w']+diag['slab_anchor_dissipation_w']
                     +diag['basal_dissipation_w']+diag['interface_shear_dissipation_w'])
            self.assertAlmostEqual(parts, diag['viscous_dissipation_w'],
                                   delta=1.e-12*diag['viscous_dissipation_w'])


class MovingHingeGeometryTests(unittest.TestCase):
    def assert_same_operator(self, first, second):
        np.testing.assert_allclose(first.torque, second.torque, atol=1.e-5, rtol=2.e-14)
        np.testing.assert_allclose(first.stiffness, second.stiffness, atol=1.e-5, rtol=2.e-14)
        first.solve()
        second.solve()
        np.testing.assert_allclose(first.x, second.x, atol=2.e-10, rtol=2.e-9)

    def test_reversing_edge_storage_preserves_polarity_and_motion(self):
        s = fixture(opposed=True)
        reverse = deepcopy(s)
        reverse.bp, reverse.bq = s.bq.copy(), s.bp.copy()
        reverse.ba, reverse.bb = s.bb.copy(), s.ba.copy()
        reverse.bn = -s.bn
        self.assert_same_operator(assembled(s), assembled(reverse))

    def test_subdividing_trench_does_not_change_force_resistance_or_solution(self):
        for pieces in (3, 17):
            self.assert_same_operator(assembled(fixture(opposed=True)),
                                      assembled(fixture(opposed=True, pieces=pieces)))

    def test_changing_owner_numbers_only_permutes_blocks(self):
        s = fixture(opposed=True)
        remapped = deepcopy(s)
        mapping = np.array([2, 0, 1])
        for name in ('bp', 'bq', 'plate', 'test_slab_owner'):
            setattr(remapped, name, mapping[getattr(s, name)])
        original, changed = assembled(s), assembled(remapped)
        indices = (3*mapping[:, None]+np.arange(3)).ravel()
        np.testing.assert_allclose(original.torque, changed.torque[indices])
        np.testing.assert_allclose(original.stiffness, changed.stiffness[np.ix_(indices, indices)])
        original.solve()
        changed.solve()
        np.testing.assert_allclose(original.x, changed.x[indices], atol=1.e-10, rtol=1.e-9)

    def test_global_rotation_rotates_force_and_solution_without_changing_work(self):
        s = fixture(opposed=True)
        # Rodrigues rotation around a non-coordinate axis avoids testing only
        # coordinate relabeling or the fixture's special equatorial symmetry.
        axis = np.array([1., 2., -3.])/math.sqrt(14.)
        angle = .79
        skew = np.array([[0., -axis[2], axis[1]], [axis[2], 0., -axis[0]],
                         [-axis[1], axis[0], 0.]])
        rotation = np.eye(3)*math.cos(angle)+(1.-math.cos(angle))*np.outer(axis, axis)+math.sin(angle)*skew
        rotated = deepcopy(s)
        for name in ('xyz', 'bmid', 'bn'):
            setattr(rotated, name, getattr(s, name)@rotation.T)
        original, changed = assembled(s), assembled(rotated)
        expected_force = original.torque.reshape(-1, 3)@rotation.T
        np.testing.assert_allclose(changed.torque.reshape(-1, 3), expected_force,
                                   rtol=1.e-13, atol=1.e-5)
        original.solve()
        changed.solve()
        np.testing.assert_allclose(changed.x.reshape(-1, 3), original.x.reshape(-1, 3)@rotation.T,
                                   rtol=2.e-7, atol=2.e-10)
        self.assertAlmostEqual(float(original.torque@original.x)/float(changed.torque@changed.x), 1., places=8)


class MovingHingeVersionTests(unittest.TestCase):
    def test_absent_saved_version_retains_fixed_trench_equations(self):
        absent = fixture(version=0)
        del absent.subduction_response_version
        self.assertEqual(balance.subduction_response_version(absent), 0)
        old, missing = assembled(fixture(version=0)), assembled(absent)
        np.testing.assert_array_equal(old.torque, missing.torque)
        np.testing.assert_array_equal(old.stiffness, missing.stiffness)
        np.testing.assert_array_equal(old.slab_over_drive, 0.)
        old.solve()
        missing.solve()
        np.testing.assert_array_equal(old.x, missing.x)

    def test_unknown_or_ambiguous_saved_version_is_rejected(self):
        for version in (True, False, np.bool_(True), '1', 1., None, -1, 2):
            with self.subTest(version=version), self.assertRaises(ValueError):
                assembled(fixture(version=version))
        self.assertEqual(balance.subduction_response_version(fixture(version=np.int64(1))), 1)

    def test_moving_hinge_rejects_active_historical_resistance(self):
        s = fixture()
        s.plate_resistance_version = 0
        with self.assertRaisesRegex(ValueError, 'passive'):
            assembled(s)

    def test_independent_retreat_cannot_be_added_to_the_moving_hinge_geometry(self):
        for retreat in ([0., 1.], [np.nan], [np.inf], [-.01]):
            s = fixture()
            s.trench_retreat_speed = np.asarray(retreat)
            with self.subTest(retreat=retreat), self.assertRaisesRegex(ValueError, 'retreat'):
                assembled(s)
        s = fixture()
        s.trench_retreat_speed = np.zeros(len(s.ba))
        assembled(s)


if __name__ == '__main__':
    unittest.main()
