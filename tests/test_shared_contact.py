"""Manufactured frozen joint operator checks, not a native collision benchmark."""
import unittest

import numpy as np
import mesh_geometry
from material_forcing import sample_belts

from benchmarks.collision_architecture import shared_contact as joint


def unit(p):
    p = np.asarray(p, float)
    return p / np.linalg.norm(p, axis=-1, keepdims=True)


def fixture(*, contacts=True):
    points = unit([[1, -.18, -.12], [1, .20, -.10], [1, .01, .20],
                   [1, -.24, -.20], [1, .24, -.18], [1, 0, .15]])
    samples = [joint.ContactSample(0, 1, unit([1, .005, z]),
        unit([-.005, 1, 0]), 50000.) for z in (-.04, .04)] if contacts else []
    return dict(points=points, faces=np.array([[0, 1, 2], [3, 4, 5]]),
        vertex_plate=np.array([0, 0, 0, 1, 1, 1]), radius_m=6371e3,
        basal_drag_pa_s_per_m=np.full(6, 1e15),
        viscosity_pa_s=np.array([2e21, 1e21]), sheet_thickness_m=np.array([4e4, 5e4]),
        basal_reference_velocity_m_s=np.zeros((6, 3)),
        other_plate_rotational_drag_n_m_s=np.eye(6)*4e38,
        external_torques_n_m=np.array([[0, 0, 1e25], [0, 0, -1e25]]),
        external_nodal_forces_n=np.zeros((6, 3)), contacts=samples)


def relative(a, b):
    return np.linalg.norm(a-b) / max(np.linalg.norm(a), np.linalg.norm(b), 1e-300)


class TestSharedContact(unittest.TestCase):
    def test_functional_derivative_and_reciprocal_cross_block(self):
        cfg = fixture()
        cfg['basal_drag_pa_s_per_m'] *= np.arange(1, 7)
        s = joint.assemble(**cfg)
        rng = np.random.default_rng(3)
        y = rng.normal(size=len(s['load_n']))*1e-9
        v = rng.normal(size=len(y))*1e-9
        _, grad = joint.evaluate(s, y)
        h = 1e-4
        fd = (joint.evaluate(s, y+h*v)[0]-joint.evaluate(s, y-h*v)[0])/(2*h)
        self.assertLess(abs(fd-grad@v)/abs(grad@v), 2e-10)
        k = s['hessian_n_s_m']; cut = s['nplate']
        self.assertLess(relative(k, k.T), 2e-16)
        self.assertGreater(np.linalg.norm(k[:cut, cut:]), 0.)
        self.assertLess(relative(k[:cut, cut:], k[cut:, :cut].T), 2e-16)
        fdg = (joint.evaluate(s, y+h*v)[1]-joint.evaluate(s, y-h*v)[1])/(2*h)
        self.assertLess(relative(fdg, k@v), 2e-11)

    def test_contact_reactions_change_plates_and_close_virtual_power(self):
        s = joint.assemble(**fixture()); solved = joint.solve(s)
        free = joint.solve(joint.assemble(**fixture(contacts=False)))
        self.assertGreater(relative(solved['plate_omega_rad_s'], free['plate_omega_rad_s']), .01)
        np.testing.assert_array_equal(solved['contact_first_force_n'], -solved['contact_second_force_n'])
        expected = np.zeros((2, 3))
        for i, (first, second) in enumerate(s['contact_plate_pairs']):
            expected[first] += s['radius_m']*np.cross(s['contact_point'][i], solved['contact_first_force_n'][i])
            expected[second] += s['radius_m']*np.cross(s['contact_point'][i], solved['contact_second_force_n'][i])
        self.assertLess(relative(expected, solved['contact_plate_torque_n_m']), 2e-13)
        self.assertLess(np.linalg.norm(expected.sum(axis=0))/np.linalg.norm(expected), 2e-15)
        d = solved['diagnostics']
        power = abs(d['external_power_w'])
        for name in ('basal_dissipation_w', 'viscous_dissipation_w', 'other_drag_dissipation_w'):
            self.assertGreaterEqual(d[name], 0.)
        for name in ('physical_power_residual_w', 'contact_virtual_power_w', 'gauge_virtual_power_w'):
            self.assertLess(abs(d[name])/power, 2e-12)

    def test_global_rotation_covariance(self):
        cfg = fixture(); original = joint.solve(joint.assemble(**cfg))
        axis = unit([.37, -.2, .8]); angle = .83
        cross = joint._cross_matrix(axis)
        q = np.eye(3)+np.sin(angle)*cross+(1-np.cos(angle))*(cross@cross)
        cfg['points'] = cfg['points']@q.T
        cfg['external_torques_n_m'] = cfg['external_torques_n_m']@q.T
        cfg['external_nodal_forces_n'] = cfg['external_nodal_forces_n']@q.T
        cfg['contacts'] = [joint.ContactSample(x.first_face, x.second_face,
            q@x.point, q@x.normal, x.length_m) for x in cfg['contacts']]
        rotated = joint.solve(joint.assemble(**cfg))
        for key in ('plate_omega_rad_s', 'total_velocity_m_s', 'residual_velocity_m_s', 'contact_plate_torque_n_m'):
            self.assertLess(relative(rotated[key], original[key]@q.T), 2e-11, key)

    def test_common_euler_trace_on_unequal_triangles(self):
        s = joint.assemble(**fixture())
        a = np.array([.2, -.3, .5])*1e-9
        y = np.r_[a, a, np.zeros(12)]
        self.assertLess(np.linalg.norm(s['contact_matrix']@y)/np.linalg.norm(a), 2e-15)
        self.assertLess(np.linalg.norm(s['strain_design']@y)**2 /
                        float(y@s['basal_hessian_n_s_m']@y), 2e-29)
        # Negative control: ordinary normalized barycentrics do not preserve
        # this common rotation on two unequal chord planes.
        c = s['contact_point'][0]; n = s['contact_normal'][0]
        velocities = np.cross(a, s['points'])
        wrong = []
        for face in s['faces']:
            alpha = np.linalg.solve(s['points'][face].T, c)
            wrong.append((alpha/alpha.sum())@velocities[face])
        self.assertGreater(abs(n@(wrong[0]-wrong[1]))/np.linalg.norm(a), 1e-4)

    def test_split_quadrature_preserves_velocity_and_aggregate_reaction(self):
        cfg = fixture(); sample = cfg['contacts'][0]; cfg['contacts'] = [sample]
        a = joint.solve(joint.assemble(**cfg))
        cfg['contacts'] = [joint.ContactSample(sample.first_face, sample.second_face,
            sample.point, sample.normal, fraction*sample.length_m) for fraction in (.25, .75)]
        b = joint.solve(joint.assemble(**cfg))
        self.assertLess(relative(a['generalized_velocity_m_s'], b['generalized_velocity_m_s']), 2e-12)
        np.testing.assert_allclose(b['contact_resultant_n'], a['contact_resultant_n'][0]*np.array([.25, .75]), rtol=2e-12)
        self.assertEqual(b['diagnostics']['redundant_rows'], 1)

    def test_residual_sheet_force_has_reciprocal_plate_response(self):
        cfg = fixture(contacts=False); cfg['external_torques_n_m'][:] = 0
        s = joint.assemble(**cfg)
        force = np.zeros((6, 3)); force[:3] = [[0, 2e18, -1e18], [0, -1e18, 0], [0, 0, 1e18]]
        p = cfg['points']; force -= np.einsum('ij,ij->i', force, p)[:, None]*p
        b = s['velocity_map'][:9, :3]
        force[:3] -= (b@np.linalg.solve(b.T@b, b.T@force[:3].ravel())).reshape(3, 3)
        cfg['external_nodal_forces_n'] = force
        free = joint.solve(joint.assemble(**cfg))
        cfg['contacts'] = fixture()['contacts']
        closed = joint.solve(joint.assemble(**cfg))
        self.assertLess(np.linalg.norm(free['plate_omega_rad_s']), 1e-12*np.linalg.norm(closed['plate_omega_rad_s']))
        self.assertGreater(np.linalg.norm(closed['plate_omega_rad_s'][1]), 1e-20)
        self.assertGreater(np.linalg.norm(closed['contact_plate_torque_n_m']), 1e20)

    def test_reference_tangency_is_checked_per_node(self):
        cfg = fixture(); p = cfg['points']
        cfg['basal_reference_velocity_m_s'][0] = np.cross([0, 0, 1e6], p[0])
        cfg['basal_reference_velocity_m_s'][1] = p[1]*1e-9
        with self.assertRaisesRegex(ValueError, 'tangent'):
            joint.assemble(**cfg)

    def test_invalid_finite_trace_rejected(self):
        cfg = fixture(); x = cfg['contacts'][0]
        for sample in (joint.ContactSample(0, 1, x.point, x.normal, 0.),
                       joint.ContactSample(0, 1, unit([1, .7, 0]), unit([-.7, 1, 0]), x.length_m),
                       joint.ContactSample(0, 0, x.point, x.normal, x.length_m)):
            cfg['contacts'] = [sample]
            with self.assertRaises(ValueError):
                joint.assemble(**cfg)

    def test_finite_contact_moves_continuously_through_old_weight_threshold(self):
        width_km = 450.
        radius_km = fixture()['radius_m']/1000.
        threshold_distance = 2*width_km/np.pi*np.arccos(np.sqrt(.9999))
        differences = []
        for offset_km in (.1, .01, .001):
            velocities = []; weights = []
            for side in (-1., 1.):
                angle = (threshold_distance+side*offset_km)/radius_km
                point = np.array([np.cos(angle), np.sin(angle), 0.])
                normal = np.array([-np.sin(angle), np.cos(angle), 0.])
                cfg = fixture()
                cfg['contacts'] = [joint.ContactSample(0, 1, point, normal, 100000.)]
                velocities.append(joint.solve(joint.assemble(**cfg))['generalized_velocity_m_s'])
                belts = dict(mid=point[None], normal=normal[None], length_km=np.array([100.]),
                    owner_a=np.array([0]), owner_b=np.array([1]), values=np.ones((1, 1)))
                weights.append(sample_belts(np.array([[1., 0., 0.]]), np.array([0]),
                    belts, width_km=width_km)[0, 0])
            self.assertGreater(weights[0], .9999)
            self.assertLess(weights[1], .9999)
            differences.append(np.linalg.norm(velocities[1]-velocities[0]))
        # Fixed positive patch and smooth trace; no binary driven-node membership.
        # This does not establish a contact birth/opening law or native trajectory.
        np.testing.assert_allclose(np.array(differences[1:])/differences[:-1], .1, rtol=2e-5)

    def test_nonfinite_power_from_finite_extreme_input_fails_closed(self):
        mesh = mesh_geometry.icosphere(0); p = mesh['vertices']
        with np.errstate(over='ignore', invalid='ignore'):
            system = joint.assemble(p, mesh['faces'], np.zeros(len(p), int), radius_m=1.,
                basal_drag_pa_s_per_m=1., viscosity_pa_s=0., sheet_thickness_m=1.,
                basal_reference_velocity_m_s=np.zeros_like(p),
                other_plate_rotational_drag_n_m_s=np.zeros((3, 3)),
                external_torques_n_m=np.array([[1e160, -.4e160, .2e160]]),
                external_nodal_forces_n=np.zeros_like(p), contacts=[])
            with self.assertRaises(RuntimeError):
                joint.solve(system)

    def test_inconsistent_external_power_ledger_is_rejected(self):
        system = joint.assemble(**fixture())
        # The assembled KKT load is unchanged: this deliberately corrupts only
        # the independently evaluated external-power ledger by one percent.
        system['torque_load_n'] *= 1.01
        with self.assertRaisesRegex(RuntimeError, 'power'):
            joint.solve(system)


if __name__ == '__main__':
    unittest.main()
