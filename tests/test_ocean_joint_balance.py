"""Frozen joint SI balance: sparse work adjoints and common-quadrature oracles."""
from copy import deepcopy
import unittest

import numpy as np
import ocean_joint_balance as joint
import ocean_traction
import viscous_sheet
from tests.test_ocean_traction import fixture


def two_plates(rotation=None):
    """Adjacent connected plates duplicate vertices along their common front."""
    points, faces, owners = [], [], []
    for owner, longitudes in enumerate((np.linspace(-.04, 0., 3), np.linspace(0., .04, 3))):
        lon, lat = np.meshgrid(longitudes, np.linspace(-.02, .02, 3))
        lon, lat = lon.ravel(), lat.ravel()
        block = np.column_stack((np.cos(lat)*np.cos(lon), np.cos(lat)*np.sin(lon), np.sin(lat)))
        start = len(points)
        points.extend(block); owners.extend([owner]*len(block))
        for row in range(2):
            for col in range(2):
                a = start+3*row+col
                faces.extend(((a, a+1, a+4), (a, a+4, a+3)))
    points, faces = np.asarray(points), np.asarray(faces)
    triangles = points[faces]
    numerator = np.abs(np.einsum('ij,ij->i', triangles[:, 0], np.cross(triangles[:, 1], triangles[:, 2])))
    denominator = 1.+sum(np.einsum('ij,ij->i', triangles[:, i], triangles[:, (i+1)%3]) for i in range(3))
    area = 2.*np.arctan2(numerator, denominator)*(6371e3)**2
    down_nodes, over_nodes = np.array([2, 5, 8]), np.array([9, 12, 15])
    normal = np.tile([0., 1., 0.], (3, 1))
    tangent = np.cross(points[down_nodes], normal)
    if rotation is not None:
        points = points@rotation.T; normal = normal@rotation.T; tangent = tangent@rotation.T
    context = ocean_traction.prepare(points, faces, area, radius_m=6371e3,
        viscosity_pa_s=1e21, thickness_m=80e3, basal_drag_pa_s_m=1e15, owner_labels=np.asarray(owners))
    down = joint.velocity_rows(18, 3, np.arange(3), down_nodes, normal)
    over = joint.velocity_rows(18, 3, np.arange(3), over_nodes, normal)
    along = joint.velocity_rows(18, 3, np.tile(np.arange(3), 2), np.r_[down_nodes, over_nodes],
                                np.r_[-tangent, tangent])
    return context, down, over, along


def hinge(context, down, over, along, *, plastic=True, unequal=False):
    mass = np.array([.7, 1., 1.6])*5e11 if unequal else 5e11
    return joint.moving_hinge_terms(down, over, along, length_m=1e5,
        excess_mass_kg_per_m=mass, dip_radians=np.deg2rad(50.), gravity_m_s2=9.81,
        retained_length_scale=.5, slab_stokes_pa_s=3e22, slab_anchor_pa_s=3e22,
        bending_pa_s=1e21 if plastic else 0., megathrust_n_per_m=1.8e12 if plastic else 0.,
        strike_slip_n_per_m=4e11 if plastic else 0., regularization_m_s=1e-10,
        response_version=1, resistance_version=1)


class OceanJointBalanceTests(unittest.TestCase):
    def test_sparse_velocity_force_maps_are_work_adjoint(self):
        rng = np.random.default_rng(4)
        rows = joint.velocity_rows(5, 3, [0, 0, 1, 2, 2], [0, 2, 1, 3, 3], rng.normal(size=(5, 3)))
        velocity, force = rng.normal(size=(5, 3)), rng.normal(size=3)
        self.assertAlmostEqual(float(rows.sample(velocity)@force), float(np.sum(velocity*rows.adjoint(force))), places=13)

    def test_passive_cone_derivatives_match_independent_energy_differences(self):
        rates = np.array([[-.9e-9, .2e-9], [.4e-9, -.1e-9]])
        width = np.array([.1e-9, .15e-9])
        value, first, second = joint._passive_values('cone', tuple(rates), width)

        def potential(x):
            return np.sqrt(np.maximum(-x[0], 0.)**2+x[1]**2+width**2)-width

        np.testing.assert_allclose(value, potential(rates), rtol=1e-14)
        step = 1e-14
        numeric_first, numeric_second = [], []
        for axis in range(2):
            offset = np.zeros_like(rates); offset[axis] = step
            numeric_first.append((potential(rates+offset)-potential(rates-offset))/(2*step))
            plus = joint._passive_values('cone', tuple(rates+offset), width)[1]
            minus = joint._passive_values('cone', tuple(rates-offset), width)[1]
            numeric_second.append((np.asarray(plus)-np.asarray(minus))/(2*step))
        np.testing.assert_allclose(first, numeric_first, rtol=2e-8, atol=2e-10)
        expected_hessian = np.array([[second[0], second[1]], [second[1], second[2]]])
        np.testing.assert_allclose(expected_hessian, numeric_second, rtol=2e-7, atol=1.)

    def test_rigid_restriction_matches_independent_same_quadrature_matrix(self):
        context, down, over, along = two_plates()
        terms = hinge(context, down, over, along, plastic=False, unequal=True)
        result = joint.solve(context, **terms, rigid_only=True)
        points = context['sheet']['points']; area = context['sheet']['data']
        basis = []
        for owner in (0, 1):
            for axis in np.eye(3):
                field = np.cross(axis, points)
                field[context['owner_labels'] != owner] = 0.
                basis.append(field)
        basis = np.asarray(basis)
        matrix = 1e15*np.einsum('ani,bni,n->ab', basis, basis, area)
        for term in terms['quadratics']:
            rates = np.asarray([term['rows'].sample(field) for field in basis])
            matrix += np.einsum('ae,be,e->ab', rates, rates, term['coefficient'])
        load = np.einsum('ani,ni->a', basis, terms['sources'][0]['force_n'])
        expected = np.einsum('a,ani->ni', np.linalg.solve(matrix, load), basis)
        np.testing.assert_allclose(result['total_velocity_m_s'], expected, rtol=2e-7, atol=1e-19)
        self.assertLess(np.linalg.norm(result['internal_velocity_m_s']), 1e-19)
        self.assertFalse(result['diagnostics']['production_discretization_equivalent'])

    def test_opposed_zero_torque_loads_drive_internal_motion_in_joint_solve(self):
        context, force = fixture()
        result = joint.solve(context, sources=[dict(name='opposed', force_n=force)])
        reference = ocean_traction.solve(context, force)
        self.assertGreater(np.linalg.norm(result['internal_velocity_m_s']), 0.)
        np.testing.assert_allclose(result['internal_velocity_m_s'], reference['internal_velocity_m_s'],
                                   rtol=2e-6, atol=1e-19)
        self.assertLess(np.linalg.norm(result['rigid_velocity_m_s']), 1e-18)
        d = result['diagnostics']
        self.assertLessEqual(abs(d['power_residual_w']), d['power_acceptance_bound_w'])

    def test_large_rigid_load_does_not_hide_resolvable_internal_motion(self):
        context, force = fixture()
        points, area = context['sheet']['points'], context['sheet']['data']
        rigid = np.cross(np.array([0., 0., 1e-9]), points)
        small_force = force*1e-10
        baseline = joint.solve(context, sources=[dict(name='small', force_n=small_force)])
        result = joint.solve(context, sources=[dict(name='large_and_small',
            force_n=1e15*area[:, None]*rigid+small_force)])
        error = np.linalg.norm(result['internal_velocity_m_s']-baseline['internal_velocity_m_s'])
        self.assertLess(error/np.linalg.norm(baseline['internal_velocity_m_s']), .001)
        # This load could disappear entirely inside a global force tolerance.
        # The separate internal gate demands the additional refinement.
        small_norm = np.linalg.norm(small_force/context['projection'].root_area[:, None])
        gates = result['diagnostics']['force_mode_residuals']
        self.assertLess(small_norm, gates[0]['acceptance_n_m_inv'])
        self.assertGreater(small_norm, gates[1]['acceptance_n_m_inv']*100.)
        self.assertGreaterEqual(result['diagnostics']['newton_iterations'], 2)

    def test_response_fingerprints_bind_context_sources_and_every_boundary_law(self):
        context, down, over, along = two_plates()
        terms = hinge(context, down, over, along, unequal=True)

        def fingerprint(value):
            return joint._terms_fingerprint(*joint._terms(context, **value))

        result = joint.solve(context, **terms)
        self.assertEqual(result['context_fingerprint'], context['context_fingerprint'])
        expected = result['frozen_terms_fingerprint']
        self.assertEqual(expected, fingerprint(deepcopy(terms)))
        for changed in ('source_name', 'source_force', 'quadratic_coefficient', 'row_vector',
                        'row_node', 'row_index', 'passive_coefficient', 'width', 'shape'):
            other = deepcopy(terms)
            if changed == 'source_name': other['sources'][0]['name'] += '_other'
            elif changed == 'source_force': other['sources'][0]['force_n'][0, 0] += 1.
            elif changed == 'quadratic_coefficient': other['quadratics'][0]['coefficient'][0] *= 2.
            elif changed.startswith('row_'):
                original = other['quadratics'][0]['rows']
                row, node, vector = original.row.copy(), original.node.copy(), original.vector.copy()
                if changed == 'row_vector': vector[0] *= 2.
                elif changed == 'row_node': node[0] = (node[0]+1) % original.node_count
                else: row[0] = (row[0]+1) % original.count
                other['quadratics'][0]['rows'] = joint.velocity_rows(original.node_count,
                    original.count, row, node, vector)
            elif changed == 'passive_coefficient': other['passive'][1]['coefficient'][0] *= 2.
            elif changed == 'width': other['passive'][1]['width_m_s'][0] *= 2.
            else: other['passive'][2]['shape'] = 'negative'
            with self.subTest(changed=changed):
                self.assertNotEqual(expected, fingerprint(other))

    def test_total_boundary_velocities_produce_coupled_resistance_and_one_work_budget(self):
        context, down, over, along = two_plates()
        terms = hinge(context, down, over, along, unequal=True)
        result = joint.solve(context, **terms)
        d = result['diagnostics']; velocity = result['total_velocity_m_s']
        self.assertEqual(set(d['source_work_w']), {'slab_gravity'})
        self.assertGreater(np.linalg.norm(result['internal_velocity_m_s']), 0.)
        self.assertGreater(np.linalg.norm(result['rigid_velocity_m_s']), 0.)
        # Every resistance is evaluated on the same total sampled velocity.
        for term in terms['quadratics']:
            rates = term['rows'].sample(velocity)
            expected = float(term['coefficient']@(rates*rates))
            self.assertAlmostEqual(d['quadratic_dissipation_w'][term['name']]/expected, 1., places=12)
        horizontal = terms['quadratics'][0]
        a = horizontal['rows'].sample(result['rigid_velocity_m_s'])
        b = horizontal['rows'].sample(result['internal_velocity_m_s'])
        cross = float(2.*horizontal['coefficient']@(a*b))
        self.assertGreater(abs(cross), d['total_source_work_w']*1e-5,
                           'Fixture must expose a cross-block omitted by sequential solves.')
        self.assertTrue(all(value >= 0 for value in d['passive_resistance_work_w'].values()))
        self.assertLessEqual(abs(d['power_residual_w']), d['power_acceptance_bound_w'])
        self.assertAlmostEqual((d['rigid_source_work_w']+d['internal_source_work_w'])/d['total_source_work_w'], 1., places=12)

    def test_zero_force_and_basal_supported_common_rotation(self):
        context, force = fixture()
        zero = joint.solve(context, sources=[])
        np.testing.assert_array_equal(zero['total_velocity_m_s'], 0.)
        points = context['sheet']['points']; area = context['sheet']['data']
        desired = np.cross(np.array([2e-10, -3e-10, 1e-9]), points)
        force = 1e15*area[:, None]*desired
        result = joint.solve(context, sources=[dict(name='supports_basal_drag', force_n=force)])
        np.testing.assert_allclose(result['total_velocity_m_s'], desired, rtol=2e-7, atol=1e-18)
        self.assertLess(np.linalg.norm(result['strain_rate_per_s']), 1e-25)
        self.assertLess(result['diagnostics']['membrane_dissipation_w'], result['diagnostics']['basal_dissipation_w']*1e-15)

    def test_frame_rotation_preserves_motion_and_work(self):
        angle = .71
        rotation = np.array([[np.cos(angle), 0., np.sin(angle)], [0., 1., 0.], [-np.sin(angle), 0., np.cos(angle)]])
        a = two_plates(); b = two_plates(rotation)
        first = joint.solve(a[0], **hinge(*a, unequal=True))
        other = joint.solve(b[0], **hinge(*b, unequal=True))
        np.testing.assert_allclose(other['total_velocity_m_s'], first['total_velocity_m_s']@rotation.T,
                                   rtol=5e-6, atol=2e-18)
        self.assertAlmostEqual(other['diagnostics']['total_source_work_w']/first['diagnostics']['total_source_work_w'], 1., places=6)

    def test_nonconvergence_leaves_context_and_sources_unchanged(self):
        context, down, over, along = two_plates()
        terms = hinge(context, down, over, along, unequal=True)
        digest = ocean_traction._context_fingerprint(context)
        forces = deepcopy(terms['sources'])
        with self.assertRaisesRegex(ValueError, 'CG did not reach'):
            joint.solve(context, **terms, max_cg_iterations=1)
        self.assertEqual(ocean_traction._context_fingerprint(context), digest)
        np.testing.assert_array_equal(terms['sources'][0]['force_n'], forces[0]['force_n'])

    def test_invalid_versions_and_duplicate_source_labels_fail_closed(self):
        context, down, over, along = two_plates()
        terms = hinge(context, down, over, along)
        with self.assertRaisesRegex(ValueError, 'unique'):
            joint.solve(context, sources=terms['sources']*2)
        with self.assertRaisesRegex(ValueError, 'Only moving-hinge v1'):
            joint.moving_hinge_terms(down, over, along, length_m=1e5, excess_mass_kg_per_m=5e11,
                dip_radians=.8, gravity_m_s2=9.81, retained_length_scale=.5, slab_stokes_pa_s=3e22,
                slab_anchor_pa_s=3e22, bending_pa_s=1e21, megathrust_n_per_m=1.8e12,
                strike_slip_n_per_m=4e11, regularization_m_s=1e-10, response_version=0, resistance_version=1)


if __name__ == '__main__':
    unittest.main()
