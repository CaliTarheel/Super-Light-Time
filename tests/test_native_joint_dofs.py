"""Full Euler slots and explicit anchored residual frames, without native activation."""
import unittest
import numpy as np

from benchmarks.collision_architecture import sparse_shared_contact as sparse
from benchmarks.collision_architecture import sparse_joint_solver as solver
from tests.test_shared_contact import fixture, relative, unit
from tests.test_sparse_shared_contact import spherical_fixture


def small_config(copies=1, extra_plates=0):
    p = np.tile(np.eye(3), (copies, 1))
    count = copies+extra_plates
    return dict(points=p, faces=np.arange(len(p)).reshape(-1, 3), vertex_plate=np.repeat(np.arange(copies), 3),
        plate_count=count, radius_m=3., basal_drag_pa_s_per_m=2., viscosity_pa_s=.5, sheet_thickness_m=1.,
        basal_reference_velocity_m_s=np.zeros_like(p), other_plate_rotational_drag_n_m_s=np.eye(3*count)*7.,
        external_torques_n_m=np.arange(1., 3*count+1.).reshape(-1, 3),
        external_nodal_forces_n=np.random.default_rng(131).normal(size=p.shape), contacts=[])


def explicit_kkt(system):
    """Independent small full-space KKT with every fixed coordinate as a row."""
    n = len(system['load_n']); cut = system['nplate']
    k = system['hessian_n_s_m'].to_dense()
    c = np.r_[system['gauge_matrix'].to_dense(),
        system['contact_weights'][:, None]*system['contact_matrix'].to_dense()]
    c = c[np.any(c != 0., axis=1)]
    fixed = np.flatnonzero(np.r_[np.zeros(cut, bool), system['fixed_residual_dofs'].ravel()])
    c = np.r_[c, np.eye(n)[fixed]]
    # This manufactured fixture has full independent constraint rows. Scale
    # rows and unknowns for a direct KKT solve, without solver projection code.
    scale = 1/np.sqrt(np.diag(k)); a = c*scale
    row_scale = np.linalg.norm(a, axis=1); a = a/row_scale[:, None]
    matrix = np.block([[scale[:, None]*k*scale[None, :], a.T], [a, np.zeros((len(a), len(a)))]] )
    solved = np.linalg.solve(matrix, np.r_[scale*system['load_n'], np.zeros(len(a))])
    y = scale*solved[:n]; lam = solved[n:]/row_scale
    fixed_reaction = np.zeros(n)
    if len(fixed): fixed_reaction[fixed] = -lam[-len(fixed):]
    return y, fixed_reaction


class NativeJointDofs(unittest.TestCase):
    def test_anchor_compatible_exchange_dimensions_and_ocean_only_zero_rows(self):
        cfg = small_config(4, 1); mask = np.zeros((12, 2), bool)
        mask[3] = True; mask[6:8] = True; mask[9, 0] = True
        cfg['fixed_residual_dofs'] = mask
        system = sparse.assemble(**cfg)
        np.testing.assert_array_equal(system['gauge_exchange_dimension'], [3, 1, 0, 2, 0])
        np.testing.assert_array_equal(system['material_plate_present'], [True, True, True, True, False])
        rotation = sparse._rotation_maps(system['points'])
        exchange = -np.einsum('nia,nij->naj', system['tangent_basis'], rotation)
        for plate, dimension in enumerate(system['gauge_exchange_dimension']):
            selected = system['vertex_plate'] == plate
            basis = system['gauge_exchange_basis'][plate, :, :dimension]
            np.testing.assert_allclose(exchange[selected][mask[selected]]@basis, 0., atol=2e-15)
        result = solver.solve(system, maximum_constraint_rows=6)
        self.assertEqual(result['diagnostics']['active_constraint_rows'], 6)
        self.assertEqual(result['diagnostics']['fixed_residual_coordinates'], 7)
        self.assertEqual(result['diagnostics']['constraint_rank'], 6)

    def test_ocean_only_forced_Euler_balance_has_no_material_gauge(self):
        cfg = small_config(1, 1)
        cfg['vertex_plate'][:] = 1  # Explicit slot zero has no material.
        cfg['external_torques_n_m'][0] = [5., -2., 3.]
        result = solver.solve(sparse.assemble(**cfg))
        np.testing.assert_allclose(result['plate_omega_rad_s'][0], cfg['external_torques_n_m'][0]/7., rtol=2e-12)
        self.assertEqual(result['diagnostics']['active_constraint_rows'], 3)
        self.assertEqual(result['diagnostics']['constraint_rows'], 6)

    def test_exact_elimination_matches_full_KKT_and_recovers_frame_reactions(self):
        for mask in (np.array([[True, True], [False, False], [False, False]]),
                     np.array([[True, False], [False, False], [False, False]]),
                     np.array([[True, True], [True, True], [False, False]]),
                     np.ones((3, 2), bool)):
            with self.subTest(mask=mask.tolist()):
                cfg = small_config(); cfg['fixed_residual_dofs'] = mask
                system = sparse.assemble(**cfg); expected, reaction = explicit_kkt(system)
                result = solver.solve(system)
                self.assertLess(relative(result['generalized_velocity_m_s'], expected), 2e-11)
                self.assertLess(relative(result['fixed_generalized_reaction_n'], reaction), 2e-11)
                np.testing.assert_array_equal(result['generalized_velocity_m_s'][3:].reshape(-1, 2)[mask], 0.)
                self.assertGreater(np.linalg.norm(result['fixed_nodal_force_n']), .01)
                torque = -cfg['radius_m']*np.cross(system['points'], result['fixed_nodal_force_n']).sum(axis=0)
                np.testing.assert_allclose(result['fixed_plate_torque_n_m'][0], torque, rtol=2e-14)
                d = result['diagnostics']
                self.assertEqual(d['fixed_virtual_power_w'], 0.)
                self.assertLessEqual(abs(d['fixed_exchange_power_error_w']), d['fixed_exchange_power_allowance_w'])
                self.assertLessEqual(abs(d['physical_power_residual_w']), d['physical_power_acceptance_allowance_w'])

    def test_many_fixed_coordinates_do_not_consume_constraint_row_budget(self):
        cfg = spherical_fixture(3); mask = np.ones((len(cfg['points']), 2), bool)
        cfg['fixed_residual_dofs'] = mask
        system = sparse.assemble(**cfg); result = solver.solve(system, maximum_constraint_rows=1)
        self.assertGreater(result['diagnostics']['fixed_residual_coordinates'], 512)
        self.assertEqual(result['diagnostics']['active_constraint_rows'], 0)
        self.assertEqual(result['diagnostics']['free_unknowns'], 3)
        np.testing.assert_array_equal(result['residual_velocity_m_s'], 0.)
        p = system['points']; area = system['nodal_area_m2']; r = cfg['radius_m']
        matrix = cfg['basal_drag_pa_s_per_m']*r*r*(np.eye(3)*area.sum()-np.einsum('n,ni,nj->ij', area,p,p))
        matrix += cfg['other_plate_rotational_drag_n_m_s']
        np.testing.assert_allclose(result['plate_omega_rad_s'][0], np.linalg.solve(matrix,cfg['external_torques_n_m'][0]), rtol=2e-11)

    def test_unresolved_independent_anchor_direction_fails_closed(self):
        cfg = small_config(2); cfg['vertex_plate'][:] = 0
        cfg['points'][3] = unit([1, 1e-14, 0])
        cfg['fixed_residual_dofs'] = np.zeros((6, 2), bool)
        cfg['fixed_residual_dofs'][[0, 3]] = True
        with self.assertRaisesRegex(ValueError, 'Anchor exchange rank is numerically unresolved'):
            sparse.assemble(**cfg)

    def test_explicit_slots_and_masks_validate_without_implicit_anchors(self):
        cfg = small_config(1, 1); cfg['plate_count'] = 1.5
        with self.assertRaises(ValueError): sparse.assemble(**cfg)
        cfg = small_config(); cfg['fixed_residual_dofs'] = np.ones((3, 2), int)
        with self.assertRaisesRegex(ValueError, 'Boolean'): sparse.assemble(**cfg)
        cfg = small_config(1, 1); cfg['vertex_plate'][:] = 1; del cfg['plate_count']
        with self.assertRaisesRegex(ValueError, 'Inferred plate slots'): sparse.assemble(**cfg)
        system = sparse.assemble(**small_config()); system['fixed_residual_dofs'][0] = True
        with self.assertRaisesRegex(ValueError, 'changed after its frame gauge'): solver.solve(system)

    def test_native_represented_geometry_is_preserved_only_when_requested(self):
        cfg = small_config(); cfg['points'][0, 0] = np.nextafter(1., 2.)
        default = sparse.assemble(**cfg)
        cfg['preserve_represented_points'] = True
        native = sparse.assemble(**cfg)
        np.testing.assert_array_equal(native['points'], cfg['points'])
        self.assertNotEqual(default['points'][0, 0], cfg['points'][0, 0])
        cfg['points'][0, 0] = 1.+1e-8
        with self.assertRaisesRegex(ValueError, 'unit spherical'): sparse.assemble(**cfg)

    def test_existing_fixed_contacts_use_full_physical_rows(self):
        cfg = fixture(); mask = np.zeros((6, 2), bool); mask[[0, 3]] = True
        cfg['fixed_residual_dofs'] = mask
        system = sparse.assemble(**cfg); expected, reaction = explicit_kkt(system)
        result = solver.solve(system)
        self.assertLess(relative(result['generalized_velocity_m_s'], expected), 2e-9)
        self.assertLess(relative(result['fixed_generalized_reaction_n'], reaction), 2e-9)
        self.assertLessEqual(np.max(result['diagnostics']['original_row_relative_error']), 1e-10)

    def test_finite_basal_replacement_resists_ocean_slot_and_attached_material(self):
        from benchmarks.collision_architecture import basal_partition as partition
        from benchmarks.collision_architecture import finite_basal_operator as basal
        from tests.test_finite_basal_operator import allocation, R, BETA, AREA
        cfg = small_config(1, 1); cfg['radius_m'] = R
        cfg['fixed_residual_dofs'] = np.ones((3, 2), bool)
        cfg['other_plate_rotational_drag_n_m_s'][:] = 0.
        cfg['external_torques_n_m'] *= 1e15; cfg['external_nodal_forces_n'][:] = 0.
        cfg['viscosity_pa_s'] = 0.
        system = sparse.assemble(**cfg)
        ocean = partition.partition_cell(-np.eye(3)[[0,2,1]], [], [], radius_m=R,
            saved_cell_area_m2=AREA, support={21:1.}, basal_drag_pa_s_per_m=BETA,
            allocation_policy='preserve-native')
        component = basal.build(system, [allocation(), ocean], {10:0}, owner_to_plate_slot={20:0,21:1},
            mantle_omega_rad_s=np.zeros(3), quadrature_relative_tolerance=2e-10, max_order=64)
        replaced = basal.replace_basal(system, component, other_drag_excludes_allocated_basal=True)
        result = solver.solve(replaced)
        expected = np.linalg.solve(component.hessian_n_s_m.to_dense()[:6,:6], replaced['load_n'][:6])
        np.testing.assert_allclose(result['generalized_velocity_m_s'][:6], expected, rtol=2e-11)
        self.assertEqual(result['diagnostics']['other_drag_dissipation_w'], 0.)
        self.assertEqual(result['diagnostics']['active_constraint_rows'], 0)
        self.assertGreater(result['diagnostics']['basal_dissipation_w'], 0.)

    def test_finite_nonlinear_laws_preserve_fixed_frame_equilibrium(self):
        from tests.test_joint_resistance import fixture as geometry_fixture
        from tests.test_local_joint_resistance import arc_parameters, patch_parameters
        from benchmarks.collision_architecture import local_joint_resistance as laws
        original = geometry_fixture(); cfg = fixture(contacts=False)
        cfg['points'] = original['points']; cfg['other_plate_rotational_drag_n_m_s'] = np.eye(6)*1e39
        cfg['fixed_residual_dofs'] = np.zeros((6,2), bool); cfg['fixed_residual_dofs'][[0,3]] = True
        system = sparse.assemble(**cfg); geometry = laws.Geometry(system)
        resistance = laws.Resistance(system, welds=[geometry.weld_arc(**arc_parameters())],
            interfaces=[geometry.interface_patch(**patch_parameters())], smoothing_speed_m_s=2e-9)
        result = solver.solve(system, resistance=resistance)
        self.assertGreater(result['diagnostics']['resistance_dissipation_w'], 0.)
        self.assertGreater(np.linalg.norm(result['fixed_generalized_reaction_n']), 0.)
        self.assertLessEqual(np.max(result['diagnostics']['componentwise_stationarity_relative']), 1e-10)
        self.assertEqual(result['diagnostics']['fixed_virtual_power_w'], 0.)


if __name__ == '__main__': unittest.main()
