"""Finite basal trace, measure, moving-reference and replacement checks."""
from pathlib import Path
import sys
import unittest
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from benchmarks.collision_architecture import basal_partition as partition
from benchmarks.collision_architecture import finite_basal_operator as basal
from benchmarks.collision_architecture import sparse_shared_contact as sparse

R = 6371000.
BETA = 2.5e-3
C = np.eye(3)
MID = np.array([0., 1., 1.])/np.sqrt(2.)
HALF = np.array([C[0], C[1], MID])
OTHER = np.array([C[0], MID, C[2]])
AREA = .5*np.pi*R**2
METRIC = np.full((3, 3), -R**2/3.)
np.fill_diagonal(METRIC, 2*AREA/3.)


def system(*, stacked=False, count=1, beta_old=1e15, eta=0., other=0.):
    n = 2 if stacked else count
    points = np.tile(C, (n, 1)); faces = np.arange(3*n).reshape(-1, 3)
    owner = np.repeat(np.arange(n), 3) if stacked else np.zeros(3*n, dtype=int)
    plates = 2 if stacked else 1
    return sparse.assemble(points, faces, owner, radius_m=R,
        basal_drag_pa_s_per_m=beta_old, viscosity_pa_s=eta, sheet_thickness_m=4e4,
        basal_reference_velocity_m_s=np.zeros_like(points),
        other_plate_rotational_drag_n_m_s=np.eye(3*plates)*other,
        external_torques_n_m=np.zeros((plates, 3)), external_nodal_forces_n=np.zeros_like(points), contacts=[])


def allocation(*, control=C, saved=AREA, material=True, stacked=False, density=1., support=None, policy='preserve-native'):
    records = [dict(triangle=C, face_id=10, sheet_id=1, owner=20)] if material else []
    if stacked:
        records.append(dict(triangle=C, face_id=11, sheet_id=2, owner=21))
    return partition.partition_cell(control, records, [(2, 1)] if stacked else [],
        radius_m=R, saved_cell_area_m2=saved*density,
        support={20:1., 21:0.} if support is None else support,
        basal_drag_pa_s_per_m=BETA, allocation_policy=policy)


def build(s, allocations=None, mantle=(0.,0.,0.), **kwargs):
    return basal.build(s, [allocation()] if allocations is None else allocations,
        {10:0, **({11:1} if s['plate_count']==2 else {})},
        owner_to_plate_slot={20:0, **({21:1} if s['plate_count']==2 else {})},
        mantle_omega_rad_s=np.asarray(mantle), quadrature_relative_tolerance=kwargs.get('tolerance', 2e-10),
        max_order=kwargs.get('max_order', 64))


class FiniteBasalOperatorTests(unittest.TestCase):
    def test_rigid_octant_matches_independent_analytic_metric_and_SI(self):
        s = system(); component = build(s)
        y = np.zeros(len(s['load_n'])); y[:3] = [.003, -.002, .007]
        expected = BETA*y[:3]@METRIC@y[:3]
        self.assertAlmostEqual(component.work(y)['basal_dissipation_w']/expected, 1., places=10)
        np.testing.assert_allclose((component.hessian_n_s_m@y)[:3], BETA*METRIC@y[:3], rtol=2e-10)
        self.assertLess(component.diagnostics[0]['quadrature'][-1]['full_rigid_metric_relative_error'], 2e-10)

    def test_moving_mantle_force_and_actual_power_share_one_measure(self):
        s = system(); mantle = np.array([.001, .002, -.003])/R
        component = build(s, mantle=mantle)
        y = np.zeros(len(s['load_n'])); y[:3] = [.003, -.002, .007]
        slip = y[:3]-R*mantle; work = component.work(y)
        self.assertAlmostEqual(work['basal_dissipation_w']/(BETA*slip@METRIC@slip), 1., places=10)
        self.assertAlmostEqual(work['basal_reference_input_power_w']/(-BETA*slip@METRIC@(R*mantle)), 1., places=10)
        np.testing.assert_allclose(component.load_n[:3], BETA*METRIC@(R*mantle), rtol=2e-10)
        self.assertAlmostEqual((work['basal_dissipation_w']-work['basal_reference_input_power_w'])/work['generalized_gradient_work_w'], 1., places=14)
        y[:3] = R*mantle
        self.assertLess(component.work(y)['basal_dissipation_w'], 1e-23)

    def test_original_radial_trace_reproduces_common_Euler_in_residual_space(self):
        s = system(); component = build(s)
        rotation = np.array([.003, -.002, .007]); y = np.zeros(len(s['load_n']))
        y[:3] = rotation
        rigid_power = component.work(y)['basal_dissipation_w']
        velocities = np.cross(rotation, s['points'])
        y[:3] = 0.
        y[s['nplate']:] = np.einsum('nij,ni->nj', s['tangent_basis'], velocities).ravel()
        self.assertAlmostEqual(component.work(y)['basal_dissipation_w']/rigid_power, 1., places=13)

    def test_control_subdivision_preserves_full_material_Gram_and_mantle_load(self):
        s = system(); mantle = [1e-10, -3e-10, 2e-10]
        whole = build(s, mantle=mantle)
        split = build(s, [allocation(control=HALF, saved=AREA/2), allocation(control=OTHER, saved=AREA/2)], mantle=mantle)
        a = whole.hessian_n_s_m.to_dense(); b = split.hessian_n_s_m.to_dense()
        scale = np.sqrt(np.diag(a)); denominator = scale[:,None]*scale[None]
        self.assertLess(np.linalg.norm((a-b)/denominator, 2), 3e-10)
        np.testing.assert_allclose(split.load_n, whole.load_n, rtol=3e-10, atol=1e-7)
        self.assertAlmostEqual(split.constant_w/whole.constant_w, 1., places=10)

    def test_upper_layer_gets_no_extra_basal_drag(self):
        s = system(stacked=True); component = build(s, [allocation(stacked=True)])
        y = np.zeros(len(s['load_n'])); y[3:6] = [1.,2.,3.]; y[s['nplate']+6:] = 5.
        self.assertEqual(component.work(y)['basal_dissipation_w'], 0.)
        np.testing.assert_array_equal(component.hessian_n_s_m@y, np.zeros_like(y))
        y[:3] = [.1,.2,.3]
        expected = BETA*y[:3]@METRIC@y[:3]
        self.assertAlmostEqual(component.work(y)['basal_dissipation_w']/expected, 1., places=10)

    def test_uncovered_fractional_plate_measure_is_analytic_and_moves_with_mantle(self):
        s = system(stacked=True)
        a = allocation(material=False, support={20:.25,21:.75})
        mantle = np.array([.001,-.002,.003])/R
        component = build(s, [a], mantle=mantle)
        y = np.zeros(len(s['load_n'])); y[:6] = [.003,-.001,.002, -.002,.006,.001]
        expected = sum(weight*BETA*(y[3*p:3*p+3]-R*mantle)@METRIC@(y[3*p:3*p+3]-R*mantle)
                       for p,weight in ((0,.25),(1,.75)))
        self.assertAlmostEqual(component.work(y)['basal_dissipation_w']/expected, 1., places=14)
        self.assertTrue(all(row['kind']=='uncovered-rigid' for row in component.diagnostics))

    def test_saved_area_density_and_explicit_resolved_policy_are_retained(self):
        s = system(stacked=True)
        base = build(s, [allocation(stacked=True)])
        changed = build(s, [allocation(stacked=True, density=1.125, support={20:.25,21:.75}, policy='resolved-material-bottom')])
        y = np.random.default_rng(9).normal(size=len(s['load_n']))
        self.assertAlmostEqual(changed.work(y)['basal_dissipation_w']/base.work(y)['basal_dissipation_w'], 1.125, places=13)
        self.assertEqual(changed.diagnostics[0]['policy'],'resolved-material-bottom')

    def test_gradient_is_work_conjugate_and_Gram_is_positive(self):
        s = system(); component = build(s, mantle=[1e-10,2e-10,-1e-10])
        rng = np.random.default_rng(200); y = rng.normal(size=len(s['load_n']))*.001
        direction = rng.normal(size=len(y)); value, gradient = component.evaluate(y)
        h = 1e-7
        finite_difference = (component.evaluate(y+h*direction)[0]-component.evaluate(y-h*direction)[0])/(2*h)
        self.assertAlmostEqual(finite_difference/(gradient@direction), 1., places=8)
        self.assertGreater(value, 0.)
        self.assertGreaterEqual(y@(component.hessian_n_s_m@y),0.)
        np.testing.assert_allclose(component.diagonal_n_s_m, np.diag(component.hessian_n_s_m.to_dense()), rtol=2e-15)
        self.assertTrue(np.all(abs(component.hessian_n_s_m@y) <= component.absolute_action(y)*(1+1e-14)))
        cancelled = np.zeros_like(y); cancelled[:3]=[.001,-.002,.003]
        cancelled[s['nplate']:] = np.einsum('nij,ni->nj',s['tangent_basis'],
            -np.cross(cancelled[:3],s['points'])).ravel()
        zero_reference = build(s)
        cost = zero_reference.work(cancelled)
        self.assertGreater(cost['arithmetic_constituent_work_bound_w'],1.)
        self.assertLess(cost['basal_dissipation_w'],cost['arithmetic_constituent_work_bound_w']*1e-27)

    def test_replacement_removes_old_nodal_drag_and_closes_total_physical_ledger(self):
        s = system(eta=700., other=1e18); component = build(s, mantle=[1e-10,2e-10,-1e-10])
        changed = basal.replace_basal(s, component, other_drag_excludes_allocated_basal=True)
        y = np.random.default_rng(20).normal(size=len(s['load_n']))*.001
        expected = component.hessian_n_s_m@y+s['strain_design'].rmatvec(s['strain_design']@y)
        expected[:s['nplate']] += s['other_plate_drag_factor'].T@(s['other_plate_drag_factor']@y[:s['nplate']])
        np.testing.assert_allclose(changed['hessian_n_s_m']@y, expected, rtol=2e-15)
        self.assertNotIn('basal_drag_weight_n_s_m', changed)
        self.assertIn('basal_drag_weight_n_s_m', s)
        work = changed['physical_work'](y)
        self.assertAlmostEqual(work['unsolved_power_defect_w']/work['generalized_gradient_work_w'], 1., places=13)
        self.assertGreaterEqual(changed['physical_work_absolute_scale_w'](y), work['basal_dissipation_w'])
        different = system(eta=700.,other=1e18,beta_old=3.)
        again = basal.replace_basal(different,component,other_drag_excludes_allocated_basal=True)
        np.testing.assert_array_equal(again['hessian_n_s_m']@y,changed['hessian_n_s_m']@y)
        with self.assertRaises(ValueError): basal.replace_basal(s,component,other_drag_excludes_allocated_basal=False)
        with self.assertRaises(ValueError): basal.replace_basal(changed,component,other_drag_excludes_allocated_basal=True)

    def test_binding_mapping_duplicate_and_arithmetic_errors_fail_closed(self):
        s = system(); component = build(s)
        wrong = dict(s); wrong['vertex_plate'] = np.ones(3,dtype=int)
        with self.assertRaises(ValueError): component.validate_system(wrong)
        with self.assertRaisesRegex(ValueError,'Repeated'): build(s,[allocation(),allocation()])
        corrupt = allocation(); corrupt['pieces'][0]['area_m2'] *= 1.001
        with self.assertRaises(ValueError): build(s,[corrupt])
        huge = allocation(); huge['basal_drag_pa_s_per_m']=1e300
        with self.assertRaises(ValueError): build(s,[huge])
        with self.assertRaises(ValueError): build(s,mantle=[complex(1,2),0,0])
        with self.assertRaises(RuntimeError): build(s,tolerance=1e-14,max_order=16)

    def test_large_unknown_space_keeps_only_local_nine_column_factors(self):
        # Layout/memory test only: the caller supplies one covered domain, not a
        # claimed global nonoverlap or completeness certificate for these faces.
        s = system(count=90); component = build(s)
        self.assertGreater(len(s['load_n']),256)
        self.assertEqual(component.basal_design.blocks.shape,(1,9,9))
        self.assertEqual(component.basal_design.columns.shape,(1,9))
        with self.assertRaises(ValueError): component.hessian_n_s_m.to_dense()
        y = np.zeros(len(s['load_n'])); y[:3]=[.001,.002,.003]
        self.assertTrue(np.isfinite(component.hessian_n_s_m@y).all())

    def test_actual_finite_basal_weld_interface_and_sparse_solver_share_ledger(self):
        from collision_interface import _precise_rotation_integral
        from benchmarks.collision_architecture import sparse_joint_solver as solver
        from tests.test_local_joint_resistance import sparse_fixture, resistances, patch_parameters
        s = sparse_fixture(); polygon = patch_parameters()['polygon']
        # This assembled replacement has no independent upper-plate drag. The
        # upper Euler degrees of freedom obtain resistance through the actual
        # finite interface, not an added diagonal or artificial mantle contact.
        s = sparse.assemble(s['points'],s['faces'],s['vertex_plate'],radius_m=R,
            basal_drag_pa_s_per_m=1e15,viscosity_pa_s=2e21,sheet_thickness_m=4e4,
            basal_reference_velocity_m_s=np.zeros_like(s['points']),
            other_plate_rotational_drag_n_m_s=np.zeros((6,6)),
            external_torques_n_m=np.array([[0,0,1e25],[0,0,-1e25]]),
            external_nodal_forces_n=np.zeros_like(s['points']),contacts=[])
        area = _precise_rotation_integral(polygon)[1]*R**2
        records = [dict(triangle=s['points'][face], face_id=10+j, sheet_id=j+1, owner=20+j)
                   for j,face in enumerate(s['faces'])]
        a = partition.partition_cell(polygon,records,[(1,2)],radius_m=R,saved_cell_area_m2=area,
            support={20:.5,21:.5},basal_drag_pa_s_per_m=1e15,allocation_policy='resolved-material-bottom')
        component = build(s,[a],mantle=[1e-17,-2e-17,1e-17])
        np.testing.assert_array_equal(component.diagonal_n_s_m[:3],0.)
        updated = basal.replace_basal(s,component,other_drag_excludes_allocated_basal=True)
        resistance,_ = resistances(updated)
        solved = solver.solve(updated,resistance=resistance)
        d = solved['diagnostics']; y = solved['generalized_velocity_m_s']
        self.assertLessEqual(np.max(d['componentwise_stationarity_relative']),1e-10)
        self.assertLessEqual(abs(d['physical_power_residual_w']),d['physical_power_acceptance_allowance_w'])
        self.assertGreater(d['resistance_dissipation_w'],0.)
        work = updated['physical_work'](y)
        self.assertGreater(work['basal_dissipation_w'],0.)
        defect = work['unsolved_power_defect_w']+resistance.evaluate(y)['resisting_power_w']
        self.assertLessEqual(abs(defect),d['physical_power_acceptance_allowance_w'])


if __name__=='__main__': unittest.main()
