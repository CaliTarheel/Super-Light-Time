"""Frozen joint solves; no native constitutive/contact-admission validation."""
import unittest
import gc
import hashlib
import json
from pathlib import Path
import time
import tracemalloc
import numpy as np

from benchmarks.collision_architecture import shared_contact as dense
from benchmarks.collision_architecture import sparse_shared_contact as sparse
from benchmarks.collision_architecture import sparse_joint_solver as solver
from tests.test_shared_contact import fixture, relative, unit
from tests.test_sparse_shared_contact import spherical_fixture


class Quartic:
    def __init__(self, size, coefficient=1e45):
        self.size = size; self.coefficient = coefficient
    def evaluate(self, y):
        if y.flags.writeable: raise AssertionError('Trial must be readonly.')
        diagonal = 3*self.coefficient*y*y
        class Hessian:
            def __matmul__(self, x): return diagonal*x
        return dict(potential_w=.25*self.coefficient*float(np.sum(y**4)),
            gradient_n=self.coefficient*y**3, resisting_power_w=self.coefficient*float(np.sum(y**4)),
            hessian_n_s_m=Hessian(), diagonal_n_s_m=diagonal, diagnostics={})


class SparseJointSolver(unittest.TestCase):
    def check_acceptance(self, result):
        d = result['diagnostics']
        self.assertLessEqual(np.max(d['componentwise_stationarity_relative']), 1e-10)
        self.assertLessEqual(np.max(d['original_row_relative_error']), 1e-10)
        self.assertLessEqual(abs(d['physical_power_residual_w']), d['physical_power_acceptance_allowance_w'])
    def test_small_linear_solution_and_contact_reactions_match_dense(self):
        cfg = fixture(); expected = dense.solve(dense.assemble(**cfg))
        result = solver.solve(sparse.assemble(**cfg))
        for name in ('generalized_velocity_m_s', 'total_velocity_m_s', 'contact_resultant_n', 'plate_omega_rad_s'):
            self.assertLess(relative(expected[name], result[name]), 2e-9, name)
        self.check_acceptance(result)
    def test_absolute_rigid_sphere_law_beyond_256_unknowns(self):
        cfg = spherical_fixture(3); system = sparse.assemble(**cfg)
        p = system['points']; a = system['nodal_area_m2']; beta = cfg['basal_drag_pa_s_per_m']; r = cfg['radius_m']
        matrix = beta*r*r*(np.eye(3)*a.sum()-np.einsum('n,ni,nj->ij', a,p,p))
        matrix += cfg['other_plate_rotational_drag_n_m_s']
        expected = np.linalg.solve(matrix, cfg['external_torques_n_m'][0])
        result = solver.solve(system)
        self.assertGreater(len(system['load_n']), 256)
        self.assertLess(relative(result['plate_omega_rad_s'][0], expected), 2e-10)
        self.check_acceptance(result)
    def test_duplicate_contact_quadrature_preserves_aggregate_reaction(self):
        cfg = fixture(); a = solver.solve(sparse.assemble(**cfg))
        split = []
        for sample in cfg['contacts']:
            for fraction in (.3, .7):
                split.append(dense.ContactSample(sample.first_face, sample.second_face,
                    sample.point, sample.normal, sample.length_m*fraction))
        cfg['contacts'] = split; b = solver.solve(sparse.assemble(**cfg))
        self.assertLess(relative(a['generalized_velocity_m_s'], b['generalized_velocity_m_s']), 2e-9)
        pair = b['contact_resultant_n'].reshape(-1, 2)
        self.assertLess(relative(pair.sum(axis=1), a['contact_resultant_n']), 2e-9)
        self.assertLess(relative(pair[:,0]/.3, pair[:,1]/.7), 2e-9)
        self.assertEqual(b['diagnostics']['redundant_rows'], 2)
    def test_common_moving_mantle_zero_work_limit(self):
        cfg = spherical_fixture(2); cfg['viscosity_pa_s']=0.
        cfg['other_plate_rotational_drag_n_m_s'][:]=0.; cfg['external_torques_n_m'][:]=0.
        speed = np.array([.2,-.3,.5])*1e-9
        cfg['basal_reference_velocity_m_s']=np.cross(speed,cfg['points'])
        result=solver.solve(sparse.assemble(**cfg))
        self.assertLess(relative(result['plate_omega_rad_s'][0], speed/cfg['radius_m']), 1e-10)
        self.check_acceptance(result)
    def test_resource_limits_fail_closed(self):
        system=sparse.assemble(**fixture())
        with self.assertRaisesRegex(ValueError,'row budget'):
            solver.solve(system,maximum_constraint_rows=2)
        with self.assertRaisesRegex(RuntimeError,'iteration budget'):
            solver.solve(system,maximum_iterations=1)
    def test_nonlinear_total_potential_stationarity_and_work(self):
        system=sparse.assemble(**fixture())
        result=solver.solve(system,resistance=Quartic(len(system['load_n'])))
        self.check_acceptance(result)
        self.assertGreater(result['diagnostics']['resistance_dissipation_w'],0.)
        self.assertGreater(result['diagnostics']['nonlinear_iterations'],1)
    def test_nonrigid_manufactured_solution_with_thousands_of_faces(self):
        system,target=manufactured_system(4)
        result=solver.solve(system)
        self.assertGreater(len(system['load_n']),5000)
        self.assertLess(relative(result['generalized_velocity_m_s'],target),2e-8)
        self.assertGreater(result['diagnostics']['cg_iterations'],5)
        self.check_acceptance(result)
    def test_independent_physical_power_gate_rejects_corrupted_ledger(self):
        system=sparse.assemble(**fixture())
        def corrupt(y):
            powers=sparse.work(system,y)
            powers['basal_dissipation_w']*=1.01
            return powers
        system['physical_work']=corrupt
        with self.assertRaisesRegex(RuntimeError,'physical power'):
            solver.solve(system)
    def test_rotated_solution_preserves_cartesian_motion_and_reaction(self):
        cfg=fixture(); original=solver.solve(sparse.assemble(**cfg))
        axis=unit([.3,-.6,.8]); angle=.71; cross=dense._cross_matrix(axis)
        q=np.eye(3)+np.sin(angle)*cross+(1-np.cos(angle))*cross@cross
        cfg['points']=cfg['points']@q.T; cfg['external_torques_n_m']=cfg['external_torques_n_m']@q.T
        cfg['contacts']=[dense.ContactSample(s.first_face,s.second_face,q@s.point,q@s.normal,s.length_m) for s in cfg['contacts']]
        result=solver.solve(sparse.assemble(**cfg))
        self.assertLess(relative(result['total_velocity_m_s'],original['total_velocity_m_s']@q.T),1e-8)
        self.assertLess(relative(result['contact_plate_torque_n_m'],original['contact_plate_torque_n_m']@q.T),1e-8)
    def test_many_original_normal_rows_at_material_mesh_scale(self):
        system,target=manufactured_system(2,paired=True)
        result=solver.solve(system)
        self.assertGreater(len(system['load_n']),600)
        self.assertEqual(len(system['contact_weights']),32)
        self.assertLess(relative(result['generalized_velocity_m_s'],target),2e-8)
        self.check_acceptance(result)
    def test_unresolved_near_dependent_contact_rows_cannot_be_silently_dropped(self):
        cfg=fixture(); sample=cfg['contacts'][0]
        normal=sample.normal+1e-7*np.cross(sample.point,sample.normal)
        cfg['contacts'].append(dense.ContactSample(sample.first_face,sample.second_face,sample.point,normal,sample.length_m))
        cfg['external_nodal_forces_n']=np.random.default_rng(7).normal(size=(6,3))*1e18
        with self.assertRaises(RuntimeError):
            solver.solve(sparse.assemble(**cfg),maximum_newton_iterations=4)
    def test_many_actual_finite_laws_solve_above_dense_unknown_limit(self):
        # Repeated material copies are a solver/load fixture, not a basal-domain
        # allocation or contact-admission model for coincident geology.
        from tests.test_local_joint_resistance import sparse_fixture,arc_parameters,patch_parameters
        from benchmarks.collision_architecture import local_joint_resistance as local
        copies=24; system=sparse_fixture(copies=copies); geometry=local.Geometry(system)
        welds=[geometry.weld_arc(**arc_parameters(first_face=2*i,second_face=2*i+1)) for i in range(copies)]
        interfaces=[geometry.interface_patch(**patch_parameters(first_face=2*i,second_face=2*i+1)) for i in range(copies)]
        resistance=local.Resistance(system,welds=welds,interfaces=interfaces,smoothing_speed_m_s=2e-9)
        result=solver.solve(system,resistance=resistance)
        self.assertGreater(len(system['load_n']),256)
        self.assertEqual(len(result['diagnostics']['resistance_diagnostics']['terms']),48)
        self.assertGreater(result['diagnostics']['resistance_dissipation_w'],0.)
        self.check_acceptance(result)
    def test_interface_supplies_resistance_to_zero_base_upper_plate_diagonal(self):
        from tests.test_local_joint_resistance import sparse_fixture,patch_parameters
        from benchmarks.collision_architecture import local_joint_resistance as local
        system=sparse_fixture(); size=len(system['load_n']); cut=system['nplate']
        # A manufactured covered upper sheet has no mantle drag. Its Euler
        # rotation has mathematically zero membrane strain; evaluate that exact
        # null direction here, rather than retaining tiny floating rigid strain.
        weight=system['basal_drag_weight_n_s_m'].copy()
        weight[system['vertex_plate']==1]=0.
        velocity=system['velocity_map'].to_dense()
        basal=np.sqrt(np.repeat(weight,3))[:,None]*velocity
        strain=system['strain_design'].to_dense(); strain[:,3:6]=0.
        other=system['other_plate_drag_factor'].copy(); other[:,3:6]=0.
        matrix=basal.T@basal+strain.T@strain
        matrix[:cut,:cut]+=other.T@other
        system['hessian_n_s_m']=matrix
        system['hessian_diagonal_n_s_m']=np.diag(matrix)
        system['hessian_absolute_action']=lambda y:abs(matrix)@abs(y)
        external=system['load_n'].copy()
        def powers(y):
            b=float((basal@y)@(basal@y)); s=float((strain@y)@(strain@y))
            o=float((other@y[:cut])@(other@y[:cut])); work=float(external@y)
            return dict(basal_dissipation_w=b,viscous_dissipation_w=s,other_drag_dissipation_w=o,
                external_power_w=work,basal_reference_input_power_w=0.,objective_w=.5*(b+s+o)-work)
        system['physical_work']=powers
        system['physical_work_absolute_scale_w']=lambda y:float(abs(y)@abs(matrix)@abs(y)+abs(external)@abs(y))
        np.testing.assert_array_equal(np.diag(matrix)[3:6],0.)
        with self.assertRaisesRegex(ValueError,'combined Hessian diagonal'):solver.solve(system)
        geometry=local.Geometry(system)
        law=local.Resistance(system,interfaces=[geometry.interface_patch(**patch_parameters())],smoothing_speed_m_s=2e-9)
        result=solver.solve(system,resistance=law)
        self.check_acceptance(result)
        self.assertGreater(np.linalg.norm(result['plate_omega_rad_s'][1]),0.)
        self.assertGreater(result['diagnostics']['resistance_dissipation_w'],0.)


def manufactured_system(level,paired=False):
    """Nonrigid known solution; source generated by the stated operator for solver verification."""
    cfg=spherical_fixture(level)
    if paired:
        p=cfg['points']; f=cfg['faces']; count=len(p)
        cfg['points']=np.r_[p,p]; cfg['faces']=np.r_[f,f+count]
        cfg['vertex_plate']=np.r_[np.zeros(count,int),np.ones(count,int)]
        cfg['basal_reference_velocity_m_s']=np.zeros((2*count,3))
        cfg['external_nodal_forces_n']=np.zeros((2*count,3))
        cfg['external_torques_n_m']=np.r_[cfg['external_torques_n_m'],-cfg['external_torques_n_m']]
        cfg['other_plate_rotational_drag_n_m_s']=np.eye(6)*1e39
        cfg['contacts']=[]
        for face_id in np.linspace(0,len(f)-1,32,dtype=int):
            point=unit(p[f[face_id]].sum(axis=0)); axis=np.eye(3)[np.argmin(abs(point))]
            cfg['contacts'].append(dense.ContactSample(face_id,face_id+len(f),point,unit(np.cross(point,axis)),100.))
    system=sparse.assemble(**cfg); cut=system['nplate']
    p=system['points']; tangent=system['tangent_basis']; rng=np.random.default_rng(143+level)
    velocity=(np.cross([.2,-.4,.7],p)+.3*rng.normal(size=p.shape))*1e-9
    target=np.r_[np.tile(np.array([.2,-.3,.4])*1e-9,system['plate_count']),np.einsum('nia,ni->na',tangent,velocity).ravel()]
    projection=solver._Constraints(system,solver._diagonal(system),512)
    target=projection.scale*projection.project(target/projection.scale)
    load=system['hessian_n_s_m']@target
    force=np.einsum('nia,na->ni',tangent,load[cut:].reshape(-1,2))
    nodal=system['velocity_map'].rmatvec(force.ravel())
    cfg['external_nodal_forces_n']=force
    cfg['external_torques_n_m']=(cfg['radius_m']*(load[:cut]-nodal[:cut])).reshape(-1,3)
    return sparse.assemble(**cfg),target


def benchmark(output_path,levels=(3,4,5,6)):
    """Explicit nonrigid solve resource probe, excluded from ordinary unit tests."""
    records=[]
    for level,paired in [(level,False) for level in levels]+[(4,True)]:
        gc.collect(); system,target=manufactured_system(level,paired=paired)
        tracemalloc.start(); start=time.perf_counter()
        result=solver.solve(system)
        elapsed=time.perf_counter()-start
        current,peak=tracemalloc.get_traced_memory(); tracemalloc.stop()
        d=result['diagnostics']; records.append(dict(level=level,faces=len(system['faces']),paired_sheets=paired,
            contact_rows=len(system['contact_weights']),
            unknowns=len(system['load_n']),solve_seconds=elapsed,tracked_peak_bytes=peak,
            cg_iterations=d['cg_iterations'],newton_iterations=d['nonlinear_iterations'],
            maximum_component_residual=float(np.max(d['componentwise_stationarity_relative'])),
            maximum_original_row_error=float(np.max(d['original_row_relative_error'])),
            joint_relative_residual=d['joint_relative_residual'],target_relative_error=relative(result['generalized_velocity_m_s'],target),
            physical_power_residual_w=d['physical_power_residual_w'],physical_power_acceptance_allowance_w=d['physical_power_acceptance_allowance_w']))
    source=Path(solver.__file__)
    receipt=dict(source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),numpy_version=np.__version__,records=records,
        scope='Known nonrigid manufactured sources: single spherical plate and paired-sheet case with32normal rows. Paired coincident sheets test solver algebra, not a physical basal allocation/admission model. No native physics/trajectory validation.',
        memory_scope='tracemalloc allocations inside solve; excludes assembly and input system arrays, not OS peak RSS.')
    Path(output_path).write_text(json.dumps(receipt,indent=2)+'\n',encoding='utf-8')
    return receipt


if __name__ == '__main__':
    import sys
    if len(sys.argv)==3 and sys.argv[1]=='--benchmark-output': print(json.dumps(benchmark(sys.argv[2]),indent=2))
    else: unittest.main()
