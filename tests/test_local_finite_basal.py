"""Actual local finite factors on tiny exact geometry, without a world claim."""
from fractions import Fraction as F
import json
from pathlib import Path
from copy import deepcopy
import unittest
import numpy as np

from benchmarks.collision_architecture import stable_basal_metric as stable
from benchmarks.collision_architecture import finite_basal_operator as finite
from benchmarks.collision_architecture.framed_basal_map import FramedBlockMap
from benchmarks.collision_architecture import sparse_shared_contact as sparse
from tests.test_stable_basal_metric import CONTROL, CHART, positive_work

R = 6371000.
BETA = 2.5e-3


def system(points):
    return sparse.assemble(points,np.array([[0,1,2]]),np.zeros(3,dtype=int),
        radius_m=R,basal_drag_pa_s_per_m=1.,viscosity_pa_s=0.,sheet_thickness_m=4e4,
        basal_reference_velocity_m_s=np.zeros((3,3)),other_plate_rotational_drag_n_m_s=np.zeros((3,3)),
        external_torques_n_m=np.zeros((1,3)),external_nodal_forces_n=np.zeros((3,3)),
        contacts=[],preserve_represented_points=True)


def component(s, geometry, mantle=None, *, beta=BETA, density=1.):
    """One declared partial piece; intentionally not a complete-cell build."""
    factor,records=finite._material_factor(s,np.arange(3),geometry,density,beta,
        geometry.area_unit*R**2*density,geometry.metric_local*R**2*density,2e-10,(8,16,32,64))
    design=FramedBlockMap(factor[None],np.arange(9)[None],np.arange(9)[None],(9,9),
        geometry.frame_hi[None],geometry.frame_lo[None])
    reference=np.zeros(9)
    if mantle is not None: reference[:3]=mantle
    return finite.FiniteBasal(s,design,design@reference,records)


class LocalFiniteBasalTests(unittest.TestCase):
    def test_public_allocation_rejects_numeric_identity_and_type_aliases(self):
        from tests.test_finite_basal_operator import system as standard_system, allocation, build
        s=standard_system(); valid=allocation()
        cases=[]
        changed=deepcopy(valid); changed['radius_m']=complex(changed['radius_m']); cases.append(changed)
        changed=deepcopy(valid); changed['native_cell_metric_m2']=changed['native_cell_metric_m2'].astype(complex); cases.append(changed)
        changed=deepcopy(valid); changed['pieces'][0]['bottom_face']=10.; cases.append(changed)
        changed=deepcopy(valid); changed['pieces'][0]['owner']=20.; cases.append(changed)
        changed=deepcopy(valid); changed['pieces'][0]['fractions']={20:True}; cases.append(changed)
        changed=deepcopy(valid); changed['support']={20.:1.,21:0.}; cases.append(changed)
        changed=deepcopy(valid); changed['support']={20:1.,21:False}; cases.append(changed)
        for i,changed in enumerate(cases):
            with self.subTest(case=i),self.assertRaises(ValueError): build(s,[changed])

    def test_both_saved_pieces_pass_with_original_material_and_coefficients(self):
        fixture=json.loads((Path(__file__).parent/'fixtures/native_basal_piece_regressions.json').read_text())
        for name,record in fixture['pieces'].items():
            with self.subTest(cell=name):
                control=np.array(record['control_triangle'])
                chart=tuple(tuple(F(v) for v in row) for row in record['chart_polygon'])
                g=stable.build(control,chart); s=system(np.array(record['material_triangle']))
                beta=record['basal_drag_pa_s_per_m']; density=record['native_measure_density']
                c=component(s,g,beta=beta,density=density)
                y=np.zeros(9); y[:3]=g.frame_hi[:,2]+np.sqrt(g.metric_local[2,2]/g.area_unit)*g.frame_hi[:,0]
                oracle=beta*R**2*density*positive_work(control,chart,y[:3])
                self.assertLess(abs(c.work(y)['basal_dissipation_w']-oracle)/oracle,2e-10)
                self.assertLess(abs(g.area_unit*R**2*density-record['area_m2'])/record['area_m2'],1e-14)

    def test_tiny_actual_operator_preserves_weak_and_oblique_power(self):
        g=stable.build(CONTROL,CHART); s=system(CONTROL); c=component(s,g)
        scale=np.sqrt(g.metric_local[2,2]/g.area_unit)
        for a in (g.frame_hi[:,2],g.frame_hi[:,2]+scale*g.frame_hi[:,0],
                  g.frame_hi[:,2]-scale*g.frame_hi[:,1],np.array([.3,-.7,.2])):
            y=np.zeros(9); y[:3]=a
            oracle=BETA*R**2*positive_work(CONTROL,CHART,a)
            self.assertLess(abs(c.work(y)['basal_dissipation_w']-oracle)/oracle,2e-10)
        self.assertLessEqual(max(max(row[k] for k in ('full_rigid_metric_relative_error',
            'compressed_rigid_metric_relative_error','operator_relative_change','area_relative_error'))
            for row in c.diagnostics[-1:]),2e-10)

    def test_generic_rotations_check_actual_represented_global_input(self):
        rng=np.random.default_rng(261)
        for _ in range(3):
            rotation,_=np.linalg.qr(rng.normal(size=(3,3)))
            if np.linalg.det(rotation)<0: rotation[:,0]*=-1
            points=CONTROL@rotation.T; g=stable.build(points,CHART); c=component(system(points),g)
            y=np.zeros(9); y[:3]=g.frame_hi[:,2]+np.sqrt(g.metric_local[2,2]/g.area_unit)*g.frame_hi[:,0]
            expected=BETA*R**2*positive_work(points,CHART,y[:3])
            self.assertLess(abs(c.work(y)['basal_dissipation_w']-expected)/expected,2e-10)

    def test_exact_split_preserves_residual_force_and_moving_reference_work(self):
        s=system(CONTROL); g=stable.build(CONTROL,CHART)
        a,b,c=CHART; m=tuple((x+y)/2 for x,y in zip(b,c))
        mantle=np.array([.001,-.002,.003])
        whole=component(s,g,mantle)
        parts=[component(s,stable.build(CONTROL,p),mantle) for p in ((a,b,m),(a,m,c))]
        y=np.random.default_rng(729).normal(size=9)*.001
        expected=whole.work(y)
        for name in ('basal_dissipation_w','basal_reference_input_power_w','generalized_gradient_work_w'):
            observed=sum(part.work(y)[name] for part in parts)
            self.assertLess(abs(observed-expected[name]),2e-10*expected['arithmetic_constituent_work_bound_w'])
        gradient=sum((part.evaluate(y)[1] for part in parts),np.zeros(9))
        scale=whole.absolute_action(y)+abs(whole.load_n)
        self.assertTrue(np.all(abs(gradient-whole.evaluate(y)[1])<=2e-10*scale))
        y[:3]=mantle; y[3:]=0.
        self.assertEqual(whole.work(y)['basal_dissipation_w'],0.)

    def test_returned_forces_have_constituent_bounded_reciprocal_work(self):
        g=stable.build(CONTROL,CHART); c=component(system(CONTROL),g)
        y=np.zeros(9); y[:3]=g.frame_hi[:,2]
        _,force=c.evaluate(y); work=c.work(y)
        # Final global force components are binary64. The small dot product
        # must be assessed against its constituents, not false relative digits.
        bound=work['arithmetic_constituent_work_bound_w']+float(abs(y)@abs(force))
        self.assertLess(abs(float(y@force)-work['generalized_gradient_work_w']),256*np.finfo(float).eps*bound)
        self.assertTrue(np.all(abs(c.hessian_n_s_m@y)<=c.absolute_action(y)*(1+1e-14)))

    def test_exact_piece_cannot_be_assigned_to_unrelated_material(self):
        g=stable.build(CONTROL,CHART)
        with self.assertRaisesRegex(ValueError,'declared bottom'):
            component(system(np.eye(3)),g)


if __name__=='__main__': unittest.main()
