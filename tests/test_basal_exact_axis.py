"""Exact provenance certificates remove only proved local-frame zeros."""
from decimal import Decimal as D
from fractions import Fraction as F
import json
from pathlib import Path
import unittest
import numpy as np
from benchmarks.collision_architecture import stable_basal_metric as stable
from benchmarks.collision_architecture import framed_basal_map as framed
from benchmarks.collision_architecture import basal_partition as partition
from benchmarks.collision_architecture import finite_basal_operator as finite
from tests.test_local_finite_basal import system,R
from tests.test_stable_basal_metric import positive_work

FIXTURE=json.loads((Path(__file__).parent/'fixtures/basal_exact_axis_31.json').read_text())
CONTROL=np.asarray(FIXTURE['control_triangle'])
CHART=tuple(tuple(F(x) for x in p) for p in FIXTURE['chart_polygon'])

def rays(control,chart=CHART):
    basis=tuple(tuple(F(float(x)) for x in p) for p in control)
    return tuple(tuple(sum((w*basis[j][k] for j,w in enumerate((1-x-y,x,y))),F(0)) for k in range(3)) for x,y in chart)

class BasalExactAxisTests(unittest.TestCase):
    def test_original_control31_has_exact_second_axis_without_decimal_residual(self):
        before=CONTROL.tobytes();metric=stable.build(CONTROL,CHART)
        self.assertEqual(stable._certified_centroid_zeros(rays(CONTROL)),(True,False,False))
        np.testing.assert_array_equal(metric.frame_hi[:,1],[-1.,0.,0.])
        np.testing.assert_array_equal(metric.frame_lo[:,1],np.zeros(3))
        self.assertEqual(tuple(D(row[1]) for row in metric._frame_decimal),(D(-1),D(0),D(0)))
        self.assertEqual(CONTROL.tobytes(),before)

    def test_saved_tiny_product_transforms_and_transpose_without_widening_range(self):
        metric=stable.build(CONTROL,CHART);value=-4.748946659612752e-216
        expected=np.array([0.,-value,0.])
        np.testing.assert_array_equal(framed.project_frames(metric.frame_hi,metric.frame_lo,[value,0.,0.]),expected)
        design=framed.FramedBlockMap(np.eye(9)[None],np.arange(9)[None],np.arange(9)[None],(9,9),metric.frame_hi[None],metric.frame_lo[None])
        local=np.zeros(9);local[1]=value
        np.testing.assert_array_equal(design.rmatvec(local),np.r_[-value,np.zeros(8)])
        # A genuine supplied nonzero product remains outside the declared
        # two-word range. The fix must never turn this into an accepted zero.
        with self.assertRaisesRegex(ValueError,'exponent'):
            framed._two_product(np.array([1e-80]),np.array([value]))
        stale=metric.frame_lo.copy();stale[0,1]=1e-80
        with self.assertRaisesRegex(ValueError,'exponent'):
            framed.project_frames(metric.frame_hi,stale,[value,0.,0.])

    def test_zero_raw_sum_with_unequal_norms_is_not_a_certificate(self):
        control=CONTROL.copy();control[2,2]=np.nextafter(control[2,2],np.inf)
        original=rays(control)
        self.assertEqual(sum((p[0] for p in original),F(0)),0)
        self.assertFalse(stable._certified_centroid_zeros(original)[0])
        metric=stable.build(control,CHART)
        self.assertNotEqual(D(metric._frame_decimal[0][2]),0)
        self.assertNotEqual(metric.frame_hi[0,2],0.)

    def test_near_axis_and_rounded_decimal_zero_are_not_exact_proofs(self):
        for epsilon in (1e-30,np.nextafter(0.,1.)):
            with self.subTest(epsilon=epsilon):
                control=CONTROL.copy();control[0,0]=epsilon
                self.assertFalse(stable._certified_centroid_zeros(rays(control))[0])
                # The tiny subnormal can disappear in ordinary Decimal sum
                # ordering. It still does not authorize the structural branch.
                if epsilon==1e-30:
                    metric=stable.build(control,CHART)
                    self.assertGreater(D(metric._frame_decimal[0][2]),0)
                    self.assertGreater(metric.frame_hi[0,2],0.)

    def test_certificate_uses_the_actual_fraction_chart_not_just_control_vertices(self):
        chart=((F(0),F(0)),(F(1,2),F(0)),(F(0),F(1)))
        self.assertTrue(stable._certified_centroid_zeros(rays(CONTROL))[0])
        self.assertFalse(stable._certified_centroid_zeros(rays(CONTROL,chart))[0])
        metric=stable.build(CONTROL,chart)
        self.assertGreater(metric.frame_hi[0,2],0.)

    def test_exact_coordinate_rotations_preserve_only_the_proved_axis(self):
        for shift in range(3):
            for signs in ([1,1,1],[-1,-1,1],[-1,1,-1],[1,-1,-1]):
                control=np.roll(CONTROL,shift,axis=1)*signs
                proof=stable._certified_centroid_zeros(rays(control))
                self.assertEqual(sum(proof),1)
                metric=stable.build(control,CHART);axis=proof.index(True)
                expected=np.zeros(3);expected[axis]=-1
                np.testing.assert_array_equal(metric.frame_hi[:,1],expected)
                np.testing.assert_array_equal(metric.frame_lo[:,1],np.zeros(3))

    def test_actual_uncovered_factor_preserves_rigid_work_and_transpose(self):
        beta=1e15
        allocation=partition.partition_cell(CONTROL,[],[],radius_m=R,
            saved_cell_area_m2=FIXTURE['saved_area_m2'],support={0:1.},
            basal_drag_pa_s_per_m=beta,allocation_policy='resolved-material-bottom')
        self.assertEqual(len(allocation['pieces']),1)
        self.assertIsNone(allocation['pieces'][0]['bottom_face'])
        s=system(CONTROL)
        component=finite.build(s,[allocation],{},owner_to_plate_slot={0:0},
            mantle_omega_rad_s=np.zeros(3),quadrature_relative_tolerance=2e-10,max_order=64)
        metric=allocation['control_stable_metric'];density=allocation['native_measure_density']
        for velocity in (np.array([.2,-.3,.1]),metric.frame_hi[:,2]):
            y=np.zeros(9);y[:3]=velocity
            truth=beta*R**2*density*positive_work(CONTROL,CHART,velocity)
            observed=component.work(y)['basal_dissipation_w']
            self.assertLess(abs(observed-truth)/truth,2e-10)
            force=component.hessian_n_s_m@y
            np.testing.assert_allclose(force,component.basal_design.rmatvec(component.basal_design@y),rtol=0.,atol=0.)

if __name__=='__main__':unittest.main()
