"""Exceptional products require an exact two-binary64 expansion."""
from fractions import Fraction as F
import json
import math
from pathlib import Path
import unittest
import numpy as np
from benchmarks.collision_architecture import framed_basal_map as framed
from tests.test_framed_basal_map import frame,reference_project
FIXTURE=json.loads((Path(__file__).parent/'fixtures/basal_exact_product_range.json').read_text())

def exact_pair(test,a,b):
    hi,lo=framed._two_product(a,b)
    aa,bb=np.broadcast_arrays(a,b)
    test.assertEqual(hi.shape,aa.shape);test.assertEqual(lo.shape,aa.shape)
    for x,y,h,l in zip(aa.flat,bb.flat,hi.flat,lo.flat):
        test.assertEqual(F(float(h))+F(float(l)),F(float(x))*F(float(y)))
    return hi,lo

class BasalExactProductTests(unittest.TestCase):
    def test_both_saved_native_exceptional_lanes_are_exactly_represented(self):
        for row in FIXTURE['cases']:
            hi,lo=exact_pair(self,np.asarray(row['a']),np.asarray(row['b']))
            self.assertEqual(float(hi),row['nearest_high'])
            self.assertEqual(float(lo),row['nearest_low'])

    def test_signed_broadcast_and_ordinary_fast_lanes_keep_exact_products(self):
        aa=np.array([[FIXTURE['cases'][0]['a']],[-FIXTURE['cases'][1]['a']]])
        bb=np.array([[FIXTURE['cases'][0]['b'],-FIXTURE['cases'][0]['b'],0.]])
        exact_pair(self,aa,bb)
        rng=np.random.default_rng(420)
        exact_pair(self,rng.normal(size=(8,1)),rng.normal(size=(1,7)))

    def test_extreme_opposite_scales_and_exact_subnormal_are_supported(self):
        for a,b in ((math.ldexp(1.,1000),math.ldexp(1.,-1000)),
                    (-math.ldexp(1.,1000),math.ldexp(1.,-1000)),
                    (np.finfo(float).max,1.),(np.nextafter(0.,1.),1.),
                    (0.,np.finfo(float).max),(np.nextafter(0.,1.),0.)):
            with self.subTest(a=a,b=b):exact_pair(self,np.asarray(a),np.asarray(b))

    def test_unrepresentable_residual_underflow_and_overflow_still_reject(self):
        for a,b in ((1e-80,-4.748946659612752e-216),(np.nextafter(0.,1.),.5),
                    (np.finfo(float).max,2.),(1e-200,1e-200)):
            with self.subTest(a=a,b=b),self.assertRaisesRegex(ValueError,'exact two-binary64'):
                framed._two_product(np.asarray(a),np.asarray(b))

    def test_nonfinite_operands_reject_even_with_zero_multiplier(self):
        for value in (np.inf,-np.inf,np.nan):
            with self.subTest(value=value),self.assertRaisesRegex(ValueError,'finite'):
                framed._two_product(np.asarray(value),np.asarray(0.))

    def test_large_finite_frame_projection_agrees_with_independent_Decimal(self):
        hi,lo=frame();value=np.ones(3)*1e300
        actual=framed.project_frames(hi,lo,value)
        expected=np.asarray(reference_project(hi,lo,value),float)
        np.testing.assert_allclose(actual,expected,rtol=4e-15,atol=0.)
        # This tests projection only; squared work at this scale overflows and
        # is still outside the physical operator's finite-output contract.

if __name__=='__main__':unittest.main()
