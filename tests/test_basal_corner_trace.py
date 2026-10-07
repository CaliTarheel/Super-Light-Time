"""Exact-corner trace regression with an independent stored140digit full Gram."""
from decimal import Decimal as D,localcontext
from fractions import Fraction as F
import json
from pathlib import Path
import unittest
import numpy as np

from benchmarks.collision_architecture import finite_basal_operator as basal
from benchmarks.collision_architecture import stable_basal_metric as stable


def fixture():
    return json.loads((Path(__file__).parent/'fixtures/basal_corner_trace_72871.json').read_text())


def setup():
    data=fixture();chart=tuple(tuple(F(v) for v in p) for p in data['chart_polygon'])
    geometry=stable.build(np.array(data['control_triangle']),chart)
    system=dict(points=np.array(data['material_triangle']),
        tangent_basis=np.array(data['oracle']['tangent_basis']),radius_m=1.)
    return data,system,geometry


def oracle_error(gram,data):
    with localcontext() as context:
        context.prec=100
        expected=[[D(v) for v in row] for row in data['oracle']['gram_unit']]
        scale=[expected[i][i].sqrt() for i in range(9)]
        delta=np.array([[float((D.from_float(float(gram[i,j]))-expected[i][j])/(scale[i]*scale[j]))
                         for j in range(9)] for i in range(9)])
    return float(np.linalg.norm(delta,2))


class BasalCornerTraceTests(unittest.TestCase):
    def test_saved72871_full_nine_coordinate_Gram_matches_independent_oracle_at_all_orders(self):
        data,system,g=setup()
        for order in (8,16,32,64,128):
            with self.subTest(order=order):
                factor,area,metric=basal._local_trace_rule(system,np.arange(3),g,order,1.,1.)
                self.assertLess(oracle_error(factor.T@factor,data),1e-11)
                self.assertLess(abs(area-g.area_unit)/g.area_unit,2e-10)
                self.assertLess(basal._metric_error(metric,g.metric_local),2e-10)

    def test_previously_failed_saved_piece_now_accepts_unchanged_gate_at16(self):
        data,system,g=setup()
        factor,records=basal._material_factor(system,np.arange(3),g,1.,1.,g.area_unit,g.metric_local,2e-10,(8,16,32,64))
        self.assertEqual(records[-1]['order'],16)
        self.assertLess(max(v for k,v in records[-1].items() if k!='order'),2e-10)
        self.assertLess(oracle_error(factor.T@factor,data),1e-11)
        system['radius_m']=data['radius_m'];scale=data['radius_m']**2*data['density']
        physical,records=basal._material_factor(system,np.arange(3),g,data['density'],data['beta'],
            data['area_m2'],g.metric_local*scale,data['tolerance'],(8,16,32,64))
        self.assertEqual(records[-1]['order'],16)
        self.assertLess(oracle_error((physical.T@physical)/(scale*data['beta']),data),1e-11)

    def test_exact_material_edge_incidence_remains_zero(self):
        data=fixture();points=np.array(data['material_triangle'])
        g=stable.build(points,((F(0),F(0)),(F(1),F(0)),(F(0),F(1))))
        alpha=basal._corner_trace_coordinates(points,g)
        np.testing.assert_array_equal(alpha-np.diag(np.diag(alpha)),np.zeros((3,3)))
        with localcontext() as context:
            context.prec=100
            expected=[1/sum(D.from_float(float(v))**2 for v in row).sqrt() for row in points]
            for a,b in zip(alpha.diagonal(),expected):self.assertLess(abs(D.from_float(a)-b),D('2e-16'))
        self.assertFalse(alpha.flags.writeable)

    def test_homogeneous_coefficients_are_not_normalized_barycentrics(self):
        points=np.eye(3);chart=((F(1,4),F(1,4)),(F(1,2),F(1,4)),(F(1,4),F(1,2)))
        g=stable.build(points,chart);alpha=basal._corner_trace_coordinates(points,g)
        self.assertTrue(np.all(alpha.sum(axis=1)>1.))
        rays=np.array([[float(1-x-y),float(x),float(y)] for x,y in chart])
        np.testing.assert_allclose(alpha,rays/np.linalg.norm(rays,axis=1)[:,None],rtol=2e-16,atol=0.)

    def test_nonzero_unrepresentable_corner_does_not_become_an_edge(self):
        tiny=F(1,10**350);points=np.eye(3)
        g=stable.build(points,((tiny,F(1,4)),(F(1,4),F(1,4)),(tiny,F(1,2))))
        with self.assertRaisesRegex(ValueError,'unresolved binary64 cast'):basal._corner_trace_coordinates(points,g)

    def test_outside_corner_and_changed_geometry_reject_without_sign_band(self):
        _,system,g=setup()
        with self.assertRaisesRegex(ValueError,'leaves its original material face'):
            basal._corner_trace_coordinates(system['points'],stable.build(np.eye(3),((F(0),F(0)),(F(1),F(0)),(F(0),F(1)))))
        g.local_polygon.flat[0]+=1e-5
        with self.assertRaisesRegex(ValueError,'modified'):basal._corner_trace_coordinates(system['points'],g)


if __name__=='__main__':unittest.main()
