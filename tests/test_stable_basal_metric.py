"""Independent positive quadrature and exact-chart representation checks."""
from copy import deepcopy
from decimal import Decimal as D, localcontext
from fractions import Fraction as F
import json
from pathlib import Path
import unittest
import numpy as np
from benchmarks.collision_architecture import stable_basal_metric as stable
from benchmarks.collision_architecture import basal_partition

FIXTURE=json.loads((Path(__file__).parent/'fixtures/basal_tiny_exact_chart.json').read_text())
CONTROL=np.array([[float(F(x)) for x in row] for row in FIXTURE['basis']])
CHART=tuple(tuple(F(x) for x in row) for row in FIXTURE['chart_polygon'])


def positive_work(control,chart,vector,order=8):
    """Independent radial Duffy integral of |vector cross r|²; no boundary formula."""
    with localcontext() as ctx:
        ctx.prec=100
        dot=lambda a,b:sum((x*y for x,y in zip(a,b)),D(0))
        cross=lambda a,b:[a[1]*b[2]-a[2]*b[1],a[2]*b[0]-a[0]*b[2],a[0]*b[1]-a[1]*b[0]]
        unit=lambda a:[x/dot(a,a).sqrt() for x in a]
        basis=[[F(float(x)) for x in row] for row in control]
        decimal=lambda q:D(q.numerator)/D(q.denominator)
        points=[]
        for x,y in chart:
            w=(1-x-y,x,y)
            points.append(unit([decimal(sum((w[j]*basis[j][k] for j in range(3)),F(0))) for k in range(3)]))
        speed=[D.from_float(float(x)) for x in vector]
        nodes,weights=np.polynomial.legendre.leggauss(order)
        total=D(0); a=points[0]
        for b,c in zip(points[1:-1],points[2:]):
            determinant=dot(a,cross(b,c))
            for uu,wu in zip(nodes,weights):
                u=(D.from_float(float(uu))+1)/2
                for vv,wv in zip(nodes,weights):
                    v=(D.from_float(float(vv))+1)/2
                    q=[a[k]+u*(b[k]-a[k])+(1-u)*v*(c[k]-a[k]) for k in range(3)]
                    length=dot(q,q).sqrt(); q=[x/length for x in q]
                    velocity=cross(speed,q)
                    total+=D.from_float(float(wu))*D.from_float(float(wv))/4*(1-u)*determinant/length**3*dot(velocity,velocity)
        return float(total)


def work(metric,vector):
    a=metric.factor_local@metric.localize(vector)
    return float(a@a)


class StableBasalMetricTests(unittest.TestCase):
    def test_saved_positive_piece_matches_independent_weak_and_oblique_work(self):
        metric=stable.build(CONTROL,CHART)
        self.assertGreater(metric.metric_local[2,2],0.)
        self.assertTrue(np.all(np.diag(metric.factor_local)>0.))
        scale=np.sqrt(metric.metric_local[2,2]/metric.area_unit)
        n=metric.frame_hi[:,2]; t=metric.frame_hi[:,0]
        for vector in (n,n+scale*t,n-scale*t,np.array([.3,-.7,.2])):
            oracle=positive_work(CONTROL,CHART,vector)
            self.assertLess(abs(work(metric,vector)-oracle)/oracle,2e-10)
        # The exact chart is primary; prior globally rounded spherical corners
        # are retained in the fixture only to expose the former failure.
        self.assertLess(abs(metric.area_unit*6371000.**2-FIXTURE['physical_area_m2']),1e-9)

    def test_generic_rotations_use_oracle_for_each_represented_input(self):
        rng=np.random.default_rng(770)
        for _ in range(4):
            rotation,_=np.linalg.qr(rng.normal(size=(3,3)))
            if np.linalg.det(rotation)<0: rotation[:,0]*=-1
            control=CONTROL@rotation.T
            metric=stable.build(control,CHART)
            scale=np.sqrt(metric.metric_local[2,2]/metric.area_unit)
            vector=metric.frame_hi[:,2]+scale*metric.frame_hi[:,0]
            oracle=positive_work(control,CHART,vector)
            self.assertLess(abs(work(metric,vector)-oracle)/oracle,2e-10)

    def test_exact_chart_subdivision_adds_every_metric_direction(self):
        whole=stable.build(CONTROL,CHART)
        a,b,c=CHART; midpoint=tuple((x+y)/2 for x,y in zip(b,c))
        first=stable.build(CONTROL,(a,b,midpoint)); second=stable.build(CONTROL,(a,midpoint,c))
        difference=first.metric_in(whole)+second.metric_in(whole)-whole.metric_local
        self.assertLess(stable._whitened_error(difference,whole.factor_local),2e-12)
        self.assertLess(abs(first.area_unit+second.area_unit-whole.area_unit)/whole.area_unit,2e-14)

    def test_actual_global_input_transform_and_transpose_agree_with_Decimal(self):
        metric=stable.build(CONTROL,CHART)
        x=np.array([.17,-.37,.81]); local=np.array([-.31,.29,.77])
        with localcontext() as ctx:
            ctx.prec=110
            q=[[D(s) for s in row] for row in metric._frame_decimal]
            expected=np.array([float(sum(D.from_float(float(x[i]))*q[i][j] for i in range(3))) for j in range(3)])
            pull=np.array([float(sum(D.from_float(float(local[j]))*q[i][j] for j in range(3))) for i in range(3)])
        np.testing.assert_array_equal(metric.localize(x),expected)
        np.testing.assert_array_equal(metric.globalize(local),pull)
        self.assertLess(abs(metric.localize(x)@local-x@metric.globalize(local)),2e-16)
        np.testing.assert_allclose(metric.localize(3*x),3*metric.localize(x),rtol=3e-16,atol=3e-16)

    def test_content_and_original_typed_geometry_are_bound(self):
        metric=stable.build(CONTROL,CHART)
        for field in ('frame_hi','frame_lo','local_polygon','metric_local','factor_local','control_triangle'):
            changed=deepcopy(metric); array=getattr(changed,field)
            array.flat[0]=np.nextafter(array.flat[0],np.inf)
            with self.assertRaisesRegex(ValueError,'modified'): changed.validate()
        with self.assertRaises(ValueError): stable.build(CONTROL.astype(object),CHART)
        with self.assertRaises(ValueError): stable.build(CONTROL,((False,0),(1,0),(0,1)))
        with self.assertRaises(ValueError): stable.build(CONTROL,tuple(reversed(CHART)))
        with self.assertRaises(ValueError): metric.localize([True,False,True])
        with self.assertRaises(ValueError): metric.localize([1,2,3],normalize='yes')

    def test_insufficient_precision_schedule_rejects_without_projection(self):
        e=F(1,10**35); chart=((F(0),F(0)),(e,F(0)),(F(0),e))
        with self.assertRaisesRegex(ValueError,'precision convergence'):
            stable.build(np.eye(3),chart,precision_schedule=(32,40))
        metric=stable.build(np.eye(3),chart,precision_schedule=(48,80,128,192,256))
        self.assertGreater(metric.precision,80)
        self.assertGreater(metric.metric_local[2,2],0.)

    def test_partition_retains_tiny_metric_and_control_frame_provenance(self):
        # Construct material geometry from exact represented vertices on a small
        # but resolved patch. The saved actual rational witness is tested above.
        p=np.array([[1.,0.,0.],[1.,1e-8,0.],[1.,0.,1e-8]])
        p/=np.linalg.norm(p,axis=1)[:,None]
        result=basal_partition.partition_cell(np.eye(3),[dict(triangle=p,face_id=8,sheet_id=3,owner=0)],[],
            radius_m=6371000.,saved_cell_area_m2=np.pi/2*6371000.**2,support={0:1.},basal_drag_pa_s_per_m=1.)
        tiny=next(row for row in result['pieces'] if row['bottom_face']==8)
        self.assertLess(tiny['area_m2'],.003)
        self.assertGreater(tiny['stable_metric'].metric_local[2,2],0.)
        self.assertEqual(tiny['stable_metric'].chart_polygon,tiny['chart_polygon'])
        np.testing.assert_array_equal(tiny['stable_metric'].control_triangle,np.eye(3))
        self.assertLess(result['scaled_metric_closure_error'],2e-10)


if __name__=='__main__': unittest.main()
