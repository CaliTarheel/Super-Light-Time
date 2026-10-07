"""Independent high-precision checks for newly born metre-scale crust."""
from decimal import Decimal, localcontext
import math
import unittest
import numpy as np

import gravitational_relaxation as gravity
import material_surface as surface
import mesh_coverage as coverage


def unit(value):
    value=np.asarray(value,float)
    return value/np.linalg.norm(value,axis=-1,keepdims=True)


def triangle(scale=1e-6,center=(1.,2.,3.)):
    n=unit(center);x=unit(np.cross(n,(0.,0.,1.)));y=np.cross(n,x)
    return unit(n+scale*(np.array([-1.,1.,0.])[:,None]*x
                         +np.array([-1.,-1.,2.])[:,None]*y))


def decimal_area(points):
    # Evaluate the original scalar determinant with 60 decimal digits on the
    # exact stored binary floats. This is independent of the translated kernel.
    with localcontext() as ctx:
        ctx.prec=60
        a,b,c=[[Decimal.from_float(float(v)) for v in p] for p in points]
        det=(a[0]*(b[1]*c[2]-b[2]*c[1])-a[1]*(b[0]*c[2]-b[2]*c[0])
             +a[2]*(b[0]*c[1]-b[1]*c[0]))
        den=1+sum(a[j]*b[j]+a[j]*c[j]+b[j]*c[j] for j in range(3))
        return 2*math.atan2(float(abs(det)),float(den))*6371.**2


def decimal_overlap(subject,clipper):
    """Independent 60-digit clipping, retaining distinct vertices of thin slivers."""
    def dot(a,b):return sum(x*y for x,y in zip(a,b))
    def cross(a,b):return [a[1]*b[2]-a[2]*b[1],a[2]*b[0]-a[0]*b[2],a[0]*b[1]-a[1]*b[0]]
    def normalized(a):
        length=dot(a,a).sqrt()
        return [x/length for x in a]
    with localcontext() as ctx:
        ctx.prec=60
        polygon,other=[[[Decimal.from_float(float(x)) for x in p] for p in triangle]
                       for triangle in (subject,clipper)]
        for j in range(3):
            a,b=other[j],other[(j+1)%3]
            normal=cross(a,b)
            distances=[Decimal(0) if p==a or p==b else dot(p,normal) for p in polygon]
            inside=[d >= 0 for d in distances]
            output=[]
            for k in range(len(polygon)):
                old=(k-1)%len(polygon)
                if inside[k] != inside[old]:
                    fraction=distances[old]/(distances[old]-distances[k])
                    fraction=min(Decimal(1),max(Decimal(0),fraction))
                    output.append(normalized([a*(1-fraction)+b*fraction
                                              for a,b in zip(polygon[old],polygon[k])]))
                if inside[k]:output.append(polygon[k])
            polygon=output
        result=0.
        for j in range(1,len(polygon)-1):
            a,b,c=polygon[0],polygon[j],polygon[j+1]
            result+=2*math.atan2(float(abs(dot(a,cross(b,c)))),
                                float(1+dot(a,b)+dot(a,c)+dot(b,c)))*6371.**2
        return result


class SmallSphericalFaceTests(unittest.TestCase):
    def test_distinct_near_coincident_vertices_preserve_thin_overlap_in_both_orders(self):
        # A real 0.02 Myr seed-37 preflight pair: the old 2e-13 duplicate
        # threshold discarded a distinct vertex and changed the reverse area
        # by 1.57e-7 km2, failing the unchanged 1e-8 gravity area check.
        first=np.array([[.939149237414537,.1766420315278987,-.2946121222241043],
                        [.9227102965896569,.19793195481431183,-.33080001485914856],
                        [.9162447251812708,.26408375489336455,-.3012563260396686]])
        second=np.array([[.9243054247446538,.13165661992940317,-.3582262081651341],
                         [.9227103034830769,.19793195452354925,-.3307999958051025],
                         [.9391492402197199,.1766420276385581,-.29461211561383843]])
        expected=decimal_overlap(first,second)
        self.assertAlmostEqual(expected,decimal_overlap(second,first),places=16)
        for a,b in ((first,second),(second,first)):
            scalar=coverage._polygon_area(coverage.clip_triangle(a,b),6371.)
            batched=coverage._intersection_areas(a[None],b[None],6371.)[0]
            measured,_,_=gravity._paired_area_derivative(a[None],b[None],6371.)
            np.testing.assert_allclose([scalar,batched,measured[0]],expected,rtol=0.,atol=5e-10)
        # The stored pair sits on a topology transition. Move it a finite
        # fraction of a metre into the same thin-overlap regime before checking
        # the fixed-topology derivative against independent virtual work.
        axis=np.array([.2,.3,-.1])
        second=unit(second-2e-8*np.cross(axis,second))
        direction=np.cross(axis,second)
        _,derivative,_=gravity._paired_area_derivative(first[None],second[None],6371.)
        h=1e-10
        finite=(decimal_overlap(first,unit(second+h*direction))-
                decimal_overlap(first,unit(second-h*direction)))/(2*h)
        predicted=derivative[0,9:]@direction.ravel()
        np.testing.assert_allclose(predicted,finite,rtol=2e-5)

    def test_coincident_edges_and_exact_duplicates_fit_the_polygon_buffer(self):
        points=triangle(3e-6)
        expected=decimal_area(points)
        area,_,_=gravity._paired_area_derivative(points[None],points[None],6371.)
        np.testing.assert_allclose(area,[expected],rtol=3e-13)
        np.testing.assert_allclose(coverage._intersection_areas(points[None],points[None],6371.),
                                   [expected],rtol=3e-13)
        degenerate=points[[0,0,1]]
        self.assertEqual(coverage._polygon_area(coverage.clip_triangle(degenerate,points),6371.),0.)
        np.testing.assert_array_equal(coverage._intersection_areas(degenerate[None],points[None],6371.),[0.])

    def test_metre_scale_areas_match_high_precision_at_arbitrary_orientation(self):
        for scale in (1e-5,1e-6,3e-7):
            for center in ((1.,2.,3.),(-2.,3.,1.),(3.,1.,-2.)):
                p=triangle(scale,center);expected=decimal_area(p)
                measured=surface.spherical_face_areas(p,np.array([[0,1,2]]))[0]
                self.assertAlmostEqual(measured/expected,1.,places=14)
                self.assertAlmostEqual(coverage._polygon_area(p,6371.)/expected,1.,places=14)

    def test_contained_overlap_and_its_derivative_use_the_same_resolved_area(self):
        outer=triangle(3e-6);inner=triangle(1e-6)
        expected=decimal_area(inner)
        scalar=coverage._polygon_area(coverage.clip_triangle(inner,outer),6371.)
        batched=coverage._intersection_areas(inner[None],outer[None],6371.)[0]
        area,derivative,_=gravity._paired_area_derivative(inner[None],outer[None],6371.)
        self.assertAlmostEqual(scalar/expected,1.,places=13)
        self.assertAlmostEqual(batched/expected,1.,places=13)
        self.assertAlmostEqual(area[0]/expected,1.,places=13)
        direction=np.array([[.2,.3,-.1],[-.1,.2,.1],[.1,-.2,.3]])
        direction-=inner*np.sum(direction*inner,axis=1)[:,None]
        h=1e-10
        measured=(decimal_area(unit(inner+h*direction))-decimal_area(unit(inner-h*direction)))/(2*h)
        predicted=derivative[0,:9]@direction.ravel()
        np.testing.assert_allclose(predicted,measured,rtol=2e-5)

    def test_tiny_column_gravity_consumes_interval_with_decreasing_energy(self):
        p=triangle();faces=np.array([[0,1,2]])
        area=np.array([decimal_area(p)]);volumes=area*60.;reference=np.array([30.])
        result,report=gravity.relax(p,faces,volumes,reference,np.array([1]),5e-6,
            rigid_mask=np.zeros(3,bool),minimum_area_km2=area*.2,maximum_area_km2=area*5.,
            constraint_version=1)
        self.assertEqual(report['completed_dt_myr'],5e-6)
        self.assertLess(report['energy_after_km4'],report['energy_before_km4'])
        final=decimal_area(result)
        expected=.5*final*(volumes[0]/final-reference[0])**2
        np.testing.assert_allclose(report['energy_after_km4'],expected,rtol=3e-13)


if __name__=='__main__':unittest.main()
