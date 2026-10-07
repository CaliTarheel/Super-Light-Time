"""Accepted clipping water remains polygons, including collinear fan corners."""
from pathlib import Path
import math
import unittest
import numpy as np
import native_subduction as capture
import native_spreading as exact
import mesh_coverage

RADIUS=6371.

def area(polygon):return mesh_coverage._polygon_area(polygon,RADIUS)
def unit(points):
    points=np.asarray(points,float);return points/np.linalg.norm(points,axis=-1,keepdims=True)
def subtract(polygon,polygons):
    context=capture._polygon_context(polygons)
    return exact._subtract_polygons(polygon,context['polygons'],context)


class SubductionPolygonContextTests(unittest.TestCase):
    def test_exact_failed_trial_polygon_remains_a_complete_blocker(self):
        path=Path(__file__).parent/'fixtures/subduction_polygon_context_004.npz'
        with np.load(path) as z:
            polygons=[z['vertices'][a:b].copy() for a,b in zip(z['offsets'][:-1],z['offsets'][1:])]
        polygon=polygons[1]
        triangles=np.array([polygon[[0,i,i+1]] for i in range(1,len(polygon)-1)])
        determinants=np.einsum('ij,ij->i',triangles[:,0],np.cross(triangles[:,1],triangles[:,2]))
        self.assertGreater(area(polygon),.99)
        self.assertLess(np.abs(determinants).min(),1e-14)
        with self.assertRaisesRegex(ValueError,'nonzero area'):
            exact.prepare_material(dict(vertices=polygon,faces=np.array([[0,i,i+1] for i in range(1,len(polygon)-1)])))
        for shift in range(len(polygon)):
            blocker=np.roll(polygon,shift,axis=0)
            for ordered in (blocker,blocker[::-1]):
                remaining=subtract(polygon,[polygons[0],ordered])
                self.assertLessEqual(math.fsum(area(p) for p in remaining),exact.AREA_TOLERANCE_KM2)
        preserved=capture._polygon_context(polygons)['polygons']
        for before,after in zip(polygons,preserved):np.testing.assert_array_equal(before,after)

    def test_meaningful_small_polygon_below_locator_floor_is_not_discarded(self):
        tiny=unit([[1,0,0],[1,1e-6,0],[1,0,5e-9]])
        total=area(tiny)
        self.assertGreater(total,exact.AREA_TOLERANCE_KM2*10.)
        with self.assertRaisesRegex(ValueError,'nonzero area'):
            exact.prepare_material(dict(vertices=tiny,faces=np.array([[0,1,2]])))
        query=unit([[1,-1e-6,-1e-6],[1,2e-6,-1e-6],[1,2e-6,2e-6],[1,-1e-6,2e-6]])
        remaining=subtract(query,[tiny])
        removed=area(query)-math.fsum(area(p) for p in remaining)
        self.assertAlmostEqual(removed,total,delta=1e-11)
        self.assertGreater(removed,.99*total)

    def test_collinear_boundary_vertices_do_not_change_union_area(self):
        polygon=unit([[1,-.001,-.001],[1,0,-.001],[1,.001,-.001],[1,.001,.001],[1,-.001,.001]])
        clean=polygon[[0,2,3,4]]
        query=unit([[1,-.002,-.002],[1,.002,-.002],[1,.002,.002],[1,-.002,.002]])
        expected=area(query)-area(clean)
        for pieces in ([polygon],[polygon,polygon],[clean,polygon]):
            self.assertAlmostEqual(math.fsum(area(p) for p in subtract(query,pieces)),expected,delta=1e-7)


if __name__=='__main__':unittest.main()
