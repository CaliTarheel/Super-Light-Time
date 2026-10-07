"""Exact geometry regression for batched rejection of disjoint union pieces."""
import math
import unittest
from unittest.mock import patch
import numpy as np
import mesh_coverage
import native_spreading as spreading
import convex_partition
from orientation import rotation_matrix
from tests._native_spreading_oracle import rectangle,material_rectangles,uncovered_rectangle_area,polygon_area


def slow_complement(polygon,context):
    """Disjoint scalar subtraction order, without the candidate acceleration."""
    parts=[polygon]
    for face in spreading._candidates(polygon,context):
        remaining=[]
        for part in parts:
            overlap,outside=convex_partition.partition(part,convex_partition.prepare(context['triangles'][face]))
            if spreading._area(overlap,context)<=spreading.AREA_TOLERANCE_KM2:
                remaining.append(part);continue
            remaining.extend(piece for piece in outside if spreading._area(piece,context)>spreading.AREA_TOLERANCE_KM2)
        parts=remaining
        if not parts:break
    return parts


class SpreadingAccelerationTests(unittest.TestCase):
    def compare(self,polygon,context):
        expected=slow_complement(polygon,context)
        with patch.object(mesh_coverage,'_clip_planes',wraps=mesh_coverage._clip_planes) as spy:
            actual=spreading.uncovered_polygons(polygon,context)
        self.assertEqual(len(actual),len(expected))
        for a,b in zip(actual,expected):np.testing.assert_array_equal(a,b)
        return actual,spy.call_count

    def test_overlapping_refined_sheets_preserve_ordered_polygons_and_independent_union(self):
        target=(-150.,150.,-100.,100.)
        rectangles=[(-170.,30.,-120.,50.),(-50.,170.,-40.,120.)]*2
        for refined in (False,True):
            for pose in ({'yaw':0.,'pitch':0.,'roll':0.},
                         {'yaw':179.,'pitch':0.,'roll':0.},
                         {'yaw':37.,'pitch':89.,'roll':23.}):
                rotation=rotation_matrix(pose)
                vertices,faces=material_rectangles(rectangles,refine=refined,rotation=rotation)
                context=spreading.prepare_material(dict(vertices=vertices,faces=faces))
                result,_=self.compare(rectangle(*target)@rotation,context)
                self.assertAlmostEqual(math.fsum(polygon_area(p) for p in result),
                    uncovered_rectangle_area(target,rectangles),delta=1e-6)

    def test_cached_intersection_planes_preserve_original_scalar_arithmetic(self):
        vertices,faces=material_rectangles([(-100.,100.,-100.,100.)],refine=True)
        vertices=vertices@rotation_matrix(dict(yaw=179.,pitch=89.,roll=37.))
        context=spreading.prepare_material(dict(vertices=vertices,faces=faces))
        for i,triangle in enumerate(vertices[faces]):
            expected=mesh_coverage._unit(np.cross(triangle,np.roll(triangle,-1,axis=0)))
            np.testing.assert_array_equal(context['intersection_planes'][i],expected)
        # A context captured before the optimization remains reconstructible.
        context.pop('intersection_planes')
        self.compare(vertices[faces[0]],context)
        self.assertIn('intersection_planes',context)

    def test_rejection_is_sufficient_and_tolerance_neighbors_take_scalar_path(self):
        plane=np.array([[0.,1.,0.],[0.,0.,1.],[1.,0.,0.]])
        polygon=rectangle(-100.,100.,-100.,100.)
        parts=[]
        for offset in (-1e-3,-2e-12,-5.1e-14,-4.9e-14,0.,5e-14,.01):
            part=polygon.copy();part[:,1]=offset
            part/=np.linalg.norm(part,axis=1)[:,None]
            parts.append(part)
        parts=parts*103 # Exercise more than two bounded groups.
        chosen=spreading._separated_parts(parts,plane)
        for i,(part,rejected) in enumerate(zip(parts,chosen)):
            if rejected:self.assertEqual(len(mesh_coverage._clip_planes(part,plane)),0)
            if i%7 in (2,3,4,5):self.assertFalse(rejected)
        self.assertTrue(np.any(chosen));self.assertTrue(np.any(~chosen))

    def test_real_water_sliver_and_shared_coast_are_not_removed(self):
        target=(-150.,150.,-100.,100.)
        for half_gap in (.0005,2e-6):
            rectangles=[(-170.,-half_gap,-120.,120.),(half_gap,170.,-120.,120.)]
            vertices,faces=material_rectangles(rectangles,refine=True)
            context=spreading.prepare_material(dict(vertices=vertices,faces=faces))
            result,_=self.compare(rectangle(*target),context)
            self.assertGreater(math.fsum(polygon_area(p) for p in result),0.)
            self.assertAlmostEqual(math.fsum(polygon_area(p) for p in result),
                uncovered_rectangle_area(target,rectangles),delta=1e-6)
            self.compare(vertices[faces[0]],context)


if __name__=='__main__':unittest.main()
