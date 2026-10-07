"""Disjoint complements, exact side oracles and the native capture regression."""
from decimal import Decimal,localcontext
import json
import math
from pathlib import Path
import unittest
import numpy as np
import convex_partition as geometry
import native_spreading as spreading
from tests._native_spreading_oracle import rectangle,uncovered_rectangle_area,polygon_area
from orientation import rotation_matrix


def unit(points):
    points=np.asarray(points,float);return points/np.linalg.norm(points,axis=-1,keepdims=True)


def determinant(a,b,c):
    # Decimal products provide an independent high-precision oracle for the
    # production Fraction fallback and the double-precision filter.
    with localcontext() as context:
        context.prec=100
        a,b,c=([Decimal.from_float(float(x)) for x in row] for row in (a,b,c))
        return float(a[0]*(b[1]*c[2]-b[2]*c[1])-a[1]*(b[0]*c[2]-b[2]*c[0])+a[2]*(b[0]*c[1]-b[1]*c[0]))


def precise_area(points):
    return abs(math.fsum(2.*math.atan2(determinant(points[0],points[i],points[i+1]),
        1.+float(points[0]@points[i])+float(points[i]@points[i+1])+float(points[i+1]@points[0]))
        for i in range(1,len(points)-1)))*6371.**2


class ConvexPartitionTests(unittest.TestCase):
    def test_near_plane_signs_match_decimal_and_share_exact_edge_corners(self):
        rng=np.random.default_rng(841)
        for width in (1e-4,1e-10,1e-14,1e-17):
            for _ in range(12):
                a=unit(rng.normal(size=3));b=unit(a+.2*rng.normal(size=3))
                normal=unit(np.cross(a,b));point=unit(.7*a+.3*b+width*normal)
                values=geometry.Edge(a,b).distances(np.array([a,b,point,unit(point-2.*width*normal)]))
                np.testing.assert_array_equal(values[:2],0.)
                for p,value in zip((point,unit(point-2.*width*normal)),values[2:]):
                    expected=determinant(a,b,p)
                    self.assertEqual(np.sign(value),np.sign(expected))
                    self.assertAlmostEqual(value,expected,delta=32.*np.finfo(float).eps*np.linalg.norm(b-a)*np.linalg.norm(p-a))
                midpoint=unit(a+b)
                self.assertEqual(geometry.Edge(a,b).distances(midpoint[None])[0],determinant(a,b,midpoint))

    def test_thin_strip_partition_conserves_area_and_does_not_copy_both_halves(self):
        for width in (1e-7,1e-14):
            subject=unit([[1.,-width,-.2],[1.,width,-.2],[1.,width,.2],[1.,-width,.2]])
            edge=geometry.Edge(unit([1.,0.,.3]),unit([1.,0.,-.3]))
            positive,negative=geometry.split(subject,edge)
            total=precise_area(subject)
            self.assertGreater(total,spreading.AREA_TOLERANCE_KM2)
            self.assertAlmostEqual(precise_area(positive)/total,.5,places=12)
            self.assertAlmostEqual((precise_area(positive)+precise_area(negative))/total,1.,places=12)
            boundary_positive=positive[positive[:,1]==0.]
            boundary_negative=negative[negative[:,1]==0.]
            self.assertEqual({tuple(p) for p in boundary_positive},{tuple(p) for p in boundary_negative})

    def test_saved_native_duplicate_blockers_never_increase_water_and_self_capture_is_empty(self):
        fixture=json.loads((Path(__file__).parent/'fixtures/capture_duplicate_halfspaces.json').read_text())
        polygon=np.array(fixture['subject']);blockers=list(map(np.array,fixture['blockers']))
        context=dict(radius_km=fixture['radius_km']);parts=[polygon]
        for blocker in blockers:
            before=math.fsum(precise_area(p) for p in parts)
            parts=[piece for p in parts for piece in spreading._subtract_polygons(p,[blocker],context)]
            self.assertLessEqual(len(parts),1)
            self.assertLessEqual(math.fsum(precise_area(p) for p in parts),before+1e-20)
        self.assertEqual(parts,[])
        for shift in range(3):
            for reverse in (False,True):
                rotated=np.roll(polygon,shift,axis=0)
                if reverse:rotated=rotated[::-1]
                self.assertEqual(spreading._subtract_polygons(polygon,[rotated],context),[])

    def test_union_matches_independent_chart_tiles_under_order_orientation_and_rotation(self):
        target=(-150.,150.,-100.,100.);rectangles=[(-170.,30.,-120.,50.),(-50.,170.,-40.,120.)]
        expected=uncovered_rectangle_area(target,rectangles)
        for pose in (dict(yaw=0.,pitch=0.,roll=0.),dict(yaw=37.,pitch=89.,roll=23.)):
            rotation=rotation_matrix(pose);subject=rectangle(*target)@rotation
            blockers=[rectangle(*box)@rotation for box in rectangles]
            for ordered in (blockers,blockers[::-1],[b[::-1] for b in blockers]*2):
                result=spreading._subtract_polygons(subject,ordered,dict(radius_km=6371.))
                self.assertAlmostEqual(math.fsum(polygon_area(p) for p in result),expected,delta=1e-6)

    def test_native_third_interval_rejects_unmodeled_detachment_after_checkpoint_replay(self):
        import checkpoint,tempfile
        from native_engine import Simulation
        from tests.test_entry_phase_depth import world
        from tests.test_entry_regions import advance
        s=world();s.retained_phase_parameters['reaction_pressure_pa']=1.2e9
        for _ in range(2):advance(s,.000005,max_source_step_myr=.000005)
        before_time=float(s.t)
        before_vertices=s.material_surface['vertices'].copy()
        before_structure={key:value.copy() for key,value in s.structure.items()}
        before_slab_mass=float(s.trench_systems[0]['slab_retained_excess_mass_kg'])
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/'before-third.npz';checkpoint.write_checkpoint(path,s,dict(config=s.config),{})
            replay,_=checkpoint.read_checkpoint(path,None,Simulation)
        # The third candidate interval exhausts its attached local neck. Entry
        # has no detachment/eduction handoff, so the coupled guard must reject
        # the whole step instead of manufacturing a post-rupture source path.
        for state in (s,replay):
            with self.assertRaisesRegex(ValueError,'lost its attached source slab'):
                advance(state,.000005,max_source_step_myr=.000005)
            self.assertEqual(float(state.t),before_time)
            self.assertEqual(float(state.trench_systems[0]['slab_retained_excess_mass_kg']),
                             before_slab_mass)
            np.testing.assert_array_equal(state.material_surface['vertices'],before_vertices)
            for key,value in before_structure.items():
                np.testing.assert_array_equal(state.structure[key],value)
        np.testing.assert_array_equal(s.support,replay.support)
        np.testing.assert_array_equal(s.material_surface['vertices'],replay.material_surface['vertices'])
        for key in s.structure:np.testing.assert_array_equal(s.structure[key],replay.structure[key])
        diagnostics=s.native_subduction_diagnostics
        self.assertLessEqual(diagnostics['unique_downgoing_water_area_km2'],diagnostics['local_water_area_before_union_km2']+1e-7)


if __name__=='__main__':unittest.main()
