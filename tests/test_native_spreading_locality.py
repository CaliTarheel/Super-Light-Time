"""Local paired footprints: fixed physical geometry and independent areas."""
import json
import math
from pathlib import Path
import unittest
import numpy as np
import mesh_geometry
import mesh_transport
import native_spreading
from orientation import rotation_matrix
from tests.test_native_spreading_pairing import scene,expected_area
from tests._native_spreading_oracle import material_rectangles,polygon_area,RADIUS


def obstructed(split=1,level=3,refine=False):
    s,axis=scene(split=split,extent=120.,level=level)
    vertices,faces=material_rectangles([(-30.,-2.,0.,140.),(2.,30.,-140.,0.)],refine=refine)
    s.material_surface=dict(vertices=vertices,faces=faces,
        vertex_owner=np.full(len(vertices),2,np.int16))
    return s,axis


def local_area(dt):
    # The finite rotated endcap has this exact straight gnomonic equation.
    # Scalar spherical excess, not production clipping, measures the result.
    x=2./RADIUS;y=120./RADIUS*(1.+math.tan(.001*dt/2.)*x)
    points=np.array([[1.,0.,-120./RADIUS],[1.,x,-y],[1.,x,y],[1.,0.,120./RADIUS]])
    points/=np.linalg.norm(points,axis=1)[:,None]
    return 2.*polygon_area(points)


def turn(s,pose):
    matrix=rotation_matrix(pose)
    for name,value in s.native_mesh.items():
        if isinstance(value,np.ndarray) and value.dtype.kind=='f' and value.ndim==2 and value.shape[1]==3:
            s.native_mesh[name]=value@matrix
    s.xyz=s.native_mesh['xyz'];s.omega=s.omega@matrix
    s.material_surface['vertices']=s.material_surface['vertices']@matrix
    for name in ('segments_start','segments_end','segment_normals'):
        s.native_boundary_geometry[name]=s.native_boundary_geometry[name]@matrix
    locator=mesh_geometry.build_locator(s.native_mesh['vertices'],s.native_mesh['faces'])
    stencil=mesh_transport.face_vertex_stencil(s.native_mesh)
    s._sample_coordinates=lambda at:mesh_transport.sample_coordinates(s.native_mesh,locator,stencil,at)


class NativeSpreadingLocalityTests(unittest.TestCase):
    def advance(self,s,dt=2.):
        before={name:value.copy() for name,value in s.material_surface.items()}
        transported=s.support.copy();birth=native_spreading.advance(s,transported,dt)
        for name,value in before.items():np.testing.assert_array_equal(s.material_surface[name],value)
        np.testing.assert_allclose(transported.sum(axis=0),s.support.sum(axis=0),atol=1e-12)
        report=s.spreading_diagnostics
        self.assertAlmostEqual(report['side_p_area_km2'],report['side_q_area_km2'],delta=1e-6)
        self.assertLessEqual(report.get('maximum_water_capacity_residual_km2',0.),1e-5)
        return float(birth@s.cell_area),birth,transported

    def test_alternating_coasts_cannot_borrow_paired_area_remotely(self):
        for dt in (1.,2.):
            for split in (1,2,4,8):
                s,_=obstructed(split)
                with self.subTest(dt=dt,split=split):
                    area,_,_=self.advance(s,dt)
                    self.assertAlmostEqual(area,local_area(dt),delta=1e-6)
                    self.assertGreater(s.spreading_diagnostics['local_unpaired_area_km2'],0.)

    def test_refinement_polar_rotation_and_owner_order_preserve_locality(self):
        for level in (3,4):
            for split in (1,4):
                s,_=obstructed(split,level,refine=True)
                s.ba,s.bb=s.bb,s.ba;s.bp,s.bq=s.bq,s.bp
                s.native_boundary_geometry['segment_normals']*=-1.
                turn(s,dict(yaw=179.,pitch=89.,roll=37.))
                area,_,_=self.advance(s)
                self.assertAlmostEqual(area,local_area(2.),delta=1e-6)

    def test_duplicate_fronts_union_before_nearly_saturated_cell_capacity(self):
        results=[]
        for split in (1,4):
            for copies in (1,20):
                s,axis=scene(split=split,extent=120.,level=3)
                vertices,faces=material_rectangles([(-2000.,-13.,-2000.,2000.),
                    (13.,2000.,-2000.,2000.),(-13.,13.,-2000.,-121.),(-13.,13.,121.,2000.)])
                s.material_surface=dict(vertices=vertices,faces=faces,vertex_owner=np.full(len(vertices),2,np.int16))
                for name,value in s.native_boundary_geometry.items():
                    s.native_boundary_geometry[name]=np.concatenate([value]*copies)
                area,birth,_=self.advance(s)
                self.assertAlmostEqual(area,expected_area(axis,2.),delta=1e-6)
                if copies>1:self.assertAlmostEqual(s.spreading_diagnostics['duplicate_paired_area_km2'],area*(copies-1),delta=1e-5)
                context=native_spreading.prepare_material(s.material_surface)
                water=0.
                for cell in np.flatnonzero(birth>0.):
                    polygon=s.native_mesh['vertices'][s.native_mesh['faces'][cell]]
                    water+=sum(polygon_area(part) for part in native_spreading.uncovered_polygons(polygon,context))
                self.assertGreater(area/water,.95)
                results.append((area,birth))
        for area,birth in results[1:]:np.testing.assert_allclose(birth,results[0][1],atol=1e-12)

    def test_noncollinear_finite_euler_stages_match_independent_polar_midpoint(self):
        def matrix(vector):
            theta=np.linalg.norm(vector);axis=vector/theta
            x,y,z=axis;cross=np.array([[0.,-z,y],[z,0.,-x],[-y,x,0.]])
            return np.eye(3)*math.cos(theta)+(1.-math.cos(theta))*np.outer(axis,axis)+math.sin(theta)*cross
        for split in (1,4):
            s,axis=scene(split=split)
            s.omega[0]=[.0003,.0002,-.001];s.omega[1]=[-.0002,.0004,.0012]
            rp,rq=matrix(s.omega[0]*2.),matrix(s.omega[1]*2.)
            u,_,vt=np.linalg.svd(rp+rq);rm=u@vt
            middle=axis@rm.T;left=axis@rp.T;right=axis@rq.T
            parea=polygon_area(np.array([middle[0],left[0],left[1],middle[1]]))
            qarea=polygon_area(np.array([middle[0],right[0],right[1],middle[1]]))
            self.assertAlmostEqual(parea,qarea,delta=1e-6)
            area,_,_=self.advance(s)
            self.assertAlmostEqual(area,parea+qarea,delta=1e-6)

    def test_actual_initial_coast_roundoff_fixture_has_no_water_interval(self):
        path=Path(__file__).parent/'fixtures/initial-coast-roundoff.json'
        fixture=json.loads(path.read_text());triangles=np.asarray(fixture['triangles'])
        vertices=triangles.reshape(-1,3);faces=np.arange(len(vertices)).reshape(-1,3)
        context=native_spreading.prepare_material(dict(vertices=vertices,faces=faces))
        self.assertEqual(native_spreading.ocean_intervals(fixture['start'],fixture['end'],context),[])


if __name__=='__main__':unittest.main()
