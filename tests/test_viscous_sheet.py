"""Constitutive, variational and geometric checks for the isolated solver."""
import importlib.util
from pathlib import Path
import sys
import unittest
import numpy as np

import viscous_sheet as sheet


def unit(value):
    value=np.asarray(value,float)
    return value/np.linalg.norm(value,axis=-1,keepdims=True)


def area(points,faces,radius):
    a,b,c=points[faces].transpose(1,0,2)
    return 2*np.arctan2(np.abs(np.einsum('ij,ij->i',a,np.cross(b,c))),
        1+np.einsum('ij,ij->i',a,b)+np.einsum('ij,ij->i',b,c)+np.einsum('ij,ij->i',c,a))*radius**2


def rotation(angle=.83,axis=(.2,.7,-.1)):
    axis=unit(axis);x,y,z=axis
    cross=np.array([[0,-z,y],[z,0,-x],[-y,x,0]])
    return np.eye(3)*np.cos(angle)+(1-np.cos(angle))*np.outer(axis,axis)+np.sin(angle)*cross


def patch(n=6,extent=20.,radius=6371.):
    xy=np.array([(x,y) for y in np.linspace(-extent,extent,n+1)
                 for x in np.linspace(-extent,extent,n+1)])
    faces=[];centres=[]
    for row in range(n):
        for col in range(n):
            a=row*(n+1)+col;b=a+1;d=a+n+1;c=d+1
            middle=len(xy)+len(centres);centres.append(xy[[a,b,c,d]].mean(axis=0))
            faces.extend(((a,b,middle),(b,c,middle),(c,d,middle),(d,a,middle)))
    xy=np.vstack((xy,centres));points=unit(np.column_stack((xy/radius,np.ones(len(xy)))))
    faces=np.asarray(faces,int)
    return points,faces,area(points,faces,radius),xy


def tangent(values,points):
    return values-np.sum(values*points,axis=1)[:,None]*points


class ViscousSheetTests(unittest.TestCase):
    def test_common_euler_rotation_has_zero_symmetric_strain(self):
        points,faces,areas,_=patch(n=5,extent=700)
        context=sheet.prepare(points,faces,areas,500.)
        velocity=np.cross(np.broadcast_to([.003,-.002,.005],points.shape),points)*6371.
        fields=sheet.strain_rate(context,velocity)
        self.assertLess(float(np.max(np.abs(fields['D']))),2e-16)
        self.assertLess(float(np.max(np.abs(fields['divergence']))),2e-16)
        np.testing.assert_allclose(sheet.apply(context,velocity),areas.sum()*0+context['data'][:,None]*velocity,rtol=1e-12,atol=1e-7)

    def test_spd_virtual_work_matches_tensor_energy(self):
        points,faces,areas,_=patch(n=3,extent=80.)
        context=sheet.prepare(points,faces,areas,90.,viscosity_weights=np.linspace(.2,1.8,len(faces)))
        rng=np.random.default_rng(412)
        u=tangent(rng.normal(size=points.shape),points);v=tangent(rng.normal(size=points.shape),points)
        au=sheet.apply(context,u);av=sheet.apply(context,v)
        self.assertAlmostEqual(float(np.sum(u*av))/float(np.sum(v*au)),1.,places=13)
        du=sheet.strain_rate(context,u);dv=sheet.strain_rate(context,v)
        expected=float(np.sum(context['data'][:,None]*u*v)+np.sum(context['face_scale']*(
            np.einsum('fij,fij->f',du['D'],dv['D'])+du['divergence']*dv['divergence'])))
        np.testing.assert_allclose(np.sum(u*av),expected,rtol=5e-14,atol=1e-8)
        self.assertGreater(float(np.sum(u*au)),float(np.sum(context['data'][:,None]*u*u)))

    def test_force_free_without_boundary_conditions_is_exact_zero(self):
        points,faces,areas,_=patch(n=3)
        zero=np.zeros_like(points);mask=np.zeros(len(points),bool)
        velocity,info=sheet.solve(points,faces,areas,zero,mask,mask,100.)
        self.assertTrue(info['converged']);self.assertEqual(info['iterations'],0)
        np.testing.assert_array_equal(velocity,zero)

    def compression(self,points,faces,areas,xy,*,normal_mode=True,length=5000.,iterations=1600):
        target=np.column_stack((-.1*xy[:,0],np.zeros((len(xy),2))))
        mask=np.zeros(len(points),bool)
        driven=np.isclose(np.abs(xy[:,0]),np.max(xy[:,0]))
        normals=np.tile([1.,0,0],(len(points),1)) if normal_mode else None
        return sheet.solve(points,faces,areas,target,mask,driven,length,
            driven_normals=normals,iterations=iterations,tolerance=2e-10)

    def test_uniaxial_compression_free_flanks_extend_at_half_rate(self):
        points,faces,areas,xy=patch(n=6,extent=20.)
        velocity,info=self.compression(points,faces,areas,xy)
        self.assertTrue(info['converged'],info)
        exx=float(xy[:,0]@velocity[:,0]/(xy[:,0]@xy[:,0]))
        eyy=float(xy[:,1]@velocity[:,1]/(xy[:,1]@xy[:,1]))
        # Free traction gives sigma_yy=2 eta H (exx+2 eyy)=0.
        self.assertAlmostEqual(eyy/-exx,.5,delta=3e-4)
        self.assertGreater(eyy,0.)
        self.assertLess(info['relative_residual'],2e-10)
        self.assertGreater(info['viscous_dissipation_quadratic'],0.)

    def test_normal_only_driving_preserves_mirrored_head_on_symmetry(self):
        points,faces,areas,xy=patch(n=4)
        velocity,info=self.compression(points,faces,areas,xy)
        self.assertTrue(info['converged'],info)
        lookup={tuple(np.round(pair,10)):i for i,pair in enumerate(xy)}
        for i,(x,y) in enumerate(xy):
            mirror=lookup[tuple(np.round([-x,y],10))]
            np.testing.assert_allclose(velocity[mirror],velocity[i]*[-1,1,1],rtol=0,atol=3e-7)

    def test_solution_rotation_covariance_through_pole(self):
        points,faces,areas,xy=patch(n=4)
        target=tangent(np.column_stack((-.1*xy[:,0],.02*xy[:,1],np.zeros(len(xy)))),points)
        force=tangent(np.broadcast_to([.02,-.015,0.],points.shape),points)
        rigid=np.zeros(len(points),bool);driven=np.isclose(np.abs(xy[:,0]),20.)
        normals=np.broadcast_to([1.,0,0],points.shape)
        reference,info=sheet.solve(points,faces,areas,target,rigid,driven,60.,body_force=force,driven_normals=normals,tolerance=1e-10,iterations=600)
        self.assertTrue(info['converged'],info)
        for angle,axis in ((1.57,[0,1,0]),(2.3,[.2,.7,.4])):
            q=rotation(angle,axis)
            moved,other=sheet.solve(points@q.T,faces,areas,target@q.T,rigid,driven,60.,
                body_force=force@q.T,driven_normals=normals@q.T,tolerance=1e-10,iterations=600)
            self.assertTrue(other['converged'],other)
            np.testing.assert_allclose(moved,reference@q.T,rtol=1e-7,atol=2e-7)

    def test_drag_only_body_force_full_driven_rigid_and_unused_nodes(self):
        points,faces,areas,_=patch(n=2)
        points=np.vstack((points,unit([1,1,1])))
        target=tangent(np.tile([1.,2.,3.],(len(points),1)),points)
        body=tangent(np.tile([-.3,.2,.1],(len(points),1)),points)
        rigid=np.zeros(len(points),bool);rigid[0]=True
        driven=np.zeros(len(points),bool);driven[1]=True
        before=[v.copy() for v in (points,faces,areas,target,body,rigid,driven)]
        velocity,info=sheet.solve(points,faces,areas,target,rigid,driven,0.,body_force=body,tolerance=1e-10)
        self.assertTrue(info['converged'],info)
        expected=target+body;expected[0]=0;expected[1]=target[1];expected[-1]=0
        np.testing.assert_allclose(velocity,expected,rtol=0,atol=2e-14)
        self.assertEqual(info['unused_vertices'],1)
        for old,current in zip(before,(points,faces,areas,target,body,rigid,driven)):np.testing.assert_array_equal(old,current)

    def test_iteration_failure_and_sliver_emit_no_unverified_velocity(self):
        points,faces,areas,xy=patch(n=3)
        velocity,info=self.compression(points,faces,areas,xy,iterations=1)
        self.assertFalse(info['converged']);self.assertIsNotNone(info['failure_reason'])
        np.testing.assert_array_equal(velocity,np.zeros_like(points))
        # A real nondegenerate, very slender spherical element is accepted as
        # geometry. An unresolved ill-conditioned response is never emitted.
        p=unit(np.array([[0.,0.,1.],[.001,0.,1.],[.0005,1e-10,1.],[0,.001,1.]]))
        f=np.array([[0,1,2],[0,2,3]],int);a=area(p,f,6371.)
        target=tangent(np.array([[1.,2.,0.],[-1,1,0.],[.3,-.4,0.],[1,-2,0.]]),p)
        mask=np.zeros(4,bool)
        answer,diagnostic=sheet.solve(p,f,a,target,mask,mask,100.,iterations=1,tolerance=1e-13)
        self.assertFalse(diagnostic['converged']);np.testing.assert_array_equal(answer,np.zeros_like(answer))
        # Spherical sagitta keeps the embedded planar triangle less slender
        # than its spherical footprint; test the actual reported planar ratio.
        self.assertLess(diagnostic['minimum_planar_quality'],1e-3)

    def test_malformed_units_degenerate_geometry_and_normals_raise(self):
        points,faces,areas,_=patch(n=2)
        zero=np.zeros_like(points);mask=np.zeros(len(points),bool)
        with self.assertRaises(ValueError):sheet.solve(points*6371,faces,areas,zero,mask,mask,100.)
        with self.assertRaises(ValueError):sheet.prepare(points,np.array([[0,0,1]]),np.ones(1),100.)
        with self.assertRaises(ValueError):sheet.prepare(points,faces,areas,100.,viscosity_weights=-1.)
        with self.assertRaises(ValueError):sheet.solve(points,faces,areas,zero,mask,~mask,100.,driven_normals=points)
        with self.assertRaises(ValueError):sheet.solve(points,faces,areas,zero,mask,mask,100.,iterations=True)


if __name__=='__main__':unittest.main()
