"""Read-only footprint map using the native affine nodal entry coordinate.

The material face supplies a barycentric coordinate `w`. Native entry energy
uses `q(w) = sum(w_i q_i)` with nodal great-circle hinge distances `q_i`.
This map gives that *same* q a horizontal angle `q cos(dip)/R` after entry,
while preserving the face point's along-hinge azimuth. Before entry it uses
angle `q/R`; the two pieces join continuously at the hinge. The map is a
finite-element approximation and must remain locally one-to-one.
"""

import math

import numpy as np

import mesh_coverage


class AffineConveyorFootprint:
    def __init__(self, triangle, hinge_normal, dip_degrees, *, radius_km=6371.):
        face = np.asarray(triangle,float)
        normal = np.asarray(hinge_normal,float)
        dip = float(dip_degrees)
        radius = float(radius_km)
        if (face.shape != (3,3) or not np.isfinite(face).all()
                or np.any(np.abs(np.linalg.norm(face,axis=1)-1.)>1e-10)
                or normal.shape != (3,) or not np.isfinite(normal).all()
                or abs(float(np.linalg.norm(normal))-1.)>1e-10
                or not math.isfinite(dip) or not 0. < dip < 90.
                or not math.isfinite(radius) or radius <= 0.
                or np.any(np.abs(face@normal)>=1.-1e-12)
                or np.dot(face[0],np.cross(face[1],face[2]))<=0.
                or mesh_coverage._polygon_area(face,radius)<=0.):
            raise ValueError('Affine footprint needs an outward unit face, hinge and finite dip.')
        self.face=face.copy()
        self.normal=normal.copy()
        self.factor=math.cos(math.radians(dip))
        self.radius_m=radius*1000.
        self.area_km2=mesh_coverage._polygon_area(face,radius)
        self.q=self.radius_m*np.arcsin(face@normal)
        samples=np.vstack((np.eye(3),np.full(3,1./3.)))
        signs=self._signed_area_ratio(samples)
        if np.any(~np.isfinite(signs)) or np.min(np.abs(signs))<=1e-9 or not np.all(signs*signs[-1]>0.):
            raise ValueError('Affine footprint folds or degenerates inside its material face.')
        self.orientation=float(np.sign(signs[-1]))
        self.face.flags.writeable=False
        self.normal.flags.writeable=False
        self.q.flags.writeable=False

    def _checked_weights(self, weights):
        bary=np.asarray(weights,float)
        if (bary.ndim<1 or bary.shape[-1]!=3 or not np.isfinite(bary).all()
                or np.any(np.abs(bary.sum(axis=-1)-1.)>1e-9)):
            raise ValueError('Affine footprint needs finite barycentric material positions.')
        return bary

    def project(self, weights):
        """Map barycentric material positions to physical spherical positions."""
        bary=self._checked_weights(weights)
        direction=bary@self.face
        parallel=direction-np.expand_dims(direction@self.normal,-1)*self.normal
        length=np.linalg.norm(parallel,axis=-1,keepdims=True)
        if np.any(length<=1e-10):
            raise ValueError('Affine footprint reaches an undefined hinge azimuth.')
        hinge=parallel/length
        q=bary@self.q
        angle=q*np.where(q>0.,self.factor,1.)/self.radius_m
        return hinge*np.expand_dims(np.cos(angle),-1)+self.normal*np.expand_dims(np.sin(angle),-1)

    def unproject(self, points):
        """Invert a physical point to the face's affine material coordinate."""
        point=np.asarray(points,float)
        if (point.ndim<1 or point.shape[-1]!=3 or not np.isfinite(point).all()
                or np.any(np.abs(np.linalg.norm(point,axis=-1)-1.)>1e-10)):
            raise ValueError('Affine footprint inverse needs unit physical points.')
        sine=np.clip(point@self.normal,-1.,1.)
        angle=np.arcsin(sine)
        if np.any((angle>0.)&(angle>=self.factor*math.pi/2.-1e-10)):
            raise ValueError('Affine footprint inverse leaves its positive entry domain.')
        parallel=point-np.expand_dims(sine,-1)*self.normal
        length=np.linalg.norm(parallel,axis=-1,keepdims=True)
        if np.any(length<=1e-10):
            raise ValueError('Affine footprint inverse has no hinge azimuth.')
        hinge=parallel/length
        tangent=np.cross(self.normal,hinge)
        target_q=angle*self.radius_m/np.where(angle>0.,self.factor,1.)
        shape=point.shape[:-1]
        rows=np.stack((np.broadcast_to(np.ones(3),shape+(3,)),
                       np.broadcast_to(self.q,shape+(3,)),
                       tangent@self.face.T),axis=-2)
        rhs=np.stack((np.ones(shape),target_q,np.zeros(shape)),axis=-1)
        try:
            bary=np.linalg.solve(rows,rhs[...,None])[...,0]
        except np.linalg.LinAlgError as error:
            raise ValueError('Affine footprint inverse is singular.') from error
        if (not np.isfinite(bary).all() or
                np.any(np.sum((bary@self.face)*hinge,axis=-1) < -1e-10)):
            raise ValueError('Affine footprint inverse selected the wrong hinge azimuth.')
        return bary

    def _signed_area_ratio(self, weights):
        bary=self._checked_weights(weights)
        direction=bary@self.face
        parallel=direction-np.expand_dims(direction@self.normal,-1)*self.normal
        length2=np.sum(parallel*parallel,axis=-1)
        if np.any(length2<=1e-20):
            raise ValueError('Affine footprint area is singular at the hinge pole.')
        du=self.face[1]-self.face[0]
        dv=self.face[2]-self.face[0]
        phi_u=np.sum(np.cross(parallel,du)*self.normal,axis=-1)/length2
        phi_v=np.sum(np.cross(parallel,dv)*self.normal,axis=-1)/length2
        q=bary@self.q
        factor=np.where(q>0.,self.factor,1.)
        beta=q*factor/self.radius_m
        beta_u=factor*(self.q[1]-self.q[0])/self.radius_m
        beta_v=factor*(self.q[2]-self.q[0])/self.radius_m
        return (self.radius_m**2*np.cos(beta)*(beta_u*phi_v-beta_v*phi_u)
                /(2.*self.area_km2*1e6))

    def reference_to_footprint_ratio(self, weights):
        """Physical horizontal area per conserved barycentric reference area."""
        ratio=self._signed_area_ratio(weights)*self.orientation
        if np.any(ratio<=0.) or not np.isfinite(ratio).all():
            raise ValueError('Affine footprint folded at the requested material site.')
        return ratio
