"""Independent dense variational oracle for a small graded spherical sheet."""
import unittest
import numpy as np
import viscous_sheet as sheet


def fixture(n=12,power=2.,extent=200.,length=500.):
    line=np.linspace(-1.,1.,n+1);line=np.sign(line)*np.abs(line)**power
    xy=np.array([(x,y) for y in line for x in line])*extent
    points=np.column_stack((xy/6371.,np.ones(len(xy))))
    points/=np.linalg.norm(points,axis=1)[:,None]
    faces=[]
    for row in range(n):
        for col in range(n):
            a=row*(n+1)+col;b=a+1;d=a+n+1;c=d+1
            faces.extend(((a,b,c),(a,c,d)))
    faces=np.array(faces,int)
    triangle=points[faces];a,b,c=triangle.transpose(1,0,2)
    area=2.*np.arctan2(np.abs(np.einsum('ij,ij->i',a,np.cross(b,c))),
        1.+np.einsum('ij,ij->i',a,b)+np.einsum('ij,ij->i',b,c)+np.einsum('ij,ij->i',c,a))*6371.**2
    axis=np.eye(3)[np.argmin(np.abs(points),axis=1)]
    first=np.cross(points,axis);first/=np.linalg.norm(first,axis=1)[:,None]
    basis=np.stack((first,np.cross(points,first)),axis=2)
    data=np.bincount(faces.ravel(),weights=np.repeat(area/3.,3),minlength=len(points))
    matrix=np.diag(np.repeat(data,2))
    # Direct assembly of the bilinear tensor form, independent of sheet.apply.
    for ids,tri,weight in zip(faces,triangle*6371.,area):
        e1,e2=tri[1]-tri[0],tri[2]-tri[0]
        cross=np.cross(e1,e2);normal=cross/np.linalg.norm(cross)
        p=np.eye(3)-np.outer(normal,normal)
        gradients=np.array([np.cross(normal,tri[2]-tri[1]),
                            np.cross(normal,tri[0]-tri[2]),
                            np.cross(normal,tri[1]-tri[0])])/np.linalg.norm(cross)
        derivatives=[]
        for corner,index in enumerate(ids):
            for dof in range(2):
                raw=np.outer(p@basis[index,:,dof],gradients[corner])
                derivatives.append(.5*(raw+raw.T))
        d=np.array(derivatives);div=np.trace(d,axis1=1,axis2=2)
        local=(np.einsum('aij,bij->ab',d,d)+np.outer(div,div))*weight*length**2
        indices=(ids[:,None]*2+np.arange(2)).ravel()
        matrix[np.ix_(indices,indices)]+=local
    x,y=xy.T/extent
    modes=np.column_stack((np.sin(2*x)+.3*np.cos(5*y)+.2*x*y,
                           -.8*np.sin(3*y)+.4*np.cos(4*x))).ravel()
    load=matrix@modes
    body=np.einsum('nia,na->ni',basis,(load/np.repeat(data,2)).reshape(-1,2))
    return points,faces,area,basis,matrix,modes,body,length


class GradedModesTests(unittest.TestCase):
    def test_graded_multimode_sheet_converges_without_discarding_krylov_progress(self):
        points,faces,area,basis,matrix,modes,body,length=fixture()
        zero=np.zeros_like(points);mask=np.zeros(len(points),bool)
        velocity,diag=sheet.solve(points,faces,area,zero,mask,mask,length,
                                 body_force=body,iterations=1024,tolerance=1e-8)
        self.assertTrue(diag['converged'],diag)
        actual=np.einsum('nia,ni->na',basis,velocity).ravel()
        load=matrix@modes
        self.assertLessEqual(np.linalg.norm(load-matrix@actual)/np.linalg.norm(load),1e-8)
        self.assertLess(np.linalg.norm(actual-modes)/np.linalg.norm(modes),5e-4)
        self.assertTrue(np.all(np.linalg.eigvalsh(matrix)>0.))


if __name__=='__main__': unittest.main(verbosity=2)
