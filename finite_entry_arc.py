"""Differentiated reference-triangle clipping at a finite inherited trench.

The face's material reference measure is fixed during a mechanical solve. A
signed linear entry coordinate and two endpoint halfspaces divide that measure;
the endpoint crossings move with the face and hinge. This module differentiates
those crossings, so a partially intersected face contributes its actual entry
energy and reciprocal force rather than being accepted or rejected wholesale.
"""
import math
import numpy as np


def endpoint_planes(normal,midpoint,half_length_km,radius_km):
    """Return inward great-circle halfspaces for a trench arc under 180 degrees."""
    n=np.asarray(normal,float);m=np.asarray(midpoint,float)
    length=float(half_length_km);radius=float(radius_km)
    if (n.shape!=(3,) or m.shape!=(3,) or not np.isfinite(n).all()
            or not np.isfinite(m).all() or abs(np.linalg.norm(n)-1.)>1e-10
            or abs(np.linalg.norm(m)-1.)>1e-10 or abs(n@m)>1e-10
            or not np.isfinite(length) or not np.isfinite(radius) or radius<=0.
            or not 0.<length<math.pi*radius/2.):
        raise ValueError('Finite entry arc needs orthogonal unit hinge geometry and a half-length below 90 degrees.')
    angle=length/radius
    tangent=np.cross(n,m)
    sine,cosine=math.sin(angle),math.cos(angle)
    return sine*m+cosine*tangent,sine*m-cosine*tangent,sine,cosine


def _clip(polygon,values,offset):
    """Clip barycentric vertices carrying Jacobians in aligned nodal fields."""
    if not polygon:return []
    output=[]
    for a,b in zip(polygon,polygon[1:]+polygon[:1]):
        va,ja=a;vb,jb=b
        ca=float(va@values);cb=float(vb@values)
        inside_a,inside_b=ca>=0.,cb>=0.
        if inside_a != inside_b:
            da=ja.T@values;db=jb.T@values
            da[offset:offset+3]+=va
            db[offset:offset+3]+=vb
            denominator=ca-cb
            t=ca/denominator
            derivative=(da-t*(da-db))/denominator
            crossing=(va+t*(vb-va),ja+t*(jb-ja)+np.outer(vb-va,derivative))
            output.append(crossing)
        if inside_b:output.append(b)
    return output


def clip_fields(fields):
    """Clip an affine reference face by any number of moving halfspaces.

    `fields[k, i]` is halfspace k's signed value at material vertex i.
    Returned polygon vertices are barycentric weights and their exact
    derivatives in field-major `(3, 3 * n_fields)` order. Values and
    derivatives apply while the clipping topology is fixed.
    """
    fields=np.asarray(fields,float)
    if (fields.ndim!=2 or fields.shape[1]!=3 or not len(fields)
            or not np.isfinite(fields).all()):
        raise ValueError('Affine clipping needs finite aligned nodal halfspaces.')
    width=3*len(fields)
    polygon=[(np.eye(3)[i],np.zeros((3,width))) for i in range(3)]
    for index,field in enumerate(fields):
        polygon=_clip(polygon,field,3*index)
        if not polygon:break
    return polygon


def clipped_reference_area(fields):
    """Return fraction, nodal-field derivative and polygon of an intersection."""
    polygon=clip_fields(fields)
    gradient=np.zeros(np.asarray(fields).shape)
    fraction=0.
    for first in range(1,len(polygon)-1):
        (a,da),(b,db),(c,dc)=polygon[0],polygon[first],polygon[first+1]
        x,y=b[1]-a[1],b[2]-a[2]
        u,v=c[1]-a[1],c[2]-a[2]
        area=x*v-y*u
        if area< -2e-13:
            raise ValueError('Affine clipping reversed its reference winding.')
        if area<=0.:continue
        dx,dy=db[1]-da[1],db[2]-da[2]
        du,dv=dc[1]-da[1],dc[2]-da[2]
        derivative=dx*v+x*dv-dy*u-y*du
        fraction+=area
        gradient+=derivative.reshape(gradient.shape)
    if (fraction>1.+2e-12 or not np.isfinite(fraction)
            or not np.isfinite(gradient).all()):
        raise ValueError('Affine clipping lost its reference area derivative.')
    return fraction,gradient,polygon


def integrate_affine_field(fields, field_index=0):
    """Integrate one nodal affine field over the multiply clipped reference face.

    Return its whole-face reference mean, derivative in every nodal field, and
    the clipped polygon. The moving-boundary and direct integrand terms are
    both included while the clipping topology is fixed.
    """
    fields=np.asarray(fields,float)
    polygon=clip_fields(fields)
    if not isinstance(field_index,(int,np.integer)) or not 0<=field_index<len(fields):
        raise ValueError('Affine integral needs a valid field index.')
    gradient=np.zeros(fields.size)
    integral=0.
    q=fields[field_index]
    for second in range(1,len(polygon)-1):
        corners=[polygon[index] for index in (0,second,second+1)]
        (a,da),(b,db),(c,dc)=corners
        x,y=b[1]-a[1],b[2]-a[2]
        u,v=c[1]-a[1],c[2]-a[2]
        area=x*v-y*u
        if area< -2e-13:
            raise ValueError('Affine integral reversed reference winding.')
        if area<=0.:continue
        dx,dy=db[1]-da[1],db[2]-da[2]
        du,dv=dc[1]-da[1],dc[2]-da[2]
        darea=dx*v+x*dv-dy*u-y*du
        mean_weights=(a+b+c)/3.
        mean_q=float(mean_weights@q)
        dmean_q=(da+db+dc).T@q/3.
        offset=3*field_index
        dmean_q[offset:offset+3]+=mean_weights
        integral+=area*mean_q
        gradient+=darea*mean_q+area*dmean_q
    if not np.isfinite(integral) or not np.isfinite(gradient).all():
        raise ValueError('Affine integral lost its field derivative.')
    return integral,gradient.reshape(fields.shape),polygon


def integrate(q,left,right):
    """Integrate positive entry distance within both finite arc endpoint planes.

    Returns reference-face mean, derivatives with respect to the three values
    of each affine field (q, left, right), positive area fraction, and the
    clipped barycentric triangles for local source quadrature.
    """
    values=[np.asarray(v,float) for v in (q,left,right)]
    if any(v.shape!=(3,) or not np.isfinite(v).all() for v in values):
        raise ValueError('Finite entry clipping needs three finite nodal fields.')
    polygon=clip_fields(values)
    mean=0.;gradient=np.zeros(9);fraction=0.;pieces=[]
    for first in range(1,len(polygon)-1):
        (a,da),(b,db),(c,dc)=polygon[0],polygon[first],polygon[first+1]
        x,y=b[1]-a[1],b[2]-a[2]
        u,v=c[1]-a[1],c[2]-a[2]
        area=x*v-y*u
        if area< -2e-13:
            raise ValueError('Finite entry clipping reversed its reference winding.')
        if area<=0.:continue
        dx,dy=db[1]-da[1],db[2]-da[2]
        du,dv=dc[1]-da[1],dc[2]-da[2]
        darea=dx*v+x*dv-dy*u-y*du
        bary=(a+b+c)/3.
        dbary=(da+db+dc)/3.
        qmean=float(bary@values[0])
        dqmean=dbary.T@values[0]
        dqmean[:3]+=bary
        mean+=area*qmean
        gradient+=darea*qmean+area*dqmean
        fraction+=area
        pieces.append((area,np.stack((a,b,c))))
    if fraction>1.+2e-12 or not np.isfinite(mean) or not np.isfinite(gradient).all():
        raise ValueError('Finite entry clipping lost its reference measure or derivative.')
    return mean,gradient[:3],gradient[3:6],gradient[6:9],fraction,pieces
