"""ISOLATED PROPOSAL: directed finite-front winding as disjoint convex regions.

An oriented fan is only a boundary representation. Areas are measured after
geometric overlay, never by adding signed fan areas. Opening regions are not
capture. Sources are joined only through actual endpoints and equal owner pair.
"""
from collections import defaultdict
import math
import numpy as np
import mesh_coverage
import native_spreading as exact
from ridge_geometry import rotate

VERSION=1
RADIUS_KM=6371.
ENDPOINT_TOLERANCE=2e-12

def _unit(x):
    x=np.asarray(x,float)
    return x/np.maximum(np.linalg.norm(x,axis=-1,keepdims=True),1e-30)

def _area(p):return mesh_coverage._polygon_area(p,RADIUS_KM)

def _atom(poly,winding):
    center=_unit(np.sum(poly,axis=0))
    angle=float(np.arctan2(np.linalg.norm(np.cross(poly,center),axis=1),poly@center).max(initial=0.))
    return dict(polygon=poly,winding=winding,center=center,angle=angle)

def _disjoint(a,b):
    limit=a['angle']+b['angle']
    return limit<math.pi and float(a['center']@b['center'])<math.cos(limit)-1e-13

def _overlay(atoms,triangle,source,sign):
    """Overlay one directed triangle; retain local integer source winding."""
    incoming=_atom(triangle,{source:sign});new_parts=[triangle];result=[]
    context={'radius_km':RADIUS_KM}
    for prior in atoms:
        if _disjoint(prior,incoming):result.append(prior);continue
        intersection=exact._intersect_polygon(prior['polygon'],triangle)
        if _area(intersection)<=exact.AREA_TOLERANCE_KM2:
            result.append(prior);continue
        result.extend(_atom(p,dict(prior['winding'])) for p in
                      exact._subtract_polygons(prior['polygon'],[triangle],context))
        winding=dict(prior['winding']);winding[source]=winding.get(source,0)+sign
        if not winding[source]:del winding[source]
        if winding:result.append(_atom(intersection,winding))
        new_parts=[piece for p in new_parts for piece in exact._subtract_polygons(p,[prior['polygon']],context)]
    result.extend(_atom(p,{source:sign}) for p in new_parts)
    return result

def partition(sources):
    """One connected front; output geometric positive winding with local maturity.

    Input sources have an oriented polygon whose positive orientation denotes
    convergence and scalar maturity. Source indices refer only to this call.
    """
    atoms=[]
    for source,row in enumerate(sources):
        polygon=np.asarray(row['polygon'],float)
        center=_unit(polygon.sum(axis=0))
        if np.any(polygon@center<=0.):raise ValueError('Finite capture ribbon must lie in one minor hemisphere.')
        for i in range(1,len(polygon)-1):
            tri=polygon[[0,i,i+1]]
            if _area(tri)<=exact.AREA_TOLERANCE_KM2:continue
            det=float(np.linalg.det(tri));sign=1 if det>0 else -1
            if sign<0:tri=tri[::-1]
            atoms=_overlay(atoms,tri,source,sign)
    result=[]
    for atom in atoms:
        if sum(atom['winding'].values())<=0:continue
        witnesses=[i for i,w in atom['winding'].items() if w>0]
        maturity=max(float(sources[i]['maturity']) for i in witnesses)
        if maturity<=0:continue
        poly=atom['polygon'];area=_area(poly)
        if area<=exact.AREA_TOLERANCE_KM2:continue
        parents=sorted({int(sources[i]['parent']) for i in witnesses})
        chosen=min(witnesses,key=lambda i:(-float(sources[i]['maturity']),int(sources[i]['parent']),i))
        result.append(dict(parent=int(sources[chosen]['parent']),down=int(sources[chosen]['down']),
                           over=int(sources[chosen]['over']),maturity=maturity,polygon=poly,
                           raw_area_km2=area,source_parents=parents,source_indices=witnesses,
                           capture_geometry_version=VERSION))
    return result

def _components(sources):
    """Endpoint proof, with neighboring bins used only to find candidates."""
    parent=list(range(len(sources)));bins=defaultdict(list)
    def find(i):
        while parent[i]!=i:parent[i]=parent[parent[i]];i=parent[i]
        return i
    for i,row in enumerate(sources):
        for point in row['endpoints']:
            key=np.floor(point/ENDPOINT_TOLERANCE).astype(np.int64)
            for dx in (-1,0,1):
                for dy in (-1,0,1):
                    for dz in (-1,0,1):
                        for j,other in bins.get((row['down'],row['over'],*(key+[dx,dy,dz])),[]):
                            if np.linalg.norm(point-other)<=ENDPOINT_TOLERANCE:
                                a,b=find(i),find(j)
                                if a!=b:parent[max(a,b)]=min(a,b)
            bins[(row['down'],row['over'],*key)].append((i,point))
    groups=defaultdict(list)
    for i in range(len(sources)):groups[find(i)].append(sources[i])
    return list(groups.values())

def capture_polygons(s,dt):
    if not np.isfinite(dt) or dt<=0:raise ValueError('Native capture needs positive finite time.')
    contour=s.native_boundary_geometry
    maturity=np.asarray(getattr(s,'trench_maturity',np.zeros(len(s.ba))),float)
    if maturity.shape!=(len(s.ba),) or not np.isfinite(maturity).all() or np.any((maturity<0)|(maturity>1)):
        raise ValueError('Native trench maturity must be finite and lie in [0,1].')
    import normal_partition
    candidates=normal_partition.capture_candidates(s)
    sources=[]
    for a,b,normal,parent in zip(contour['segments_start'],contour['segments_end'],contour['segment_normals'],contour['contact_index']):
        parent=int(parent);p=int(s.down[parent])
        if not candidates[parent] or p not in (s.bp[parent],s.bq[parent]):continue
        q=int(s.bq[parent] if p==s.bp[parent] else s.bp[parent])
        if not (s.active[p] and s.active[q]):continue
        normal=np.asarray(normal,float)*(1 if p==s.bp[parent] else -1)
        endpoints=np.array([a,b],float)
        # This directed orientation is physical polarity, not the sign of an
        # arbitrarily chosen first fan triangle in a potentially crossed quad.
        if np.cross(a,b)@normal>0:endpoints=endpoints[::-1]
        incoming=rotate(endpoints,np.asarray(s.omega[p])*dt)
        overriding=rotate(endpoints,np.asarray(s.omega[q])*dt)
        polygon=np.array([incoming[0],incoming[1],overriding[1],overriding[0]])
        sources.append(dict(parent=parent,down=p,over=q,maturity=float(maturity[parent]),
                            endpoints=endpoints,polygon=polygon))
    result=[]
    for group in _components(sources):result.extend(partition(group))
    result.sort(key=lambda row:(-row['maturity'],row['down'],row['over'],tuple(np.round(row['polygon'].mean(axis=0),14))))
    return result
