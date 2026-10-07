"""Exact free-outline cancellation across differently subdivided fault sides.

Cartesian bins select nearby candidate edges. Only oppositely oriented,
coincident great-circle intervals cancel. Real uncovered intervals survive;
no nearest-neighbor snapping, tiny-gap closing or all-edge quadratic matrix.
"""
from __future__ import annotations
import numpy as np


def _unit(x):return x/np.maximum(np.linalg.norm(x,axis=-1,keepdims=True),1e-30)


def free_outline(vertices,faces):
    vertices,faces=np.asarray(vertices,float),np.asarray(faces)
    if not len(faces):
        return dict(a=np.empty((0,3)),b=np.empty((0,3)),face=np.empty(0,np.int32),
            joins=np.empty((0,2),np.int32),diagnostics=dict(candidate_edge_pairs=0,
                partial_contact_pairs=0,cancelled_contact_length_km=0.,raw_free_edges=0))
    pair=np.array([[0,1],[1,2],[2,0]])
    original=faces[:,pair].reshape(-1,2)
    # Cancel actual shared-index internal edges first, in opposite directed
    # pairs. Coincident coordinates alone do not make an internal seam: two
    # independent sheets can have the same-facing coast at the same place.
    # Duplicated fault vertices proceed to the geometric opposite-interval
    # test below, which also handles nonconforming subdivisions.
    edges=np.sort(original,axis=1)
    _,inverse,counts=np.unique(edges,axis=0,return_inverse=True,return_counts=True)
    order=np.argsort(inverse,kind='stable');starts=np.r_[0,np.cumsum(counts)[:-1]]
    joins=[];internal=np.zeros(len(original),bool)
    for start,count in zip(starts[counts>1],counts[counts>1]):
        members=order[start:start+count]
        forward=members[original[members,0]<original[members,1]]
        backward=members[original[members,0]>original[members,1]]
        for first,second in zip(forward,backward):
            internal[first]=True;internal[second]=True
            joins.append((int(first//3),int(second//3)))
    selected=np.flatnonzero(~internal)
    a,b=vertices[original[selected]].transpose(1,0,2)
    owner_face=selected//3
    normal=_unit(np.cross(a,b));tangent=np.cross(normal,a)
    length=np.arctan2(np.linalg.norm(np.cross(a,b),axis=1),np.sum(a*b,axis=1))
    mid=_unit(a+b);radius=2.*np.sin(length/4.)+2e-11
    resolution=32
    low=np.clip(np.floor((mid-radius[:,None]+1)*resolution/2).astype(int),0,resolution-1)
    high=np.clip(np.floor((mid+radius[:,None]+1)*resolution/2).astype(int),0,resolution-1)
    bins={};edge_keys=[];global_edges=[]
    for edge,(lo,hi) in enumerate(zip(low,high)):
        if np.prod(hi-lo+1)>2048:
            global_edges.append(edge);edge_keys.append(None);continue
        keys=[]
        for x in range(lo[0],hi[0]+1):
            for y in range(lo[1],hi[1]+1):
                for z in range(lo[2],hi[2]+1):
                    key=int(x+resolution*(y+resolution*z));keys.append(key)
                    bins.setdefault(key,[]).append(edge)
        edge_keys.append(keys)
    covered=[[] for _ in a];tested=0;contacts=0
    for i,keys in enumerate(edge_keys):
        possible=(np.arange(i+1,len(a)) if keys is None else
                  np.unique([edge for key in keys for edge in bins[key]]+global_edges))
        possible=possible[possible>i].astype(np.int64)
        if not len(possible):continue
        tested+=len(possible)
        exact=((normal[possible]@normal[i]<-1.+1e-12)
               &(np.abs(a[possible]@normal[i])<2e-11)
               &(np.abs(b[possible]@normal[i])<2e-11))
        for j in possible[exact]:
            # Opposite direction means j runs from angle theta backwards by
            # its minor-arc length. Try periodic copies for very long arcs.
            theta=float(np.arctan2(a[j]@tangent[i],a[j]@a[i]))
            for shift in (-2*np.pi,0.,2*np.pi):
                lo=max(0.,theta+shift-length[j]);hi=min(length[i],theta+shift)
                if hi-lo<=2e-11:continue
                covered[i].append((lo,hi))
                # The same physical interval in j's opposite coordinate.
                covered[j].append((max(0.,theta+shift-hi),min(length[j],theta+shift-lo)))
                joins.append((int(owner_face[i]),int(owner_face[j])));contacts+=1
    out_a=[];out_b=[];out_faces=[]
    cancelled=0.
    for i,intervals in enumerate(covered):
        cursor=0.
        uncovered=[]
        for lo,hi in sorted(intervals):
            if lo>cursor+2e-11:uncovered.append((cursor,lo))
            if hi>cursor:cancelled+=hi-max(cursor,lo)
            cursor=max(cursor,hi)
        if cursor<length[i]-2e-11:uncovered.append((cursor,length[i]))
        for lo,hi in uncovered:
            out_a.append(a[i] if lo==0. else a[i]*np.cos(lo)+tangent[i]*np.sin(lo))
            out_b.append(b[i] if hi==length[i] else a[i]*np.cos(hi)+tangent[i]*np.sin(hi))
            out_faces.append(owner_face[i])
    return dict(a=np.asarray(out_a,float).reshape(-1,3),b=np.asarray(out_b,float).reshape(-1,3),
        face=np.asarray(out_faces,np.int32),joins=np.asarray(joins,np.int32).reshape(-1,2),
        diagnostics=dict(candidate_edge_pairs=tested,partial_contact_pairs=contacts,
            cancelled_contact_length_km=float(cancelled*6371./2.),raw_free_edges=len(a)))
