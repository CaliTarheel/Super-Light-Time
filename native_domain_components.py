"""Joint native control/material connectivity without nearest-domain aliases.

Material connectivity comes from shared indexed edges, never proximity or
overlap alone. Positive-area intersections attach material to same-owner control
terrain. This allows a real material bridge to join coarse control components,
while a tiny island over foreign control terrain keeps an independent identity.
"""
from __future__ import annotations
import numpy as np


def classify(control_mesh,control_owner,material_surface,coverage):
    """Return deterministic ``control_labels``, ``material_labels`` and ``count``.

    Labels describe current connectivity only. The caller owns persistence,
    display names, material/reference-area accounting and new-domain events.
    Neither geometry, ownership nor the supplied coverage is modified.
    """
    owners=np.asarray(control_owner)
    material_owners=np.asarray(material_surface['face_owner'])
    faces=np.asarray(material_surface['faces'])
    control_edges=np.asarray(control_mesh['edge_faces'])
    control=np.asarray(coverage['control_index'])
    material=np.asarray(coverage['material_index'])
    area=np.asarray(coverage['area_km2'],float)
    n,m=len(owners),len(material_owners)
    if (owners.shape!=(n,) or owners.dtype.kind not in 'iu' or np.any(owners<0)
            or material_owners.shape!=(m,) or material_owners.dtype.kind not in 'iu'
            or np.any(material_owners<0)
            or faces.shape!=(m,3) or faces.dtype.kind not in 'iu'
            or np.any(faces<0) or np.any(faces>=len(material_surface['vertices']))
            or control_edges.ndim!=2 or control_edges.shape[1:]!=(2,)
            or control_edges.dtype.kind not in 'iu' or np.any(control_edges<0)
            or np.any(control_edges>=n)
            or control.ndim!=1 or material.shape!=control.shape or area.shape!=control.shape
            or control.dtype.kind not in 'iu' or material.dtype.kind not in 'iu'
            or np.any(control<0) or np.any(control>=n) or np.any(material<0) or np.any(material>=m)
            or not np.isfinite(area).all() or np.any(area<0)):
        raise ValueError('Joint domains require aligned native owners, connectivity and finite coverage.')
    parent=np.arange(n+m,dtype=np.int64)

    def find(i):
        while parent[i]!=i:
            parent[i]=parent[parent[i]]
            i=int(parent[i])
        return i

    def join(a,b):
        for left,right in zip(a,b):
            left,right=find(int(left)),find(int(right))
            if left!=right:
                parent[max(left,right)]=min(left,right)

    a,b=control_edges.T
    same=owners[a]==owners[b]
    join(a[same],b[same])
    if m:
        # Group references to each actual indexed edge. Sorting endpoints is
        # independent of face winding and coordinate rotation. Nonmanifold
        # material inputs, if supplied, form a spanning star for each owner;
        # no quadratic clique is needed to preserve their connectivity.
        edges=np.concatenate((faces[:,(0,1)],faces[:,(1,2)],faces[:,(2,0)]))
        edges.sort(axis=1)
        face_index=np.tile(np.arange(m,dtype=np.int64),3)
        order=np.lexsort((material_owners[face_index],edges[:,1],edges[:,0]))
        sorted_edges=edges[order]
        sorted_face=face_index[order]
        sorted_owner=material_owners[sorted_face]
        continuation=(np.all(sorted_edges[1:]==sorted_edges[:-1],axis=1)
                      &(sorted_owner[1:]==sorted_owner[:-1]))
        join(n+sorted_face[:-1][continuation],n+sorted_face[1:][continuation])
    attached=(area>0)&(owners[control]==material_owners[material])
    join(control[attached],n+material[attached])
    roots=np.fromiter((find(i) for i in range(n+m)),dtype=np.int64,count=n+m)
    unique,labels=np.unique(roots,return_inverse=True)
    return dict(control_labels=labels[:n].astype(np.int32),
                material_labels=labels[n:].astype(np.int32),count=len(unique))
