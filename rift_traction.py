"""Force-derived loading for progressive continental rifting.

The plate solver already computes slab gravity from the conserved retained slab
inventory. This module samples that same local line force onto continental
material and supplies a reduced membrane-compliance target. Rigid motion is
removed by rift_mechanics, so a uniform traction that only spins/translates one
connected continent cannot manufacture strain.

The N/m -> km/Myr mapping is an explicit reduced compliance, not a stress solve.
"""
from __future__ import annotations

import math
import numpy as np

import slab_memory

VERSION=1
RADIUS_KM=6371.
DEFAULTS=dict(version=VERSION,enabled=False,reference_strength_n_per_m=1e13,
              mobility_km_myr=12.,reach_km=900.,max_equivalent_speed_km_myr=40.,
              boundary_motion_fraction=.25)


def normalize(value=None):
    if value is None:return dict(DEFAULTS)
    if not isinstance(value,dict) or set(value)-set(DEFAULTS):
        raise ValueError('Rift traction settings contain unknown fields.')
    out=dict(DEFAULTS,**value)
    if type(out['version']) is not int or out['version']!=VERSION or type(out['enabled']) is not bool:
        raise ValueError('Rift traction requires version 1 and a boolean enabled flag.')
    for key,low,high in (
        ('reference_strength_n_per_m',1e11,1e15),('mobility_km_myr',.1,100.),
        ('reach_km',100.,3000.),('max_equivalent_speed_km_myr',1.,200.),
        ('boundary_motion_fraction',0.,1.)):
        v=out[key]
        if isinstance(v,(bool,np.bool_)) or not isinstance(v,(int,float,np.number)) or not np.isfinite(v) or not low<=float(v)<=high:
            raise ValueError(f'Rift traction {key} must be finite and between {low} and {high}.')
        out[key]=float(v)
    return out


def enabled(s):
    return bool(getattr(s,'config',{}).get('rift_traction',{}).get('enabled',False))


def _unit(v):
    v=np.asarray(v,float)
    return v/np.maximum(np.linalg.norm(v,axis=-1,keepdims=True),1e-30)


def loading(s,mesh):
    """Return equivalent soft velocity loads derived from local slab line force."""
    settings=normalize(s.config.get('rift_traction'))
    count=len(mesh['xyz'])
    loads=np.zeros((count,3));weights=np.zeros(count)
    if not settings['enabled'] or not count:
        return loads,weights,dict(version=VERSION,enabled=False)

    owners,slab_load,_=slab_memory.line_load(s)
    owners=np.asarray(owners,int);slab_load=np.asarray(slab_load,float)
    length=np.asarray(s.bl,float)
    active=(owners>=0)&(slab_load>0.)&(length>1e-10)
    if not np.any(active):
        return loads,weights,dict(version=VERSION,enabled=True,trench_edges=0,
            trench_length_km=0.,max_line_force_n_per_m=0.,max_equivalent_speed_km_myr=0.,
            boundary_motion_fraction=settings['boundary_motion_fraction'])

    dip=math.radians(slab_memory.SUBDUCTION_DIP_DEG)
    line_force=9.81*slab_load*np.sin(dip)
    # Convert the physical integrated-strength ratio into a bounded equivalent
    # membrane-loading speed. The force itself and conversion are both reported.
    equivalent=np.clip(settings['mobility_km_myr']*line_force/settings['reference_strength_n_per_m'],
                       0.,settings['max_equivalent_speed_km_myr'])
    points=np.asarray(mesh['xyz'],float)
    areas=np.asarray(mesh['area'],float)
    mesh_owners=np.asarray(mesh['owners'],int)
    total_force_n=0.
    reaction_length=0.

    import plate_balance
    moving=plate_balance.subduction_response_version(s)==1
    edges=np.flatnonzero(active)
    for edge in edges:
        down=int(owners[edge])
        over=int(s.bq[edge] if s.bp[edge]==down else s.bp[edge])
        toward=np.asarray(s.bn[edge],float)*(1. if s.bp[edge]==down else -1.)
        midpoint=np.asarray(s.bmid[edge],float)
        toward=_unit((toward-midpoint*np.dot(toward,midpoint))[None])[0]
        cases=[(down,toward,1.)]
        if moving:
            cases.append((over,-toward,1.))
            reaction_length+=float(length[edge])
        for owner,direction,scale in cases:
            nodes=np.flatnonzero(mesh_owners==owner)
            if not len(nodes):continue
            # Physical reach is independent of mesh density; broader material
            # nodes receive smoothly tapered soft constraints.
            dot=np.clip(points[nodes]@midpoint,-1.,1.)
            distance=RADIUS_KM*np.arccos(dot)
            taper=np.maximum(1.-distance/settings['reach_km'],0.)**2
            use=taper>0.
            if not np.any(use):continue
            target=nodes[use]
            w=taper[use]*float(length[edge])
            vector=direction*(equivalent[edge]*scale)
            loads[target]+=w[:,None]*vector
            weights[target]+=w
        total_force_n+=float(line_force[edge]*length[edge]*1000.)

    loaded=weights>0.
    loads[loaded]/=weights[loaded,None]
    loads[loaded]-=points[loaded]*np.sum(loads[loaded]*points[loaded],axis=1)[:,None]
    # Convert physical trace length accumulated at a node into a dimensionless
    # soft-constraint weight, matching the existing boundary-loading scale.
    soft=np.minimum(weights/800.,3.)
    return loads,soft,dict(version=VERSION,enabled=True,trench_edges=int(len(edges)),
        trench_length_km=float(length[edges].sum()),
        reaction_length_km=float(reaction_length),
        max_line_force_n_per_m=float(line_force[edges].max(initial=0.)),
        integrated_slab_force_n=float(total_force_n),
        loaded_nodes=int(np.count_nonzero(loaded)),
        loaded_material_area_km2=float(areas[loaded].sum()),
        max_equivalent_speed_km_myr=float(np.linalg.norm(loads,axis=1).max(initial=0.)),
        reference_strength_n_per_m=settings['reference_strength_n_per_m'],
        mobility_km_myr=settings['mobility_km_myr'],
        reach_km=settings['reach_km'],
        boundary_motion_fraction=settings['boundary_motion_fraction'],
        moving_hinge_reaction=bool(moving),
        force_source='conserved slab line load; g * excess_mass_per_length * sin(dip)',
        compliance='explicit reduced N/m to km/Myr membrane loading; not a resolved intraplate stress tensor')
