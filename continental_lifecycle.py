"""Versioned ordinary continental-entry and slab-breakoff lifecycle.

This policy promotes the already-tested coupled slab-tether/native-entry path
into ordinary reviewed-physics stepping. It does not add a second force law.
The constitutive controls are explicit starting-world/run configuration and are
recorded because breakoff timing is not calibrated by this reduced model.
"""
from __future__ import annotations

from copy import deepcopy
import math
import numpy as np

from slab_tether import Neck

VERSION = 1
DEFAULTS = dict(version=VERSION, enabled=False,
    neck_viscosity_pa_s=1e23, neck_thickness_km=80., neck_length_km=100.,
    failure_opening_km=80., support_radius_km=1500.,
    max_source_step_myr=.25, replacement_delay_myr=20.,
    replacement_exclusion_km=600.)


def normalize(value=None):
    if value is None:
        return dict(DEFAULTS)
    if not isinstance(value, dict) or set(value)-set(DEFAULTS):
        raise ValueError('Continental lifecycle settings contain unknown fields.')
    result=dict(DEFAULTS,**value)
    if type(result['version']) is not int or result['version']!=VERSION or type(result['enabled']) is not bool:
        raise ValueError('Continental lifecycle requires version 1 and a boolean enabled flag.')
    ranges=(
        ('neck_viscosity_pa_s',1e20,1e25),('neck_thickness_km',10.,200.),
        ('neck_length_km',10.,500.),('failure_opening_km',5.,500.),
        ('support_radius_km',100.,10000.),('max_source_step_myr',.001,1.),
        ('replacement_delay_myr',0.,200.),('replacement_exclusion_km',0.,3000.))
    for key,low,high in ranges:
        value=result[key]
        if isinstance(value,(bool,np.bool_)) or not isinstance(value,(int,float,np.number)) or not np.isfinite(value) or not low<=float(value)<=high:
            raise ValueError(f'Continental lifecycle {key} must be finite and between {low} and {high}.')
        result[key]=float(value)
    return result


def enabled(s):
    return getattr(s,'continental_lifecycle_version',0)==VERSION


def neck(settings):
    settings=normalize(settings)
    return Neck(viscosity_pa_s=settings['neck_viscosity_pa_s'],
                thickness_m=settings['neck_thickness_km']*1000.,
                length_m=settings['neck_length_km']*1000.,
                failure_opening_m=settings['failure_opening_km']*1000.)


def _localize_existing_trenches(s,settings):
    """Install local neck histories on inherited loaded slabs without inventing mass."""
    import slab_memory
    import slab_tether_history as history
    import slab_tether_local as local
    import trench_history

    trench_history.prepare(s)
    material=neck(settings)
    for row in s.trench_systems:
        if row['phase'] in ('shutdown','joined') or local.enabled(row):
            continue
        slab_memory.validate_row(row,require_mass=True)
        if history.FIELD not in row:
            history.initialize(row,material)
        elif row.get(history.VERSION_FIELD)!=1:
            raise ValueError('Continental lifecycle cannot reinterpret an existing non-v1 slab neck history.')
        edges=np.flatnonzero((np.asarray(s.trench_id)==row['id']) & (np.asarray(s.bl)>1e-10))
        if len(edges):
            anchors=np.asarray(s.bmid)[edges]
            lengths=np.asarray(s.bl)[edges]
        else:
            anchors=np.asarray(row['geometry_xyz'],float)
            lengths=np.ones(len(anchors))
        if not len(anchors):
            raise ValueError('Continental lifecycle needs resolved trench geometry for inherited slab localization.')
        fractions=lengths/lengths.sum()
        local.localize(row,anchors,fractions,support_radius_km=settings['support_radius_km'])


def initialize(s):
    settings=normalize(s.config.get('continental_lifecycle'))
    if not settings['enabled']:
        return dict(enabled=False,version=0)
    if getattr(s,'physics_profile_version',0)!=1:
        raise ValueError('Continental lifecycle requires reviewed_v1 physics.')
    if float(s.t)!=0. or int(s.steps)!=0:
        raise ValueError('Continental lifecycle is a fresh-world policy, not a checkpoint migration.')
    if getattr(s,'continental_lifecycle_version',0)==VERSION:
        return deepcopy(s.continental_lifecycle_diagnostics)
    s.continental_lifecycle_version=VERSION
    s.automatic_entry_version=1
    s.trench_shutdown_version=1
    s.replacement_subduction_version=1
    s.continental_lifecycle_settings=settings
    _localize_existing_trenches(s,settings)
    s.continental_lifecycle_diagnostics=dict(version=VERSION,enabled=True,
        initialized_myr=0.,constitutive=dict(settings),
        mechanics='Existing coupled slab-tether force/damage, automatic zero-work continental entry and conservative native sources.',
        breakoff='Local neck damage reaches one; represented retained slab inventory retires through the existing rupture transaction.',
        replacement='Independent convergent oceanic contacts may initiate after the configured delay/exclusion test; the detached trace does not reverse immediately.',
        limitation='Reduced neck rheology and finite-front entry; active entry/stack overlap and complete collision-channel rheology remain unresolved.')
    if hasattr(s,'_record'):
        s._record('continental_lifecycle','Reviewed world enables coupled continental entry and slab breakoff.',
                  details=deepcopy(s.continental_lifecycle_diagnostics))
    return deepcopy(s.continental_lifecycle_diagnostics)


def replacement_allowed(s,edges):
    """Mask fresh convergent edges that are independent of recent breakoff traces."""
    edges=np.asarray(edges,int)
    if not len(edges):
        return np.zeros(0,bool)
    if getattr(s,'replacement_subduction_version',0)!=1:
        return np.zeros(len(edges),bool)
    settings=getattr(s,'continental_lifecycle_settings',normalize(None))
    allowed=np.ones(len(edges),bool)
    now=float(s.t)
    import trench_history
    detached=[row for row in getattr(s,'trench_systems',[])
              if row.get('phase')=='shutdown'
              and row.get('episodes',[{}])[-1].get('shutdown_reason')=='slab_necks_detached']
    if not detached:
        return np.zeros(len(edges),bool)
    for row in detached:
        if row['phase']!='shutdown':
            continue
        reason=row.get('episodes',[{}])[-1].get('shutdown_reason')
        if reason!='slab_necks_detached':
            continue
        ended=float(row.get('episodes',[{}])[-1].get('end_myr',row.get('last_seen_myr',now)))
        if now-ended>=settings['replacement_delay_myr']:
            continue
        distance=trench_history._nearest(np.asarray(s.bmid)[edges],np.asarray(row['geometry_xyz'],float))
        allowed &= distance>settings['replacement_exclusion_km']
    return allowed


def advance(s,dt):
    if not enabled(s):
        raise ValueError('Continental lifecycle is not enabled.')
    settings=s.continental_lifecycle_settings
    import slab_tether_native
    report=slab_tether_native.advance(s,float(dt),new_neck=neck(settings),
        support_radius_km=settings['support_radius_km'],
        max_source_step_myr=settings['max_source_step_myr'],
        absolute_tolerance=1e-7,relative_tolerance=1e-6)
    s.continental_lifecycle_last_step=deepcopy(report)
    return report


def snapshot(s):
    if not enabled(s):
        return {}
    return dict(continental_lifecycle_version=VERSION,
        continental_lifecycle_diagnostics=deepcopy(getattr(s,'continental_lifecycle_diagnostics',{})),
        continental_lifecycle_last_step=deepcopy(getattr(s,'continental_lifecycle_last_step',None)))
