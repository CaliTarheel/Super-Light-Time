"""Coupled plate-force/damage integration at fixed geometry and inventories.

Every derivative evaluation solves the actual nonlinear plate balance. Adaptive
Bogacki-Shampine 3(2) steps bound damage error; a bracketed first-crossing solve
localizes rupture and restarts mechanics with the conservatively retired patch.
All work occurs on a private state. This is not the native transport/source
timestep: geometry, thermal state, feeding and geological time remain frozen.
"""
from copy import copy, deepcopy
import math
import numpy as np
import plate_balance as pb
import slab_memory as slab
import slab_tether_history as history
import slab_tether_local as local
from slab_tether import Neck, SECONDS_PER_MYR
from dataclasses import replace


def _layout(s):
    result=[]
    identities=set()
    for row_index,row in enumerate(s.trench_systems):
        slab.validate_row(row,require_mass=True)
        if row['id'] in identities:
            raise ValueError('Coupled attachment integration needs unique trench identities.')
        identities.add(row['id'])
        if row['phase']=='joined':continue
        if not local.enabled(row):
            raise ValueError('Coupled attachment integration requires explicit local neck histories.')
        for channel_index,channel in enumerate(row[history.FIELD]):
            if channel['neck']['damage']<1. and channel['retained_area_km2']>0.:
                result.append((row_index,channel_index))
    return result


def _damage(s,layout):
    return np.array([s.trench_systems[r][history.FIELD][c]['neck']['damage'] for r,c in layout])


def _stage(s,layout,damage):
    trial=copy(s)
    trial.trench_systems=deepcopy(s.trench_systems)
    for (r,c),d in zip(layout,damage):
        # One-sided continuation used only while bracketing the event. Do not
        # retire inventory at intermediate RK stages or use post-event forces
        # to locate a pre-event trajectory. Accepted crossings retire atomically.
        trial.trench_systems[r][history.FIELD][c]['neck']['damage']=float(min(d,np.nextafter(1.,0.)))
    return trial


def _rate(model,layout):
    at={(model.s.trench_systems[r]['id'],c):i for i,(r,c) in enumerate(layout)}
    result=np.zeros(len(layout))
    for record in model.tether_assembly['channels']:
        key=(record['trench_id'],record['channel_index'])
        if key not in at:continue
        index=at[key]
        row_index,channel_index=layout[index]
        channel=model.s.trench_systems[row_index][history.FIELD][channel_index]
        speeds=record['inlet_rows']@model.x*pb.CM_YR_M_S
        hinge=record['hinge_rows']@model.x*pb.CM_YR_M_S
        mantle,neck=record['mantle_drag_n_s_m'],record['neck_drag_n_s_m']
        scale=max(mantle,neck)
        drive=np.maximum(record['weight_n']/scale-(mantle/scale)
            *(speeds+record['dip_cosine']*hinge),0.)
        resistance=mantle/scale+neck/scale
        result[index]=float(record['edge_shares']@drive)/resistance/channel['neck']['failure_opening_m']*SECONDS_PER_MYR
    if not np.isfinite(result).all() or np.any(result<0):
        raise ValueError('Coupled damage rates must be finite and irreversible.')
    return result


def integrate_frozen_geometry(s,dt_myr,*,max_step_myr=.1,absolute_tolerance=1e-8,
                              relative_tolerance=1e-7,max_force_solves=20000,
                              stop_after_interval=False,continental_entry_domain=None):
    """Return a private evolved state and a complete mechanical-time report.

    The caller's state, RNG and geological clock are unchanged, including on
    failure. Returned inventory changes are rupture transfers only. The rotation
    integral is an integral of Euler-vector components, not an advected geometry
    or a claim that finite rotations commute. No part of dt is silently dropped.
    """
    for name,value in (('duration',dt_myr),('maximum step',max_step_myr),
                       ('absolute tolerance',absolute_tolerance),('relative tolerance',relative_tolerance)):
        if isinstance(value,(bool,np.bool_)) or not np.isscalar(value) or not np.isfinite(value) or value<=0:
            raise ValueError('Coupled attachment '+name+' must be finite and positive.')
    if isinstance(max_force_solves,(bool,np.bool_)) or not isinstance(max_force_solves,(int,np.integer)) or max_force_solves<1:
        raise ValueError('Coupled attachment force-solve budget must be a positive integer.')
    if type(stop_after_interval) is not bool:
        raise ValueError('Single-interval integration must be explicitly selected.')
    if pb.resistance_version(s)!=1:
        raise ValueError('Coupled rupture requires explicitly selected passive plate resistance version 1.')
    if continental_entry_domain is not None and not isinstance(continental_entry_domain,dict):
        raise ValueError('A frozen continental-entry domain must be an explicit parameter dictionary.')
    import entry_regions
    if continental_entry_domain is not None and entry_regions.enabled(s):
        raise ValueError('Persistent entry and a second frozen entry domain cannot both load the same balance.')
    entry_domain=deepcopy(continental_entry_domain)
    state=deepcopy(s)
    _layout(state)
    elapsed=0.;next_step=min(float(max_step_myr),float(dt_myr))
    solves=0;rejected=0;accepted=0;events=[];intervals=[]
    rotation_integral=np.zeros_like(np.asarray(s.omega,float))
    maximum_residual=0.

    def evaluate(base,layout,damage):
        nonlocal solves,maximum_residual
        if solves>=max_force_solves:
            raise ValueError('Coupled attachment force-solve budget exhausted; no state committed.')
        model=pb.Balance(_stage(base,layout,damage),1.,slab_tethers=True)
        if entry_domain is not None:
            import continental_entry
            continental_entry.apply_to_balance(model,**entry_domain)
        model.solve();solves+=1
        residual=model._relative_residual(model._evaluate(model.x,pb.HUBER_CONTINUATION_KM_MYR[-1]*pb.KM_MYR_CM_YR)[1])
        maximum_residual=max(maximum_residual,residual)
        return _rate(model,layout),model.rotation(),model

    def trial(base,layout,y,k1,omega1,h):
        k2,w2,_=evaluate(base,layout,y+.5*h*k1)
        k3,w3,_=evaluate(base,layout,y+.75*h*k2)
        end=y+h*(2.*k1/9.+k2/3.+4.*k3/9.)
        k4,_,_=evaluate(base,layout,end)
        low=y+h*(7.*k1/24.+k2/4.+k3/3.+k4/8.)
        scale=absolute_tolerance+relative_tolerance*np.maximum(np.abs(y),np.abs(end))
        error=float(np.max(np.abs(end-low)/scale,initial=0.))
        integral=h*(2.*omega1/9.+w2/3.+4.*w3/9.)
        return end,error,integral

    while elapsed<float(dt_myr):
        layout=_layout(state);y=_damage(state,layout)
        k1,omega1,_=evaluate(state,layout,y)
        remaining=float(dt_myr)-elapsed
        h=min(next_step,remaining,float(max_step_myr))
        # Numerical step limiter, not a physical damage/velocity cap.
        if len(k1) and float(k1.max())>0.:h=min(h,.05/float(k1.max()))
        if elapsed+h==elapsed or h<=0.:
            raise ValueError('Coupled attachment cannot resolve remaining time; no state committed.')
        end,error,integral=trial(state,layout,y,k1,omega1,h)
        if error>1.:
            rejected+=1;next_step=h*max(.1,.8*error**(-1./3.))
            continue
        ruptured=[]
        if np.any(end>=1.):
            low,high=0.,h
            high_result=(end,error,integral)
            # Find the first crossing of ANY currently attached patch. Each
            # probe re-solves plate forces at all RK stages on the pre-event side.
            time_tolerance=max(16.*np.spacing(max(elapsed+h,1.)),
                               .05*absolute_tolerance/max(float(k1.max()),1e-300))
            for _ in range(64):
                if high-low<=time_tolerance:break
                middle=.5*(low+high)
                candidate=trial(state,layout,y,k1,omega1,middle)
                if np.any(candidate[0]>=1.):high=middle;high_result=candidate
                else:low=middle
            else:
                raise ValueError('Coupled rupture event did not localize; no state committed.')
            h=high;end,error,integral=high_result
            if error>1.:
                rejected+=1;next_step=h*max(.1,.8*error**(-1./3.))
                continue
            ruptured=np.flatnonzero(end>=1.).tolist()
        start=elapsed
        for index,((r,c),d) in enumerate(zip(layout,end)):
            channel=state.trench_systems[r][history.FIELD][c]
            if index in ruptured:
                detached=history.rupture(state.trench_systems[r],c,replace(Neck(**channel['neck']),damage=1.))
                events.append(dict(time_myr=elapsed+h,trench_id=int(state.trench_systems[r]['id']),
                                   channel_index=int(c),**detached))
            else:
                channel['neck']['damage']=float(d)
        # Use the exact remaining interval when accepted; accumulated roundoff
        # must not create an unresolvable sliver or silently lose an interval.
        elapsed=float(dt_myr) if h==remaining else elapsed+h
        rotation_integral+=integral;accepted+=1
        intervals.append(dict(start_myr=start,end_myr=elapsed,damage_error_ratio=error,
                              rupture_count=len(ruptured)))
        next_step=min(float(max_step_myr),h*(2. if error==0. else min(2.,max(.2,.9*error**(-1./3.)))))
        if stop_after_interval:break
    layout=_layout(state)
    _,omega,final_model=evaluate(state,layout,_damage(state,layout))
    state.omega=omega
    report=dict(requested_dt_myr=float(dt_myr),advanced_dt_myr=elapsed,remaining_dt_myr=float(dt_myr)-elapsed,
        accepted_intervals=accepted,rejected_trials=rejected,force_solves=solves,
        maximum_scaled_force_residual=maximum_residual,events=events,intervals=intervals,
        rotation_integral_rad=rotation_integral,final_damage_rate_per_myr=_rate(final_model,layout),
        integration='adaptive coupled force/damage at frozen geometry and source inventories',
        continental_entry_domain_supplied=entry_domain is not None,
        persistent_entry_regions=entry_regions.enabled(s),
        frozen_continental_entry_energy_j=final_model.notes.get('continental_entry_energy_j',0.),
        geological_time_advanced=False,transport_advanced=False)
    return state,report
