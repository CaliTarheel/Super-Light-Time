"""Explicit experimental native stepping with coupled local slab rupture.

Accepted mechanical intervals advance the real native transport/source/topology
path once. Damage and rupture commit after accepted capture feeds its local slab,
before identities change. All work is staged on a private world and committed
only after the entire requested interval succeeds. Explicit registered entry
regions can supply plate/sheet work and advected hinges. Opt-in automatic
zero-work admission is available at finite inherited fronts. Ocean/stack
partition, burial and polarity remain incomplete; this is not a completed
collision model.
"""
from copy import deepcopy
from dataclasses import asdict, replace
import math
import numpy as np
import plate_balance as pb
import slab_memory as slab
import slab_tether_history as history
import slab_tether_local as local
import slab_tether_evolution as evolution
from slab_tether import Neck


def _source_limit(now, target, ceiling):
    """Consume an endpoint within roundoff of the next source interval.

    Repeatedly adding a decimal ceiling such as 0.05 Myr can leave one ulp of
    the requested duration. That residue is not a physical source interval;
    accepting it would run a full force and native source transaction for
    effectively zero time. Exceeding the ceiling by a few ulps at the final
    interval preserves its intended numerical meaning.
    """
    remaining = target-now
    limit = min(remaining, ceiling)
    if 0. < remaining-limit <= 4.*np.spacing(max(abs(target), 1.)):
        return remaining
    return limit


def _initialize_empty_histories(s,new_neck,radius):
    """Only newly empty histories can acquire a fresh supplied constitutive law."""
    for row in s.trench_systems:
        if row['phase']=='joined' or local.enabled(row):continue
        slab.validate_row(row,require_mass=True)
        if any(row[k]!=0. for k in slab.FIELDS+slab.MASS_FIELDS):
            raise ValueError('Loaded slab histories require explicit spatial localization before native rupture stepping.')
        edges=np.flatnonzero((np.asarray(s.trench_id)==row['id']) & (np.asarray(s.bl)>1e-10))
        if len(edges):
            anchors=np.asarray(s.bmid)[edges]
            lengths=np.asarray(s.bl)[edges]
        else:
            anchors=np.asarray(row['geometry_xyz'],float)
            lengths=np.ones(len(anchors))
        if not len(anchors) or row.get('length_km',0.)<=0.:
            raise ValueError('New local neck history lacks a resolved positive trace.')
        history.initialize(row,new_neck)
        local.localize(row,anchors,lengths/lengths.sum(),support_radius_km=radius)


def _prepare_capture(s):
    """Refresh current trench polarity/maturity; capture applies local attachment."""
    import trench_history
    trench_history.prepare(s)


def _commit_damage(s,mechanical):
    """After real feed/retirement, update damage and retire the CURRENT inventory."""
    endpoints={r['id']:r for r in mechanical.trench_systems}
    events=[]
    for row in s.trench_systems:
        target=endpoints.get(row['id'])
        if target is None or not local.enabled(row):continue
        if len(row[history.FIELD])!=len(target[history.FIELD]):
            raise ValueError('Trench identity changed before its mechanical event could commit.')
        for index,(current,end) in enumerate(zip(row[history.FIELD],target[history.FIELD])):
            damage=end['neck']['damage']
            if damage<current['neck']['damage']:
                raise ValueError('Native mechanical event cannot heal prior damage.')
            if damage==1. and current['neck']['damage']<1.:
                detached=history.rupture(row,index,replace(Neck(**current['neck']),damage=1.))
                events.append(dict(trench_id=int(row['id']),channel_index=index,
                                   geological_time_myr=float(s.t),**detached))
            else:
                # rupture replaces the row's records atomically; reacquire the
                # actual current record instead of mutating an earlier alias.
                row[history.FIELD][index]['neck']['damage']=damage
        slab.validate_row(row,require_mass=True)
        import trench_history
        if not trench_history._local_neck_state(row)[1]:
            import entry_regions
            detached_regions=entry_regions.mark_source_detached(s,row['id'])
            for event in events:
                if event['trench_id']==row['id']:
                    event['detached_entry_region_ids']=detached_regions
    return events


def advance(s,dt_myr,*,new_neck,support_radius_km,max_source_step_myr=.05,
            absolute_tolerance=1e-8,relative_tolerance=1e-7,max_force_solves=20000):
    """Atomically advance a native experiment, consuming the complete interval.

    Forces/damage are coupled within each accepted interval. Native geometry,
    feeding and other source terms update between intervals (operator splitting).
    max_source_step_myr controls this additional error; damage tolerance alone
    cannot certify transport/source convergence. This API is deliberately explicit
    and is not selected by ordinary Simulation.step or a live-world migration.
    """
    from native_engine import Simulation
    import world_design
    import lip_events
    if not isinstance(s,Simulation) or not pb.enabled(s) or not slab.conservative(s) or pb.resistance_version(s)!=1:
        raise ValueError('Native rupture stepping requires native instantaneous balance, passive resistance and conserved slab mass.')
    if not isinstance(new_neck,Neck) or new_neck.damage!=0.:
        raise ValueError('New trenches require an explicitly supplied undamaged neck law.')
    for name,value in (('duration',dt_myr),('source step',max_source_step_myr),('support radius',support_radius_km)):
        if isinstance(value,(bool,np.bool_)) or not np.isscalar(value) or not np.isfinite(value) or value<=0.:
            raise ValueError('Native rupture '+name+' must be finite and positive.')
    if max_source_step_myr>5. or support_radius_km>math.pi*local.RADIUS_KM:
        raise ValueError('Native rupture source step or support radius exceeds its geometric/time domain.')
    if isinstance(max_force_solves,(bool,np.bool_)) or not isinstance(max_force_solves,(int,np.integer)) or max_force_solves<1:
        raise ValueError('Native rupture force-solve budget must be positive.')
    work=deepcopy(s)
    # This explicit coupled path has represented neck damage and rupture.
    # Preserve that constitutive choice across its native sources/checkpoints:
    # a quiet or buoyant label cannot retire an attached slab by elapsed time.
    import trench_history
    trench_history._rupture_governed(work)
    work.trench_shutdown_version=1
    start=float(work.t);target=start+float(dt_myr)
    if not np.isfinite(target) or target<=start:
        raise ValueError('Native rupture cannot resolve the requested geological time interval.')
    solves=0;intervals=[];events=[]
    import entry_regions
    entry_enabled=entry_regions.enabled(work)
    admissions=[]
    while float(work.t)<target:
        work._prepare_step()
        _initialize_empty_histories(work,new_neck,support_radius_km)
        _prepare_capture(work)
        limit=_source_limit(float(work.t),target,float(max_source_step_myr))
        for transition in (world_design.next_transition(work),lip_events.next_transition(work)):
            if transition is not None and float(work.t)<transition<float(work.t)+limit:
                limit=transition-float(work.t)
        forecast=None
        if entry_regions.automatic_enabled(work) and len(work.material_surface['faces']):
            if solves>=max_force_solves:
                raise ValueError('Native automatic-entry forecast exceeds its force-solve budget; no state committed.')
            forecast_balance=pb.Balance(work,1.,slab_tethers=True)
            forecast_balance.solve();solves+=1
            forecast=forecast_balance.rotation()
        admission=entry_regions.admit_approaching(work,limit,forecast_omega=forecast)
        admissions.append(admission)
        entry_enabled=entry_regions.enabled(work)
        mechanical,report=evolution.integrate_frozen_geometry(work,limit,
            max_step_myr=limit,absolute_tolerance=absolute_tolerance,relative_tolerance=relative_tolerance,
            max_force_solves=max_force_solves-solves,stop_after_interval=True)
        solves+=report['force_solves']
        h=report['advanced_dt_myr'];epoch=float(work.t)
        if h<=0. or epoch+h==epoch:
            raise ValueError('Native rupture failed to advance resolved time; no state committed.')
        work.omega=report['rotation_integral_rad']/h
        work._boundaries();_prepare_capture(work)
        # Capture consumes the accepted average motion; local mature/failed
        # patch gates are based on the beginning of this interval. Failure is
        # committed at its endpoint, after incoming material has been booked.
        actual_events=[]
        remesh_before=(work.continental_entry_regions['remesh_energy_change_j'] if entry_enabled else 0.)
        work._advance_step(h,after_slab_source=lambda staged:actual_events.extend(_commit_damage(staged,mechanical)))
        _initialize_empty_histories(work,new_neck,support_radius_km)
        intervals.append(dict(start_myr=epoch,end_myr=float(work.t),force_solves=report['force_solves'],
                              rejected_trials=report['rejected_trials'],rupture_count=len(report['events']),
                              automatic_entry_admitted_face_ids=admission['admitted_face_ids'],
                              automatic_entry_forecast_force_solves=int(forecast is not None)))
        if entry_enabled:
            after=entry_regions.energy_j(work)
            remesh=work.continental_entry_regions['remesh_energy_change_j']-remesh_before
            mechanical_energy=work.continental_entry_regions['last_mechanical_energy_j']
            intervals[-1].update(entry_energy_after_j=after,entry_energy_after_motion_j=mechanical_energy,
                entry_remesh_energy_change_j=remesh,entry_source_energy_change_j=after-mechanical_energy-remesh)
        events.extend(actual_events)
    _prepare_capture(work)
    final_admission=entry_regions.admit_approaching(work,0.)
    entry_enabled=entry_regions.enabled(work)
    if solves>=max_force_solves:
        raise ValueError('Native rupture final force solve exceeds its budget; no state committed.')
    model=pb.Balance(work,1.,slab_tethers=True);model.solve();solves+=1
    work.omega=model.rotation()
    work.plate_balance_diagnostics=model.diagnostics(work.omega)
    work._boundaries();_prepare_capture(work)
    diagnostics=dict(version=1,start_myr=start,end_myr=float(work.t),requested_dt_myr=float(dt_myr),
        advanced_dt_myr=float(work.t)-start,source_intervals=intervals,ruptures=events,force_solves=solves,
        new_neck=asdict(new_neck),support_radius_km=float(support_radius_km),
        max_source_step_myr=float(max_source_step_myr),absolute_tolerance=float(absolute_tolerance),
        relative_tolerance=float(relative_tolerance),
        method='coupled local force/damage, native source updates between accepted intervals',
        persistent_entry_regions=entry_enabled,
        automatic_entry_admissions=admissions,
        final_zero_work_admission=final_admission,
        limitation='experimental stepping; automatic face admission is opt-in; stack partition, local burial and inherited polarity remain incomplete')
    work.slab_tether_native_diagnostics=deepcopy(diagnostics)
    s.__dict__.clear();s.__dict__.update(work.__dict__)
    return diagnostics
