"""Reproducible qualification metrics for long Highland 65 native runs.

This module separates hard numerical/topological qualification from softer
worldbuilding advisories. The thresholds are operational review gates, not
claims that a passing world reproduces Earth.
"""
from __future__ import annotations

from copy import deepcopy
import math
import traceback

import numpy as np


VERSION=1
DEFAULT_POLICY=dict(
    minimum_ocean_fraction=.02,
    minimum_active_plates=2,
    maximum_slab_ledger_relative_residual=1e-8,
    supercontinent_advisory_fraction=.50,
    recent_window_myr=50.,
    stagnation_advisory_cm_yr=.10,
    stalled_trench_advisory_fraction=.50,
)


def component_areas(mask,edge_faces,area):
    import mesh_geometry
    mask=np.asarray(mask,bool);area=np.asarray(area,float)
    if mask.shape!=area.shape or not np.isfinite(area).all() or np.any(area<=0.):
        raise ValueError('Qualification component areas require aligned positive finite cell areas.')
    labels,count=mesh_geometry.connected_components(mask,np.asarray(edge_faces))
    if not count:return np.empty(0,float)
    return np.sort(np.bincount(labels[mask],weights=area[mask],minlength=count))[::-1]


def _slab_ledger(s):
    import slab_memory
    if not slab_memory.enabled(s):
        return dict(area_relative_residual=0.,mass_relative_residual=0.)
    report=slab_memory.snapshot(s)['slab_memory_diagnostics']
    initial_area=float(report.get(slab_memory.INITIAL_AREA_FIELD,0.))
    fed_area=float(report.get('fed_area_km2',0.))
    retained_area=float(report.get('retained_area_km2',0.))
    retired_area=float(report.get('retired_area_km2',0.))
    area_source=initial_area+fed_area
    area_residual=area_source-retained_area-retired_area
    result=dict(area_source_km2=area_source,area_relative_residual=abs(area_residual)/max(area_source,1.))
    if slab_memory.conservative(s):
        initial_mass=float(report.get('slab_initial_excess_mass_kg',0.))
        fed_mass=float(report.get('slab_fed_excess_mass_kg',0.))
        retained_mass=float(report.get('slab_retained_excess_mass_kg',0.))
        retired_mass=float(report.get('slab_retired_excess_mass_kg',0.))
        mass_source=initial_mass+fed_mass
        mass_residual=mass_source-retained_mass-retired_mass
        result.update(mass_source_kg=mass_source,
            mass_relative_residual=abs(mass_residual)/max(mass_source,1.))
    else:
        result['mass_relative_residual']=0.
    return result


def measure(s,*,breakoff_count=0):
    """Measure one live native state without retaining its large frame arrays."""
    frame=s.snapshot()
    stats=frame['stats']
    area=np.asarray(s.cell_area,float);earth=float(area.sum())
    land=np.asarray(s.crust)>0;ocean=~land
    continents=component_areas(land,s.native_mesh['edge_faces'],area)
    oceans=component_areas(ocean,s.native_mesh['edge_faces'],area)
    plate=np.asarray(s.plate)
    mixed=0
    for p in np.unique(plate):
        cells=plate==p
        if np.any(cells&land) and np.any(cells&ocean):mixed+=1

    a,b=np.asarray(s.edge_a),np.asarray(s.edge_b)
    coast=land[a]^land[b]
    passive=coast&(plate[a]==plate[b])
    active_coast=coast&~passive
    edge_length=np.asarray(s.edge_length,float)

    boundary={name:float(stats.get(f'{name}_length_km',0.))
              for name in ('ridge','subduction','transform','collision','rift')}
    phase={}
    for row in getattr(s,'rift_systems',[]):
        key=str(row.get('phase','unknown'));phase[key]=phase.get(key,0)+1
    trench_phase={}
    for row in getattr(s,'trench_systems',[]):
        key=str(row.get('phase','unknown'));trench_phase[key]=trench_phase.get(key,0)+1
    process=getattr(s,'process_totals',{})
    slab=_slab_ledger(s)
    stalled=float(getattr(s,'subduction_polarity_diagnostics',{}).get('stalled_length_km',0.))
    return dict(
        time_myr=float(s.t),
        continental_fraction=float(area[land].sum()/earth),
        emergent_land_fraction=float(stats.get('land_fraction',0.)),
        ocean_fraction=float(area[ocean].sum()/earth),
        continent_count=int(len(continents)),
        largest_continent_fraction=float(continents[0]/earth) if len(continents) else 0.,
        ocean_basin_count=int(len(oceans)),
        largest_ocean_basin_fraction=float(oceans[0]/earth) if len(oceans) else 0.,
        active_plates=int(stats.get('active_plates',0)),
        mixed_plates=int(mixed),
        passive_margin_length_km=float(edge_length[passive].sum()),
        active_coast_length_km=float(edge_length[active_coast].sum()),
        boundary_length_km=boundary,
        total_boundary_mechanism_length_km=float(sum(boundary.values())),
        mean_plate_speed_cm_yr=float(stats.get('mean_plate_speed_cm_yr',0.)),
        max_plate_speed_cm_yr=float(stats.get('max_plate_speed_cm_yr',0.)),
        stalled_trench_length_km=stalled,
        rift_system_phases=phase,
        rift_breakthrough_count=int(process.get('rift_events_breakup',0)),
        trench_system_phases=trench_phase,
        breakoff_count=int(breakoff_count),
        ocean_created_km2=float(process.get('ocean_created_km2',0.)),
        ocean_consumed_km2=float(process.get('ocean_consumed_km2',0.)),
        accreted_km2=float(process.get('accreted_km2',0.)),
        slab_ledger=slab,
    )


def _finite(value):
    if isinstance(value,dict):return all(_finite(v) for v in value.values())
    if isinstance(value,(list,tuple)):return all(_finite(v) for v in value)
    if value is None or isinstance(value,(bool,str)):return True
    if isinstance(value,(int,np.integer)):return True
    if isinstance(value,(float,np.floating)):return math.isfinite(float(value))
    return False


def evaluate(samples,*,status='completed',requested_duration_myr=None,policy=None):
    """Hard qualification plus non-fatal worldbuilding advisories."""
    controls=dict(DEFAULT_POLICY)
    if policy is not None:
        controls.update(policy)
    policy=controls
    failures=[];advisories=[]
    if status!='completed':failures.append('run_did_not_complete')
    if not samples:
        failures.append('no_samples')
        return dict(passed=False,failures=failures,advisories=advisories,policy=policy)
    if not all(_finite(row) for row in samples):failures.append('nonfinite_metric')
    terminal=float(samples[-1]['time_myr'])
    if requested_duration_myr is not None and terminal < float(requested_duration_myr)-1e-8:
        failures.append('terminal_time_short')
    if min(row['ocean_fraction'] for row in samples)<policy['minimum_ocean_fraction']:
        failures.append('ocean_floor_lost')
    if min(row['active_plates'] for row in samples)<policy['minimum_active_plates']:
        failures.append('plate_system_collapsed')
    residual=max(max(row['slab_ledger']['area_relative_residual'],
                     row['slab_ledger']['mass_relative_residual']) for row in samples)
    if residual>policy['maximum_slab_ledger_relative_residual']:
        failures.append('slab_inventory_failed')

    if max(row['largest_continent_fraction'] for row in samples)>=policy['supercontinent_advisory_fraction']:
        advisories.append('supercontinent_dominance')
    if samples[-1]['rift_breakthrough_count']==0:
        advisories.append('no_continental_breakthrough_observed')
    if samples[-1]['ocean_created_km2']<=0.:
        advisories.append('no_new_ocean_observed')
    start=max(samples[-1]['time_myr']-policy['recent_window_myr'],0.)
    recent=[row for row in samples if row['time_myr']>=start-1e-10]
    if recent and max(row['max_plate_speed_cm_yr'] for row in recent)<policy['stagnation_advisory_cm_yr']:
        advisories.append('recent_near_stagnation')
    ratio=max((row['stalled_trench_length_km']/
               max(row['stalled_trench_length_km']+row['boundary_length_km']['subduction'],1e-30))
              for row in samples)
    if ratio>=policy['stalled_trench_advisory_fraction']:
        advisories.append('large_stalled_trench_share')
    return dict(passed=not failures,failures=failures,advisories=advisories,
                policy=policy,maximum_slab_ledger_relative_residual=float(residual))


def qualification_config(seed,*,duration_myr=500.,sample_myr=10.,dt_myr=2.,
                         width=192,mesh_level=4,coast_geometry_level=4,
                         plate_count=12,mechanics_nodes=1024):
    for name,value,maximum in (('duration_myr',duration_myr,1000.),
                               ('sample_myr',sample_myr,None),
                               ('dt_myr',dt_myr,5.)):
        if (isinstance(value,(bool,np.bool_)) or not isinstance(value,(int,float,np.number))
                or not math.isfinite(float(value)) or float(value)<=0.
                or (maximum is not None and float(value)>maximum)):
            raise ValueError(f'{name} must be finite and positive'
                             + (f' and at most {maximum:g}.' if maximum is not None else '.'))
    if isinstance(seed,(bool,np.bool_)) or not isinstance(seed,(int,np.integer)) or not 0<=int(seed)<2**32:
        raise ValueError('seed must be an unsigned 32-bit integer.')
    return dict(seed=int(seed),width=int(width),height=int(width)//2,
        duration_myr=float(duration_myr),dt_myr=float(dt_myr),
        snapshot_myr=max(float(sample_myr),float(dt_myr)),
        plate_count=int(plate_count),mechanics_nodes=int(mechanics_nodes),
        mesh_level=int(mesh_level),coast_geometry_level=int(coast_geometry_level),
        physics_profile='reviewed_v1')


class _ProgressError(RuntimeError):
    """Evidence-writer failures must not be mistaken for simulation failures."""


def run_seed(seed,*,duration_myr=500.,sample_myr=10.,dt_myr=2.,
             width=192,mesh_level=4,coast_geometry_level=4,
             plate_count=12,mechanics_nodes=1024,policy=None,progress=None):
    """Run one seed, publishing diagnostics, not restart checkpoints.

    ``progress`` receives a detached dictionary before initialization, after
    each sample and at termination. A writer failure propagates rather than
    silently continuing an expensive campaign without its evidence.
    """
    config=qualification_config(seed,duration_myr=duration_myr,sample_myr=sample_myr,
        dt_myr=dt_myr,width=width,mesh_level=mesh_level,
        coast_geometry_level=coast_geometry_level,plate_count=plate_count,
        mechanics_nodes=mechanics_nodes)
    result=dict(version=VERSION,seed=int(seed),status='running',
                requested_config=deepcopy(config),samples=[])
    samples=result['samples'];breakoffs=0;s=None

    def publish():
        if progress is not None:
            try:
                progress(deepcopy(result))
            except Exception as exc:
                raise _ProgressError('Unable to preserve qualification evidence.') from exc

    def sample():
        samples.append(measure(s,breakoff_count=breakoffs))
        publish()
        if not _finite(samples[-1]):
            raise FloatingPointError('Non-finite qualification metric; failing sample retained.')

    publish()
    try:
        import native_engine
        initial=native_engine.make_initial(config,preset='highland65')
        s=native_engine.Simulation(config,initial)
        result['realized_config']=deepcopy(s.config)
        result['last_time_myr']=float(s.t)
        if not math.isfinite(float(s.t)) or float(s.t)!=0.:
            raise ValueError('Qualification requires a fresh world at finite time zero.')
        sample()
        next_sample=min(float(sample_myr),float(duration_myr))
        while s.t<float(duration_myr)-1e-8:
            before=float(s.t)
            step=min(float(dt_myr),float(duration_myr)-before,
                     max(next_sample-before,1e-12))
            report=s.step(step)
            after=float(s.t)
            result['last_time_myr']=after
            if not math.isfinite(after) or after<=before:
                raise RuntimeError('Simulation.step did not advance finite geological time.')
            if after>before+step+1e-8:
                raise RuntimeError('Simulation.step exceeded its requested interval.')
            if isinstance(report,dict):
                breakoffs+=len(report.get('ruptures',[]))
            if s.t>=next_sample-1e-8 or s.t>=float(duration_myr)-1e-8:
                sample()
                next_sample=min(float(duration_myr),next_sample+float(sample_myr))
        result['status']='completed'
    except _ProgressError:
        raise
    except KeyboardInterrupt:
        result.update(status='interrupted',error=dict(type='KeyboardInterrupt',message='Run interrupted.'))
    except Exception as exc:
        result.update(status='failed',error=dict(type=type(exc).__name__,message=str(exc),
                                               traceback=traceback.format_exc()))
    if s is not None:
        result['realized_config']=deepcopy(s.config)
    result['qualification']=evaluate(samples,status=result['status'],
        requested_duration_myr=duration_myr,policy=policy)
    publish()
    return result
