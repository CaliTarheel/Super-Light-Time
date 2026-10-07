"""Version-2 finite-volume neck patches carried with the overriding plate.

Spherical nearest-anchor cells localize state at an explicitly supplied material
resolution. This is a finite-volume approximation, not subcell damage recovery.
Uniform old histories can only be localized as an explicit current-state baseline.
"""
from copy import deepcopy
import math
import numpy as np
import slab_tether_history as history

VERSION = 2
GEOMETRY = ('anchor_xyz', 'support_radius_km')
INITIAL_AREA = 'initial_area_km2'
LEDGER = (INITIAL_AREA, 'fed_area_km2', 'retained_area_km2', 'retained_buoyancy_area_km2',
          'retired_area_km2', 'initial_excess_mass_kg', 'fed_excess_mass_kg',
          'retained_excess_mass_kg', 'retired_excess_mass_kg')
EXTENSIVE = tuple(dict.fromkeys((*history.EXTENSIVE, *LEDGER)))
RADIUS_KM = 6371.


def enabled(row):
    return row.get(history.VERSION_FIELD) == VERSION


def validate_channel(channel):
    import slab_memory as slab
    anchor = np.asarray(channel['anchor_xyz'], float)
    radius = channel['support_radius_km']
    if (anchor.shape != (3,) or not np.isfinite(anchor).all() or
            abs(float(np.linalg.norm(anchor))-1.) > 1e-10 or
            isinstance(radius, (bool, np.bool_)) or not np.isscalar(radius) or
            not np.isfinite(radius) or not 0. < radius <= math.pi*RADIUS_KM):
        raise ValueError('Local neck patches require a unit spherical anchor and explicit finite support radius.')
    # Older version-2 channels predate inherited initial areas. Absence means
    # zero, while every newly localized channel records the label explicitly.
    record = {'slab_'+k: channel.get(k,0.) if k==INITIAL_AREA else channel[k] for k in LEDGER}
    record['length_km'] = channel['reference_width_m']/1000.
    slab.refresh_line_load(record)
    slab.validate_row(record, require_mass=True)
    for key in ('area_km2', 'excess_mass_kg'):
        if channel['detached_'+key] > channel['retired_'+key]+max(1e-7, channel['retired_'+key]*2e-11):
            raise ValueError('A local neck detached subtotal exceeds its own retirement ledger.')


def localize(row, anchors, width_fractions, *, support_radius_km):
    """Stage an explicit spatial baseline, without inventing past local damage.

    Each existing neck channel is partitioned across the supplied anchor cells.
    Existing differences in damage are preserved. Previously unlocalized retired
    material is allocated by reference width, above each channel's detached floor;
    initial/fed area and mass labels retain the row's proportions. This records a baseline,
    not a reconstruction of where historical feeding and retirement occurred.
    """
    import slab_memory as slab
    slab.validate_row(row, require_mass=True)
    if row.get(history.VERSION_FIELD) != 1:
        raise ValueError('Local neck initialization requires an explicit version-1 history.')
    points, fractions = np.asarray(anchors, float), np.asarray(width_fractions, float)
    if (points.ndim != 2 or points.shape[1] != 3 or fractions.shape != (len(points),) or not len(points)
            or not np.isfinite(points).all() or not np.isfinite(fractions).all() or np.any(fractions <= 0.)
            or np.any(np.abs(np.linalg.norm(points, axis=1)-1.) > 1e-10)
            or not math.isclose(float(fractions.sum()), 1., rel_tol=0., abs_tol=1e-12)):
        raise ValueError('Local neck baseline needs unit anchors and positive fractions summing to one.')
    old = row[history.FIELD]
    width = math.fsum(c['reference_width_m'] for c in old)
    if width <= 0.:
        raise ValueError('Local neck baseline requires positive reference width.')
    retired_area = row['slab_retired_area_km2']-math.fsum(c['detached_area_km2'] for c in old)
    retired_mass = row['slab_retired_excess_mass_kg']-math.fsum(c['detached_excess_mass_kg'] for c in old)
    total_mass = row['slab_initial_excess_mass_kg']+row['slab_fed_excess_mass_kg']
    initial_fraction = row['slab_initial_excess_mass_kg']/total_mass if total_mass > 0. else 0.
    total_area = row.get(slab.INITIAL_AREA_FIELD,0.)+row['slab_fed_area_km2']
    initial_area_fraction = row.get(slab.INITIAL_AREA_FIELD,0.)/total_area if total_area > 0. else 0.
    result = []
    for channel in old:
        base = deepcopy(channel)
        share = base['reference_width_m']/width
        base['retired_area_km2'] = base['detached_area_km2']+max(0., retired_area)*share
        source_area = base['retained_area_km2']+base['retired_area_km2']
        base[INITIAL_AREA] = source_area*initial_area_fraction
        base['fed_area_km2'] = source_area-base[INITIAL_AREA]
        base['retired_excess_mass_kg'] = base['detached_excess_mass_kg']+max(0., retired_mass)*share
        source_mass = base['retained_excess_mass_kg']+base['retired_excess_mass_kg']
        base['initial_excess_mass_kg'] = source_mass*initial_fraction
        base['fed_excess_mass_kg'] = source_mass-base['initial_excess_mass_kg']
        for point, fraction in zip(points, fractions):
            local = {k: base[k]*float(fraction) for k in EXTENSIVE}
            local.update(neck=deepcopy(base['neck']), anchor_xyz=point.tolist(), support_radius_km=support_radius_km)
            result.append(local)
    staged = dict(row, **{history.FIELD: result, history.VERSION_FIELD: VERSION})
    staged.setdefault(slab.INITIAL_AREA_FIELD,0.)
    history.validate(staged)
    row[slab.INITIAL_AREA_FIELD] = staged[slab.INITIAL_AREA_FIELD]
    row[history.FIELD] = result
    row[history.VERSION_FIELD] = VERSION


def allocation(row, points, lengths_m):
    """Allocate current edge quadrature lengths to nearest persistent patches.

    Coincident channels share a cell in proportion to their reference widths;
    their rheologies are not averaged. Failed patches remain in the partition,
    so breaking one patch does not spread neighboring slab force into its gap.
    Outside every patch's explicit support radius no force is allocated.
    """
    history.validate(row)
    if not enabled(row):
        raise ValueError('Local slab allocation requires version-2 neck patches.')
    points, lengths = np.asarray(points, float), np.asarray(lengths_m, float)
    if (points.ndim != 2 or points.shape[1] != 3 or lengths.shape != (len(points),)
            or not np.isfinite(points).all() or not np.isfinite(lengths).all() or np.any(lengths < 0.)
            or np.any(np.abs(np.linalg.norm(points, axis=1)-1.) > 1e-10)):
        raise ValueError('Neck allocation needs unit quadrature points and finite nonnegative lengths.')
    channels = row[history.FIELD]
    result = np.zeros((len(channels), len(points)))
    if not channels or not len(points):
        return result
    anchors = np.asarray([c['anchor_xyz'] for c in channels])
    radii = np.asarray([c['support_radius_km'] for c in channels])
    widths = np.asarray([c['reference_width_m'] for c in channels])
    for begin in range(0, len(points), 1024):
        end = min(begin+1024, len(points))
        cosine = np.clip(anchors@points[begin:end].T, -1., 1.)
        eligible = (cosine >= np.cos(radii[:, None]/RADIUS_KM)-2e-14) & (widths[:, None] > 0.)
        score = np.where(eligible, cosine, -np.inf)
        best = score.max(axis=0)
        tied = eligible & (score >= best[None, :]-2e-14)
        weights = tied*widths[:, None]
        total = weights.sum(axis=0)
        result[:, begin:end] = np.divide(weights, total, out=np.zeros_like(weights), where=total > 0.)*lengths[begin:end]
    return result


def transfer_fractions(row, s, edges, selected):
    """Fraction of each patch's resolved trace transferred to a daughter."""
    lengths = allocation(row, np.asarray(s.bmid)[edges], np.asarray(s.bl)[edges]*1000.)
    moved = lengths[:, selected].sum(axis=1)
    retained = lengths[:, ~selected].sum(axis=1)
    # Sum complementary nonnegative pieces together. Independently reducing
    # the full trace can round below an entirely transferred patch's subtotal,
    # making its fraction exceed one and rejecting a conservative owner split.
    # This bounds the ratio by construction; invalid supplied shares still fail
    # the partition guard, and unsupported patches remain with their owner.
    total = moved+retained
    # Unobserved inventory stays with its existing owner; no nearest-owner guess.
    return np.divide(moved, total, out=np.zeros_like(moved), where=total > 0.)


def aggregate(row):
    for key in LEDGER:
        row['slab_'+key] = math.fsum(c.get(key,0.) if key==INITIAL_AREA else c[key] for c in row[history.FIELD])


def advect(row, rotation):
    if not enabled(row):
        return
    from ridge_geometry import rotate
    for channel in row[history.FIELD]:
        channel['anchor_xyz'] = rotate(np.asarray(channel['anchor_xyz']), rotation).tolist()


def retire(row, decay):
    """Apply exact no-feed retention to each local extensive ledger."""
    for channel in row[history.FIELD]:
        for key in ('area_km2', 'excess_mass_kg'):
            old = channel['retained_'+key]
            channel['retained_'+key] = old*decay
            channel['retired_'+key] += old-channel['retained_'+key]
        channel['retained_buoyancy_area_km2'] *= decay


def validate_feed(feed):
    """Saved local accepted-water provenance closes to its containing trench."""
    entries=feed.get('neck_feeds')
    if not isinstance(entries,list):
        raise ValueError('Local neck feeding requires resolved capture provenance.')
    seen=set()
    for entry in entries:
        index=entry.get('channel_index')
        if isinstance(index,(bool,np.bool_)) or not isinstance(index,(int,np.integer)) or index<0 or index in seen:
            raise ValueError('Local neck feed indices must be unique nonnegative integers.')
        seen.add(index)
        values=[entry.get(k) for k in ('area_km2','buoyancy_area_km2','excess_mass_kg')]
        if any(isinstance(v,(bool,np.bool_)) or not isinstance(v,(int,float,np.number)) or not np.isfinite(v) or v<0 for v in values):
            raise ValueError('Local neck feeds require finite nonnegative area and mass.')
        area,buoyancy,mass=values
        if buoyancy>1.8*area+1e-7 or (area==0. and mass!=0.):
            raise ValueError('Local neck feed must have physical area for its mass and bounded buoyancy.')
    for key in ('area_km2','buoyancy_area_km2'):
        total=math.fsum(e[key] for e in entries)
        if not math.isclose(total,float(feed[key]),rel_tol=2e-11,abs_tol=1e-7):
            raise ValueError('Local neck feeding does not close to the accepted trench capture.')


def advance_source(row, feed, decay, retain_feed):
    """Exact retention with resolved, accepted source area and cooling mass."""
    if feed is None:
        feed=dict(area_km2=0.,buoyancy_area_km2=0.,neck_feeds=[])
    validate_feed(feed)
    entries={e['channel_index']:e for e in feed['neck_feeds']}
    if any(index>=len(row[history.FIELD]) for index in entries):
        raise ValueError('Accepted local slab feed references a missing neck patch.')
    for index,channel in enumerate(row[history.FIELD]):
        entry=entries.get(index,dict(area_km2=0.,buoyancy_area_km2=0.,excess_mass_kg=0.))
        if channel['neck']['damage']==1. and entry['area_km2']>0.:
            raise ValueError('New local feed cannot reattach a failed slab neck.')
        for key in ('area_km2','excess_mass_kg'):
            old,gain=channel['retained_'+key],entry[key]
            new=old*decay+gain*retain_feed
            channel['retained_'+key]=new
            channel['fed_'+key]+=gain
            channel['retired_'+key]+=old+gain-new
        channel['retained_buoyancy_area_km2']=channel['retained_buoyancy_area_km2']*decay+entry['buoyancy_area_km2']*retain_feed
    aggregate(row)
    history.validate(row)


def advance_damage(row, assembly, x, dt_myr):
    """Stage patch damage to the first rupture under frozen plate inlet speeds.

    Positive local opening is integrated BEFORE averaging within a finite-volume
    patch. Under frozen geometry/weight/drag/inlet speeds that average admits the
    same exact monotone damage integral as one neck. This is not a coupled-time
    plate solve: the caller must re-solve forces and retain the returned remainder.
    """
    from dataclasses import asdict
    import slab_tether as tether
    import plate_balance as pb
    history.validate(row)
    if not enabled(row):
        raise ValueError('Damage advancement requires explicitly localized neck histories.')
    if isinstance(dt_myr,(bool,np.bool_)) or not np.isscalar(dt_myr) or not np.isfinite(dt_myr) or dt_myr<0:
        raise ValueError('Local damage requires a finite nonnegative interval.')
    controls = []
    seen = set()
    accepted = float(dt_myr)
    for record in assembly['channels']:
        if record['trench_id'] != row['id']:
            continue
        index = record['channel_index']
        channel = row[history.FIELD][index]
        if index in seen or channel!=record['channel_state']:
            raise ValueError('Local damage must use the same channel state as its frozen force assembly.')
        seen.add(index)
        material = tether.Neck(**channel['neck'])
        width = channel['reference_width_m']
        weight, drag = record['weight_n'], record['mantle_drag_n_s_m']
        speeds = record['inlet_rows']@np.asarray(x,float)*pb.CM_YR_M_S
        positive = float(record['edge_shares']@np.maximum(weight-drag*speeds,0.))
        equivalent_inlet = (weight-positive)/drag
        args = (material,weight/width,drag/width,equivalent_inlet)
        _, event = tether.advance(*args,dt_myr)
        if event['ruptured']:
            accepted = min(accepted,event['rupture_time_myr'])
        controls.append((index,args))
    staged = deepcopy(row)
    events = []
    for index,args in controls:
        updated,event = tether.advance(*args,accepted)
        if event['ruptured']:
            history.rupture(staged,index,updated)
            events.append(index)
        else:
            staged[history.FIELD][index]['neck'] = asdict(updated)
    history.validate(staged)
    return staged,dict(advanced_dt_myr=accepted,remaining_dt_myr=float(dt_myr)-accepted,
                       ruptured_channel_indices=events,
                       integration='exact local opening average with frozen plate inlet speeds')
