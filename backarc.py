"""Subduction-linked back-arc loading and finite arc-plate separation.

This is a kinematic worldbuilding closure, not a slab or mantle-flow solver.
A persistent local load can split a coherent arc strip; subsequent real
boundary motion determines whether its basin spreads, pauses, or closes. No
permanent mantle kick is used.

The loading driver is versioned (``backarc_driver_version``):

0  Swept trench retreat relative to the overriding plate, the forearc sweep's
   ``trench_retreat_speed``. Saved worlds without the field keep this law.
1  Sea anchor, for moving-hinge worlds (``subduction_response_version`` 1),
   whose rigid-carrier hinge has no forearc sweep and whose
   ``trench_retreat_speed`` is identically zero. The force balance's slab
   anchor (plate_balance SLAB_ANCHOR_PA_S) resists the overriding plate's
   own trench-normal velocity in the stationary-mantle frame. Where the solved
   upper plate moves away from a slab that reaches below its lithosphere, the
   anchored slab would leave the hinge behind at exactly that speed; the rigid
   carrier suppresses the lag, and a deformable back-arc would take it up as
   extension (Uyeda & Kanamori 1979; Heuret & Lallemand 2005). That speed,
   in km/Myr, replaces the swept retreat as the loading rate. Nothing else in
   the loading gate changes, and trench_retreat_speed stays zero so arc supply
   and the force-balance guard keep their meaning.
2  Effective-reservoir loading, for Lite worlds. A declared force-bearing
   trench with actual oceanic convergence supplies the unresolved reservoir;
   the overriding plate's solved trench-normal motion away from it supplies
   the kinematic loading rate. No slab mass, depth or neck gate is inferred,
   and this law neither moves the trench nor drives the upper plate.
"""
from __future__ import annotations

import math
from copy import deepcopy
import numpy as np
import trench_history
import normal_partition
import slab_memory
import effective_subduction

from trench_dynamics import retreat_speed
from backarc_geometry import choose_arc_sliver

RADIUS_KM = 6371.
SECTOR_RADIUS_KM = 2200.
LOADING_MEMORY_MYR = 80.
RUPTURE_LOADING_KM = 60.
QUIET_DELAY_MYR = 10.
# Existing admission policy for a loading edge and its sector (km/Myr).
LOADING_EDGE_KM_MYR = .4
DRIVER_VERSION = 1
EFFECTIVE_DRIVER_VERSION = 2
DRIVER_FRAME = 'stationary mantle (plate_balance absolute frame)'
DRIVER_LAW = 'overriding plate trench-normal velocity away from a slab anchored below its lithosphere'
DRIVER_DIAGNOSTIC_NUMBERS = ('anchored_edges', 'anchored_length_km', 'loading_length_km',
                             'max_speed_km_myr', 'mean_speed_km_myr')
EFFECTIVE_DRIVER_LAW = 'overriding plate trench-normal velocity away from a declared effective reservoir during actual oceanic convergence'
EFFECTIVE_DRIVER_DIAGNOSTIC_NUMBERS = ('force_bearing_edges', 'force_bearing_length_km',
                                      'loading_length_km', 'max_speed_km_myr', 'mean_speed_km_myr')
_MECHANISM = {
    0: 'relative trench retreat, coherent arc separation and measured boundary motion',
    1: ('overriding plate moving away from its anchored slab in the mantle frame, '
        'coherent arc separation and measured boundary motion'),
    2: ('overriding plate moving away from a declared effective reservoir in the mantle frame, '
        'coherent arc separation and measured boundary motion'),
}


def driver_version(s):
    """Missing state keeps the recorded swept-retreat law."""
    version = getattr(s, 'backarc_driver_version', 0)
    if (isinstance(version, (bool, np.bool_))
            or not isinstance(version, (int, np.integer))
            or version not in (0, DRIVER_VERSION, EFFECTIVE_DRIVER_VERSION)):
        raise ValueError('Unsupported back-arc loading driver version.')
    return int(version)


def overriding_thickness_km(s, cells):
    """Overriding lithosphere thickness from the balance's own cooling law.

    Oceanic columns use plate_balance.plate_thickness_m with its cap, the same
    thickness the hinge-bending term uses. Continental, cratonic and arc
    columns all take the cap: the model carries no separate continental
    geotherm. Real cratonic lithosphere is about 200-250 km thick and arc
    lithosphere is hot and thin, so this anchors cratonic upper plates too
    easily and arc-crust upper plates too hard (PHYSICS_REPAIRS, sea anchor,
    "Not addressed").
    """
    import plate_balance
    cells = np.asarray(cells, int)
    cap = plate_balance.PLATE_THICKNESS_CAP_KM
    age = np.asarray(s.age, float)[cells]
    ocean = np.asarray(s.crust)[cells] == 0
    if not np.all(np.isfinite(age[ocean])):
        # A NaN thickness would silently read as "not anchored".
        raise ValueError('Oceanic overriding cells need finite ages for the sea-anchor depth test.')
    oceanic = plate_balance.plate_thickness_m(np.where(ocean, age, 0.), cap)/1e3
    return np.where(ocean, oceanic, float(cap))


def loading_speed(s):
    """Per boundary edge back-arc loading speed, km/Myr, under the saved driver.

    Drivers 1 and 2 also store ``s.backarc_driver_diagnostics``.
    """
    if driver_version(s) == 0:
        return np.asarray(s.trench_retreat_speed, float)
    if driver_version(s) == EFFECTIVE_DRIVER_VERSION:
        return _effective_loading_speed(s)
    import plate_balance
    if plate_balance.subduction_response_version(s) != 1:
        # A fixed trench already supplies swept retreat; both would double count.
        raise ValueError('The sea-anchor back-arc driver requires the moving-hinge subduction response.')
    speed = np.zeros(len(s.ba))
    # Attachment and slab length come from slab_memory.line_load, the per-edge
    # attachment the reviewed force balance uses. The experimental slab-tether
    # path gives the balance's anchor tether owners and extents instead; this
    # depth test is not reconciled with those (no saved world enables tethers).
    owners, load, length = slab_memory.line_load(s)
    down = np.asarray(s.down, int)
    bp, bq = np.asarray(s.bp, int), np.asarray(s.bq, int)
    edges = np.flatnonzero(normal_partition.subduction(s) & (owners >= 0) & (owners == down) & (load > 0.))
    anchored = np.zeros(0, int)
    if len(edges):
        over = np.where(bp[edges] == down[edges], bq[edges], bp[edges])
        above = np.where(bp[edges] == over, np.asarray(s.ba)[edges], np.asarray(s.bb)[edges])
        normal = np.asarray(s.bn, float)[edges]
        # Unit trench normal from the downgoing plate into the overriding plate
        # (bn points from bp to bq).
        inland = np.where((bq[edges] == over)[:, None], normal, -normal)
        velocity = np.cross(np.asarray(s.omega, float)[over], np.asarray(s.bmid, float)[edges])*RADIUS_KM
        away = np.sum(velocity*inland, axis=1)
        # Mantle resistance acts only on slab that reaches below the overriding
        # lithosphere; a shallower slab is not yet a sea anchor. This depth
        # condition belongs to the back-arc driver only: plate_balance applies
        # its anchor to every attached slab, scaled by slab_length/862 km. Past
        # the condition the full kinematic lag counts, whatever that scale.
        depth = length[edges]*math.sin(math.radians(slab_memory.SUBDUCTION_DIP_DEG))
        hold = depth > overriding_thickness_km(s, above)
        anchored = edges[hold]
        speed[anchored] = np.maximum(away[hold], 0.)
    bl = np.asarray(s.bl, float)
    anchored_length = float(bl[anchored].sum())
    s.backarc_driver_diagnostics = dict(version=DRIVER_VERSION,
        anchored_edges=int(len(anchored)), anchored_length_km=anchored_length,
        loading_length_km=float(bl[speed > LOADING_EDGE_KM_MYR].sum()),
        max_speed_km_myr=float(speed.max(initial=0.)),
        mean_speed_km_myr=(float(np.sum(speed[anchored]*bl[anchored])/anchored_length)
                           if anchored_length > 0. else 0.),
        frame=DRIVER_FRAME, law=DRIVER_LAW)
    return speed


def _effective_loading_speed(s):
    """Use solved upper-plate motion without inventing a resolved sea anchor."""
    import plate_balance
    if not effective_subduction.enabled(s) or plate_balance.subduction_response_version(s) != 0:
        raise ValueError('The effective-reservoir back-arc driver requires effective fixed-trench subduction.')
    owners, force, live = effective_subduction.line_state(s)
    down = np.asarray(s.down, int)
    bp, bq = np.asarray(s.bp, int), np.asarray(s.bq, int)
    speed = np.zeros(len(s.ba))
    edges = np.flatnonzero(normal_partition.subduction(s) & live
                          & (owners == down) & (force > 0.))
    if len(edges):
        normal = np.asarray(s.bn, float)[edges]
        midpoint = np.asarray(s.bmid, float)[edges]
        relative = np.cross(np.asarray(s.omega, float)[bq[edges]]
                            - np.asarray(s.omega, float)[bp[edges]], midpoint)*RADIUS_KM
        closing = np.sum(relative*normal, axis=1) < 0.
        # Cached colors or normal speeds cannot create a converging mechanism.
        closing &= ((np.asarray(s.plate)[np.asarray(s.ba)[edges]] == bp[edges])
                    & (np.asarray(s.plate)[np.asarray(s.bb)[edges]] == bq[edges]))
        edges, normal, midpoint = edges[closing], normal[closing], midpoint[closing]
        over = np.where(bp[edges] == down[edges], bq[edges], bp[edges])
        inland = np.where((bq[edges] == over)[:, None], normal, -normal)
        velocity = np.cross(np.asarray(s.omega, float)[over], midpoint)*RADIUS_KM
        speed[edges] = np.maximum(np.sum(velocity*inland, axis=1), 0.)
    bl = np.asarray(s.bl, float)
    force_bearing_length = float(bl[edges].sum())
    s.backarc_driver_diagnostics = dict(version=EFFECTIVE_DRIVER_VERSION,
        force_bearing_edges=int(len(edges)), force_bearing_length_km=force_bearing_length,
        loading_length_km=float(bl[speed > LOADING_EDGE_KM_MYR].sum()),
        max_speed_km_myr=float(speed.max(initial=0.)),
        mean_speed_km_myr=(float(np.sum(speed[edges]*bl[edges])/force_bearing_length)
                           if force_bearing_length > 0. else 0.),
        frame=DRIVER_FRAME, law=EFFECTIVE_DRIVER_LAW)
    return speed


def snapshot_fields(s):
    if driver_version(s) == 0:
        return {}
    if not isinstance(getattr(s, 'backarc_driver_diagnostics', None), dict):
        loading_speed(s)
    return dict(backarc_driver_version=driver_version(s),
                backarc_driver_diagnostics=deepcopy(s.backarc_driver_diagnostics))


def validate_frame(frame):
    version = frame.get('backarc_driver_version', 0)
    if type(version) is not int or version not in (0, DRIVER_VERSION, EFFECTIVE_DRIVER_VERSION):
        raise ValueError('Unsupported back-arc loading driver version.')
    if version == 0:
        if 'backarc_driver_diagnostics' in frame:
            raise ValueError('Back-arc driver diagnostics require their driver version.')
        return
    if version == DRIVER_VERSION:
        if frame.get('subduction_response_version') != 1:
            raise ValueError('The sea-anchor back-arc driver requires the moving-hinge subduction response.')
        law, numbers = DRIVER_LAW, DRIVER_DIAGNOSTIC_NUMBERS
        count, length = 'anchored_edges', 'anchored_length_km'
    else:
        if (type(frame.get('effective_subduction_version')) is not int
                or frame['effective_subduction_version'] != effective_subduction.VERSION
                or type(frame.get('subduction_response_version')) is not int
                or frame['subduction_response_version'] != 0):
            raise ValueError('The effective-reservoir back-arc driver requires effective fixed-trench subduction.')
        law, numbers = EFFECTIVE_DRIVER_LAW, EFFECTIVE_DRIVER_DIAGNOSTIC_NUMBERS
        count, length = 'force_bearing_edges', 'force_bearing_length_km'
    row = frame.get('backarc_driver_diagnostics')
    if (not isinstance(row, dict) or type(row.get('version')) is not int or row['version'] != version
            or row.get('frame') != DRIVER_FRAME or row.get('law') != law):
        raise ValueError('Back-arc driver diagnostics must declare their versioned loading law.')
    for name in numbers:
        value = row.get(name)
        if type(value) not in (int, float) or not math.isfinite(value) or value < 0:
            raise ValueError('Back-arc driver diagnostics must be finite and nonnegative.')
    if (type(row[count]) is not int
            or row['loading_length_km'] > row[length]*(1+1e-12)+1e-9
            or row['mean_speed_km_myr'] > row['max_speed_km_myr']*(1+1e-12)+1e-9):
        raise ValueError('Back-arc driver diagnostics are inconsistent.')


def _unit(v):
    return np.asarray(v, float)/max(float(np.linalg.norm(v)), 1e-30)


def slots(s):
    return {int(s.plate_uid[p]): int(p) for p in np.flatnonzero(s.active)}


def _near(s, edges, center, radius=SECTOR_RADIUS_KM):
    return edges[s.bmid[edges] @ np.asarray(center) > math.cos(radius/RADIUS_KM)]


def trench_edges(s, down, over):
    """Require a current oceanic downgoing side and actual normal convergence."""
    edges = np.flatnonzero(normal_partition.subduction(s) & (s.down == down)
                          & ((s.bp == over) | (s.bq == over)))
    if not len(edges):
        return edges
    below = np.where(s.bp[edges] == down, s.ba[edges], s.bb[edges])
    dv = np.cross(s.omega[s.bq[edges]]-s.omega[s.bp[edges]], s.bmid[edges])*RADIUS_KM
    normal = np.sum(dv*s.bn[edges], axis=1)
    shear = np.linalg.norm(dv-normal[:, None]*s.bn[edges], axis=1)
    closing = (normal < 0. if normal_partition.enabled(s) else normal < -np.maximum(2., .35*shear))
    valid = ((s.crust[below] == 0) & closing
             & (s.plate[s.ba[edges]] == s.bp[edges])
             & (s.plate[s.bb[edges]] == s.bq[edges]))
    if hasattr(s, 'trench_systems'):
        valid &= trench_history.weights(s)[edges] > 0
    return edges[valid]


def _frame(s, edges, down):
    length = s.bl[edges]
    center = _unit(np.sum(s.bmid[edges]*length[:, None], axis=0))
    normals = np.where((s.bp[edges] == down)[:, None], s.bn[edges], -s.bn[edges])
    direction = np.sum(normals*length[:, None], axis=0)
    direction = _unit(direction-center*np.dot(center, direction))
    return center, direction


def _record(s, row, phase, description, owners):
    row['phase'] = phase
    row['phase_serial'] = row.get('phase_serial', 0)+1
    s._record('backarc_'+phase, description,
              ('backarc', row['id'], row['phase_serial']), plates=owners,
              xyz=row['center'], details={'backarc_id': row['id'],
                  'parent_plate_uid': row['parent_plate_uid'],
                  'arc_plate_uid': row['arc_plate_uid'],
                  'downgoing_plate_uid': row['downgoing_plate_uid'],
                  'opening_km': row['opening_km'], 'loading_km': row['loading_km'],
                  'mechanism': _MECHANISM[driver_version(s)]})


def driven_pairs(s):
    """Arc contacts using plate motion must not also receive forearc remapping."""
    if not getattr(s, 'backarc_basins', []):
        return set()
    current = slots(s)
    return {(current[r['downgoing_plate_uid']], current[r['arc_plate_uid']])
            for r in s.backarc_basins if r['phase'] != 'closed'
            and r['downgoing_plate_uid'] in current and r['arc_plate_uid'] in current}


def apply_motion(s, dt):
    """A bounded, transient hinge-retreat correction after ordinary forces.

    The parent and arc receive equal area-weighted opposing angular changes.
    Existing faster extension is retained; strong convergence is not simply
    overwritten. A vanished/reversed trench supplies no correction.
    """
    if driver_version(s) == EFFECTIVE_DRIVER_VERSION:
        # Lite records loading from solved motion; it supplies no separate
        # retreat correction, even if called outside the reviewed engine.
        return
    current = slots(s)
    area = np.bincount(s.plate, weights=s.cell_area, minlength=s.capacity)
    for row in s.backarc_basins:
        if row['phase'] == 'closed' or row['arc_plate_uid'] is None:
            continue
        if any(row[k] not in current for k in ('parent_plate_uid', 'arc_plate_uid', 'downgoing_plate_uid')):
            continue
        parent, arc, down = (current[row[k]] for k in ('parent_plate_uid', 'arc_plate_uid', 'downgoing_plate_uid'))
        edges = _near(s, trench_edges(s, down, arc), row['center'], 3000.)
        if hasattr(s, 'trench_systems') and row.get('trench_id') is not None:
            edges = edges[s.trench_id[edges] == row['trench_id']]
        if not len(edges) or min(area[parent], area[arc]) <= 0:
            continue
        center, direction = _frame(s, edges, down)
        below = np.where(s.bp[edges] == down, s.ba[edges], s.bb[edges])
        craton = float(s.mass[(s.parcel_plate == arc) & (s.kind == 2)].sum()/max(area[arc], 1.))
        velocity = np.cross(s.omega[s.bq[edges]]-s.omega[s.bp[edges]], s.bmid[edges])*RADIUS_KM
        convergence = -np.sum(velocity*s.bn[edges], axis=1)
        retained=trench_history.slab_memory.enabled(s)
        rollback = float(np.average(retreat_speed(convergence, s.age[below], craton,
            slab_buoyancy=trench_history.slab_pull_weights(s)[edges] if retained else None)
            * (1. if retained else trench_history.weights(s)[edges]), weights=s.bl[edges]))
        relative = float(np.dot(np.cross(s.omega[arc]-s.omega[parent], center)*RADIUS_KM, direction))
        change = min(rollback, max(0., relative+rollback))*(1.-math.exp(-dt/12.))
        change *= min(1., s.config['rift_strength'])
        torque = np.cross(center, -direction)*(change/RADIUS_KM)
        fraction = area[arc]/(area[parent]+area[arc])
        s.omega[arc] += torque*(1-fraction)
        s.omega[parent] -= torque*fraction


def advect_records(s, dt, rotate):
    current = slots(s)
    for row in s.backarc_basins:
        if row['parent_plate_uid'] in current:
            p = current[row['parent_plate_uid']]
            row['center'] = _unit(rotate(np.asarray(row['center']), s.omega[p]*dt)).tolist()


def _commit(s, row, edges, parent, down):
    choice = choose_arc_sliver(s, edges, seed=s.config['seed']+row['id']*7919)
    if choice is None:
        return False
    free = np.flatnonzero(~s.active)
    if not len(free):
        s._ensure_plate_capacity(s.capacity+1)
        free = np.flatnonzero(~s.active)
    if not len(free):
        return False
    arc = int(free[0])
    region = choice['region']
    parcel_side = (s.parcel_plate == parent) & choice['parcel_side']
    trace_side = (s.trace_plate == parent) & choice['trace_side']
    s.active[arc] = True
    s.count = max(s.count, arc+1)
    s._new_plate_identity(arc, parent)
    s.names[arc] = f"Arc plate {row['id']:02d}"
    s.polarity[arc, :] = s.polarity[:, arc] = -1
    s.polarity[arc, down] = s.polarity[down, arc] = down
    s.collision_clock[arc, :] = s.collision_clock[:, arc] = 0
    s.support[arc] = 0
    s.support[arc, region] = s.support[parent, region]
    s.support[parent, region] = 0
    s.plate[region] = arc
    inherited = trench_history.transfer_overriding(s, parent, arc)
    if row.get('trench_id') in inherited:
        row['trench_id'] = inherited[row['trench_id']]
    s.parcel_plate[parcel_side] = arc
    s.trace_plate[trace_side] = arc
    s._transfer_ridge_hosts(parent, arc)
    s.mantle[arc] = s.mantle[parent].copy()
    # Only the currently evidenced retreat initializes relative motion.
    center, direction = np.asarray(choice['center']), np.asarray(choice['direction'])
    opening = np.cross(center, -direction)*min(8., row['loading_rate_km_myr'])/RADIUS_KM
    fraction = choice['area_fraction']
    old = s.omega[parent].copy()
    s.omega[parent] = old-opening*fraction
    s.omega[arc] = old+opening*(1-fraction)
    s.centres[arc] = _unit(np.sum(s.xyz[region]*s.cell_area[region, None], axis=0))
    s.born[arc] = s.t
    s.rift_clock[arc] = s.ocean_rift_clock[arc] = 0
    row.update(arc_plate_uid=int(s.plate_uid[arc]), rupture_myr=float(s.t),
               center=center.tolist(), last_active_myr=float(s.t),
               opening_km=0., closing_km=0., loading_rate_km_myr=0.)
    s.process_totals['backarc_ruptures'] += 1
    # rift_events is incremented by BOTH this path and continental/oceanic breakup in
    # native_topology, so a nonzero total never meant a plate had split. Keep the total
    # for continuity with saved history and record the source alongside it.
    s.process_totals['rift_events'] += 1
    s.process_totals['rift_events_backarc'] = s.process_totals.get('rift_events_backarc', 0.)+1
    cause = ('Sustained retreat' if driver_version(s) == 0 else
             'Sustained motion of the overriding plate away from its anchored slab'
             if driver_version(s) == DRIVER_VERSION else
             'Sustained motion of the overriding plate away from its declared effective reservoir')
    _record(s, row, 'rifting',
            f"{cause} fractured the plate behind its volcanic arc, separating {s.names[arc]} from {s.names[parent]}; subsequent relative motion controls basin opening.",
            (parent, arc, down))
    return True


def update(s, dt):
    """Update recorded phases and prepare at most one coherent rupture per step.

    Called after actual trench migration, before deformation. New ruptures are
    committed separately after deformation, so cached boundary owners stay valid.
    """
    current = slots(s)
    pending = None
    for row in s.backarc_basins:
        if row['arc_plate_uid'] is None or row['phase'] == 'closed':
            continue
        if row['parent_plate_uid'] not in current or row['arc_plate_uid'] not in current:
            row['extension_rate_km_myr'] = 0.
            _record(s, row, 'closed', f"Back-arc basin {row['id']} lost its separate plate boundary during plate incorporation.", ())
            continue
        parent, arc = current[row['parent_plate_uid']], current[row['arc_plate_uid']]
        contact = np.flatnonzero(((s.bp == parent) & (s.bq == arc)) | ((s.bp == arc) & (s.bq == parent)))
        contact = _near(s, contact, row['center'], 3000.)
        rate, shortening, ocean_length = 0., 0., 0.
        if len(contact):
            length = s.bl[contact]
            ext = np.isin(s.bcode[contact], [1, 5])
            con = normal_partition.convergence(s)[contact]
            rate = float(np.sum(np.maximum(s.normal_speed[contact], 0)*ext*length)/max(length.sum(), 1.))
            shortening = float(np.sum(np.maximum(-s.normal_speed[contact], 0)*con*length)/max(length.sum(), 1.))
            ocean_length = float(length[s.bcode[contact] == 1].sum())
            row['center'] = _unit(np.sum(s.bmid[contact]*length[:, None], axis=0)).tolist()
        row['extension_rate_km_myr'] = rate
        row['opening_km'] += rate*dt
        row['closing_km'] += shortening*dt
        if rate > 1.:
            row['last_active_myr'] = float(s.t)
            phase = 'spreading' if ocean_length >= 300. else 'rifting'
            if row['phase'] != phase:
                _record(s, row, phase,
                        f"Back-arc basin {row['id']} {'has an active oceanic spreading boundary' if phase == 'spreading' else 'is extending behind its arc'}.", (parent, arc))
        elif s.t-row['last_active_myr'] >= QUIET_DELAY_MYR and row['phase'] != 'quiet':
            _record(s, row, 'quiet', f"Opening in back-arc basin {row['id']} has become quiet as relative extension waned.", (parent, arc))
        # Compression consumes opening; mere age never erases a basin.
        if row['opening_km'] > 20. and row['closing_km'] >= row['opening_km'] and shortening > 1.:
            _record(s, row, 'closed', f"Convergence has consumed the recorded opening of back-arc basin {row['id']}.", (parent, arc))

    # Advance prospective basins only with the measured loading driver:
    # relative forearc retreat (0), sea-anchor hinge lag (1), or the declared
    # effective reservoir (2). The latter two do not advect the trench.
    version = driver_version(s)
    # Drivers 1/2 measure every step so their diagnostics stay current; driver 0
    # reads the swept retreat only where a candidate pair needs it, as before.
    loading = loading_speed(s) if version else None
    sub = np.flatnonzero(normal_partition.subduction(s))
    tracked = hasattr(s, 'trench_systems')
    maturity = trench_history.weights(s)
    pairs = sorted({(int(s.down[k]), int(s.bq[k] if s.down[k] == s.bp[k] else s.bp[k]),
                     int(s.trench_id[k]) if tracked else 0)
                    for k in sub if maturity[k] > 0})
    seen = set()
    for down, over, trench_id in pairs:
        if down < 0 or not s.active[down] or not s.active[over]:
            continue
        if (down, over) in driven_pairs(s):
            continue
        # One back-arc rupture at a time on a host, avoiding nested parallel axes.
        if any(r['phase'] != 'closed' and r['arc_plate_uid'] is not None
               and s.t-r['last_active_myr'] < 60.
               and int(s.plate_uid[over]) in (r['parent_plate_uid'], r['arc_plate_uid']) for r in s.backarc_basins):
            continue
        edges = trench_edges(s, down, over)
        if tracked:
            edges = edges[s.trench_id[edges] == trench_id]
        if not len(edges):
            continue
        if loading is None:
            loading = loading_speed(s)
        speed = loading[edges]
        good = edges[speed > LOADING_EDGE_KM_MYR]
        if not len(good):
            continue
        strongest = good[np.argmax(loading[good]*s.bl[good])]
        edges = _near(s, edges, s.bmid[strongest])
        if s.bl[edges].sum() < 800.:
            continue
        center, _ = _frame(s, edges, down)
        rate = float(np.average(loading[edges], weights=s.bl[edges]))
        if rate <= LOADING_EDGE_KM_MYR or s.config['rift_strength'] <= 0:
            continue
        matches = [r for r in s.backarc_basins if r['arc_plate_uid'] is None
                   and r['parent_plate_uid'] == int(s.plate_uid[over])
                   and r['downgoing_plate_uid'] == int(s.plate_uid[down])
                   and (not tracked or r.get('trench_id') == trench_id)
                   and np.dot(r['center'], center) > math.cos(2000./RADIUS_KM)]
        if matches:
            row = max(matches, key=lambda r: np.dot(r['center'], center))
        else:
            row = dict(id=s.next_backarc_id, started_myr=float(s.t), phase='loading',
                       trench_id=trench_id if tracked else None,
                       parent_plate_uid=int(s.plate_uid[over]), arc_plate_uid=None,
                       downgoing_plate_uid=int(s.plate_uid[down]), center=center.tolist(),
                       opening_km=0., loading_km=0., extension_rate_km_myr=0., loading_rate_km_myr=rate,
                       last_active_myr=float(s.t), last_attempt_myr=-1000.)
            s.next_backarc_id += 1
            s.backarc_basins.append(row)
            _record(s, row, 'loading', (f"Persistent trench retreat is loading the overriding plate behind the arc of {s.names[over]}."
                                        if version == 0 else
                                        f"{s.names[over]} is moving away from its anchored slab, loading the plate behind its arc."
                                        if version == DRIVER_VERSION else
                                        f"{s.names[over]} is moving away from its declared effective reservoir, loading the plate behind its arc."), (over, down))
        seen.add(row['id'])
        row['center'] = center.tolist()
        row['loading_rate_km_myr'] = rate
        row['last_active_myr'] = float(s.t)
        decay = math.exp(-dt/LOADING_MEMORY_MYR)
        row['loading_km'] = row['loading_km']*decay+rate*s.config['rift_strength']*LOADING_MEMORY_MYR*(1-decay)
        if row['phase'] == 'quiet':
            _record(s, row, 'loading', (f"Retreat is again loading the prospective back-arc basin {row['id']}."
                                        if version == 0 else
                                        f"Motion away from the anchored slab is again loading the prospective back-arc basin {row['id']}."
                                        if version == DRIVER_VERSION else
                                        f"Motion away from the declared effective reservoir is again loading the prospective back-arc basin {row['id']}."), (over, down))
        if (row['loading_km'] >= RUPTURE_LOADING_KM and s.t-row['started_myr'] >= 8.
                and s.t-row['last_attempt_myr'] >= 8. and pending is None):
            row['last_attempt_myr'] = float(s.t)
            pending = (row['id'], edges.copy(), over, down)
    for row in s.backarc_basins:
        if row['arc_plate_uid'] is None and row['id'] not in seen:
            row['loading_km'] *= math.exp(-dt/LOADING_MEMORY_MYR)
            row['extension_rate_km_myr'] = 0.
            row['loading_rate_km_myr'] = 0.
            if row['phase'] != 'quiet' and s.t-row['last_active_myr'] >= QUIET_DELAY_MYR:
                owners = tuple(current[u] for u in (row['parent_plate_uid'], row['downgoing_plate_uid']) if u in current)
                _record(s, row, 'quiet', (f"Retreat loading of prospective back-arc basin {row['id']} has waned before rupture."
                                          if version == 0 else
                                          f"Anchored-slab loading of prospective back-arc basin {row['id']} has waned before rupture."
                                          if version == DRIVER_VERSION else
                                          f"Effective-reservoir loading of prospective back-arc basin {row['id']} has waned before rupture."), owners)
    return pending


def commit_pending(s, pending):
    if pending is None:
        return False
    identity, edges, parent, down = pending
    if not s.active[parent] or not s.active[down]:
        return False
    # Ordinary collision may have moved a cached cell earlier in this step.
    valid = s._valid_loading_edges()
    edges = edges[valid[edges]]
    if not len(edges):
        return False
    row = next(r for r in s.backarc_basins if r['id'] == identity)
    return _commit(s, row, edges, parent, down)


def protected_hosts(s):
    """Prevent a second generic fracture while the dedicated arc system exists."""
    return {u for r in s.backarc_basins if r['arc_plate_uid'] is not None and r['phase'] != 'closed'
            and s.t-r['last_active_myr'] < 60.
            for u in (r['parent_plate_uid'], r['arc_plate_uid'])}
