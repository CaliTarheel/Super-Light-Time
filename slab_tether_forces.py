"""Frozen-state slab attachment assembly for the actual plate balance.

This does not enable a timestep/polarity policy. Version 1 channels are spread
uniformly over the matched trace; version 2 uses persistent local anchor cells.
Each retains its own material and damage. Reference neck width sets integrated
neck resistance; current width sets the depth-window allocation. Neither force
evaluation nor rematching changes mass.
"""
import math
from copy import deepcopy
import numpy as np
import slab_memory as slab
import slab_tether_history as history
import slab_tether_local as local
from slab_tether import Neck


def prepare(balance):
    """Return a PSD stiffness and work-conjugate drive in Balance's coordinates.

    Fixed-hinge intake is the incoming plate's normal speed. A moving hinge
    replaces it by incoming minus overriding speed. The independent slab speed
    is relative to that hinge; its absolute mantle velocity has horizontal
    component h+u*cos(dip) and vertical component u*sin(dip), where h is the
    overriding plate's normal speed. Eliminating u from one quadratic energy
    gives reciprocal plate forces and positive mantle/neck dissipation.
    """
    import plate_balance as pb
    s = balance.s
    if not slab.conservative(s):
        raise ValueError('Slab attachment forces require conservative excess-mass histories.')
    stiffness = np.zeros((balance.size, balance.size))
    drive = np.zeros(balance.size)
    down_drive = np.zeros(balance.size)
    over_drive = np.zeros(balance.size)
    count = len(balance.bp)
    owners = np.full(count, -1, int)
    loads = np.zeros(count)
    extents = np.zeros(count)
    line_drives = np.zeros(count)
    records = []
    slots = {int(s.plate_uid[p]): int(p) for p in balance.plates}
    ids = np.asarray(s.trench_id)
    live = np.zeros(count, bool); live[balance.live] = True
    moving = getattr(balance, 'subduction_response_version', 0) == 1
    dip = math.radians(slab.SUBDUCTION_DIP_DEG)
    cosine = math.cos(dip) if moving else 0.
    sine = math.sin(dip) if moving else 1.
    for row in s.trench_systems:
        slab.validate_row(row, require_mass=True)
        if row['phase'] in ('shutdown', 'joined'):
            continue
        down, over = slots.get(row['downgoing_plate_uid']), slots.get(row['overriding_plate_uid'])
        if down is None or over is None:
            continue
        select = live & (ids == row['id']) & (((balance.bp == down) & (balance.bq == over)) |
                                             ((balance.bp == over) & (balance.bq == down)))
        edges = np.flatnonzero(select)
        if not len(edges):
            continue
        if history.FIELD not in row:
            raise ValueError('Explicit slab neck histories are required before attachment force assembly.')
        channels = row[history.FIELD]
        total_reference_width = math.fsum(c['reference_width_m'] for c in channels if c['neck']['damage'] < 1.)
        if total_reference_width <= 0.:
            continue
        trace_width = float(balance.length_m[edges].sum())
        if local.enabled(row):
            allocated = local.allocation(row,balance.radial[edges],balance.length_m[edges])
        else:
            allocated = np.asarray([c['reference_width_m']/total_reference_width*balance.length_m[edges]
                                    for c in channels])
        inlet = balance._absolute_rows(edges, np.full(len(edges), down))
        sign = np.where(balance.bp[edges] == down, 1., -1.)[:, None]
        inlet *= sign
        hinge = (balance._absolute_rows(edges, np.full(len(edges), over))*sign
                 if moving else np.zeros_like(inlet))
        intake = inlet-hinge
        for index, channel in enumerate(channels):
            material = Neck(**channel['neck'])
            area = channel['retained_area_km2']
            width = float(allocated[index].sum())
            if material.damage == 1. or area <= 0. or width <= 0.:
                continue
            shares = allocated[index]/width
            geometric = dict(length_km=width/1000., slab_retained_area_km2=area)
            fraction = slab.coupled_fraction(geometric)
            extent = slab.slab_length_km(geometric)
            mass = channel['retained_excess_mass_kg']*fraction
            weight = pb.GRAVITY_M_S2*mass*math.sin(math.radians(slab.SUBDUCTION_DIP_DEG))
            mantle_drag = pb.SLAB_STOKES_PA_S*(extent/slab.upper_mantle_length_km())*width
            neck_drag = material.coefficient_pa_s*channel['reference_width_m']
            scale = max(mantle_drag, neck_drag)
            total_scaled = mantle_drag/scale+neck_drag/scale
            alpha = (neck_drag/scale)/total_scaled
            transmitted_drive, transmitted_drag = alpha*weight, alpha*mantle_drag
            # Sum-of-squares Schur complement of the independent slab speed.
            coupled = intake+cosine*hinge
            stiffness += np.einsum('e,ei,ej->ij', shares*transmitted_drag*pb.CM_YR_M_S**2, coupled, coupled)
            if moving:
                stiffness += np.einsum('e,ei,ej->ij',
                    shares*mantle_drag*sine*sine*pb.CM_YR_M_S**2, hinge, hinge)
            hinge_drive = -weight*(mantle_drag/scale)/total_scaled*cosine
            down_drive += np.einsum('e,ei->i', shares*transmitted_drive*pb.CM_YR_M_S, inlet)
            over_drive += np.einsum('e,ei->i',
                shares*(hinge_drive-transmitted_drive)*pb.CM_YR_M_S, hinge)
            owners[edges[shares>0]] = down
            loads[edges] += mass*shares/balance.length_m[edges]
            line_drives[edges] += transmitted_drive*shares/balance.length_m[edges]
            extents[edges] += allocated[index]/balance.length_m[edges]*extent
            records.append(dict(trench_id=int(row['id']), channel_index=index, edges=edges.copy(),
                channel_state=deepcopy(channel),
                edge_shares=shares.copy(), inlet_rows=intake.copy(), hinge_rows=hinge.copy(),
                dip_cosine=cosine, dip_sine=sine, moving_hinge=moving, weight_n=weight,
                mantle_drag_n_s_m=mantle_drag, neck_drag_n_s_m=neck_drag,
                drive_n=transmitted_drive, drag_n_s_m=transmitted_drag,
                coupled_mass_kg=mass, coupled_area_km2=area*fraction,
                transmission_fraction=alpha))
    drive = down_drive+over_drive
    if not np.isfinite(stiffness).all() or not np.isfinite(drive).all():
        raise ValueError('Slab attachment assembly exceeds its finite constitutive range.')
    return dict(stiffness=stiffness, drive=drive, down_drive=down_drive,
                over_drive=over_drive, owners=owners, loads=loads,
                extents=extents, channels=records, line_drives=line_drives)


def prepare_horizontal(balance):
    """Read-only alternative with neck slip measured in horizontal speed.

    Reuse the existing live trace, local neck and conserved slab inventory
    selection, but replace its quadratic force law. No native Balance call
    selects this result until entry depth, sources and coordinates agree.
    """
    import plate_balance as pb

    if getattr(balance, 'subduction_response_version', 0) != 1:
        raise ValueError('Horizontal intake needs the moving-hinge channel inventory.')
    inherited = prepare(balance)
    stiffness = np.zeros_like(inherited['stiffness'])
    down_drive = np.zeros_like(inherited['drive'])
    over_drive = np.zeros_like(inherited['drive'])
    line_drives = np.zeros_like(inherited['line_drives'])
    channels = []
    for record in inherited['channels']:
        cs, cn, weight = (record[key] for key in
                          ('mantle_drag_n_s_m', 'neck_drag_n_s_m', 'weight_n'))
        cosine, sine = record['dip_cosine'], record['dip_sine']
        denominator = cs + cn * cosine**2
        q = record['inlet_rows']
        h = record['hinge_rows']
        d = q + h
        shares = record['edge_shares']
        coupled = q + cosine**2 * h
        drag = cs * cn / denominator
        stiffness += np.einsum('e,ei,ej->ij',
            shares * drag * pb.CM_YR_M_S**2, coupled, coupled)
        stiffness += np.einsum('e,ei,ej->ij',
            shares * cs * sine**2 * pb.CM_YR_M_S**2, h, h)
        down_force = weight * cn * cosine / denominator
        over_force = -weight * (cn + cs) * cosine / denominator
        down_drive += np.einsum('e,ei->i',
            shares * down_force * pb.CM_YR_M_S, d)
        over_drive += np.einsum('e,ei->i',
            shares * over_force * pb.CM_YR_M_S, h)
        edges = record['edges']
        line_drives[edges] += down_force * shares / balance.length_m[edges]
        changed = dict(record, horizontal_intake=True,
                       drive_n=down_force, drag_n_s_m=drag,
                       horizontal_drive_gain=cn * cosine / denominator)
        changed.pop('transmission_fraction', None)
        channels.append(changed)
    drive = down_drive + over_drive
    if not np.isfinite(stiffness).all() or not np.isfinite(drive).all():
        raise ValueError('Horizontal slab assembly exceeded its finite range.')
    return dict(inherited, stiffness=stiffness, drive=drive,
                down_drive=down_drive, over_drive=over_drive,
                line_drives=line_drives, channels=channels,
                scope='read-only horizontal-intake assembly; native law unchanged')


def _recover_channel(channel, x):
    """Recover the same eliminated slab coordinates for all local force readers."""
    import plate_balance as pb
    v = channel['inlet_rows']@np.asarray(x, float)*pb.CM_YR_M_S
    h = channel['hinge_rows']@np.asarray(x, float)*pb.CM_YR_M_S
    cs, cn, weight = (channel[k] for k in ('mantle_drag_n_s_m','neck_drag_n_s_m','weight_n'))
    cosine, sine = channel['dip_cosine'], channel['dip_sine']
    if channel.get('horizontal_intake', False):
        denominator = cs + cn * cosine**2
        u = (weight + cn * cosine * v - cs * cosine * h) / denominator
        opening = cosine * u - v
    else:
        scale = max(cs, cn)
        denominator = cs/scale+cn/scale
        u = (weight/scale+(cn/scale)*v-(cs/scale)*cosine*h)/denominator
        opening = (weight/scale-(cs/scale)*(v+cosine*h))/denominator
    return v, h, u, opening


def edge_force_ledger(assembly, x):
    """Read-only work-conjugate slab forces on each matched channel edge.

    Rows have Balance's generalized coordinates, W/(cm/year). Their sum is
    ``assembly['drive'] - assembly['stiffness'] @ x``; their work is the
    recovered physical plate work in :func:`audit`. Multiple channels may
    contribute rows to the same edge, so callers must accumulate those rows.
    No second raw slab-weight pull or equal-and-opposite upper reaction should
    be added: these net rows already contain the current transmitted neck
    traction and eliminated-slab mantle reaction.

    This export excludes Balance's separate anchor, bending, basal, collision,
    and plastic-interface terms. It supplies current local force/onset data,
    not a finite breakup speed or a spatial cell interpolation. In particular,
    velocity-dependent resistance must be reevaluated for daughter motion.
    Both the ordinary and diagnostic horizontal-intake assemblies are supported;
    calling this reader does not select or enable either evolution law.
    """
    import plate_balance as pb
    x = np.asarray(x, float)
    size = len(assembly['drive'])
    if x.shape != (size,) or not np.isfinite(x).all():
        raise ValueError('Local slab force export requires finite Balance coordinates of the assembled size.')
    edges, forces, channel_ids = [], [], []
    for index, channel in enumerate(assembly['channels']):
        v, h, u, opening = _recover_channel(channel, x)
        cs, cn = channel['mantle_drag_n_s_m'], channel['neck_drag_n_s_m']
        cosine = channel['dip_cosine']
        traction = cn*opening
        # q = v_down - h on the moving-hinge branch. Thus T*q - C_s*(h+c*u)*h
        # gives both the downgoing traction and the overriding mantle reaction.
        local = channel['edge_shares'][:, None]*pb.CM_YR_M_S*(
            traction[:, None]*channel['inlet_rows']
            -(cs*(h+cosine*u))[:, None]*channel['hinge_rows'])
        edges.append(np.asarray(channel['edges'], int).copy())
        forces.append(local)
        channel_ids.append(np.full(len(local), index, int))
    if not edges:
        return dict(edges=np.zeros(0, int), channel_index=np.zeros(0, int),
                    net_generalized_forces=np.zeros((0, size)))
    result = np.vstack(forces)
    if not np.isfinite(result).all():
        raise ValueError('Local slab force export exceeds its finite constitutive range.')
    return dict(edges=np.concatenate(edges), channel_index=np.concatenate(channel_ids),
                net_generalized_forces=result)


def audit(assembly, x):
    """Recover eliminated velocities and compare plate work with slab power."""
    totals = dict(weight_power_w=0., plate_power_w=0., mantle_dissipation_w=0., neck_dissipation_w=0.)
    for channel in assembly['channels']:
        v, h, u, opening = _recover_channel(channel, x)
        cs, cn, weight = (channel[k] for k in ('mantle_drag_n_s_m','neck_drag_n_s_m','weight_n'))
        cosine, sine = channel['dip_cosine'], channel['dip_sine']
        traction = cn*opening
        shares = channel['edge_shares']
        totals['weight_power_w'] += float(shares@(weight*u))
        totals['plate_power_w'] += float(shares@(traction*v-cs*(h+cosine*u)*h))
        totals['mantle_dissipation_w'] += float(shares@(cs*((h+cosine*u)**2+(sine*u)**2)))
        totals['neck_dissipation_w'] += float(shares@(cn*opening**2))
    totals['power_residual_w'] = (totals['weight_power_w']-totals['plate_power_w']-
                                  totals['mantle_dissipation_w']-totals['neck_dissipation_w'])
    return totals
