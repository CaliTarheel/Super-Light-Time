"""Finite ridge-subduction thermal and volcanic episodes for worldbuilding.

An actual oceanic spreading edge must meet a trench through their common
downgoing cell, involve a third plate, and approach the hinge on the sphere.
Young seafloor alone never triggers an episode. The 32 Myr lifetime, 400 m
thermal support, and 200 m construction budget are explicit model choices,
not universal predictions of ridge subduction or an added mantle force.
"""
from __future__ import annotations

import math
import numpy as np
import normal_partition

RADIUS_KM = 6371.
CONTACT_DISTANCE_KM = 350.
YOUNG_AGE_MYR = 20.
EPISODE_DURATION_MYR = 32.
EPISODE_PEAK_MYR = 8.
EPISODE_RADIUS_KM = 450.
LATCH_DISTANCE_KM = 600.
LATCH_MEMORY_MYR = 80.


def _unit(vector):
    vector = np.asarray(vector, dtype=float)
    return vector / max(float(np.linalg.norm(vector)), 1e-30)


def _distance(a, b):
    return RADIUS_KM * math.acos(float(np.clip(np.dot(a, b), -1., 1.)))


def detect_ridge_trench_contacts(s, *, max_distance_km=CONTACT_DISTANCE_KM,
                                young_age_myr=YOUNG_AGE_MYR):
    """Read boundary geometry and return independently evidenced contacts.

    Shared downgoing cells enforce resolved ridge/trench adjacency; the
    physical-distance cap excludes huge coarse cells and spherical shortcuts.
    Axis motion is estimated as the mean motion of the two spreading plates.
    Both incoming plates need not retain a trench contact after the encounter.
    """
    ridge = np.flatnonzero((s.bcode == 1) & (s.crust[s.ba] == 0)
                           & (s.crust[s.bb] == 0))
    sub = np.flatnonzero(normal_partition.subduction(s))
    if not len(ridge) or not len(sub):
        return []
    by_cell = {}
    for k in ridge:
        if max(float(s.age[s.ba[k]]), float(s.age[s.bb[k]])) > young_age_myr:
            continue
        for plate, cell in ((s.bp[k], s.ba[k]), (s.bq[k], s.bb[k])):
            by_cell.setdefault((int(plate), int(cell)), []).append(int(k))
    contacts = []
    for k in sub:
        down = int(s.down[k])
        if down < 0 or down not in (int(s.bp[k]), int(s.bq[k])):
            continue
        down_a = down == s.bp[k]
        below, above = (s.ba[k], s.bb[k]) if down_a else (s.bb[k], s.ba[k])
        over = int(s.bq[k] if down_a else s.bp[k])
        if s.crust[below] != 0 or s.age[below] > young_age_myr:
            continue
        direction = _unit(s.bn[k] if down_a else -s.bn[k])
        trench = _unit(s.bmid[k])
        for r in by_cell.get((down, int(below)), ()):
            other = int(s.bq[r] if s.bp[r] == down else s.bp[r])
            if len({int(s.plate_uid[p]) for p in (down, other, over)}) != 3:
                continue
            axis = _unit(s.bmid[r])
            distance = _distance(axis, trench)
            if distance > max_distance_km:
                continue
            # Positive is the overriding side. Allow only a small geometric
            # reconstruction tolerance, never a distant backarc heat trigger.
            side_km = RADIUS_KM * math.asin(float(np.clip(np.dot(axis, direction), -1., 1.)))
            if side_km > 25.:
                continue
            axis_omega = .5 * (s.omega[down] + s.omega[other])
            relative = np.cross(axis_omega - s.omega[over], axis) * RADIUS_KM
            approach = float(np.dot(relative, direction))
            if approach <= .5:
                continue
            angle = 180. / RADIUS_KM
            center = _unit(math.cos(angle) * trench + math.sin(angle) * direction)
            if s.plate[int(s._indices(center[None, :])[0])] != over:
                center = _unit(s.xyz[above])
            contacts.append(dict(
                overriding_plate_uid=int(s.plate_uid[over]),
                incoming_plate_uids=sorted((int(s.plate_uid[down]), int(s.plate_uid[other]))),
                downgoing_plate_uid=int(s.plate_uid[down]), center=center.tolist(),
                evidence=dict(ridge_trench_distance_km=distance,
                              axis_approach_km_myr=approach,
                              incoming_age_myr=float(s.age[below]),
                              ridge_max_age_myr=float(max(s.age[s.ba[r]], s.age[s.bb[r]])),
                              detection_distance_limit_km=float(max_distance_km),
                              shared_downgoing_cell=int(below),
                              ridge_boundary_index=int(r), trench_boundary_index=int(k),
                              xyz=trench.tolist(), normal=direction.tolist())))
    return sorted(contacts, key=lambda row: (row['evidence']['ridge_trench_distance_km'],
                                            row['overriding_plate_uid'],
                                            row['incoming_plate_uids'],
                                            row['evidence']['trench_boundary_index']))


def _rotate(point, rotation):
    angle = float(np.linalg.norm(rotation))
    if angle < 1e-15:
        return _unit(point)
    axis = np.asarray(rotation, float) / angle
    point = np.asarray(point, float)
    return _unit(point * math.cos(angle) + np.cross(axis, point) * math.sin(angle)
                 + axis * np.dot(axis, point) * (1 - math.cos(angle)))


def update_ridge_interactions(s, dt):
    """Advect, detect and latch; mutate only plain episode bookkeeping.

    Call after the step's advection and boundary refresh, with current ``s.t``.
    Existing centers use the same Euler motion as their overriding plate.
    Return new episodes for the engine's event catalogue. A continuous contact
    refreshes its latch but never restarts or replenishes its thermal pulse.
    """
    if not math.isfinite(float(dt)) or dt < 0:
        raise ValueError('Episode elapsed time must be finite and nonnegative.')
    now = float(s.t)
    episodes = getattr(s, 'ridge_episodes', [])
    live = getattr(s, 'active', np.ones(len(s.plate_uid), bool))
    slots = {int(s.plate_uid[p]): int(p) for p in np.flatnonzero(live)}
    for episode in episodes:
        slot = slots.get(int(episode['overriding_plate_uid']))
        if slot is not None:
            episode['center'] = _rotate(episode['center'], s.omega[slot] * dt).tolist()
    episodes = [row for row in episodes if now <= row['end_myr']
                or now - row['last_seen_myr'] <= LATCH_MEMORY_MYR]
    created = []
    next_id = int(getattr(s, 'next_ridge_episode_id', 1))
    for contact in detect_ridge_trench_contacts(s):
        nearby = [row for row in episodes
                  if row['overriding_plate_uid'] == contact['overriding_plate_uid']
                  and _distance(row['center'], contact['center']) <= LATCH_DISTANCE_KM]
        if nearby:
            nearest = min(nearby, key=lambda row: _distance(row['center'], contact['center']))
            nearest['last_seen_myr'] = now
            continue
        episode = dict(contact, id=next_id, started_myr=now,
                       peak_myr=now + EPISODE_PEAK_MYR,
                       end_myr=now + EPISODE_DURATION_MYR, last_seen_myr=now,
                       radius_km=EPISODE_RADIUS_KM, thermal_peak_m=400.,
                       volcanic_budget_m=200., mechanism='parameterized ridge-subduction thermal window')
        next_id += 1
        episodes.append(episode)
        created.append(episode)
    s.ridge_episodes = episodes
    s.next_ridge_episode_id = next_id
    return created


def reassign_ridge_episodes(s, source_uid, target_uid, select=None):
    """Transfer selected episode hosts during a physical weld or fracture.

    ``select`` is an optional boolean mask matching the episode list. Stable
    UIDs prevent accidental attachment to a later plate reusing an array slot.
    The engine decides which centers belong to the transferred material.
    """
    episodes = getattr(s, 'ridge_episodes', [])
    selected = np.ones(len(episodes), bool) if select is None else np.asarray(select, bool)
    if selected.shape != (len(episodes),):
        raise ValueError('Episode reassignment mask must match the episode list.')
    for use, episode in zip(selected, episodes):
        if use and episode['overriding_plate_uid'] == int(source_uid):
            episode['overriding_plate_uid'] = int(target_uid)


def _profile(age, peak, duration):
    if age <= 0. or age >= duration:
        return 0.
    if age <= peak:
        return math.sin(math.pi * age / (2 * peak)) ** 2
    return math.cos(math.pi * (age - peak) / (2 * (duration - peak))) ** 2


def _profile_integral(age, peak, duration):
    age = min(max(float(age), 0.), duration)
    if age <= peak:
        return age / 2 - peak / (2 * math.pi) * math.sin(math.pi * age / peak)
    fall = duration - peak
    after = age - peak
    return peak / 2 + after / 2 + fall / (2 * math.pi) * math.sin(math.pi * after / fall)


def sample_ridge_effects(episodes, xyz, owner_uids, time_myr, dt_myr=0.):
    """Sample owner-clipped effects; thermal support is never accumulated.

    Return ``thermal_support_m`` (instantaneous, capped at 600 m), ``heat``
    (0..1), and ``volcanic_addition_m`` (exact nonnegative integral over
    [time_myr-dt_myr, time_myr]). At a fixed plate-attached center one episode
    can add at most its 200 m construction budget, independently of timestep.
    It adds no buoyant area, new island, ordinary arc flux or motion force.
    """
    points = np.asarray(xyz, dtype=float)
    owners = np.asarray(owner_uids)
    if points.shape != (len(owners), 3):
        raise ValueError('Ridge samples need one three-vector per owner UID.')
    if not math.isfinite(float(dt_myr)) or dt_myr < 0:
        raise ValueError('Ridge sampling interval must be finite and nonnegative.')
    thermal = np.zeros(len(points), np.float32)
    heat = np.zeros(len(points), np.float32)
    volcanic = np.zeros(len(points), np.float32)
    for episode in episodes:
        start = float(episode['started_myr'])
        duration = float(episode['end_myr']) - start
        peak = float(episode['peak_myr']) - start
        age = float(time_myr) - start
        pulse = _profile(age, peak, duration)
        integrated = (_profile_integral(age, peak, duration)
                      - _profile_integral(age - dt_myr, peak, duration)) / (duration / 2)
        if pulse <= 0 and integrated <= 0:
            continue
        cells = np.flatnonzero(owners == int(episode['overriding_plate_uid']))
        if not len(cells):
            continue
        distance = RADIUS_KM * np.arccos(np.clip(points[cells] @ np.asarray(episode['center']), -1., 1.))
        within = distance < float(episode['radius_km'])
        cells, distance = cells[within], distance[within]
        weight = np.cos(np.pi * distance / (2 * float(episode['radius_km']))) ** 2
        thermal[cells] += float(episode['thermal_peak_m']) * pulse * weight
        heat[cells] = np.maximum(heat[cells], pulse * weight)
        volcanic[cells] += float(episode['volcanic_budget_m']) * max(integrated, 0.) * weight
    np.minimum(thermal, 600., out=thermal)
    return dict(thermal_support_m=thermal, heat=heat, volcanic_addition_m=volcanic)
