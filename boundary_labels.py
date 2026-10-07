"""Map and statistics labels taken from the mechanism that actually runs.

The kinematic code in `s.bcode` is a gate for processes: a boundary is called
divergent, convergent or sliding by comparing its normal speed with
`max(2 km/Myr, 0.15 * sliding speed)`. Normal partition version 1 then subducts
and collides on any closing motion, whatever that gate said. The map and the
length statistics read the gate, so a margin the engine is subducting could be
drawn as a transform, and inherited trenches needed a separate override to look
like subduction at all.

These labels answer a different question: what is this boundary doing? A closing
edge with a downgoing plate and water coming in is subduction. A closing edge
with continental material on both sides, or one whose incoming plate has become
buoyant and jammed the margin, is collision. Spreading keeps the exact
classification of the production path. Everything else that still exists is
sliding. `s.bcode` is untouched, so no process changes; only what is displayed,
counted and exported does. See IMPLEMENTATION_GAPS.md G116 and G117.
"""
from __future__ import annotations

from copy import deepcopy
from numbers import Integral

import numpy as np

import normal_partition

VERSION = 1
NONE, RIDGE, SUBDUCTION, TRANSFORM, COLLISION, RIFT = range(6)
MINIMUM_LENGTH_KM = 1e-10


def enabled(s):
    version = getattr(s, 'boundary_label_version', 0)
    if isinstance(version, (bool, np.bool_)) or not isinstance(version, Integral) or version not in (0, VERSION):
        raise ValueError('Unsupported boundary label version.')
    return int(version) == VERSION


def upgrade(s):
    """Select mechanism labels. Display only: no physical state changes."""
    if enabled(s):
        return deepcopy(getattr(s, 'boundary_label_migration', {}))
    report = dict(version=VERSION, time_myr=float(s.t),
                  selection='Map, statistics and exports label each boundary by the mechanism that runs on it.',
                  physical_state_changed=False, kinematic_code_retained=True)
    s.boundary_label_version = VERSION
    s.boundary_label_migration = report
    return deepcopy(report)


def incoming_water(s, down=None):
    """Whether each contact's incoming side carries any ocean."""
    import native_subduction
    owner = s.down if down is None else down
    if native_subduction.enabled(s):
        return native_subduction.edge_ocean_fraction(s, owner) > 1e-10
    owner = np.asarray(owner)
    below = np.where(owner == np.asarray(s.bp), np.asarray(s.ba), np.asarray(s.bb))
    return (np.asarray(s.crust)[below] == 0) & (owner >= 0)


def carries_material(s):
    """Whether either side of each contact carries buoyant crust.

    Measured on the material surface where that geometry exists: a margin can
    hold continental material inside a control cell the coarse crust flag still
    calls ocean, and a stalled trench is exactly that case.
    """
    import native_subduction
    if native_subduction.enabled(s):
        water_if = [native_subduction.edge_ocean_fraction(s, side) for side in (s.bp, s.bq)]
        return (water_if[0] < 1.-1e-10) | (water_if[1] < 1.-1e-10)
    crust = np.asarray(s.crust)
    return (crust[np.asarray(s.ba)] > 0) | (crust[np.asarray(s.bb)] > 0)


def codes(s):
    """Per-contact display codes, in the same 0-5 vocabulary as `s.bcode`."""
    kinematic = np.asarray(s.bcode, np.uint8)
    if not enabled(s):
        return kinematic.copy()
    live = np.asarray(s.bl, float) > MINIMUM_LENGTH_KM
    display = np.where(live, TRANSFORM, NONE).astype(np.uint8)
    # Production decides divergence exactly; keep what it resolved.
    display[live & (kinematic == RIDGE)] = RIDGE
    display[live & (kinematic == RIFT)] = RIFT
    closing = normal_partition.convergence(s)
    # Continental material in a closing boundary is a collision, whether the
    # two sides are both continental or a buoyant plate has arrived and jammed
    # the margin. A stalled trench keeps no polarity (physics_profile.
    # stall_buoyant_incoming clears s.down), so polarity cannot identify it.
    display[live & closing & carries_material(s)] = COLLISION
    polarity = (np.asarray(s.down) == np.asarray(s.bp)) | (np.asarray(s.down) == np.asarray(s.bq))
    display[live & closing & polarity & incoming_water(s)] = SUBDUCTION
    return display


def lengths(s, display=None):
    """Boundary length in km for each label, from the same codes the map shows."""
    display = codes(s) if display is None else np.asarray(display, np.uint8)
    length = np.asarray(s.bl, float)
    return {name: float(length[display == code].sum()) for name, code in
            (('ridge', RIDGE), ('subduction', SUBDUCTION), ('transform', TRANSFORM),
             ('collision', COLLISION), ('rift', RIFT))}


def disagreement(s, display=None):
    """What the gate would have shown against what the mechanism is doing.

    Reported so the change is visible in every frame rather than silent.
    """
    display = codes(s) if display is None else np.asarray(display, np.uint8)
    kinematic = np.asarray(s.bcode, np.uint8)
    length = np.asarray(s.bl, float)
    moved = display != kinematic
    rows = {}
    for code, name in ((SUBDUCTION, 'subduction'), (COLLISION, 'collision'),
                       (TRANSFORM, 'transform'), (RIDGE, 'ridge'), (RIFT, 'rift')):
        gained = moved & (display == code)
        if np.any(gained):
            rows[name] = dict(contacts=int(np.count_nonzero(gained)),
                              length_km=float(length[gained].sum()),
                              from_kinematic={str(int(c)): int(np.count_nonzero(gained & (kinematic == c)))
                                              for c in np.unique(kinematic[gained])})
    return dict(version=VERSION, relabelled_contacts=int(np.count_nonzero(moved)),
                relabelled_length_km=float(length[moved].sum()), by_label=rows,
                policy='Labels follow the running mechanism; s.bcode remains the kinematic gate.')
