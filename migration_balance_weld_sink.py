"""Upgrade a saved world's state for the force balance, the suture weld and the mass sink.

Called once by validation/balance-weld-sink-20260914/migrate.py against a Simulation that
checkpoint.read_checkpoint has restored, never by the engine. It is deliberately thin: each
module already knows how to create its own fields, and duplicating that knowledge here is
how a migration and an engine drift apart. What this module owns is the ORDER, which is not
optional -- every ensure_fields below is version gated, so the flags have to be set first or
the helpers will quietly do nothing and the first migrated step will fail on a missing field
rather than here, where it can still be diagnosed.

It must not touch two things. config is compared against the embedded manifest by
read_checkpoint, so changing it makes the archive unreadable. rng is carried across verbatim
so the saved stream continues; note separately that deleting the Ornstein-Uhlenbeck mantle
draw removes the only per-step draw on the native path, so the stream shifts from this epoch
by consuming less, not by being reseeded.

Idempotent: running it twice is a no-op, because every helper it calls is.
"""
from __future__ import annotations

import numpy as np

import collision_contacts
import eclogite_sink
import plate_balance
import slab_memory


def upgrade_state(s):
    """Set the version flags, create the new state, and prove the result is consistent."""
    report = {'time_myr': float(s.t), 'steps': int(getattr(s, 'steps', -1))}
    before = _column_volume_km3(s)

    # 1. Flags first: everything below is gated on them.
    s.plate_balance_version = plate_balance.VERSION
    import normal_partition
    report['normal_partition'] = normal_partition.upgrade(s)
    s.suture_weld_version = collision_contacts.SUTURE_WELD_VERSION
    report['foundering_inventory'] = eclogite_sink.upgrade_inventory(s)
    report['versions'] = dict(plate_balance=s.plate_balance_version,
                              suture_weld=s.suture_weld_version,
                              foundering=s.foundering_version)

    # 2. Per-contact weld clocks. A contact that predates the weld has no record of when it
    #    last converged; ensure_fields seeds it from the row's own history so a long-dead
    #    suture starts cold (strong) rather than hot.
    contacts_before = len(getattr(s, 'collision_contacts', []) or [])
    collision_contacts.ensure_fields(s)
    import weld_geometry
    report['local_weld'] = weld_geometry.upgrade(s)
    import collision_interface
    report['interface_shear'] = collision_interface.upgrade(s)
    report['collision_contacts'] = contacts_before
    report['welded_contacts'] = sum(
        1 for row in (getattr(s, 'collision_contacts', None) or [])
        if float(row.get('suture_strength', 0.)) >= .95)

    # 3. The foundering ledger, the root clock and the mantle reservoir.
    eclogite_sink.ensure_fields(s)
    eclogite_sink.validate_alignment(s)
    report['foundering_depth'] = eclogite_sink.upgrade_depth_integration(s)
    report['foundered_m_columns'] = int(len(s.structure[eclogite_sink.COLUMN_FIELD]))
    report['foundered_m_traces'] = int(len(s.trace_structure[eclogite_sink.COLUMN_FIELD]))
    report['mantle_return_km3'] = dict(s.mantle_return_km3)
    report['root_clock_heated_faces'] = int(np.count_nonzero(
        np.asarray(s.parcel_root_age_myr, float) >= eclogite_sink.HEATING_DELAY_MYR))

    # 4. Slab line densities. The balance sums slab weight over attached trench edges, and a
    #    run that predates the line load records only accumulated shortening, so reconstruct
    #    it from that. Without this every trench would start the first migrated step pulling
    #    nothing and the continents would briefly go slack.
    report['slab_bootstrap'] = slab_memory.bootstrap_slab_state(s)

    # 5. Prove no mass moved. This upgrade adds bookkeeping; it must not touch a column.
    after = _column_volume_km3(s)
    report['column_volume_km3'] = after
    drift = abs(after - before)
    if drift > max(1e-6, 1e-12*max(abs(before), 1.)):
        raise ValueError(f'Migration changed crustal volume by {drift:g} km3; it must not.')
    report['column_volume_drift_km3'] = drift
    if abs(float(s.mantle_return_km3['total'])) > 0.:
        raise ValueError('Migration must open the mantle ledger at zero.')
    return report


def _column_volume_km3(s):
    """Total crustal volume, as the column budget measures it."""
    structure = getattr(s, 'structure', None)
    if not structure or 'thickness_km' not in structure:
        return 0.
    area = np.asarray(s.mass, float)
    thickness = np.asarray(structure['thickness_km'], float)
    factor = np.asarray(structure.get('area_factor', np.ones(len(thickness))), float)
    return float(np.sum(area*factor*thickness))
