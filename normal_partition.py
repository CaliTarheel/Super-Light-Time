"""Versioned normal convergence independent of a boundary's display label.

This selects geometry, not an established subduction mechanism. Incoming-water,
local polarity, elapsed-convergence and shortening/maturity gates still belong
to trench_history and finite native capture. Tangential traction remains active
at the same boundary in plate_balance.
"""
from copy import deepcopy
import numpy as np

VERSION = 1


def enabled(s):
    value = getattr(s, 'normal_partition_version', 0)
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)) or value not in (0, VERSION):
        raise ValueError('Unsupported normal-motion partition version.')
    return value == VERSION


def convergence(s):
    if not enabled(s):
        return np.isin(s.bcode, [2, 4])
    speed, length = np.asarray(s.normal_speed, float), np.asarray(s.bl, float)
    if speed.shape != length.shape or not np.isfinite(speed).all() or not np.isfinite(length).all() or np.any(length < 0):
        raise ValueError('Normal convergence requires aligned finite physical boundary geometry.')
    return (speed < 0.) & (length > 1e-10)


def subduction(s):
    if not enabled(s):
        return np.asarray(s.bcode) == 2
    return convergence(s) & ((s.down == s.bp) | (s.down == s.bq))


def collision(s):
    if not enabled(s):
        return np.asarray(s.bcode) == 4
    return convergence(s) & (s.crust[s.ba] > 0) & (s.crust[s.bb] > 0)


def capture_candidates(s):
    if not enabled(s):
        return np.asarray(s.bcode) == 2
    # The finite ribbon handles opening/closing locally. Do not discard a
    # locally closing piece because a parent's mean velocity hides it.
    return (np.asarray(s.bl) > 1e-10) & ((s.down == s.bp) | (s.down == s.bq))


def upgrade(s):
    enabled(s)  # validate the prior version
    if getattr(s, 'normal_partition_version', 0):
        return deepcopy(getattr(s, 'normal_partition_migration', {}))
    report = dict(from_version=0, to_version=VERSION, time_myr=float(s.t),
                  initiation_model_changed=False, maturity_created=False,
                  policy='normal convergence and along-strike sliding can act simultaneously')
    s.normal_partition_version = VERSION
    s.normal_partition_migration = report
    return deepcopy(report)


def snapshot(s):
    if not enabled(s):
        return {}
    return dict(normal_partition_version=VERSION,
                normal_partition_migration=deepcopy(getattr(s, 'normal_partition_migration', {})))


def validate_frame(frame):
    from types import SimpleNamespace
    enabled(SimpleNamespace(normal_partition_version=frame.get('normal_partition_version', 0)))
    if frame.get('normal_partition_version', 0) and frame.get('native_subduction_version') != 1:
        raise ValueError('Saved normal-motion partition requires the finite native subduction model.')
