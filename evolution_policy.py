"""Saved numerical evolution policies, independent of fresh-world initialization.

Historical checkpoints without these fields retain their prior profile-based
behavior. An explicit migration can enable complete coupled contact and adaptive
rollback without rerunning initialization or changing the world's physics profile.
These policies change numerical acceptance, not forces or material properties.
"""
from copy import deepcopy
import math
from numbers import Integral


VERSION = 1
FIELDS = ('adaptive_timestep_version', 'complete_contact_version')


def _validate(value, name):
    if isinstance(value, bool) or not isinstance(value, Integral) or value not in (0, 1):
        raise ValueError(f'Unsupported {name}; expected integer version 0 or 1.')
    return int(value)


def _version(s, name):
    if hasattr(s, name):
        return _validate(getattr(s, name), name)
    return _validate(getattr(s, 'physics_profile_version', 0), 'physics_profile_version')


def adaptive_timestep_version(s):
    return _version(s, 'adaptive_timestep_version')


def complete_contact_version(s):
    return _version(s, 'complete_contact_version')


def versions(s):
    """Validate both policies before a coupled step changes any state."""
    return dict(adaptive_timestep_version=adaptive_timestep_version(s),
                complete_contact_version=complete_contact_version(s))


def upgrade(s):
    """Explicitly select both current policies without changing physical state.

    This is valid at a historical checkpoint. It adds only the two version
    scalars and their provenance, never config, RNG, profile, or initial geology.
    Calling it again after successful activation returns the original receipt.
    """
    previous = versions(s)
    epoch = float(s.t)
    if not math.isfinite(epoch) or epoch < 0:
        raise ValueError('Evolution-policy migration requires a finite nonnegative epoch.')
    selected = dict.fromkeys(FIELDS, VERSION)
    recorded = getattr(s, 'evolution_policy_migration', None)
    if recorded is not None:
        if (not isinstance(recorded, dict) or recorded.get('version') != VERSION
                or recorded.get('to_versions') != selected):
            raise ValueError('Unsupported evolution-policy migration record.')
        if all(hasattr(s, name) for name in FIELDS) and previous == selected:
            return deepcopy(recorded)
    report = dict(version=VERSION, time_myr=epoch,
                  from_versions=previous, to_versions=selected,
                  selection='Explicit checkpoint migration; fresh initialization was not run.',
                  physical_state_changed=False, config_changed=False, rng_changed=False,
                  physics_profile_changed=False)
    # All validation and report construction precede mutation.
    for name, value in selected.items():
        setattr(s, name, value)
    s.evolution_policy_migration = report
    return deepcopy(report)


def snapshot(s):
    """Expose actual evolution selections even for a migrated historical profile."""
    result = versions(s)
    if hasattr(s, 'evolution_policy_migration'):
        result['evolution_policy_migration'] = deepcopy(s.evolution_policy_migration)
    if hasattr(s, 'timestep_diagnostics'):
        result['timestep_diagnostics'] = deepcopy(s.timestep_diagnostics)
    return result
