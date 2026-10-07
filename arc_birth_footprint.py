"""Conserved-volume search for a physically admissible juvenile birth footprint.

The source equivalent area funds a 25-km column V=25*A. Actual footprint area
may spread to V/8, the retained minimum column. No source position, volume,
column bound, constructive-slope bound, or geographic admission is changed.
This bounded search returns a verified feasible profile, not a global optimum.
"""
from numbers import Integral
import numpy as np
import arc_birth_profile as profile

VERSION = 1
BISECTION_STEPS = 14
AREA_MARGIN = 1e-8


def version(s):
    value = getattr(s, 'native_arc_footprint_version', 0)
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Integral) or value not in (0, VERSION):
        raise ValueError('Unsupported juvenile footprint capacity version.')
    # Explicit old-profile fixtures do not activate the new footprint policy.
    if value and profile.version(s) != 1:
        return 0
    return int(value)


def propose(factory, source_area_km2, basement_m):
    """Search the original footprint through the volume-funded column bound.

    ``factory(physical_area)`` must retain the original source and strike.
    Every returned profile is checked on its exact shared mesh. A failed upper
    endpoint retains the entire source pending; no under-thickness apron is
    used to force a birth. Bisection only retains independently passing plans.
    """
    source = float(source_area_km2)
    if not np.isfinite(source) or source <= 0:
        raise ValueError('A juvenile footprint needs positive finite source area.')
    maximum = source * profile.SOURCE_COLUMN_KM / profile.MECHANICAL_REFERENCE_KM
    count = 0

    def evaluate(area):
        nonlocal count
        count += 1
        plan = factory(area)
        return plan, profile.birth(plan, source, basement_m)

    original, initial = evaluate(source)
    selected, result = original, initial
    if not initial['diagnostics']['admissible']:
        upper = maximum * (1. - AREA_MARGIN)
        selected, result = evaluate(upper)
        if result['diagnostics']['admissible']:
            lower = source
            for _ in range(BISECTION_STEPS):
                middle = (lower + upper) * .5
                candidate, measured = evaluate(middle)
                if measured['diagnostics']['admissible']:
                    upper, selected, result = middle, candidate, measured
                else:
                    lower = middle
        else:
            selected = None
    search = dict(version=VERSION, evaluations=count,
        minimum_footprint_area_km2=source, maximum_footprint_area_km2=maximum,
        nominal_profile=initial['diagnostics'],
        selected_footprint_area_km2=(result['diagnostics']['actual_area_km2'] if selected is not None else 0.),
        admissible=selected is not None,
        policy='fixed volume; original source and strike; verified 8-75 km columns and 20-degree slope')
    return dict(plan=selected, nominal_plan=original, profile=result, search=search)


def snapshot_fields(s):
    if not version(s):
        return {}
    return dict(arc_footprint_version=VERSION, arc_footprint_policy=dict(
        version=VERSION, minimum_column_km=profile.MECHANICAL_REFERENCE_KM,
        source_column_km=profile.SOURCE_COLUMN_KM,
        maximum_area_ratio=profile.SOURCE_COLUMN_KM/profile.MECHANICAL_REFERENCE_KM,
        bisection_steps=BISECTION_STEPS,
        policy='source equivalent area and actual footprint differ; volume and original source retained'))


def validate_frame(frame):
    tag = frame.get('arc_footprint_version', 0)
    if isinstance(tag, (bool, np.bool_)) or not isinstance(tag, Integral) or tag not in (0, VERSION):
        raise ValueError('Unsupported saved juvenile footprint capacity version.')
    metadata = frame.get('arc_footprint_policy')
    if not tag:
        if metadata is not None:
            raise ValueError('Juvenile footprint capacity metadata requires its version.')
        return 0
    if frame.get('arc_birth_profile_version') != 1 or frame.get('arc_emplacement_version') != 1:
        raise ValueError('Juvenile footprint capacity requires physical profiles and geographic admission.')
    expected = dict(version=VERSION, minimum_column_km=8., source_column_km=25.,
                    maximum_area_ratio=25./8., bisection_steps=BISECTION_STEPS)
    if not isinstance(metadata, dict) or any(
            isinstance(metadata.get(key), bool) or metadata.get(key) != value for key, value in expected.items()):
        raise ValueError('Invalid saved juvenile footprint capacity metadata.')
    if not isinstance(metadata['version'], Integral) or not isinstance(metadata['bisection_steps'], Integral):
        raise ValueError('Juvenile footprint policy versions and search counts must be integers.')
    return int(tag)
