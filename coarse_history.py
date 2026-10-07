"""Named reduced native history mode: rigid sheets and retained crustal stacks.

Uses existing finite advection, force laws and conservative column processes.
It does not claim internal P1 strain or gravitational relaxation was solved.
"""
from copy import deepcopy
import math

MODE = 'coarse_rigid_sheet_history_v1'


def normalize(value):
    if value is None:
        return None
    if type(value) is not str or value != MODE:
        raise ValueError('Unsupported history_mode; choose coarse_rigid_sheet_history_v1 or null.')
    return value


def select_config(incoming, mode):
    """Normalize explicit mode without silently accepting conflicting switches."""
    normalize(mode)
    if mode is None:
        return
    for key in ('deforming_regions', 'adaptive_refinement'):
        if key in incoming and (type(incoming[key]) is bool or incoming[key] != 0):
            raise ValueError(MODE + ' requires ' + key + '=0.')
        incoming[key] = 0


def enabled(s):
    return normalize(s.config.get('history_mode')) == MODE


def validate(s):
    if not enabled(s):
        return
    for key in ('deforming_regions','adaptive_refinement'):
        if type(s.config.get(key)) is not int or s.config[key] != 0:
            raise ValueError('Coarse rigid-sheet configuration changed: ' + key)
    for key in ('collision_surface_version','material_mechanics_version'):
        if type(getattr(s,key,None)) is not int or getattr(s,key) != 1:
            raise ValueError('Coarse history requires retained native stacked-column mechanics: ' + key)
    marker=getattr(s,'coarse_history_policy',None)
    if not isinstance(marker,dict) or marker.get('mode') != MODE or marker.get('version') != 1:
        raise ValueError('Coarse history requires an explicit state/source transition marker.')
    if not math.isfinite(float(s.t)) or float(s.t) < marker['source_time_myr']:
        raise ValueError('Coarse history cannot precede its declared transition epoch.')


def activate(s, *, source_receipt):
    """Explicit paused-state migration; never loads, evolves, remeshes or saves.

    The caller owns immutable checkpoint/history preservation and any separately
    reviewed coalescence. Observer diagnostics are compacted through its dedicated
    helper, with the original121 evidence referenced rather than relabeled.
    """
    if not isinstance(source_receipt,dict) or not source_receipt:
        raise ValueError('Coarse transition requires immutable source provenance.')
    if normalize(s.config.get('history_mode')) is not None:
        raise ValueError('Coarse history activation is one-shot.')
    for key in ('collision_surface_version','material_mechanics_version'):
        if type(getattr(s,key,None)) is not int or getattr(s,key) != 1:
            raise ValueError('Missing native stack/column mechanics for coarse transition.')
    epoch=float(s.t)
    if not math.isfinite(epoch) or epoch < 0:
        raise ValueError('Invalid coarse transition epoch.')
    import coarse_diagnostics
    report=coarse_diagnostics.compact(s,source_receipt)
    updated=dict(s.config)
    updated.update(history_mode=MODE,deforming_regions=0,adaptive_refinement=0)
    s.config=updated
    s.coarse_history_policy=dict(version=1,mode=MODE,source_time_myr=epoch,
        source_receipt=deepcopy(source_receipt),diagnostic_compaction=deepcopy(report),
        mechanics='finite rigid material-sheet advection with retained overlapping-column support',
        internal_p1_strain_resolved=False,gravitational_relaxation_resolved=False,
        physical_force_laws_changed=False,automatic_material_refinement=False,
        arc_birth_and_explicit_coalescence='retained physical sources and separately reviewed conservative remaps',
        limitations='Internal sheet shortening/thinning and gravitational spreading are unresolved; stacking and conservative column/arc/erosion processes remain active.')
    validate(s)
    return deepcopy(s.coarse_history_policy)


def annotate_snapshot(s, frame):
    if not enabled(s):
        return
    validate(s)
    deformation=frame.get('deformation_diagnostics',{})
    if (not isinstance(deformation,dict) or deformation.get('model') != 'rigid material transport'
            or type(deformation.get('deforming_vertices')) is not int
            or deformation['deforming_vertices'] != 0
            or 'gravitational_relaxation' in deformation):
        raise ValueError('Coarse frame cannot claim a fine contact/gravity solve.')
    frame['history_mode']=MODE
    frame['coarse_history_policy']=deepcopy(s.coarse_history_policy)
    frame['coarse_history_policy']['frame_time_myr']=float(s.t)
    frame['coarse_history_policy']['model_interval_claim']='Only an original completed Simulation.step advances physical time.'
