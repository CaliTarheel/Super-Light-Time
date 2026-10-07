"""Run the baseline matrix and preserve failed scientific comparison gates.

python -m benchmarks.collision_architecture.study --output /chosen/directory
Exit status 1 means at least one declared gate failed, not that evidence is lost.
"""
import argparse
from pathlib import Path

import numpy as np

from . import runner


def measures(report):
    rows = report['rows']
    initial, final = rows[0], rows[-1]
    # Equal initial per-body volumes in this fixture: this is also volume weighted.
    mean_h = lambda row: np.mean([x['mean_thickness_km'] for x in row['sides']])
    passive_j = sum((b['time_myr']-a['time_myr'])*runner.balance.SECONDS_PER_MYR*
                   sum(a['resisting_power_w'].values()) for a, b in zip(rows, rows[1:]))
    bracket = next(([a['time_myr'], b['time_myr']] for a, b in zip(rows, rows[1:])
                    if a['overlap_km2'] == 0. and b['overlap_km2'] > 0.), None)
    return dict(
        thickness_increment_km=float(mean_h(final)-mean_h(initial)),
        accumulated_compression=float(np.mean([.5*(x['accumulated_absolute_log_area']-x['signed_log_area']) for x in final['sides']])),
        maximum_displacement_km=final['maximum_displacement_km'],
        passive_power_left_quadrature_j=passive_j, first_contact_bracket_myr=bracket,
        max_thickness_km=final['max_thickness_km'], overlap_km2=final['overlap_km2'],
        maximum_local_volume_relative_error=max(r['maximum_local_volume_relative_error'] for r in rows),
        maximum_plate_relative_residual=max(r['plate_relative_residual'] for r in rows),
        limited_face_step_count=sum((r['last_step'] or {}).get('limited_faces', 0) for r in rows))


def refinement(coarse, medium, fine):
    floors = dict(thickness_increment_km=runner.TOL['refinement_thickness_floor_km'],
                  accumulated_compression=1e-8,
                  maximum_displacement_km=runner.TOL['refinement_displacement_floor_km'],
                  passive_power_left_quadrature_j=0.)
    result = {}
    for key, floor in floors.items():
        old = abs(coarse[key]-medium[key])
        error = abs(medium[key]-fine[key])
        limit = max(floor, runner.TOL['refinement_rtol']*abs(fine[key]))
        decreasing = error < old or max(error, old) <= floor
        result[key] = dict(coarse_medium_difference=old, medium_fine_difference=error,
                           allowed_difference=limit, decreases=bool(decreasing),
                           pass_gate=bool(error <= limit and decreasing))
    return result


def swapped_fields(first, second):
    """Compare corresponding material, after the same proper rotation as fixture."""
    nv = len(first.material_surface['vertices'])//2
    nf = len(first.structure['thickness_km'])//2
    mapped = np.r_[second.material_surface['vertices'][nv:], second.material_surface['vertices'][:nv]]@runner.MIRROR
    original = first.material_surface['vertices']
    angle = np.arctan2(np.linalg.norm(np.cross(mapped, original), axis=1), np.einsum('ij,ij->i', mapped, original))
    thickness = np.r_[second.structure['thickness_km'][nf:], second.structure['thickness_km'][:nf]]
    result = dict(maximum_position_difference_rad=float(angle.max()),
                  maximum_thickness_difference_km=float(np.max(np.abs(thickness-first.structure['thickness_km']))))
    result['pass_gate'] = result['maximum_position_difference_rad'] <= runner.TOL['swap_position_rad'] and result['maximum_thickness_difference_km'] <= runner.TOL['swap_thickness_km']
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    reports, states = {}, {}
    summary = dict(schema=1, source=runner.source_manifest(), tolerances=runner.TOL,
                   complete=False, collision_validated=False, runs={})
    cases = [('rest', 1.), ('symmetric', 1.), ('symmetric', .5), ('symmetric', .25),
             ('weak0', 1.), ('weak1', 1.)]
    for case, dt in cases:
        name = f'{case}-dt{dt:g}'
        state = runner.fixture(case)
        report = runner.run_case(state, dt=dt, end=12., output=args.output/name)
        reports[name], states[name] = report, state
        summary['runs'][name] = dict(complete=report['complete'], numerical_gates_pass=report['numerical_gates_pass'],
            contact_coverage=report['contact_coverage'], wall_seconds=report['wall_seconds'], error=report['error'],
            measures=measures(report) if report['complete'] else None)
        runner.atomic_json(args.output/'summary.json', summary)
        print(name, 'complete:', report['complete'], 'numerical:', report['numerical_gates_pass'], flush=True)
    if not all(r['complete'] and r['numerical_gates_pass'] for r in reports.values()):
        return 1
    summary['refinement'] = refinement(*(summary['runs'][f'symmetric-dt{dt:g}']['measures'] for dt in (1., .5, .25)))
    summary['symmetric_covariance'] = swapped_fields(states['symmetric-dt1'], states['symmetric-dt1'])
    summary['strength_swap_covariance'] = swapped_fields(states['weak0-dt1'], states['weak1-dt1'])
    base = states['symmetric-dt1'].structure['thickness_km']
    changed = states['weak0-dt1'].structure['thickness_km']
    difference = float(np.max(np.abs(changed-base)))
    summary['strength_response'] = dict(maximum_thickness_difference_km=difference, pass_gate=difference > runner.TOL['swap_thickness_km'])
    summary['complete'] = True
    summary['comparison_gates_pass'] = bool(all(v['pass_gate'] for v in summary['refinement'].values()) and
        summary['symmetric_covariance']['pass_gate'] and summary['strength_swap_covariance']['pass_gate'] and
        summary['strength_response']['pass_gate'] and all(r['contact_coverage'] for k, r in reports.items() if not k.startswith('rest')))
    runner.atomic_json(args.output/'summary.json', summary)
    print('Comparison gates pass:', summary['comparison_gates_pass'], flush=True)
    return 0 if summary['comparison_gates_pass'] else 1


if __name__ == '__main__':
    raise SystemExit(main())
