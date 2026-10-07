"""Read-only emergent-plausibility audit of a saved run.

Component tests and short preflights check that an operator conserves what it
claims. They cannot say whether a 500 Myr world came out looking like a planet.
This reads the saved frames of a finished or running experiment and compares its
emergent behaviour with broad Earth-like bands and with the rules of thumb.

The bands are deliberately wide, and a band is not a calibration target: a world
outside one is a question to investigate, not a failed test. Anything the frames
do not record is reported as unavailable rather than guessed.

    python run_audit.py --run 20260921-011257-b2f85b
    python run_audit.py --run <id> --json audit.json --frames 20

Nothing here writes to the run. See IMPLEMENTATION_GAPS.md G120, C02 and C03.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

EARTH_AREA_KM2 = 510.1e6
# Observed ranges, for orientation. Sources: plate speeds and net rotation from
# present-day plate motions; ocean age from the age grid; boundary lengths from
# the plate-boundary compilation. Each entry is (low, high, note).
BANDS = {
    'mean_plate_speed_cm_yr': (2.0, 7.0, 'Earth about 4-5 cm/yr, area weighted'),
    'fastest_plate_cm_yr': (5.0, 12.0, 'Earth about 8-10 cm/yr (Pacific, Cocos, Nazca)'),
    'net_rotation_deg_myr': (0.1, 0.26, 'observed net lithospheric rotation; two-sided (C03)'),
    'mean_ocean_age_myr': (40.0, 100.0, 'Earth about 64 Myr'),
    'subduction_over_ridge_length': (0.5, 1.6, 'Earth about 50,000 km of trench to 60,000 km of ridge'),
    'ocean_consumed_over_created': (0.7, 1.4, 'a steady planet closes as much ocean as it opens'),
    'unexplained_ocean_fraction': (0.0, 0.15, 'created minus consumed, as a fraction of created (C02)'),
    'active_plates': (8, 45, 'Earth has about 7 major and dozens of smaller plates'),
    'continental_breakups_per_100myr': (0.5, 8.0, 'Wilson cycles keep continents splitting and reassembling'),
    'collisions_per_100myr': (1.0, 20.0, 'collision begins between distinct plate pairs'),
    'accreted_mkm2_per_100myr': (0.05, 10.0, 'terrane collage: arcs and microcontinents dock'),
    'land_fraction': (0.15, 0.45, 'Earth about 0.29'),
    'max_elevation_km': (2.5, 10.0, 'Earth about 8.8 km'),
    'slab_attached_speed_ratio': (1.5, 12.0, 'Rule II: plates with slabs attached move several times faster'),
}


def _verdict(value, band):
    if value is None or not math.isfinite(value):
        return 'unavailable'
    low, high, _ = band
    if low <= value <= high:
        return 'pass'
    # Warn within a factor of two of the band, fail beyond it.
    if value < low:
        return 'warn' if value >= low/2. else 'fail'
    return 'warn' if value <= high*2. else 'fail'


def load_run(run_dir):
    """Return (manifest, frame stats by index) without opening any frame file."""
    run_dir = Path(run_dir)
    manifest = json.loads((run_dir/'manifest.json').read_text(encoding='utf-8'))
    frames = [dict(index=int(row['index']), time_myr=float(row['time_myr']), stats=dict(row.get('stats') or {}))
              for row in manifest.get('frames', [])]
    frames.sort(key=lambda row: row['index'])
    return manifest, frames


def sample_indices(count, wanted):
    """Evenly spaced frame indices, always including the last one."""
    if count <= 0:
        return []
    if wanted >= count:
        return list(range(count))
    step = (count-1)/float(max(wanted-1, 1))
    return sorted({int(round(i*step)) for i in range(wanted)} | {count-1})


def read_frames(run_dir, indices):
    """Read the JSON side of selected frames. Missing frames are skipped."""
    out = {}
    for index in indices:
        path = Path(run_dir)/f'frame_{index:04d}.json'
        if path.is_file():
            out[index] = json.loads(path.read_text(encoding='utf-8'))
    return out


def _rate(frames, key, window_myr):
    """Change in a cumulative counter per 100 Myr over the last window."""
    if len(frames) < 2:
        return None
    last = frames[-1]
    span = [row for row in frames if row['time_myr'] >= last['time_myr']-window_myr]
    if len(span) < 2:
        span = frames[-2:]
    first = span[0]
    elapsed = last['time_myr']-first['time_myr']
    if elapsed <= 0:
        return None
    a, b = first['stats'].get(key), last['stats'].get(key)
    if a is None or b is None:
        return None
    return (float(b)-float(a))/elapsed*100.


def slab_attachment(frame):
    """Mean speed of plates with an attached slab against the rest (Rule II).

    A plate is slab-attached when it is the downgoing side of a trench that is
    past initiation and still carries retained slab mass.
    """
    speeds = (frame.get('plate_balance_diagnostics') or {}).get('mean_speed_cm_yr')
    plates = frame.get('plates')
    if not isinstance(speeds, dict) or not isinstance(plates, list):
        return None
    name_by_uid = {int(row['uid']): row['name'] for row in plates if 'uid' in row and 'name' in row}
    attached = set()
    for row in frame.get('trench_systems') or []:
        if row.get('phase') in ('shutdown', 'joined'):
            continue
        if float(row.get('slab_retained_excess_mass_kg') or 0.) <= 0.:
            continue
        name = name_by_uid.get(int(row.get('downgoing_plate_uid', -1)))
        if name is not None:
            attached.add(name)
    with_slab = [float(v) for k, v in speeds.items() if k in attached]
    without = [float(v) for k, v in speeds.items() if k not in attached]
    if not with_slab or not without:
        return None
    mean_with = sum(with_slab)/len(with_slab)
    mean_without = sum(without)/len(without)
    return dict(attached_plates=len(with_slab), other_plates=len(without),
                attached_mean_cm_yr=mean_with, other_mean_cm_yr=mean_without,
                ratio=(mean_with/mean_without if mean_without > 0 else None))


def audit(run_dir, window_myr=100., frame_count=12):
    run_dir = Path(run_dir)
    manifest, frames = load_run(run_dir)
    if not frames:
        raise ValueError('The run has no saved frames to audit.')
    last = frames[-1]
    stats = last['stats']
    sampled = read_frames(run_dir, sample_indices(len(frames), frame_count))
    latest = sampled.get(last['index'])

    measures = {}

    def put(name, value, detail=None):
        band = BANDS.get(name)
        measures[name] = dict(value=value, verdict=_verdict(value, band) if band else 'reported',
                              band=list(band[:2]) if band else None,
                              reference=band[2] if band else None, detail=detail)

    put('mean_plate_speed_cm_yr', stats.get('mean_plate_speed_cm_yr'))
    put('fastest_plate_cm_yr', stats.get('max_plate_speed_cm_yr'))
    put('mean_ocean_age_myr', stats.get('mean_ocean_age_myr'))
    put('active_plates', stats.get('active_plates'))
    put('land_fraction', stats.get('land_fraction'))
    put('max_elevation_km', (stats.get('max_elevation_m') or 0.)/1000. if stats.get('max_elevation_m') is not None else None)

    ridge, trench = stats.get('ridge_length_km'), stats.get('subduction_length_km')
    put('subduction_over_ridge_length', (trench/ridge) if ridge else None,
        detail=dict(ridge_km=ridge, subduction_km=trench, transform_km=stats.get('transform_length_km'),
                    collision_km=stats.get('collision_length_km')))

    created, consumed = stats.get('ocean_created_km2'), stats.get('ocean_consumed_km2')
    put('ocean_consumed_over_created', (consumed/created) if created else None,
        detail=dict(created_mkm2=(created or 0.)/1e6, consumed_mkm2=(consumed or 0.)/1e6))
    put('unexplained_ocean_fraction', ((created-consumed)/created) if created else None,
        detail=dict(unexplained_mkm2=((created or 0.)-(consumed or 0.))/1e6,
                    note='cumulative created minus measured subduction; the rest leaves through support '
                         'normalization and hinge transfer, which the frames do not separate (C02)'))

    put('continental_breakups_per_100myr', _rate(frames, 'rift_events_breakup', window_myr))
    put('collisions_per_100myr', _rate(frames, 'collision_events', window_myr))
    accreted = _rate(frames, 'accreted_km2', window_myr)
    put('accreted_mkm2_per_100myr', (accreted/1e6) if accreted is not None else None)

    net_rotation = None
    if latest:
        balance = latest.get('plate_balance_diagnostics') or {}
        net_rotation = balance.get('net_rotation_deg_myr')
    put('net_rotation_deg_myr', net_rotation,
        detail=dict(bound=(latest or {}).get('plate_balance_diagnostics', {}).get('net_rotation_bound_deg_myr'),
                    note='the drag calibration claims this bound pins it; a run far below it is not pinned (C03)'))

    attachment = slab_attachment(latest) if latest else None
    put('slab_attached_speed_ratio', (attachment or {}).get('ratio'), detail=attachment)

    plate_counts = [row['stats'].get('active_plates') for row in frames if row['stats'].get('active_plates')]
    merges = sum(1 for a, b in zip(plate_counts, plate_counts[1:]) if b is not None and a is not None and b < a)
    measures['plate_count_decreases'] = dict(
        value=merges, verdict='pass' if merges else 'warn', band=None,
        reference='plates should sometimes merge or be absorbed after collision',
        detail=dict(first=plate_counts[0] if plate_counts else None, last=plate_counts[-1] if plate_counts else None))

    verdicts = [row['verdict'] for row in measures.values()]
    return dict(
        run_id=manifest.get('run_id'), state=manifest.get('state'),
        time_myr=last['time_myr'], frames=len(frames), window_myr=window_myr,
        sampled_frames=sorted(sampled),
        summary=dict(passed=verdicts.count('pass'), warned=verdicts.count('warn'),
                     failed=verdicts.count('fail'), unavailable=verdicts.count('unavailable')),
        measures=measures,
        caveats=['Bands are broad plausibility ranges for orientation, not calibration targets.',
                 'Statistics come from saved frames; a running experiment is audited as far as it has been saved.',
                 'Per-frame diagnostics record the last accepted substep, so per-step quantities are samples.'])


def format_report(result):
    lines = [f"Run {result['run_id']} - {result['state']} at {result['time_myr']:g} Myr, "
             f"{result['frames']} frames (rates over the last {result['window_myr']:g} Myr)", '']
    mark = {'pass': 'ok  ', 'warn': 'WARN', 'fail': 'FAIL', 'unavailable': '--  ', 'reported': '    '}
    width = max(len(name) for name in result['measures'])
    for name, row in result['measures'].items():
        value = row['value']
        shown = 'unavailable' if value is None else (f'{value:,.3g}' if isinstance(value, float) else str(value))
        band = f"[{row['band'][0]:g}, {row['band'][1]:g}]" if row['band'] else ''
        lines.append(f"  {mark[row['verdict']]} {name:<{width}}  {shown:>12}  {band:<14} {row['reference'] or ''}")
    s = result['summary']
    lines += ['', f"  {s['passed']} within band, {s['warned']} outside, {s['failed']} far outside, "
                  f"{s['unavailable']} unavailable"]
    return '\n'.join(lines)


def main(argv=None):
    parser = argparse.ArgumentParser(description='Audit a saved run against broad plausibility bands.')
    parser.add_argument('--run', required=True, help='run id under output/runs, or a path to a run directory')
    parser.add_argument('--window', type=float, default=100., help='window in Myr for per-100-Myr rates')
    parser.add_argument('--frames', type=int, default=12, help='how many frames to open for per-frame diagnostics')
    parser.add_argument('--json', help='also write the full result to this path')
    args = parser.parse_args(argv)
    run_dir = Path(args.run)
    if not run_dir.is_dir():
        run_dir = Path(__file__).resolve().parent/'output'/'runs'/args.run
    result = audit(run_dir, window_myr=args.window, frame_count=args.frames)
    print(format_report(result))
    if args.json:
        Path(args.json).write_text(json.dumps(result, indent=1), encoding='utf-8')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
