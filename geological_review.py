"""Build a reproducible basin and mountain review from existing saved frames.

This is an offline review command. It neither constructs a simulation nor
changes a run. All distances follow explicitly selected spherical paths.
"""
from __future__ import annotations

import argparse
import datetime
import json
import math
import sys
from pathlib import Path

import numpy as np

from review_history_io import digest, load_history, verify_history


DEFAULT_SELECTION = {
    'transects': [
        {'id': 'northwest-trench', 'waypoints_lon_lat': [[-135., 20.], [-60., 20.]]},
        {'id': 'southern-collision', 'waypoints_lon_lat': [[-125., -49.], [-80., -49.]]},
    ],
    'marker_ids': [],
    'mountain_samples_per_transect': 63,
    'sample_spacing_km': 25.,
}


def xyz(lon_lat):
    if len(lon_lat) != 2:
        raise ValueError('Each waypoint requires longitude and latitude.')
    lon, lat = map(float, lon_lat)
    if not math.isfinite(lon) or not math.isfinite(lat) or not -90 <= lat <= 90:
        raise ValueError('Waypoint coordinates must be finite and latitude within [-90,90].')
    lon, lat = np.radians([lon, lat])
    return [float(np.cos(lat)*np.cos(lon)), float(np.cos(lat)*np.sin(lon)), float(np.sin(lat))]


def choose_nearby_markers(frames, selection):
    """Choose actual saved IDs near declared locations, retaining the distance.

    This is a marker selection aid, never a claim that the nearest marker is a
    summit or represents every material sheet at that location.
    """
    chosen = list(selection.get('marker_ids', []))
    records = []
    for request in selection.get('marker_locations', []):
        epoch = float(request['time_myr'])
        matches = [f for f in frames if float(f['time_myr']) == epoch]
        if len(matches) != 1:
            raise ValueError('A marker-selection epoch must be among the loaded saved frames.')
        frame = matches[0]
        positions = np.asarray(frame.get('trace_xyz', []), float)
        ids = np.asarray(frame.get('trace_id', []))
        if (positions.ndim != 2 or positions.shape[1:] != (3,) or not len(positions)
                or ids.shape != (len(positions),) or not np.isfinite(positions).all()
                or not np.allclose(np.linalg.norm(positions, axis=1), 1., atol=1e-6)):
            raise ValueError('Saved material marker positions and identities are required.')
        point = np.asarray(xyz(request['lon_lat']))
        angles = np.arctan2(np.linalg.norm(np.cross(positions, point), axis=1), positions@point)
        index = int(np.argmin(angles))
        uid = int(ids[index])
        chosen.append(uid)
        records.append(dict(time_myr=epoch, requested_lon_lat=request['lon_lat'],
                            trace_id=uid, selection_distance_km=float(angles[index]*6371.),
                            meaning='Nearest saved marker; not an exact summit or full crustal-stack sample.'))
    return sorted(set(chosen)), records


def build_report(loaded, selection):
    here = Path(__file__).resolve().parent
    source_hashes = {p.name: digest(p) for p in here.glob('*.py')}

    def verify_analysis_sources():
        if {p.name: digest(p) for p in here.glob('*.py')} != source_hashes:
            raise ValueError('Analysis source changed during review; use a settled candidate.')
        for name in source_hashes:
            module = sys.modules.get(Path(name).stem)
            if module is not None and Path(getattr(module, '__file__', '')).resolve() != here/name:
                raise ValueError(f'Analysis helper was imported from another source directory: {name}')

    import basin_history_review
    import mountain_history_review
    verify_analysis_sources()

    transects = selection['transects']
    if len(transects) > 2:
        raise ValueError('This compact combined review accepts at most two transects.')
    samples = selection.get('mountain_samples_per_transect', 63)
    if type(samples) is not int or samples < 3 or samples*len(transects) > 128:
        raise ValueError('Choose at most128 total mountain-profile samples per epoch.')
    paths = []
    for transect in transects:
        waypoints = transect['waypoints_lon_lat']
        if len(waypoints) != 2:
            raise ValueError('Combined mountain profiles require two-ended minor great-circle paths.')
        paths.append(dict(name=transect['id'], start_xyz=xyz(waypoints[0]), end_xyz=xyz(waypoints[1]),
                          samples=samples))
    marker_ids, marker_selection = choose_nearby_markers(loaded['frames'], selection)
    basin = basin_history_review.review_frames(loaded['frames'], transects,
                committed_manifest=loaded['manifest'],
                sample_spacing_km=selection.get('sample_spacing_km', 25.))
    mountains = mountain_history_review.review_frames(loaded['frames'], transects=paths,
                                                       marker_ids=marker_ids)
    verify_history(loaded)
    verify_analysis_sources()
    return dict(created_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
                provenance=loaded['provenance'], selection=selection, marker_selection=marker_selection,
                analysis_source_sha256=source_hashes,
                numpy_version=np.__version__, basins=basin, mountains=mountains,
                scope='Read-only saved-history review. Geographic profiles and material trajectories are separate observations.',
                limitations=['Sparse exact epochs do not establish transition times between snapshots.',
                             'A selected profile is not a global extremum search or complete geologic basin identity.',
                             'No procedural terrain detail, new physics, or integration is introduced.'])


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('run_path', type=Path)
    parser.add_argument('--epochs', default='0,50,latest', help='Comma-separated exact committed epochs or latest.')
    parser.add_argument('--selection', type=Path, help='JSON defining explicit transects and optional marker IDs/locations.')
    parser.add_argument('--output', required=True, type=Path, help='New report directory outside the source run.')
    args = parser.parse_args(argv)
    output = args.output.resolve()
    run = args.run_path.resolve()
    if output == run or run in output.parents:
        parser.error('Write review artifacts outside the preserved source run.')
    if output.exists():
        parser.error('Use a new output directory to preserve prior review artifacts.')
    selection = json.loads(args.selection.read_text(encoding='utf-8-sig')) if args.selection else DEFAULT_SELECTION
    loaded = load_history(run, [x.strip() for x in args.epochs.split(',')])
    report = build_report(loaded, selection)
    encoded = json.dumps(report, indent=2, allow_nan=False) + '\n'
    output.mkdir(parents=True)
    (output/'review.json').write_text(encoded, encoding='utf-8')
    (output/'selection.json').write_text(json.dumps(selection, indent=2)+'\n', encoding='utf-8')
    from geological_review_report import write_markdown
    write_markdown(report, output)
    print(json.dumps(dict(report=str(output/'review.json'),
                          run_id=loaded['provenance']['run_id'],
                          epochs=loaded['provenance']['selected_epochs_myr'])))


if __name__ == '__main__':
    main()
