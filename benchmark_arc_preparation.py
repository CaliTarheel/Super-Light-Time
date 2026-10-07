"""Compare immutable baseline modules and candidate preparation on identical inputs.

Run with single-thread BLAS, --baseline CHECKOUT [--frame SAVED_FRAME.npz].
Results are microbenchmarks, not a prediction of complete timestep speedup.
"""
import argparse
import importlib.util
import json
from pathlib import Path
import statistics
import timeit
from unittest.mock import patch
import numpy as np
import arc_surface
import arc_birth_profile
import material_reconstruction
import native_arc_material


def load(root, name):
    spec = importlib.util.spec_from_file_location('baseline_'+name, Path(root)/(name+'.py'))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def equal(a, b):
    if isinstance(a, dict):
        assert a.keys() == b.keys()
        for key in a: equal(a[key], b[key])
    else:
        np.testing.assert_array_equal(a, b)


def compare(label, before, after, number=100):
    equal(before(), after())
    times = [[], []]
    for round_index in range(7):
        for index in ((0, 1) if round_index % 2 == 0 else (1, 0)):
            times[index].append(timeit.timeit((before, after)[index], number=number)/number)
    old, new = map(statistics.median, times)
    return dict(case=label, baseline_seconds=old, candidate_seconds=new,
                speedup=old/new, exact_outputs=True, repetitions=number, rounds=7)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--baseline', required=True)
    parser.add_argument('--frame')
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    old_reconstruction = load(args.baseline, 'material_reconstruction')
    old_surface = load(args.baseline, 'arc_surface')
    rows = []
    geom = native_arc_material._patch([1., 0., 0.], [0., 1., 0.], 10000.)
    born = arc_birth_profile.birth(geom, 10000., -5500.)
    values = (geom['vertices'], geom['faces'], born['area_km2'], born['height_m'], -5500.)
    def measure_with(reconstruction, surface):
        with patch.object(arc_birth_profile, 'material_reconstruction', reconstruction), patch.object(arc_birth_profile, 'arc_surface', surface):
            return arc_birth_profile.measure(*values)
    rows.append(compare('24-face arc profile measure',
        lambda: measure_with(old_reconstruction, old_surface),
        lambda: measure_with(material_reconstruction, arc_surface), 300))
    if args.frame:
        with np.load(args.frame, allow_pickle=False) as z:
            vertices, faces = z['material_vertices'], z['material_faces']
            owners, ids = z['material_owner'], z['material_arc_id']
            areas = z['material_actual_area_km2']
        stencil = material_reconstruction.prepare(vertices, faces, owners, area_km2=areas)
        rows.append(compare('saved frame arc boundaries', lambda: old_surface.prepare(stencil, ids), lambda: arc_surface.prepare(stencil, ids), 10))
        for label, owner in [('mixed owners', owners), ('one owner', np.zeros(len(faces), int))]:
            rows.append(compare('saved frame stencil '+label,
                lambda: old_reconstruction.prepare(vertices, faces, owner, area_km2=areas),
                lambda: material_reconstruction.prepare(vertices, faces, owner, area_km2=areas), 10))
    result = dict(numpy=np.__version__, baseline=args.baseline, frame=args.frame, results=rows,
                  scope='Exact preparation/profile outputs only; no full-step speedup measured.')
    Path(args.output).write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))


if __name__ == '__main__': main()
