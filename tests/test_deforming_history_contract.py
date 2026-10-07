"""Versioned deforming epochs retain fields through public persistence APIs."""
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

import numpy as np

import gospl_export
import mesh_history
import server
from orientation import inverse_cells, rotation_matrix
try:
    from .test_material_transport import complete_schema, tracked, refine
except ImportError:
    from test_material_transport import complete_schema, tracked, refine


def saved_surface():
    source = complete_schema(tracked())
    for key in list(source):
        if key.startswith('trace_'): del source[key]
    n = source['width']*source['height']; m = len(source['material_faces'])
    source['age'] = np.zeros(n, np.float32)
    source['crust'][:4] = 1
    source['deformation_weight'] = np.full(n, -1., np.float32)
    source['deformation_weight'][:4] = [0., .25, .75, 1.]
    source['geometric_strain_percent'] = np.zeros(n, np.float32)
    source['geometric_strain_percent'][:4] = [-2., 0., 1., 4.]
    source['refinement_level'] = np.full(n, -1., np.float32)
    source['refinement_level'][:4] = np.arange(4)
    source.update(material_parent_id=np.full(m, -1, np.int64),
                  material_refinement_level=np.zeros(m, np.int32),
                  material_actual_area_km2=np.ones(m), material_deformation_weight=np.array([.25, 1.]),
                  material_rigid=np.zeros(m, np.uint8), material_geometric_log_area=np.array([-.02, .03]))
    return source


class DeformingHistoryContractTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory(); self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.manager = server.SimulationManager(self.root/'runs')
        self.manager.current = dict(run_id='synthetic-deforming', frame_count=1, frames=[dict(time_myr=0.)], config={})
        self.manager.path().mkdir(parents=True)

    def test_server_save_native_reader_and_oriented_review_retain_versioned_fields(self):
        source = saved_surface(); self.manager.save_frame(0, source)
        stored = gospl_export.load_frame(self.manager.path(), 0)
        native = self.manager.native_frame(0)
        for name in set(mesh_history.DTYPES).intersection(source):
            np.testing.assert_array_equal(native[name], source[name], err_msg=name)
            np.testing.assert_array_equal(stored[name], source[name], err_msg=name)
        lean = self.manager.frame(0)
        self.assertFalse(set(mesh_history.DTYPES).intersection(lean))
        self.assertEqual(lean['material_transport_version'], 1)
        for name in ('deformation_weight', 'geometric_strain_percent', 'refinement_level'):
            np.testing.assert_array_equal(lean[name], source[name])
        angles = dict(yaw=42., pitch=65., roll=-17.); matrix = rotation_matrix(angles)
        oriented = self.manager.native_frame(0, angles)
        for name in ('material_root_id', 'material_parent_id', 'material_reference_corners', 'material_erosion_total_m'):
            np.testing.assert_array_equal(oriented[name], source[name], err_msg=name)
        np.testing.assert_allclose(oriented['material_vertices'], source['material_vertices']@matrix, atol=2e-15)
        at = inverse_cells(np.arange(source['width']*source['height']), source['width'], source['height'], angles)
        np.testing.assert_array_equal(oriented['refinement_level'], source['refinement_level'][at])
        self.assertTrue(np.isfinite(oriented['geometric_strain_percent']).all())
        self.assertTrue(set(np.unique(oriented['refinement_level'])).issubset({-1., 0., 1., 2., 3.}))

    def test_invalid_deformation_review_bounds_fail_before_overwriting_saved_frame(self):
        source = saved_surface(); self.manager.save_frame(0, source)
        baseline = (self.manager.path()/'frame_0000.npz').read_bytes()
        invalid = [('deformation_weight', 0, 1.1), ('deformation_weight', 0, -.5),
                   ('deformation_weight', 0, -1.), ('deformation_weight', 9, 0.),
                   ('refinement_level', 0, .5), ('refinement_level', 0, 4.),
                   ('refinement_level', 0, -1.), ('refinement_level', 9, 0.),
                   ('geometric_strain_percent', 0, -100.), ('geometric_strain_percent', 0, np.nan)]
        for name, cell, value in invalid:
            with self.subTest(name=name, cell=cell, value=value):
                bad = deepcopy(source); bad[name][cell] = value
                with self.assertRaises(ValueError): self.manager.save_frame(0, bad)
                self.assertEqual((self.manager.path()/'frame_0000.npz').read_bytes(), baseline)

    def test_versioned_chart_fields_survive_actual_saved_rotated_package(self):
        first = saved_surface(); second, _ = refine(first); second['time_myr'] = 2.
        # Synthetic refinement helper only transfers physical export columns;
        # complete the independent diagnostic arrays for server validation.
        m = len(second['material_faces'])
        for name in set(mesh_history.DTYPES).intersection(second):
            if name.startswith('material_') and name not in ('material_vertices', 'material_faces', 'material_reference_corners'):
                if len(second[name]) != m:
                    second[name] = np.resize(second[name], m)
        self.manager.current.update(frame_count=2, frames=[dict(time_myr=0.), dict(time_myr=2.)])
        self.manager.save_frame(0, first); self.manager.save_frame(1, second)
        meta = gospl_export.build_history(self.manager.path(), self.root/'gospl', self.manager.current,
            subdivisions=1, dt_years=100000, orientation=dict(yaw=42., pitch=65., roll=-17.))
        self.assertEqual(meta['material_transport_version'], 1)
        self.assertTrue(json.loads((self.root/'gospl'/'validation.json').read_text())['passed'])
        self.assertTrue({'material_transport.py', 'arc_surface.py'}.issubset(meta['native_sampling_sources_sha256']))

    def test_captured_engine_and_exporter_import_closures_include_deformation_helpers(self):
        closure = server.capture_auxiliary_sources(server.ENGINE_SOURCE)
        needed = {'native_material_evolution.py', 'native_material_adaptivity.py', 'deforming_regions.py',
                  'adaptive_material.py', 'material_transport.py', 'native_frame_sampling.py'}
        self.assertTrue(needed.issubset(closure), needed-set(closure))
        captured = self.root/'source'; captured.mkdir()
        (captured/'tectonics.py').write_bytes(server.ENGINE_SOURCE)
        for name, payload in closure.items(): (captured/name).write_bytes(payload)
        # Import from the frozen directory only; no live application paths and
        # no model trajectory is needed to verify the complete source closure.
        subprocess.run([sys.executable, '-I', '-c',
            "import sys; sys.path.insert(0, '.'); import tectonics, native_material_adaptivity, material_transport"],
            cwd=captured, check=True, capture_output=True, text=True)
        self.assertEqual(hashlib.sha256((captured/'material_transport.py').read_bytes()).hexdigest(),
                         hashlib.sha256(gospl_export.NATIVE_SAMPLING_SOURCES['material_transport.py']).hexdigest())


if __name__ == '__main__': unittest.main()
