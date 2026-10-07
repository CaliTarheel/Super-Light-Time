"""Exact-root goSPL forcing carries physical collision support once."""
from copy import deepcopy
import hashlib
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

import numpy as np

import collision_contacts
import collision_surface
import crustal_structure
import gospl_export as export
import native_frame_sampling as sampling
from tests.test_material_transport import query, refine, tracked
from tests.test_native_gospl_sampling import write_frame
from validate_gospl import validate_package


def supported(source=None, support=(200., 600.)):
    """Saved physical support is a separate datum from the rock columns."""
    first = tracked(source)
    count = len(first['material_faces'])
    first.update(collision_contact_version=1, collision_surface_version=1,
        collision_contacts=[], material_collision_sheet=np.ones(count, np.int64),
        material_exposed_fraction=np.ones(count), material_burial_myr=np.zeros(count),
        material_collision_suture=np.zeros(count), arc_material_version=1, arc_surface_version=2,
        material_arc_id=np.zeros(count, np.int64), material_arc_basal_m=np.zeros(count),
        material_collision_support_m=np.broadcast_to(support, (count,)).copy())
    first['material_collision_load_thickness_km'] = first['material_collision_support_m']/crustal_structure.AIRY_M_PER_KM
    return first


def refined(first):
    second, transfer = refine(first)
    for name in (*collision_contacts.ARRAY_FIELDS, *collision_surface.ARRAY_FIELDS,
                 'material_arc_id', 'material_arc_basal_m'):
        # Dense-load support is optional in histories predating that policy.
        if name in first: second[name] = first[name][transfer['source_face']]
    second['time_myr'] = 2.
    return second, transfer


class GoSPLCollisionSupportTests(unittest.TestCase):
    def test_pure_remesh_inherits_root_support_without_extra_uplift(self):
        first = supported()
        second, _ = refined(first)
        self.assertGreater(len(second['material_faces']), len(first['material_faces']))
        faces = np.array([0, 0, 1, 1])
        points = query(first, faces, np.array([[.2,.2,.6], [.6,.3,.1], [.7,.2,.1], [.1,.3,.6]]))
        fields, info = export.interval_forcing(points, first, second, {})
        np.testing.assert_array_equal(fields['collision_support_rate'], 0.)
        np.testing.assert_array_equal(fields['geometric_rate'], 0.)
        np.testing.assert_array_equal(fields['upsub'], 0.)
        np.testing.assert_array_equal(fields['relaxation_correction'], 0.)
        np.testing.assert_allclose(fields['hdisp'], 0., atol=1e-14)
        self.assertGreater(np.abs(fields['display_geometric_rate']).max()*2e6, 1.)
        np.testing.assert_array_equal(fields['display_geometric_rate'], fields['reconstruction_difference_rate'])
        self.assertEqual(info['collision_surface_version'], 1)
        self.assertEqual(info['collision_support_change_rms_m'], 0.)

    def test_support_change_is_counted_once_and_erosion_correction_is_unchanged(self):
        first = supported()
        second = deepcopy(first)
        second['time_myr'] = 2.
        support_change, erosion = np.array([120., -80.]), np.array([15., 30.])
        second['material_collision_support_m'] += support_change
        second['material_collision_load_thickness_km'] = second['material_collision_support_m']/crustal_structure.AIRY_M_PER_KM
        second['material_height_m'] -= erosion
        second['material_erosion_total_m'] += erosion
        points = np.vstack((query(first, np.array([0, 1]), np.full((2, 3), 1/3)), [0., 0., -1.]))
        fields, info = export.interval_forcing(points, first, second, {})
        np.testing.assert_allclose(fields['collision_support_rate'][:2], support_change/2e6, rtol=2e-7)
        np.testing.assert_allclose(fields['geometric_rate'][:2], (support_change-erosion)/2e6, rtol=2e-7)
        np.testing.assert_allclose(fields['relaxation_correction'][:2], erosion/2e6, rtol=2e-7)
        np.testing.assert_allclose(fields['upsub'][:2], support_change/2e6, rtol=2e-7)
        np.testing.assert_allclose(fields['upsub'], fields['geometric_rate']+fields['relaxation_correction'], atol=1e-11)
        self.assertEqual(fields['collision_support_rate'][2], 0.)
        np.testing.assert_array_equal(fields['correction_supported'], [1, 1, 0])
        self.assertEqual(info['relaxation_fallback_nodes'], 0)
        self.assertIn('included once in geometric_rate', info['collision_support_forcing'])

    def test_support_change_follows_root_descendants_with_new_face_ids(self):
        first = supported()
        second, transfer = refined(first)
        change = np.array([75., -125.])
        second['material_collision_support_m'] += change[transfer['source_face']]
        second['material_collision_load_thickness_km'] = second['material_collision_support_m']/crustal_structure.AIRY_M_PER_KM
        faces = np.array([0, 0, 1, 1])
        points = query(first, faces, np.array([[.2,.2,.6], [.6,.3,.1], [.7,.2,.1], [.1,.3,.6]]))
        fields, _ = export.interval_forcing(points, first, second, {})
        np.testing.assert_allclose(fields['collision_support_rate'], change[faces]/2e6, rtol=2e-7)
        np.testing.assert_allclose(fields['upsub'], change[faces]/2e6, rtol=2e-7)
        np.testing.assert_array_equal(fields['relaxation_correction'], 0.)

    def test_mixed_collision_surface_versions_are_rejected_in_both_directions(self):
        for first_new in (False, True):
            first, second = supported(), supported()
            second['time_myr'] = 2.
            old = second if first_new else first
            for name in ('collision_surface_version', *collision_surface.ARRAY_FIELDS):
                old.pop(name, None)
            with self.subTest(first_new=first_new), self.assertRaisesRegex(ValueError, 'mix collision surface versions'):
                export.interval_forcing(np.array([[1., 0., 0.]]), first, second, {})

    def test_legacy_unflagged_forcing_keeps_its_components_and_metadata(self):
        first = tracked()
        second = deepcopy(first)
        second['time_myr'] = 2.
        second['material_height_m'] += 80.
        second['material_erosion_total_m'] += 10.
        points = query(first, np.array([0, 1]), np.full((2, 3), 1/3))
        fields, info = export.interval_forcing(points, first, second, {})
        np.testing.assert_allclose(fields['geometric_rate'], 80./2e6, rtol=2e-7)
        np.testing.assert_allclose(fields['relaxation_correction'], 10./2e6, rtol=2e-7)
        np.testing.assert_allclose(fields['upsub'], 90./2e6, rtol=2e-7)
        self.assertNotIn('collision_support_rate', fields)
        self.assertNotIn('collision_surface_version', info)
        self.assertNotIn('collision_support_forcing', info)

    def test_actual_package_records_diagnostic_versions_and_standalone_source_closure(self):
        first = supported(support=200.)
        second = deepcopy(first)
        second['time_myr'] = 2.
        second['material_collision_support_m'] += 220.
        second['material_collision_load_thickness_km'] = second['material_collision_support_m']/crustal_structure.AIRY_M_PER_KM
        with tempfile.TemporaryDirectory() as temporary:
            folder = Path(temporary)
            run, output = folder/'run', folder/'export'
            run.mkdir()
            for index, source in enumerate((first, second)):
                write_frame(run, index, source)
            manifest = dict(run_id='collision-support-forcing', config={'dt_myr':2.},
                frames=[{'time_myr':0.}, {'time_myr':2.}])
            meta = export.build_history(run, output, manifest, subdivisions=2, dt_years=100000)
            self.assertTrue(validate_package(output)['passed'])
            self.assertEqual(meta['collision_surface_version'], 1)
            self.assertIn('collision_support_rate', meta['units'])
            for row in meta['source_frames']:
                self.assertEqual(row['surface_reconstruction']['collision_contact_version'], 1)
                self.assertEqual(row['surface_reconstruction']['collision_surface_version'], 1)
            self.assertEqual(hashlib.sha256((output/'source'/'collision_surface.py').read_bytes()).hexdigest(),
                meta['native_sampling_sources_sha256']['collision_surface.py'])
            with np.load(output/'input'/'mesh.npz') as mesh:
                sample = sampling.sample_frame(first, mesh['v']/export.RADIUS_M)
                np.testing.assert_array_equal(mesh['z'], sample['elevation'].astype(np.float32))
            with np.load(output/'input'/'forcing_0000.npz') as data:
                land = sample['material_face'] >= 0
                self.assertTrue(land.any())
                np.testing.assert_allclose(data['collision_support_rate'][land], 220./2e6, rtol=2e-7)
                np.testing.assert_array_equal(data['collision_support_rate'][~land], 0.)
                np.testing.assert_allclose(data['upsub'][land], 220./2e6, rtol=2e-7)
                np.testing.assert_allclose(data['upsub'], data['geometric_rate']+data['relaxation_correction'], atol=1e-11)
            self.assertIn('not added again to `upsub`', (output/'README.md').read_text(encoding='utf-8'))
            for index, source in enumerate((first, second)):
                write_frame(output/'source', index, source)
            # Exercise only captured sources, without the workspace on sys.path.
            program = """import sys;sys.path.insert(0,'.')
import numpy as np,gospl_export as e
a=e.load_frame('.',0);b=e.load_frame('.',1)
fields,info=e.interval_forcing(np.array([[1.,0.,0.]]),a,b,{})
np.testing.assert_allclose(fields['collision_support_rate'],220./2e6,rtol=2e-7)
np.testing.assert_allclose(fields['upsub'],220./2e6,rtol=2e-7)
assert info['collision_surface_version']==1
assert 'collision_surface.py' in e.NATIVE_SAMPLING_SOURCES
"""
            subprocess.run([sys.executable, '-I', '-c', program], cwd=output/'source',
                check=True, capture_output=True, text=True)


if __name__ == '__main__':
    unittest.main()
