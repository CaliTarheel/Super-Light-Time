"""Independent free-edge emergence tests beneath another material margin."""
from copy import deepcopy
from types import SimpleNamespace
import unittest
import numpy as np

import collision_contacts
import collision_surface
import material_surface
import mesh_coverage
import native_frame_sampling as sampling
import structure_engine
from orientation import orient_frame, rotation_matrix
from tests.test_collision_contacts import rectangle
from tests.test_collision_surface_acceptance import saved, unit

R = 6371.


def layer_fixture(*, triple=False, lower_arc=False, upper_arc=False):
    # The upper sheet's right coast is50km from the lower sheet's left edge.
    # Crossing x=0 enters lower material while staying in the SAME upper face
    # inside its finite continental margin. Optionalthird lower layer shares
    # that entry and remains reachable through the transitive sheet order.
    patches = [rectangle(-800., 50., 0, nx=7, ny=5, kind=3 if upper_arc else 1),
               rectangle(0., 800., 1, nx=7, ny=5, kind=3 if lower_arc else 1)]
    if triple:
        patches.append(rectangle(0., 600., 2, nx=7, ny=5))
    surface = material_surface.initialize_surface(*patches[0])
    for patch in patches[1:]: material_surface.append_surface(surface, *patch)
    n = len(surface['faces']); owner = surface['face_owner']
    s = SimpleNamespace(collision_surface_version=1, material_surface=surface,
        parcel_patch=surface['face_id'].copy(), parcel_plate=owner.copy(), kind=surface['face_kind'].copy(),
        mass=surface['reference_area_km2'].copy(), pos=material_surface.face_centres(surface),
        relief=np.choose(owner, [1100., 900., 600.])-220., suture=np.zeros(n),
        material_lineage=dict(root_id=np.arange(n)+1000), t=0.,
        plate_uid=np.arange(len(patches))+101, trace_xyz=np.empty((0, 3)), trace_patch=np.empty(0, np.int64))
    structure_engine.initialize_parcels(s)
    collision_contacts.ensure_fields(s)
    s.collision_contacts = [dict(id=i+1, top_sheet=i+1, under_sheet=i+2, state='active')
                            for i in range(len(patches)-1)]
    collision_surface.refresh(s)
    frame = saved(s, margins=True)
    if lower_arc: frame['material_arc_id'][owner == 1] = 8
    if upper_arc: frame['material_arc_id'][owner == 0] = 7
    return s, frame


class CollisionLayerEmergenceTests(unittest.TestCase):
    def test_ordered_overlap_inventory_is_pair_direction_invariant(self):
        s,_=layer_fixture(triple=True)
        surface=s.material_surface;sheets=s.parcel_collision_sheet
        graph=collision_surface.descendants(s.collision_contacts)
        overlap=mesh_coverage.material_overlaps(surface['vertices'],surface['faces'],sheets,
                                                radius_km=surface.get('radius_km',R))
        pairs=collision_surface.ordered_overlap_pairs(surface,sheets,graph,overlap)
        reversed_pairs=collision_surface.ordered_overlap_pairs(surface,sheets,graph,
            dict(first=overlap['second'],second=overlap['first'],area_km2=overlap['area_km2']))
        self.assertGreater(len(pairs['area_km2']),0)
        for field in pairs:np.testing.assert_array_equal(pairs[field],reversed_pairs[field])
        self.assertTrue(all(int(sheets[lower]) in graph[int(sheets[upper])]
                            for upper,lower in zip(pairs['upper_face'],pairs['lower_face'])))
        volume=float(pairs['area_km2']@s.structure['thickness_km'][pairs['lower_face']])
        self.assertAlmostEqual(volume,s.collision_stack_diagnostics['lower_column_load_volume_km3'])

    def check_emergence(self, **options):
        s, frame = layer_fixture(**options)
        inventory = deepcopy(frame)
        jumps = []
        for distance in (.1, .01, .001):
            points = unit([[1., -distance/R, 75./R], [1., distance/R, 75./R]])
            result = sampling.sample_frame(frame, points)
            self.assertEqual(result['material_face'][0], result['material_face'][1])
            self.assertTrue(np.all(result['plate'] == 0))
            for name in ('elevation', 'raw_elevation_m', 'selected_sheet_height_m',
                         'physical_stack_support_m', 'display_thermal_support_m'):
                self.assertTrue(np.isfinite(result[name]).all(), name)
            np.testing.assert_allclose(result['raw_elevation_m'], result['selected_sheet_height_m']+
                result['physical_stack_support_m']+result['display_thermal_support_m'], atol=1e-10)
            np.testing.assert_array_equal(result['clipping_delta_m'], 0.)
            jumps.append({name:float(abs(np.diff(result[name])[0])) for name in
                ('raw_elevation_m', 'selected_sheet_height_m', 'physical_stack_support_m')})
        for name in jumps[0]:
            self.assertLess(jumps[2][name], max(1e-7, .15*jumps[1][name]), (name, jumps))
            self.assertLess(jumps[2][name], .2, (name, jumps))
        # Surface reconstruction cannot repair its own discontinuity by
        # mutating the saved raw columns, support/load, identities or geometry.
        for key, expected in inventory.items():
            if isinstance(expected, np.ndarray): np.testing.assert_array_equal(frame[key], expected, err_msg=key)
        np.testing.assert_array_equal(s.relief+220., frame['material_height_m'])
        return frame

    def test_ordinary_lower_sheet_enters_beneath_upper_margin_continuously(self):
        self.check_emergence()

    def test_two_lower_sheets_enter_through_transitive_order_continuously(self):
        self.check_emergence(triple=True)

    def test_lower_arc_toe_enters_beneath_upper_margin_continuously(self):
        self.check_emergence(lower_arc=True)

    def test_upper_arc_keeps_its_continuous_toe_over_lower_sheet_entry(self):
        self.check_emergence(upper_arc=True)

    def test_layer_emergence_is_rotation_equivariant(self):
        _, frame = layer_fixture(triple=True, lower_arc=True)
        points = unit([[1., -.001/R, 75./R], [1., .001/R, 75./R]])
        reference = sampling.sample_frame(frame, points)
        for angles in (dict(yaw=179.), dict(pitch=89.), dict(yaw=51., pitch=-71., roll=13.)):
            result = sampling.sample_frame(orient_frame(frame, angles), points@rotation_matrix(angles))
            np.testing.assert_array_equal(result['material_face'], reference['material_face'])
            for name in ('raw_elevation_m', 'selected_sheet_height_m', 'physical_stack_support_m'):
                np.testing.assert_allclose(result[name], reference[name], atol=2e-7)


if __name__ == '__main__': unittest.main()
