"""A new arc plate follows a real spherical trench without tearing material."""
from copy import deepcopy
import unittest

import numpy as np

from backarc_geometry import RADIUS_KM, TrenchStrip, choose_arc_sliver
from fracture import component_labels
from material_geometry import patch_centres
from raster_engine import Simulation, _rotate


def arc_margin_world(width=128, rotation=None, material=True):
    """Two prescribed plates with juvenile overriding crust beside a real trench.

    Rotate the physical world before rasterization to exercise poles and seams;
    the returned boundary classification and finite-footprint crust are real.
    No simulation method is replaced or frozen.
    """
    height = width//2
    rotation = np.eye(3) if rotation is None else np.asarray(rotation)
    lon, lat = np.meshgrid((np.arange(width)+.5)*2*np.pi/width-np.pi,
                          np.pi/2-(np.arange(height)+.5)*np.pi/height)
    xyz = np.column_stack((np.cos(lat).ravel()*np.cos(lon).ravel(),
                           np.cos(lat).ravel()*np.sin(lon).ravel(), np.sin(lat).ravel()))
    local = xyz@rotation.T
    local_lon = np.arctan2(local[:, 1], local[:, 0])
    local_lat = np.arcsin(np.clip(local[:, 2], -1., 1.))
    arc = (local_lon > 0) & (local_lon < np.deg2rad(12.)) & (np.abs(local_lat) < np.deg2rad(35.))
    crust = (arc*3*material).astype(np.uint8)
    s = Simulation(dict(width=width, height=height, seed=9, plate_count=4,
                        duration_myr=64, dt_myr=2, snapshot_myr=2, erosion=0.),
                   dict(width=width, height=height, crust=crust))
    s.plate = (local[:, 1] > 0).astype(np.int16)
    s.active[:] = False
    s.active[:2] = True
    s.count = 2
    s.plate_uid[:] = 0
    s.plate_uid[:2] = [1, 2]
    s.next_plate_uid = 3
    s.names[:2] = ['Incoming ocean plate', 'Arc-bearing overriding plate']
    s.omega[:] = 0
    s.omega[0] = np.array([0., 0., .005])@rotation
    s.mantle[:] = 0
    s.support[:] = 0
    s.support[s.plate, np.arange(s.n)] = 1
    s.parcel_plate[:] = 1
    s.trace_plate[:] = 1
    s.age[:] = np.where(s.plate == 0, 80., 20.)
    s.polarity[:] = -1
    s._rasterize()
    s._boundaries()
    local_mid = s.bmid@rotation.T
    edges = np.flatnonzero((s.bcode == 2) & (s.down == 0)
                          & ((s.bp == 1) | (s.bq == 1))
                          & (local_mid[:, 0] > .8)
                          & (np.abs(local_mid[:, 2]) < np.sin(np.deg2rad(20.))))
    return s, edges


class BackarcGeometryTests(unittest.TestCase):
    def test_real_margin_yields_attached_curving_sliver_and_connected_parent(self):
        s, edges = arc_margin_world()
        self.assertGreater(len(edges), 8)
        choice = choose_arc_sliver(s, edges, seed=14)
        self.assertIsNotNone(choice)
        child = choice['region']
        self.assertEqual(component_labels(child, s.w, s.h)[1], 1)
        self.assertEqual(component_labels((s.plate == 1) & ~child, s.w, s.h)[1], 1)
        self.assertTrue(np.all(s.plate[child] == 1))
        perimeter = child[s.edge_a] != child[s.edge_b]
        owners = np.where(child[s.edge_a[perimeter]], s.plate[s.edge_b[perimeter]], s.plate[s.edge_a[perimeter]])
        self.assertEqual(set(owners), {0, 1})
        self.assertGreater(choice['length_km'], 300.)
        self.assertGreater(choice['area_fraction'], 0.)
        self.assertLess(choice['area_fraction'], .05)
        np.testing.assert_allclose(np.linalg.norm(choice['path'], axis=1), 1., atol=1e-14)
        self.assertGreater(np.dot(choice['direction'], [0., 1., 0.]), .95)
        self.assertAlmostEqual(np.dot(choice['direction'], choice['center']), 0., places=14)
        width = choice['classifier'].width_at(choice['classifier'].midpoints)
        self.assertGreater(np.ptp(width), 50., 'A finite arc must not become a straight constant-width grid cut')
        self.assertGreaterEqual(float(width.min()), 350.)
        self.assertLessEqual(float(width.max()), 900.)

    def test_proposal_is_seeded_nonmutating_and_trace_ownership_follows_whole_patches(self):
        s, edges = arc_margin_world()
        arrays = {k: v.copy() for k, v in vars(s).items() if isinstance(v, np.ndarray)}
        rng_state = deepcopy(s.rng.bit_generator.state)
        first = choose_arc_sliver(s, edges, seed=14)
        again = choose_arc_sliver(s, edges, seed=14)
        other = choose_arc_sliver(s, edges, seed=27)
        self.assertIsNotNone(first)
        self.assertIsNotNone(other)
        for key in ('region', 'parcel_side', 'trace_side', 'path'):
            np.testing.assert_array_equal(first[key], again[key])
        self.assertFalse(np.array_equal(first['path'], other['path']))
        self.assertEqual(s.rng.bit_generator.state, rng_state)
        for name, original in arrays.items():
            np.testing.assert_array_equal(getattr(s, name), original, err_msg=name)
        ids, _, inverse = patch_centres(s.pos, s.mass, s.parcel_patch)
        sides = first['parcel_side']
        self.assertTrue(np.any(sides))
        self.assertTrue(np.any(~sides))
        patch_side = np.array([sides[inverse == i][0] for i in range(len(ids))])
        np.testing.assert_array_equal(sides, patch_side[inverse])
        np.testing.assert_array_equal(first['trace_side'], patch_side[np.searchsorted(ids, s.trace_patch)])
        # Reassignment consumes no crust and loses no hidden subcell: the same
        # immutable reference-area parcels are partitioned between two owners.
        self.assertAlmostEqual(float(s.mass[sides].sum()+s.mass[~sides].sum()), float(s.mass.sum()), places=6)

    def test_hidden_craton_footprint_and_protected_group_across_cut_are_rejected(self):
        s, edges = arc_margin_world()
        good = choose_arc_sliver(s, edges, seed=14)
        self.assertIsNotNone(good)
        # Put strong material at the inland cut but leave the surface raster
        # juvenile. Protection must read the enduring material, not visible pixels.
        candidates = np.flatnonzero((s.pos[:, 1] > np.sin(np.deg2rad(3.))) & (np.abs(s.pos[:, 2]) < .1))
        k = candidates[np.argmin(np.abs(good['classifier'].signed_distance(s.pos[candidates])))]
        same = s.parcel_patch == s.parcel_patch[k]
        s.kind[same] = 2
        s.parcel_craton[same] = 301
        self.assertFalse(np.any(s.crust == 2))
        self.assertIsNone(choose_arc_sliver(s, edges, seed=14))
        # Even a craton with no point in the rupture belt is indivisible when
        # occlusion or a long connected material body spans both sides.
        s.kind[same] = 3
        s.parcel_craton[:] = -1
        inside = np.flatnonzero(good['parcel_side'])[0]
        outside = np.flatnonzero(~good['parcel_side'])[-1]
        s.parcel_craton[[inside, outside]] = 902
        self.assertIsNone(choose_arc_sliver(s, edges, seed=14))

    def test_foreign_plate_at_new_perimeter_rejects_the_entire_proposal(self):
        s, edges = arc_margin_world()
        good = choose_arc_sliver(s, edges, seed=14)
        self.assertIsNotNone(good)
        child = good['region']
        adjacent = (child[s.edge_a] != child[s.edge_b]) & (s.plate[s.edge_a] == 1) & (s.plate[s.edge_b] == 1)
        k = np.flatnonzero(adjacent)[len(np.flatnonzero(adjacent))//2]
        outside = int(s.edge_b[k] if child[s.edge_a[k]] else s.edge_a[k])
        s.plate[outside] = 2
        s.active[2] = True
        self.assertIsNone(choose_arc_sliver(s, edges, seed=14))

    def test_finite_sector_cannot_cut_an_isthmus_into_detached_parent_fragments(self):
        s, edges = arc_margin_world()
        good = choose_arc_sliver(s, edges, seed=14)
        self.assertIsNotNone(good)
        # Restrict this parent to a thin, long strip: removing the finite middle
        # arc would leave northern and southern bodies with the same parent.
        thin_parent = (s.plate == 1) & (s.xyz[:, 1] < np.sin(np.deg2rad(3.))) & (s.xyz[:, 0] > .5)
        s.plate[(s.plate == 1) & ~thin_parent] = 0
        s.support[:] = 0
        s.support[s.plate, np.arange(s.n)] = 1
        s._boundaries()
        edges = np.flatnonzero((s.bcode == 2) & (s.down == 0) & (s.bmid[:, 0] > .9) & (np.abs(s.bmid[:, 2]) < .25))
        self.assertEqual(component_labels(s.plate == 1, s.w, s.h)[1], 1)
        self.assertIsNone(choose_arc_sliver(s, edges, seed=14))

    def test_actual_polar_and_seam_margins_remain_connected_and_coherent(self):
        for angle in ([0., 0., np.pi], [0., -np.pi/2, 0.]):
            with self.subTest(rotation=angle):
                rotation = _rotate(np.eye(3), np.asarray(angle))
                s, edges = arc_margin_world(rotation=rotation)
                choice = choose_arc_sliver(s, edges, seed=14)
                self.assertIsNotNone(choice)
                self.assertEqual(component_labels(choice['region'], s.w, s.h)[1], 1)
                self.assertEqual(component_labels((s.plate == 1) & ~choice['region'], s.w, s.h)[1], 1)
                ids, _, inverse = patch_centres(s.pos, s.mass, s.parcel_patch)
                for i in range(len(ids)):
                    self.assertEqual(len(np.unique(choice['parcel_side'][inverse == i])), 1)
                if angle[1]:
                    self.assertGreater(np.max(np.abs(s.xyz[choice['region'], 2])), .99)

    def test_continuous_classifier_is_rotation_equivariant_and_width_is_bounded(self):
        s, edges = arc_margin_world()
        c = choose_arc_sliver(s, edges, seed=14)['classifier']
        rotation = _rotate(np.eye(3), np.array([.8, -.6, 1.1]))
        moved = TrenchStrip(c.center@rotation, c.direction@rotation,
                            c.midpoints@rotation, c.normals@rotation,
                            c.lengths.copy(), c.width_km, c.phases.copy())
        np.testing.assert_allclose(c.signed_distance(s.xyz), moved.signed_distance(s.xyz@rotation), atol=2e-13)
        for width in (350., 900.):
            c.width_km = width
            values = c.width_at(s.xyz)
            self.assertGreaterEqual(float(values.min()), 350.)
            self.assertLessEqual(float(values.max()), 900.)

    def test_invalid_or_nonconvergent_edges_cannot_make_a_sliver(self):
        s, edges = arc_margin_world()
        self.assertIsNone(choose_arc_sliver(s, [], seed=14))
        self.assertIsNone(choose_arc_sliver(s, [-1], seed=14))
        self.assertIsNone(choose_arc_sliver(s, [len(s.ba)], seed=14))
        for width in (349., 901., np.nan):
            with self.assertRaises(ValueError):
                choose_arc_sliver(s, edges, seed=14, width_km=width)
        s.bcode[edges[0]] = 1
        self.assertIsNone(choose_arc_sliver(s, edges, seed=14))

    def test_bare_ocean_sector_cannot_be_declared_an_arc_sliver(self):
        s, edges = arc_margin_world(material=False)
        self.assertGreater(len(edges), 8)
        self.assertEqual(len(s.pos), 0)
        self.assertIsNone(choose_arc_sliver(s, edges, seed=14))


if __name__ == '__main__':
    unittest.main()
