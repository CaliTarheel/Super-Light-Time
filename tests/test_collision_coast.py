"""Collision interiors use exposed union coasts with continuous sheet entry."""
import unittest
from unittest.mock import patch
import numpy as np
import collision_coast
import material_surface
import continental_margin
import native_frame_sampling as sampling
from tests import test_collision_layer_emergence as emergence
from tests import test_collision_surface_acceptance as acceptance
from tests.test_collision_contacts import rectangle


def revised_saved(*args, **kwargs):
    frame = acceptance.saved(*args, **kwargs)
    frame['collision_coast_version'] = 1
    return frame


class CollisionCoastTests(unittest.TestCase):
    def test_lower_entry_continuity_including_arcs_and_three_layers(self):
        check = emergence.CollisionLayerEmergenceTests()
        with patch.object(emergence, 'saved', revised_saved):
            for options in ({}, dict(triple=True), dict(lower_arc=True), dict(upper_arc=True)):
                with self.subTest(options=options): check.check_emergence(**options)
            check.test_layer_emergence_is_rotation_equivariant()

    def test_upper_toe_continuity_and_provenance(self):
        original = acceptance.saved
        def revised(*args, **kwargs):
            frame=original(*args, **kwargs);frame['collision_coast_version']=1;return frame
        # Reuse the geometric approach-to-edge test with the new saved policy.
        with patch.object(acceptance, 'saved', revised):
            tests = acceptance.CollisionSurfaceAcceptanceTests()
            tests.test_upper_free_edge_and_lower_emergence_share_a_continuous_interface()

    def test_buried_edges_are_not_ocean_coasts(self):
        _, frame = emergence.layer_fixture()
        frame['collision_coast_version'] = 1
        ctx = sampling.prepare(frame)
        points = emergence.unit([[1., 25./6371., 0.]])
        _, faces, _ = sampling.geometry.locate_points(points, ctx['material_locator'], all_hits=True)
        old = continental_margin.distance_km(ctx['continental_margin'], np.repeat(points, len(faces), axis=0), faces)
        new = continental_margin.distance_km(ctx['collision_coast'], np.repeat(points, len(faces), axis=0), faces)
        self.assertTrue(np.all(old < 30.))
        self.assertTrue(np.all(new > 150.))
        result = sampling.sample_frame(frame, points, prepared=ctx)
        self.assertGreater(result['elevation'][0], 500.)
        self.assertGreater(ctx['collision_coast']['covered_source_edges'], 0)

    def test_isolated_coast_and_arc_profile_unchanged_without_support(self):
        for arc in (False, True):
            tri=acceptance.triangle()
            s=acceptance.world([(tri, 35., 1000., False)])
            acceptance.collision_surface.refresh(s)
            frame=acceptance.saved(s, margins=True)
            if arc: frame['material_arc_id'][:]=1
            points=acceptance.unit(np.array([tri.mean(axis=0), .001*tri[0]+.4995*tri[1]+.4995*tri[2]]))
            before=sampling.sample_frame(frame, points)
            frame['collision_coast_version']=1
            after=sampling.sample_frame(frame, points)
            np.testing.assert_allclose(after['elevation'], before['elevation'], atol=1e-7, rtol=0.)
            np.testing.assert_array_equal(after['material_face'],before['material_face'])

    def test_policy_requires_explicit_valid_tag_and_support(self):
        for bad in (True, -1, 3, 1.0):
            with self.assertRaises(ValueError): collision_coast.version(dict(collision_coast_version=bad, collision_surface_version=1))
        with self.assertRaises(ValueError): collision_coast.version(dict(collision_coast_version=1))
        self.assertEqual(collision_coast.version({}),0)

    def test_partial_edge_clipping_keeps_exposed_remainder_and_real_gap(self):
        for right_start in (0., 210.):
            surface=material_surface.initialize_surface(*rectangle(-600.,200.,0,width=400.))
            material_surface.append_surface(surface,*rectangle(right_start,800.,1,width=100.))
            v,f=surface['vertices'],surface['faces']
            old=continental_margin.prepare(v,f,surface['face_kind'],np.zeros(len(f),int))
            new=collision_coast.prepare(v,f,old)
            xyz=emergence.unit([[1.,190./6371.,0.],[1.,190./6371.,300./6371.]])
            locator=sampling.geometry.build_locator(v,f)
            chosen,_=sampling.geometry.locate_points(xyz,locator)
            distance=continental_margin.distance_km(new,xyz,chosen)
            self.assertLess(distance[1],11.)  # Uncovered part of the same edge.
            if right_start==0.: self.assertGreater(distance[0],90.)
            else: self.assertLess(distance[0],11.)  # Ten-km water gap remains.

    def test_export_rejects_mixed_coast_revisions(self):
        from copy import deepcopy
        from tests.test_gospl_collision_support import supported
        import gospl_export
        first=supported();second=deepcopy(first);second['time_myr']=2.
        for target in (first,second):
            target['collision_coast_version']=1
            with self.assertRaisesRegex(ValueError,'mix collision coast versions'):
                gospl_export.interval_forcing(np.array([[1.,0.,0.]]),first,second,{})
            del target['collision_coast_version']

    def test_reconstruction_does_not_force_submerged_columns_above_sea_level(self):
        _,frame=emergence.layer_fixture()
        frame['collision_coast_version']=1
        frame['material_height_m'][:]=-2500.
        frame['material_collision_support_m'][:]=0.
        frame['material_collision_load_thickness_km'][:]=0.
        result=sampling.sample_frame(frame,emergence.unit([[1.,25./6371.,0.]]))
        self.assertAlmostEqual(result['elevation'][0],-2500.,places=7)


if __name__ == '__main__': unittest.main()
