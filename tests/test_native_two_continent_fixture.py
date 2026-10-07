"""The native collision probe must begin with two real, separate crust bodies."""

import unittest

import numpy as np

import mesh_coverage
import plate_balance
import structure_engine
from experiments.reduced_collision_trajectory import initial_world, two_continent_world
from ridge_geometry import rotate


class TwoContinentFixtureTests(unittest.TestCase):
    def test_prescribed_blocks_have_matching_material_and_trace_columns(self):
        for world,expected in ((initial_world(automatic_finite_front=True),(1e6,0.)),
                               (two_continent_world(),(1e6,1.4e6))):
            area=world.material_surface['area_km2']
            for owner,wanted in enumerate(expected):
                self.assertAlmostEqual(float(area[world.parcel_plate==owner].sum()),wanted,delta=1e-6)
            self.assertEqual(float(world.native_arc_pending['area'].sum()),0.)
            by_id={int(uid):int(kind) for uid,kind in zip(world.parcel_patch,world.kind)}
            np.testing.assert_array_equal(world.trace_kind,
                [by_id[int(uid)] for uid in world.trace_patch])
            np.testing.assert_allclose(structure_engine.material_height(world.kind,world.relief),
                world.structure['reference_elevation_m'],atol=1e-10)
            np.testing.assert_allclose(structure_engine.material_height(
                world.trace_kind,world.trace_relief_m),
                world.trace_structure['reference_elevation_m'],atol=1e-10)

    def test_two_blocks_start_separate_but_can_meet_under_solved_relative_motion(self):
        world=two_continent_world()
        surface=world.material_surface
        faces=surface['faces'];vertices=surface['vertices']
        owner=world.parcel_plate;sheets=world.parcel_collision_sheet
        def cross_area(points):
            overlaps=mesh_coverage.material_overlaps(points,faces,sheets)
            first,second,area=(np.asarray(overlaps[key]) for key in
                ('first','second','area_km2'))
            return float(area[owner[first]!=owner[second]].sum())
        self.assertEqual(cross_area(vertices),0.)
        balance=plate_balance.Balance(world,1.,slab_tethers=True)
        balance.solve()
        rate=balance.rotation()[0]-balance.rotation()[1]
        moved=vertices.copy()
        incoming_vertices=np.unique(faces[owner==0])
        moved[incoming_vertices]=rotate(moved[incoming_vertices],rate*10.)
        self.assertGreater(cross_area(moved),1000.)


if __name__=='__main__':unittest.main()
