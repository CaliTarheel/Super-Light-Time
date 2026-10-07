"""Prescribed trench flux survives arc emergence and changes in grid resolution."""
import gc
import unittest
from unittest.mock import patch

import numpy as np
import trench_history

from raster_engine import RADIUS_KM, Simulation, _unit


def static_trench(width):
    """Hold one 100-km trench at 30 km/Myr to isolate arc construction.

    Call the real deformation/deposition processes while excluding changing
    plate motion and boundary classification from this controlled flux test.
    """
    height = width//2
    simulation = Simulation(dict(width=width, height=height, duration_myr=160,
                                 dt_myr=2, snapshot_myr=2, plate_count=4),
                            dict(width=width, height=height, crust=np.zeros(width*height, np.uint8)))
    below = (height//2)*width+width//2
    above = below+1
    simulation.plate[:] = 0
    simulation.plate[below] = 1
    simulation.active[:] = False
    simulation.active[:2] = True
    simulation.count = 2
    simulation.plate_uid[:] = 0
    simulation.plate_uid[:2] = [1, 2]
    simulation.next_plate_uid = 3
    simulation.support[:] = 0.
    simulation.support[simulation.plate, np.arange(simulation.n)] = 1.
    simulation.ba, simulation.bb = np.array([below]), np.array([above])
    simulation.bp, simulation.bq, simulation.down = np.array([1]), np.array([0]), np.array([1])
    simulation.bcode = np.array([2], np.uint8)
    simulation.bl, simulation.normal_speed = np.array([100.]), np.array([-30.])
    simulation.bmid = _unit(simulation.xyz[[below]]+simulation.xyz[[above]])
    delta = simulation.xyz[[above]]-simulation.xyz[[below]]
    simulation.bn = _unit(delta-simulation.bmid*np.sum(delta*simulation.bmid, axis=1)[:, None])
    # This fixture prescribes supply from an already-developed slab, rather
    # than testing initiation or changes in plate velocity. Replace the random
    # world's records with this one known mature trench; lifecycle maturation
    # itself is exercised separately by test_trench_history.
    simulation.age[below], simulation.age[above] = 80., 20.
    del simulation.trench_systems
    trench_history.initialize(simulation)
    for row in simulation.trench_systems:
        row.update(phase='mature', maturity=1., active_myr=10., shortening_km=300.)
    trench_history.prepare(simulation)
    angle = 180/RADIUS_KM
    point = np.cos(angle)*simulation.bmid+np.sin(angle)*simulation.bn
    arc_cell = int(simulation._indices(point)[0])
    return simulation, arc_cell


class ArcGrowthTests(unittest.TestCase):
    def test_retreat_addition_uses_bounded_rate_and_never_exceeds_realized_supply(self):
        # The same 100-km segment at 30 km/Myr has a requested 4.5 km/Myr
        # retreat contribution. A diffuse pair transfer cannot magnify it.
        for swept_area, expected_retreat in ((1000000., 4.5), (500., 2.5)):
            with self.subTest(swept_area=swept_area):
                simulation, _ = static_trench(64)
                simulation.age[:] = 80.
                below = int(simulation.ba[0])
                # Most transferred membership lies away from this one visible
                # segment; retain an internally consistent area/gain budget.
                simulation.plate[simulation.xyz[:, 0] < 0] = 1
                cells = np.flatnonzero(simulation.plate == 1)
                gain = np.full(len(cells), swept_area/simulation.cell_area[cells].sum())
                self.assertIn(below, cells)
                self.assertTrue(np.all(gain < 1.))
                transfer = {(1, 0): dict(cells=cells, gain=gain, area_km2=swept_area)}
                with patch('raster_engine.extend_forearcs', return_value=transfer), \
                        patch.object(simulation, '_boundaries'):
                    simulation._migrate_trenches(2.)
                np.testing.assert_allclose(simulation.trench_retreat_speed, [expected_retreat])
                simulation.t, simulation.steps = 2., 1
                simulation._deform_and_accrete(2.)
                simulation._rasterize()
                # Baseline 114 km² plus .015 of the bounded retreat sweep.
                expected_area = 114.+expected_retreat*100.*2.*.015
                self.assertAlmostEqual(simulation.process_totals['arc_added_km2'], expected_area)
                self.assertAlmostEqual(float(simulation.mass.sum()), expected_area)
                self.assertAlmostEqual(float(simulation.land_mass.sum()), expected_area)
                self.assertEqual(simulation.process_totals['trench_swept_km2'], swept_area)
                eligible = simulation.process_totals['trench_arc_sweep_km2']
                self.assertAlmostEqual(eligible, expected_retreat*100.*2.)
                self.assertLessEqual(eligible, swept_area)

    def test_volcanic_offset_is_180_km_from_hinge_across_seam_and_pole(self):
        for width in (64, 256):
            for location in ('equator', 'seam', 'pole'):
                with self.subTest(width=width, location=location):
                    simulation, _ = static_trench(width)
                    if location == 'seam':
                        below, above = simulation.h//2*width+width-1, simulation.h//2*width
                    elif location == 'pole':
                        below, above = width//4, 3*width//4
                    else:
                        below, above = int(simulation.ba[0]), int(simulation.bb[0])
                    simulation.plate[:] = 0
                    simulation.plate[below] = 1
                    simulation.ba, simulation.bb = np.array([below]), np.array([above])
                    simulation.bmid = _unit(simulation.xyz[[below]]+simulation.xyz[[above]])
                    delta = simulation.xyz[[above]]-simulation.xyz[[below]]
                    simulation.bn = _unit(delta-simulation.bmid*np.sum(delta*simulation.bmid, axis=1)[:, None])
                    with patch.object(simulation, '_indices', wraps=simulation._indices) as indices, \
                            patch.object(simulation, '_add_arc_crust'):
                        simulation._deform_and_accrete(2.)
                    proposed = indices.call_args_list[0].args[0]
                    distance = np.arccos(np.clip(np.sum(proposed*simulation.bmid, axis=1), -1, 1))*RADIUS_KM
                    np.testing.assert_allclose(distance, 180., atol=1e-8)
                    np.testing.assert_allclose(np.linalg.norm(proposed, axis=1), 1., atol=1e-14)
                    self.assertTrue(np.all(np.sum(proposed*simulation.bn, axis=1) > 0))

    def test_integrated_trench_flux_is_independent_of_resolution_and_emergence(self):
        # This physical setup supplies 114 km² per two-Myr step and 9120 km²
        # in total. Coarse/fine maps reach visible arc crust at different times.
        emerged = {}
        for width in (128, 256, 512):
            with self.subTest(width=width):
                simulation, _ = static_trench(width)
                first_visible_mass = None
                for step in range(80):
                    simulation.t += 2
                    simulation.steps += 1
                    simulation._deform_and_accrete(2.)
                    simulation._rasterize()
                    self.assertAlmostEqual(simulation.process_totals['arc_added_km2'], (step+1)*114., places=8)
                    self.assertAlmostEqual(float(simulation.mass.sum()), (step+1)*114., places=8)
                    self.assertAlmostEqual(float(simulation.land_mass.sum()), (step+1)*114., places=8)
                    if first_visible_mass is None and np.any(simulation.crust == 3):
                        first_visible_mass = float(simulation.mass.sum())
                emerged[width] = first_visible_mass
                self.assertAlmostEqual(simulation.process_totals['arc_added_km2'], 9120., places=8)
                if first_visible_mass is not None:
                    self.assertGreater(simulation.mass.sum(), first_visible_mass,
                                       'A mapped island arc must continue receiving prescribed juvenile flux')
                del simulation
                gc.collect()
        self.assertIsNone(emerged[128], 'The coarse case must actually exercise unresolved crust')
        self.assertIsNotNone(emerged[512], 'The fine case must actually exercise visible arc crust')

    def test_continental_crust_and_cratons_receive_no_juvenile_arc_addition(self):
        for kind in (1, 2):
            with self.subTest(crust=kind):
                simulation, cell = static_trench(64)
                # Place a resolved continental/cratonic patch under the same
                # prescribed volcanic belt, preserving its actual parcel data.
                simulation._add_arc_crust(np.array([cell]), np.array([simulation.cell_area[cell]]))
                simulation.kind[:] = kind
                simulation.trace_kind[:] = kind
                simulation.trace_origin_kind[:] = kind
                simulation._rasterize()
                self.assertEqual(simulation.crust[cell], kind)
                before_mass = simulation.mass.copy()
                before_added = simulation.process_totals['arc_added_km2']
                before_relief = simulation.relief.copy()
                simulation.t, simulation.steps = 2., 1
                simulation._deform_and_accrete(2.)
                np.testing.assert_array_equal(simulation.mass, before_mass)
                self.assertEqual(simulation.process_totals['arc_added_km2'], before_added)
                self.assertFalse(np.any(simulation.kind == 3))
                self.assertGreater(float(simulation.relief[0]), float(before_relief[0]),
                                   'Continental arc-related relief growth must still operate')

    def test_replenishment_and_coalescence_preserve_mass_and_marker_identity(self):
        simulation, cell = static_trench(64)
        simulation.steps = 1
        simulation._add_arc_crust(np.array([cell, cell+1]), np.array([100., 200.]))
        self.assertEqual(len(simulation.mass), 2)
        original_markers = simulation.trace_id.copy()
        simulation.pos[1] = simulation.pos[0]
        simulation._add_arc_crust(np.array([cell]), np.array([400.]))
        self.assertAlmostEqual(float(simulation.mass.sum()), 700.)
        simulation._coalesce_arcs()
        self.assertEqual(len(simulation.mass), 1)
        np.testing.assert_array_equal(simulation.trace_id, original_markers)
        np.testing.assert_array_equal(simulation.trace_origin_kind, [3, 3])
        self.assertTrue(np.all(simulation.trace_patch == simulation.parcel_patch[0]))
        simulation._rasterize()
        self.assertAlmostEqual(float(simulation.land_mass.sum()), 700., places=8)
        simulation.t, simulation.steps = 2., 2
        simulation._deform_and_accrete(2.)
        simulation._rasterize()
        self.assertAlmostEqual(float(simulation.mass.sum()), 814., places=8)
        self.assertAlmostEqual(float(simulation.land_mass.sum()), 814., places=8)
        self.assertAlmostEqual(simulation.process_totals['arc_added_km2'], 814., places=8)


if __name__ == '__main__':
    unittest.main()
