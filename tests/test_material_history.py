"""Material-history identity and accounting checks, independent of terrain QA."""
import unittest

import numpy as np

from raster_engine import DEFAULT_CONFIG, Simulation


def world(**kwargs):
    return Simulation(dict(width=64, height=32, duration_myr=40, snapshot_myr=2, **kwargs))


class MaterialHistoryTests(unittest.TestCase):
    def test_every_step_default_and_independent_material_budget(self):
        self.assertEqual(DEFAULT_CONFIG["snapshot_myr"], 2)
        simulation = world()
        frames = list(simulation.snapshots())
        self.assertEqual([f["time_myr"] for f in frames], list(range(0, 41, 2)))
        birth_relief = {}
        previous_ids = set()
        for frame in frames:
            ids = frame["trace_id"]
            self.assertEqual(len(ids), len(np.unique(ids)))
            self.assertTrue(previous_ids <= set(ids))
            previous_ids = set(ids)
            np.testing.assert_allclose(np.linalg.norm(frame["trace_xyz"], axis=1), 1, atol=1e-13)
            self.assertTrue(np.all(frame["trace_birth_myr"] <= frame["time_myr"]))
            self.assertTrue(np.all(frame["trace_uplift_m"] >= 0))
            self.assertTrue(np.all(frame["trace_extension_m"] >= 0))
            for i, marker in enumerate(ids):
                birth_relief.setdefault(int(marker), frame["trace_relief_m"][i])
            expected = np.array([birth_relief[int(marker)] for marker in ids])
            expected += frame["trace_uplift_m"] - frame["trace_extension_m"]
            expected += frame["trace_adjustment_m"] - frame["trace_erosion_m"]
            np.testing.assert_allclose(frame["trace_relief_m"], expected, atol=1e-9)
            self.assertEqual(frame["history_version"], 1)
        initial_position = frames[0]["trace_xyz"].copy()
        frames[-1]["trace_xyz"][:] = 0
        np.testing.assert_array_equal(frames[0]["trace_xyz"], initial_position)
        simulation.events[-1]["details"]["mutated"] = True
        self.assertNotIn("mutated", frames[-1]["events"][-1]["details"])

    def test_arc_traces_survive_coalescence_and_welding(self):
        cfg = dict(width=64, height=32, duration_myr=2)
        simulation = Simulation(cfg, {"width": 64, "height": 32, "crust": np.zeros(2048)})
        self.assertEqual(len(simulation.trace_id), 0)
        p = int(np.bincount(simulation.plate).argmax())
        cells = np.flatnonzero(simulation.plate == p)[10:12]
        simulation.t = 2
        simulation._add_arc_crust(cells, np.array([100., 100.]))
        self.assertEqual(len(simulation.trace_id), 2)
        ids, xyz = simulation.trace_id.copy(), simulation.trace_xyz.copy()
        np.testing.assert_array_equal(simulation.trace_birth_myr, [2., 2.])
        simulation.pos[1] = simulation.pos[0]
        simulation._coalesce_arcs()
        self.assertEqual(len(simulation.pos), 1)
        np.testing.assert_array_equal(simulation.trace_id, ids)
        np.testing.assert_array_equal(simulation.trace_xyz, xyz)
        q = (p + 1) % simulation.count
        simulation._weld(p, q)
        np.testing.assert_array_equal(simulation.trace_id, ids)
        np.testing.assert_array_equal(simulation.trace_plate, [q, q])
        np.testing.assert_array_equal(simulation.trace_kind, [1, 1])
        np.testing.assert_array_equal(simulation.trace_origin_kind, [3, 3])
        event = simulation.events[-1]
        self.assertEqual(event["plate_uids"], [int(simulation.plate_uid[p]), int(simulation.plate_uid[q])])
        self.assertIn("lon", event)
        self.assertIn("lat", event)

    def test_split_reused_slots_have_distinct_plate_identity(self):
        simulation = world()
        p = int(np.bincount(simulation.parcel_plate, weights=simulation.mass).argmax())
        # The initial supercontinent now consists of small craton-preserving
        # shards. Consolidate buoyant material into a coherent parent before
        # testing two real fractures and slot reuse; a single initial craton
        # block need not itself be splittable.
        simulation.parcel_plate[:] = p
        simulation.trace_plate[:] = p
        land = simulation.crust > 0
        simulation.plate[land] = p
        simulation.support[:, land] = 0
        simulation.support[p, land] = 1
        q = int(np.flatnonzero(~simulation.active)[0])
        initial_ids = simulation.trace_id.copy()
        initial_positions = simulation.trace_xyz.copy()
        simulation._split(p)
        self.assertTrue(simulation.active[q])
        first_uid = int(simulation.plate_uid[q])
        self.assertEqual(simulation.plate_parent_uid[q], simulation.plate_uid[p])
        self.assertTrue(np.any(simulation.trace_plate == q))
        np.testing.assert_array_equal(simulation.trace_id, initial_ids)
        np.testing.assert_array_equal(simulation.trace_xyz, initial_positions)
        event = simulation.events[-1].copy()
        # A consumed-slot fixture: consolidate its material and membership, then
        # split the same continent again to exercise the real slot-reuse path.
        simulation.plate[simulation.plate == q] = p
        simulation.parcel_plate[simulation.parcel_plate == q] = p
        simulation.trace_plate[simulation.trace_plate == q] = p
        simulation.support[p] += simulation.support[q]
        simulation.support[q] = 0
        simulation.active[q] = False
        simulation._split(p)
        self.assertTrue(simulation.active[q])
        self.assertNotEqual(int(simulation.plate_uid[q]), first_uid)
        self.assertEqual(simulation.plate_generation[q], 2)
        self.assertEqual(event["plate_uids"][-1], first_uid)
        np.testing.assert_array_equal(simulation.trace_id, initial_ids)

    def test_submerged_columns_do_not_rise_through_negative_erosion(self):
        import structure_engine
        simulation = world()
        simulation.relief[:] = -1200
        simulation.kind[:] = 1
        simulation.trace_relief_m[:] = -1200
        simulation.trace_kind[:] = 1
        structure_engine.initialize_parcels(simulation)
        # This deliberate submerged starting fixture has neutral columns at
        # its specified height, with no active cooling or foreland load.
        simulation.trace_structure = structure_engine._new(simulation.trace_kind, simulation.trace_relief_m)
        simulation.bcode[:] = 3
        zero = np.zeros(simulation.n)
        structure_engine.deform(simulation, zero, zero, zero, 2)
        np.testing.assert_array_equal(simulation.trace_erosion_m, 0.)
        np.testing.assert_array_equal(simulation.trace_adjustment_m, 0.)
        np.testing.assert_allclose(simulation.trace_relief_m, -1200., atol=1e-10)

    def test_zero_raster_area_marker_owner_keeps_real_plate_metadata(self):
        simulation = world()
        p = int(simulation.trace_plate[0])
        q = (p + 1) % simulation.count
        simulation.plate[simulation.plate == p] = q
        # A zero-dominant-area remnant can still own tracked buoyant material.
        simulation.names[p] = "Recorded remnant name"
        simulation.born[p] = 23.  # Dynamics' cooldown is not its creation date.
        frame = simulation.snapshot()
        metadata = {plate["uid"]: plate for plate in frame["plates"]}
        self.assertTrue(set(frame["trace_plate_uid"]) <= set(metadata))
        owner = metadata[int(frame["trace_plate_uid"][0])]
        self.assertEqual(owner["name"], "Recorded remnant name")
        self.assertEqual(owner["area_km2"], 0)
        self.assertEqual(owner["created_myr"], 0)
        self.assertTrue(owner["active"])
        self.assertAlmostEqual(sum(p["area_km2"] for p in frame["plates"])/simulation.earth_area, 1., places=12)


if __name__ == "__main__":
    unittest.main()
