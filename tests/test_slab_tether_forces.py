"""Expanded-velocity and power oracles for actual plate attachment assembly."""
from copy import deepcopy
from dataclasses import replace
import math
import unittest
import numpy as np
import plate_balance as pb
import slab_memory as slab
import slab_tether_forces as forces
import slab_tether_history as history
from slab_tether import Neck
from tests.test_slab_tether import neck
from tests.test_slab_depth_window import inventory, attached


def fixture(extent=300., damage=.3, edges=(400., 600.)):
    s = attached(inventory(extent), edges)
    s.xyz = np.vstack((np.eye(3), -np.eye(3)))
    s.cell_area = np.full(6, 1000.)
    s.plate = np.array([0, 0, 0, 1, 1, 1])
    s.crust = np.zeros(6, int); s.age = np.full(6, 150.)
    s.bmid = np.tile([1., 0., 0.], (len(edges), 1))
    s.bn = np.tile([0., 1., 0.], (len(edges), 1))
    s.omega = np.zeros((2, 3)); s.config = {}
    history.initialize(s.trench_systems[0], neck(damage=damage))
    return s


class SlabTetherForceTests(unittest.TestCase):
    def test_actual_balance_matches_expanded_independent_slab_velocities(self):
        s = fixture()
        s.bmid[1] = [0., 1., 0.]; s.bn[1] = [0., 0., 1.]
        model = pb.Balance(s, 1., slab_tethers=True)
        row = s.trench_systems[0]; material = Neck(**row[history.FIELD][0]['neck'])
        width = 1e6
        cs = 3e22*(300.*math.sin(math.radians(50.))/660.)*width
        cn = 4.*material.viscosity_pa_s*material.thickness_m/material.length_m*(1.-material.damage)**2*width
        weight = row[slab.RETAINED_MASS_FIELD]*9.81*math.sin(math.radians(50.))
        # Two independent slab speeds, one for each differently oriented edge.
        expanded = np.zeros((8, 8)); right = np.zeros(8)
        background = model.stiffness-model.tether_assembly['stiffness']
        expanded[:6, :6] = background
        for e, share in enumerate((.4, .6)):
            r = np.r_[np.cross(s.bmid[e], s.bn[e]), [0., 0., 0.]]
            expanded[:6, :6] += cn*share*pb.CM_YR_M_S**2*np.outer(r, r)
            expanded[:6, 6+e] = expanded[6+e, :6] = -cn*share*pb.CM_YR_M_S**2*r
            expanded[6+e, 6+e] = (cs+cn)*share*pb.CM_YR_M_S**2
            right[6+e] = weight*share*pb.CM_YR_M_S
        oracle = np.linalg.solve(expanded, right)
        condensed = np.linalg.solve(model.stiffness, model.torque)
        np.testing.assert_allclose(condensed, oracle[:6], rtol=3e-13, atol=1e-12)
        self.assertGreater(np.linalg.eigvalsh(model.stiffness).min(), 0.)
        # Actual nonlinear plate solve also meets its force acceptance criterion.
        model.solve()
        gradient = model._evaluate(model.x, pb.HUBER_CONTINUATION_KM_MYR[-1]*pb.KM_MYR_CM_YR)[1]
        self.assertLessEqual(model._relative_residual(gradient), pb.FORCE_RELATIVE_TOLERANCE)

    def test_power_and_virtual_work_in_actual_euler_coordinates(self):
        model = pb.Balance(fixture(), 1., slab_tethers=True)
        assembly = model.tether_assembly
        rng = np.random.default_rng(2026)
        for _ in range(20):
            x = rng.normal(size=6)*20.
            report = forces.audit(assembly, x)
            expected = float(x@(assembly['drive']-assembly['stiffness']@x))
            scale = max(abs(report['weight_power_w']), abs(expected), 1.)
            self.assertLess(abs(report['plate_power_w']-expected)/scale, 1e-13)
            self.assertLess(abs(report['power_residual_w'])/scale, 1e-13)
            self.assertGreaterEqual(report['neck_dissipation_w'], 0.)
            self.assertGreaterEqual(report['mantle_dissipation_w'], 0.)

    def test_subdivision_and_reversed_edge_orientation_preserve_force_and_drag(self):
        base = pb.Balance(fixture(edges=(1000.,)), 1., slab_tethers=True)
        for count in (2, 10):
            s = fixture(edges=(1000./count,)*count)
            s.bp[::2], s.bq[::2] = 1, 0
            s.ba[::2], s.bb[::2] = 1, 0
            s.bn[::2] *= -1.
            model = pb.Balance(s, 1., slab_tethers=True)
            np.testing.assert_allclose(model.stiffness, base.stiffness, rtol=2e-14)
            np.testing.assert_allclose(model.torque, base.torque, rtol=2e-14)

    def test_split_and_rejoin_keep_distinct_damage_channels(self):
        s = fixture(edges=(300., 700.))
        before = pb.Balance(s, 1., slab_tethers=True)
        parent = s.trench_systems[0]
        child = {k: deepcopy(parent[k]) for k in ('phase','maturity','downgoing_plate_uid','overriding_plate_uid')}
        child.update(id=2, length_km=300.)
        slab.partition(parent, child, .3)
        parent['length_km'] = 700.; slab.refresh_line_load(parent)
        s.trench_systems.append(child); s.trench_id[0] = 2
        split = pb.Balance(s, 1., slab_tethers=True)
        np.testing.assert_allclose(split.stiffness, before.stiffness, rtol=2e-14)
        np.testing.assert_allclose(split.torque, before.torque, rtol=2e-14)
        child[history.FIELD][0]['neck']['damage'] = .9
        unequal = pb.Balance(s, 1., slab_tethers=True)
        slab.join(child, parent); child['phase'] = 'joined'
        s.trench_id[:] = 1; parent['length_km'] = 1000.; slab.refresh_line_load(parent)
        joined = pb.Balance(s, 1., slab_tethers=True)
        np.testing.assert_allclose(joined.stiffness, unequal.stiffness, rtol=2e-14)
        np.testing.assert_allclose(joined.torque, unequal.torque, rtol=2e-14)
        self.assertEqual(len(joined.tether_assembly['channels']), 2)

    def test_rupture_removes_transmitted_drive_and_drag_but_retains_mass(self):
        s = fixture()
        row = s.trench_systems[0]; initial = row[slab.RETAINED_MASS_FIELD]
        material = Neck(**row[history.FIELD][0]['neck'])
        history.rupture(row, 0, replace(material, damage=1.))
        model = pb.Balance(s, 1., slab_tethers=True)
        np.testing.assert_array_equal(model.tether_assembly['stiffness'], 0.)
        np.testing.assert_array_equal(model.drivers['slab'], 0.)
        self.assertEqual(len(model.trench_edges), 0)
        self.assertEqual(row['slab_retired_excess_mass_kg'], initial)

    def test_depth_window_limits_both_forces_without_mutation(self):
        cap = 660./math.sin(math.radians(50.))
        shallow = pb.Balance(fixture(extent=cap), 1., slab_tethers=True)
        s = fixture(extent=10000.)
        original = deepcopy(s.trench_systems)
        deep = pb.Balance(s, 1., slab_tethers=True)
        np.testing.assert_allclose(deep.stiffness, shallow.stiffness, rtol=2e-14)
        np.testing.assert_allclose(deep.torque, shallow.torque, rtol=2e-14)
        self.assertEqual(s.trench_systems, original)

    def test_no_matched_trace_no_force_and_explicit_history_required(self):
        s = fixture(); s.trench_id[:] = 0
        model = pb.Balance(s, 1., slab_tethers=True)
        np.testing.assert_array_equal(model.drivers['slab'], 0.)
        s.trench_id[:] = 1
        del s.trench_systems[0][history.FIELD]; del s.trench_systems[0][history.VERSION_FIELD]
        with self.assertRaisesRegex(ValueError, 'Explicit slab neck histories'):
            pb.Balance(s, 1., slab_tethers=True)
        pb.Balance(s, 1.)  # Standard policy does not require or select this experiment.

    def test_native_mesh_force_solution_and_diagnostics_survive_checkpoint(self):
        from pathlib import Path
        from types import SimpleNamespace
        import tempfile
        import checkpoint
        from tests.test_native_processes import ocean_fixture
        s = ocean_fixture(level=2, two=True)
        s._boundaries()
        row = fixture().trench_systems[0]
        row['downgoing_plate_uid'], row['overriding_plate_uid'] = map(int, s.plate_uid[:2])
        s.trench_systems = [row]
        s.trench_id = np.zeros(len(s.ba), int)
        edges = np.flatnonzero(s.bl > 0.)[:4]
        self.assertEqual(len(edges), 4)
        s.trench_id[edges] = row['id']
        before = deepcopy(row)
        original_omega = s.omega.copy()
        model = pb.Balance(s, .25, slab_tethers=True)
        model.solve()
        report = model.diagnostics(model.rotation())
        self.assertLessEqual(report['scaled_force_residual'], pb.FORCE_RELATIVE_TOLERANCE)
        power = report['slab_tether_power']
        self.assertLess(abs(power['power_residual_w'])/max(abs(power['weight_power_w']), 1.), 1e-12)
        self.assertEqual(report['slab_stokes_dissipation_w'], power['mantle_dissipation_w'])
        self.assertIn('independently eliminated', report['subduction_response'])
        self.assertEqual(s.trench_systems[0], before)
        np.testing.assert_array_equal(s.omega, original_omega)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)/'state.npz'
            checkpoint.write_checkpoint(path, s, dict(config=s.config), {})
            resumed, _ = checkpoint.read_checkpoint(path, None, SimpleNamespace)
        continuation = pb.Balance(resumed, .25, slab_tethers=True)
        continuation.solve()
        np.testing.assert_array_equal(continuation.x, model.x)
        self.assertEqual(continuation.diagnostics(continuation.rotation()), report)

    def test_neutrally_buoyant_attached_inventory_resists_without_manufacturing_weight(self):
        s = fixture()
        row = s.trench_systems[0]
        for key in slab.MASS_FIELDS: row[key] = 0.
        row[history.FIELD][0]['retained_excess_mass_kg'] = 0.
        slab.refresh_line_load(row)
        model = pb.Balance(s, 1., slab_tethers=True)
        np.testing.assert_array_equal(model.drivers['slab'], 0.)
        self.assertGreater(np.trace(model.tether_assembly['stiffness']), 0.)
        self.assertEqual(len(model.trench_edges), len(s.ba))


if __name__ == '__main__': unittest.main()
