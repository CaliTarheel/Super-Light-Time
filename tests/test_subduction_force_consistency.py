"""Force onset follows developed local history, not a boundary label alone."""
from copy import deepcopy
import unittest
from unittest.mock import patch
import numpy as np
import trench_history as history
from trench_dynamics import retreat_speed
from raster_engine import Simulation
from test_trench_history import fixture, step


class SubductionForceConsistencyTests(unittest.TestCase):
    def test_real_force_path_has_basal_startup_and_only_developed_slab_response(self):
        s=fixture();history.initialize(s)
        s.cell_area=np.ones(s.n);s.earth_area=float(s.n)
        s.kind=np.empty(0,np.uint8);s.mass=np.empty(0);s.parcel_plate=np.empty(0,int)
        s.mantle=np.array([[.001,0.,0.],[-.001,0.,0.]])
        s.rng=np.random.default_rng(3);s.config=dict(slab_pull=1.,ridge_push=0.)
        without=deepcopy(s);without.config['slab_pull']=0.
        Simulation._forces(s,2.);Simulation._forces(without,2.)
        np.testing.assert_array_equal(s.omega,without.omega)
        self.assertGreater(np.linalg.norm(s.omega),0.,'Basal driving can initiate convergence with no initial slab pull.')
        for _ in range(5):step(s)
        without=deepcopy(s);without.config['slab_pull']=0.
        Simulation._forces(s,2.);Simulation._forces(without,2.)
        self.assertGreater(np.linalg.norm(s.omega-without.omega),0.)

    def test_uninitialized_and_initial_histories_have_no_slab_pull(self):
        s = fixture()
        np.testing.assert_array_equal(history.slab_pull_weights(s), 0.)
        np.testing.assert_array_equal(history.slab_pull_weights(s, legacy=True), 1.)
        history.initialize(s)
        before = deepcopy(s.trench_systems)
        np.testing.assert_array_equal(history.slab_pull_weights(s), 0.)
        self.assertEqual(s.trench_systems, before)

    def test_local_maturity_scales_force_like_consumption(self):
        s = fixture(((28, 35, 31), (28, 35, 95)))
        history.initialize(s)
        for _ in range(5):
            step(s)
        np.testing.assert_allclose(history.slab_pull_weights(s), history.weights(s))
        # A remote fresh system never borrows the other system's developed slab.
        s.trench_systems[1]['maturity'] = 0.
        history.prepare(s)
        np.testing.assert_array_equal(history.slab_pull_weights(s)[s.trench_id == 2], 0.)
        np.testing.assert_array_equal(history.slab_pull_weights(s)[s.trench_id == 1], 1.)

    def test_slow_shortening_cannot_mature_from_time_alone(self):
        s = fixture()
        s.normal_speed[:] = -3.
        history.initialize(s)
        for _ in range(5):
            step(s)
        np.testing.assert_allclose(history.slab_pull_weights(s), .3)

    def test_inactive_and_buoyant_contacts_supply_no_pull(self):
        s = fixture()
        history.initialize(s)
        for _ in range(5):
            step(s)
        s.crust[s.ba] = 1
        np.testing.assert_array_equal(history.slab_pull_weights(s), 0.)
        s.crust[s.ba] = 0
        s.bcode[:] = 3
        np.testing.assert_array_equal(history.slab_pull_weights(s), 0.)
        s.bcode[:] = 2
        s.normal_speed[:] = 2
        np.testing.assert_array_equal(history.slab_pull_weights(s), 0.)

    def test_actual_fractional_ocean_support_scales_force(self):
        s = fixture()
        history.initialize(s)
        for _ in range(5):
            step(s)
        with patch.object(history.native_subduction, 'enabled', return_value=True), \
             patch.object(history.native_subduction, 'edge_ocean_fraction', return_value=np.full(len(s.ba), .25)):
            np.testing.assert_allclose(history.slab_pull_weights(s), .25)

    def test_cooling_proxy_is_shared_with_retreat_and_has_no_newborn_floor(self):
        np.testing.assert_allclose(history.ocean_buoyancy([0., 20., 80., 1000.]), [0., .5, 1., 1.8])
        self.assertEqual(float(retreat_speed(20., 0., 0.)), 0.)
        self.assertAlmostEqual(float(retreat_speed(20., 20., 0.)), 1.5)
        for value in [-1., np.nan, np.inf]:
            with self.assertRaises(ValueError):
                history.ocean_buoyancy(value)


if __name__ == '__main__':
    unittest.main()
