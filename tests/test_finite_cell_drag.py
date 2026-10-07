"""Basal drag integrates the finite native cell rather than its centroid."""
import unittest
from types import SimpleNamespace

import numpy as np

import plate_balance
from mesh_geometry import geometry, icosphere


class FiniteCellDragTests(unittest.TestCase):
    def test_one_occupied_native_cell_has_spin_drag(self):
        mesh = icosphere(2)
        count = len(mesh['faces'])
        support = np.zeros((1, count))
        support[0, 0] = 1.
        state = SimpleNamespace(native_mesh=mesh, support=support,
                                plate=np.zeros(count, int),
                                cell_area=mesh['area_km2'], xyz=mesh['xyz'],
                                crust=np.zeros(count, int))
        model = plate_balance.Balance.__new__(plate_balance.Balance)
        model.s = state
        model.plates = [0]
        model.slot = {0: 0}
        model.size = 3
        model.bp = np.zeros(0, int)
        model.live = np.zeros(0, int)
        model.trench = np.zeros(0, bool)
        model.notes = {}
        model._viscous()
        radial = mesh['xyz'][0]
        self.assertGreater(radial@model.basal[0]@radial, 0.)
        self.assertGreater(np.linalg.eigvalsh(model.basal[0]).min(), 0.)

    def test_octant_analytic(self):
        mesh = dict(vertices=np.eye(3), faces=np.array([[0, 1, 2]]))
        expected = np.full((3, 3), -2/(3*np.pi))
        np.fill_diagonal(expected, 2/3)
        np.testing.assert_allclose(plate_balance._cell_rotation_metric(mesh)[0],
                                   expected, atol=2e-15)

    def test_subdivision_preserves_integral(self):
        vertices = np.eye(3)
        vertices = np.vstack((vertices,
                              (vertices[0]+vertices[1])/np.sqrt(2),
                              (vertices[1]+vertices[2])/np.sqrt(2),
                              (vertices[2]+vertices[0])/np.sqrt(2)))
        mesh = geometry(vertices, [[0, 3, 5], [3, 1, 4],
                                   [5, 4, 2], [3, 4, 5]], require_closed=False)
        average = np.einsum('c,cij->ij', mesh['area_km2'],
                            plate_balance._cell_rotation_metric(mesh))/mesh['area_km2'].sum()
        octant = dict(vertices=np.eye(3), faces=[[0, 1, 2]])
        np.testing.assert_allclose(average,
                                   plate_balance._cell_rotation_metric(octant)[0],
                                   atol=2e-15)

    def test_single_cell_spin_and_rotation(self):
        mesh = icosphere(6)
        metric = plate_balance._cell_rotation_metric(mesh)
        self.assertGreater(np.linalg.eigvalsh(metric).min(), 1e-6)
        radial = mesh['xyz'][0]
        self.assertGreater(radial@metric[0]@radial, 1e-6)
        axis = np.array([1., 2., 3.])
        axis /= np.linalg.norm(axis)
        rotation = 2*np.outer(axis, axis)-np.eye(3)
        rotated = dict(vertices=mesh['vertices']@rotation.T, faces=mesh['faces'])
        np.testing.assert_allclose(plate_balance._cell_rotation_metric(rotated),
                                   rotation@metric@rotation.T, atol=3e-12, rtol=0)
        weighted = np.einsum('c,cij->ij', mesh['area_km2'], metric)/mesh['area_km2'].sum()
        np.testing.assert_allclose(weighted, (2/3)*np.eye(3), atol=1e-14)


if __name__ == '__main__':
    unittest.main()
