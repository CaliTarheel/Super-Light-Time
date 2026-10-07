"""The arc patch and growth geometries solve for area, and cheaply.

Both used a fifty-two step bisection whose every step rebuilt the patch's
spherical areas. _patch alone runs thousands of times per step, and the profiler
attributed most of a production step to this path, so the evaluation count is
part of the contract and is asserted here.
"""
import unittest

import numpy as np

import material_surface
import native_arc_material as arcs


def counted(function):
    """Run `function` while counting spherical-area evaluations."""
    calls = []
    real = material_surface.spherical_face_areas

    def spy(vertices, faces, *args, **kwargs):
        calls.append(1)
        return real(vertices, faces, *args, **kwargs)
    material_surface.spherical_face_areas = spy
    try:
        return function(), len(calls)
    finally:
        material_surface.spherical_face_areas = real


def bisected_scale(evaluate, low, high, target):
    """The replaced fifty-two step bisection, as the reference answer."""
    for _ in range(52):
        middle = (low+high)*.5
        if evaluate(middle)[1].sum() < target:
            low = middle
        else:
            high = middle
    return (low+high)*.5


class ArcGeometrySolverTests(unittest.TestCase):
    def test_patch_areas_sum_to_the_requested_area(self):
        for area in (1., 25., 340., 4000.):
            for centre, strike in (([1., 0., 0.], [0., 1., 0.]), ([0., 0., 1.], [1., 0., 0.]),
                                   ([.3, -.5, .81], [-.6, .2, .77])):
                patch = arcs._patch(centre, strike, area)
                self.assertAlmostEqual(patch['area_km2'].sum()/area, 1., places=9)
                # The reference areas are rescaled to the request exactly.
                self.assertAlmostEqual(patch['reference_area_km2'].sum(), area, places=6)
                self.assertTrue(np.all(patch['area_km2'] > 0.))
                radius = np.linalg.norm(patch['vertices'], axis=1)
                np.testing.assert_allclose(radius, 1., rtol=0, atol=1e-12)

    def test_patch_matches_the_bisection_it_replaced(self):
        # Agreement well inside the bisection's own final bracket, so the
        # solved geometry is the same answer reached by a cheaper route.
        for area in (12., 200., 2500.):
            centre, strike = [.2, .9, -.39], [.7, -.1, .71]
            patch = arcs._patch(centre, strike, area)
            reference = _reference_patch(centre, strike, area)
            self.assertLess(float(np.abs(patch['vertices']-reference).max())*6371e3, 1e-3)

    def test_patch_costs_a_handful_of_evaluations_not_fifty_two(self):
        for area in (5., 120., 3000.):
            _, calls = counted(lambda: arcs._patch([1., 0., 0.], [0., 1., 0.], area))
            self.assertLessEqual(calls, 20, f'{calls} area evaluations for {area} km2')
            self.assertGreaterEqual(calls, 2)

    def test_patch_still_refuses_an_area_beyond_a_minor_cap(self):
        with self.assertRaises(ValueError):
            arcs._patch([1., 0., 0.], [0., 1., 0.], 4e8)
        with self.assertRaises(ValueError):
            arcs._patch([1., 0., 0.], [0., 1., 0.], 0.)

    def test_solver_keeps_its_bracket_and_reports_an_unbracketed_target(self):
        def evaluate(scale):
            return None, np.array([scale**2])
        solved = arcs._solve_scale(evaluate, 0., 4., 9., low_area=0., high_area=16.)
        self.assertAlmostEqual(solved, 3., places=9)
        for target, low, high in ((25., 0., 4.), (-1., 0., 4.)):
            with self.assertRaises(ValueError):
                arcs._solve_scale(evaluate, low, high, target, low_area=0., high_area=16.)

    def test_solver_survives_a_flat_region(self):
        # A curve that is flat away from the root would stall a pure secant;
        # the bracket must still deliver the answer.
        def evaluate(scale):
            return None, np.array([0. if scale < 3. else (scale-3.)**3])
        solved = arcs._solve_scale(evaluate, 0., 6., 1., low_area=0., high_area=27.)
        self.assertAlmostEqual(solved, 4., places=6)

    def test_growth_geometry_adds_exactly_the_requested_area(self):
        surface = _oval_surface(900.)
        selected = np.arange(len(surface['faces']))
        for addition in (30., 300.):
            plan, calls = counted(lambda: arcs._grow_geometry(surface, selected, addition))
            self.assertIsNotNone(plan)
            self.assertAlmostEqual(float(plan['added_area_km2'].sum()), addition, places=6)
            self.assertLessEqual(calls, 20, f'{calls} area evaluations for +{addition} km2')
            grown = float(plan['area_km2'].sum())
            self.assertAlmostEqual(grown/(surface['area_km2'][selected].sum()+addition), 1., places=9)

    def test_growth_geometry_refuses_more_than_a_minor_cap_holds(self):
        surface = _oval_surface(900.)
        selected = np.arange(len(surface['faces']))
        self.assertIsNone(arcs._grow_geometry(surface, selected, 4e8))


def _reference_patch(centre, strike, area):
    """Rebuild a patch with the original bisection, for comparison."""
    centre = arcs._unit(np.asarray(centre, float))
    direction = np.asarray(strike, float)
    along = arcs._unit(direction-centre*np.dot(centre, direction))
    if np.linalg.norm(along) < .5:
        along = arcs._unit(np.cross(centre, [0., 0., 1.] if abs(centre[2]) < .9 else [0., 1., 0.]))
    across = np.cross(centre, along)
    phi = np.arange(arcs.RAYS)*2*np.pi/arcs.RAYS
    modulation = 1.+.075*np.cos(3*phi+.4)
    xy = np.column_stack((1.5*np.cos(phi), np.sin(phi)/1.5))*modulation[:, None]
    coordinates = np.vstack((np.zeros((1, 2)), .48*xy, xy))
    faces = []
    for i in range(arcs.RAYS):
        j = (i+1) % arcs.RAYS
        faces.extend(((0, 1+i, 1+j), (1+i, 1+arcs.RAYS+i, 1+arcs.RAYS+j), (1+i, 1+arcs.RAYS+j, 1+j)))
    faces = np.asarray(faces, np.int64)
    tangents = coordinates[:, 0, None]*along+coordinates[:, 1, None]*across
    distance = np.linalg.norm(tangents, axis=1)
    bearing = tangents/np.maximum(distance[:, None], 1e-30)

    def evaluate(scale):
        angles = distance*scale
        vertices = centre*np.cos(angles)[:, None]+bearing*np.sin(angles)[:, None]
        vertices[0] = centre
        return vertices, material_surface.spherical_face_areas(vertices, faces)
    scale = bisected_scale(evaluate, 0., 1.1/distance.max(), area)
    return evaluate(scale)[0]


def _oval_surface(area_km2):
    """A single arc patch as a standalone material surface."""
    patch = arcs._patch([1., 0., 0.], [0., 1., 0.], area_km2)
    faces = patch['faces']
    return material_surface.initialize_surface(
        patch['vertices'], faces, np.zeros(len(faces), int), np.full(len(faces), 3, np.uint8))


if __name__ == '__main__':
    unittest.main()
