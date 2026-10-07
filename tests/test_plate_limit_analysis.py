"""Rigid-plastic limit analysis: max-flow exactness, envelopes, analytic cap."""
import itertools
import math
import unittest

import numpy as np

import mesh_geometry
import plate_limit_analysis as pla


class StrengthEnvelopeTests(unittest.TestCase):
    def test_ocean_strengthens_with_age_and_continent_weakens_with_heat_flow(self):
        ocean = [pla.oceanic_strength_n_per_m(age) for age in (10, 20, 40, 80, 150)]
        self.assertTrue(np.all(np.diff(ocean) > 0.))
        continent = [pla.continental_strength_n_per_m(q) for q in (45, 55, 65, 75, 85)]
        self.assertTrue(np.all(np.diff(continent) < 0.))
        # Scotese Rule III in numbers: ordinary continent is weaker than
        # mature ocean; very young ocean is comparable.
        self.assertLess(pla.continental_strength_n_per_m(65), pla.oceanic_strength_n_per_m(40)/2)
        self.assertLess(pla.oceanic_strength_n_per_m(10), pla.continental_strength_n_per_m(65))
        # Brittle compression and strike-slip exceed brittle tension.
        for mode in ('compression', 'shear'):
            self.assertGreater(pla.oceanic_strength_n_per_m(60, mode), pla.oceanic_strength_n_per_m(60))

    def test_cell_strengths_select_by_crust(self):
        tension, _, _ = pla.cell_strengths([0, 1, 2, 3], [60., 0., 0., 0.])
        self.assertAlmostEqual(tension[0], np.interp(60, pla._AGE_TABLE,
                               [pla.oceanic_strength_n_per_m(a) for a in pla._AGE_TABLE]))
        self.assertAlmostEqual(tension[1], pla.continental_strength_n_per_m(65.))
        self.assertGreater(tension[2], tension[1])
        self.assertLess(tension[3], tension[1])


class MaxFlowTests(unittest.TestCase):
    def test_best_piece_matches_enumeration(self):
        rng = np.random.default_rng(4)
        for _ in range(150):
            n = int(rng.integers(2, 8))
            a, b = rng.integers(0, n, 12), rng.integers(0, n, 12)
            keep = a != b
            a, b = a[keep], b[keep]
            weight, forward, backward = rng.normal(size=n), rng.random(len(a)), rng.random(len(a))

            def value(piece):
                return (weight[piece].sum()-forward[piece[a] & ~piece[b]].sum()
                        - backward[piece[b] & ~piece[a]].sum())
            best = max(value(np.array(bits, bool)) for bits in itertools.product((0, 1), repeat=n))
            found, piece = pla._best_piece(weight, a, b, forward, backward)
            self.assertAlmostEqual(found, best, places=9)
            self.assertAlmostEqual(value(piece), best, places=9)


def hemisphere(level, pull, strength):
    mesh = mesh_geometry.icosphere(level)
    xyz, (ea, eb), mid = mesh['xyz'], mesh['edge_faces'].T, mesh['edge_mid']
    length = mesh['edge_length']*1e3
    inside = xyz[:, 2] > 0.
    torque = np.zeros_like(xyz)
    for e in np.flatnonzero(inside[ea] != inside[eb]):
        here, there = (ea[e], eb[e]) if inside[ea[e]] else (eb[e], ea[e])
        normal = xyz[there]-xyz[here]
        normal -= mid[e]*(normal@mid[e])
        torque[here] += pla.RADIUS_M*np.cross(mid[e], normal/np.linalg.norm(normal))*pull*length[e]
    cells = np.flatnonzero(inside)
    local = -np.ones(len(xyz), int)
    local[cells] = np.arange(len(cells))
    keep = inside[ea] & inside[eb]
    edges = np.column_stack((local[ea[keep]], local[eb[keep]]))
    s = np.broadcast_to(np.asarray(strength, float), (keep.sum(),)) if np.isscalar(strength) else strength
    return xyz[cells], torque[cells], edges, mid[keep], length[keep], s, xyz, cells, ea[keep], eb[keep]


class AnalyticCapTests(unittest.TestCase):
    def test_uniform_rim_pull_on_uniform_hemisphere_gives_pull_over_strength(self):
        # A hemisphere pulled uniformly outward at its rim is in isotropic
        # membrane tension N0, so its collapse load factor is N0/S. The best
        # mechanism opens a meridian about an axis on the rim.
        pull, strength = 5e12, 4e12
        points, torque, edges, mid, length, s, *_ = hemisphere(3, pull, strength)
        self.assertLess(np.linalg.norm(torque.sum(axis=0)), 1e-9*np.linalg.norm(torque, axis=1).sum())
        best = pla.analyse(points, torque, edges, mid, length, (s, 3*s, s/math.sqrt(3)))
        self.assertAlmostEqual(best['ratio'], pull/strength, delta=.05*pull/strength)
        # Upper bound: a discrete admissible mechanism never under-estimates.
        self.assertGreaterEqual(best['ratio'], pull/strength*(1-1e-3))
        self.assertLess(abs(best['axis'][2]), .1)
        self.assertAlmostEqual(best['piece'].mean(), .5, delta=.1)

    def test_forbidden_edges_off_the_optimal_cut_change_nothing(self):
        # Uncuttable edges carry enormous capacities; they must not inflate the
        # max-flow tolerance and hide the optimum, which can avoid them.
        pull, strength = 5e12, 4e12
        points, torque, edges, mid, length, s, xyz, cells, ea, eb = hemisphere(3, pull, strength)
        cluster = (xyz[:, 0] > .75) & (xyz[:, 2] < .3)
        blocked = cluster[ea] & cluster[eb]
        self.assertTrue(blocked.any())
        free = pla.analyse(points, torque, edges, mid, length, (s, 3*s, s/math.sqrt(3)))
        held = pla.analyse(points, torque, edges, mid, length, (s, 3*s, s/math.sqrt(3)), forbidden=blocked)
        self.assertFalse(np.any(held['cut'] & blocked))
        self.assertAlmostEqual(held['ratio'], free['ratio'], delta=.03*free['ratio'])

    def test_weak_band_localizes_and_forbidden_edges_are_respected(self):
        pull, strong, weak = 5e12, 2e13, 3e12
        points, torque, edges, mid, length, _, xyz, cells, ea, eb = hemisphere(3, pull, 0.)
        # A weak meridional band (continent-like) across the cap.
        band = np.abs(xyz[:, 0]) < .25
        weak_edge = band[ea] | band[eb]
        s = np.where(weak_edge, weak, strong)
        best = pla.analyse(points, torque, edges, mid, length, (s, 3*s, s/math.sqrt(3)))
        self.assertGreater(length[best['cut'] & weak_edge].sum()/length[best['cut']].sum(), .8)
        blocked = pla.analyse(points, torque, edges, mid, length, (s, 3*s, s/math.sqrt(3)),
                              forbidden=weak_edge)
        self.assertFalse(np.any(blocked['cut'] & weak_edge))
        self.assertLess(blocked['ratio'], best['ratio'])
        self.assertLess(blocked['ratio'], pull/strong*1.2)


if __name__ == '__main__':
    unittest.main()
