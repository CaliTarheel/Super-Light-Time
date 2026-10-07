"""Shared two-plate strip work and reciprocal basal reaction."""

import unittest

import numpy as np

from channel_contact_union_work import integrate_two_plate_contact_union


def fixture(*, split=False, count=2, nodes=33):
    offsets = np.linspace(0., 320000., nodes)
    preferred = np.full((count, len(offsets)), 20000.)
    base = np.broadcast_to(
        10000. + 25000. * np.exp(-((offsets - 130000.) / 60000.)**2),
        preferred.shape).copy()
    rows = ([dict(lower_offset_m=80000., upper_offset_m=125000.),
             dict(lower_offset_m=125000., upper_offset_m=180000.)]
            if split else
            [dict(lower_offset_m=80000., upper_offset_m=180000.)])
    lines = dict(line_weight_m=np.array([1000., 2000.][:count]),
                 contact_intervals=[list(rows) for _ in range(count)])
    return lines, offsets, preferred, base


def solve(lines, offsets, preferred, base):
    return integrate_two_plate_contact_union(
        lines, offsets, preferred, base, 0., 1e23, 1e23, 20000.)


class ContactUnionWorkTests(unittest.TestCase):
    def test_adjacent_source_cells_share_contact_nodes_without_duplicate_work(self):
        offsets=np.linspace(0.,160000.,33)
        preferred=(offsets*np.tan(np.deg2rad(50.)))[None,:]
        base=np.full_like(preferred,35000.)
        whole=dict(line_weight_m=np.array([1000.]),contact_intervals=[[
            dict(lower_offset_m=0.,upper_offset_m=80000.)]])
        tiled=dict(line_weight_m=whole['line_weight_m'],contact_intervals=[[
            dict(lower_offset_m=float(a),upper_offset_m=float(b))
            for a,b in zip(np.linspace(0.,80000.,17)[:-1],
                           np.linspace(0.,80000.,17)[1:])]])
        def run(lines):
            return integrate_two_plate_contact_union(lines,offsets,preferred,base,
                0.,1e23,1e23,20000.,free_upper_hinge=True)
        a,b=run(whole),run(tiled)
        np.testing.assert_array_equal(a['contact_mask'],b['contact_mask'])
        np.testing.assert_allclose(a['energy_j'],b['energy_j'],rtol=2e-13)
        np.testing.assert_allclose(a['hinge_line_reaction_n_per_m'],
                                   b['hinge_line_reaction_n_per_m'],rtol=2e-13)

    def test_affine_projected_pair_can_load_real_hinge_contact_ray(self):
        from tests.test_channel_entry_zones import paired
        from entry_projected_contact import projected_pair_areas
        from channel_contact_line_union import spherical_contact_line_union

        surface,_,_,_,_,spec=paired()
        lower=surface['vertices'][surface['faces'][0]]
        upper=surface['vertices'][surface['faces'][1]]
        normal=spec['hinge_normals'][0]
        geometry=projected_pair_areas(lower,upper,normal,50.,refinement=1,
            include_cells=True,coordinate_discretization='affine_nodal')
        lines=spherical_contact_line_union([
            dict(zone_id=str(index),polygon=cell['physical_polygon'])
            for index,cell in enumerate(geometry['contact_cells'])],
            normal,area_tolerance=1e-3)
        index=next(index for index,row in enumerate(lines['contact_intervals'])
                   if row[0]['lower_offset_m']<1e-6)
        one=dict(line_weight_m=lines['line_weight_m'][index:index+1],
                 contact_intervals=[lines['contact_intervals'][index]])
        offsets=np.linspace(0.,160000.,65)
        preferred=(offsets*np.tan(np.deg2rad(50.)))[None,:]
        base=np.full_like(preferred,35000.)
        result=integrate_two_plate_contact_union(one,offsets,preferred,base,
            0.,1e23,1e23,20000.,free_upper_hinge=True)
        self.assertTrue(result['contact_mask'][0,0])
        self.assertGreater(result['hinge_line_reaction_n_per_m'][0],0.)
        np.testing.assert_allclose(result['basal_reference_derivative_n'][0,0],
            one['line_weight_m'][0]*result['hinge_line_reaction_n_per_m'][0],
            rtol=2e-13)
        self.assertGreaterEqual(result['minimum_gap_m'],-1e-6)

    def test_free_hinge_union_carries_half_weighted_edge_force(self):
        offsets=np.linspace(0.,160000.,33)
        preferred=(offsets*np.tan(np.deg2rad(50.)))[None,:]
        base=np.full_like(preferred,35000.)
        lines=dict(line_weight_m=np.array([1200.]),contact_intervals=[[
            dict(lower_offset_m=0.,upper_offset_m=80000.)]])
        def solve_edge(reference):
            return integrate_two_plate_contact_union(
                lines,offsets,preferred,reference,0.,1e23,1e23,20000.,
                free_upper_hinge=True)
        with self.assertRaisesRegex(ValueError,'infeasible at an anchored end'):
            integrate_two_plate_contact_union(
                lines,offsets,preferred,base,0.,1e23,1e23,20000.)
        result=solve_edge(base)
        self.assertGreater(result['hinge_line_reaction_n_per_m'][0],0.)
        np.testing.assert_allclose(result['basal_reference_derivative_n'][0,0],
            result['hinge_line_reaction_n_per_m'][0]*lines['line_weight_m'][0],
            rtol=2e-13)
        delta=1.
        plus,minus=base.copy(),base.copy()
        plus[0,0]+=delta;minus[0,0]-=delta
        measured=(solve_edge(plus)['energy_j']-solve_edge(minus)['energy_j'])/(2.*delta)
        np.testing.assert_allclose(measured,
                                   result['basal_reference_derivative_n'][0,0],rtol=2e-8)

    def test_shared_strip_work_is_invariant_to_contact_zone_split(self):
        whole = fixture()
        divided = fixture(split=True)
        a = solve(*whole)
        b = solve(*divided)
        self.assertGreater(a['energy_j'], 0.)
        np.testing.assert_allclose(a['energy_j'], b['energy_j'], rtol=1e-12)
        np.testing.assert_array_equal(a['contact_mask'], b['contact_mask'])
        self.assertGreater(a['reaction_pa'].max(), 0.)
        self.assertLess(a['maximum_free_force_residual_n_per_m'], 10.)
        self.assertGreaterEqual(a['minimum_gap_m'], -1e-6)
        weighted = dict(whole[0], line_weight_m=2. * whole[0]['line_weight_m'])
        doubled = solve(weighted, *whole[1:])
        np.testing.assert_allclose(doubled['energy_j'], 2. * a['energy_j'])

    def test_integrated_upper_base_virtual_work_matches_reaction(self):
        lines, offsets, preferred, base = fixture()
        original = solve(lines, offsets, preferred, base)
        ray, node = 0, 13
        delta = .01
        plus, minus = base.copy(), base.copy()
        plus[ray, node] += delta
        minus[ray, node] -= delta
        derivative = (solve(lines, offsets, preferred, plus)['energy_j']
                      - solve(lines, offsets, preferred, minus)['energy_j']) / (2. * delta)
        np.testing.assert_allclose(
            derivative, original['basal_reference_derivative_n'][ray, node],
            rtol=3e-5)
        self.assertEqual(original['basal_reference_derivative_n'][0, 0], 0.)
        outside = base.copy()
        outside[0, 0] += 10000.
        self.assertEqual(solve(lines, offsets, preferred, outside)['energy_j'],
                         original['energy_j'])

    def test_normal_grid_refinement_over_same_physical_span(self):
        coarse = solve(*fixture(nodes=33))
        fine = solve(*fixture(nodes=65))
        np.testing.assert_allclose(coarse['energy_j'], fine['energy_j'], rtol=.01)
        np.testing.assert_allclose(coarse['upper_uplift_m'][0, 13],
                                   fine['upper_uplift_m'][0, 26], rtol=.01)

    def test_missing_or_duplicate_contact_nodes_fail_closed(self):
        lines, offsets, preferred, base = fixture(count=1)
        lines['contact_intervals'][0] = [dict(lower_offset_m=80500.,
                                              upper_offset_m=80600.)]
        with self.assertRaisesRegex(ValueError, 'miss or duplicate'):
            solve(lines, offsets, preferred, base)
        lines['contact_intervals'][0] = [
            dict(lower_offset_m=80000., upper_offset_m=130000.),
            dict(lower_offset_m=120000., upper_offset_m=180000.)]
        with self.assertRaisesRegex(ValueError, 'miss or duplicate'):
            solve(lines, offsets, preferred, base)
        lines['contact_intervals'][0] = [
            dict(lower_offset_m=-1., upper_offset_m=180000.)]
        with self.assertRaisesRegex(ValueError, 'beyond the anchored strip'):
            solve(lines, offsets, preferred, base)


if __name__ == '__main__':
    unittest.main()
