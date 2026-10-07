"""Material and physical area checks for projected constant-dip contacts."""

import unittest

import numpy as np

import entry_projected_contact as contact
import channel_pair_zones
from channel_contact_line_union import spherical_contact_line_union
from ridge_geometry import rotate


def unit(rows):
    points=np.asarray(rows,float)
    return points/np.linalg.norm(points,axis=-1,keepdims=True)


LOWER=unit([[1.,.02,-.01],[1.,.04,-.01],[1.,.03,.02]])
FULL=unit([[1.,-.05,-.05],[1.,.10,-.05],[1.,.02,.12]])
PARTIAL=unit([[1.,.01,-.02],[1.,.025,-.02],[1.,.018,.04]])
AWAY=unit([[1.,-.1,-.04],[1.,-.04,-.04],[1.,-.07,.04]])
HINGE=np.array([0.,1.,0.])


class ProjectedContactTests(unittest.TestCase):
    def test_affine_contact_cells_feed_one_nonlocal_hinge_union(self):
        from tests.test_channel_entry_zones import paired

        surface,_,_,_,_,spec=paired()
        lower=surface['vertices'][surface['faces'][0]]
        upper=surface['vertices'][surface['faces'][1]]
        normal=spec['hinge_normals'][0]
        geometry=contact.projected_pair_areas(
            lower,upper,normal,50.,refinement=1,include_cells=True,
            coordinate_discretization='affine_nodal')
        cells=geometry['contact_cells']
        self.assertTrue(cells)
        self.assertEqual(geometry['coordinate_discretization'],'affine_nodal')
        self.assertGreaterEqual(min(float((cell['physical_polygon']@normal).min())
                                    for cell in cells),-1e-12)
        zones=[dict(zone_id=str(index),polygon=cell['physical_polygon'])
               for index,cell in enumerate(cells)]
        union=spherical_contact_line_union(zones,normal,area_tolerance=1e-3)
        np.testing.assert_allclose(union['polygon_area_m2']/1e6,
                                   geometry['physical_contact_area_km2'],rtol=2e-13)
        self.assertLess(abs(union['relative_area_residual']),1e-3)
        self.assertTrue(all(intervals for intervals in union['contact_intervals']))
        np.testing.assert_allclose(sum(cell['material_reference_area_km2'] for cell in cells),
                                   geometry['material_reference_contact_area_km2'],rtol=2e-13)

    def test_affine_pair_refines_without_crossing_material_or_physical_budget(self):
        from tests.test_channel_entry_zones import paired

        surface,_,_,_,_,spec=paired()
        lower=surface['vertices'][surface['faces'][0]]
        upper=surface['vertices'][surface['faces'][1]]
        rows=[contact.projected_pair_areas(lower,upper,spec['hinge_normals'][0],50.,
              refinement=level,coordinate_discretization='affine_nodal')
              for level in (2,3,4)]
        for row in rows:
            self.assertGreater(row['physical_contact_area_km2'],0.)
            self.assertLess(row['material_reference_contact_area_km2'],
                            row['material_reference_entered_area_km2'])
            self.assertLess(row['physical_contact_area_km2'],
                            row['physical_entered_area_km2'])
        self.assertLess(abs(rows[-1]['physical_contact_area_km2']-
                            rows[-2]['physical_contact_area_km2'])/
                        rows[-1]['physical_contact_area_km2'],1e-5)
        with self.assertRaisesRegex(ValueError,'bounded refinement'):
            contact.projected_pair_areas(lower,upper,spec['hinge_normals'][0],50.,
                coordinate_discretization='unknown')

    def test_admissible_native_pair_keeps_material_and_physical_contact_cells(self):
        from tests.test_channel_entry_zones import paired

        surface,_,_,_,_,spec=paired()
        lower=surface['vertices'][surface['faces'][0]]
        upper=surface['vertices'][surface['faces'][1]]
        before_lower,before_upper=lower.copy(),upper.copy()
        source=channel_pair_zones.partition_entry_pair(
            lower,upper,spec['hinge_normals'][0])
        rows=[contact.projected_pair_areas(lower,upper,spec['hinge_normals'][0],50.,
                                           refinement=level,include_cells=True)
              for level in (2,3,4)]
        for row in rows:
            self.assertTrue(row['contact_cells'])
            self.assertAlmostEqual(sum(cell['material_reference_area_km2']
                                       for cell in row['contact_cells'])/
                                   row['material_reference_contact_area_km2'],1.,places=12)
            self.assertAlmostEqual(sum(cell['physical_area_km2']
                                       for cell in row['contact_cells'])/
                                   row['physical_contact_area_km2'],1.,places=12)
            self.assertAlmostEqual(row['legacy_unprojected_contact_area_km2']/
                                   source['contact_area_km2'],1.,places=12)
            self.assertLess(row['max_affine_coordinate_mismatch_m'],50.)
        final=rows[-1]
        self.assertGreater(source['contact_area_km2']/final['physical_contact_area_km2'],1.7)
        self.assertLess(abs(rows[-1]['physical_contact_area_km2']-
                            rows[-2]['physical_contact_area_km2'])/
                        rows[-1]['physical_contact_area_km2'],1e-6)
        for cell in final['contact_cells']:
            self.assertEqual(cell['material_barycentric_polygon'].shape[1],3)
            self.assertTrue(np.allclose(cell['material_barycentric_polygon'].sum(axis=1),1.))
            self.assertGreater(cell['material_reference_area_km2'],0.)
            self.assertGreater(cell['physical_area_km2'],0.)
        np.testing.assert_array_equal(lower,before_lower)
        np.testing.assert_array_equal(upper,before_upper)

    def test_full_and_absent_upper_contacts_preserve_both_area_budgets(self):
        for level in (0,1,2,3):
            full=contact.projected_pair_areas(LOWER,FULL,HINGE,50.,refinement=level)
            empty=contact.projected_pair_areas(LOWER,AWAY,HINGE,50.,refinement=level)
            self.assertAlmostEqual(full['material_reference_contact_area_km2']/
                                   full['material_reference_entered_area_km2'],1.,places=10)
            self.assertAlmostEqual(full['physical_contact_area_km2']/
                                   full['physical_entered_area_km2'],1.,places=12)
            self.assertEqual(empty['physical_contact_area_km2'],0.)
            self.assertEqual(empty['material_reference_contact_area_km2'],0.)
            self.assertLess(full['physical_entered_area_km2'],
                            full['material_reference_entered_area_km2'])

    def test_partial_contact_refines_and_rotates_with_both_plates(self):
        rows=[contact.projected_pair_areas(LOWER,PARTIAL,HINGE,50.,refinement=level)
              for level in range(5)]
        for row in rows:
            self.assertGreater(row['physical_contact_area_km2'],0.)
            self.assertLess(row['physical_contact_area_km2'],row['physical_entered_area_km2'])
            self.assertGreater(row['material_reference_contact_area_km2'],
                               row['physical_contact_area_km2'])
            self.assertLess(row['material_reference_contact_area_km2'],
                            row['material_reference_entered_area_km2'])
            self.assertGreater(row['max_affine_coordinate_mismatch_m'],0.)
        self.assertLess(abs(rows[-1]['physical_contact_area_km2']-
                            rows[-2]['physical_contact_area_km2']),
                        abs(rows[1]['physical_contact_area_km2']-
                            rows[0]['physical_contact_area_km2']))
        self.assertLess(abs(rows[-1]['physical_contact_area_km2']-
                            rows[-2]['physical_contact_area_km2'])/
                        rows[-1]['physical_contact_area_km2'],1e-6)
        self.assertGreater(rows[-1]['physical_contact_area_km2'],
                           5.*rows[-1]['legacy_unprojected_contact_area_km2'])
        axis=[.08,-.03,.1]
        turned=contact.projected_pair_areas(rotate(LOWER,axis),rotate(PARTIAL,axis),
            rotate(HINGE,axis),50.,refinement=3)
        for key in ('material_reference_contact_area_km2','physical_contact_area_km2'):
            np.testing.assert_allclose(turned[key],rows[3][key],rtol=1e-9)

    def test_finite_trench_keeps_entry_and_contact_material_budgets(self):
        midpoint=np.array([1.,0.,0.])
        row=contact.projected_pair_areas(LOWER,FULL,HINGE,50.,
            finite_midpoint=midpoint,finite_half_length_km=50.,refinement=3)
        unlimited=contact.projected_pair_areas(LOWER,FULL,HINGE,50.,refinement=3)
        self.assertGreater(row['material_reference_entered_area_km2'],0.)
        self.assertLess(row['material_reference_entered_area_km2'],
                        unlimited['material_reference_entered_area_km2'])
        np.testing.assert_allclose(row['material_reference_contact_area_km2'],
                                   row['material_reference_entered_area_km2'],rtol=1e-10)

    def test_rejects_invalid_level_and_geometry(self):
        with self.assertRaisesRegex(ValueError,'bounded refinement'):
            contact.projected_pair_areas(LOWER,FULL,HINGE,50.,refinement=8)
        with self.assertRaisesRegex(ValueError,'matched finite endpoints'):
            contact.projected_pair_areas(LOWER,FULL,HINGE,50.,finite_half_length_km=10.)
        with self.assertRaises(ValueError):
            contact.projected_pair_areas(LOWER,LOWER[::-1],HINGE,50.)

    def test_native_source_bookkeeping_fixture_is_beyond_physical_slab_window(self):
        import entry_regions
        from tests.test_channel_entry_zones import native_contact_world

        s,lower_id,upper_id=native_contact_world()
        surface=s.material_surface
        by_id={int(value):index for index,value in enumerate(surface['face_id'])}
        spec=entry_regions.specification(s)
        local=list(spec['face_ids']).index(lower_id)
        lower=surface['vertices'][surface['faces'][by_id[lower_id]]]
        upper=surface['vertices'][surface['faces'][by_id[upper_id]]]
        self.assertGreater(float(np.arcsin(lower@spec['hinge_normals'][local]).min()*6371.),
                           9000.)
        with self.assertRaisesRegex(ValueError,'upper-mantle slab window'):
            contact.projected_pair_areas(lower,upper,spec['hinge_normals'][local],
                                         spec['dip_degrees'][local])


if __name__=='__main__':
    unittest.main()
