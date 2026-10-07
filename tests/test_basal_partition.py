"""Independent common-measure and finite-domain acceptance checks."""
from pathlib import Path
import sys
import unittest
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from benchmarks.collision_architecture import basal_partition as basal
import plate_balance

R = 6371000.
C = np.eye(3)
M = np.array([0., 1., 1.])/np.sqrt(2.)
HALF = np.array([C[0], C[1], M])
OTHER = np.array([C[0], M, C[2]])
AREA = .5*np.pi*R**2


def record(triangle, face=10, sheet=1, owner=0):
    return dict(triangle=triangle, face_id=face, sheet_id=sheet, owner=owner)


def allocate(material=(), above=(), support=None, control=C, saved=AREA, **kwargs):
    return basal.partition_cell(control, material, above, radius_m=R,
        saved_cell_area_m2=saved, support={0: 1.} if support is None else support,
        basal_drag_pa_s_per_m=1.5e15, **kwargs)


class BasalPartitionTests(unittest.TestCase):
    def test_empty_fractional_cell_matches_native_finite_cell_metric(self):
        result = allocate(support={0: .25, 1: .75})
        native = plate_balance._cell_rotation_metric(dict(vertices=C, faces=np.array([[0,1,2]])))[0]*AREA
        np.testing.assert_allclose(result['native_cell_metric_m2'], native, rtol=2e-15)
        for owner, weight in ((0,.25), (1,.75)):
            np.testing.assert_allclose(result['remaining_metric_m2'][owner], weight*native, rtol=2e-15)
            np.testing.assert_array_equal(result['material_metric_m2'][owner], np.zeros((3,3)))
        # Analytic octant: diagonal2A/3; off-diagonal-R²/3.
        analytic = np.full((3,3), -R**2/3)
        np.fill_diagonal(analytic, 2*AREA/3)
        np.testing.assert_allclose(native, analytic, rtol=2e-15)

    def test_partial_material_is_one_bottom_patch_plus_positive_complement(self):
        result = allocate([record(HALF)])
        material = [p for p in result['pieces'] if p['bottom_face'] is not None]
        uncovered = [p for p in result['pieces'] if p['bottom_face'] is None]
        self.assertEqual(len(material),1); self.assertEqual(len(uncovered),1)
        self.assertAlmostEqual(material[0]['area_m2']/AREA, .5, places=14)
        self.assertAlmostEqual(uncovered[0]['area_m2']/AREA, .5, places=14)
        a, b = result['material_metric_m2'][0], result['remaining_metric_m2'][0]
        np.testing.assert_allclose(a+b, result['native_cell_metric_m2'], rtol=2e-14)
        self.assertGreater(np.linalg.eigvalsh(a).min(),0)
        self.assertGreater(np.linalg.eigvalsh(b).min(),0)

    def test_same_owner_stack_only_bottom_sheet_touches_mantle(self):
        result = allocate([record(C,face=10,sheet=1),record(HALF,face=11,sheet=2)],above=[(2,1)])
        self.assertTrue(all(p['bottom_face']==10 for p in result['pieces']))
        np.testing.assert_allclose(result['material_metric_m2'][0],result['native_cell_metric_m2'],rtol=2e-14)
        np.testing.assert_array_equal(result['remaining_metric_m2'][0],np.zeros((3,3)))
        # Upper owner differs, but it still gets NO basal contact or drag.
        changed = allocate([record(C,face=10,sheet=1),record(HALF,face=11,sheet=2,owner=1)],above=[(2,1)])
        self.assertTrue(all(p['owner']==0 for p in changed['pieces']))

    def test_order_requires_bottom_unique_and_rejects_same_sheet_overlap(self):
        with self.assertRaisesRegex(ValueError,'unique ordered bottom'):
            allocate([record(C),record(HALF,face=11,sheet=2)])
        with self.assertRaisesRegex(ValueError,'within one material sheet'):
            allocate([record(C),record(HALF,face=11,sheet=1)])
        with self.assertRaisesRegex(ValueError,'cycle'):
            allocate([record(C)],above=[(1,2),(2,1)])

    def test_area_budget_does_not_authorize_mixed_support(self):
        # Material occupies exactlyhalfcell and owner0 owns half the cell measure.
        # Those equal totals cannot supply a local positive remainder.
        with self.assertRaisesRegex(basal.UnsupportedBasalOwnership,'subcell mantle ownership'):
            allocate([record(HALF)],support={0:.5,1:.5})
        full = allocate()['native_cell_metric_m2']
        material = allocate([record(HALF)])['material_metric_m2'][0]
        eigenvalues = np.linalg.eigvalsh(.5*full-material)
        self.assertLess(eigenvalues[0], -1e11)
        self.assertGreater(eigenvalues[-1], 1e11)
        with self.assertRaises(basal.UnsupportedBasalOwnership):
            allocate([record(HALF)], support={1:1.})

    def test_native_saved_measure_weight_is_preserved_explicitly(self):
        original = allocate([record(HALF)])
        changed = allocate([record(HALF)],saved=AREA*1.125)
        self.assertAlmostEqual(changed['native_measure_density'],1.125)
        for key in ('material_metric_m2','remaining_metric_m2'):
            np.testing.assert_allclose(changed[key][0],1.125*original[key][0],rtol=2e-15)

    def test_subdivision_and_rotation_preserve_full_rigid_work(self):
        full = allocate([record(C)])
        split = allocate([record(HALF),record(OTHER,face=11)])
        np.testing.assert_allclose(split['material_metric_m2'][0],full['material_metric_m2'][0],rtol=2e-14)
        # A90degree exact rotation avoids changing shared represented planes.
        rotation=np.array([[0.,-1.,0.],[1.,0.,0.],[0.,0.,1.]])
        rotated=allocate([record(C@rotation.T)],control=C@rotation.T)
        np.testing.assert_allclose(rotated['material_metric_m2'][0],rotation@full['material_metric_m2'][0]@rotation.T,rtol=2e-14)
        omega=np.array([.3,-.2,.7])*1e-15
        observed=omega@(split['material_metric_m2'][0]+split['remaining_metric_m2'][0])@omega
        expected=omega@full['native_cell_metric_m2']@omega
        self.assertAlmostEqual(observed/expected,1.,places=13)

    def test_tiny_positive_piece_is_retained_without_area_floor(self):
        width=1e-5
        tiny=np.array([[1.,0.,0.],[1.,width,0.],[1.,0.,width]])
        tiny/=np.linalg.norm(tiny,axis=1)[:,None]
        result=allocate([record(tiny)])
        pieces=[p for p in result['pieces'] if p['bottom_face'] is not None]
        self.assertEqual(len(pieces),1)
        self.assertGreater(pieces[0]['area_m2'],0.)
        self.assertLess(pieces[0]['area_m2'],1e6)
        with self.assertRaises(basal.UnsupportedBasalOwnership):
            allocate([record(tiny)],support={0:.999,1:.001})

    def test_invalid_inputs_fail_before_allocation(self):
        for support in ({0:1.1},{0:-.1,1:1.1},{0:True},{0:float('nan')}):
            with self.assertRaises(ValueError):
                allocate(support=support)
        for triangle in (C[::-1], C.astype(complex), C.astype(object), C*2):
            with self.assertRaises(ValueError):
                allocate([record(triangle)])
        with self.assertRaisesRegex(ValueError,'unique'):
            allocate([record(HALF),record(OTHER)])

    def test_explicit_resolved_policy_changes_owner_work_with_positive_pieces(self):
        result=allocate([record(HALF)], support={0:.5,1:.5},allocation_policy='resolved-material-bottom')
        pure=allocate([record(HALF)])
        np.testing.assert_allclose(result['material_metric_m2'][0],pure['material_metric_m2'][0],rtol=2e-14)
        for owner in (0,1):
            np.testing.assert_allclose(result['remaining_metric_m2'][owner],.5*pure['remaining_metric_m2'][0],rtol=2e-14)
        delta=result['per_owner_legacy_metric_change_m2']
        np.testing.assert_allclose(delta[0]+delta[1],np.zeros((3,3)),atol=AREA*2e-15)
        self.assertLess(np.linalg.eigvalsh(delta[1]).min(),0.)
        self.assertEqual(result['reassigned_material_pieces'],1)
        self.assertAlmostEqual(result['reassigned_material_area_m2']/AREA,.5,places=13)
        self.assertEqual(result['allocation_policy'],'resolved-material-bottom')
        with self.assertRaises(ValueError): allocate(allocation_policy='dominant-owner')
        # A pure legacy owner can likewise change only by explicit selection.
        changed=allocate([record(C)],support={0:0.,1:1.},allocation_policy='resolved-material-bottom')
        np.testing.assert_allclose(changed['material_metric_m2'][0],changed['native_cell_metric_m2'],rtol=2e-14)
        self.assertGreater(np.linalg.norm(changed['per_owner_legacy_metric_change_m2'][1]),0.)

    def test_control_subdivision_and_three_layer_order(self):
        full=allocate([record(C,face=1,sheet=1),record(C,face=2,sheet=2),record(C,face=3,sheet=3)],above=[(3,2),(2,1)])
        permuted=allocate([record(C,face=3,sheet=3),record(C,face=1,sheet=1),record(C,face=2,sheet=2)],above=[(2,1),(3,2)])
        self.assertEqual(full['pieces'][0]['bottom_face'],1)
        np.testing.assert_array_equal(full['material_metric_m2'][0],permuted['material_metric_m2'][0])
        divided=[allocate([record(C)],control=triangle,saved=AREA/2) for triangle in (HALF,OTHER)]
        total=sum((p['material_metric_m2'][0] for p in divided),np.zeros((3,3)))
        np.testing.assert_allclose(total,full['material_metric_m2'][0],rtol=2e-14)
        axis=C.sum(axis=0); axis/=np.linalg.norm(axis)
        self.assertAlmostEqual((axis@total@axis)/(axis@full['native_cell_metric_m2']@axis),1.,places=13)

    def test_unrepresentable_derived_si_values_reject(self):
        for radius,beta in ((R,1e300),(1e300,1.),(1e-300,1.)):
            with self.assertRaises(ValueError):
                basal.partition_cell(C,[],[],radius_m=radius,saved_cell_area_m2=AREA,
                    support={0:1.},basal_drag_pa_s_per_m=beta)


if __name__=='__main__':
    unittest.main()
