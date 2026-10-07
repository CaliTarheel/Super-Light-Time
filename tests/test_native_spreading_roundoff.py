"""Capacity roundoff must not turn old ocean ownership negative."""
import unittest
import json
from pathlib import Path
import numpy as np
from native_spreading import _bounded_pair_areas


class SpreadingRoundoffTests(unittest.TestCase):
    def test_captured_332_myr_transaction(self):
        fixture=json.loads((Path(__file__).parent/'fixtures/spreading_roundoff_332.json').read_text())
        cells=np.asarray(fixture['cells'])
        paired=np.asarray(fixture['paired'])
        area=np.asarray(fixture['cell_area'])
        before=np.asarray(fixture['transported'])
        makers=np.asarray(fixture['makers'])
        def mix(patches):
            deposit=np.bincount(cells,weights=np.repeat(patches,2),minlength=len(area))
            created=np.zeros_like(before)
            np.add.at(created,(makers,cells),np.repeat(patches,2))
            birth=deposit/area
            return before*(1.-birth)+before.sum(axis=0)*created/area,birth
        broken,raw_birth=mix(paired)
        self.assertEqual(np.count_nonzero(raw_birth>1.),6)
        self.assertEqual(np.count_nonzero(broken<0.),24)
        accepted,factors,deposit=_bounded_pair_areas(cells,paired,np.asarray(fixture['capacity']),area)
        repaired,birth=mix(accepted)
        self.assertTrue(np.isfinite(repaired).all())
        self.assertTrue(np.all(repaired>=0.))
        self.assertTrue(np.all((birth>=0.) & (birth<=1.)))
        np.testing.assert_allclose(repaired.sum(axis=0),before.sum(axis=0),rtol=0.,atol=2e-15)
        self.assertGreater(2.*sum(paired-accepted),0.)
        self.assertLess(2.*sum(paired-accepted),1e-8)
        # The generated budget uses exactly the same accepted areas as birth.
        self.assertAlmostEqual(float(birth@area),2.*sum(accepted),delta=1e-9)

    def test_full_cell_preserves_pairing_and_nonnegative_old_support(self):
        cell_area=np.array([5783.233200915268,6000.])
        paired=np.array([cell_area[0]*(1.+4e-15)])
        accepted,factors,deposited=_bounded_pair_areas([0,1],paired,cell_area,cell_area)
        birth=deposited/cell_area
        self.assertTrue(np.all((birth>=0.) & (birth<=1.)))
        self.assertEqual(deposited[0],deposited[1])
        self.assertLess(paired[0]-accepted[0],1e-8)
        # A third owner has no newly generated crust. Its retained support
        # exposes the old negative-factor bug even when the maker stays positive.
        before=np.array([[.3,.1],[.2,.7],[.5,.2]])
        created=np.array([[accepted[0],0.],[0.,accepted[0]],[0.,0.]])
        after=before*(1.-birth)+before.sum(axis=0)*created/cell_area
        self.assertTrue(np.all(after>=0.))
        np.testing.assert_allclose(after.sum(axis=0),before.sum(axis=0),rtol=0.,atol=1e-15)
        np.testing.assert_allclose(after[2],before[2]*(1.-birth),rtol=0.,atol=0.)

    def test_shared_cell_limits_all_incident_pairs_and_both_partners(self):
        pairs=np.array([.3,.7])*(1.+2e-15)
        accepted,_,deposit=_bounded_pair_areas([0,1,0,2],pairs,np.array([1.,2.,2.]),np.array([1.,2.,2.]))
        self.assertLessEqual(deposit[0],1.)
        self.assertEqual(deposit[1],accepted[0])
        self.assertEqual(deposit[2],accepted[1])
        self.assertAlmostEqual(deposit[0],deposit[1]+deposit[2])

    def test_partial_water_capacity_is_respected(self):
        accepted,_,deposit=_bounded_pair_areas([0,1],[2.+1e-7],np.array([2.,3.]),np.array([10.,10.]))
        self.assertTrue(np.all(deposit<=2.))
        self.assertEqual(deposit[0],deposit[1])

    def test_ordinary_geometry_is_bitwise_unchanged(self):
        pairs=np.array([.1,.2])
        accepted,factors,_=_bounded_pair_areas([0,1,0,2],pairs,np.ones(3),np.ones(3))
        np.testing.assert_array_equal(accepted,pairs)
        np.testing.assert_array_equal(factors,np.ones(2))

    def test_material_overfill_and_nonfinite_values_still_fail(self):
        for value in (1.01,np.nan,np.inf,-.1):
            with self.subTest(value=value), self.assertRaises(ValueError):
                _bounded_pair_areas([0,1],[value],np.ones(2),np.ones(2))


if __name__=='__main__': unittest.main()
