"""Independent work check for a proposed local stack-plus-entry closure.

This evaluates the existing two energy terms together without enabling active
entry/stack overlap in the simulator. A passing algebraic check does not prove
that the assumed vertical geometry is geologically adequate.
"""
import unittest

import numpy as np

import continental_entry as entry
import gravitational_relaxation as gravity
import material_surface
import mesh_coverage
from ridge_geometry import rotate
from tests.test_continental_entry import planar_oracle, unit


FACES=np.array([[0,1,2],[3,4,5]])
SHEETS=np.array([1,2])
NORMAL=np.array([[0.,1.,0.]])
DIP=50.
THICKNESS=35.
RHO_LOWER=2800.
RHO_UPPER=2800.
RHO_MANTLE=3300.
G=9.81


def geometry():
    lower=unit([[1.,-.02,-.02],[1.,.03,-.01],[1.,.01,.04]])
    return np.vstack((lower,rotate(lower,[0.,0.,.008])))


def physical_energy(points,normal,volumes):
    """Column moment, pair interaction and extra mantle descent in SI units."""
    area=material_surface.spherical_face_areas(points,FACES)
    h=volumes/area
    # The advected reference thickness is 35 km. In this virtual-work fixture
    # rigid rotation preserves each face area, leaving the self term zero.
    self_energy=.5*G*RHO_LOWER*(1.-RHO_LOWER/RHO_MANTLE)*1e12*np.sum(area*(h-THICKNESS)**2)
    pair=mesh_coverage.material_overlaps(points,FACES,SHEETS)['area_km2']
    stack=G*RHO_UPPER*(1.-RHO_LOWER/RHO_MANTLE)*1e12*float(pair.sum())*h[0]*h[1]
    q=6371e3*np.arcsin(points[FACES[0]]@normal[0])
    mean_depth=planar_oracle(q)[0]*np.sin(np.deg2rad(DIP))
    descent=G*(RHO_MANTLE-RHO_LOWER)*volumes[0]*1e9*mean_depth
    return self_energy+stack+descent


class EntryStackWorkOracleTests(unittest.TestCase):
    def test_ordered_intersections_resolve_local_depth_without_assigning_contact(self):
        points=geometry()
        fractions=entry.reference_entry_fractions(points[FACES[:1]],NORMAL,radius_km=6371.)
        area_only=entry.entry_stack_overlap_ledger(points,FACES,SHEETS,np.array([0]),
            NORMAL,fractions,radius_km=6371.)
        ledger=entry.ordered_entry_stack_depth_ledger(points,FACES,SHEETS,
            np.array([0]),NORMAL,np.array([DIP]),fractions,{2:{1}},radius_km=6371.)
        self.assertEqual(len(ledger),2)
        self.assertTrue(all(row['entered_is_lower'] and row['upper_face']==1
                            and row['lower_face']==0 for row in ledger))
        np.testing.assert_allclose(sum(row['area_km2'] for row in ledger),
                                   sum(row['area_km2'] for row in area_only),rtol=2e-13)
        self.assertLess(min(row['minimum_entry_depth_m'] for row in ledger),1e-6)
        self.assertGreater(max(row['maximum_entry_depth_m'] for row in ledger),100e3)
        self.assertLess(max(row['maximum_entry_depth_m'] for row in ledger),150e3)
        horizontal=entry.ordered_entry_stack_depth_ledger(points,FACES,SHEETS,
            np.array([0]),NORMAL,np.array([DIP]),fractions,{2:{1}},radius_km=6371.,
            coordinate_mode='horizontal_surface')
        depth_ratio=1./np.cos(np.deg2rad(DIP))
        np.testing.assert_allclose([row['maximum_entry_depth_m'] for row in horizontal],
            [row['maximum_entry_depth_m']*depth_ratio for row in ledger],rtol=3e-15)
        np.testing.assert_array_equal([row['area_km2'] for row in horizontal],
                                      [row['area_km2'] for row in ledger])
        rotated=rotate(points,[.2,-.1,.3]);normal=rotate(NORMAL,[.2,-.1,.3])
        turned=entry.ordered_entry_stack_depth_ledger(rotated,FACES,SHEETS,
            np.array([0]),normal,np.array([DIP]),fractions,{2:{1}},radius_km=6371.)
        np.testing.assert_allclose([row['area_km2'] for row in ledger],
                                   [row['area_km2'] for row in turned],rtol=3e-12)
        np.testing.assert_allclose([row['maximum_entry_depth_m'] for row in ledger],
                                   [row['maximum_entry_depth_m'] for row in turned],rtol=3e-12)
        reversed_order=entry.ordered_entry_stack_depth_ledger(points,FACES,SHEETS,
            np.array([0]),NORMAL,np.array([DIP]),fractions,{1:{2}},radius_km=6371.)
        self.assertTrue(all(not row['entered_is_lower'] and row['upper_face']==0
                            and row['lower_face']==1 for row in reversed_order))
        with self.assertRaisesRegex(ValueError,'no persistent vertical order'):
            entry.ordered_entry_stack_depth_ledger(points,FACES,SHEETS,
                np.array([0]),NORMAL,np.array([DIP]),fractions,{},radius_km=6371.)

    def test_independent_energy_and_two_plate_virtual_work(self):
        points=geometry()
        area=material_surface.spherical_face_areas(points,FACES)
        volumes=area*THICKNESS
        profile=dict(dense_fraction=np.zeros(2),sheet_order={2:{1}})
        self.assertGreater(mesh_coverage.material_overlaps(points,FACES,SHEETS)['area_km2'].sum(),0.)
        fraction=entry.reference_entry_fractions(points[FACES[:1]],NORMAL,radius_km=6371.)
        self.assertGreater(fraction[0],0.)
        self.assertLess(fraction[0],1.)
        self.assertTrue(entry.entry_stack_overlap_ledger(points,FACES,SHEETS,np.array([0]),
            NORMAL,fraction,radius_km=6371.))

        column,gradient,_=gravity.energy_gradient(points,FACES,volumes/area,
            np.full(2,THICKNESS),SHEETS,density_profile=profile)
        ramp=entry.evaluate(points,FACES[:1],volumes[:1],
            volumes[:1]*RHO_LOWER*1e9,NORMAL,[DIP])
        actual=column*gravity.ENERGY_UNIT_J+ramp['energy_j']
        expected=physical_energy(points,NORMAL,volumes)
        np.testing.assert_allclose(actual,expected,rtol=3e-13)

        gradient=gradient*gravity.ENERGY_UNIT_J+ramp['vertex_gradient_j']
        lower_torque=np.cross(points[:3],gradient[:3]).sum(axis=0)
        upper_torque=(np.cross(points[3:],gradient[3:]).sum(axis=0)
                      +ramp['hinge_potential_torque_n_m'][0])
        np.testing.assert_allclose(lower_torque+upper_torque,0.,
            atol=max(np.linalg.norm(lower_torque),1.)*3e-12)
        axis=unit([.3,.5,-.2]);step=2e-7
        for owner,predicted in ((0,lower_torque),(1,upper_torque)):
            def shifted(sign):
                moved=points.copy()
                select=slice(3*owner,3*owner+3)
                moved[select]=rotate(moved[select],axis*(sign*step))
                normal=rotate(NORMAL,axis*(sign*step)) if owner==1 else NORMAL
                return physical_energy(moved,normal,volumes)
            observed=(shifted(1.)-shifted(-1.))/(2*step)
            np.testing.assert_allclose(predicted@axis,observed,rtol=2e-6)


if __name__=='__main__':unittest.main()
