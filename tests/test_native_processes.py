"""Native process ledgers and physical juvenile-material geometry."""
import unittest
import numpy as np

from native_engine import Simulation
import native_processes as processes
import mesh_transport
import material_surface
import trench_history


def ocean_fixture(level=2, two=False):
    s=Simulation(dict(width=96,height=48,mesh_level=level,duration_myr=4,plate_count=4),
        dict(width=48,height=24,crust=np.zeros(48*24,np.uint8)))
    # Existing transport/source unit cases deliberately exercise the previous
    # small-patch birth law. New column/capacity/cohort tests explicitly enable
    # version one and verify physical formation, budgets and source geometry.
    s.native_arc_birth_profile_version=0
    # Profile-only legacy tests retain their nominal footprint; new footprint
    # capacity tests explicitly enable version one after constructing this fixture.
    s.native_arc_footprint_version=0
    s.omega[:]=0.;s.mantle[:]=0.
    s.age[:]=80.;s.primordial_fraction[:]=1.
    if two:
        # This fixture explicitly replaces the source partition with a new
        # hemispheric ocean world; its original imported interfaces no longer apply.
        s.native_initial_ownership_consumed=True
        s.active[:2]=True;s.count=2;s.plate_uid[1]=2;s.names[1]='Other ocean'
        s.plate=(s.xyz[:,1]>=0).astype(np.int16)
        s.support[:]=0.;s.support[s.plate,np.arange(s.n)]=1.
    return s


def reset_trenches(s):
    s._boundaries()
    del s.trench_systems
    trench_history.initialize(s)


class NativeProcessTests(unittest.TestCase):
    def test_common_euler_motion_carries_ocean_age_and_damage(self):
        s=ocean_fixture(two=True)
        s.omega[:2]=[.004,-.002,.003]
        s.age=80.+20.*s.xyz[:,2]
        s.ocean_history['damage']=.3+.2*s.xyz[:,0]
        s.ocean_history['weakness']=.4+.1*s.xyz[:,1]
        reset_trenches(s)
        initial=s.age.copy();damage=s.ocean_history['damage'].copy();support=s.support.copy()
        age_expected=mesh_transport.advect(s.native_mesh,initial+2.,s.omega[0],2.)
        damage_expected=mesh_transport.advect(s.native_mesh,damage,s.omega[0],2.)
        processes.advect_ocean(s,2.)
        np.testing.assert_allclose(s.age,age_expected,atol=6e-14)
        np.testing.assert_allclose(s.ocean_history['damage'],damage_expected,atol=3e-16)
        np.testing.assert_allclose(s.primordial_fraction,1.,atol=3e-16)
        self.assertGreater(np.max(np.abs(s.support-support)),0.)
        self.assertEqual(s.process_totals['ocean_created_km2'],0.)
        self.assertLess(s.process_totals['ocean_consumed_km2'],1e-6)

    def test_measured_native_ridge_birth_is_paired_and_changes_only_new_fraction(self):
        s=ocean_fixture(level=3,two=True)
        s.omega[0]=[0,0,-.005];s.omega[1]=[0,0,.005]
        s.ocean_history['damage'][:]=.6
        reset_trenches(s)
        processes.advect_ocean(s,2.)
        d=s.spreading_diagnostics
        self.assertGreater(d['generated_area_km2'],0.)
        self.assertAlmostEqual(d['side_p_area_km2'],d['side_q_area_km2'],places=7)
        birth=1-s.primordial_fraction
        np.testing.assert_allclose(s.age,82.*(1-birth)+birth,atol=7e-14)
        np.testing.assert_allclose(s.ocean_history['damage'],.6*(1-birth),atol=4e-16)
        self.assertAlmostEqual(float(s.cell_area@birth),d['generated_area_km2'],places=6)
        self.assertTrue(np.all((birth>=0)&(birth<=1)))

    def test_ocean_consumption_requires_mature_local_trench(self):
        young=ocean_fixture(level=3,two=True)
        mature=ocean_fixture(level=3,two=True)
        for s in (young,mature):
            s.omega[0]=[0,0,-.005];s.omega[1]=[0,0,.005]
            reset_trenches(s)
        for _ in range(5):
            mature.t+=2.;trench_history.update(mature,2.)
        processes.advect_ocean(young,2.)
        processes.advect_ocean(mature,2.)
        self.assertEqual(young.ocean_diagnostics['consumed_area_km2'],0.)
        self.assertGreater(mature.ocean_diagnostics['consumed_area_km2'],0.)
        self.assertGreater(young.ocean_diagnostics['unresolved_contact_overlap_km2'],0.)
        self.assertLess(mature.ocean_diagnostics['unresolved_contact_overlap_km2'],
                        young.ocean_diagnostics['unresolved_contact_overlap_km2'])

    def test_arc_birth_has_exact_small_area_and_grows_existing_geometry(self):
        s=ocean_fixture()
        result=processes.add_arc_crust(s,np.array([17,17]),np.array([600.,400.]))
        self.assertEqual(result['new_faces'],24)
        self.assertEqual(len(s.material_surface['faces']),24)
        self.assertEqual(len(s.mass),24)
        self.assertAlmostEqual(s.mass.sum(),1000.)
        self.assertLess(s.mass.sum(),s.cell_area[17]*.01)
        np.testing.assert_allclose(s.material_surface['area_km2'],s.mass,rtol=3e-11)
        face_id=s.parcel_patch.copy();old=s.material_surface['vertices'].copy()
        processes.add_arc_crust(s,np.array([17]),np.array([2500.]))
        self.assertEqual(len(s.mass),24)
        np.testing.assert_array_equal(s.parcel_patch,face_id)
        self.assertAlmostEqual(s.mass.sum(),3500.)
        np.testing.assert_allclose(s.material_surface['area_km2'],s.mass,rtol=3e-11)
        self.assertGreater(np.max(np.abs(s.material_surface['vertices']-old)),0.)
        self.assertEqual(s.process_totals['arc_added_km2'],3500.)
        self.assertEqual(len(s.structure['thickness_km']),len(s.mass))
        self.assertEqual(len(s.trace_structure['thickness_km']),len(s.trace_id))

    def test_small_arc_budget_is_retained_and_materializes_without_full_cell(self):
        s=ocean_fixture()
        first=processes.add_arc_crust(s,np.array([9]),np.array([.0004]))
        self.assertEqual(first['new_faces'],0)
        self.assertEqual(len(s.mass),0)
        self.assertAlmostEqual(first['pending_area_km2'],.0004)
        second=processes.add_arc_crust(s,np.array([9]),np.array([.0007]))
        self.assertEqual(second['new_faces'],24)
        self.assertAlmostEqual(s.mass.sum(),.0011)
        self.assertEqual(second['pending_area_km2'],0.)
        np.testing.assert_allclose(s.material_surface['area_km2'],s.mass,rtol=1e-6)

    def test_new_arcs_do_not_modify_continental_shared_vertices(self):
        s=Simulation(dict(width=96,height=48,mesh_level=2,duration_myr=4,plate_count=4))
        # Historical tiny-patch insertion isolation; the column-derived birth
        # law and its pending capacity are exercised by profile-one tests.
        s.native_arc_birth_profile_version=0
        old=s.material_surface['vertices'].copy()
        old_faces=s.material_surface['faces'].copy()
        cells=np.flatnonzero(s.crust == 0)[:2]
        original_mass=s.mass.sum()
        processes.add_arc_crust(s,cells,np.array([1000.,2000.]))
        np.testing.assert_array_equal(s.material_surface['vertices'][:len(old)],old)
        np.testing.assert_array_equal(s.material_surface['faces'][:len(old_faces)],old_faces)
        self.assertAlmostEqual(s.mass.sum()-original_mass,3000.,places=6)
        self.assertEqual(len(s.mass),len(s.material_surface['faces']))

    def test_arc_geometry_rotates_through_pole_with_mass_and_identity(self):
        s=ocean_fixture()
        processes.add_arc_crust(s,np.array([17]),np.array([2000.]))
        original=s.material_surface['vertices'].copy();patch=s.parcel_patch.copy()
        s.omega[0]=[.04,.01,0]
        for _ in range(40):
            material_surface.advect_surface(s.material_surface,s.omega,2.)
        s._sync_material()
        np.testing.assert_array_equal(s.parcel_patch,patch)
        np.testing.assert_allclose(s.material_surface['area_km2'],s.mass,rtol=4e-11)
        self.assertGreater(np.max(np.abs(s.material_surface['vertices']-original)),.2)


if __name__ == '__main__':
    unittest.main()
