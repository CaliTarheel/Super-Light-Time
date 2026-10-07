"""Conserved material force under horizontal contact-area projection."""

import unittest

import numpy as np

import entry_contact_channel
import entry_projected_contact
import entry_projected_loads
import finite_entry_arc
from tests.test_channel_entry_zones import paired
from tests.test_entry_projected_contact import LOWER,FULL,HINGE


def fixture(refinement):
    surface,_,_,_,_,spec=paired()
    lower=surface['vertices'][surface['faces'][0]]
    upper=surface['vertices'][surface['faces'][1]]
    return entry_projected_contact.projected_pair_areas(
        lower,upper,spec['hinge_normals'][0],50.,refinement=refinement,
        include_cells=True)


class ProjectedLoadTests(unittest.TestCase):
    def test_full_projected_contact_recovers_old_entry_energy_exactly(self):
        for level,mode in ((0,'spherical_point'),(2,'spherical_point'),
                           (4,'spherical_point'),(2,'affine_nodal')):
            geometry=entry_projected_contact.projected_pair_areas(
                LOWER,FULL,HINGE,50.,refinement=level,include_cells=True,
                coordinate_discretization=mode)
            face=geometry['face_reference_area_km2']
            volume=face*35.
            mass=volume*2800e9
            result=entry_projected_loads.uniform_contact_entry_work(
                LOWER,HINGE,50.,volume,mass,geometry)
            q=6371e3*np.arcsin(LOWER@HINGE)
            reference_mean=finite_entry_arc.integrate(q,np.ones(3),np.ones(3))[0]
            pressure=float(entry_contact_channel.buoyancy_surface_load(volume,mass,face))
            expected=pressure*face*1e6*reference_mean*np.sin(np.deg2rad(50.))
            np.testing.assert_allclose(result['contact_entry_work_j'],expected,rtol=2e-10)
            self.assertGreater(result['contact_entry_work_j'],0.)

    def test_partial_pair_replaces_only_its_own_signed_entry_work(self):
        surface,_,_,_,_,spec=paired()
        lower=surface['vertices'][surface['faces'][0]]
        upper=surface['vertices'][surface['faces'][1]]
        rows=[]
        for level in (3,4):
            geometry=entry_projected_contact.projected_pair_areas(
                lower,upper,spec['hinge_normals'][0],50.,refinement=level,
                include_cells=True)
            face=geometry['face_reference_area_km2']
            volume=face*35.
            result=entry_projected_loads.uniform_contact_entry_work(
                lower,spec['hinge_normals'][0],50.,volume,volume*2800e9,geometry)
            rows.append(result)
            full_mean=finite_entry_arc.integrate(
                6371e3*np.arcsin(lower@spec['hinge_normals'][0]),
                np.ones(3),np.ones(3))[0]
            pressure=float(entry_contact_channel.buoyancy_surface_load(
                volume,volume*2800e9,face))
            full=pressure*face*1e6*full_mean*np.sin(np.deg2rad(50.))
            self.assertGreater(result['contact_entry_work_j'],0.)
            self.assertLess(result['contact_entry_work_j'],full)
            self.assertTrue(all(cell['mean_entry_depth_m']>=0. for cell in result['cells']))
        self.assertLess(abs(rows[1]['contact_entry_work_j']-rows[0]['contact_entry_work_j'])/
                        rows[1]['contact_entry_work_j'],1e-6)

    def test_finite_front_and_dense_mass_preserve_signed_local_work(self):
        midpoint=np.array([1.,0.,0.])
        geometry=entry_projected_contact.projected_pair_areas(
            LOWER,FULL,HINGE,50.,finite_midpoint=midpoint,
            finite_half_length_km=50.,refinement=3,include_cells=True)
        face=geometry['face_reference_area_km2']
        volume=face*35.
        light=entry_projected_loads.uniform_contact_entry_work(
            LOWER,HINGE,50.,volume,volume*2800e9,geometry)
        dense=entry_projected_loads.uniform_contact_entry_work(
            LOWER,HINGE,50.,volume,volume*3450e9,geometry)
        left,right,_,_=finite_entry_arc.endpoint_planes(HINGE,midpoint,50.,6371.)
        mean=finite_entry_arc.integrate(6371e3*np.arcsin(LOWER@HINGE),
                                        LOWER@left,LOWER@right)[0]
        for mass,result in ((volume*2800e9,light),(volume*3450e9,dense)):
            pressure=float(entry_contact_channel.buoyancy_surface_load(volume,mass,face))
            expected=pressure*face*1e6*mean*np.sin(np.deg2rad(50.))
            np.testing.assert_allclose(result['contact_entry_work_j'],expected,rtol=2e-10)
        self.assertLess(dense['contact_entry_work_j'],0.)

    def test_cells_conserve_buoyancy_while_pressure_follows_physical_area(self):
        geometry=fixture(4)
        face=geometry['face_reference_area_km2']
        volume=face*35.
        mass=volume*2800e9
        result=entry_projected_loads.uniform_cell_buoyancy(
            volume,mass,face,geometry['contact_cells'])
        reference=float(entry_contact_channel.buoyancy_surface_load(volume,mass,face))
        material=geometry['material_reference_contact_area_km2']
        physical=geometry['physical_contact_area_km2']
        np.testing.assert_allclose(result['contact_buoyancy_force_n'],
                                   reference*material*1e6,rtol=2e-13)
        mean=result['contact_buoyancy_force_n']/(physical*1e6)
        self.assertAlmostEqual(mean/reference,material/physical,places=12)
        self.assertGreater(mean/reference,1.5)
        self.assertAlmostEqual(sum(cell['material_mass_kg'] for cell in result['cells'])/
                               (mass*material/face),1.,places=12)
        self.assertAlmostEqual(sum(cell['material_volume_km3'] for cell in result['cells'])/
                               (volume*material/face),1.,places=12)
        for cell in result['cells']:
            self.assertAlmostEqual(cell['mean_buoyancy_pa']*cell['physical_area_km2']*1e6/
                                   cell['buoyancy_force_n'],1.,places=12)

    def test_signed_dense_crust_and_refinement_keep_extensive_force(self):
        reports=[]
        for level in (3,4):
            geometry=fixture(level)
            face=geometry['face_reference_area_km2']
            volume=face*35.
            result=entry_projected_loads.uniform_cell_buoyancy(
                volume,volume*3450e9,face,geometry['contact_cells'])
            self.assertLess(result['contact_buoyancy_force_n'],0.)
            self.assertTrue(all(cell['mean_buoyancy_pa']<0. for cell in result['cells']))
            reports.append(result)
        self.assertLess(abs(reports[1]['contact_buoyancy_force_n']-
                            reports[0]['contact_buoyancy_force_n'])/
                        abs(reports[1]['contact_buoyancy_force_n']),1e-6)

    def test_rejects_invented_material_and_missing_physical_area(self):
        with self.assertRaisesRegex(ValueError,'more material'):
            entry_projected_loads.uniform_cell_buoyancy(35.,35.*2800e9,1.,[
                dict(material_reference_area_km2=2.,physical_area_km2=1.)])
        with self.assertRaisesRegex(ValueError,'positive finite areas'):
            entry_projected_loads.uniform_cell_buoyancy(35.,35.*2800e9,1.,[
                dict(material_reference_area_km2=.5,physical_area_km2=0.)])
        with self.assertRaisesRegex(ValueError,'positive uniform material inventory'):
            entry_projected_loads.uniform_cell_buoyancy(0.,1.,1.,[])


if __name__=='__main__':
    unittest.main()
