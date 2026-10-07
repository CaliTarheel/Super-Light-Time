import unittest
from types import SimpleNamespace
import numpy as np
import enhanced_rifting as er
import progressive_rifting
import viscous_sheet

class EnhancedRiftingTests(unittest.TestCase):
    def test_config_and_spatial_reproducibility(self):
        config=er.normalize(dict(enabled=True,seed=41))
        xyz=np.random.default_rng(3).normal(size=(80,3));xyz/=np.linalg.norm(xyz,axis=1)[:,None]
        whole=er.seeded_weakness(xyz,config)
        np.testing.assert_array_equal(whole,np.r_[er.seeded_weakness(xyz[:40],config),er.seeded_weakness(xyz[40:],config)])
        self.assertTrue(np.all((whole>=0)&(whole<=.25)))
        self.assertGreater(np.ptp(whole),.05)
        for bad in [dict(enabled=1),dict(seed=True),dict(amplitude=1),dict(correlation_km=0)]:
            with self.assertRaises(ValueError):er.normalize(bad)

    def test_weakness_changes_resistance_without_creating_damage(self):
        config=er.normalize(dict(enabled=True))
        s=SimpleNamespace(config={'enhanced_rifting':config},mass=np.ones(2),kind=np.ones(2,int),
            pos=np.array([[1.,0.,0.],[0.,1.,0.]]),suture=np.array([0.,1.]),parcel_patch=np.arange(2),
            structure={'thickness_km':np.full(2,35.),'reference_thickness_km':np.full(2,35.),
                'rift_heat_m':np.zeros(2),'lip_heat_m':np.array([0.,800.])})
        er.initialize(s); weights=er.material_viscosity(s)
        self.assertLess(weights[1],weights[0])
        np.testing.assert_array_equal(progressive_rifting.material_fields(s)['rift_damage'],np.zeros(2))
        np.testing.assert_array_equal(progressive_rifting.evolve_damage(np.zeros(2),np.zeros(2),weights,2),np.zeros(2))

    def test_realized_motion_replaces_unrealized_loading(self):
        s=SimpleNamespace(config={'enhanced_rifting':{'enabled':True}})
        mesh=dict(bases=np.array([1,2]),owner_uids=np.array([1,1]),edges=np.array([[0,1]]))
        s.rift_realized_motion=dict(**mesh,strain=np.array([-.01]),edge_extension=np.array([-2.]),velocity=np.zeros((2,3)),dt_myr=2.)
        response=dict(edge_extension=np.array([100.]),velocity=np.ones((2,3)))
        tensile,compression=er.realized_response(s,mesh,response,2.)
        np.testing.assert_array_equal(tensile,[0.]);np.testing.assert_array_equal(compression,[.01])
        np.testing.assert_array_equal(response['edge_extension'],[-2.])
        with self.assertRaises(ValueError):er.realized_response(s,mesh,response,1.)

    def test_heterogeneous_operator_is_positive(self):
        xyz=np.array([[1.,0.,0.],[1.,.02,0.],[1.,0.,.02],[1.,.02,.02]])
        xyz/=np.linalg.norm(xyz,axis=1)[:,None]
        faces=np.array([[0,1,2],[1,3,2]])
        from material_surface import spherical_face_areas
        areas=spherical_face_areas(xyz,faces)
        one=viscous_sheet.prepare(xyz,faces,areas,100.)
        weak=viscous_sheet.prepare(xyz,faces,areas,100.,viscosity_weights=np.array([.2,1.]))
        self.assertEqual(weak['viscosity_weights'][0],.2)
        self.assertTrue(np.all(one['viscosity_weights']==1))

    def test_weak_belt_localizes_actual_extension_and_unloaded_stays_still(self):
        from test_viscous_sheet import patch
        xyz,faces,areas,xy=patch(n=6,extent=300.)
        mask=np.zeros(len(xyz),bool)
        driven=np.isclose(np.abs(xy[:,0]),300.)
        target=np.column_stack((xy[:,0]/100.,np.zeros((len(xy),2))))
        normals=np.tile([1.,0.,0.],(len(xyz),1))
        centre=np.mean(xy[faces,0],axis=1)
        weights=np.where(np.abs(centre)<100.,.2,1.)
        velocity,diagnostic=viscous_sheet.solve(xyz,faces,areas,target,mask,driven,3000.,
            viscosity_weights=weights,driven_normals=normals,tolerance=1e-9,iterations=1600)
        self.assertTrue(diagnostic['converged'],diagnostic)
        context=viscous_sheet.prepare(xyz,faces,areas,3000.,viscosity_weights=weights)
        deformation=viscous_sheet.strain_rate(context,velocity)['D']
        # The tangent x axis is global x up to the patch's spherical tilt.
        exx=deformation[:,0,0]
        weak=np.abs(centre)<100.;strong=np.abs(centre)>180.
        self.assertGreater(float(np.mean(exx[weak])),1.5*float(np.mean(exx[strong])))
        zero,info=viscous_sheet.solve(xyz,faces,areas,np.zeros_like(xyz),mask,mask,3000.,
            viscosity_weights=weights,tolerance=1e-9,iterations=1600)
        self.assertTrue(info['converged']);np.testing.assert_array_equal(zero,np.zeros_like(xyz))

    def test_weak_collision_columns_spread_faster_and_conserve_volume(self):
        from test_gravitational_relaxation import fixture
        from material_surface import spherical_face_areas
        from gravitational_relaxation import relax
        points,faces,thickness,reference,sheets=fixture()
        area=spherical_face_areas(points,faces);volume=area*thickness
        runs=[]
        for weight in (1.,.25):
            result,info=relax(points,faces,volume,reference,sheets,.5,
                rigid_mask=np.zeros(len(points),bool),minimum_area_km2=area*.5,
                maximum_area_km2=area*2.,viscosity_weights=np.full(len(faces),weight),
                constraint_version=1,tolerance=1e-9)
            self.assertLess(info['energy_after_km4'],info['energy_before_km4'])
            final_area=spherical_face_areas(result,faces)
            np.testing.assert_allclose(final_area*(volume/final_area),volume,rtol=2e-14)
            runs.append((np.linalg.norm(result-points),info['energy_after_km4']))
        self.assertGreater(runs[1][0],runs[0][0])
        self.assertLess(runs[1][1],runs[0][1])

if __name__=='__main__':unittest.main()
