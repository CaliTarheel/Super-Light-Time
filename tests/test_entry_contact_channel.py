"""Independent local equilibrium and work checks for candidate channel contact."""

import unittest

import numpy as np

import entry_contact_channel as channel


class EntryContactChannelTests(unittest.TestCase):
    def test_conserved_mass_sets_signed_buoyancy_without_new_inventory(self):
        area=np.array([10.,10.])
        volume=area*35.
        mass=volume*1e9*np.array([2800.,3500.])
        load=channel.buoyancy_surface_load(volume,mass,area)
        np.testing.assert_allclose(load,9.81*35000.*np.array([500.,-200.]),rtol=1e-14)

    def test_unilateral_equilibrium_and_work_on_both_branches(self):
        preferred=np.array([10.,50.,50.,20.])*1000.
        base=np.full(4,35e3)
        buoyancy=np.array([1e7,1e7,-1e7,-1e7])
        stiffness=np.full(4,1000.)
        result=channel.equilibrate(preferred,base,buoyancy,stiffness)
        np.testing.assert_array_equal(result['contact'],[True,False,False,True])
        self.assertTrue(np.all(result['depth_m']>=base))
        self.assertTrue(np.all(result['reaction_pa']>=0.))
        np.testing.assert_array_equal(result['reaction_pa'][~result['contact']],0.)
        np.testing.assert_allclose(result['reaction_pa']*result['mantle_gap_m'],0.,atol=1e-5)
        np.testing.assert_allclose(stiffness*(result['depth_m']-preferred)+buoyancy,
                                   result['reaction_pa'],rtol=1e-14,atol=1e-5)
        for offset in (100.,1000.,10000.):
            trial=result['depth_m']+offset
            energy=.5*stiffness*(trial-preferred)**2+buoyancy*trial
            self.assertTrue(np.all(energy>=result['energy_j_m2']))
        step=.01
        for name,argument,derivative in (
            ('preferred_depth_m',preferred,'preferred_depth_derivative_n_m2'),
            ('upper_base_depth_m',base,'upper_base_derivative_n_m2'),
            ('buoyancy_n_m2',buoyancy,'buoyancy_derivative_m')):
            plus=argument+step;minus=argument-step
            values=dict(preferred_depth_m=preferred,upper_base_depth_m=base,
                        buoyancy_n_m2=buoyancy,bending_stiffness_n_m3=stiffness)
            values[name]=plus
            high=channel.equilibrate(**values)['energy_j_m2']
            values[name]=minus
            low=channel.equilibrate(**values)['energy_j_m2']
            np.testing.assert_allclose((high-low)/(2*step),result[derivative],rtol=2e-7,atol=1e-4)

    def test_contact_to_channel_energy_and_force_are_continuous(self):
        base=35e3;buoyancy=1e7;stiffness=1000.
        switch=base+buoyancy/stiffness
        at=channel.equilibrate(switch,base,buoyancy,stiffness)
        left=channel.equilibrate(switch-.001,base,buoyancy,stiffness)
        right=channel.equilibrate(switch+.001,base,buoyancy,stiffness)
        tangent=float(at['preferred_depth_derivative_n_m2'])*.001
        self.assertAlmostEqual(float(at['energy_j_m2']),float(left['energy_j_m2'])+tangent,delta=1e-3)
        self.assertAlmostEqual(float(at['energy_j_m2']),float(right['energy_j_m2'])-tangent,delta=1e-3)
        self.assertAlmostEqual(float(left['reaction_pa']),0.,delta=1.)
        self.assertAlmostEqual(float(right['reaction_pa']),0.,delta=1.)
        self.assertAlmostEqual(float(left['preferred_depth_derivative_n_m2']),
                               float(right['preferred_depth_derivative_n_m2']),delta=1.)

    def test_free_channel_has_the_existing_buoyancy_work_slope(self):
        buoyancy=9.81*500.*35000.
        stiffness=1000.
        first=channel.equilibrate(300e3,35e3,buoyancy,stiffness)
        second=channel.equilibrate(330e3,35e3,buoyancy,stiffness)
        self.assertFalse(bool(first['contact']))
        self.assertFalse(bool(second['contact']))
        self.assertAlmostEqual(float(first['preferred_depth_derivative_n_m2']),buoyancy,
                               delta=buoyancy*1e-14)
        self.assertAlmostEqual(float(second['energy_j_m2']-first['energy_j_m2']),
                               buoyancy*30e3,delta=buoyancy*30e3*1e-14)

    def test_upper_uplift_shares_contact_work_and_rigid_limit(self):
        preferred=10e3;base=35e3;buoyancy=1e7
        lower_stiffness=1000.;upper_stiffness=5000.
        result=channel.equilibrate(preferred,base,buoyancy,lower_stiffness,
            upper_restoring_stiffness_n_m3=upper_stiffness)
        z=float(result['depth_m']);w=float(result['upper_uplift_m'])
        reaction=float(result['reaction_pa'])
        self.assertTrue(bool(result['contact']))
        self.assertGreater(w,0.)
        self.assertGreaterEqual(z+w,base-1e-8)
        self.assertAlmostEqual(z+w,base,delta=1e-8)
        self.assertAlmostEqual(lower_stiffness*(z-preferred)+buoyancy,
                               reaction,delta=1e-6)
        self.assertAlmostEqual(upper_stiffness*w,reaction,delta=1e-6)
        for dz,dw in ((100.,0.),(0.,100.),(100.,-100.),(-100.,100.)):
            trial_z,trial_w=z+dz,w+dw
            self.assertGreaterEqual(trial_z+trial_w,base-1e-8)
            trial_energy=(.5*lower_stiffness*(trial_z-preferred)**2
                          +buoyancy*trial_z+.5*upper_stiffness*trial_w**2)
            self.assertGreater(trial_energy,float(result['energy_j_m2']))
        for name,value,derivative in (
            ('preferred_depth_m',preferred,'preferred_depth_derivative_n_m2'),
            ('upper_base_depth_m',base,'upper_base_derivative_n_m2'),
            ('buoyancy_n_m2',buoyancy,'buoyancy_derivative_m')):
            step=100. if name=='buoyancy_n_m2' else .01
            inputs=dict(preferred_depth_m=preferred,upper_base_depth_m=base,
                buoyancy_n_m2=buoyancy,bending_stiffness_n_m3=lower_stiffness,
                upper_restoring_stiffness_n_m3=upper_stiffness)
            inputs[name]=value+step
            high=float(channel.equilibrate(**inputs)['energy_j_m2'])
            inputs[name]=value-step
            low=float(channel.equilibrate(**inputs)['energy_j_m2'])
            self.assertAlmostEqual((high-low)/(2*step),float(result[derivative]),
                                   delta=max(abs(float(result[derivative]))*2e-7,1e-4))
        rigid=channel.equilibrate(preferred,base,buoyancy,lower_stiffness)
        stiff=channel.equilibrate(preferred,base,buoyancy,lower_stiffness,
            upper_restoring_stiffness_n_m3=1e15)
        np.testing.assert_allclose(stiff['depth_m'],rigid['depth_m'],rtol=2e-12)
        np.testing.assert_allclose(stiff['energy_j_m2'],rigid['energy_j_m2'],rtol=2e-12)

    def test_conserved_crust_and_mantle_restoring_scale_keep_contact_finite(self):
        area=10.;volume=area*35.;mass=volume*1e9*2800.
        buoyancy=float(channel.buoyancy_surface_load(volume,mass,area))
        response=channel.equilibrate(10e3,35e3,buoyancy,1000.,
            upper_restoring_stiffness_n_m3=3300.*9.81)
        self.assertTrue(bool(response['contact']))
        self.assertGreater(float(response['upper_uplift_m']),5e3)
        self.assertLess(float(response['upper_uplift_m']),7e3)
        self.assertGreater(float(response['depth_m']),28e3)
        self.assertLess(float(response['depth_m']),30e3)

    def test_invalid_inputs_fail_before_a_contact_state_is_returned(self):
        for values in ((-1.,1.,1.,1.),(1.,-1.,1.,1.),(1.,1.,1.,0.),
                       (1.,1.,np.nan,1.)):
            with self.assertRaises(ValueError):channel.equilibrate(*values)
        with self.assertRaises(ValueError):channel.buoyancy_surface_load(1.,1.,0.)


if __name__=='__main__':unittest.main()
