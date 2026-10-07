"""Independent force, work and opening-integral oracles for a viscous slab neck."""
import unittest
import numpy as np
import slab_tether as tether


def neck(**updates):
    values=dict(viscosity_pa_s=1e23,thickness_m=80000.,length_m=100000.,failure_opening_m=100000.)
    values.update(updates)
    return tether.Neck(**values)


class SlabTetherTests(unittest.TestCase):
    def test_condensed_force_matches_independent_two_body_stokes_balance(self):
        basal=2e21;mantle=1e21;weight=1e13;external=-2e12
        for damage in (0.,.2,.8,1.):
            material=neck(damage=damage)
            connection=4e23*.8*(1.-damage)**2
            matrix=np.array([[basal+connection,-connection],[-connection,mantle+connection]])
            plate,slab=np.linalg.solve(matrix,np.array([external,weight]))
            terms=tether.condensed(material,weight,mantle)
            condensed_plate=(external+terms['drive_n_per_m'])/(basal+terms['drag_pa_s'])
            self.assertAlmostEqual(condensed_plate/plate,1.,places=12)
            report=tether.solve(material,weight,mantle,plate)
            self.assertAlmostEqual(report['slab_speed_m_s']/slab,1.,places=12)
            input_power=external*plate+weight*slab
            dissipation=basal*plate**2+mantle*slab**2+connection*(slab-plate)**2
            self.assertAlmostEqual(input_power/dissipation,1.,places=12)
            self.assertGreater(np.linalg.eigvalsh(matrix).min(),0.)

    def test_driver_and_drag_share_transmission_and_detachment_removes_both(self):
        full=tether.condensed(neck(),1e13,1e21)
        damaged=tether.condensed(neck(damage=.9),1e13,1e21)
        self.assertLess(damaged['drive_n_per_m'],full['drive_n_per_m'])
        self.assertAlmostEqual(damaged['drive_n_per_m']/full['drive_n_per_m'],
                               damaged['drag_pa_s']/full['drag_pa_s'])
        final=tether.solve(neck(damage=1.),1e13,1e21,2e-9)
        self.assertEqual(final['plate_traction_n_per_m'],0.)
        self.assertEqual(final['drag_pa_s'],0.)
        self.assertAlmostEqual(final['slab_speed_m_s'],1e-8,places=20)
        self.assertAlmostEqual(final['weight_power_w_per_m'],final['mantle_dissipation_w_per_m'])

    def test_parallel_channels_preserve_work_and_subdivision_without_averaging_damage(self):
        materials=[neck(damage=.1),neck(damage=.9)]
        weights=np.array([1e13,3e12]);drags=np.array([1e21,2e21]);widths=np.array([2e5,4e5])
        combined=tether.parallel(materials,weights,drags,widths)
        speed=1e-9
        expected=sum(width*tether.solve(material,weight,drag,speed)['plate_traction_n_per_m']
            for material,weight,drag,width in zip(materials,weights,drags,widths))
        self.assertAlmostEqual((combined['drive_n']-combined['drag_n_s_per_m']*speed)/expected,1.,places=13)
        split=tether.parallel([materials[0],materials[0],materials[1]],
            [weights[0],weights[0],weights[1]],[drags[0],drags[0],drags[1]],
            [widths[0]/3.,2.*widths[0]/3.,widths[1]])
        for key in combined:self.assertAlmostEqual(split[key]/combined[key],1.,places=13)
        # Conserving weight while averaging damage does not conserve response.
        averaged=tether.parallel([neck(damage=float(widths@[.1,.9]/widths.sum()))],
            [float(widths@weights/widths.sum())],[float(widths@drags/widths.sum())],[widths.sum()])
        self.assertGreater(abs(averaged['drive_n']-combined['drive_n'])/combined['drive_n'],.01)

    def test_exact_power_identity_and_nonnegative_dissipation_for_signed_loading(self):
        rng=np.random.default_rng(834)
        for _ in range(100):
            material=neck(damage=rng.uniform())
            report=tether.solve(material,rng.uniform(-1e13,1e13),10.**rng.uniform(20.,24.),rng.uniform(-1e-8,1e-8))
            scale=max(abs(report['weight_power_w_per_m']),abs(report['plate_power_w_per_m']),1.)
            self.assertLess(abs(report['power_residual_w_per_m'])/scale,2e-12)
            self.assertGreaterEqual(report['neck_dissipation_w_per_m'],0.)
            self.assertGreaterEqual(report['mantle_dissipation_w_per_m'],0.)

    def test_virtual_work_derivative_of_eliminated_functional_is_plate_resistance(self):
        material=neck(damage=.7)
        c=4e23*.8*.3**2;drag=1e21;weight=1e13
        def potential(v):
            u=(weight+c*v)/(drag+c)
            return .5*drag*u*u+.5*c*(u-v)**2-weight*u
        for v in (-1e-8,0.,2e-9,2e-8):
            epsilon=1e-12
            derivative=(potential(v+epsilon)-potential(v-epsilon))/(2.*epsilon)
            report=tether.solve(material,weight,drag,v)
            self.assertAlmostEqual(derivative/-report['plate_traction_n_per_m'],1.,places=8)

    def test_damage_matches_independent_differential_integration_and_opening(self):
        material=neck(damage=.2)
        weight=1e13;drag=1e21;v=2e-9;seconds=5.*365.25*86400.*1e6
        intact=4e23*.8
        def rhs(y):
            delta_v=(weight-drag*v)/(drag+intact*(1.-y[0])**2)
            return np.array([delta_v/100000.,delta_v])
        numerical=np.array([.2,0.]);dt=seconds/10000.
        for _ in range(10000):
            a=rhs(numerical);b=rhs(numerical+.5*dt*a)
            c=rhs(numerical+.5*dt*b);d=rhs(numerical+dt*c)
            numerical+=dt*(a+2.*b+2.*c+d)/6.
        result,report=tether.advance(material,weight,drag,v,5.)
        self.assertAlmostEqual(result.damage,numerical[0],places=11)
        self.assertAlmostEqual(report['opening_m']/numerical[1],1.,places=10)
        self.assertEqual(material.damage,.2)
        self.assertFalse(report['ruptured'])

    def test_failure_is_localized_exactly_and_leaves_remainder_for_coupled_resolve(self):
        material=neck(damage=.2)
        force=1e13;drag=1e21;inlet=2e-9
        expected=(100000./(force-drag*inlet))*(drag*.8+(4e23*.8)*.8**3/3.)/(365.25*86400.*1e6)
        result,report=tether.advance(material,force,drag,inlet,expected*2.)
        self.assertEqual(result.damage,1.)
        self.assertTrue(report['ruptured'])
        self.assertAlmostEqual(report['rupture_time_myr'],expected,places=12)
        self.assertAlmostEqual(report['remaining_dt_myr'],expected,places=12)
        self.assertEqual(report['final']['plate_traction_n_per_m'],0.)

    def test_time_partition_and_event_time_are_invariant_under_constant_external_state(self):
        initial=neck(damage=.1)
        whole,_=tether.advance(initial,1e13,1e21,1e-9,8.)
        split=initial
        for _ in range(160):split,_=tether.advance(split,1e13,1e21,1e-9,.05)
        self.assertAlmostEqual(split.damage,whole.damage,places=12)
        _,a=tether.advance(initial,1e13,1e21,1e-9,100.)
        _,b=tether.advance(split,1e13,1e21,1e-9,100.)
        self.assertAlmostEqual(a['rupture_time_myr'],8.+b['rupture_time_myr'],places=10)

    def test_small_time_and_stiff_neck_keep_relative_opening_resolution(self):
        material=neck()
        dt=1e-30
        expected=1e13/(1e21+4e23*.8)/100000.*(365.25*86400.*1e6)*dt
        result,report=tether.advance(material,1e13,1e21,0.,dt)
        self.assertAlmostEqual(result.damage/expected,1.,places=12)
        stiff=neck(viscosity_pa_s=1e40)
        report=tether.solve(stiff,1e13,1e21,0.)
        self.assertGreater(report['neck_extension_speed_m_s'],0.)
        self.assertAlmostEqual(report['neck_extension_speed_m_s']/(1e13/(1e21+4e40*.8)),1.,places=12)

    def test_compression_does_not_heal_and_zero_relative_opening_does_not_damage(self):
        initial=neck(damage=.4)
        for v in (1e-8,2e-8):
            result,report=tether.advance(initial,1e13,1e21,v,100000.)
            self.assertEqual(result,initial)
            self.assertEqual(report['opening_m'],0.)
            self.assertFalse(report['ruptured'])
        result,report=tether.advance(initial,-1e13,1e21,0.,100000.)
        self.assertEqual(result,initial)
        self.assertGreater(report['initial']['neck_dissipation_w_per_m'],0.)

    def test_force_rheology_and_plate_feed_control_failure_not_elapsed_collision_time(self):
        material=neck()
        def failure(model,weight=1e13,v=0.):
            return tether.advance(model,weight,1e21,v,1e6)[1]['rupture_time_myr']
        baseline=failure(material)
        self.assertGreater(failure(neck(viscosity_pa_s=1e24)),baseline)
        self.assertLess(failure(neck(viscosity_pa_s=1e22)),baseline)
        self.assertAlmostEqual(failure(material,weight=2e13),baseline/2.,places=12)
        self.assertGreater(failure(material,v=5e-9),baseline)

    def test_invalid_constitutive_state_is_rejected_without_defaults(self):
        for key in ('viscosity_pa_s','thickness_m','length_m','failure_opening_m'):
            for value in (0.,-1.,np.nan,True):
                with self.assertRaises(ValueError):neck(**{key:value})
        for value in (-.1,1.1,np.nan,True):
            with self.assertRaises(ValueError):neck(damage=value)
        for value in (-1.,np.nan,True):
            with self.assertRaises(ValueError):tether.advance(neck(),1e13,1e21,0.,value)
        with self.assertRaises(ValueError):tether.solve(neck(),1e13,0.,0.)


if __name__=='__main__':unittest.main()
