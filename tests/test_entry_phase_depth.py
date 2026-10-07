"""Independent depth moments, local source integration, and native continuation."""
from copy import deepcopy
import math
import pickle
import unittest
from unittest.mock import patch
import numpy as np
import entry_depth as depth
import entry_phase_depth as coupling
import dense_crust as phase
import crust_inventory as inventory
import phase_evolution as evolution
import crustal_structure as columns
import mesh_coverage
import finite_entry_arc
from tests.test_entry_regions import world as entry_world


def moment(corners,power):
    # Dirichlet(1,1,1) moments of the uniform reference triangle. This oracle
    # never uses the production one-dimensional coordinate density.
    total=0.
    for i in range(power+1):
        for j in range(power-i+1):
            k=power-i-j
            total+=2.*math.factorial(power)/math.factorial(power+2)*corners[0]**i*corners[1]**j*corners[2]**k
    return total


def world():
    s=entry_world()
    evolution.upgrade(s,1000.,constitutive_parameters=dict(surface_temperature_c=1000.,
        mantle_temperature_c=1000.,geotherm_c_per_km=0.,supporting_yield_stress_pa=1e9))
    coupling.upgrade(s)
    return s


class EntryPhaseDepthTests(unittest.TestCase):
    def test_horizontal_surface_source_depth_is_read_only_and_policy_gated(self):
        s=world()
        before=pickle.dumps(s)
        legacy=coupling.prepare(s)
        horizontal=coupling.prepare(s,coordinate_mode='horizontal_surface')
        self.assertEqual(pickle.dumps(s),before)
        self.assertEqual(legacy['coordinate_mode'],'along_slab_conveyor')
        self.assertEqual(horizontal['coordinate_mode'],'horizontal_surface')
        self.assertEqual(set(legacy['corners']),set(horizontal['corners']))
        self.assertTrue(legacy['corners'])
        spec=legacy['specification']
        for index,face in enumerate(legacy['corners']):
            old=legacy['corners'][face]
            new=horizontal['corners'][face]
            scale=1./math.cos(math.radians(spec['dip_degrees'][index]))
            np.testing.assert_allclose(new,old*scale,rtol=2e-14,atol=1e-10)
            self.assertGreaterEqual(float(new.max()),float(old.max()))
            evaluate=lambda z:np.column_stack((z,z*z))
            x0,w0,_=depth.quadrature(old,evaluate)
            x1,w1,_=depth.quadrature(new,evaluate)
            np.testing.assert_allclose(w1@evaluate(x1),(w0@evaluate(x0))*[scale,scale*scale],
                                       rtol=2e-12,atol=1e-9)
        with self.assertRaisesRegex(ValueError,'coordinate mode is unsupported'):
            coupling.prepare(s,coordinate_mode='unknown')
        prepared=evolution.prepare(s)
        prepared['entry_depth']=horizontal
        with self.assertRaisesRegex(ValueError,'cannot enter the native source policy'):
            coupling.expand(s,s.structure,.25,[],np.array([],int),np.array([],float),prepared,
                            evolution.parameters(s.retained_phase_parameters))
        self.assertEqual(pickle.dumps(s),before)

    def test_disjoint_stack_and_entry_split_local_phase_sources(self):
        s=world();prepared=evolution.prepare(s)
        face=next(iter(prepared['entry_depth']['corners']))
        prepared['entry_depth']['corners'][face]=np.array([-20.,-20.,100.])
        triangle=prepared['entry_depth']['triangles'][face]
        def point(bary):
            value=np.asarray(bary)@triangle
            return value/np.linalg.norm(value)
        near01=point([.9,.1,0.]);near02=point([.9,0.,.1])
        stack=np.array([triangle[0],near01,near02])
        free=np.array([near01,triangle[1],triangle[2],near02])
        area=float(s.material_surface['area_km2'][face])
        stack_weight=mesh_coverage._polygon_area(stack,6371.)/area
        free_weight=mesh_coverage._polygon_area(free,6371.)/area
        self.assertAlmostEqual(stack_weight+free_weight,1.,places=11)
        rows=[dict(face=face,polygon=stack,weight=stack_weight,
                   upper=np.array([face+1]),pairs=np.array([],int)),
              dict(face=face,polygon=free,weight=free_weight,
                   upper=np.array([],int),pairs=np.array([],int))]
        selected=np.array([face,face]);weights=np.array([stack_weight,free_weight])
        controls=evolution.parameters(s.retained_phase_parameters)
        expanded,index,share,report=coupling.expand(s,s.structure,.25,rows,selected,weights,
                                                    prepared,controls)
        self.assertTrue(np.all(index==face))
        self.assertAlmostEqual(float(share.sum()),1.,places=10)
        self.assertGreater(report['entry_columns'],0)
        self.assertFalse(any('entry_depth_km' in row for row in expanded if len(row['upper'])))
        observed=sum(w*row.get('entry_depth_km',0.)**2 for row,w in zip(expanded,share))
        # A 10%-edge stack near corner 0 is entirely in q<0. The positive
        # depth distribution should therefore retain the whole-face moment.
        self.assertAlmostEqual(observed/(100.**4/86400.),1.,places=5)
        prepared['regions']=[r for r in prepared['regions'] if r['face']!=face]+rows
        evolved,phase_report,_=evolution.evolve(s,deepcopy(s.structure),.25,prepared=prepared)
        phase.validate(evolved)
        before=phase.mass_volume(s.structure)[face]
        after=(phase.mass_volume(evolved)[face]+evolved[inventory.RETURNED][face]
               -s.structure[inventory.RETURNED][face])
        self.assertAlmostEqual(after/before,1.,places=12)
        self.assertGreater(phase_report['diagnostics']['entry_depth']['local_source_evaluations'],0)
        heat=(evolved[phase.HEAT][face]+evolved[phase.HEAT_RETURNED][face]
              -evolved[phase.HEAT_BATH][face])
        self.assertAlmostEqual(heat/s.structure[phase.HEAT][face],1.,places=12)
        active=np.array([point([.1,0.,.9]),point([0.,.1,.9]),triangle[2]])
        rows[0]=dict(rows[0],polygon=active)
        with self.assertRaisesRegex(ValueError,'active continental stack'):
            coupling.expand(s,s.structure,.25,rows,selected,weights,prepared,controls)

    def test_finite_entry_pieces_retain_area_outside_disjoint_stack(self):
        from tests.test_finite_entry_arc import geometry as finite_geometry,unit as finite_unit
        points,_,normal,midpoint=finite_geometry()
        inactive=finite_unit(np.array([[.95,.05,0.],[.8,.1,.1],[.95,0.,.05]])@points)
        left,right,_,_=finite_entry_arc.endpoint_planes(normal[0],midpoint[0],50.,6371.)
        q=6371e3*np.arcsin(points@normal[0])
        _,_,_,_,_,parts=finite_entry_arc.integrate(q,points@left,points@right)
        depth_state=dict(triangles={0:points},corners={0:q*np.sin(np.deg2rad(50.))/1000.},
                         finite_pieces={0:dict(barycentric_triangles=[bary for _,bary in parts])})
        total=sum(piece_area for piece_area,_ in coupling._entered_region_triangles(
            depth_state,0,points,6371.))
        stack=sum(piece_area for piece_area,_ in coupling._entered_region_triangles(
            depth_state,0,inactive,6371.))
        whole=sum(mesh_coverage._polygon_area(
            finite_unit(bary@points),6371.) for _,bary in parts)
        self.assertLessEqual(stack,1e-8)
        self.assertAlmostEqual(total/whole,1.,places=10)

    def test_finite_endpoint_and_disjoint_stack_share_phase_face(self):
        s=world();row=s.continental_entry_regions['regions'][0]
        face=int(np.flatnonzero(s.parcel_entry_region>0)[0])
        triangle=s.material_surface['vertices'][s.material_surface['faces'][face]]
        normal=row['hinge_normal']
        midpoint=triangle[2]-normal*(triangle[2]@normal)
        row['finite_midpoint']=midpoint/np.linalg.norm(midpoint)
        row['finite_half_length_km']=5.
        prepared=evolution.prepare(s)
        def point(bary):
            value=np.asarray(bary)@triangle
            return value/np.linalg.norm(value)
        a,b=point([.9,.1,0.]),point([.9,0.,.1])
        stack=np.array([triangle[0],a,b])
        free=np.array([a,triangle[1],triangle[2],b])
        area=float(s.material_surface['area_km2'][face])
        stack_weight=mesh_coverage._polygon_area(stack,6371.)/area
        free_weight=mesh_coverage._polygon_area(free,6371.)/area
        rows=[dict(face=face,polygon=stack,weight=stack_weight,
                   upper=np.array([face+1]),pairs=np.array([],int)),
              dict(face=face,polygon=free,weight=free_weight,
                   upper=np.array([],int),pairs=np.array([],int))]
        selected=np.array([face,face]);weights=np.array([stack_weight,free_weight])
        controls=evolution.parameters(s.retained_phase_parameters)
        expanded,index,share,_=coupling.expand(s,s.structure,.25,rows,selected,weights,
                                               prepared,controls)
        self.assertTrue(np.all(index==face))
        self.assertAlmostEqual(float(share.sum()),1.,places=10)
        self.assertFalse(any('entry_depth_km' in item for item in expanded if len(item['upper'])))
        self.assertGreater(sum(w*item.get('entry_depth_km',0.) for item,w in zip(expanded,share)),0.)
        prepared['regions']=[item for item in prepared['regions'] if item['face']!=face]+rows
        evolved,_,_=evolution.evolve(s,deepcopy(s.structure),.25,prepared=prepared)
        before=phase.mass_volume(s.structure)[face]
        after=(phase.mass_volume(evolved)[face]+evolved[inventory.RETURNED][face]
               -s.structure[inventory.RETURNED][face])
        self.assertAlmostEqual(after/before,1.,places=12)

    def test_reference_moments_equal_corner_limits_and_negative_atom(self):
        for corners in ([2.,5.,11.],[2.,2.,11.],[2.,11.,11.],[5.,5.,5.]):
            x,w,r=depth.quadrature(corners,lambda z:np.column_stack([z**p for p in range(5)]))
            self.assertTrue(np.all(w>0.));self.assertAlmostEqual(w.sum(),1.,places=14)
            for p in range(5):self.assertAlmostEqual(float(w@x**p)/moment(corners,p),1.,places=12)
        # q=(-1,-1,1): density is (1-q)/2; half the edge distance is
        # negative and 3/4 of the triangle remains exactly at depth zero.
        x,w,_=depth.quadrature([-1.,-1.,1.],lambda z:np.column_stack([z,z*z,z**3]))
        self.assertEqual(depth.distribution([-1.,-1.,1.])[0],[(0.,.75)])
        for p in (1,2,3):self.assertAlmostEqual(float(w@x**p),1./(2.*(p+1)*(p+2)),places=14)
        x,w,_=depth.quadrature([-8.,-2.,-1.],lambda z:z[:,None])
        np.testing.assert_array_equal(x,[0.]);np.testing.assert_array_equal(w,[1.])

    def test_nonlinear_threshold_is_integrated_before_averaging_and_refines(self):
        # q=(0,0,100) has density (100-q)/5000. A narrow physical threshold
        # is supplied explicitly; a mean-depth source would be exactly zero.
        cutoff=70.
        function=lambda z:np.maximum(z-cutoff,0.)**2
        expected=(100.-cutoff)**4/60000.
        x,w,_=depth.quadrature([0.,0.,100.],lambda z:function(z)[:,None],breakpoints=[cutoff])
        self.assertAlmostEqual(float(w@function(x))/expected,1.,places=13)
        self.assertEqual(function(np.array([100./3.]))[0],0.)
        # Exponential integral: integration by parts gives this independent
        # primitive for the linear triangular depth density.
        def exact(k):return 2.*(np.expm1(k)-k)/(k*k)
        previous=None
        for tolerance in (1e-3,1e-6,1e-9):
            x,w,_=depth.quadrature([0.,0.,1.],lambda z:np.exp(10.*z)[:,None],relative_tolerance=tolerance)
            error=abs(float(w@np.exp(10.*x))-exact(10.))
            if previous is not None:self.assertLessEqual(error,previous+1e-11)
            previous=error
        self.assertLess(previous,1e-9)
        with self.assertRaisesRegex(ValueError,'panel budget'):
            depth.quadrature([0.,0.,1.],lambda z:np.exp(100.*z)[:,None],relative_tolerance=1e-12,max_panels=1)

    def test_marker_uses_reference_coordinates_then_positive_depth(self):
        triangle=np.array([[1.,0.,0.],[1.,.1,0.],[1.,0.,.1]])
        triangle/=np.linalg.norm(triangle,axis=1,keepdims=True)
        corners=np.array([-20.,10.,60.])
        for barycentric in ([.8,.1,.1],[.1,.2,.7]):
            point=np.asarray(barycentric)@triangle;point/=np.linalg.norm(point)
            self.assertAlmostEqual(depth.point_depth(corners,triangle,point),max(float(np.asarray(barycentric)@corners),0.),places=12)

    def test_reference_subdivision_preserves_nonlinear_local_source_mean(self):
        q=np.array([-20.,10.,80.]);mid=(q+np.roll(q,-1))/2.
        children=([q[0],mid[0],mid[2]],[mid[0],q[1],mid[1]],
                  [mid[2],mid[1],q[2]],[mid[0],mid[1],mid[2]])
        function=lambda z:np.column_stack([np.exp(-z/8.),np.maximum(z-30.,0.)**2])
        def integrate(corners):
            x,w,_=depth.quadrature(corners,function,breakpoints=[30.],relative_tolerance=1e-9)
            return w@function(x)
        np.testing.assert_allclose(sum(map(integrate,children))/4.,integrate(q),rtol=2e-10,atol=1e-12)

    def test_endpoint_layer_cannot_hide_between_interior_quadrature_nodes(self):
        cap=.002
        x,w,_=depth.quadrature([0.,0.,1.],lambda z:np.minimum(z,cap)[:,None],relative_tolerance=1e-10,absolute_tolerance=1e-14)
        expected=cap-cap**2+cap**3/3.
        self.assertAlmostEqual(float(w@np.minimum(x,cap))/expected,1.,places=9)

    def test_local_phase_conversion_matches_independent_depth_integral_and_mass_heat(self):
        s=world();prepared=evolution.prepare(s);face=next(iter(prepared['entry_depth']['corners']))
        # Controlled signed affine source geometry isolates source integration
        # from the separately tested spherical hinge mechanics.
        prepared['entry_depth']['corners'][face]=np.array([-20.,-20.,100.])
        start=deepcopy(s.structure);dt=2.
        actual,report,_=evolution.evolve(s,start,dt,prepared=prepared)
        controls=evolution.parameters(s.retained_phase_parameters)
        def direct(z):
            z=np.atleast_1d(z)
            local={k:np.full(len(z),v[face]) for k,v in start.items()}
            returned=np.zeros(len(z));contraction=np.zeros(len(z))
            for _ in range(8):
                forcing=evolution.conditions(local,z,z*3300.*1000.,controls)
                result=phase.advance(local,forcing['eligible_fraction'],1000.,forcing['thermal_tau_myr'],.25,
                    reaction_tau_myr=controls['reaction_tau_myr'],drainage_rate_myr=forcing['drainage_rate_myr'],
                    minimum_thickness_km=columns.MIN_THICKNESS_KM)
                returned+=result['returned_km'];contraction+=result['contraction_km']
            return local,returned,contraction
        # Independently integrate the analytic density and its negative atom.
        zero_weight=1.-(100./120.)**2
        estimates=[]
        for intervals in (16384,32768):
            z=np.linspace(0.,100.,intervals+1)
            weights=np.ones(intervals+1);weights[1:-1:2]=4.;weights[2:-1:2]=2.
            weights*=100./intervals/3.*2.*(100.-z)/120.**2
            values=direct(z)[0];zero=direct(0.)[0]
            estimates.append({key:zero_weight*zero[key][0]+weights@values[key] for key in values})
        for key in (phase.DENSE,phase.CONVERTED,phase.HEAT,inventory.REMAINING):
            expected=estimates[-1][key]
            np.testing.assert_allclose(estimates[0][key],expected,rtol=1e-7,atol=1e-10)
            np.testing.assert_allclose(actual[key][face],expected,rtol=3e-6,atol=1e-9)
        initial_mass=phase.mass_volume(start)[face]
        final_mass=phase.mass_volume(actual)[face]+actual[inventory.RETURNED][face]-start[inventory.RETURNED][face]
        self.assertAlmostEqual(final_mass/initial_mass,1.,places=12)
        self.assertAlmostEqual((actual[phase.HEAT][face]+actual[phase.HEAT_RETURNED][face]
            -actual[phase.HEAT_BATH][face])/start[phase.HEAT][face],1.,places=12)
        self.assertGreater(actual[phase.DENSE][face],0.)
        self.assertLessEqual(report['diagnostics']['entry_depth']['maximum_estimated_normalized_error'],1.)

    def test_source_failure_and_stale_geometry_leave_original_columns_unchanged(self):
        s=world();prepared=evolution.prepare(s);before=deepcopy(s.structure)
        prepared['entry_depth']['epoch_myr']-=1.
        with self.assertRaisesRegex(ValueError,'stale'):evolution.evolve(s,s.structure,1.,prepared=prepared)
        for key in before:np.testing.assert_array_equal(before[key],s.structure[key])
        stale=evolution.prepare(s);del stale['entry_depth']
        with self.assertRaisesRegex(ValueError,'selected entry-depth policy'):
            evolution.evolve(s,s.structure,1.,prepared=stale)
        with patch.object(depth,'quadrature',side_effect=ValueError('unresolved entry source')):
            with self.assertRaisesRegex(ValueError,'unresolved'):evolution.evolve(s,s.structure,1.)
        for key in before:np.testing.assert_array_equal(before[key],s.structure[key])

    def test_hot_and_cold_markers_use_their_local_entry_pressure_and_temperature(self):
        s=world();prepared=evolution.prepare(s);face=next(iter(prepared['entry_depth']['corners']))
        selected=np.flatnonzero(s.trace_patch==s.parcel_patch[face]);self.assertEqual(len(selected),1)
        marker=int(selected[0]);triangle=prepared['entry_depth']['triangles'][face]
        s.retained_phase_parameters.update(surface_temperature_c=0.,mantle_temperature_c=1300.,geotherm_c_per_km=15.)
        controls=evolution.parameters(s.retained_phase_parameters)
        prepared['entry_depth']['corners'][face]=np.array([-100.,-100.,200.])
        start=deepcopy(s.trace_structure)
        for barycentric in ([.8,.1,.1],[.1,.1,.8]):
            point=np.asarray(barycentric)@triangle;s.trace_xyz[marker]=point/np.linalg.norm(point)
            local_depth=max(float(np.asarray(barycentric)@prepared['entry_depth']['corners'][face]),0.)
            actual,_,_=evolution.evolve(s,start,.25,trace=True,prepared=prepared)
            expected={key:value[marker:marker+1].copy() for key,value in start.items()}
            forcing=evolution.conditions(expected,local_depth,local_depth*3300.*1000.,controls)
            phase.advance(expected,forcing['eligible_fraction'],forcing['bath_temperature_c'],forcing['thermal_tau_myr'],.25,
                reaction_tau_myr=controls['reaction_tau_myr'],drainage_rate_myr=forcing['drainage_rate_myr'],minimum_thickness_km=columns.MIN_THICKNESS_KM)
            for key in (phase.DENSE,phase.HEAT,phase.HEAT_BATH,inventory.REMAINING):
                np.testing.assert_allclose(actual[key][marker],expected[key][0],rtol=3e-13,atol=1e-10)

    def test_unentered_region_matches_original_sources_and_invalid_upgrade_is_atomic(self):
        s=world();prepared=evolution.prepare(s)
        for face in prepared['entry_depth']['corners']:prepared['entry_depth']['corners'][face]=np.array([-20.,-10.,-1.])
        before=deepcopy(s.structure);actual,_,_=evolution.evolve(s,before,1.,prepared=prepared)
        del s.entry_phase_depth
        expected,_,_=evolution.evolve(s,before,1.)
        for key in actual:np.testing.assert_array_equal(actual[key],expected[key])
        with self.assertRaises(ValueError):coupling.upgrade(s,max_panels=0)
        self.assertFalse(hasattr(s,'entry_phase_depth'))

    def test_face_heating_crosses_reaction_threshold_locally_with_conserved_heat(self):
        s=world();s.retained_phase_parameters.update(surface_temperature_c=0.,mantle_temperature_c=1300.,
            geotherm_c_per_km=15.,diffusion_length_km=5.,reaction_pressure_pa=1.3e9)
        for key in (phase.HEAT,phase.HEAT_BASELINE):s.structure[key]=phase.mass_volume(s.structure)*300.
        prepared=evolution.prepare(s);face=next(iter(prepared['entry_depth']['corners']))
        prepared['entry_depth']['corners'][face]=np.array([-10.,-10.,100.])
        start=deepcopy(s.structure);dt=1.
        actual,_,_=evolution.evolve(s,start,dt,prepared=prepared)
        controls=evolution.parameters(s.retained_phase_parameters)
        def local(z):
            z=np.atleast_1d(z);column={key:np.full(len(z),value[face]) for key,value in start.items()}
            for _ in range(4):
                forcing=evolution.conditions(column,z,z*3300.*1000.,controls)
                phase.advance(column,forcing['eligible_fraction'],forcing['bath_temperature_c'],forcing['thermal_tau_myr'],.25,
                    reaction_tau_myr=controls['reaction_tau_myr'],drainage_rate_myr=forcing['drainage_rate_myr'],minimum_thickness_km=columns.MIN_THICKNESS_KM)
            return column
        self.assertEqual(local(0.)[phase.DENSE][0],0.)
        self.assertGreater(local(100.)[phase.DENSE][0],0.)
        estimates=[]
        for intervals in (65536,131072):
            z=np.linspace(0.,100.,intervals+1);weights=np.ones(intervals+1)
            weights[1:-1:2]=4.;weights[2:-1:2]=2.;weights*=100./intervals/3.*2.*(100.-z)/110.**2
            states=local(z);zero=local(0.)
            estimates.append({key:(1.-(100./110.)**2)*zero[key][0]+weights@states[key] for key in (phase.DENSE,phase.HEAT,phase.HEAT_BATH)})
        for key,value in estimates[-1].items():
            np.testing.assert_allclose(estimates[0][key],value,rtol=2e-7,atol=1e-9)
            np.testing.assert_allclose(actual[key][face],value,rtol=4e-6,atol=1e-9)
        phase.validate(actual)

    def test_native_pressure_coupling_checkpoint_and_atomic_failure(self):
        from pathlib import Path
        import tempfile,checkpoint
        from native_engine import Simulation
        from tests.test_entry_regions import advance
        s=world();s.retained_phase_parameters['reaction_pressure_pa']=1.2e9
        report=advance(s,.000005,max_source_step_myr=.000005)
        self.assertGreater(s.structure[phase.DENSE][0],0.)
        self.assertTrue(report['persistent_entry_regions'])
        # Real native sources may add material, so use the production ledger.
        self.assertLess(abs(s.material_column_budget['phase_mass_residual_kg'])/s.material_column_budget['after_columns_mass_kg'],1e-12)
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/'entry-phase.npz';checkpoint.write_checkpoint(path,s,dict(config=s.config),{})
            resumed,_=checkpoint.read_checkpoint(path,None,Simulation)
        expected=advance(s,.000005,max_source_step_myr=.000005)
        observed=advance(resumed,.000005,max_source_step_myr=.000005)
        self.assertEqual(expected,observed)
        for key in s.structure:np.testing.assert_array_equal(s.structure[key],resumed.structure[key])
        np.testing.assert_array_equal(s.omega,resumed.omega)
        # Inject a source failure on the first interval, before the unrelated
        # third-interval capture fragmentation case recorded in the work log.
        s=world();s.retained_phase_parameters['reaction_pressure_pa']=1.2e9
        before=deepcopy(s.__dict__)
        with patch.object(depth,'quadrature',side_effect=ValueError('entry source budget exhausted')):
            with self.assertRaisesRegex(ValueError,'budget exhausted'):advance(s,.000005,max_source_step_myr=.000005)
        self.assertEqual(s.t,before['t'])
        np.testing.assert_array_equal(s.material_surface['vertices'],before['material_surface']['vertices'])
        for key in s.structure:np.testing.assert_array_equal(s.structure[key],before['structure'][key])


if __name__=='__main__':unittest.main()
