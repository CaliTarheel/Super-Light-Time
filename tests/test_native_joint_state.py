"""Native mapping and independent SI virtual-work checks; no live evolution."""
from copy import deepcopy
from types import SimpleNamespace
import unittest

import numpy as np

import enhanced_rifting
import gravitational_relaxation as gravity
import mesh_geometry
import plate_balance
from benchmarks.collision_architecture import native_joint_state as port
from benchmarks.collision_architecture import sparse_shared_contact as sparse
from benchmarks.collision_architecture import sparse_joint_solver as solver
from tests.test_shared_contact import fixture, unit


def native_fixture():
    cfg = fixture(contacts=False); p, f = cfg['points'], cfg['faces']
    owner = np.where(cfg['vertex_plate'] == 0, 1, 6)
    area = mesh_geometry.spherical_area(p, f, radius_km=6371.)
    active = np.zeros(8, bool); active[[1, 3, 6]] = True
    mesh = dict(vertices=p.copy(), faces=f.copy(), vertex_owner=owner,
        face_owner=owner[f[:, 0]], face_id=np.array([101, 907]), face_kind=np.ones(2, int),
        area_km2=area, radius_km=6371.)
    return SimpleNamespace(t=642., material_mechanics_version=1, material_surface=mesh,
        active=active, plate_uid=np.arange(8)+21, parcel_plate=mesh['face_owner'].copy(),
        parcel_patch=mesh['face_id'].copy(), parcel_collision_sheet=np.array([11, 22]),
        structure=dict(thickness_km=np.array([40., 50.]), reference_thickness_km=np.array([32., 35.])),
        config={}, collision_contacts=[], retained_dense_crust_version=0)


class NativeJointStateTests(unittest.TestCase):
    def test_all_active_plate_slots_and_read_only_source(self):
        s = native_fixture(); before = port.source_signature(s); keys = set(vars(s))
        m = port.prepare(s)
        np.testing.assert_array_equal(m.active_native_slots, [1, 3, 6])
        np.testing.assert_array_equal(m.vertex_plate, [0, 0, 0, 2, 2, 2])
        np.testing.assert_array_equal(m.plate_uids, [22, 24, 27])
        self.assertEqual(m.unknown_count, 21)
        self.assertEqual(port.source_signature(s), before); self.assertEqual(keys, set(vars(s)))
        self.assertFalse(m.points.flags.writeable)
        self.assertFalse(np.shares_memory(m.points, s.material_surface['vertices']))

    def test_velocity_round_trip_and_total_material_motion(self):
        s = native_fixture(); m = port.prepare(s); rng = np.random.default_rng(7)
        omega = rng.normal(size=(8, 3))*1e-16; omega[~s.active] = 0.
        residual = np.einsum('nik,nk->ni', m.tangent_basis, rng.normal(size=(6, 2))*1e-10)
        y = m.pack_velocity(omega, residual); actual = m.unpack_velocity(y)
        np.testing.assert_allclose(actual['native_omega_rad_s'], omega, rtol=2e-15)
        np.testing.assert_allclose(actual['residual_velocity_m_s'], residual, rtol=2e-15, atol=1e-25)
        expected = np.cross(omega[s.material_surface['vertex_owner']]*m.radius_m, m.points)+residual
        np.testing.assert_allclose(actual['total_velocity_m_s'], expected, rtol=2e-15, atol=1e-25)
        with self.assertRaisesRegex(ValueError, 'tangent'):
            m.pack_velocity(omega, residual+1e-10*m.points)

    def test_selected_craton_cores_preserve_mobile_rims(self):
        s = native_fixture(); p = unit([[1, x, y] for y in (-.02, 0., .02) for x in (-.02, 0., .02)])
        f = []
        for row in (0, 1):
            for column in (0, 1):
                a = 3*row+column; f.extend([[a,a+1,a+4],[a,a+4,a+3]])
        f = np.array(f); n = len(f)
        s.material_surface = dict(vertices=p, faces=f, vertex_owner=np.ones(9, int),
            face_owner=np.ones(n, int), face_id=np.arange(n)+200, face_kind=np.full(n, 2),
            area_km2=mesh_geometry.spherical_area(p,f,radius_km=6371.),radius_km=6371.)
        s.parcel_patch=np.arange(n)+200; s.parcel_plate=np.ones(n,int)
        s.parcel_collision_sheet=np.full(n,11); s.config={'enhanced_rifting':{'enabled':True,'craton_margin_km':150.}}
        s.structure=dict(thickness_km=np.full(n,40.),reference_thickness_km=np.full(n,32.))
        m = port.prepare(s); expected, _ = enhanced_rifting.craton_anchors(s.material_surface,150.)
        np.testing.assert_array_equal(m.craton_anchor_vertices, expected)
        self.assertEqual(int(expected.sum()),1); self.assertEqual(m.fixed_residual_dofs.sum(),2)

    def test_point_contact_keeps_native_guard_without_material_weld(self):
        s = native_fixture(); p=unit([[1,0,0],[1,.1,0],[1,0,.1],[1,-.1,0],[1,0,-.1]])
        f=np.array([[0,1,2],[0,3,4]])
        s.material_surface.update(vertices=p,faces=f,vertex_owner=np.ones(5,int),face_owner=np.ones(2,int),
            area_km2=mesh_geometry.spherical_area(p,f,radius_km=6371.))
        s.parcel_plate=np.ones(2,int); s.parcel_collision_sheet[:]=11
        m=port.prepare(s)
        np.testing.assert_array_equal(np.flatnonzero(m.corner_guard_vertices),[0])
        self.assertTrue(m.fixed_residual_dofs[0].all())
        y=np.zeros(m.unknown_count); y[9]=1e-9
        with self.assertRaisesRegex(ValueError,'prescribed'):
            m.unpack_velocity(y)

    def test_mutations_and_invalid_native_identity_fail(self):
        for mutate in (lambda s:s.parcel_patch.__setitem__(0,999),
                       lambda s:s.material_surface['vertex_owner'].__setitem__(0,3),
                       lambda s:s.plate_uid.__setitem__(3,s.plate_uid[1]),
                       lambda s:setattr(s,'active',s.active.astype(float))):
            s=native_fixture(); mutate(s)
            with self.assertRaises(ValueError): port.prepare(s)
        s=native_fixture(); m=port.prepare(s); s.structure['thickness_km'][0]+=1
        with self.assertRaisesRegex(ValueError,'changed'): m.validate_source(s)
        s=native_fixture(); m=port.prepare(s); m.vertex_plate.flags.writeable=True; m.vertex_plate[0]=1
        with self.assertRaisesRegex(ValueError,'modified'): m.validate_source(s)

    def test_gravity_force_matches_conserved_volume_energy_derivative(self):
        s=native_fixture(); m=port.prepare(s); result=port.gravitational_load(s,m)
        rng=np.random.default_rng(11); y=rng.normal(size=m.unknown_count)*1e-9
        velocity=m.unpack_velocity(y)['total_velocity_m_s']
        volume=s.material_surface['area_km2']*s.structure['thickness_km']
        def energy(seconds):
            points=unit(m.points+seconds*velocity/m.radius_m)
            return gravity.ENERGY_UNIT_J*gravity.reference_energy(points,m.faces,volume,
                s.structure['reference_thickness_km'],m.sheets,radius=m.radius_m/1000.)
        seconds=1e9
        derivative=(energy(seconds)-energy(-seconds))/(2*seconds)
        work=float(np.sum(result['external_nodal_forces_n']*velocity))
        self.assertLess(abs(derivative+work)/max(abs(work),abs(derivative)),2e-7)
        rigid=m.radius_m*np.cross(m.points,result['external_nodal_forces_n'])
        np.testing.assert_allclose(result['rigid_share_torque_n_m'][0],rigid[:3].sum(axis=0),rtol=2e-15)
        np.testing.assert_array_equal(result['rigid_share_torque_n_m'][1],np.zeros(3))
        self.assertFalse(result['add_separate_collision_torque'])

    def test_driver_units_and_collision_exclusion_by_independent_virtual_power(self):
        s=native_fixture(); m=port.prepare(s); rng=np.random.default_rng(9)
        drivers={name:rng.normal(size=9)*1e12 for name in ('slab','ridge','collision')}
        actual=port.rigid_driver_torques(m,drivers,m.active_native_slots)
        omega=rng.normal(size=(3,3))*1e-16
        expected=float((drivers['slab']+drivers['ridge'])@(m.radius_m*omega.ravel()/plate_balance.CM_YR_M_S))
        self.assertAlmostEqual(float(np.sum(actual['external_torques_n_m']*omega))/expected,1.,places=14)
        drivers['collision']*=1e6
        changed=port.rigid_driver_torques(m,drivers,m.active_native_slots)
        np.testing.assert_array_equal(actual['external_torques_n_m'],changed['external_torques_n_m'])
        with self.assertRaisesRegex(ValueError,'order'):
            port.rigid_driver_torques(m,drivers,m.active_native_slots[::-1])

    def test_active_entry_and_partial_authored_response_not_silently_dropped(self):
        s=native_fixture(); s.continental_entry_regions={'version':1}; m=port.prepare(s)
        with self.assertRaisesRegex(ValueError,'coupled entry'): port.gravitational_load(s,m)
        s=native_fixture(); s.config['world_design']=dict(enabled=True,interventions=[dict(id='weak',kind='weak',
            lon_deg=0.,lat_deg=0.,radius_deg=30.,strength=.5,start_myr=0.,end_myr=1000.,reason='test')])
        with self.assertRaisesRegex(ValueError,'Partial authored'): port.prepare(s)

    def test_binding_distinguishes_nested_dictionary_structure(self):
        self.assertNotEqual(port._digest({'a':{'b':1},'c':2}),port._digest({'a':{'b':1,'c':2}}))
        s=native_fixture(); s.config={'enhanced_rifting':{'enabled':False},'world_design':{'enabled':False}}
        m=port.prepare(s)
        s.config={'enhanced_rifting':{'enabled':False,'world_design':{'enabled':False}}}
        with self.assertRaisesRegex(ValueError,'changed'): m.validate_source(s)

    def test_original_types_ids_versions_and_geometry_are_validated(self):
        valid=native_fixture(); valid.material_surface['face_kind'][0]=3
        port.prepare(valid)  # Native retained material includes island arcs.
        mutations=[
            lambda s:s.material_surface.__setitem__('vertices',s.material_surface['vertices'].astype(complex)+1j),
            lambda s:s.material_surface['face_id'].__setitem__(0,-1),
            lambda s:s.material_surface.__setitem__('face_id',np.array([101,2**63],dtype=np.uint64)),
            lambda s:s.material_surface['face_kind'].__setitem__(0,99),
            lambda s:s.material_surface.__setitem__('radius_km',True),
            lambda s:setattr(s,'native_arc_birth_profile_version',2),
            lambda s:setattr(s,'material_mechanics_version',1.),
            lambda s:s.material_surface.__setitem__('area_km2',2*s.material_surface['area_km2']),
        ]
        for mutate in mutations:
            s=native_fixture(); mutate(s)
            with self.subTest(mutate=mutate),self.assertRaises(ValueError): port.prepare(s)

    def test_native_ledger_rejects_wrong_radius_and_complex_values(self):
        s=native_fixture(); m=port.prepare(s)
        drivers={name:np.zeros(9) for name in ('slab','ridge','collision')}
        drivers['slab']=np.ones(9,dtype=complex)*1j
        with self.assertRaisesRegex(ValueError,'real'):port.rigid_driver_torques(m,drivers,m.active_native_slots)
        drivers['slab']=np.ones(9)
        s.material_surface['radius_km']*=.5; s.material_surface['area_km2']*=.25
        m=port.prepare(s)
        with self.assertRaisesRegex(ValueError,'Earth radius'):port.rigid_driver_torques(m,drivers,m.active_native_slots)

    def test_adapter_to_solver_with_ocean_slot_anchors_and_native_gravity(self):
        # A manufactured small native state and explicitly chosen resistance;
        # this is integration/work coverage, not a calibrated production solve.
        s=native_fixture(); s.material_surface['face_kind'][0]=2
        m=port.prepare(s); g=port.gravitational_load(s,m)
        system=sparse.assemble(**m.assembly_geometry(),basal_drag_pa_s_per_m=1e15,
            viscosity_pa_s=2e21,sheet_thickness_m=s.structure['thickness_km']*1000.,
            basal_reference_velocity_m_s=np.zeros_like(m.points),
            other_plate_rotational_drag_n_m_s=np.eye(3*m.plate_count)*4e38,
            external_torques_n_m=np.zeros((m.plate_count,3)),
            external_nodal_forces_n=g['external_nodal_forces_n'],contacts=[])
        np.testing.assert_array_equal(system['points'],m.points)
        np.testing.assert_array_equal(system['gauge_exchange_dimension'],[0,0,3])
        np.testing.assert_allclose(system['nodal_load_n'][:9].reshape(3,3)*m.radius_m,
            g['rigid_share_torque_n_m'],rtol=4e-14,atol=1e8)
        solved=solver.solve(system)
        fields=m.unpack_velocity(solved['generalized_velocity_m_s'])
        np.testing.assert_allclose(fields['total_velocity_m_s'],solved['total_velocity_m_s'],rtol=3e-14,atol=1e-24)
        np.testing.assert_array_equal(fields['residual_velocity_m_s'][:3],0.)
        d=solved['diagnostics']
        self.assertLessEqual(abs(d['physical_power_residual_w']),d['physical_power_acceptance_allowance_w'])
        m.validate_source(s)


if __name__=='__main__': unittest.main()
