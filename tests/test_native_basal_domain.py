"""Complete control measure and native finite-bottom adapter checks."""
from copy import deepcopy
from fractions import Fraction
from itertools import product
from pathlib import Path
from types import SimpleNamespace
import sys
import unittest

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from benchmarks.collision_architecture import native_basal_domain as native
from benchmarks.collision_architecture import sparse_shared_contact as sparse
from benchmarks.collision_architecture import finite_basal_operator as finite
from collision_interface import _precise_rotation_integral

R = 6371000.


def sphere():
    points = np.r_[np.eye(3), -np.eye(3)]
    faces = []
    for signs in product((0,3), repeat=3):
        face = np.arange(3)+signs
        if np.linalg.det(points[face]) < 0: face = face[[0,2,1]]
        faces.append(face)
    return points, np.asarray(faces)


def state(layers=1, *, mixed=False, active=None, triangle=None, rotation=None):
    cp, cf = sphere()
    triangle = cp[cf[0]] if triangle is None else triangle
    if rotation is not None:
        cp = cp@rotation.T; triangle = triangle@rotation.T
    owners = np.arange(layers)
    active = np.ones(layers,bool) if active is None else np.asarray(active,bool)
    slots = np.flatnonzero(active)
    owners = slots[:layers]
    points = np.tile(triangle, (layers,1)); faces = np.arange(3*layers).reshape(-1,3)
    support = np.zeros((len(active),len(cf)))
    support[slots[0]] = 1.
    if mixed: support[slots] = 1./len(slots)
    return SimpleNamespace(material_surface=dict(vertices=points,faces=faces,
        face_id=np.arange(100,100+layers),face_owner=owners,vertex_owner=np.repeat(owners,3),radius_km=R/1000.),
        native_mesh=dict(vertices=cp,faces=cf,radius_km=R/1000.), cell_area=np.full(len(cf),np.pi/2*(R/1000.)**2),
        support=support,crust=np.zeros(len(cf),int),active=active,plate_uid=np.arange(len(active))+20,
        parcel_collision_sheet=np.arange(layers)+10,parcel_plate=owners,parcel_patch=np.arange(100,100+layers),
        collision_contacts=[dict(top_sheet=11+i,under_sheet=10+i) for i in range(layers-1)])


def system(s):
    domain = native.prepare(s, allocation_policy='resolved-material-bottom')
    p, f = domain.points, domain.faces; n = len(domain.active_native_slots)
    result = sparse.assemble(p,f,domain.vertex_plate,radius_m=R, basal_drag_pa_s_per_m=1.,
        viscosity_pa_s=0.,sheet_thickness_m=4e4,basal_reference_velocity_m_s=np.zeros_like(p),
        other_plate_rotational_drag_n_m_s=np.zeros((3*n,3*n)), external_torques_n_m=np.zeros((n,3)),
        external_nodal_forces_n=np.zeros_like(p),contacts=[])
    return domain, result


def build(domain, sys):
    return domain.build(sys,mantle_omega_rad_s=np.array([1.,-2.,3.])*1e-15,
                        quadrature_relative_tolerance=2e-10,max_order=64)


class NativeBasalDomainTests(unittest.TestCase):
    def test_global_measure_and_rigid_moving_mantle(self):
        s = state(); domain, sys = system(s); result = build(domain,sys)
        self.assertTrue(result.certificate['global_allocation_complete'])
        self.assertEqual(result.certificate['material_faces_covered'],1)
        self.assertEqual(result.certificate['control_count'],8)
        self.assertEqual(len(result.component.diagnostics),8)
        omega = np.array([1.,-2.,3.])*1e-15
        y = np.zeros(len(sys['load_n'])); y[:3] = [.003,.001,-.002]
        slip = y[:3]-R*omega
        beta = domain._data['beta'][0]
        expected = beta*(8*np.pi*R**2/3)*float(slip@slip)
        self.assertAlmostEqual(result.component.work(y)['basal_dissipation_w']/expected,1.,places=10)
        y[:3] = R*omega
        self.assertLess(result.component.work(y)['basal_dissipation_w'],expected*1e-20)

    def test_control_holes_duplicates_and_orientation_reject(self):
        s = state()
        for faces in (s.native_mesh['faces'][:-1], np.r_[s.native_mesh['faces'],s.native_mesh['faces'][:1]],
                      s.native_mesh['faces'][:,[0,2,1]]):
            altered = deepcopy(s); altered.native_mesh['faces'] = faces
            altered.cell_area=np.full(len(faces),s.cell_area[0]); altered.support=np.ones((1,len(faces)))
            altered.crust=np.zeros(len(faces),int)
            with self.assertRaises(ValueError): native.prepare(altered,allocation_policy='resolved-material-bottom')
        doubled = deepcopy(s)
        doubled.native_mesh['vertices']=np.tile(s.native_mesh['vertices'],(2,1))
        doubled.native_mesh['faces']=np.r_[s.native_mesh['faces'],s.native_mesh['faces']+6]
        doubled.cell_area=np.tile(s.cell_area,2); doubled.support=np.ones((1,16)); doubled.crust=np.zeros(16,int)
        with self.assertRaisesRegex(ValueError,'exactly once'):
            native.prepare(doubled,allocation_policy='resolved-material-bottom')

    def test_mixed_support_explicit_policy_and_ocean_only_slot_mapping(self):
        s=state(active=[True,False,True],mixed=True)
        strict=native.prepare(s,allocation_policy='preserve-native')
        with self.assertRaises(native.NativeBasalBlocker) as caught: strict.probe_cells([0])
        self.assertEqual(caught.exception.witness['control_id'],0)
        domain=native.prepare(s,allocation_policy='resolved-material-bottom')
        self.assertEqual(domain.owner_to_plate_slot,{0:0,2:1})
        probe=domain.probe_cells([0,1]); self.assertFalse(probe['global_allocation_complete'])
        self.assertEqual(probe['allocations'][0]['pieces'][0]['fractions'],{0:1.})
        self.assertEqual(probe['allocations'][1]['pieces'][0]['fractions'],{0:.5,2:.5})
        domain.validate_system(dict(plate_count=2,nplate=6,radius_m=R,points=domain.points,
                                   faces=domain.faces,vertex_plate=domain.vertex_plate))

    def test_triple_stack_only_unique_lowest_gets_drag_and_same_owner_order_required(self):
        s=state(3,mixed=True); domain,sys=system(s); result=build(domain,sys)
        covered=[d for d in result.component.diagnostics if d['kind']=='bottom-material']
        self.assertEqual(len(covered),1); self.assertEqual(covered[0]['bottom_face'],100)
        bad=deepcopy(s); bad.collision_contacts=[dict(top_sheet=11,under_sheet=10),dict(top_sheet=12,under_sheet=10)]
        with self.assertRaisesRegex(native.NativeBasalBlocker,'unique saved layer order'):
            native.prepare(bad,allocation_policy='resolved-material-bottom').probe_cells([0])
        bad=deepcopy(s); bad.collision_contacts=[]
        bad.material_surface['face_owner'][:]=0; bad.material_surface['vertex_owner'][:]=0; bad.parcel_plate[:]=0
        with self.assertRaises(native.NativeBasalBlocker):
            native.prepare(bad,allocation_policy='resolved-material-bottom').probe_cells([0])

    def test_finite_candidates_equal_exhaustive_and_global_face_fallback(self):
        s=state(2); domain=native.prepare(s,allocation_policy='resolved-material-bottom')
        for i in range(8):
            actual,_=domain.candidates(i)
            exact=native._positive_candidates(domain._controls[i],domain._triangles,np.arange(2))
            np.testing.assert_array_equal(actual,exact)
        # Large triangles at high resolution enter the native global-face list.
        locator=native._fresh_locator(domain.points,domain.faces,bin_resolution=128)
        self.assertEqual(len(locator['global_faces']),2)
        broad=native.mesh_coverage._candidates(None,domain._centres[0],domain._chords[0],locator)
        np.testing.assert_array_equal(native._positive_candidates(domain._controls[0],domain._triangles,broad),[0,1])

    def test_subdivision_and_rotation_preserve_complete_metric(self):
        s=state(); domain,sys=system(s); whole=build(domain,sys)
        split=deepcopy(s); cp=split.native_mesh['vertices']; cf=split.native_mesh['faces']; faces=[]; areas=[]
        # Split every indexed triangle about an interior centre, preserving all shared edges.
        for i,face in enumerate(cf):
            centre=cp[face].sum(axis=0); centre/=np.linalg.norm(centre); at=len(cp); cp=np.r_[cp,centre[None]]
            for a,b in zip(face,np.roll(face,-1)):
                child=[a,b,at]; faces.append(child)
                areas.append(_precise_rotation_integral(cp[child])[1]*(R/1000.)**2)
        split.native_mesh.update(vertices=cp,faces=np.asarray(faces)); split.cell_area=np.asarray(areas)
        split.support=np.ones((1,len(faces))); split.crust=np.zeros(len(faces),int)
        sub=native.prepare(split,allocation_policy='resolved-material-bottom'); divided=build(sub,sys)
        a=whole.component.hessian_n_s_m.to_dense(); b=divided.component.hessian_n_s_m.to_dense()
        scale=np.sqrt(np.diag(a)); self.assertLess(np.linalg.norm((a-b)/(scale[:,None]*scale[None]),2),2e-9)
        q=np.array([[0.,0.,1.],[1.,0.,0.],[0.,1.,0.]])
        rotated=state(rotation=q); rd,rs=system(rotated); rb=build(rd,rs)
        np.testing.assert_allclose(rb.component.hessian_n_s_m.to_dense()[:3,:3],q@a[:3,:3]@q.T,rtol=2e-10,atol=np.max(abs(a))*1e-13)

    def test_stale_source_system_and_duplicate_probe_controls_reject(self):
        s=state(); domain,sys=system(s); component=build(domain,sys).component
        with self.assertRaises(ValueError): domain.probe_cells([0,0])
        changed=dict(sys,vertex_plate=sys['vertex_plate']+1)
        with self.assertRaises(ValueError): domain.validate_system(changed)
        s.support[0,0]=.5
        with self.assertRaisesRegex(ValueError,'changed'): component.validate_system(sys)

    def test_arbitrarily_thin_positive_geometry_is_not_an_area_floor(self):
        triangle=np.array([[1.,0.,0.],[1.,1e-8,0.],[1.,1e-8,1e-8]])
        triangle/=np.linalg.norm(triangle,axis=1)[:,None]
        s=state(triangle=triangle)
        d=native.prepare(s,allocation_policy='resolved-material-bottom')
        np.testing.assert_array_equal(d.candidates(0)[0],[0])
        # Even if later metric/cast gates reject, discovery cannot erase this region.
        self.assertLess(_precise_rotation_integral(triangle)[1]*(R/1000.)**2,1e-8)

    def test_native_337_trace_uses_backward_stable_solve_without_gate_change(self):
        points=np.array([[-.1573993364881588,-.683905756038392,-.7123891954056033],
                         [-.15874684124694943,-.6840470938399152,-.7119543621632515],
                         [-.15996232812930478,-.6829844086065944,-.7127021475902491]])
        polygon=np.array([[-.15990375243841362,-.6830356803684022,-.7126661555734132],
                          [-.15820378135738994,-.683626850647247,-.7124786962698252],
                          [-.1588789856258886,-.6839316878241715,-.7120357535380291]])
        axes=np.eye(3)[np.argmin(abs(points),axis=1)]; first=np.cross(points,axes)
        first/=np.linalg.norm(first,axis=1)[:,None]
        sys=dict(points=points,tangent_basis=np.stack((first,np.cross(points,first)),axis=2),radius_m=R)
        factor,area,metric=finite._trace_rule(sys,np.arange(3),polygon,8,1.,1.)
        analytic,expected_area=_precise_rotation_integral(polygon)
        self.assertAlmostEqual(area/(expected_area*R**2),1.,places=11)
        np.testing.assert_allclose(metric,analytic*R**2,rtol=2e-12)
        np.testing.assert_allclose(factor[:,:3].T@factor[:,:3],analytic*R**2,rtol=2e-12)
        # An independent sample reconstruction identifies the old arithmetic
        # failure; this threshold is exactly the production research gate.
        nodes,_=np.polynomial.legendre.leggauss(8)
        u,v=np.meshgrid(.5*(nodes+1),.5*(nodes+1),indexing='ij'); u,v=u.ravel(),v.ravel()
        a,b,c=polygon; q=a+u[:,None]*(b-a)+(1-u)[:,None]*v[:,None]*(c-a)
        q/=np.linalg.norm(q,axis=1)[:,None]
        good=np.linalg.solve(points.T,q.T).T
        self.assertLess(np.max(np.linalg.norm(good@points-q,axis=1)),256*np.finfo(float).eps)

    def test_derived_owner_mapping_locator_and_original_boolean_values_reject(self):
        s=state(active=[True,True,True],mixed=True)
        for mutate in (lambda d:d.owner_to_plate_slot.update({1:2,2:1}),
                       lambda d:d.face_id_to_index.update({100:1}),
                       lambda d:d._locator.update(global_faces=np.empty(0,int),candidates=np.empty(0,int))):
            domain=native.prepare(s,allocation_policy='resolved-material-bottom'); mutate(domain)
            with self.assertRaisesRegex(ValueError,'mapping changed'): domain.probe_cells([0])
        for key in ('radius','area'):
            altered=state()
            if key=='radius':
                altered.material_surface['radius_km']=True; altered.native_mesh['radius_km']=True
            else: altered.cell_area=np.ones(8,bool)
            with self.assertRaises(ValueError): native.prepare(altered,allocation_policy='resolved-material-bottom')

    def test_local_broadcast_predicate_matches_independent_exact_original_edges(self):
        rng=np.random.default_rng(703)
        a=rng.normal(size=(4,1,3)); a/=np.linalg.norm(a,axis=-1,keepdims=True)
        b=a+1e-7*rng.normal(size=a.shape); b/=np.linalg.norm(b,axis=-1,keepdims=True)
        p=np.concatenate((a,b,(a+b)/np.linalg.norm(a+b,axis=-1,keepdims=True),
                          rng.normal(size=(4,5,3))),axis=1)
        measured=native.edge_distances(p,a,b)
        for row in range(4):
            aa=list(map(Fraction,a[row,0])); bb=list(map(Fraction,b[row,0]))
            coefficient=(aa[1]*bb[2]-aa[2]*bb[1],aa[2]*bb[0]-aa[0]*bb[2],aa[0]*bb[1]-aa[1]*bb[0])
            for j,point in enumerate(p[row]):
                exact=sum((c*Fraction(float(x)) for c,x in zip(coefficient,point)),Fraction(0))
                self.assertEqual(np.sign(measured[row,j]),(exact>0)-(exact<0))
        np.testing.assert_array_equal(native.edge_distances(np.zeros((4,3)),a[:,0],b[:,0]),np.zeros(4))
        with self.assertRaisesRegex(ValueError,'representable'):
            native.edge_distances([0.,0.,1e-320],[1.,0.,0.],[0.,1e-320,0.])


if __name__=='__main__': unittest.main()
