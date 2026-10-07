"""Exact juvenile footprint exclusion, native regressions and source budgets."""
from pathlib import Path
from types import SimpleNamespace
from copy import deepcopy
import hashlib,json,pickle,sys,unittest
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'tests'),str(ROOT)]
import arc_material_exclusion as exact
import arc_emplacement_geometry as geometry
import native_arc_material as arcs
import material_surface
from test_native_processes import ocean_fixture
from test_arc_source_geometry import add,positions_in_cell
FIXTURE=json.loads((ROOT/'tests/fixtures/arc_exact_exclusion_642.json').read_text())
FACE=np.array([[0,1,2]],dtype=int)
def unit(p):
    p=np.asarray(p,float);return p/np.linalg.norm(p,axis=-1,keepdims=True)
def triangle(yz):
    p=unit(np.column_stack((np.ones(3),np.asarray(yz,float))))
    if np.linalg.det(p)<0:p=p[[0,2,1]]
    return p
def context(p):return geometry._context(p,FACE,6371.)
def world_context(p):
    s=ocean_fixture(level=2);s.material_surface=dict(vertices=p,faces=FACE,radius_km=6371.)
    return s,geometry.prepare(s)

class ExactArcExclusionTests(unittest.TestCase):
    def test_all_three_native_subfloor_birth_pairs_are_rejected(self):
        for record in FIXTURE['pairs']:
            blocker,candidate=np.asarray(record['original_triangles'])
            with self.subTest(faces=record['face_ids']):
                self.assertLess(record['area_km2'],1e-8)
                witness=exact.obstruction(candidate[None],[context(blocker)])
                self.assertIsNotNone(witness)
                _,prepared=world_context(blocker)
                result=geometry.inspect(prepared,candidate,FACE,0)
                self.assertFalse(result['admissible'])
                self.assertIsNotNone(result['exact_material_obstruction'])
                # Existing area bookkeeping rounded these losses to zero.
                self.assertLessEqual(result['material_obstruction_km2'],result['admission_tolerance_km2'])
    def test_actual_small_positive_survives_proper_rotations(self):
        blocker,candidate=np.asarray(FIXTURE['pairs'][2]['original_triangles'])
        rng=np.random.default_rng(851)
        for _ in range(8):
            q,_=np.linalg.qr(rng.normal(size=(3,3)))
            if np.linalg.det(q)<0:q[:,0]*=-1
            self.assertIsNotNone(exact.obstruction((candidate@q)[None],[context(blocker@q)]))
    def test_boundary_touch_is_clear_but_arbitrarily_small_positive_is_not(self):
        blocker=triangle([[-.2,0],[.2,0],[0,.2]])
        touch=triangle([[-.2,0],[0,-.2],[.2,0]])
        self.assertIsNone(exact.obstruction(touch[None],[context(blocker)]))
        penetrate=touch.copy();penetrate[:,2]+=1e-14;penetrate=unit(penetrate)
        self.assertIsNotNone(exact.obstruction(penetrate[None],[context(blocker)]))
    def test_growth_preserves_old_overlap_and_rejects_only_new_obstruction(self):
        old=triangle([[-.2,-.2],[.2,-.2],[0,.2]])
        grown=triangle([[-.24,-.24],[.24,-.24],[0,.24]])
        old_blocker=triangle([[-.04,-.04],[.04,-.04],[0,.04]])
        self.assertIsNone(exact.obstruction(grown[None],[context(old_blocker)],context(old)))
        new_blocker=triangle([[-.03,-.23],[.03,-.23],[0,-.21]])
        self.assertIsNotNone(exact.obstruction(grown[None],[context(new_blocker)],context(old)))
    def test_active_and_excluded_old_faces_do_not_become_new_obstructions(self):
        p=triangle([[-.1,-.1],[.1,-.1],[0,.1]])
        ctx=context(p);ctx['active']=np.array([False])
        self.assertIsNone(exact.obstruction(p[None],[ctx]))
        ctx['active'][:]=True;ctx['exclude']=np.array([0])
        self.assertIsNone(exact.obstruction(p[None],[ctx]))
    def test_malformed_occupancy_masks_fail_closed(self):
        p=triangle([[-.1,-.1],[.1,-.1],[0,.1]])
        invalid_active=(np.array([1]),np.array([True,False]),np.array([[True]]),np.array(True))
        invalid_excluded=(np.array([True]),np.array([0.]),np.array([[0]]),
                          np.array([-1]),np.array([1]),np.array([2],dtype=np.uint64))
        for key,values in (('active',invalid_active),('exclude',invalid_excluded)):
            for value in values:
                with self.subTest(key=key,value=value):
                    ctx=context(p);ctx[key]=value
                    with self.assertRaisesRegex(ValueError,'Material exclusion'):
                        exact.obstruction(p[None],[ctx])
        ctx=context(p);ctx['exclude']=np.empty(0,dtype=int)
        self.assertIsNotNone(exact.obstruction(p[None],[ctx]))
    def test_cached_material_and_old_geometry_changes_fail_closed(self):
        p=triangle([[-.1,-.1],[.1,-.1],[0,.1]])
        for stale_old in (False,True):
            with self.subTest(stale_old=stale_old):
                blocker=context(p);old=context(p) if stale_old else None
                exact.obstruction(p[None],[blocker],old)
                stale=old if stale_old else blocker
                stale['triangles']=stale['triangles'].copy()
                stale['triangles'][0,0,0]=np.nextafter(stale['triangles'][0,0,0],np.inf)
                with self.assertRaisesRegex(ValueError,'context geometry changed'):
                    exact.obstruction(p[None],[blocker],old)
    def test_empty_and_global_candidate_contexts_are_complete(self):
        empty=geometry._context(np.empty((0,3)),np.empty((0,3),int),6371.)
        p=triangle([[.01,.01],[.02,.01],[.01,.02]])
        self.assertIsNone(exact.obstruction(p[None],[empty]))
        large=np.eye(3);idx=exact._Index(large[None]);self.assertEqual(len(idx.global_faces),1)
        self.assertIsNotNone(exact.obstruction(p[None],[context(large)]))
    def test_birth_admission_checks_actual_post_normalization_geometry(self):
        data=FIXTURE['normalization_crossing'];blocker=np.asarray(data['blocker']);candidate=np.asarray(data['candidate'])
        self.assertIsNone(exact.obstruction(candidate[None],[context(blocker)]))
        stored=unit(candidate);self.assertIsNotNone(exact.obstruction(stored[None],[context(blocker)]))
        _,ctx=world_context(blocker)
        # A fixed obstructed proposal must keep every source unit pending.
        result=geometry.admit(ctx,lambda amount:dict(vertices=candidate,faces=FACE),1.,0)
        self.assertEqual(result['accepted_area_km2'],0.)
        self.assertEqual(result['pending_area_km2'],1.)
    def test_accepted_backtracking_preserves_location_and_stored_clearance(self):
        blocker=triangle([[-.05,0],[.05,0],[0,.05]])
        s,ctx=world_context(blocker);point=unit([1,0,-5/6371.])
        result=geometry.admit(ctx,lambda amount:arcs._patch(point,[0,1,0],amount),2000.,0)
        self.assertGreater(result['accepted_area_km2'],0.)
        self.assertLess(result['accepted_area_km2'],2000.)
        self.assertEqual(result['accepted_area_km2']+result['pending_area_km2'],2000.)
        plan=result['plan'];np.testing.assert_array_equal(plan['vertices'][0],point)
        stored=material_surface.initialize_surface(plan['vertices'],plan['faces'],np.zeros(len(plan['faces']),int),np.full(len(plan['faces']),3))
        self.assertIsNone(exact.obstruction(stored['vertices'][stored['faces']],[context(blocker)]))
        self.assertEqual(hashlib.sha256(stored['vertices'][stored['faces']].tobytes()).hexdigest(),plan['_emplacement_stored_triangle_sha256'])
    def test_stale_admitted_birth_rejects_before_state_identifiers_change(self):
        s=ocean_fixture(level=2);point=positions_in_cell(s)[:1][0]
        plan=arcs._patch(point,[0,1,0],10.)
        plan['_emplacement_stored_triangle_sha256']='0'*64
        before=pickle.dumps(s,protocol=5)
        with self.assertRaisesRegex(ValueError,'stored footprint changed'):
            arcs._birth(s,0,0,10.,prepared_patch=plan)
        self.assertEqual(before,pickle.dumps(s,protocol=5))
    def test_recursive_frozen_source_capture_contains_exact_guard(self):
        import server
        sources=server.capture_auxiliary_sources((ROOT/'tectonics.py').read_bytes())
        self.assertEqual(sources['arc_material_exclusion.py'],(ROOT/'arc_material_exclusion.py').read_bytes())

    def test_native_birth_keeps_source_volume_and_pending_budget(self):
        s=ocean_fixture(level=2);s.native_arc_emplacement_version=1
        p=positions_in_cell(s)[:1];report=add(s,p,[100.])
        self.assertAlmostEqual(report['added_area_km2']+report['pending_area_km2'],100.,places=9)
        self.assertAlmostEqual(float(s.material_surface['area_km2']@s.structure['thickness_km']),25*report['added_area_km2'],places=6)
        self.assertEqual(report['emplacement_geometry']['sources'][0]['accepted_footprint']['material_exclusion_version'],1)
        restored=pickle.loads(pickle.dumps(s,protocol=5));before=deepcopy(s.native_arc_pending)
        add(restored,np.empty((0,3)),[],owners=np.empty(0,int))
        np.testing.assert_array_equal(before['xyz'],restored.native_arc_pending['xyz'])

if __name__=='__main__':unittest.main()

