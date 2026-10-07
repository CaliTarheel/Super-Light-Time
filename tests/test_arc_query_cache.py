"""Context-local memoization must preserve every exact admission decision."""
from fractions import Fraction as F
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
import json,sys,unittest
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'tests'),str(ROOT)]
import arc_emplacement_geometry as geometry
import arc_material_exclusion as exact
import convex_partition
import mesh_geometry
import native_spreading as clipping
from test_arc_emplacement_pruning import material,probes,reference_remaining

FACE=np.array([[0,1,2]],dtype=int)
FIXTURE=json.loads((ROOT/'tests/fixtures/arc_exact_exclusion_642.json').read_text())


def triangle(yz):
    p=np.column_stack((np.ones(3),np.asarray(yz,float)))
    p/=np.linalg.norm(p,axis=1)[:,None]
    return p if np.linalg.det(p)>0 else p[[0,2,1]]


def world(vertices,faces=FACE):
    # This fixture prepares geometry only: no Simulation constructor or step.
    mesh=mesh_geometry.icosphere(2)
    s=SimpleNamespace(material_surface=dict(vertices=vertices,faces=faces,radius_km=6371.),
        native_mesh=mesh,native_locator=mesh_geometry.build_locator(mesh['vertices'],mesh['faces']),
        plate=np.zeros(len(mesh['faces']),np.int16),support=np.ones((1,len(mesh['faces']))))
    return geometry.prepare(s)


def reference_obstruction(triangles,contexts,old=None):
    """Original exact face loop, with no cached planes or shared query bounds."""
    indices=[exact._index(c) for c in contexts]
    old_index=None if old is None else exact._index(old)
    reference=[(F(0),F(0)),(F(1),F(0)),(F(0),F(1))]
    for candidate,t in enumerate(triangles):
        basis=exact._rays(t)
        old_planes=[] if old_index is None else [exact._planes(old_index.exact(int(i)),basis)
                                                 for i in old_index.candidates(t)]
        for number,(context,index) in enumerate(zip(contexts,indices)):
            selected=index.candidates(t)
            if 'active' in context:selected=selected[context['active'][selected]]
            if 'exclude' in context:selected=selected[~np.isin(selected,context['exclude'])]
            for face in selected:
                inside,_=exact._partition(reference,exact._planes(index.exact(int(face)),basis))
                if not inside:continue
                parts=[inside]
                for planes in old_planes:
                    parts=[q for p in parts for q in exact._partition(p,planes)[1]]
                    if not parts:break
                if parts:
                    return dict(candidate_face=int(candidate),material_context=number,material_face=int(face),
                        exact_chart_area=str(exact._area(parts[0])),
                        chart_polygon=[[str(x),str(y)] for x,y in parts[0]])
    return None


class ArcQueryCacheTests(unittest.TestCase):
    def assert_nested_equal(self,a,b):
        if isinstance(a,np.ndarray):
            self.assertEqual(a.dtype,b.dtype);np.testing.assert_array_equal(a,b)
        elif isinstance(a,dict):
            self.assertEqual(a.keys(),b.keys())
            for key in a:self.assert_nested_equal(a[key],b[key])
        elif isinstance(a,(list,tuple)):
            self.assertEqual(len(a),len(b))
            for x,y in zip(a,b):self.assert_nested_equal(x,y)
        else:self.assertEqual(a,b)

    def test_locator_queries_are_complete_sorted_and_reused(self):
        context=material(level=2)
        queries=probes(8)+[np.eye(3),-np.eye(3)]
        with patch.object(clipping,'_candidates',wraps=clipping._candidates) as call:
            for p in queries:
                expected=clipping._candidates(p,context)
                first=geometry._candidates(p,context)
                before=call.call_count
                second=geometry._candidates(p.copy(),context)
                self.assertEqual(before,call.call_count)
                np.testing.assert_array_equal(first,expected)
                np.testing.assert_array_equal(second,expected)
                self.assertFalse(first.flags.writeable)
                self.assertTrue(np.all(np.diff(first)>0))

    def test_lookup_snapshot_does_not_mutate_native_locator(self):
        locator=mesh_geometry.build_locator(*[mesh_geometry.icosphere(1)[k] for k in ('vertices','faces')])
        copied=geometry._lookup_locator(locator)
        for key in ('keys','offsets','candidates','global_faces'):
            np.testing.assert_array_equal(copied[key],locator[key])
            self.assertFalse(copied[key].flags.writeable)
            self.assertFalse(np.shares_memory(copied[key],locator[key]))
            self.assertTrue(locator[key].flags.writeable)

    def test_different_polygons_sharing_a_box_receive_the_complete_list(self):
        context=material(level=2);p=triangle([[-.005,-.005],[.005,-.005],[0,.005]])
        q=np.nextafter(p,np.inf)
        self.assertFalse(np.array_equal(p,q))
        first=geometry._candidates(p,context)
        before=len(context['_candidate_boxes'])
        np.testing.assert_array_equal(geometry._candidates(q,context),clipping._candidates(q,context))
        self.assertEqual(len(context['_candidate_boxes']),before)
        np.testing.assert_array_equal(first,clipping._candidates(q,context))
        index=exact._Index(context['triangles'])
        first=index.candidates(p);before=len(index.candidate_boxes)
        self.assertGreater(before,0)
        second=index.candidates(q)
        self.assertEqual(len(index.candidate_boxes),before)
        np.testing.assert_array_equal(first,second)

    def test_memo_eviction_repeats_complete_queries_without_truncating(self):
        context=material(level=2);p=triangle([[-.005,-.005],[.005,-.005],[0,.005]])
        geometry._candidates(p,context)
        context['_candidate_boxes']={('other',i):np.empty(0,int) for i in range(4096)}
        expected=clipping._candidates(p,context)
        np.testing.assert_array_equal(geometry._candidates(p,context),expected)
        self.assertEqual(len(context['_candidate_boxes']),4096)
        self.assertFalse(('other',0) in context['_candidate_boxes'])
        index=exact._Index(context['triangles']);expected=index.candidates(p).copy()
        index.candidate_boxes={('other',i):np.empty(0,int) for i in range(4096)}
        np.testing.assert_array_equal(index.candidates(p),expected)
        self.assertEqual(len(index.candidate_boxes),4096)
        self.assertFalse(('other',0) in index.candidate_boxes)

    def test_partition_preparations_reuse_without_changing_pieces(self):
        context=material(level=2);queries=probes(6)
        with patch.object(convex_partition,'prepare',wraps=convex_partition.prepare) as call:
            first=[geometry._remaining(p,[context]) for p in queries]
            warmed=call.call_count
            second=[geometry._remaining(p,[context]) for p in queries]
            self.assertGreater(warmed,0);self.assertEqual(warmed,call.call_count)
        expected=[reference_remaining(p,[material(level=2)]) for p in queries]
        self.assert_nested_equal(first,expected);self.assert_nested_equal(second,expected)

    def test_memoized_candidates_keep_active_and_excluded_masks_live(self):
        p=triangle([[-.1,-.1],[.1,-.1],[0,.1]])
        context=geometry._context(p,FACE,6371.)
        self.assertEqual(geometry._remaining(p,[context]),[])
        context['active']=np.array([False])
        self.assert_nested_equal(geometry._remaining(p,[context]),[p])
        context['active'][:]=True;context['exclude']=np.array([0])
        self.assert_nested_equal(geometry._remaining(p,[context]),[p])
        context['exclude']=np.empty(0,int)
        self.assertEqual(geometry._remaining(p,[context]),[])

    def test_context_replacement_and_commit_never_reuse_old_occupancy(self):
        p=triangle([[-.1,-.1],[.1,-.1],[0,.1]])
        distant=triangle([[.5,.5],[.6,.5],[.5,.6]])
        context=world(np.empty((0,3)),np.empty((0,3),int))
        self.assert_nested_equal(geometry._remaining(p,[context['material']]),[p])
        surface=dict(vertices=p,faces=FACE)
        geometry.commit(context,surface,np.array([0]))
        added=context['updates'][0]['context']
        self.assertEqual(geometry._remaining(p,[added]),[])
        # Replacing the same component retires its whole warmed context.
        geometry.commit(context,dict(vertices=distant,faces=FACE),np.array([0]))
        self.assertEqual(len(context['updates']),1)
        self.assertIsNot(added,context['updates'][0]['context'])
        self.assert_nested_equal(geometry._remaining(p,[context['updates'][0]['context']]),[p])
        # Explicit locator replacement also retires any lookup memo.
        lookup=geometry._context(p,FACE,6371.)
        geometry._candidates(p,lookup)
        old_memo=lookup['_candidate_boxes']
        lookup['locator']=geometry._lookup_locator(lookup['locator'])
        geometry._candidates(p,lookup)
        self.assertIsNot(old_memo,lookup['_candidate_boxes'])

    def test_exact_normals_and_first_witness_match_original_face_loop(self):
        for row in FIXTURE['pairs']:
            blocker,candidate=np.asarray(row['original_triangles'])
            contexts=[geometry._context(np.empty((0,3)),np.empty((0,3),int),6371.),
                      geometry._context(blocker,FACE,6371.)]
            index=exact._index(contexts[1]);basis=exact._rays(candidate)
            self.assertEqual(index.planes(0,basis),exact._planes(index.exact(0),basis))
            self.assertEqual(exact.obstruction(candidate[None],contexts),
                             reference_obstruction(candidate[None],contexts))
            contexts[1]['active']=np.array([False])
            self.assertIsNone(exact.obstruction(candidate[None],contexts))
            contexts[1]['active'][:]=True
            self.assertEqual(exact.obstruction(candidate[None],contexts),
                             reference_obstruction(candidate[None],contexts))

    def test_certified_candidate_box_is_shared_across_all_exact_indices(self):
        p=triangle([[-.1,-.1],[.1,-.1],[0,.1]])
        contexts=[geometry._context(p,FACE,6371.) for _ in range(3)]
        old=geometry._context(p,FACE,6371.)
        for c in contexts+[old]:exact._index(c)
        with patch.object(exact,'_box',wraps=exact._box) as call:
            self.assertIsNone(exact.obstruction(p[None],contexts,old))
            self.assertEqual(call.call_count,1)
        self.assertEqual(exact.obstruction(p[None],contexts,old),reference_obstruction(p[None],contexts,old))

    def test_exact_bin_memo_keeps_global_faces_and_geometry_tamper_guard(self):
        p=triangle([[.01,.01],[.02,.01],[.01,.02]])
        context=geometry._context(np.eye(3),FACE,6371.);index=exact._index(context)
        with patch.object(exact,'_keys',wraps=exact._keys) as call:
            first=index.candidates(p);before=call.call_count
            np.testing.assert_array_equal(index.candidates(p.copy()),first)
            self.assertEqual(call.call_count,before)
        np.testing.assert_array_equal(first,[0])
        context['triangles'][0,0,0]=np.nextafter(context['triangles'][0,0,0],np.inf)
        with self.assertRaisesRegex(ValueError,'context geometry changed'):
            exact.obstruction(p[None],[context])

    def test_old_footprint_cache_matches_cold_inspection_and_refreshes_changes(self):
        old=triangle([[-.01,-.01],[.01,-.01],[0,.01]])
        grown=triangle([[-.015,-.015],[.015,-.015],[0,.015]])
        context=world(np.empty((0,3)),np.empty((0,3),int));cache={}
        real=geometry._context
        with patch.object(geometry,'_context',wraps=real) as call:
            first=geometry.inspect(context,grown,FACE,0,old_vertices=old,old_faces=FACE,_old_cache=cache)
            second=geometry.inspect(context,grown,FACE,0,old_vertices=old.copy(),old_faces=FACE,_old_cache=cache)
            self.assertEqual(call.call_count,1)
        self.assert_nested_equal(first,second)
        self.assert_nested_equal(first,geometry.inspect(context,grown,FACE,0,old_vertices=old,old_faces=FACE))
        changed=old.copy();changed[:,1]*=.9;changed/=np.linalg.norm(changed,axis=1)[:,None]
        revised=geometry.inspect(context,grown,FACE,0,old_vertices=changed,old_faces=FACE,_old_cache=cache)
        self.assert_nested_equal(revised,geometry.inspect(context,grown,FACE,0,old_vertices=changed,old_faces=FACE))
        context['radius_km']=7000.
        revised=geometry.inspect(context,grown,FACE,0,old_vertices=changed,old_faces=FACE,_old_cache=cache)
        self.assert_nested_equal(revised,geometry.inspect(context,grown,FACE,0,old_vertices=changed,old_faces=FACE))

    def test_cached_admission_preserves_full_plan_search_and_diagnostics(self):
        old=triangle([[-.01,-.01],[.01,-.01],[0,.01]])
        blocker=triangle([[-.005,.018],[.005,.018],[0,.028]])
        def factory(amount):
            span=.01+amount*.003
            return dict(vertices=triangle([[-span,-span],[span,-span],[0,span]]),faces=FACE)
        options=dict(old_vertices=old,old_faces=FACE)
        cached=geometry.admit(world(blocker),factory,8.,0,**options)
        real=geometry.inspect
        def cold(*args,**kwargs):
            kwargs.pop('_old_cache',None)
            return real(*args,**kwargs)
        with patch.object(geometry,'inspect',side_effect=cold),patch.object(geometry,'_candidates',clipping._candidates):
            expected=geometry.admit(world(blocker),factory,8.,0,**options)
        self.assertGreater(cached['diagnostics']['geometry_evaluations'],2)
        self.assertGreater(cached['accepted_area_km2'],0.)
        self.assert_nested_equal(cached,expected)

    def test_factory_none_does_not_prepare_invalid_old_geometry(self):
        context=world(np.empty((0,3)),np.empty((0,3),int))
        with patch.object(geometry,'_context',side_effect=AssertionError('unexpected preparation')):
            result=geometry.admit(context,lambda amount:None,1.,0,old_vertices=np.array([np.nan]))
        self.assertEqual(result['accepted_area_km2'],0.)
        self.assertEqual(result['diagnostics']['requested_footprint']['reason'],'unrepresentable_shared_mesh')


if __name__=='__main__':unittest.main()
