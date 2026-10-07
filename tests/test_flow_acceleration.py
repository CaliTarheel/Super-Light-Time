"""Exact flow/cut/residual oracles and frozen backend identity."""
from copy import deepcopy
import itertools
import unittest
from unittest.mock import patch
import numpy as np
import flow_acceleration as acceleration
import plate_limit_analysis as pla

class NativeFlowTests(unittest.TestCase):
    def test_highland50_preserves_initial_design_and_physics_metadata(self):
        import initial_worlds
        config=dict(width=192,height=96,seed=37)
        baseline=initial_worlds.make_highland65(config)
        half=initial_worlds.make_highland65(config,continental_fraction=.5)
        self.assertLess(abs(half['continental_fraction']-.5),.001)
        self.assertEqual((half['continent_count'],half['ocean_component_count'],half['craton_count']),(6,1,18))
        for key in ('initial_plate_topology','initial_subduction','continental_lifecycle','rift_traction'):
            self.assertEqual(half[key],baseline[key])
        self.assertEqual(half['preset'],'highland50')

    def test_ocean_rupture_transaction_is_exact_against_reference(self):
        import native_engine,checkpoint
        from tests.test_force_rifting import WORLD,FAST
        config=dict(WORLD,force_limit_rifting=FAST)
        with acceleration.reference_backend():
            reference=native_engine.Simulation(config);reference.step(.5)
        actual=native_engine.Simulation(config);actual.step(.5)
        first={};second={}
        def canonical(x):
            if isinstance(x,list):return [canonical(v) for v in x]
            if isinstance(x,dict):
                x={k:canonical(v) for k,v in x.items()}
                if x.get('t')=='set':x['v'].sort(key=lambda v:repr(v))
            return x
        self.assertEqual(canonical(checkpoint._encode(vars(actual),first)),
                         canonical(checkpoint._encode(vars(reference),second)))
        self.assertEqual(first.keys(),second.keys())
        for key in first:np.testing.assert_array_equal(first[key],second[key],err_msg=key)

    def test_ordered_native_matches_reference_capacities_and_tied_cuts(self):
        if acceleration.provenance()['name'] != 'native_ordered_dinic':
            self.skipTest('No local native backend build')
        rng=np.random.default_rng(41)
        for trial in range(300):
            n=int(rng.integers(2,35));flow=pla._Flow(n+2)
            scales=rng.choice([0.,1e-12,1.,1e12],size=n)
            weight=rng.choice([-1.,0.,1.],size=n)*scales
            for i,w in enumerate(weight):
                if w>0:flow.add(n,i,w)
                elif w<0:flow.add(i,n+1,-w)
            for _ in range(n*4):
                a,b=rng.integers(0,n,2)
                if a!=b:
                    capacity=float(rng.choice([0.,1e-14,1.,2.,1e12,1e30]))
                    flow.add(int(a),int(b),capacity,float(rng.choice([0.,1.,capacity])))
            reference=deepcopy(flow);tolerance=1e-13*(float(abs(weight).sum()) or 1.)
            expected=reference._run_python(n,n+1,tolerance)
            actual=flow.run(n,n+1,tolerance)
            with self.subTest(trial=trial):
                self.assertEqual(actual[0],expected[0])
                np.testing.assert_array_equal(actual[1],expected[1])
                np.testing.assert_array_equal(flow.cap,reference.cap)
                # A repeat call exercises already-saturated residual networks.
                again=flow.run(n,n+1,tolerance)
                oracle=reference._run_python(n,n+1,tolerance)
                self.assertEqual(again[0],oracle[0])
                np.testing.assert_array_equal(again[1],oracle[1])
                np.testing.assert_array_equal(flow.cap,reference.cap)

    @staticmethod
    def _piece_graphs(rng, count):
        for trial in range(count):
            n=int(rng.integers(2,40));e=int(rng.integers(0,4*n))
            a,b=rng.integers(0,n,(2,e));keep=a!=b;a,b=a[keep],b[keep]
            if trial%3==0 and len(a):a,b=np.r_[a,a[:4],b[:4]],np.r_[b,b[:4],a[:4]]
            weight=rng.choice([-1.,0.,1.],size=n)*rng.choice([1e-12,1.,1e12,1e26],size=n)
            if trial%7==0:weight[:]=0.
            forward=rng.choice([0.,1e-14,1.,1e12,1e30],size=len(a))*rng.random(len(a))
            backward=rng.choice([0.,1.,1e30],size=len(a))*rng.random(len(a))
            yield weight,a,b,forward,backward

    def test_direct_flow_arrays_reproduce_flow_graph_layout(self):
        rng=np.random.default_rng(67)
        for weight,a,b,forward,backward in self._piece_graphs(rng,200):
            n=len(weight);flow=pla._Flow(n+2)
            for i,w in enumerate(weight):
                if w>0.:flow.add(n,i,w)
                elif w<0.:flow.add(i,n+1,-w)
            for i,j,f,r in zip(a,b,forward,backward):flow.add(int(i),int(j),f,r)
            offset,adjacency,to,cap=pla._flow_arrays(weight,a,b,forward,backward)
            np.testing.assert_array_equal(offset,np.r_[0,np.cumsum([len(h) for h in flow.head])])
            np.testing.assert_array_equal(adjacency,[e for h in flow.head for e in h])
            np.testing.assert_array_equal(to,flow.to)
            self.assertEqual(cap.tobytes(),np.array(flow.cap,float).tobytes())

    def test_direct_native_best_piece_is_bitwise_reference(self):
        if acceleration.provenance()['name'] != 'native_ordered_dinic':
            self.skipTest('No local native backend build')
        rng=np.random.default_rng(71)
        for weight,a,b,forward,backward in self._piece_graphs(rng,300):
            actual=pla._best_piece(weight,a,b,forward,backward)
            with acceleration.reference_backend():expected=pla._best_piece(weight,a,b,forward,backward)
            self.assertEqual(np.float64(actual[0]).tobytes(),np.float64(expected[0]).tobytes())
            np.testing.assert_array_equal(actual[1],expected[1])
        with self.assertRaises(ValueError):
            pla._best_piece(np.array([1.,-1.]),np.array([0]),np.array([1]),np.array([np.nan]),np.array([0.]))

    def test_small_graph_objective_and_cut_match_independent_enumeration(self):
        rng=np.random.default_rng(53)
        for _ in range(60):
            n=6;a,b=rng.integers(0,n,(2,14));keep=a!=b;a,b=a[keep],b[keep]
            weight=rng.normal(size=n);forward=rng.random(len(a));backward=rng.random(len(a))
            def value(piece):
                return weight[piece].sum()-forward[piece[a]&~piece[b]].sum()-backward[piece[b]&~piece[a]].sum()
            best=max(value(np.array(bits,bool)) for bits in itertools.product((0,1),repeat=n))
            actual,piece=pla._best_piece(weight,a,b,forward,backward)
            self.assertAlmostEqual(actual,best,places=10)
            self.assertAlmostEqual(value(piece),best,places=10)
            with acceleration.reference_backend():reference=pla._best_piece(weight,a,b,forward,backward)
            self.assertEqual(actual,reference[0]);np.testing.assert_array_equal(piece,reference[1])

    def test_pure_stage_matches_reference_without_mutating_inputs(self):
        from tests.test_plate_limit_analysis import hemisphere
        points,torque,edges,mid,length,strength,*_=hemisphere(2,5e12,4e12)
        args=(points,torque,edges,mid,length,(strength,3*strength,strength/np.sqrt(3)))
        blocked=np.zeros(len(edges),bool);glue=[]
        before=deepcopy((args,blocked,glue))
        actual=pla.search_prepared(args,blocked,glue)
        with acceleration.reference_backend():reference=pla.search_prepared(args,blocked,glue)
        for key in actual:np.testing.assert_array_equal(actual[key],reference[key],err_msg=key)
        for x,y in zip(args,before[0]):
            if isinstance(x,tuple):
                for first,second in zip(x,y):np.testing.assert_array_equal(first,second)
            else:np.testing.assert_array_equal(x,y)
        np.testing.assert_array_equal(blocked,before[1]);self.assertEqual(glue,before[2])

    def test_reference_choice_is_local_and_frozen_binary_is_captured(self):
        native=acceleration.provenance()
        with acceleration.reference_backend():self.assertEqual(acceleration.provenance()['name'],'python_ordered_dinic')
        self.assertEqual(acceleration.provenance(),native)
        import server
        sources=server.capture_auxiliary_sources(server.ENGINE_SOURCE)
        self.assertIn('flow_acceleration.py',sources)
        if acceleration.build is not None:self.assertIn('_flow_native_build.py',sources)
        with patch.object(acceleration,'build',None):
            self.assertIsNone(acceleration.run(2,[[],[]],[],[],0,1,0.))
        self.assertEqual(acceleration.provenance(),native)

    def test_unresolved_search_returns_empty_piece_at_input_resolution(self):
        from tests.test_plate_limit_analysis import hemisphere
        points,torque,edges,mid,length,strength,*_=hemisphere(1,0.,4e12)
        args=(points,torque,edges,mid,length,(strength,3*strength,strength/np.sqrt(3)))
        actual=pla.search_prepared(args,np.ones(len(edges),bool),[])
        self.assertIsNone(actual['axis'])
        self.assertEqual(actual['ratio'],0.)
        np.testing.assert_array_equal(actual['piece'],np.zeros(len(points),bool))
        np.testing.assert_array_equal(actual['cut'],np.zeros(len(edges),bool))

    def test_invalid_native_graph_rejects_before_entering_c(self):
        if acceleration.provenance()['name'] != 'native_ordered_dinic':
            self.skipTest('No local native backend build')
        for head,to,cap in (([[0],[1]],[1,0],[float('nan'),0.]),
                            ([[0],[1]],[2,0],[1.,0.]),
                            ([[2],[1]],[1,0],[1.,0.])):
            with self.assertRaises(ValueError):acceleration.run(2,head,to,cap,0,1,0.)
        with self.assertRaises(ValueError):acceleration.run(2,[[],[]],[],[],0,0,0.)

    def test_backend_compatibility_rejects_changed_binary_identity(self):
        import checkpoint
        expected=dict(engine_sha256='engine',auxiliary_sources_sha256={},numpy_version=np.__version__,
                      flow_backend=acceleration.provenance())
        checkpoint.verify_compatibility(expected,deepcopy(expected))
        mutated=deepcopy(expected);mutated['flow_backend']={'name':'other'}
        with self.assertRaisesRegex(ValueError,'max-flow backend'):
            checkpoint.verify_compatibility(mutated,expected)

if __name__=='__main__':unittest.main()
