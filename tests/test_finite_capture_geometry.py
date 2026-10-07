"""Isolated physical footprint contract screens; not release evidence."""
from pathlib import Path
import copy,importlib.util,itertools,json,math,sys,unittest
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
FIXTURES=ROOT/'tests/fixtures'
import finite_capture_geometry as candidate
import native_subduction,native_spreading,mesh_coverage
from tests.test_native_subduction_oracle import fixture,strip_area
from tests._native_spreading_oracle import polygon_area

def unit(p):return np.asarray(p,float)/np.linalg.norm(p,axis=-1,keepdims=True)
def row(p,m=1.,parent=0):return dict(polygon=np.asarray(p),maturity=m,parent=parent,down=0,over=1)
def area(rows):return math.fsum(r['raw_area_km2'] for r in rows)
def actual(split=1):
    d=json.loads((FIXTURES/'finite_capture_024_motion.json').read_text(encoding='utf-8'))
    s,_,_=fixture(level=2);s.bq[:]=2;s.omega[0]=d['omega_incoming'];s.omega[2]=d['omega_overriding']
    a,b=np.array(d['source_endpoints']);u=np.linspace(0,1,split+1)[:,None];p=unit(a*(1-u)+b*u)
    s.native_boundary_geometry=dict(segments_start=p[:-1],segments_end=p[1:],segment_normals=np.tile(d['source_normal'],(split,1)),contact_index=np.zeros(split,int))
    return s

class FiniteCaptureGeometryTests(unittest.TestCase):
    def test_original_convex_strip_area_and_cell_partition(self):
        for split in (1,2,4):
            s,axis,_=fixture(split=split,level=3)
            rows=candidate.capture_polygons(s,2.)
            self.assertAlmostEqual(area(rows),strip_area(axis,2.),delta=1e-5)
            context=native_spreading.prepare_material(s.material_surface)
            total=math.fsum(float(native_spreading.integrate_polygon(s.native_mesh,r['polygon'],context)['area_km2'].sum()) for r in rows)
            self.assertAlmostEqual(total,area(rows),delta=1e-5)

    def test_concave_polygon_cyclic_root_and_order_keep_actual_area(self):
        p=unit([[1,0,0],[1,.02,0],[1,.005,.005],[1,0,.02]])
        expected=polygon_area(p[:3])+polygon_area(p[[0,2,3]])
        for shift in range(4):
            r=candidate.partition([row(np.roll(p,shift,axis=0))])
            self.assertAlmostEqual(area(r),expected,delta=1e-6)
            for item in r:
                q=item['polygon'];self.assertGreaterEqual(float((unit(np.cross(q,np.roll(q,-1,axis=0)-q))@q.T).min()),-1e-12)

    def test_actual_crossed_source_is_subdivision_invariant(self):
        values=[]
        for split in (1,2,4,8,16):
            s=actual(split);r=candidate.capture_polygons(s,2.);values.append(area(r))
            self.assertAlmostEqual(area(r),.331266933,delta=1e-6)
            for item in r:self.assertGreater(item['raw_area_km2'],0)
        self.assertLess(max(values)-min(values),1e-6)

    def test_actual_reverse_order_and_rigid_rotation_preserve_area(self):
        s=actual(4);expected=area(candidate.capture_polygons(s,2.))
        g=s.native_boundary_geometry
        a=g['segments_start'].copy();g['segments_start']=g['segments_end'][::-1];g['segments_end']=a[::-1]
        self.assertAlmostEqual(area(candidate.capture_polygons(s,2.)),expected,delta=1e-6)
        Q=np.array([[0.,0.,1.],[0.,1.,0.],[-1.,0.,0.]])
        for key in ('segments_start','segments_end','segment_normals'):g[key]=g[key]@Q.T
        s.omega=s.omega@Q.T
        self.assertAlmostEqual(area(candidate.capture_polygons(s,2.)),expected,delta=1e-6)

    def test_common_motion_and_opening_have_no_capture(self):
        s,_,_=fixture();s.omega[:]=s.omega[0]
        self.assertEqual(candidate.capture_polygons(s,2.),[])
        s,_,_=fixture();s.omega*=-1
        self.assertEqual(candidate.capture_polygons(s,2.),[])

    def test_local_maturity_is_not_broadcast_across_adjacent_regions(self):
        a=unit([[1,0,0],[1,.01,0],[1,.01,.01],[1,0,.01]])
        b=unit([[1,.01,0],[1,.02,0],[1,.02,.01],[1,.01,.01]])
        expected=polygon_area(a)+.25*polygon_area(b)
        for source in ([row(a,1,0),row(b,.25,1)],[row(b,.25,1),row(a,1,0)]):
            rows=candidate.partition(source)
            self.assertAlmostEqual(math.fsum(r['raw_area_km2']*r['maturity'] for r in rows),expected,delta=1e-6)

    def test_overlapping_positive_capture_uses_local_maximum_maturity(self):
        p=unit([[1,0,0],[1,.02,0],[1,.02,.01],[1,0,.01]])
        for source in itertools.permutations([row(p,.2,0),row(p,.8,1)]):
            r=candidate.partition(source)
            self.assertAlmostEqual(area(r),polygon_area(p),delta=1e-6)
            self.assertAlmostEqual(math.fsum(x['raw_area_km2']*x['maturity'] for x in r),.8*polygon_area(p),delta=1e-6)

    def test_opposite_winding_cancels_geometry_before_surviving_local_maturity(self):
        p=unit([[1,0,0],[1,.02,0],[1,.02,.01],[1,0,.01]])
        self.assertEqual(candidate.partition([row(p,.6,0),row(p[::-1],1.,1)]),[])
        for source in itertools.permutations([row(p,.2,0),row(p[::-1],1.,1),row(p,.7,2)]):
            r=candidate.partition(source)
            self.assertAlmostEqual(area(r),polygon_area(p),delta=1e-6)
            self.assertAlmostEqual(math.fsum(x['raw_area_km2']*x['maturity'] for x in r),.7*polygon_area(p),delta=1e-6)

    def test_nearby_disconnected_fronts_do_not_share_winding_or_maturity(self):
        a,b,c=unit([[1,0,0],[1,.01,0],[1,.02,0]])
        base=dict(parent=0,down=0,over=1,maturity=1,polygon=unit([[1,0,0],[1,.01,0],[1,.01,.01],[1,0,.01]]))
        rows=[dict(base,endpoints=np.array([a,b])),dict(base,endpoints=np.array([b,c]))]
        self.assertEqual(len(candidate._components(rows)),1)
        rows[1]['endpoints']=unit(rows[1]['endpoints']+np.array([0,0,1e-8]))
        self.assertEqual(len(candidate._components(rows)),2)

    def test_missing_or_zero_runtime_selector_preserves_legacy_outputs(self):
        spec=importlib.util.spec_from_file_location('unaltered_subduction_reference',FIXTURES/'native_subduction_before_geometry.py')
        ref=importlib.util.module_from_spec(spec);spec.loader.exec_module(ref)
        for version in (None,0):
            s,_,transport=fixture();s.t=0.
            if version is not None:s.native_subduction_geometry_version=version
            old=copy.deepcopy(s);old_rows=ref.capture_polygons(old,2.);new_rows=native_subduction.capture_polygons(s,2.)
            self.assertEqual(len(old_rows),len(new_rows))
            for a,b in zip(old_rows,new_rows):
                self.assertEqual(set(a),set(b))
                for key in a:
                    if isinstance(a[key],np.ndarray):np.testing.assert_array_equal(a[key],b[key])
                    else:self.assertEqual(a[key],b[key])
            np.testing.assert_array_equal(ref.removal(old,transport,2.),native_subduction.removal(s,transport,2.))
            self.assertEqual(old.native_subduction_diagnostics,s.native_subduction_diagnostics)

    def test_policy_schema_rejects_stripping_or_mismatching_tags(self):
        s,_,transport=fixture();s.t=0.;s.native_subduction_geometry_version=1
        initial=native_subduction.snapshot_metadata(s)
        self.assertEqual(initial['native_subduction_geometry_version'],1)
        native_subduction.removal(s,transport,2.);s.t=2.
        frame=dict(native_subduction.snapshot_metadata(s),mesh_version=1,time_myr=2.)
        native_subduction.validate_frame(frame)
        mutants=[]
        for value in (None,0,True,1.,2):
            f=copy.deepcopy(frame)
            if value is None:f.pop('native_subduction_geometry_version')
            else:f['native_subduction_geometry_version']=value
            mutants.append(f)
        for key in ('capture_geometry_version','capture_geometry_method'):
            f=copy.deepcopy(frame);f['native_subduction_diagnostics'].pop(key);mutants.append(f)
        f=copy.deepcopy(frame);f['native_subduction_diagnostics']['capture_geometry_version']=True;mutants.append(f)
        import mesh_history,native_frame_sampling
        for mutant in mutants:
            for validator in (native_subduction.validate_frame,mesh_history.arrays,native_frame_sampling.prepare):
                with self.subTest(validator=validator.__name__),self.assertRaisesRegex(ValueError,'geometry'):
                    validator(mutant)

    def test_fresh_initial_snapshot_declares_policy_through_shared_loaders(self):
        from tests.test_native_engine import world
        import mesh_history,native_frame_sampling
        s=world();frame=s.snapshot()
        self.assertEqual(s.native_subduction_geometry_version,1)
        self.assertEqual(frame['native_subduction_geometry_version'],1)
        self.assertEqual(frame['native_subduction_diagnostics']['capture_geometry_version'],1)
        mesh_history.arrays(frame);native_frame_sampling.prepare(frame)

if __name__=='__main__':unittest.main(verbosity=2)
