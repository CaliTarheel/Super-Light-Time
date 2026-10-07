"""Independent Decimal checks of framed actions, reciprocal work and bounds."""
from decimal import Decimal as D, localcontext
from pathlib import Path
import sys
import unittest
import numpy as np

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from benchmarks.collision_architecture.framed_basal_map import FramedBlockMap, project_frames

POLYGON=np.array([[-.047658386204569435,-.8675924763644063,-.4949868414213182],
                  [-.04765837903619681,-.8675924801785472,-.4949868354262357],
                  [-.047658390989513015,-.8675924811800351,-.4949868325199785]])

# Stored factors/frames from the independent 140-digit cell36312 split audit.
# Freezing the represented values tests this map independently of the metric
# builder; the original transpose lost ~5e-9 relatively before indexed scatter.
SPLIT_BLOCKS=np.array([
 [[5.162611987204398e-9,-1.3017908707139753e-26,-7.183400087510397e-35],
  [0.,5.162611987204398e-9,-6.216951013904279e-35],[0.,0.,1.3984265024287213e-17]],
 [[5.162611987049634e-9,-1.1250948975090428e-26,-5.913932594524294e-35],
  [0.,5.162611987049634e-9,-9.440118044059491e-35],[0.,0.,1.4681874162691045e-17]]])
SPLIT_HI=np.array([
 [[0.,-.9988636934968782,-.04765838660482186],
  [-.4955499340677273,.04139509527676766,-.8675924786053061],
  [.8685794510840509,.02361711033979373,-.49498683745502453]],
 [[0.,-.9988636936108853,-.047658384215364315],
  [-.4955499320105326,.04139509325727009,-.8675924798766864],
  [.8685794522577405,.023617109057655628,-.4949868354566637]]])
SPLIT_LO=np.array([
 [[0.,5.2535430165308196e-17,3.2113956807381653e-18],
  [9.242082824394304e-18,-2.581381609044902e-18,1.3928920308799656e-17],
  [3.768995984350473e-18,-1.622824073114261e-18,9.365149371254079e-18]],
 [[0.,-4.601986451307339e-17,-3.1724952967654058e-18],
  [1.5876419700178877e-17,-2.3800350089504446e-18,4.7067783106656474e-17],
  [4.3904458449797855e-17,-4.2299769727242305e-19,1.6635364080052833e-17]]])
SPLIT_INPUT=np.array([[6.106770018118501e-18,-6.174937704064716e-18,1.3984265024287213e-17],
                     [-6.106769977778514e-18,6.174937688563969e-18,1.4681874162691045e-17]])


def dec(x): return D.from_float(float(x))
def dot(a,b): return sum((x*y for x,y in zip(a,b)),D(0))
def cross(a,b): return [a[1]*b[2]-a[2]*b[1],a[2]*b[0]-a[0]*b[2],a[0]*b[1]-a[1]*b[0]]
def unit(a):
    length=dot(a,a).sqrt(); return [x/length for x in a]


def frame():
    with localcontext() as context:
        context.prec=100
        points=[unit(list(map(dec,row))) for row in POLYGON]
        n=unit([sum(p[k] for p in points) for k in range(3)])
        t=unit(cross(n,[D(1),D(0),D(0)])); u=cross(n,t)
        q=[[t[i],u[i],n[i]] for i in range(3)]
        hi=np.array(q,float)
        lo=np.array([[float(q[i][j]-dec(hi[i,j])) for j in range(3)] for i in range(3)])
    return hi,lo


def reference_project(hi,lo,v,transpose=True):
    with localcontext() as context:
        context.prec=100
        q=[[dec(hi[i,j])+dec(lo[i,j]) for j in range(3)] for i in range(3)]
        if transpose: q=list(map(list,zip(*q)))
        return [dot(row,list(map(dec,v))) for row in q]


def map_for(blocks=None,*,count=1,rows=None,columns=None,shape=None):
    hi,lo=frame()
    blocks=np.tile(np.eye(9),(count,1,1)) if blocks is None else blocks
    rows=np.arange(count*9).reshape(count,9) if rows is None else rows
    columns=np.arange(count*9).reshape(count,9) if columns is None else columns
    shape=(count*9,count*9) if shape is None else shape
    return FramedBlockMap(blocks,rows,columns,shape,np.tile(hi,(count,1,1)),np.tile(lo,(count,1,1)))


class FramedBasalMapTests(unittest.TestCase):
    def test_saved_centroid_and_weak_obliques_against_100_digit_oracle(self):
        hi,lo=frame()
        vectors=[hi[:,2],hi[:,2]+3e-9*hi[:,0],hi[:,2]-1e-8*hi[:,1]]
        for v in vectors:
            actual=project_frames(hi,lo,v); truth=reference_project(hi,lo,v)
            for result,expected in zip(actual,truth):
                with localcontext() as context:
                    context.prec=100
                    # Dot1 promises doubled working accuracy, not arbitrary
                    # relative precision for an exactly cancelling result.
                    budget=D('4e-16')*abs(expected)+D('2e-31')
                    self.assertLessEqual(abs(dec(result)-expected),budget)
        actual=project_frames(hi,lo,vectors[1]); truth=np.array(reference_project(hi,lo,vectors[1]),float)
        naive=(hi+lo).T@vectors[1]
        self.assertLess(abs(actual[0]-truth[0]),1e-24)
        self.assertGreater(abs(naive[0]-truth[0]),1e-19)

    def test_broadcast_projection_and_decimal_scale_range(self):
        hi,lo=frame(); vectors=np.array([[.2,-.3,.5],[1e-120,-2e-120,3e-120],[1e90,-2e90,3e90]])
        result=project_frames(hi[None],lo[None],vectors)
        for v,got in zip(vectors,result):
            expected=np.array(reference_project(hi,lo,v),float)
            np.testing.assert_allclose(got,expected,rtol=5e-15,atol=0.)
        np.testing.assert_array_equal(project_frames(hi,lo,np.zeros(3)),np.zeros(3))

    def test_compensated_factor_dot_preserves_extreme_cancellation(self):
        block=np.zeros((1,9,9)); block[0,0,3:6]=[1e16,1.,-1e16]
        design=map_for(block)
        x=np.zeros(9); x[3:6]=1.
        self.assertEqual((design@x)[0],1.)
        # Transpose has the same compensated local dot arithmetic.
        block=np.zeros((1,9,9)); block[0,3:6,3]=[1e16,1.,-1e16]
        design=map_for(block); x=np.zeros(9); x[3:6]=1.
        self.assertEqual(design.rmatvec(x)[3],1.)

    def test_saved_split_centroid_transpose_keeps_child_low_words(self):
        blocks=np.zeros((2,9,9)); blocks[:,:3,:3]=SPLIT_BLOCKS
        values=np.zeros((2,9)); values[:,:3]=SPLIT_INPUT
        rows=np.arange(18).reshape(2,9); columns=np.tile(np.arange(9),(2,1))
        design=FramedBlockMap(blocks,rows,columns,(18,9),SPLIT_HI,SPLIT_LO)
        actual=design.rmatvec(values.ravel())
        with localcontext() as context:
            context.prec=140
            truth=[D(0)]*3
            for b,h,l,v in zip(SPLIT_BLOCKS,SPLIT_HI,SPLIT_LO,SPLIT_INPUT):
                local=[dot(list(map(dec,column)),list(map(dec,v))) for column in b.T]
                q=[[dec(h[i,j])+dec(l[i,j]) for j in range(3)] for i in range(3)]
                truth=[old+dot(row,local) for old,row in zip(truth,q)]
            for got,expected in zip(actual[:3],truth):
                self.assertLessEqual(abs(dec(got)-expected),D('4e-16')*abs(expected))
        # Altering piece order or using a further indexed split must preserve
        # the represented map, not just a permissive total-work estimate.
        reverse=FramedBlockMap(blocks[::-1],rows,columns,(18,9),SPLIT_HI[::-1],SPLIT_LO[::-1])
        np.testing.assert_array_equal(reverse.rmatvec(values[::-1].ravel()),actual)
        split=FramedBlockMap(np.repeat(blocks/2.,2,axis=0),np.arange(36).reshape(4,9),
            np.tile(np.arange(9),(4,1)),(36,9),np.repeat(SPLIT_HI,2,axis=0),np.repeat(SPLIT_LO,2,axis=0))
        np.testing.assert_array_equal(split.rmatvec(np.repeat(values,2,axis=0).ravel()),actual)

    def test_compensated_indexed_reduction_keeps_residual_cancellation(self):
        blocks=np.tile(np.eye(9),(3,1,1)); columns=np.tile(np.arange(9),(3,1))
        design=map_for(blocks,count=3,columns=columns,shape=(27,9))
        values=np.zeros((3,9)); values[:,3]=[1e16,1.,-1e16]
        self.assertEqual(design.rmatvec(values.ravel())[3],1.)
        order,levels=design._transpose_plan
        for array in (order,*(a for level in levels for a in level)):
            self.assertFalse(array.flags.writeable)
        # Plan storage scales with actual contributions, not (maxdegree*N).
        self.assertLess(sum(a.size for level in levels for a in level),5*columns.size)
        # Strongly uneven degrees must not keep thousands of singleton groups
        # in the tree while one highly shared coordinate is reduced.
        count=1024; columns=np.arange(count*9).reshape(count,9); columns[:,0]=0
        uneven=map_for(count=count,columns=columns)
        _,levels=uneven._transpose_plan
        self.assertLess(sum(a.size for level in levels for a in level),5*columns.size)

    def test_tiny_mode_action_power_and_global_reaction_against_decimal(self):
        hi,lo=frame(); block=np.zeros((1,9,9)); block[0,:3,:3]=np.diag([1.,1.,1e-8])
        design=map_for(block); x=np.zeros(9); x[:3]=hi[:,2]+3e-9*hi[:,0]
        forward=design@x
        with localcontext() as context:
            context.prec=100
            local=reference_project(hi,lo,x[:3]); expected=[local[0],local[1],local[2]*dec(1e-8)]
            work=sum(v*v for v in expected)
            measured=dec(float(forward@forward))
            self.assertLess(abs(measured-work)/work,D('2e-15'))
        force=design.rmatvec(forward)
        with localcontext() as context:
            context.prec=100
            reaction=reference_project(hi,lo,[forward[0],forward[1],forward[2]*1e-8],False)
            np.testing.assert_allclose(force[:3],np.array(reaction,float),rtol=3e-15,atol=0.)
            # Global virtual work uses an independent accurate sum; ordinary
            # x@force can lose digits when large tangential terms cancel.
            power=dot(list(map(dec,x[:3])),list(map(dec,force[:3])))
            self.assertLess(abs(power-work)/work,D('2e-8'))

    def test_reciprocal_indexed_transpose_diagonal_and_bounds(self):
        rng=np.random.default_rng(18); count=4
        rows=np.arange(36).reshape(4,9)%20
        columns=(np.arange(36).reshape(4,9)+3)%17
        blocks=rng.normal(size=(count,9,9))
        design=map_for(blocks,count=count,rows=rows,columns=columns,shape=(20,17))
        x=rng.normal(size=17); v=rng.normal(size=20); dense=design.to_dense()
        np.testing.assert_allclose(design@x,dense@x,rtol=1e-13,atol=1e-14)
        np.testing.assert_allclose(design.rmatvec(v),dense.T@v,rtol=1e-13,atol=1e-14)
        self.assertAlmostEqual(float(v@(design@x)),float(x@design.rmatvec(v)),places=12)
        np.testing.assert_allclose(design.diagonal(),np.sum(dense*dense,axis=0),rtol=3e-14,atol=1e-14)
        self.assertTrue(np.all(design.absolute_forward(x)>=abs(dense)@abs(x)))
        self.assertTrue(np.all(design.absolute_action(x)>=abs(dense).T@(abs(dense)@abs(x))))
        self.assertTrue(np.all(design.diagonal()>=0.))

    def test_duplicate_columns_coalesce_before_diagonal_squaring(self):
        block=np.zeros((1,9,9)); block[0,0,3]=2.; block[0,0,4]=3.
        columns=np.arange(9)[None]; columns[0,4]=3
        design=map_for(block,columns=columns)
        self.assertEqual(design.diagonal()[3],25.)

    def test_mantle_reference_uses_same_frame_and_physical_power_identity(self):
        hi,lo=frame(); block=np.zeros((1,9,9)); block[0,:3,:3]=np.diag([2.,3.,1e-8])
        design=map_for(block); mantle=hi[:,2]+2e-9*hi[:,1]
        reference=block[0,:,:3]@project_frames(hi,lo,mantle)
        x=np.zeros(9); x[:3]=mantle
        np.testing.assert_allclose(design@x,reference,rtol=2e-15,atol=1e-30)
        x[:3]+=1e-9*hi[:,0]
        slip=design@x-reference; gradient=design.rmatvec(slip)
        diss=float(slip@slip); mantle_input=-float(slip@reference)
        with localcontext() as context:
            context.prec=100
            power=dot(list(map(dec,x)),list(map(dec,gradient)))
            expected=dec(diss-mantle_input)
            self.assertLess(abs(power-expected),D('2e-24'))

    def test_detached_readonly_arrays_and_bounded_dense_conversion(self):
        blocks=np.eye(9)[None]; hi,lo=frame(); rows=np.arange(9)[None]; columns=rows.copy()
        design=FramedBlockMap(blocks,rows,columns,(9,9),hi[None],lo[None])
        blocks[:]=0.; hi[:]=0.; rows[:]=0
        self.assertGreater(design.storage_bytes,blocks.nbytes)
        for name in ('blocks','rows','columns','frame_hi','frame_lo'):
            with self.assertRaises(ValueError): getattr(design,name).flat[0]=0.
        with self.assertRaises(ValueError): map_for(shape=(9,300)).to_dense()

    def test_invalid_input_and_unsupported_arithmetic_fail_closed(self):
        hi,lo=frame()
        for bad in (np.ones((3,3)),np.diag([1.,1.,-1.]),np.full((3,3),np.nan)):
            with self.assertRaises(ValueError): project_frames(bad,np.zeros((3,3)),np.ones(3))
        with self.assertRaises(ValueError): project_frames(hi,lo,np.array([True,False,True]))
        with self.assertRaises(ValueError): project_frames(hi,lo,np.ones(3)*np.finfo(float).max)
        with self.assertRaises(ValueError): project_frames(hi,lo,np.ones(3)*1e-300)
        with self.assertRaises(ValueError): map_for(np.full((1,9,9),1e250))@np.full(9,1e250)
        with self.assertRaises(ValueError): map_for(np.full((1,9,9),1e-200)).diagonal()
        with self.assertRaises(ValueError): map_for(rows=np.full((1,9),-.5))
        with self.assertRaises(ValueError): map_for(shape=(9,True))


if __name__=='__main__': unittest.main()
