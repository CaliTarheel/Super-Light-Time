"""Indexed local factors with compensated, uncollapsed hi/lo Euler frames.

Frames store global Q columns. Forward: F [Q.T a, z]; transpose: [Q w, r].
Q is never precomposed into a stored global factor. Compensated dot products
use error-free Split/TwoProduct/TwoSum of Ogita, Rump and Oishi (2004),
Algorithms4--6. Transpose factor/frame products and indexed reduction retain
two words until the final global rounding; this is not an interval certificate.
"""
from dataclasses import dataclass
from functools import wraps
from fractions import Fraction
import numpy as np

_EPS = np.finfo(float).eps
_SPLITTER = float(2**27+1)
_OPERAND_MIN = float(2.**-968)
_OPERAND_MAX = float(2.**968)


def _checked(function):
    @wraps(function)
    def call(*args, **kwargs):
        try:
            with np.errstate(over='raise',under='raise',invalid='raise',divide='raise'):
                return function(*args,**kwargs)
        except (FloatingPointError,OverflowError) as error:
            raise ValueError('Framed map is outside its resolved compensated arithmetic range.') from error
    return call


def _real(value,name):
    value=np.asarray(value)
    if value.dtype.kind not in 'fiu' or (value.dtype.kind=='f' and value.dtype.itemsize>8):
        raise ValueError(f'{name} requires real binary64-compatible values.')
    result=np.asarray(value,float)
    if not np.isfinite(result).all(): raise ValueError(f'{name} must be finite.')
    return result


def _vector(value,count):
    result=_real(value,'Vector')
    if result.shape!=(count,): raise ValueError('Vector does not align with framed map.')
    return result


def _two_sum(a,b):
    value=a+b
    z=value-a
    error=(a-(value-z))+(b-z)
    return value,error


def _two_product(a,b):
    a,b=np.broadcast_arrays(a,b)
    if not np.isfinite(a).all() or not np.isfinite(b).all():
        raise ValueError('Compensated product operands must be finite.')
    active=(a!=0.)&(b!=0.)
    aa,bb=abs(a),abs(b)
    exponents=np.frexp(a)[1]+np.frexp(b)[1]
    exceptional=active&((aa<_OPERAND_MIN)|(bb<_OPERAND_MIN)|(aa>_OPERAND_MAX)|(bb>_OPERAND_MAX)
                       |(exponents < -900)|(exponents > 900))
    original_a,original_b=a,b
    # Zero products are exact and must not split an otherwise extreme operand.
    # Exceptional lanes are evaluated separately, never through unsafe Split.
    fast=active&~exceptional
    a=np.where(fast,a,0.); b=np.where(fast,b,0.)
    value=a*b
    c=_SPLITTER*a; ah=c-(c-a); al=a-ah
    c=_SPLITTER*b; bh=c-(c-b); bl=b-bh
    error=al*bl-(((value-ah*bh)-al*bh)-ah*bl)
    if np.any(exceptional):
        # This is an exact representability test, not a relaxed exponent bound.
        # Two floats must recover the original binary64 operands' product with
        # no lost residual. Genuine underflow/overflow still fails closed.
        value=np.asarray(value); error=np.asarray(error)
        for index in np.flatnonzero(exceptional):
            exact=Fraction(float(original_a.flat[index]))*Fraction(float(original_b.flat[index]))
            try:
                if abs(exact)>Fraction(float(np.finfo(float).max)): raise OverflowError
                hi=float(exact)
                if not np.isfinite(hi): raise OverflowError
                lo=float(exact-Fraction(hi))
            except OverflowError as problem:
                raise ValueError('Compensated product exponent exceeds exact two-binary64 representation.') from problem
            if not np.isfinite(lo) or Fraction(hi)+Fraction(lo)!=exact:
                raise ValueError('Compensated product exponent has a residual outside exact two-binary64 representation.')
            value.flat[index]=hi; error.flat[index]=lo
    return value,error


def _dot(a,b):
    """Dot1 on the final dimension, broadcasting all other dimensions."""
    a,b=np.broadcast_arrays(a,b)
    if not a.shape[-1]: return np.zeros(a.shape[:-1])
    total,correction=_two_product(a[...,0],b[...,0])
    for index in range(1,a.shape[-1]):
        product,error=_two_product(a[...,index],b[...,index])
        total,roundoff=_two_sum(total,product)
        correction=correction+(roundoff+error)
    result=total+correction
    if not np.isfinite(result).all(): raise ValueError('Compensated dot product is nonfinite.')
    return result


def _add_pairs(ah,al,bh,bl):
    """Add two normalized two-word expansions without an intermediate cast."""
    sh,sl=_two_sum(ah,bh)
    th,tl=_two_sum(al,bl)
    # All leading-sum roundoff remains represented through normalization.
    sh,sl=_two_sum(sh,sl+th)
    return _two_sum(sh,sl+tl)


def _dot_pair(a,b):
    """Two-word dot; unlike _dot, preserve the low word for later operators."""
    a,b=np.broadcast_arrays(a,b)
    hi=np.zeros(a.shape[:-1]); lo=hi.copy()
    for index in range(a.shape[-1]):
        ph,pl=_two_product(a[...,index],b[...,index])
        hi,lo=_add_pairs(hi,lo,ph,pl)
    return hi,lo


def _frame_pair(hi,lo,value_hi,value_lo):
    # Q times a two-word vector: all four products are required. Dropping the
    # local low word or rounding each child before scatter loses weak modes.
    coefficients=np.concatenate((hi,hi,lo,lo),axis=-1)
    vectors=np.concatenate((value_hi,value_lo,value_hi,value_lo),axis=-1)
    return _dot_pair(coefficients,vectors[...,None,:])


def _reduction_plan(indices):
    """O(entries) immutable indexed tree, no max-degree-by-index padding."""
    flat=indices.ravel()
    order=np.argsort(flat,kind='stable')
    current=flat[order]
    levels=[]
    while len(current):
        starts=np.flatnonzero(np.r_[True,current[1:]!=current[:-1]])
        sizes=np.diff(np.r_[starts,len(current)])
        counts=np.repeat(sizes,sizes)
        ranks=np.arange(len(current))-np.repeat(starts,sizes)
        done=np.flatnonzero(counts==1)
        # Finalized singleton groups leave the tree immediately. Retaining
        # them at every level would cost O(unique_indices*log(max_degree)).
        left=np.flatnonzero((ranks%2==0)&(counts>1))
        paired=(left+1<len(current))
        paired[paired]=current[left[paired]]==current[left[paired]+1]
        destinations=np.flatnonzero(paired)
        right=left[paired]+1
        levels.append((done,current[done],left,destinations,right))
        current=current[left]
    for array in (order,*(a for level in levels for a in level)):
        array.setflags(write=False)
    return order,tuple(levels)


def _scatter_pairs(plan,hi,lo,count,*,contribution_indices=None):
    order,levels=plan
    hi,lo=hi.ravel()[order],lo.ravel()[order]
    result=np.zeros(count)
    residual=np.zeros(count) if contribution_indices is not None else None
    for done,indices,left,destinations,right in levels:
        if contribution_indices is None:
            result[indices]=hi[done]+lo[done]
        else:
            destinations_in_result=np.searchsorted(contribution_indices,indices)
            result[destinations_in_result]=hi[done]
            residual[destinations_in_result]=lo[done]
        next_hi,next_lo=hi[left].copy(),lo[left].copy()
        next_hi[destinations],next_lo[destinations]=_add_pairs(
            next_hi[destinations],next_lo[destinations],hi[right],lo[right])
        hi,lo=next_hi,next_lo
    if not np.isfinite(result).all(): raise ValueError('Framed indexed accumulation is nonfinite.')
    if residual is not None:
        if not np.isfinite(residual).all(): raise ValueError('Framed indexed expansion is nonfinite.')
        return result,residual
    return result


def _frame_values(hi,lo):
    hi,lo=_real(hi,'Frame hi'),_real(lo,'Frame lo')
    if hi.shape!=lo.shape or hi.shape[-2:]!=(3,3):
        raise ValueError('Frames require aligned (...,3,3) hi/lo arrays.')
    q=hi+lo
    defect=np.swapaxes(q,-1,-2)@q-np.eye(3)
    if np.any(abs(defect)>2e-12) or np.any(np.linalg.det(q)<=0.):
        raise ValueError('Framed maps require proper orthonormal frames within binary64 input tolerance.')
    return hi,lo


def _frame_product(hi,lo,value,transpose):
    # For Q.T, output j sums Q[i,j]*v[i]; for Q, output i sums Q[i,j]*w[j].
    if transpose: hi,lo=np.swapaxes(hi,-1,-2),np.swapaxes(lo,-1,-2)
    terms=np.concatenate((hi,lo),axis=-1)
    vectors=np.concatenate((value,value),axis=-1)[...,None,:]
    return _dot(terms,vectors)


@_checked
def project_frames(frame_hi,frame_lo,global_vectors):
    """Return (frame_hi+frame_lo).T @ vectors without rounding Q first.

    Leading dimensions broadcast; frames' final axes are3x3, vectors' final
    axis is3. A vector is the actually represented input, not an inferred
    exact rotated or normalized vector. Windows extended precision is unused.
    """
    hi,lo=_frame_values(frame_hi,frame_lo)
    value=_real(global_vectors,'Global vectors')
    if value.shape[-1:]!=(3,): raise ValueError('Frame projection requires three-vectors.')
    return _frame_product(hi,lo,value,True)


def _up(value):
    result=np.array(value,copy=True)
    take=result>0.
    result[take]=np.nextafter(result[take],np.inf)
    if not np.isfinite(result).all(): raise ValueError('Absolute bound exceeds finite arithmetic range.')
    return result


def _positive_dot(a,b):
    a,b=np.broadcast_arrays(a,b)
    result=np.zeros(a.shape[:-1])
    for index in range(a.shape[-1]):
        result=_up(result+_up(a[...,index]*b[...,index]))
    return result


def _scatter(indices,values,count,*,upper=False):
    result=np.bincount(indices.ravel(),weights=values.ravel(),minlength=count)
    if upper:
        # All terms are nonnegative. Cover ordinary indexed summation's
        # per-index gamma_n error; inflation itself rounds outward by one ULP.
        terms=np.bincount(indices.ravel(),minlength=count)
        fraction=terms*_EPS
        if np.any(fraction>=.25): raise ValueError('Indexed bound has too many summed terms.')
        result=_up(result/(1.-fraction))
    if not np.isfinite(result).all(): raise ValueError('Framed indexed accumulation is nonfinite.')
    return result


@dataclass(frozen=True)
class FramedBlockMap:
    blocks: np.ndarray
    rows: np.ndarray
    columns: np.ndarray
    shape: tuple
    frame_hi: np.ndarray
    frame_lo: np.ndarray

    @_checked
    def __post_init__(self):
        b=_real(self.blocks,'Blocks')
        r,c=np.asarray(self.rows),np.asarray(self.columns)
        shape=self.shape
        if (not isinstance(shape,tuple) or len(shape)!=2
                or any(isinstance(n,(bool,np.bool_)) or not isinstance(n,(int,np.integer))
                       or n<0 or n>np.iinfo(np.int64).max for n in shape)
                or b.ndim!=3 or b.shape[1:]!=(9,9) or r.shape!=(len(b),9) or c.shape!=(len(b),9)
                or r.dtype.kind not in 'iu' or c.dtype.kind not in 'iu'
                or np.any(r<0) or np.any(c<0) or np.any(r>=shape[0]) or np.any(c>=shape[1])):
            raise ValueError('Finite9x9 framed blocks and indexed shape must align.')
        hi,lo=_frame_values(self.frame_hi,self.frame_lo)
        if hi.shape!=(len(b),3,3): raise ValueError('One hi/lo frame per local block is required.')
        for name,value in [('blocks',b),('rows',r.astype(np.int64)),('columns',c.astype(np.int64)),
                           ('frame_hi',hi),('frame_lo',lo)]:
            value=value.copy(); value.setflags(write=False); object.__setattr__(self,name,value)
        object.__setattr__(self,'_disjoint_rows',len(np.unique(self.rows))==self.rows.size)
        object.__setattr__(self,'_transpose_plan',_reduction_plan(self.columns))

    @_checked
    def __matmul__(self,value):
        local=_vector(value,self.shape[1])[self.columns].copy()
        local[:,:3]=_frame_product(self.frame_hi,self.frame_lo,local[:,:3],True)
        output=_dot(self.blocks,local[:,None,:])
        return _scatter(self.rows,output,self.shape[0])

    @_checked
    def rmatvec(self,value):
        local=_vector(value,self.shape[0])[self.rows]
        hi,lo=_dot_pair(np.swapaxes(self.blocks,1,2),local[:,None,:])
        hi[:,:3],lo[:,:3]=_frame_pair(self.frame_hi,self.frame_lo,hi[:,:3],lo[:,:3])
        return _scatter_pairs(self._transpose_plan,hi,lo,self.shape[1])

    @_checked
    def rmatvec_contributions(self,value):
        """Unique global indices and hi/lo forces for compensated outer sums.

        This preserves the low word beyond each map, without allocating full
        global vectors. Callers must combine both words before final rounding.
        """
        local=_vector(value,self.shape[0])[self.rows]
        hi,lo=_dot_pair(np.swapaxes(self.blocks,1,2),local[:,None,:])
        hi[:,:3],lo[:,:3]=_frame_pair(self.frame_hi,self.frame_lo,hi[:,:3],lo[:,:3])
        indices=np.unique(self.columns)
        hi,lo=_scatter_pairs(self._transpose_plan,hi,lo,len(indices),contribution_indices=indices)
        return indices,hi,lo

    @property
    def storage_bytes(self):
        order,levels=self._transpose_plan
        return (sum(getattr(self,key).nbytes for key in ('blocks','rows','columns','frame_hi','frame_lo'))
                +order.nbytes+sum(a.nbytes for level in levels for a in level))

    @_checked
    def absolute_forward(self,value):
        local=abs(_vector(value,self.shape[1])[self.columns])
        frame=_up(abs(self.frame_hi)+abs(self.frame_lo))
        local[:,:3]=_positive_dot(np.swapaxes(frame,1,2),local[:,:3,None].swapaxes(1,2))
        output=_positive_dot(abs(self.blocks),local[:,None,:])
        return _scatter(self.rows,output,self.shape[0],upper=True)

    @_checked
    def absolute_action(self,value):
        local=self.absolute_forward(value)[self.rows]
        output=_positive_dot(abs(np.swapaxes(self.blocks,1,2)),local[:,None,:])
        frame=_up(abs(self.frame_hi)+abs(self.frame_lo))
        output[:,:3]=_positive_dot(frame,output[:,None,:3])
        return _scatter(self.columns,output,self.shape[1],upper=True)

    def _basis_blocks(self):
        """Temporary coordinate columns for diagonal diagnostics, never action."""
        result=self.blocks.copy()
        for j in range(3):
            vector=np.concatenate((self.frame_hi[:,j,:],self.frame_lo[:,j,:]),axis=1)
            factor=np.concatenate((self.blocks[:,:,:3],self.blocks[:,:,:3]),axis=2)
            result[:,:,j]=_dot(factor,vector[:,None,:])
        return result

    @_checked
    def diagonal(self):
        coefficients=self._basis_blocks()
        if self._disjoint_rows:
            # Normal finite-basal path: no global81-entry-per-piece grouping.
            # Coalesce the at-most9 local repeated columns before each square.
            result=np.zeros(self.shape[1])
            for j in range(9):
                first=~np.any(self.columns[:,:j]==self.columns[:,j,None],axis=1)
                mask=(self.columns==self.columns[:,j,None]).astype(float)
                column=_dot(coefficients,mask[:,None,:])
                weight=np.sum(column*column,axis=1)*first
                result+=_scatter(self.columns[:,j],weight,self.shape[1])
            if not np.isfinite(result).all(): raise ValueError('Framed diagonal is nonfinite.')
            return result
        # Repeated local columns or output rows require coalescing the SAME
        # matrix entry before squaring, not merely summing individual squares.
        rows=np.broadcast_to(self.rows[:,:,None],coefficients.shape).ravel()
        columns=np.broadcast_to(self.columns[:,None,:],coefficients.shape).ravel()
        keys=np.column_stack((rows,columns))
        unique,inverse=np.unique(keys,axis=0,return_inverse=True)
        values=np.bincount(inverse,weights=coefficients.ravel())
        return _scatter(unique[:,1],values*values,self.shape[1])

    def to_dense(self):
        """Small audit-only basis actions; never used to apply a physical map."""
        if self.shape[1]>256 or self.shape[0]*self.shape[1]>2_000_000:
            raise ValueError('Dense conversion is limited to small parity audits.')
        if self.shape[1]==0: return np.empty(self.shape)
        return np.column_stack([self@basis for basis in np.eye(self.shape[1])])
