"""Exact-chart analytic basal metric retained in a local positive representation.

Research arithmetic component. Precision agreement is an explicit numerical
convergence guard, not a formal directed-rounding error certificate. Global
binary64 matrices are diagnostics, never the weak-axis positivity oracle.
"""
from dataclasses import dataclass, fields
from decimal import Decimal as D, localcontext
from fractions import Fraction
import hashlib
import json
import numpy as np


def _dot(a, b): return sum((x*y for x,y in zip(a,b)), D(0))
def _cross(a, b): return [a[1]*b[2]-a[2]*b[1], a[2]*b[0]-a[0]*b[2], a[0]*b[1]-a[1]*b[0]]
def _unit(a):
    n = _dot(a,a).sqrt()
    if not n: raise ValueError('Zero spherical ray is unresolved.')
    return [x/n for x in a]
def _decimal(x):
    return D(x.numerator)/D(x.denominator) if isinstance(x,Fraction) else D.from_float(float(x))
def _atan(x):
    scale = 1
    while abs(x) > D('.1'):
        x /= 1+(1+x*x).sqrt(); scale *= 2
    term = total = x; square = -x*x; denominator = 1
    while True:
        term *= square; denominator += 2
        following = total+term/denominator
        if following == total: return total*scale
        total = following
def _transpose(a): return [list(row) for row in zip(*a)]
def _multiply(a,b):
    return [[_dot(row,col) for col in zip(*b)] for row in a]
def _local_matrix(matrix, frame):
    return _multiply(_multiply(_transpose(frame),matrix),frame)


def _certified_centroid_zeros(rays):
    """Sufficient exact zero proof for the sum of normalized original rays.

    Rays with the same exact squared length have the same normalization
    denominator. A component that sums to zero separately in every such group
    is therefore zero after normalization too. A raw component sum or a rounded
    Decimal zero alone is insufficient. This proves selected zeros only; it
    does not claim to recognize every possible algebraic cancellation.
    """
    groups={}
    for ray in rays:
        squared=sum((value*value for value in ray),Fraction(0))
        if squared<=0: raise ValueError('Zero spherical ray is unresolved.')
        total=groups.setdefault(squared,[Fraction(0)]*3)
        for j in range(3): total[j]+=ray[j]
    if not groups: raise ValueError('A spherical centroid needs original rays.')
    return tuple(all(total[j]==0 for total in groups.values()) for j in range(3))


def _real(value, name):
    raw=np.asarray(value)
    if raw.dtype.kind not in 'fiu' or (raw.dtype.kind=='f' and raw.dtype.itemsize>8):
        raise ValueError(name+' requires real binary64-compatible input.')
    result=np.asarray(raw,float)
    if not np.isfinite(result).all(): raise ValueError(name+' requires finite values.')
    return result


def _provenance(control, chart):
    a=_real(control,'control_triangle')
    if a.shape!=(3,3) or not np.allclose(np.linalg.norm(a,axis=1),1.,rtol=0.,atol=2e-12):
        raise ValueError('Control requires three represented unit vectors.')
    basis=tuple(tuple(Fraction(float(x)) for x in row) for row in a)
    dot=lambda x,y:sum((u*v for u,v in zip(x,y)),Fraction(0))
    cross=lambda x,y:(x[1]*y[2]-x[2]*y[1],x[2]*y[0]-x[0]*y[2],x[0]*y[1]-x[1]*y[0])
    if dot(basis[0],cross(basis[1],basis[2]))<=0: raise ValueError('Control winding must be positive.')
    if np.any(a@a.sum(axis=0)<=0): raise ValueError('Control must occupy a common open hemisphere.')
    polygon=[]
    for point in chart:
        if len(point)!=2 or any(isinstance(x,(bool,np.bool_)) or not isinstance(x,(Fraction,int,np.integer)) for x in point):
            raise ValueError('Chart coordinates must be exact Fractions or integers.')
        x,y=(Fraction(v) for v in point)
        if x<0 or y<0 or x+y>1: raise ValueError('Chart polygon leaves its original control triangle.')
        polygon.append((x,y))
    if len(polygon)<3 or len(set(polygon))!=len(polygon): raise ValueError('Chart requires distinct convex corners.')
    turns=[]
    for i,b in enumerate(polygon):
        x,y=polygon[i-1],polygon[(i+1)%len(polygon)]
        turns.append((b[0]-x[0])*(y[1]-b[1])-(b[1]-x[1])*(y[0]-b[0]))
    if any(t<0 for t in turns) or not any(t>0 for t in turns):
        raise ValueError('Chart requires positive convex winding.')
    for edge_a,edge_b in zip(polygon,polygon[1:]+polygon[:1]):
        if any((edge_b[0]-edge_a[0])*(c[1]-edge_a[1])-(edge_b[1]-edge_a[1])*(c[0]-edge_a[0])<0 for c in polygon):
            raise ValueError('Chart must be a simple convex polygon.')
    return a.copy(),basis,tuple(polygon)


def _analytic(basis,chart,precision):
    # Keep the exact chart provenance for algebraic frame identities. Decimal
    # cancellation, even at high precision, cannot certify an exact zero.
    rays=[]
    for x,y in chart:
        weights=(1-x-y,x,y)
        rays.append(tuple(sum((weights[j]*basis[j][k] for j in range(3)),Fraction(0)) for k in range(3)))
    zero=_certified_centroid_zeros(rays)
    with localcontext() as ctx:
        ctx.prec=precision
        b=[[_decimal(x) for x in row] for row in basis]
        points=[_unit([_decimal(x) for x in ray]) for ray in rays]
        normal=_unit([D(0) if zero[k] else sum(row[k] for row in points) for k in range(3)])
        chosen=min(range(3),key=lambda i:abs(normal[i]))
        axis=[D(0)]*3; axis[chosen]=D(1)
        first=_unit(_cross(normal,axis))
        # For exact unit n perpendicular to e_j, n x unit(n x e_j)=-e_j.
        # Preserve that proved identity instead of exporting Decimal residuals
        # such as 1e-80 into the compensated binary64 transform.
        second=[-value for value in axis] if zero[chosen] else _cross(normal,first)
        frame=_transpose([first,second,normal])
        area=D(0); a=points[0]
        for b,c in zip(points[1:-1],points[2:]):
            determinant=_dot(a,_cross([v-u for u,v in zip(a,b)],[v-u for u,v in zip(a,c)]))
            denominator=1+_dot(a,b)+_dot(b,c)+_dot(c,a)
            if determinant<0 or denominator<=0: raise ValueError('Unresolved convex analytic polygon.')
            area+=2*_atan(determinant/denominator)
        if area<=0: raise ValueError('Positive chart has unresolved spherical area.')
        boundary=[[D(0)]*3 for _ in range(3)]
        for a,b in zip(points,points[1:]+points[:1]):
            normal_edge=_cross(a,b); midpoint=[x+y for x,y in zip(a,b)]; denominator=1+_dot(a,b)
            if denominator<=0: raise ValueError('Antipodal analytic edge.')
            for i in range(3):
                for j in range(3):
                    boundary[i][j]-=(normal_edge[i]*midpoint[j]+midpoint[i]*normal_edge[j])/denominator
        metric=[[(2*area/3 if i==j else D(0))+boundary[i][j]/6 for j in range(3)] for i in range(3)]
        local=_local_matrix(metric,frame)
        local_points=[[_dot(column,point) for column in zip(*frame)] for point in points]
        return dict(area=area,metric=metric,frame=frame,local=local,points=points,local_points=local_points)


def _factor(matrix):
    if not np.isfinite(matrix).all() or np.any(np.diag(matrix)<=0):
        raise ValueError('Local positive metric is unresolved in binary64.')
    scale=np.sqrt(np.diag(matrix)); normalized=matrix/scale[:,None]/scale[None,:]
    try: factor=np.linalg.cholesky(normalized).T*scale[None,:]
    except np.linalg.LinAlgError as error: raise ValueError('Local positive metric has unresolved pivots.') from error
    if not np.isfinite(factor).all() or np.any(np.diag(factor)<=0): raise ValueError('Local metric factor is nonfinite.')
    return factor


def _whitened_error(difference, factor):
    # C.T C=J; C^-T difference C^-1 checks EVERY physical direction.
    x=np.linalg.solve(factor.T,difference)
    x=np.linalg.solve(factor.T,x.T).T
    result=float(np.linalg.norm(x,2))
    if not np.isfinite(result): raise ValueError('Local weak-axis comparison is unresolved.')
    return result


def _encode(value):
    if isinstance(value,np.ndarray): return ['array',str(value.dtype),list(value.shape),value.tobytes().hex()]
    if isinstance(value,Fraction): return ['fraction',str(value.numerator),str(value.denominator)]
    if isinstance(value,tuple): return ['tuple',[_encode(x) for x in value]]
    if isinstance(value,(str,int,float,bool)): return [type(value).__name__,value]
    raise ValueError('Unsupported stable metric provenance type.')


@dataclass(frozen=True)
class StableMetric:
    control_triangle: np.ndarray
    chart_polygon: tuple
    frame_hi: np.ndarray
    frame_lo: np.ndarray
    local_polygon: np.ndarray
    polygon: np.ndarray
    metric_local: np.ndarray
    factor_local: np.ndarray
    metric_global: np.ndarray
    area_unit: float
    precision: int
    convergence_error: float
    _frame_decimal: tuple
    _metric_decimal: tuple
    _area_decimal: str
    signature: str=''

    def _signature(self):
        values=[(field.name,_encode(getattr(self,field.name))) for field in fields(self) if field.name!='signature']
        return hashlib.sha256(json.dumps(values,separators=(',',':'),allow_nan=False).encode()).hexdigest()

    def validate(self):
        if self.signature!=self._signature(): raise ValueError('Stable basal geometry/frame/metric was modified.')
        return self

    def localize(self, values, *, normalize=False):
        self.validate(); a=_real(values,'Local-frame values')
        if not isinstance(normalize,(bool,np.bool_)): raise ValueError('normalize must be Boolean.')
        if a.shape[-1:]!=(3,): raise ValueError('Local-frame vectors must end in three coordinates.')
        with localcontext() as ctx:
            ctx.prec=self.precision
            columns=list(zip(*[[D(x) for x in row] for row in self._frame_decimal]))
            result=[]
            for row in a.reshape(-1,3):
                p=[D.from_float(float(x)) for x in row]
                if normalize: p=_unit(p)
                result.append([float(_dot(column,p)) for column in columns])
        result=np.asarray(result).reshape(a.shape)
        if not np.isfinite(result).all(): raise ValueError('Local transform exceeds finite arithmetic.')
        return result

    def globalize(self, values):
        self.validate(); a=_real(values,'Global-frame values')
        if a.shape[-1:]!=(3,): raise ValueError('Global-frame vectors must end in three coordinates.')
        with localcontext() as ctx:
            ctx.prec=self.precision
            frame=[[D(x) for x in row] for row in self._frame_decimal]
            result=[[float(_dot(row,[D.from_float(float(x)) for x in v])) for row in frame] for v in a.reshape(-1,3)]
        result=np.asarray(result).reshape(a.shape)
        if not np.isfinite(result).all(): raise ValueError('Global transform exceeds finite arithmetic.')
        return result

    def metric_in(self, other):
        self.validate(); other.validate()
        with localcontext() as ctx:
            ctx.prec=max(self.precision,other.precision)
            local=_local_matrix([[D(x) for x in row] for row in self._metric_decimal],
                [[D(x) for x in row] for row in other._frame_decimal])
        return np.array(local,float)


def build(control_triangle, chart_polygon, *, relative_tolerance=2e-12, precision_schedule=(48,80,128,192)):
    control,basis,chart=_provenance(control_triangle,chart_polygon)
    if (isinstance(relative_tolerance,(bool,np.bool_)) or not isinstance(relative_tolerance,(int,float))
            or not np.isfinite(relative_tolerance) or not 0<relative_tolerance<.01):
        raise ValueError('Precision agreement tolerance must be finite and positive below .01.')
    schedule=tuple(precision_schedule)
    if (len(schedule)<2 or any(isinstance(p,bool) or not isinstance(p,int) or not 32<=p<=4096 for p in schedule)
            or any(b<=a for a,b in zip(schedule,schedule[1:]))):
        raise ValueError('Precision schedule requires increasing integer decimal precisions, at least32.')
    previous=None
    for precision in schedule:
        try: current=_analytic(basis,chart,precision)
        except ValueError:
            # Exact positive convex chart/hemisphere was already validated;
            # unresolved analytic signs at this precision require escalation.
            previous=None
            continue
        metric=np.array(current['local'],float)
        try: factor=_factor(metric)
        except ValueError:
            previous=current; continue
        if previous is not None:
            with localcontext() as ctx:
                ctx.prec=precision
                old_local=_local_matrix(previous['metric'],current['frame'])
                diff=np.array([[float(old_local[i][j]-current['local'][i][j]) for j in range(3)] for i in range(3)])
                area_error=float(abs(previous['area']-current['area'])/current['area'])
            error=max(_whitened_error(diff,factor),area_error)
            if error<=relative_tolerance:
                with localcontext() as ctx:
                    ctx.prec=precision
                    hi=np.array(current['frame'],float)
                    lo=np.array([[float(current['frame'][i][j]-D.from_float(float(hi[i,j]))) for j in range(3)] for i in range(3)])
                result=StableMetric(control,chart,hi,lo,np.array(current['local_points'],float),
                    np.array(current['points'],float),metric,factor,np.array(current['metric'],float),
                    float(current['area']),precision,error,
                    tuple(tuple(str(x) for x in row) for row in current['frame']),
                    tuple(tuple(str(x) for x in row) for row in current['metric']),str(current['area']))
                if not np.isfinite(result.area_unit) or result.area_unit<=0: raise ValueError('Area cast is unresolved.')
                object.__setattr__(result,'signature',result._signature())
                return result.validate()
        previous=current
    raise ValueError('Analytic local basal metric did not meet its full weak-axis precision convergence gate.')
