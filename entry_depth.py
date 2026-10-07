"""Reference-area integration of a local, positive affine entry depth.

For a uniform material triangle the distribution of a linear coordinate is
piecewise linear between its three corner values. Negative coordinates all
map to depth zero. Keeping that atom and the positive density avoids averaging
depth before a pressure or temperature threshold is applied.
"""
import numpy as np
from numpy.polynomial.legendre import Legendre

def _lobatto(order):
    polynomial=Legendre.basis(order-1)
    nodes=np.r_[-1.,polynomial.deriv().roots(),1.]
    return nodes,2./(order*(order-1)*polynomial(nodes)**2)

_LOBATTO={order:_lobatto(order) for order in (4,8)}


def distribution(corners):
    """Return a zero-depth atom and positive linear-density intervals.

    An interval is (left depth, right depth, density at left, density at right).
    Equal positive corner values instead give a single point mass.
    """
    q=np.asarray(corners,float)
    if q.shape!=(3,) or not np.isfinite(q).all():
        raise ValueError('Local entry depth needs three finite signed corner values.')
    a,b,c=sorted(q)
    if a==c:return [(max(float(a),0.),1.)],[]
    atoms=[];panels=[];zero=0.
    for lo,hi,p0,p1 in ((a,b,0.,2./(c-a)),(b,c,2./(c-a),0.)):
        if lo==hi:continue
        slope=(p1-p0)/(hi-lo)
        if lo<0.:
            end=min(hi,0.)
            zero+=(end-lo)*(p0+(p0+slope*(end-lo)))/2.
        if hi>0.:
            start=max(lo,0.)
            panels.append((float(start),float(hi),float(p0+slope*(start-lo)),float(p1)))
    if zero>0.:atoms.append((0.,float(zero)))
    return atoms,panels


def _rule(panel,order):
    lo,hi,p0,p1=panel
    x,w=_LOBATTO[order]
    fraction=(x+1.)/2.
    nodes=lo+(hi-lo)*fraction
    weights=w*(hi-lo)/2.*(p0+(p1-p0)*fraction)
    mass=(hi-lo)*(p0+p1)/2.
    weights*=mass/weights.sum()
    return nodes,weights


def quadrature(corners,evaluate,*,relative_tolerance=1e-6,absolute_tolerance=1e-10,
               max_panels=1024,breakpoints=()):
    """Adapt positive quadrature to the caller's complete local source result.

    evaluate(depths) returns one row per depth, with columns for all state and
    budget values whose means must converge. A paired 4/8-point Lobatto estimate
    includes panel endpoints: evolving columns can shift a thermal cap just
    beyond an initial breakpoint, where interior-only Gauss nodes miss it.
    This is an error estimate, not a rigorous
    bound for arbitrary functions; callers must also test tolerance refinement.
    A budget failure raises instead of returning an under-resolved result.
    """
    if (not np.isfinite(relative_tolerance) or relative_tolerance<=0.
            or not np.isfinite(absolute_tolerance) or absolute_tolerance<=0.
            or isinstance(max_panels,(bool,np.bool_)) or not isinstance(max_panels,(int,np.integer)) or max_panels<1):
        raise ValueError('Entry quadrature needs positive tolerances and an integer panel budget.')
    atoms,intervals=distribution(corners)
    cuts=np.asarray(breakpoints,float)
    if cuts.ndim!=1 or not np.isfinite(cuts).all():raise ValueError('Entry quadrature breakpoints must be finite.')
    split=[]
    for lo,hi,p0,p1 in intervals:
        points=np.unique(np.r_[lo,cuts[(cuts>lo)&(cuts<hi)],hi])
        for left,right in zip(points[:-1],points[1:]):
            split.append((left,right,p0+(p1-p0)*(left-lo)/(hi-lo),p0+(p1-p0)*(right-lo)/(hi-lo)))
    def values(nodes):
        result=np.asarray(evaluate(np.asarray(nodes)),float)
        if result.ndim!=2 or result.shape[0]!=len(nodes) or not np.isfinite(result).all():
            raise ValueError('Entry source quadrature received nonfinite or unaligned local results.')
        return result
    atom_nodes=np.array([a[0] for a in atoms]);atom_weights=np.array([a[1] for a in atoms])
    atom_mean=atom_weights@values(atom_nodes) if atoms else None
    panels=[]
    evaluations=len(atoms)
    def assess(panel):
        nonlocal evaluations
        x4,w4=_rule(panel,4);x8,w8=_rule(panel,8)
        y4=values(x4);y8=values(x8);evaluations+=12
        high=w8@y8;low=w4@y4
        return dict(panel=panel,nodes=x8,weights=w8,mean=high,error=np.abs(high-low))
    if len(split)>max_panels:raise ValueError('Entry source quadrature exceeds its panel budget.')
    panels=[assess(panel) for panel in split]
    if not panels:
        return atom_nodes,atom_weights,dict(panels=0,evaluations=evaluations,normalized_error=0.)
    while True:
        mean=sum((p['mean'] for p in panels),np.zeros_like(panels[0]['mean']))
        if atom_mean is not None:mean+=atom_mean
        error=sum((p['error'] for p in panels),np.zeros_like(mean))
        tolerance=absolute_tolerance+relative_tolerance*np.abs(mean)
        scaled=error/tolerance
        if np.max(scaled,initial=0.)<=1.:break
        if len(panels)>=max_panels:
            raise ValueError('Entry source quadrature did not converge within its panel budget.')
        worst=max(range(len(panels)),key=lambda i:float(np.max(panels[i]['error']/tolerance)))
        lo,hi,p0,p1=panels.pop(worst)['panel'];middle=lo+(hi-lo)/2.;pm=(p0+p1)/2.
        if middle==lo or middle==hi:raise ValueError('Entry source quadrature cannot resolve another depth interval.')
        panels.extend((assess((lo,middle,p0,pm)),assess((middle,hi,pm,p1))))
    nodes=np.concatenate([atom_nodes]+[p['nodes'] for p in panels])
    weights=np.concatenate([atom_weights]+[p['weights'] for p in panels])
    if not np.isfinite(weights).all() or abs(float(weights.sum())-1.)>2e-13 or np.any(weights<0.):
        raise ValueError('Entry depth measure lost its material reference area.')
    positive=weights>0.;nodes=nodes[positive];weights=weights[positive]
    weights/=weights.sum()
    return nodes,weights,dict(panels=len(panels),evaluations=evaluations,normalized_error=float(np.max(scaled,initial=0.)))


def point_depth(corners,triangle,point):
    """Reference-linear depth at a projective material marker, not mean depth."""
    triangle=np.asarray(triangle,float);point=np.asarray(point,float)
    if triangle.shape!=(3,3) or point.shape!=(3,) or not np.isfinite(triangle).all() or not np.isfinite(point).all():
        raise ValueError('Entry marker needs a finite material triangle and position.')
    weights=np.linalg.solve(triangle.T,point)
    total=float(weights.sum())
    if total<=0. or np.any(weights/total < -2e-10):
        raise ValueError('Entry marker left its registered material face.')
    weights/=total
    q=np.asarray(corners,float)
    if q.shape!=(3,) or not np.isfinite(q).all():raise ValueError('Entry marker needs three finite depth coordinates.')
    return max(float(weights@q),0.)
