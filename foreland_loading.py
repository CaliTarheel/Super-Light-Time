"""Finite mountain-load redistribution beside retained native collision belts.

This is a regional correction to existing local compensation, not a new
uncompensated-load model. A constant-rigidity, infinite-plane Kelvin kernel is
evaluated at local spherical distances. Only the positive difference from local
Airy response supplies the existing foreland accommodation state. Negative
forebulge/uplift, water/sediment loading, stresses and mechanical work are absent.

Kernel basis: Wickert (2016), https://doi.org/10.5194/gmd-9-997-2016.
"""
from __future__ import annotations

import numpy as np
from functools import lru_cache
from itertools import product
import material_surface

VERSION = 1
RADIUS_KM = 6371.
DEFAULTS = dict(elastic_thickness_km=40., young_modulus_pa=70e9, poisson_ratio=.25,
                mantle_density_kg_m3=3300., load_density_kg_m3=2800., gravity_m_s2=9.81,
                background_height_m=500., quadrature_edge_fraction=.5,
                kernel_cutoff_alpha=12.)


def parameters(overrides=None):
    result = dict(DEFAULTS)
    if overrides:
        if set(overrides)-set(result):
            raise ValueError('Unknown distributed foreland parameter.')
        result.update(overrides)
    for key, value in result.items():
        if isinstance(value, (bool, np.bool_)) or not np.isscalar(value) or not np.isfinite(value):
            raise ValueError('Foreland parameters must be finite scalars: '+key)
        result[key] = float(value)
    if (not 5 <= result['elastic_thickness_km'] <= 120
            or result['young_modulus_pa'] <= 0 or not 0 <= result['poisson_ratio'] < .5
            or min(result['mantle_density_kg_m3'], result['load_density_kg_m3'], result['gravity_m_s2']) <= 0
            or result['background_height_m'] < 0
            or not 0 < result['quadrature_edge_fraction'] <= .5
            or not 10 <= result['kernel_cutoff_alpha'] <= 12):
        raise ValueError('Distributed foreland parameters are outside their physical/numerical bounds.')
    rigidity = result['young_modulus_pa']*(1000.*result['elastic_thickness_km'])**3/(12*(1-result['poisson_ratio']**2))
    result['rigidity_n_m'] = rigidity
    result['alpha_km'] = (rigidity/(result['mantle_density_kg_m3']*result['gravity_m_s2']))**.25/1000.
    result['density_ratio'] = result['load_density_kg_m3']/result['mantle_density_kg_m3']
    return result


def _kelvin_table():
    """Numpy-only K0 series on a finite range; no external special-function ABI.

    K0(z)=-(log(z/2)+gamma)*I0(z)+sum(H_k*(z*z/4)**k/(k!)**2).
    For z=x*exp(i*pi/4), kei(x)=imag(K0(z)); kei(0)=-pi/4.
    """
    x=np.linspace(0.,12.,12001)
    z=x[1:]*np.exp(1j*np.pi/4)
    term=np.ones(len(z),complex);bessel=term.copy();harmonic_sum=np.zeros(len(z),complex)
    harmonic=0.
    for k in range(1,81):
        term*=z*z/(4*k*k);harmonic+=1./k
        bessel+=term;harmonic_sum+=harmonic*term
    value=np.empty(len(x));value[0]=-np.pi/4
    value[1:]=(-(np.log(z/2)+np.euler_gamma)*bessel+harmonic_sum).imag
    return x,value


_KELVIN_X,_KELVIN_VALUE=_kelvin_table()


@lru_cache(maxsize=8)
def _kernel_mass(cutoff):
    if not 10 <= cutoff <= 12:
        raise ValueError('Kernel cutoff must be10–12 flexural parameters.')
    x=np.r_[_KELVIN_X[_KELVIN_X<cutoff],cutoff]
    y=-x*np.interp(x,_KELVIN_X,_KELVIN_VALUE)
    return float(np.sum((y[1:]+y[:-1])*.5*np.diff(x)))


def flexural_kernel(distance_km, alpha_km, cutoff_alpha=12.):
    """Signed, downward-positive response per unit area; integral tends to one.

    Signed kei lobes are retained. One global radial-integral normalization
    corrects the truncated tail (0.024% at12alpha), never the available local
    load. This prevents a uniform compensated load acquiring a spurious basin.
    The untruncated r=0 limit is1/(8alpha**2); all points share the tail factor.
    """
    distance = np.asarray(distance_km, float)
    if (not np.isfinite(distance).all() or np.any(distance < 0)
            or not np.isfinite(alpha_km) or alpha_km <= 0):
        raise ValueError('Kernel distances must be nonnegative and alpha positive.')
    value = -np.interp(distance/alpha_km,_KELVIN_X,_KELVIN_VALUE)/(2*np.pi*alpha_km**2*_kernel_mass(float(cutoff_alpha)))
    return np.where(distance <= cutoff_alpha*alpha_km, value, 0.)


def _unit(values):
    values = np.asarray(values, float)
    norm = np.linalg.norm(values, axis=-1)
    if not np.isfinite(values).all() or np.any(np.abs(norm-1) > 1e-5):
        raise ValueError('Foreland geometry requires finite unit XYZ vectors.')
    return values/norm[..., None]


def triangle_quadrature(triangles, integrated_weights, maximum_edge_km, *, radius_km=RADIUS_KM):
    """Refine real finite triangles before applying a three-point area rule.

    Children use their spherical areas. Weights sum to each supplied parent
    integral, so increasing quadrature resolution never adds mountain load.
    """
    tri = _unit(triangles).copy()
    weights = np.asarray(integrated_weights, float).copy()
    if tri.ndim != 3 or tri.shape[1:] != (3, 3) or weights.shape != (len(tri),):
        raise ValueError('Quadrature triangles and integrated weights must align.')
    if (not np.isfinite(weights).all() or np.any(weights < 0)
            or not np.isfinite(maximum_edge_km) or maximum_edge_km <= 0):
        raise ValueError('Quadrature weights and edge length must be positive/finite.')
    out_points, out_weights = [], []
    bary = np.array([[2/3, 1/6, 1/6], [1/6, 2/3, 1/6], [1/6, 1/6, 2/3]])
    for depth in range(12):
        if not len(tri):
            break
        edge = np.arccos(np.clip(np.einsum('nij,nij->ni', tri, np.roll(tri, -1, axis=1)), -1., 1.)).max(axis=1)*radius_km
        ready = edge <= maximum_edge_km*(1+1e-12)
        if np.any(ready):
            points = np.einsum('ij,njk->nik', bary, tri[ready])
            points /= np.linalg.norm(points, axis=2)[:, :, None]
            out_points.append(points.reshape(-1, 3))
            out_weights.append(np.repeat(weights[ready]/3., 3))
        tri, weights = tri[~ready], weights[~ready]
        if not len(tri):
            break
        mid = tri+np.roll(tri, -1, axis=1)
        mid /= np.linalg.norm(mid, axis=2)[:, :, None]
        a,b,c = np.moveaxis(tri, 1, 0); ab,bc,ca = np.moveaxis(mid, 1, 0)
        children = np.stack((np.stack((a,ab,ca),axis=1), np.stack((ab,b,bc),axis=1),
                             np.stack((ca,bc,c),axis=1), np.stack((ab,bc,ca),axis=1)),axis=1)
        flat = children.reshape(-1,3,3)
        area = material_surface.spherical_face_areas(flat.reshape(-1,3),
            np.arange(len(flat)*3).reshape(-1,3), radius_km).reshape(-1,4)
        weights = (weights[:,None]*area/area.sum(axis=1)[:,None]).reshape(-1)
        tri = flat
    else:
        raise ValueError('Foreland source quadrature exceeded its geometric refinement bound.')
    if not out_points:
        return np.empty((0,3)), np.empty(0)
    return np.concatenate(out_points), np.concatenate(out_weights)


def prepare(*, vertices, faces, area_km2, face_ids, owners, kinds, achieved_height_m,
            collision_support_m, collision_suture, exposed_fraction, collision_sheets,
            contacts, parameter_overrides=None, signed_support=False):
    """Prepare ephemeral native load quadrature; no caller arrays are mutated."""
    cfg = parameters(parameter_overrides)
    vertices = _unit(vertices)
    faces = np.asarray(faces)
    if faces.ndim != 2 or faces.shape[1] != 3 or faces.dtype.kind not in 'iu':
        raise ValueError('Foreland material faces must be integral triangles.')
    if np.any(faces < 0) or np.any(faces >= len(vertices)):
        raise ValueError('Foreland material faces reference absent vertices.')
    n = len(faces)
    arrays = dict(area_km2=area_km2, face_ids=face_ids, owners=owners, kinds=kinds,
                  achieved_height_m=achieved_height_m, collision_support_m=collision_support_m,
                  collision_suture=collision_suture, exposed_fraction=exposed_fraction,
                  collision_sheets=collision_sheets)
    arrays = {key:np.asarray(value) for key,value in arrays.items()}
    if any(value.shape != (n,) or not np.isfinite(value).all() for value in arrays.values()):
        raise ValueError('Distributed foreland face arrays must align and be finite.')
    for key in ('face_ids','owners','kinds','collision_sheets'):
        if arrays[key].dtype.kind not in 'iu':
            raise ValueError('Foreland identities must be integral: '+key)
    if (len(np.unique(arrays['face_ids'])) != n or np.any(arrays['area_km2'] <= 0)
            or (not signed_support and np.any(arrays['collision_support_m'] < 0))
            or np.any((arrays['collision_suture'] < 0)|(arrays['collision_suture'] > 1))
            or np.any((arrays['exposed_fraction'] < 0)|(arrays['exposed_fraction'] > 1))):
        raise ValueError('Invalid distributed foreland identity, area, support or exposure.')
    receivers = {}
    for row in contacts:
        for key in ('top_sheet','under_sheet'):
            sheet = int(row[key])
            receivers.setdefault(sheet,set()).update((int(row['top_owner']),int(row['under_owner'])))
    eligible = (np.isin(arrays['kinds'], (1,2))
        & ((arrays['collision_support_m'] > 1e-8)|(arrays['collision_suture'] > 1e-8))
        & np.isin(arrays['collision_sheets'], list(receivers)))
    height = np.where(eligible, np.maximum(arrays['achieved_height_m']-cfg['background_height_m'],0.),0.)
    integral = arrays['area_km2']*arrays['exposed_fraction']*height
    selected = np.flatnonzero(integral > 0)
    source_points, source_weights, receiver_groups = [], [], []
    for sheet in np.unique(arrays['collision_sheets'][selected]):
        at = selected[arrays['collision_sheets'][selected]==sheet]
        points, weight = triangle_quadrature(vertices[faces[at]],integral[at],
            cfg['quadrature_edge_fraction']*cfg['alpha_km'])
        source_points.append(points);source_weights.append(weight)
        receiver_groups.extend([tuple(sorted(receivers[int(sheet)]))]*len(points))
    xyz = np.concatenate(source_points) if source_points else np.empty((0,3))
    weights = np.concatenate(source_weights) if source_weights else np.empty(0)
    bins = {}
    chord=2*np.sin(cfg['kernel_cutoff_alpha']*cfg['alpha_km']/(2*RADIUS_KM))
    for owner in sorted({owner for group in receiver_groups for owner in group}):
        indices = np.array([i for i,group in enumerate(receiver_groups) if owner in group],int)
        cell=np.floor((xyz[indices]+1.)/chord).astype(int)
        owner_bins={}
        for index,key in zip(indices,map(tuple,cell)):
            owner_bins.setdefault(key,[]).append(int(index))
        bins[owner]={key:np.asarray(value,int) for key,value in owner_bins.items()}
    diagnostics = dict(version=VERSION,state='prepared',model='finite exposed mountain-load redistribution relative to local compensation',
        source_faces=len(selected),quadrature_points=len(xyz),source_exposed_area_km2=float(np.sum(arrays['area_km2'][selected]*arrays['exposed_fraction'][selected])),
        source_height_area_m_km2=float(integral.sum()),quadrature_height_area_m_km2=float(weights.sum()),
        quadrature_integral_residual_m_km2=float(weights.sum()-integral.sum()),parameters=cfg,
        receiver_owner_count=len(bins),sampled_target_points=0,maximum_unbounded_target_m=0.,
        maximum_target_m=0.,capped_target_points=0,
        source_policy='Achieved compensated height above500m, retained collision support/suture, actual exposed face area; no boundary maximum.',
        correction_policy='Signed Kelvin response minus the SAME face-average local Airy load used by source quadrature; only positive regional correction enters existing foreland accommodation.',
        limitations='Constant40km elastic-thickness default, local spherical-distance approximation to an infinite planar kernel; achieved topography is a redistribution proxy, not independently reconstructed uncompensated load. Air-filled foundation; no water/sediment/forebulge/work solve.',
        truncated_kernel_radial_integral=_kernel_mass(cfg['kernel_cutoff_alpha']),
        kernel_normalization='One global signed radial integral; no query/load dependent renormalization.',
        changes_material_volume=False,changes_gravity_energy=False)
    return dict(version=VERSION,parameters=cfg,vertices=vertices,faces=faces,face_ids=arrays['face_ids'].copy(),
                local_load_height_m=height*arrays['exposed_fraction'],
                source_xyz=xyz,source_weights=weights,spatial_bins=bins,spatial_bin_chord=chord,diagnostics=diagnostics)


def prepare_native(s):
    mesh = s.material_surface
    # These achieved scalar heights include the same explicit stack support
    # that the native control surface uses. They are not control-cell maxima.
    from structure_engine import material_height
    context = prepare(vertices=mesh['vertices'],faces=mesh['faces'],area_km2=mesh['area_km2'],
        face_ids=s.parcel_patch,owners=s.parcel_plate,kinds=s.kind,
        achieved_height_m=material_height(s.kind,s.relief)+s.parcel_collision_support_m,
        collision_support_m=s.parcel_collision_support_m,collision_suture=s.parcel_collision_suture,
        exposed_fraction=s.parcel_exposed_fraction,collision_sheets=s.parcel_collision_sheet,
        contacts=s.collision_contacts,parameter_overrides=getattr(s,'foreland_loading_parameters',None),
        signed_support=getattr(s,'retained_dense_crust_version',0)==1)
    s.foreland_loading_diagnostics = context['diagnostics']
    return context


def regional_response(points, owners, context):
    """Downward-positive distributed response before local compensation removal."""
    points = _unit(points)
    owners = np.asarray(owners)
    if points.ndim != 2 or points.shape[1] != 3 or owners.shape != (len(points),):
        raise ValueError('Foreland query coordinates and owners must align.')
    result = np.zeros(len(points))
    cfg=context['parameters'];cutoff=cfg['kernel_cutoff_alpha']*cfg['alpha_km']
    chord=context['spatial_bin_chord']
    offsets=tuple(product((-1,0,1),repeat=3))
    for owner in np.unique(owners):
        if int(owner) not in context['spatial_bins']:continue
        bins=context['spatial_bins'][int(owner)]
        query=np.flatnonzero(owners==owner)
        cells=np.floor((points[query]+1.)/chord).astype(int)
        for index,cell in zip(query,cells):
            parts=[bins[key] for offset in offsets if (key:=tuple(cell+offset)) in bins]
            if not parts:continue
            at=np.concatenate(parts)
            delta=context['source_xyz'][at]-points[index]
            at=at[np.einsum('ij,ij->i',delta,delta)<=chord*chord*(1+1e-12)]
            distance=np.arccos(np.clip(context['source_xyz'][at]@points[index],-1.,1.))*RADIUS_KM
            result[index]=cfg['density_ratio']*np.dot(context['source_weights'][at],
                flexural_kernel(distance,cfg['alpha_km'],cfg['kernel_cutoff_alpha']))
    return result


def target(points, owners, face_ids, context, *, max_depth_m=1800.):
    """Return regional correction using exact material IDs for local subtraction."""
    points=_unit(points);ids=np.asarray(face_ids)
    if ids.shape!=(len(points),) or ids.dtype.kind not in 'iu' or not np.isfinite(max_depth_m) or max_depth_m<0:
        raise ValueError('Foreland targets require aligned persistent face IDs and a nonnegative cap.')
    order=np.argsort(context['face_ids']);sorted_ids=context['face_ids'][order]
    at=np.searchsorted(sorted_ids,ids);valid=at<len(order)
    valid[valid]&=sorted_ids[at[valid]]==ids[valid]
    if not valid.all():raise ValueError('A regional foreland target has no exact source material face.')
    face=order[at];tri=context['vertices'][context['faces'][face]]
    bary=np.linalg.solve(np.swapaxes(tri,1,2),points[...,None])[...,0]
    bary/=bary.sum(axis=1)[:,None]
    if np.any(bary < -1e-7):raise ValueError('A foreland material marker left its recorded triangle.')
    # Subtract the same face-average load that was integrated above. Smoothing
    # only this term would dilute isolated high faces and invent self-load pits.
    local=context['local_load_height_m'][face]
    response=regional_response(points,owners,context)
    raw=np.maximum(response-context['parameters']['density_ratio']*local,0.)
    result=np.minimum(raw,max_depth_m)
    diag=context['diagnostics'];diag['state']='evaluated'
    diag['sampled_target_points']+=len(points)
    diag['maximum_unbounded_target_m']=max(diag['maximum_unbounded_target_m'],float(raw.max(initial=0.)))
    diag['maximum_target_m']=max(diag['maximum_target_m'],float(result.max(initial=0.)))
    diag['capped_target_points']+=int(np.count_nonzero(raw>max_depth_m))
    return result
