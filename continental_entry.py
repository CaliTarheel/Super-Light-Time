"""Thin-sheet buoyancy energy for explicitly supplied continental-entry domains.

These are explicit plate-force and fixed-hinge sheet components, not an enabled
entry or polarity policy. A caller must identify material entering beneath an oceanic
upper plate. It must not add this ramp energy to burial already represented by
the continental stack potential. Domain transport, material remeshing, burial
state and inherited-polarity integration remain separate required work.

The conveyor distance normal to a local great-circle hinge is interpolated
linearly in each material reference triangle. Its positive part is integrated
before averaging. Depth is distance*sin(dip), consistent with the existing
fixed-hinge down-dip material-speed convention. Existing crust volume/mass is
displaced through a uniform mantle; it is neither removed nor duplicated.
"""
import math
import numpy as np
import column_density
import material_surface
import finite_entry_arc
import entry_depth
from entry_overlap_geometry import clipped_entry_stack_intersections as _entry_stack_intersections


class EntryGeometryError(ValueError):
    """A trial leaves the explicitly resolved oceanic entry domain."""


def validate_finite_front(normals,midpoints,half_lengths_km,radius_km):
    """Validate optional finite arc geometry; clipped energy can leave its arc."""
    if midpoints is None and half_lengths_km is None:
        return
    middle=np.asarray(midpoints,float)
    half=np.asarray(half_lengths_km,float)
    normals=np.asarray(normals,float)
    if (middle.shape!=(len(normals),3) or half.shape!=(len(normals),)
            or np.any(half<=0.) or np.any(np.isnan(half))):
        raise ValueError('Finite entry fronts require aligned positive arc bounds.')
    finite=np.isfinite(half)
    if not np.any(finite):
        return
    for normal,middle,length in zip(normals[finite],middle[finite],half[finite]):
        finite_entry_arc.endpoint_planes(normal,middle,length,radius_km)


class FrozenPotential:
    """Entry potential at fixed hinge, topology, composition and face volumes.

    This is a transient mechanical-stage object, not persisted domain history.
    The caller supplies hinges at this stage's epoch. Every trial checks the
    selected material for new stack overlap; volumes/composition cannot change
    inside an energy-decreasing mechanical solve.
    """

    def __init__(self, vertices, faces, volumes_km3, masses_kg, sheets,
                 face_indices, hinge_normals, dip_degrees, *, radius_km=6371.,
                 finite_midpoints=None,finite_half_lengths_km=None,
                 coordinate_mode='along_slab_conveyor'):
        self.faces=np.array(faces,copy=True)
        self.volumes=np.array(volumes_km3,float,copy=True)
        self.masses=np.array(masses_kg,float,copy=True)
        self.sheets=np.array(sheets,copy=True)
        self.indices=np.array(face_indices,copy=True)
        self.normals=np.array(hinge_normals,float,copy=True)
        self.dips=np.array(dip_degrees,float,copy=True)
        self.finite_midpoints=(None if finite_midpoints is None else np.array(finite_midpoints,float,copy=True))
        self.finite_half_lengths_km=(None if finite_half_lengths_km is None else
                                     np.array(finite_half_lengths_km,float,copy=True))
        self.radius=radius_km
        self.coordinate_mode=coordinate_mode
        if (self.faces.ndim!=2 or self.faces.shape[1]!=3 or self.faces.dtype.kind not in 'iu'
                or self.volumes.shape!=(len(self.faces),) or self.masses.shape!=self.volumes.shape
                or not np.isfinite(self.volumes).all() or np.any(self.volumes<=0.)
                or not np.isfinite(self.masses).all() or np.any(self.masses<=0.)
                or self.sheets.shape!=self.volumes.shape or self.sheets.dtype.kind not in 'iu'
                or self.indices.ndim!=1 or self.indices.dtype.kind not in 'iu'
                or len(np.unique(self.indices))!=len(self.indices)
                or np.any(self.indices<0) or np.any(self.indices>=len(self.faces))):
            raise ValueError('Frozen entry needs aligned conserved inventories and unique material indices.')
        self.evaluate(vertices,self.faces,self.volumes,self.sheets,radius=radius_km)
        for array in (self.faces,self.volumes,self.masses,self.sheets,self.indices,self.normals,self.dips,
                      self.finite_midpoints,self.finite_half_lengths_km):
            if array is None:continue
            array.flags.writeable=False

    @classmethod
    def from_state(cls,s,*,face_ids,hinge_normals,dip_degrees,
                   finite_midpoints=None,finite_half_lengths_km=None,
                   coordinate_mode='along_slab_conveyor'):
        mesh=s.material_surface;ids=np.asarray(face_ids);saved=np.asarray(mesh['face_id'])
        if (ids.ndim!=1 or ids.dtype.kind not in 'iu' or len(np.unique(ids))!=len(ids)
                or len(np.unique(saved))!=len(saved)):
            raise ValueError('Frozen entry needs unique persistent material identities.')
        lookup={int(ident):i for i,ident in enumerate(saved)}
        if any(int(ident) not in lookup for ident in ids):
            raise ValueError('Frozen entry material identity is absent.')
        volume,mass=native_inventory(s)
        return cls(mesh['vertices'],mesh['faces'],volume,mass,
            getattr(s,'parcel_collision_sheet',saved),np.array([lookup[int(ident)] for ident in ids],int),
            hinge_normals,dip_degrees,radius_km=mesh.get('radius_km',6371.),
            finite_midpoints=finite_midpoints,finite_half_lengths_km=finite_half_lengths_km,
            coordinate_mode=coordinate_mode)

    def evaluate(self,points,faces,volumes,sheets,*,radius):
        if (not np.array_equal(faces,self.faces) or not np.array_equal(sheets,self.sheets)
                or np.shape(volumes)!=self.volumes.shape
                or not np.allclose(volumes,self.volumes,rtol=2e-13,atol=0.) or radius!=self.radius):
            raise ValueError('Frozen entry topology, inventory or radius changed during mechanics.')
        triangles=np.asarray(points)[self.faces[self.indices]]
        if np.any(np.abs(np.einsum('fvi,fi->fv',triangles,self.normals))>=1.-1e-12):
            raise EntryGeometryError('Entry trial reaches a hinge-coordinate pole.')
        result=evaluate(points,self.faces[self.indices],self.volumes[self.indices],self.masses[self.indices],
                        self.normals,self.dips,radius_km=radius,
                        finite_midpoints=self.finite_midpoints,
                        finite_half_lengths_km=self.finite_half_lengths_km,
                        coordinate_mode=self.coordinate_mode)
        if active_stack_overlap(points,self.faces,self.sheets,self.indices,self.normals,
                result['entered_reference_fraction'],radius_km=radius,
                finite_midpoints=self.finite_midpoints,
                finite_half_lengths_km=self.finite_half_lengths_km):
            raise EntryGeometryError('Active entry domain intersects a material stack; resolve oceanic subregions.')
        # Bound arithmetic cancellation by the signed components' magnitudes,
        # never by a possibly cancelling total potential alone.
        result['absolute_energy_j']=float(math.fsum(np.abs(result['face_energy_j'])))
        return result

    def validate_density(self,profile):
        fraction=np.zeros(len(self.volumes)) if profile is None else np.asarray(profile['dense_fraction'],float)
        if fraction.shape!=self.volumes.shape or not np.isfinite(fraction).all() or np.any((fraction<0.)|(fraction>1.)):
            raise ValueError('Entry and column phase profiles must describe the same conserved mass.')
        mass=self.volumes*1e9*(column_density.RHO_CRUST+
            (column_density.RHO_DENSE-column_density.RHO_CRUST)*fraction)
        if not np.allclose(mass,self.masses,rtol=2e-13,atol=0.):
            raise ValueError('Entry and column phase profiles describe different conserved mass.')


def positive_linear_average(values):
    """Exact mean positive part and its nodal derivatives on a reference triangle.

    The moving zero contour contributes no boundary term because its integrand
    is zero. Derivatives are integrals of barycentric shape functions over the
    positive region, so their sum is its material-area fraction.
    """
    q=np.asarray(values,float)
    if q.ndim!=2 or q.shape[1]!=3 or not np.isfinite(q).all():
        raise ValueError('Entry coordinates must be finite values on three material corners.')
    mean=np.zeros(len(q));weights=np.zeros_like(q)
    for row,v in enumerate(q):
        if np.all(v<=0.):continue
        if np.all(v>=0.):
            mean[row]=float(v.mean());weights[row]=1./3.;continue
        positive=v>0.
        if np.count_nonzero(positive)==1:
            i=int(np.argmax(v));j,k=[n for n in range(3) if n!=i]
            t=v[i]/(v[i]-v[j]);u=v[i]/(v[i]-v[k]);fraction=t*u
            weights[row,i]=fraction*(3.-t-u)/3.
            weights[row,j]=fraction*t/3.;weights[row,k]=fraction*u/3.
            mean[row]=v[i]*fraction/3.
        else:
            # Integrate the positive quadrilateral as two positive triangles.
            # Subtracting the negative triangle from the whole loses a thin
            # entered sliver when negative and positive distances differ greatly.
            i=int(np.argmin(v));j,k=[n for n in range(3) if n!=i]
            a=v[j]/(v[j]-v[i]);b=v[k]/(v[k]-v[i]);second=a*(1.-b)
            mean[row]=(b*(v[j]+v[k])+second*v[j])/3.
            weights[row,i]=(b*b+second*(a+b))/3.
            weights[row,j]=(b+second*(2.-a))/3.
            weights[row,k]=(b*(2.-b)+second*(1.-b))/3.
    return mean,weights


def reference_entry_fractions(triangles,normals,*,radius_km,
                              finite_midpoints=None,finite_half_lengths_km=None):
    """Geometric active area before reading mass, for stack exclusivity guards."""
    q=radius_km*1000.*np.arcsin(np.clip(np.einsum('fvi,fi->fv',triangles,normals),-1.,1.))
    fraction=positive_linear_average(q)[1].sum(axis=1)
    if finite_half_lengths_km is not None:
        for row in np.flatnonzero(np.isfinite(finite_half_lengths_km)):
            left,right,_,_=finite_entry_arc.endpoint_planes(normals[row],finite_midpoints[row],
                finite_half_lengths_km[row],radius_km)
            fraction[row]=finite_entry_arc.integrate(q[row],triangles[row]@left,triangles[row]@right)[4]
    return fraction


def entry_stack_overlap_ledger(points,faces,sheets,indices,normals,entry_fractions,*,radius_km,
                               finite_midpoints=None,finite_half_lengths_km=None):
    """Measure unresolved local entry/stack overlap without assigning energy.

    A physical combined law will need these same face pairs and subface areas
    to account for buoyancy and overburden once. Returning the geometry alone
    does not make an active intersection mechanically admissible.
    """
    return list(_entry_stack_intersections(points,faces,sheets,indices,normals,entry_fractions,
        radius_km=radius_km,finite_midpoints=finite_midpoints,
        finite_half_lengths_km=finite_half_lengths_km))


def ordered_entry_stack_depth_ledger(points,faces,sheets,indices,normals,dip_degrees,
                                     entry_fractions,order_graph,*,radius_km,
                                     finite_midpoints=None,finite_half_lengths_km=None,
                                     coordinate_mode='along_slab_conveyor'):
    """Report entered overlap's actual upper/lower identity and local depth.

    ``order_graph`` maps each upper sheet to its transitive lower sheets, as
    returned by ``collision_surface.descendants``. Depth at each intersection
    vertex uses the same projective reference-linear coordinate as the entry
    source, not an average over the whole face. Bounds are geometric evidence;
    they assign no basal interface, pressure, energy, or force.
    """
    if (not isinstance(coordinate_mode,str)
            or coordinate_mode not in ('along_slab_conveyor','horizontal_surface')):
        raise ValueError('Ordered entry depth needs a supported dip coordinate.')
    points=np.asarray(points,float);faces=np.asarray(faces)
    sheets=np.asarray(sheets);indices=np.asarray(indices)
    normals=np.asarray(normals,float);dip=np.asarray(dip_degrees,float)
    entry_fractions=np.asarray(entry_fractions,float)
    if (indices.ndim!=1 or indices.dtype.kind not in 'iu' or sheets.shape!=(len(faces),)
            or np.any(indices<0) or np.any(indices>=len(faces))
            or normals.shape!=(len(indices),3) or dip.shape!=(len(indices),)
            or entry_fractions.shape!=(len(indices),) or len(np.unique(indices))!=len(indices)
            or not np.isfinite(normals).all() or not np.isfinite(entry_fractions).all()
            or np.any((entry_fractions<0.)|(entry_fractions>1.+2e-12))
            or not np.isfinite(dip).all()
            or np.any((dip<=0.)|(dip>=90.))):
        raise ValueError('Ordered entry depth needs aligned selected faces, hinges and dips.')
    slot={int(face):local for local,face in enumerate(indices)}
    triangles=points[faces]
    result=[]
    for row in _entry_stack_intersections(points,faces,sheets,indices,normals,entry_fractions,
            radius_km=radius_km,finite_midpoints=finite_midpoints,
            finite_half_lengths_km=finite_half_lengths_km,include_polygon=True):
        entered,stack=row['entered_face'],row['stack_face']
        entered_sheet,stack_sheet=int(sheets[entered]),int(sheets[stack])
        if entered_sheet in order_graph.get(stack_sheet,()):
            upper,lower=stack,entered
        elif stack_sheet in order_graph.get(entered_sheet,()):
            upper,lower=entered,stack
        else:
            raise ValueError('Entered stack intersection has no persistent vertical order.')
        local=slot[entered]
        triangle=triangles[entered]
        q=radius_km*1000.*np.arcsin(np.clip(triangle@normals[local],-1.,1.))
        depth_factor=(np.sin(np.deg2rad(dip[local]))
                      if coordinate_mode=='along_slab_conveyor'
                      else np.tan(np.deg2rad(dip[local])))
        depths=np.array([entry_depth.point_depth(q,triangle,point)*depth_factor
                         for point in row['polygon']])
        result.append(dict(entered_face=entered,stack_face=stack,
                           upper_face=upper,lower_face=lower,
                           entered_is_lower=entered==lower,
                           area_km2=row['area_km2'],
                           minimum_entry_depth_m=float(depths.min()),
                           maximum_entry_depth_m=float(depths.max())))
    return result


def active_stack_overlap(points,faces,sheets,indices,normals,entry_fractions,*,radius_km,
                         finite_midpoints=None,finite_half_lengths_km=None):
    """Reject the first active entry/stack intersection until a shared law exists."""
    return next(_entry_stack_intersections(points,faces,sheets,indices,normals,entry_fractions,
        radius_km=radius_km,finite_midpoints=finite_midpoints,
        finite_half_lengths_km=finite_half_lengths_km),None) is not None


def evaluate(vertices,faces,volumes_km3,masses_kg,hinge_normals,dip_degrees,*,radius_km=6371.,
             finite_midpoints=None,finite_half_lengths_km=None,
             coordinate_mode='along_slab_conveyor'):
    """Return energy and exact derivatives of this reference-element model.

    Hinge normals point toward the overriding side and rotate with that plate.
    Their rotational reaction is retained; a common rotation does no work.
    Signed buoyancy is intentional: sufficiently dense material lowers its
    energy on entry. This is a thin sheet in uniform mantle, not a resolved
    flexural ramp, free surface, or pressure/temperature solution.
    """
    if (not isinstance(coordinate_mode, str)
            or coordinate_mode not in ('along_slab_conveyor', 'horizontal_surface')):
        raise ValueError('Entry energy needs an explicit supported dip coordinate.')
    p=np.asarray(vertices,float);f=np.asarray(faces)
    volume=np.asarray(volumes_km3,float);mass=np.asarray(masses_kg,float)
    normal=np.asarray(hinge_normals,float);dip=np.asarray(dip_degrees,float)
    if (p.ndim!=2 or p.shape[1]!=3 or not np.isfinite(p).all()
            or np.any(np.abs(np.linalg.norm(p,axis=1)-1.)>1e-10)
            or f.ndim!=2 or f.shape[1]!=3 or f.dtype.kind not in 'iu'
            or np.any(f<0) or np.any(f>=len(p))
            or volume.shape!=(len(f),) or mass.shape!=volume.shape or dip.shape!=volume.shape
            or normal.shape!=(len(f),3) or not np.isfinite(normal).all()
            or np.any(np.abs(np.linalg.norm(normal,axis=1)-1.)>1e-10)
            or not np.isfinite(volume).all() or np.any(volume<=0.)
            or not np.isfinite(mass).all() or np.any(mass<=0.)
            or not np.isfinite(dip).all() or np.any((dip<=0.)|(dip>=90.))
            or not np.isscalar(radius_km) or isinstance(radius_km,(bool,np.bool_))
            or not np.isfinite(radius_km) or radius_km<=0.):
        raise ValueError('Entry energy requires aligned positive inventories and valid spherical material/hinge geometry.')
    triangles=p[f]
    signed=np.einsum('ij,ij->i',triangles[:,0],np.cross(triangles[:,1]-triangles[:,0],triangles[:,2]-triangles[:,0]))
    if np.any(signed<=0.):raise ValueError('Entry energy needs outward nondegenerate material faces.')
    cosine=np.einsum('fvi,fi->fv',triangles,normal)
    if np.any(np.abs(cosine)>=1.-1e-12):
        raise ValueError('Entry domain reaches a hinge-coordinate pole; resolve a local domain instead.')
    radius_m=float(radius_km)*1000.
    distance=radius_m*np.arcsin(cosine)
    mean,weights=positive_linear_average(distance)
    validate_finite_front(normal,finite_midpoints,finite_half_lengths_km,radius_km)
    endpoint_left=np.zeros_like(distance);endpoint_right=np.zeros_like(distance)
    left_planes=np.zeros_like(normal);right_planes=np.zeros_like(normal)
    midpoint=np.zeros_like(normal);arc_sine=np.zeros(len(f));arc_cosine=np.zeros(len(f))
    fraction=weights.sum(axis=1)
    if finite_half_lengths_km is not None:
        lengths=np.asarray(finite_half_lengths_km,float)
        middle=np.asarray(finite_midpoints,float)
        for row in np.flatnonzero(np.isfinite(lengths)):
            left,right,sine_angle,cosine_angle=finite_entry_arc.endpoint_planes(
                normal[row],middle[row],lengths[row],radius_km)
            average,dq,dl,dr,part,_=finite_entry_arc.integrate(
                distance[row],triangles[row]@left,triangles[row]@right)
            mean[row]=average;weights[row]=dq
            endpoint_left[row]=dl;endpoint_right[row]=dr
            left_planes[row]=left;right_planes[row]=right
            midpoint[row]=middle[row];arc_sine[row]=sine_angle;arc_cosine[row]=cosine_angle
            fraction[row]=part
    depth_factor=(np.sin(np.deg2rad(dip)) if coordinate_mode=='along_slab_conveyor'
                  else np.tan(np.deg2rad(dip)))
    # Use the same SI-volume multiplication as native_inventory. Associating
    # rho*V*1e9 differently can leave a spurious load for exactly neutral mass.
    buoyancy_mass=column_density.RHO_MANTLE*(volume*1e9)-mass
    coefficient=9.81*buoyancy_mass*depth_factor
    energy=coefficient*mean
    factor=coefficient[:,None]*radius_m*weights/np.sqrt(1.-cosine*cosine)
    corner_gradient=factor[:,:,None]*normal[:,None,:]
    corner_gradient+=coefficient[:,None,None]*(endpoint_left[:,:,None]*left_planes[:,None,:]
                                               +endpoint_right[:,:,None]*right_planes[:,None,:])
    corner_gradient-=triangles*np.sum(corner_gradient*triangles,axis=2)[:,:,None]
    normal_gradient=np.sum(factor[:,:,None]*triangles,axis=1)
    normal_gradient+=coefficient[:,None]*arc_cosine[:,None]*np.sum(
        (endpoint_left-endpoint_right)[:,:,None]*np.cross(midpoint[:,None,:],triangles),axis=1)
    normal_gradient-=normal*np.sum(normal_gradient*normal,axis=1)[:,None]
    midpoint_gradient=coefficient[:,None]*(arc_sine[:,None]*np.sum(
        (endpoint_left+endpoint_right)[:,:,None]*triangles,axis=1)
        +arc_cosine[:,None]*np.sum((endpoint_left-endpoint_right)[:,:,None]*
                                    np.cross(triangles,normal[:,None,:]),axis=1))
    midpoint_gradient-=midpoint*np.sum(midpoint_gradient*midpoint,axis=1)[:,None]
    gradient=np.zeros_like(p)
    np.add.at(gradient,f.ravel(),corner_gradient.reshape(-1,3))
    entering_torque=np.sum(np.cross(triangles,corner_gradient),axis=1)
    hinge_torque=np.cross(normal,normal_gradient)+np.cross(midpoint,midpoint_gradient)
    if not all(np.isfinite(a).all() for a in (energy,gradient,normal_gradient,midpoint_gradient,entering_torque,hinge_torque)):
        raise ValueError('Entry energy exceeds the finite arithmetic domain.')
    return dict(energy_j=float(math.fsum(energy)),face_energy_j=energy,
        vertex_gradient_j=gradient,hinge_normal_gradient_j=normal_gradient,
        hinge_midpoint_gradient_j=midpoint_gradient,
        entering_potential_torque_n_m=entering_torque,hinge_potential_torque_n_m=hinge_torque,
        mean_entry_depth_m=mean*depth_factor,entered_reference_fraction=fraction,
        corner_entry_coordinate_m=distance,corner_entry_depth_m=distance*depth_factor[:,None],
        buoyancy_mass_kg=buoyancy_mass,volumes_km3=volume.copy(),masses_kg=mass.copy(),
        coordinate=('reference-linear great-circle conveyor distance; depth=distance*sin(dip)'
                    if coordinate_mode=='along_slab_conveyor' else
                    'horizontal great-circle distance; depth=distance*tan(dip); read-only candidate'))


def native_inventory(s):
    """Read mass from the existing ordinary/dense volumes; no new inventory."""
    mesh=s.material_surface
    areas=material_surface.spherical_face_areas(mesh['vertices'],mesh['faces'],mesh.get('radius_km',6371.))
    thickness=np.asarray(s.structure['thickness_km'],float)
    reference=np.asarray(s.mass,float);factor=np.asarray(s.structure['area_factor'],float)
    if (thickness.shape!=areas.shape or not np.isfinite(thickness).all() or np.any(thickness<=0.)
            or reference.shape!=areas.shape or factor.shape!=areas.shape
            or not np.isfinite(reference).all() or not np.isfinite(factor).all()
            or np.any(reference<=0.) or np.any(factor<=0.)
            or not np.allclose(areas,reference*factor,rtol=1e-8,atol=0.)):
        raise ValueError('Entry needs physical native crust columns.')
    volume=reference*factor*thickness
    options=column_density.options(s)
    fraction=options['density_profile']['dense_fraction'] if options else np.zeros(len(volume))
    density=column_density.RHO_CRUST+(column_density.RHO_DENSE-column_density.RHO_CRUST)*fraction
    return volume,volume*1e9*density


def _plate_drive(size, slots, incoming, overriding, entering_torque, hinge_torque, scale):
    """Map a frozen potential's reciprocal torques to solver coordinates.

    The incoming face and the overriding hinge act on their own plate slots.
    This calculation does not change a balance, so callers can validate its
    finite range before publishing any force or energy diagnostics.
    """
    drive=np.zeros(size)
    for plate,torque in zip(incoming,entering_torque):
        slot=slots[int(plate)];drive[3*slot:3*slot+3]-=torque*scale
    for plate,torque in zip(overriding,hinge_torque):
        slot=slots[int(plate)];drive[3*slot:3*slot+3]-=torque*scale
    return drive


def apply_to_balance(balance,*,face_ids,hinge_normals,dip_degrees,overriding_plate_uids,
                     finite_midpoints=None,finite_half_lengths_km=None):
    """Add frozen entry forces to an actual plate balance, with reciprocal torques.

    This explicit experimental call does not infer domain eligibility, mutate
    material or install a timestep policy. Face IDs must still exist and be
    unique. A removed/remeshed ID is rejected rather than silently reassigned.
    The same nodal gradient drives the explicit fixed-hinge sheet stage;
    connecting both components to native entry evolution remains required.
    """
    import plate_balance as pb
    if not isinstance(balance,pb.Balance) or getattr(balance,'resistance_version',0)!=1:
        raise ValueError('Entry forces require an actual balance with passive resistance version 1.')
    if getattr(balance,'_continental_entry_applied',False):
        raise ValueError('Entry energy has already been included in this balance.')
    s=balance.s;mesh=s.material_surface
    ids=np.asarray(face_ids);over=np.asarray(overriding_plate_uids)
    saved=np.asarray(mesh['face_id'])
    if (ids.ndim!=1 or ids.dtype.kind not in 'iu' or len(np.unique(ids))!=len(ids)
            or len(np.unique(saved))!=len(saved) or over.shape!=ids.shape or over.dtype.kind not in 'iu'):
        raise ValueError('Entry domains require unique persistent face IDs and aligned overriding identities.')
    lookup={int(ident):i for i,ident in enumerate(saved)}
    if any(int(ident) not in lookup for ident in ids):
        raise ValueError('Entry material identity is absent; update its domain through the real material lifecycle.')
    indices=np.array([lookup[int(ident)] for ident in ids],int)
    validate_finite_front(hinge_normals,finite_midpoints,finite_half_lengths_km,
        mesh.get('radius_km',6371.))
    incoming=np.asarray(s.parcel_plate)[indices]
    slots={int(s.plate_uid[p]):p for p in balance.plates}
    if any(int(uid) not in slots for uid in over) or any(int(p) not in balance.slot for p in incoming):
        raise ValueError('Entry domain refers to an inactive plate.')
    overriding=np.array([slots[int(uid)] for uid in over],int)
    if np.any(incoming==overriding):raise ValueError('A plate cannot override its own entry domain.')
    sheets=np.asarray(getattr(s,'parcel_collision_sheet',mesh['face_id']))
    radius=mesh.get('radius_km',6371.)
    fractions=reference_entry_fractions(
        np.asarray(mesh['vertices'])[np.asarray(mesh['faces'])[indices]],np.asarray(hinge_normals),
        radius_km=radius,finite_midpoints=finite_midpoints,
        finite_half_lengths_km=finite_half_lengths_km)
    if active_stack_overlap(mesh['vertices'],mesh['faces'],sheets,indices,np.asarray(hinge_normals),
            fractions,radius_km=radius,finite_midpoints=finite_midpoints,
            finite_half_lengths_km=finite_half_lengths_km):
        raise ValueError('Active entry domain intersects a material stack; resolve its oceanic subregions before adding ramp energy.')
    volume,mass=native_inventory(s)
    result=evaluate(mesh['vertices'],np.asarray(mesh['faces'])[indices],volume[indices],mass[indices],
                    hinge_normals,dip_degrees,radius_km=radius,
                    finite_midpoints=finite_midpoints,finite_half_lengths_km=finite_half_lengths_km)
    if not np.array_equal(result['entered_reference_fraction']>0.,fractions>0.):
        raise ValueError('Entry active area changed between geometry guard and force assembly.')
    drive=_plate_drive(balance.size,balance.slot,incoming,overriding,
        result['entering_potential_torque_n_m'],result['hinge_potential_torque_n_m'],
        pb.CM_YR_M_S/pb.RADIUS_M)
    combined=balance.torque+drive
    if not np.isfinite(drive).all() or not np.isfinite(combined).all():
        raise ValueError('Entry torque exceeds the finite force range.')
    balance.torque=combined;balance.drivers['continental_entry']=drive
    balance.notes['continental_entry_energy_j']=result['energy_j']
    balance.notes['continental_entry_scope']='explicit frozen thin-sheet domains; lifecycle/polarity coupling incomplete'
    balance._continental_entry_applied=True
    result.update(face_ids=ids.copy(),face_indices=indices,drive=drive)
    return result
