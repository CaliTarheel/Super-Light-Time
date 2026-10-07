"""Conservative artwork ownership and its finite initial physical interfaces.

The imported closed partition is authoritative until the first physical
transport. Afterwards the existing finite-volume ocean representation evolves
its own interfaces. This module neither moves material nor alters the control
contact graph. Zero-length graph contacts have no fabricated physical front.
"""
from __future__ import annotations

import numpy as np
import mesh_coverage

VERSION = 1


def _unit(value):
    value = np.asarray(value, float)
    return value / np.maximum(np.linalg.norm(value, axis=-1, keepdims=True), 1e-30)


def active(s):
    return (getattr(s, 'native_initial_ownership_version', 0) == VERSION
            and not getattr(s, 'native_initial_ownership_consumed', True))


def initialize(s, fitted):
    """Integrate every initial owner, including water, onto control areas."""
    owners = np.asarray(fitted['face_owner'])
    hits = mesh_coverage.intersections(s.native_mesh, fitted['vertices'], fitted['faces'],
                                       locator=s.native_locator)
    support = np.zeros_like(s.support, dtype=float)
    np.add.at(support, (owners[hits['material_index']], hits['control_index']), hits['area_km2'])
    support /= s.cell_area
    expected = np.bincount(owners, weights=fitted['area_km2'], minlength=len(support))
    actual = support @ s.cell_area
    partition_error = float(np.max(np.abs(support.sum(axis=0)-1.)))
    owner_error = float(np.max(np.abs(actual-expected)))
    if partition_error > 1e-8 or owner_error > max(1e-4, float(s.cell_area.sum())*1e-11):
        raise ValueError('Initial owner partition does not conservatively cover the control sphere.')
    # Keep roundoff-scale partition error measurable rather than introducing
    # an owner-changing normalization. Transport already accepts these sums.
    s.support = support
    s.native_initial_ownership_version = VERSION
    s.native_initial_ownership_consumed = False
    s.native_initial_owner_interfaces = interfaces(fitted)
    s.initial_geometry_diagnostics.update(
        initial_owner_area_km2=expected.tolist(),
        initial_owner_area_max_error_km2=owner_error,
        initial_owner_partition_max_error=partition_error,
        initial_owner_geometry_scope='Fixed imported partition through first physical transport; evolved ocean interfaces use finite-volume support.')
    return hits


def interfaces(fitted):
    """Extract each genuine owner interface once from the closed partition."""
    vertices = np.asarray(fitted['vertices'], float)
    faces = np.asarray(fitted['faces'])
    owners = np.asarray(fitted['face_owner'])
    kinds = np.asarray(fitted['face_kind'])
    edges = np.sort(faces[:, [[0, 1], [1, 2], [2, 0]]], axis=2).reshape(-1, 2)
    unique, inverse, counts = np.unique(edges, axis=0, return_inverse=True, return_counts=True)
    if np.any(counts != 2):
        raise ValueError('Initial ownership requires a closed conforming source partition.')
    adjacent = np.argsort(inverse, kind='stable').reshape(-1, 2)//3
    left, right = adjacent.T
    changed = owners[left] != owners[right]
    left, right = left[changed], right[changed]
    start, end = vertices[unique[changed]].transpose(1, 0, 2)
    normal = _unit(np.cross(start, end))
    centres = _unit(vertices[faces].sum(axis=1))
    normal *= np.where(np.sum(normal*(centres[right]-centres[left]), axis=1) >= 0., 1., -1.)[:, None]
    length = 6371.*np.arctan2(np.linalg.norm(np.cross(start, end), axis=1), np.sum(start*end, axis=1))
    return dict(start=start, end=end, normal=normal, length_km=length,
                owner_a=owners[left].copy(), owner_b=owners[right].copy(),
                kind_a=kinds[left].copy(), kind_b=kinds[right].copy(),
                ocean=(kinds[left] == 0) & (kinds[right] == 0))


def reconstruct(s, edge_indices):
    """Assign exact source segments once to the unchanged same-owner graph.

    A segment is assigned to its nearest same-pair graph contact. This changes
    only the force quadrature location/length; source endpoints remain exact.
    Entirely unresolved source pairs are explicit missing geometry, never raw
    edges or remote different-owner substitutes.
    """
    mesh = s.native_mesh
    edge_indices = np.asarray(edge_indices, np.int64)
    a, b = np.asarray(mesh['edge_faces'])[edge_indices].T
    p, q = s.plate[a], s.plate[b]
    count = len(a)
    raw_mid = np.asarray(mesh['edge_mid'])[edge_indices]
    mid = raw_mid.copy()
    normal = np.asarray(mesh['edge_normal'])[edge_indices].copy()
    lengths = np.zeros(count)
    sum_mid = np.zeros((count, 3)); sum_normal = np.zeros((count, 3))
    source = s.native_initial_owner_interfaces
    source_mid = _unit(source['start']+source['end'])
    parents = np.full(len(source_mid), -1, np.int32)
    oriented = np.zeros_like(source_mid)
    distance = np.zeros(len(source_mid))
    pairs = np.sort(np.column_stack((source['owner_a'], source['owner_b'])), axis=1)
    graph_pairs = np.sort(np.column_stack((p, q)), axis=1)
    for pp, qq in np.unique(pairs, axis=0):
        selected = np.flatnonzero((pairs[:, 0] == pp) & (pairs[:, 1] == qq))
        contacts = np.flatnonzero((graph_pairs[:, 0] == pp) & (graph_pairs[:, 1] == qq))
        if not len(contacts):
            continue
        nearest = np.empty(len(selected), np.int32)
        d = np.empty(len(selected))
        # Bound working memory without adding an optional SciPy dependency.
        for first in range(0, len(selected), 128):
            take = selected[first:first+128]
            distance2 = np.sum((source_mid[take, None]-raw_mid[contacts][None])**2, axis=2)
            chosen = distance2.argmin(axis=1)
            nearest[first:first+len(take)] = chosen
            d[first:first+len(take)] = np.sqrt(distance2[np.arange(len(take)), chosen])
        parent = contacts[nearest]
        parents[selected] = parent
        distance[selected] = 2.*6371.*np.arcsin(np.clip(d/2., 0., 1.))
        sign = np.where(p[parent] == source['owner_a'][selected], 1., -1.)
        oriented[selected] = source['normal'][selected]*sign[:, None]
    mapped = parents >= 0
    # A single control contact can receive ocean/ocean and coastal source
    # pieces, or coast pieces of opposite local polarity. Keep those finite
    # material fronts separate instead of assigning their summed length to
    # one inherited slab. Control connectivity and source geometry stay exact.
    extra_edges = []
    if 'kind_a' in source and 'kind_b' in source:
        ids = np.flatnonzero(mapped)
        forward = p[parents[ids]] == source['owner_a'][ids]
        water_p = np.where(forward, source['kind_a'][ids], source['kind_b'][ids]) == 0
        water_q = np.where(forward, source['kind_b'][ids], source['kind_a'][ids]) == 0
        kind = water_p.astype(np.int8) + 2*water_q.astype(np.int8)
        for contact in np.unique(parents[ids]):
            local = ids[parents[ids] == contact]
            categories = kind[parents[ids] == contact]
            for category in np.unique(categories)[1:]:
                new_contact = count + len(extra_edges)
                extra_edges.append(edge_indices[contact])
                parents[local[categories == category]] = new_contact
        if extra_edges:
            edge_indices = np.concatenate((edge_indices, np.asarray(extra_edges, np.int64)))
            count = len(edge_indices)
            raw_mid = np.asarray(mesh['edge_mid'])[edge_indices]
            mid = raw_mid.copy()
            normal = np.asarray(mesh['edge_normal'])[edge_indices].copy()
            lengths = np.zeros(count)
            sum_mid = np.zeros((count, 3)); sum_normal = np.zeros((count, 3))
    assigned = parents[mapped]
    weights = source['length_km'][mapped]
    np.add.at(lengths, assigned, weights)
    np.add.at(sum_mid, assigned, source_mid[mapped]*weights[:, None])
    np.add.at(sum_normal, assigned, oriented[mapped]*weights[:, None])
    represented = lengths > 1e-10
    mid[represented] = _unit(sum_mid[represented])
    tangent = sum_normal-mid*np.sum(sum_normal*mid, axis=1)[:, None]
    degenerate = represented & (np.linalg.norm(tangent, axis=1) < 1e-10*lengths)
    if np.any(degenerate):
        raise ValueError('Initial physical fronts cancel within one control contact; refine the control graph.')
    normal[represented] = _unit(tangent[represented])
    return dict(midpoints=mid, normals=normal, lengths=lengths, refined=represented,
                graph_edge_indices=edge_indices.copy(),
                segments_start=source['start'][mapped].copy(), segments_end=source['end'][mapped].copy(),
                segment_normals=oriented[mapped], contact_index=assigned,
                diagnostics=dict(representation='exact fixed initial owner partition',
                    initial_ownership_version=VERSION, edges=count, refined_edges=int(represented.sum()),
                    material_front_split_contacts=len(extra_edges),
                    zero_geometry_edges=int((~represented).sum()), fallback_edges=0,
                    contour_segments=int(mapped.sum()), unmapped_segments=int((~mapped).sum()),
                    unmapped_length_km=float(source['length_km'][~mapped].sum()),
                    raw_length=float(np.asarray(mesh['edge_length'])[edge_indices].sum()),
                    corrected_length=float(lengths.sum()), source_length_km=float(source['length_km'].sum()),
                    maximum_graph_distance_km=float(distance.max(initial=0.))))


def validate_frame(frame):
    """Validate the small saved initialization contract without re-clipping."""
    version=frame.get('native_initial_ownership_version',0)
    mesh_diagnostics=frame.get('mesh_diagnostics',{})
    initial=mesh_diagnostics.get('initial_geometry',{}) if isinstance(mesh_diagnostics,dict) else {}
    boundary=frame.get('boundary_geometry_diagnostics',{})
    marked=('native_initial_ownership_active' in frame
            or isinstance(initial,dict) and 'initial_owner_area_km2' in initial
            or isinstance(boundary,dict) and 'initial_ownership_version' in boundary)
    if version==0 and not isinstance(version,(bool,np.bool_)):
        if marked:raise ValueError('Initial owner geometry metadata requires its version.')
        return
    integer=lambda x:isinstance(x,(int,np.integer)) and not isinstance(x,(bool,np.bool_))
    if not integer(version) or version!=VERSION:
        raise ValueError('Unsupported initial owner geometry version.')
    is_active=frame.get('native_initial_ownership_active')
    if not isinstance(is_active,(bool,np.bool_)) or not isinstance(initial,dict) or not isinstance(boundary,dict):
        raise ValueError('Initial owner geometry needs a boolean state and diagnostic objects.')
    def number(value,name):
        if (isinstance(value,(bool,np.bool_)) or not isinstance(value,(int,float,np.number))
                or not np.isfinite(value) or value<0):
            raise ValueError('Invalid initial ownership '+name+'.')
        return float(value)
    try:
        owners=np.asarray(initial['initial_owner_area_km2'],float)
        areas=np.asarray(frame['mesh_area_km2'],float)
    except (KeyError,TypeError,ValueError) as exc:
        raise ValueError('Initial ownership requires recorded owner and control areas.') from exc
    if (owners.ndim!=1 or not len(owners) or areas.ndim!=1 or not len(areas)
            or not np.isfinite(owners).all() or not np.isfinite(areas).all()
            or np.any(owners<0) or np.any(areas<=0)):
        raise ValueError('Initial ownership requires finite nonnegative owner areas and positive controls.')
    tolerance=max(1e-4,float(areas.sum())*1e-11)
    if abs(float(owners.sum()-areas.sum()))>tolerance:
        raise ValueError('Initial owner areas do not cover the saved control sphere.')
    if number(initial.get('initial_owner_area_max_error_km2'),'area error')>tolerance:
        raise ValueError('Initial owner projection exceeds its area error tolerance.')
    if number(initial.get('initial_owner_partition_max_error'),'partition error')>1e-8:
        raise ValueError('Initial owner fractions do not partition their controls.')
    if not isinstance(initial.get('initial_owner_geometry_scope'),str) or not initial['initial_owner_geometry_scope']:
        raise ValueError('Initial owner geometry requires its limited scope.')
    epoch=number(frame.get('time_myr'),'epoch')
    if is_active and epoch!=0.:
        raise ValueError('The fixed initial owner geometry cannot remain active after time zero.')
    if not is_active:
        if 'initial_ownership_version' in boundary:
            raise ValueError('Consumed initial ownership retains stale fixed boundary geometry.')
        return
    if (not integer(boundary.get('initial_ownership_version'))
            or boundary['initial_ownership_version']!=VERSION
            or boundary.get('representation')!='exact fixed initial owner partition'):
        raise ValueError('Active initial ownership requires its exact boundary geometry record.')
    counts={}
    for name in ('edges','refined_edges','zero_geometry_edges','fallback_edges','contour_segments','unmapped_segments'):
        value=boundary.get(name)
        if not integer(value) or value<0:raise ValueError('Invalid initial boundary '+name+'.')
        counts[name]=int(value)
    codes=np.asarray(frame.get('native_boundary_code',[]))
    speeds=np.asarray(frame.get('native_boundary_normal_speed_km_myr',[]),float)
    if (codes.shape!=(counts['edges'],) or codes.dtype.kind not in 'iu'
            or speeds.shape!=codes.shape or not np.isfinite(speeds).all()
            or counts['refined_edges']+counts['zero_geometry_edges']!=len(codes)
            or counts['fallback_edges']!=0
            or int(np.count_nonzero(codes==0))!=counts['zero_geometry_edges']
            or np.any(speeds[codes==0]!=0.)):
        raise ValueError('Initial control contacts disagree with their represented physical fronts.')
    values={name:number(boundary.get(name),'boundary '+name) for name in
            ('unmapped_length_km','raw_length','corrected_length','source_length_km','maximum_graph_distance_km')}
    tolerance=max(1e-5,values['source_length_km']*1e-10)
    if abs(values['corrected_length']+values['unmapped_length_km']-values['source_length_km'])>tolerance:
        raise ValueError('Initial boundary length is not partitioned into mapped and unresolved geometry.')
    rows=frame.get('boundary_segments')
    if not isinstance(rows,list):raise ValueError('Initial owner geometry requires finite boundary segments.')
    if (not counts['refined_edges']<=counts['contour_segments']<=len(rows)
            or (counts['unmapped_segments']==0)!=(values['unmapped_length_km']==0.)):
        raise ValueError('Initial source segment counts disagree with represented and unresolved lengths.')
    lengths=[];represented=set()
    for row in rows:
        if not isinstance(row,dict):raise ValueError('Initial boundary segment must be an object.')
        parent=row.get('contact_index')
        try:points=np.asarray(row['geometry_xyz'],float)
        except (KeyError,TypeError,ValueError) as exc:raise ValueError('Invalid initial boundary endpoints.') from exc
        if (not integer(parent) or not 0<=parent<len(codes) or codes[parent]==0
                or points.shape!=(2,3) or not np.isfinite(points).all()
                or np.any(abs(np.linalg.norm(points,axis=1)-1.)>1e-8)):
            raise ValueError('Invalid initial boundary segment or empty parent contact.')
        lengths.append(6371.*np.arctan2(np.linalg.norm(np.cross(*points)),float(points[0]@points[1])))
        represented.add(int(parent))
    if abs(float(np.sum(lengths))-values['corrected_length'])>tolerance:
        raise ValueError('Saved initial segments disagree with mapped physical source length.')
    if len(represented)!=counts['refined_edges']:
        raise ValueError('Initial physical segments do not cover every represented contact.')
