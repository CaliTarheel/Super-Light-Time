"""Export saved spherical histories as goSPL meshes and geometric forcing.

This adapter does not rerun tectonics. Geometric vertical forcing follows the
saved surface after plate motion and adds back an estimate of recorded relief
relaxation. Its component fields and coverage diagnostics remain inspectable.
"""
from __future__ import annotations

import ast
import hashlib
import json
import math
from pathlib import Path
import orientation as globe_orientation
import ridge_interaction as ridge_windows
import native_frame_sampling
import material_reconstruction
import native_boundary_geometry
import mesh_geometry
import mesh_transport
import material_transport
import arc_surface
import re
import zipfile

import numpy as np

RADIUS_M = 6_371_000.0
VERSION = 1
CHUNK = 65_536
EXPORTER_SOURCE = Path(__file__).read_bytes()
VALIDATOR_SOURCE = (Path(__file__).parent/'validate_gospl.py').read_bytes()
ORIENTATION_SOURCE = Path(globe_orientation.__file__).read_bytes()
RIDGE_INTERACTION_SOURCE = Path(ridge_windows.__file__).read_bytes()


def capture_native_sampling_sources(root=None):
    """Freeze every local helper imported by the native sampler, recursively."""
    root=Path(root) if root is not None else Path(native_frame_sampling.__file__).parent
    captured={};pending=['native_frame_sampling.py']
    while pending:
        name=pending.pop()
        if name in captured:continue
        path=root/name
        if not path.is_file():continue
        payload=path.read_bytes();captured[name]=payload
        for node in ast.walk(ast.parse(payload)):
            modules=([item.name for item in node.names] if isinstance(node,ast.Import)
                     else [node.module] if isinstance(node,ast.ImportFrom) and node.level==0 and node.module else [])
            pending.extend(module.split('.')[0]+'.py' for module in modules)
    return dict(sorted(captured.items()))


NATIVE_SAMPLING_SOURCES = capture_native_sampling_sources()


class ExportCancelled(Exception):
    pass


def check_cancel(cancel):
    if cancel and cancel():
        raise ExportCancelled('goSPL export cancelled.')


def icosphere(level, cancel=None):
    """Closed, outward-oriented spherical triangles with no polar singularity."""
    if isinstance(level, bool) or int(level) != level or not 0 <= level <= 9:
        raise ValueError('Mesh subdivision must be an integer from 0 to 9.')
    phi = (1 + math.sqrt(5)) / 2
    vertices = np.array([[-1,phi,0],[1,phi,0],[-1,-phi,0],[1,-phi,0],
                         [0,-1,phi],[0,1,phi],[0,-1,-phi],[0,1,-phi],
                         [phi,0,-1],[phi,0,1],[-phi,0,-1],[-phi,0,1]], float)
    vertices /= np.linalg.norm(vertices, axis=1)[:, None]
    faces = np.array([[0,11,5],[0,5,1],[0,1,7],[0,7,10],[0,10,11],
                      [1,5,9],[5,11,4],[11,10,2],[10,7,6],[7,1,8],
                      [3,9,4],[3,4,2],[3,2,6],[3,6,8],[3,8,9],
                      [4,9,5],[2,4,11],[6,2,10],[8,6,7],[9,8,1]], np.int32)
    for _ in range(int(level)):
        check_cancel(cancel)
        count, nfaces = len(vertices), len(faces)
        a, b, c = faces.T
        edges = np.concatenate((np.column_stack((a,b)), np.column_stack((b,c)), np.column_stack((c,a))))
        edges.sort(axis=1)
        keys = edges[:,0].astype(np.int64)*count + edges[:,1]
        del edges
        unique, inverse = np.unique(keys, return_inverse=True)
        del keys
        middle = vertices[unique//count] + vertices[unique%count]
        middle /= np.linalg.norm(middle, axis=1)[:,None]
        ab, bc, ca = (inverse.reshape(3,nfaces)+count).astype(np.int32)
        vertices = np.concatenate((vertices,middle))
        faces = np.concatenate((np.column_stack((a,ab,ca)), np.column_stack((b,bc,ab)),
                                np.column_stack((c,ca,bc)), np.column_stack((ab,bc,ca))))
    return vertices, faces


def cells_at(points, width, height):
    lon = np.arctan2(points[:,1], points[:,0])
    lat = np.arcsin(np.clip(points[:,2],-1,1))
    x = np.floor((lon+np.pi)*width/(2*np.pi)).astype(int)%width
    y = np.clip(np.floor((np.pi/2-lat)*height/np.pi).astype(int),0,height-1)
    return y*width+x


def sample(field, points):
    """Bilinear, seam-wrapped sampling with unique values at both poles."""
    field = np.asarray(field)
    h,w = field.shape
    x = (np.arctan2(points[:,1],points[:,0])+np.pi)*w/(2*np.pi)-.5
    y = (np.pi/2-np.arcsin(np.clip(points[:,2],-1,1)))*h/np.pi-.5
    ix,iy = np.floor(x).astype(int),np.floor(y).astype(int)
    fx,fy = x-ix,y-iy
    values = []
    for row in (iy,iy+1):
        crossed = (row<0)|(row>=h)
        yy = np.where(row<0,-row-1,np.where(row>=h,2*h-row-1,row)).clip(0,h-1)
        xx = ix+crossed*(w//2)
        values.append(field[yy,xx%w]*(1-fx)+field[yy,(xx+1)%w]*fx)
    result = values[0]*(1-fy)+values[1]*fy
    north,south = np.clip(-2*y,0,1),np.clip(2*(y-h+1),0,1)
    return result*(1-north-south)+field[0].mean()*north+field[-1].mean()*south


def load_frame(run_path, index):
    stem = Path(run_path)/f'frame_{index:04d}'
    metadata_bytes = stem.with_suffix('.json').read_bytes()
    frame = json.loads(metadata_bytes)
    with np.load(stem.with_suffix('.npz'),allow_pickle=False) as archive:
        frame.update({key:archive[key] for key in archive.files})
    h,w = int(frame['height']),int(frame['width'])
    if w != 2*h:
        raise ValueError('goSPL export requires a global 2:1 history grid.')
    for key in ('elevation','plate','crust'):
        if frame[key].size != w*h or not np.isfinite(frame[key]).all():
            raise ValueError(f'Invalid saved {key} field in frame {index}.')
    return frame


def matched_traces(first, second):
    required = ('trace_id','trace_xyz','trace_plate_uid','trace_erosion_m')
    if any(key not in frame for frame in (first,second) for key in required):
        raise ValueError('Evolving goSPL export requires saved material markers and erosion counters; this older history has none.')
    _, a,b = np.intersect1d(first['trace_id'],second['trace_id'],return_indices=True)
    return a,b


def rotation(omega, duration):
    omega = np.asarray(omega,float)
    speed = np.linalg.norm(omega)
    if speed < 1e-15:
        return np.eye(3)
    axis = omega/speed
    x,y,z = axis
    cross = np.array([[0,-z,y],[z,0,-x],[-y,x,0]])
    angle = speed*duration
    # Returned matrix operates on row vectors.
    return (np.eye(3)*np.cos(angle)+(1-np.cos(angle))*np.outer(axis,axis)+np.sin(angle)*cross).T


def plate_rotations(first, second, a, b, config, *, recorded_only=False):
    """Recover rigid motion from material where possible; report every fallback."""
    dt = second['time_myr']-first['time_myr']
    last = {p['uid']:p for p in second['plates']}
    matrices,diagnostics = {},[]
    for plate in first['plates']:
        pid,uid = int(plate['id']),int(plate['uid'])
        if recorded_only:
            start = end = np.empty((0, 3))
        else:
            subset = first['trace_plate_uid'][a] == uid
            start,end = first['trace_xyz'][a[subset]],second['trace_xyz'][b[subset]]
        method = 'material_rigid_fit'
        residual = None
        matrix = None
        if len(start)>=3:
            u,s,v = np.linalg.svd(start.T@end)
            if s[1] > 1e-9:
                parity = np.eye(3);parity[2,2] = np.linalg.det(u@v)
                matrix = u@parity@v
                residual = float(np.sqrt(np.mean(np.sum((start@matrix-end)**2,axis=1)))*RADIUS_M)
                if residual > 1.0:
                    method = 'nonrigid_material_fit_approximation'
        if matrix is None:
            method = 'recorded_euler_approximation'
            initial = np.asarray(plate['angular_velocity'])
            final = np.asarray(last.get(uid,plate)['angular_velocity'])
            omega = final if dt <= config.get('dt_myr',2)+1e-7 else (initial+final)/2
            matrix = rotation(omega,dt)
        matrices[pid] = matrix
        diagnostics.append(dict(plate_id=pid,plate_uid=uid,method=method,
                                matched_markers=len(start),rigid_fit_rms_m=residual))
    return matrices,diagnostics


def recorded_thermal_grid(frame, cancel=None):
    """Reconstruct only the saved instantaneous slab-window surface overlay.

    Sampling this original source raster uses the same interpolation as the
    recorded elevation. Evaluating an analytic dome at a mesh vertex instead
    would subtract a different signal near grid-scale coasts and pulse edges.
    Legacy frames take the unchanged path without allocating a zero raster.
    """
    episodes = frame.get('ridge_episodes', [])
    if not episodes:
        return None
    width, height = int(frame['width']), int(frame['height'])
    owners = np.asarray(frame['plate']).reshape(-1)
    uid_by_id = {int(row['id']): int(row['uid']) for row in frame['plates']}
    result = np.empty(width*height, np.float32)
    for offset in range(0, len(result), CHUNK):
        check_cancel(cancel)
        cells = np.arange(offset, min(offset+CHUNK, len(result)))
        lon = (cells % width+.5)*2*np.pi/width-np.pi
        lat = np.pi/2-(cells//width+.5)*np.pi/height
        points = np.column_stack((np.cos(lat)*np.cos(lon), np.cos(lat)*np.sin(lon), np.sin(lat)))
        slots, inverse = np.unique(owners[cells], return_inverse=True)
        uids = np.array([uid_by_id[int(slot)] for slot in slots], np.int64)[inverse]
        result[cells] = ridge_windows.sample_ridge_effects(
            episodes, points, uids, frame['time_myr'])['thermal_support_m']
    return result.reshape(height, width)


def erosion_correction(first, second, a, b, config, thermal_grid=None):
    """Recorded signed relaxation loss on a bounded, owner-restricted grid.

    Missing support uses the model's documented exponential relaxation rate.
    It never derives erosion from differences between fixed geographic heights.
    """
    width = min(256,int(first['width']));height = width//2
    lon,lat = np.meshgrid((np.arange(width)+.5)*2*np.pi/width-np.pi,
                          np.pi/2-(np.arange(height)+.5)*np.pi/height)
    xyz = np.column_stack((np.cos(lat).ravel()*np.cos(lon).ravel(),
                           np.cos(lat).ravel()*np.sin(lon).ravel(),np.sin(lat).ravel()))
    source = cells_at(xyz,first['width'],first['height'])
    owners = first['plate'][source].reshape(height,width)
    crust = first['crust'][source].reshape(height,width)
    elevation = first['elevation'][source].reshape(height,width)
    uid_by_id = {int(p['id']):int(p['uid']) for p in first['plates']}
    uids = np.array([uid_by_id.get(int(p),-1) for p in owners.ravel()])
    trace_cells = cells_at(first['trace_xyz'][a],width,height)
    valid = (uids[trace_cells] == first['trace_plate_uid'][a]) & (crust.ravel()[trace_cells]>0)
    loss = second['trace_erosion_m'][b]-first['trace_erosion_m'][a]
    if not np.isfinite(loss).all():
        raise ValueError('Saved erosion counters contain non-finite values.')
    count = np.bincount(trace_cells[valid],minlength=width*height)
    sums = np.bincount(trace_cells[valid],weights=loss[valid],minlength=width*height)
    known = count.reshape(height,width)>0
    values = np.divide(sums,count,out=np.zeros_like(sums,dtype=float),where=count>0).reshape(height,width)
    # Propagate only through cells of the same moving plate and crust class.
    for _ in range(24):
        needed = ~known & (crust>0)
        if not needed.any():break
        total = np.zeros_like(values);weights=np.zeros_like(values)
        for dy,dx in ((0,1),(0,-1),(1,0),(-1,0)):
            def shift(field):
                result = np.roll(field,(dy,dx),(0,1))
                if dy==1:result[0]=np.roll(field[0],width//2)
                elif dy==-1:result[-1]=np.roll(field[-1],width//2)
                return result
            usable = shift(known)&(shift(owners)==owners)&(shift(crust)==crust)&needed
            total += shift(values)*usable;weights += usable
        added = weights>0
        if not added.any():break
        values[added]=total[added]/weights[added];known|=added
    dt = second['time_myr']-first['time_myr']
    # Source relief is estimated from its basement; overlap and roughness make
    # this fallback approximate. Keep that support flag in exported forcing.
    if thermal_grid is None:
        thermal_grid = recorded_thermal_grid(first)
    internal_elevation = elevation if thermal_grid is None else elevation-thermal_grid.ravel()[source].reshape(height,width)
    relief = np.clip(internal_elevation-220-np.where(crust==2,180,0)+np.where(crust==3,100,0),-900,7500)
    fallback = relief*(1-np.exp(-dt*config.get('erosion',1)/np.where(crust==2,350.,180.)))
    if first.get('structure_version'):
        if 'erosion_rate_m_myr' not in first:
            raise ValueError('Crustal-column history lacks its recorded net erosion-rate field.')
        # New columns rebound as they lose material. The saved NET rate omits
        # that paired erosion/rebound contribution together; using the legacy
        # decay rule here would invent forcing in unsupported marker regions.
        rate = np.asarray(first['erosion_rate_m_myr']).reshape(-1)
        if rate.size != int(first['width'])*int(first['height']) or not np.isfinite(rate).all() or np.any(rate < 0):
            raise ValueError('Saved net erosion-rate field is invalid.')
        fallback = rate[source].reshape(height,width)*dt
    values = np.where(known,values,fallback)
    values[crust==0]=0
    return values,known.astype(np.uint8),owners,crust


def interval_forcing(vertices, first, second, config, cancel=None, native_contexts=None):
    duration = float(second['time_myr']-first['time_myr'])
    if duration<=0:raise ValueError('History times must increase strictly.')
    if first.get('collision_surface_version', 0) != second.get('collision_surface_version', 0):
        raise ValueError('A forcing interval cannot mix collision surface versions.')
    if first.get('surface_erosion_version', 0) != second.get('surface_erosion_version', 0):
        raise ValueError('Cannot mix surface erosion policies in one export interval.')
    if first.get('collision_coast_version', 0) != second.get('collision_coast_version', 0):
        raise ValueError('A forcing interval cannot mix collision coast versions.')
    if native_frame_sampling.is_native(first) != native_frame_sampling.is_native(second):
        raise ValueError('A forcing interval cannot mix raster and native mesh histories.')
    if native_frame_sampling.is_native(first):
        return _native_interval_forcing(vertices, first, second, config, cancel, native_contexts)
    a,b = matched_traces(first,second)
    matrices,diagnostics = plate_rotations(first,second,a,b,config)
    thermal_grid = recorded_thermal_grid(first,cancel)
    correction,supported,corr_owner,corr_crust = erosion_correction(first,second,a,b,config,thermal_grid)
    years = duration*1_000_000
    fields = dict(hdisp=np.empty((len(vertices),3),np.float32),
                  upsub=np.empty(len(vertices),np.float32),
                  geometric_rate=np.empty(len(vertices),np.float32),
                  relaxation_correction=np.empty(len(vertices),np.float32),
                  correction_supported=np.empty(len(vertices),np.uint8))
    fallback_count=0;land_count=0
    for offset in range(0,len(vertices),CHUNK):
        check_cancel(cancel)
        section=slice(offset,offset+CHUNK);points=vertices[section]
        cells=cells_at(points,first['width'],first['height'])
        owners=first['plate'][cells];crust=first['crust'][cells]
        moved=np.empty_like(points)
        for pid in np.unique(owners):
            if int(pid) not in matrices:raise ValueError('A saved plate has no recorded motion metadata.')
            mask=owners==pid;moved[mask]=points[mask]@matrices[int(pid)]
        start_z=sample(first['elevation'].reshape(first['height'],first['width']),points)
        end_z=sample(second['elevation'].reshape(second['height'],second['width']),moved)
        corr_cells=cells_at(points,correction.shape[1],correction.shape[0])
        good=(corr_owner.ravel()[corr_cells]==owners)&(corr_crust.ravel()[corr_cells]==crust)
        loss=correction.ravel()[corr_cells].copy()
        internal_start_z=start_z if thermal_grid is None else start_z-sample(thermal_grid,points)
        fallback=np.clip(internal_start_z-220-np.where(crust==2,180,0)+np.where(crust==3,100,0),-900,7500)*(1-np.exp(-duration*config.get('erosion',1)/np.where(crust==2,350.,180.)))
        if first.get('structure_version'):
            fallback = np.asarray(first['erosion_rate_m_myr']).reshape(-1)[cells]*duration
        loss[~good]=fallback[~good];loss[crust==0]=0
        support=good&(supported.ravel()[corr_cells]>0)&(crust>0)
        land_count+=int((crust>0).sum());fallback_count+=int(((crust>0)&~support).sum())
        fields['hdisp'][section]=(moved-points)*RADIUS_M/years
        fields['geometric_rate'][section]=(end_z-start_z)/years
        fields['relaxation_correction'][section]=loss/years
        fields['upsub'][section]=(end_z-start_z+loss)/years
        fields['correction_supported'][section]=support
    return fields,dict(start_myr=first['time_myr'],end_myr=second['time_myr'],
                       motion=diagnostics,land_nodes=land_count,
                       relaxation_fallback_nodes=fallback_count,
                       erosion_convention='net denudation after rebound' if first.get('structure_version') else 'signed relief relaxation',
                       relaxation_supported_fraction=1-fallback_count/max(1,land_count))


def _native_interval_forcing(vertices, first, second, config, cancel, contexts):
    """Pull native surfaces back along recorded motion, without raster probes."""
    if contexts is None:
        contexts = (native_frame_sampling.prepare(first), native_frame_sampling.prepare(second))
    initial, final = contexts
    if initial['material_transport_version'] != final['material_transport_version']:
        raise ValueError('A forcing interval cannot mix material transport versions.')
    if initial['material_transport_version'] == 1:
        return _deforming_interval_forcing(vertices, first, second, config, cancel, contexts)
    duration = float(second['time_myr']-first['time_myr'])
    years = duration*1_000_000.
    a, b = matched_traces(first, second)
    matrices, diagnostics = plate_rotations(first, second, a, b, config)
    correction, supported = native_frame_sampling.material_erosion_correction(first, second, a, b, initial)
    correction = native_frame_sampling.prepare_material_correction(initial, correction, supported)
    fields = dict(hdisp=np.empty((len(vertices),3),np.float32),
                  upsub=np.empty(len(vertices),np.float32),
                  geometric_rate=np.empty(len(vertices),np.float32),
                  relaxation_correction=np.empty(len(vertices),np.float32),
                  correction_supported=np.empty(len(vertices),np.uint8))
    land_count = fallback_count = 0
    for offset in range(0, len(vertices), CHUNK):
        check_cancel(cancel)
        section = slice(offset, offset+CHUNK)
        points = vertices[section]
        start = native_frame_sampling.sample_frame(first, points, initial)
        moved = np.empty_like(points)
        for owner in np.unique(start['plate']):
            if int(owner) not in matrices:
                raise ValueError('A saved native plate has no recorded motion metadata.')
            selected = start['plate'] == owner
            moved[selected] = points[selected]@matrices[int(owner)]
        end = native_frame_sampling.sample_frame(second, moved, final)
        material = start['material_face']
        land = material >= 0
        loss, support = native_frame_sampling.sample_material_correction(initial, start, correction)
        difference = end['elevation']-start['elevation']
        fields['hdisp'][section] = (moved-points)*RADIUS_M/years
        fields['geometric_rate'][section] = difference/years
        fields['relaxation_correction'][section] = loss/years
        fields['upsub'][section] = (difference+loss)/years
        fields['correction_supported'][section] = support
        land_count += int(land.sum())
        fallback_count += int((land & ~support).sum())
    return fields, dict(start_myr=first['time_myr'], end_myr=second['time_myr'],
                        motion=diagnostics, land_nodes=land_count,
                        sampling='native spherical material containment and ocean reconstruction',
                        erosion_convention='net denudation after rebound',
                        relaxation_fallback_nodes=fallback_count,
                        relaxation_supported_fraction=1-fallback_count/max(1,land_count))


def _deforming_interval_forcing(vertices, first, second, config, cancel, contexts):
    """Exact material correspondence; column differences exclude remesh texture.

    Reconstructing continuous endpoint surfaces changes vertex averages when
    triangle topology changes. That representational difference is reported
    separately, never added to the physical column/erosion increment.
    """
    initial, final = contexts
    collision_version = initial.get('collision_surface_version', 0)
    if collision_version != final.get('collision_surface_version', 0):
        raise ValueError('A forcing interval cannot mix collision surface versions.')
    if initial.get('surface_erosion_version', 0) != final.get('surface_erosion_version', 0):
        raise ValueError('Cannot mix surface erosion policies in one export interval.')
    if initial.get('collision_coast_version', 0) != final.get('collision_coast_version', 0):
        raise ValueError('A forcing interval cannot mix collision coast versions.')
    years = float(second['time_myr']-first['time_myr'])*1_000_000.
    matrices, diagnostics = plate_rotations(first, second, None, None, config, recorded_only=True)
    fields = {name: np.empty(len(vertices), np.float32)
              for name in ('upsub', 'geometric_rate', 'relaxation_correction',
                           'display_geometric_rate', 'reconstruction_difference_rate')}
    fields['hdisp'] = np.empty((len(vertices), 3), np.float32)
    fields['correction_supported'] = np.empty(len(vertices), np.uint8)
    if collision_version == 1:
        fields['collision_support_rate'] = np.empty(len(vertices), np.float32)
    land_count = 0
    for offset in range(0, len(vertices), CHUNK):
        check_cancel(cancel)
        section = slice(offset, offset+CHUNK)
        points = vertices[section]
        start = native_frame_sampling.sample_frame(first, points, initial)
        source_face = start['material_face']
        land = source_face >= 0
        moved = np.empty_like(points)
        for owner in np.unique(start['plate'][~land]):
            selected = ~land & (start['plate'] == owner)
            moved[selected] = points[selected]@matrices[int(owner)]
        mapped = material_transport.map_hits(initial['material_transport'], final['material_transport'],
                                             source_face[land], start['material_weights'][land])
        moved[land] = mapped['xyz']
        end = native_frame_sampling.sample_frame(second, moved, final)
        display_difference = end['elevation']-start['elevation']
        difference = display_difference.copy()
        loss = np.zeros(len(points))
        collision_change = np.zeros(len(points)) if collision_version == 1 else None
        if land.any():
            before, after = source_face[land], mapped['face']
            uids = np.array([final['uids'][int(owner)] for owner in final['material_owner'][after]], np.int64)
            thermal = ridge_windows.sample_ridge_effects(second.get('ridge_episodes', []),
                                                         moved[land], uids, second['time_myr'])['thermal_support_m']
            difference[land] = (final['material_height_m'][after]-initial['material_height_m'][before]
                                + thermal-start['thermal_support_m'][land])
            if collision_version == 1:
                # Support is an explicit physical displacement, separate from
                # raw rock columns. Follow the same exact material descendant;
                # changing endpoint vertex averages must not manufacture uplift.
                collision_change[land] = (final['material_collision_support_m'][after]
                                           - initial['material_collision_support_m'][before])
                difference[land] += collision_change[land]
            loss[land] = (final['material_transport']['erosion_total_m'][after]
                          - initial['material_transport']['erosion_total_m'][before])
        fields['hdisp'][section] = (moved-points)*RADIUS_M/years
        fields['geometric_rate'][section] = difference/years
        fields['relaxation_correction'][section] = loss/years
        fields['upsub'][section] = (difference+loss)/years
        fields['display_geometric_rate'][section] = display_difference/years
        fields['reconstruction_difference_rate'][section] = (display_difference-difference)/years
        fields['correction_supported'][section] = land
        if collision_version == 1:
            # Diagnostic component of geometric_rate; never an extra upsub term.
            fields['collision_support_rate'][section] = collision_change/years
        land_count += int(land.sum())
    info = dict(start_myr=first['time_myr'], end_myr=second['time_myr'],
                        motion=diagnostics, material_transport_version=1, land_nodes=land_count,
                        material_motion='exact projective root-chart correspondence between saved triangular surfaces',
                        ocean_motion='recorded Euler finite rotation approximation',
                        sampling='native containment; material column change plus instantaneous thermal change',
                        erosion_convention='difference of cumulative material net denudation after rebound',
                        reconstruction_difference_rms_m=float(np.sqrt(np.mean(
                            np.square(fields['reconstruction_difference_rate'].astype(float)*years)))),
                        relaxation_fallback_nodes=0, relaxation_supported_fraction=1.)
    if collision_version == 1:
        info.update(collision_surface_version=1,
            sampling='native containment; material column, physical collision support and instantaneous thermal changes',
            collision_support_forcing='exact descendant-minus-source material_collision_support_m; included once in geometric_rate; zero for ocean',
            collision_support_rate_units='m/year',
            collision_support_change_rms_m=float(np.sqrt(np.mean(
                np.square(fields['collision_support_rate'].astype(float)*years)))))
    return fields, info


def checked_times(manifest, start_index, end_index, dt_years):
    frames=manifest.get('frames',[])
    for value in (start_index,end_index):
        if isinstance(value,bool) or int(value)!=value:raise ValueError('Select integer frame indices.')
    if not 0<=start_index<end_index<len(frames):
        raise ValueError('Choose at least two saved epochs in increasing order.')
    if isinstance(dt_years,bool) or not math.isfinite(float(dt_years)) or float(dt_years)!=int(dt_years) or dt_years<1:
        raise ValueError('goSPL time step must be a positive whole number of years.')
    times=np.array([float(row['time_myr'])*1_000_000 for row in frames[start_index:end_index+1]])
    if not np.isfinite(times).all() or np.any(np.diff(times)<=0) or np.any(np.abs(times-np.rint(times))>1e-3):
        raise ValueError('goSPL export needs strictly increasing epochs at whole-year times.')
    times=np.rint(times).astype(np.int64)
    offsets=times-times[0]
    if np.any(offsets%int(dt_years)):
        divisor=int(np.gcd.reduce(np.diff(times)))
        raise ValueError(f'The goSPL step must divide every saved interval exactly. Choose a divisor of {divisor:,} years (for example {min(divisor,1):,}).')
    return times


def write_json(path, value):
    Path(path).write_text(json.dumps(value,indent=2,allow_nan=False),encoding='utf-8')


def yaml_text(config):
    # JSON is also valid YAML. Explicit decimal scientific notation also works
    # with goSPL's YAML 1.1 loader, which otherwise treats some exponents as text.
    value=json.dumps(config,indent=2,allow_nan=False)
    return re.sub(r'(?<=: )(-?\d+)e([+-]\d+)(?=[,\n])',r'\1.0e\2',value)+'\n'


def source_fingerprint(run_path,index):
    result=dict(index=index,**{suffix[1:]+'_sha256':hashlib.sha256((Path(run_path)/f'frame_{index:04d}{suffix}').read_bytes()).hexdigest()
                              for suffix in ('.npz','.json')})
    metadata=json.loads((Path(run_path)/f'frame_{index:04d}.json').read_text(encoding='utf-8'))
    names=('mesh_version','surface_reconstruction_version','owner_reconstruction_version',
           'material_transport_version','arc_material_version','arc_surface_version',
           'continental_margin_version','continental_margin_parameters','continental_margin_revision',
           'collision_contact_version','collision_surface_version','collision_coast_version','surface_erosion_version')
    result['surface_reconstruction']={name:metadata[name] for name in names if name in metadata}
    return result


def build_history(run_path, output_path, manifest, start_index=0, end_index=None,
                  subdivisions=7, dt_years=100_000, rainfall_m_yr=1., progress=None,cancel=None,
                  orientation=None):
    """Write one immutable, native goSPL history package with bounded frames."""
    orientation=globe_orientation.normalize_orientation(orientation)
    rotation=globe_orientation.rotation_matrix(orientation)
    if end_index is None:end_index=len(manifest['frames'])-1
    times=checked_times(manifest,start_index,end_index,dt_years)
    rainfall_m_yr=float(rainfall_m_yr)
    if not math.isfinite(rainfall_m_yr) or not 0<=rainfall_m_yr<=20:
        raise ValueError('Rainfall must be between 0 and 20 metres per year.')
    output_path=Path(output_path);data_path=output_path/'input';data_path.mkdir(parents=True,exist_ok=True)
    source_path=output_path/'source';source_path.mkdir(exist_ok=True)
    update=lambda p,msg: progress(p,msg) if progress else None
    update(.01,'Building a closed spherical mesh')
    vertices,faces=icosphere(subdivisions,cancel)
    # Evaluate the original history in its original coordinates. Only the
    # output basis changes; fitting motion on regridded frames would blur it.
    source_vertices=vertices@rotation.T if any(orientation.values()) else vertices
    first=load_frame(run_path,start_index)
    first_context=native_frame_sampling.prepare(first) if native_frame_sampling.is_native(first) else None
    # Fail clearly for old histories before writing a deceptively usable file.
    if 'trace_erosion_m' not in first and first.get('material_transport_version', 0) != 1:
        raise ValueError('This history predates material erosion counters and cannot supply corrected evolving goSPL forcing.')
    initial_z=np.empty(len(vertices),np.float32)
    for offset in range(0,len(vertices),CHUNK):
        check_cancel(cancel);section=slice(offset,offset+CHUNK)
        initial_z[section]=(native_frame_sampling.sample_frame(first,source_vertices[section],first_context)['elevation']
                            if first_context is not None else
                            sample(first['elevation'].reshape(first['height'],first['width']),source_vertices[section]))
    np.savez_compressed(data_path/'mesh.npz',v=vertices*RADIUS_M,c=faces,z=initial_z)
    del faces,initial_z
    start_year=0;end_year=int(times[-1]-times[0]);intervals=[];diagnostics=[]
    fingerprints=[source_fingerprint(run_path,start_index)]
    count=end_index-start_index
    for number,index in enumerate(range(start_index,end_index)):
        check_cancel(cancel)
        update(.05+.85*number/count,f'Converting interval {number+1} of {count}')
        second=load_frame(run_path,index+1)
        second_context=native_frame_sampling.prepare(second) if native_frame_sampling.is_native(second) else None
        if abs(first['time_myr']*1e6-times[number])>.001 or abs(second['time_myr']*1e6-times[number+1])>.001:
            raise ValueError('Saved frame times do not match the captured history manifest.')
        fields,info=interval_forcing(source_vertices,first,second,manifest.get('config',{}),cancel,
                                     (first_context,second_context) if first_context is not None else None)
        if any(orientation.values()):
            fields['hdisp']=(fields['hdisp']@rotation).astype(np.float32)
        # Reference fields allow inspection against the subsequent goSPL output.
        fields['reference_z_end']=np.empty(len(vertices),np.float32)
        fields['reference_crust_end']=np.empty(len(vertices),np.uint8)
        fields['reference_boundary_end']=np.empty(len(vertices),np.uint8)
        for offset in range(0,len(vertices),CHUNK):
            check_cancel(cancel);section=slice(offset,offset+CHUNK);points=source_vertices[section]
            if second_context is not None:
                reference=native_frame_sampling.sample_frame(second,points,second_context)
                fields['reference_z_end'][section]=reference['elevation']
                fields['reference_crust_end'][section]=reference['crust']
                fields['reference_boundary_end'][section]=reference['boundary']
            else:
                cells=cells_at(points,second['width'],second['height'])
                fields['reference_z_end'][section]=sample(second['elevation'].reshape(second['height'],second['width']),points)
                fields['reference_crust_end'][section]=second['crust'][cells]
                fields['reference_boundary_end'][section]=second['boundary'][cells]
        fields['start_year']=np.array(int(times[number]-times[0]))
        fields['end_year']=np.array(int(times[number+1]-times[0]))
        name=f'input/forcing_{number:04d}'
        np.savez_compressed(output_path/(name+'.npz'),**fields)
        intervals.append(dict(start=int(times[number]-times[0]),end=int(times[number+1]-times[0]),
                              hdisp=[name,'hdisp'],upsub=[name,'upsub']))
        diagnostics.append(info);fingerprints.append(source_fingerprint(run_path,index+1))
        first=second
        first_context=second_context
        del fields
    output_interval=int(np.gcd.reduce(np.diff(times)))
    config=dict(name=f'Deep Time {manifest["run_id"]}: elapsed {times[0]/1e6:g} to {times[-1]/1e6:g} Myr',
                domain=dict(npdata=['input/mesh','v','c','z'],radius=RADIUS_M,flowdir=5,advect='interp',fast=False,seadepo=True),
                time=dict(start=start_year,end=end_year,dt=int(dt_years),tout=output_interval),
                spl=dict(K=3.0e-8,d=0.0,m=0.5,n=1.0,G=0.0),
                diffusion=dict(hillslopeKa=.02,hillslopeKm=.2,nonlinKm=100.,clinSlp=.00001),
                sea=dict(position=0.),climate=[dict(start=start_year,uniform=rainfall_m_yr)],
                tectonics=intervals,output=dict(dir='gospl-output',makedir=True))
    (output_path/'input.yml').write_text(yaml_text(config),encoding='utf-8')
    config['domain']['fast']=True;config['output']['dir']='gospl-forcing-check'
    (output_path/'forcing-check.yml').write_text(yaml_text(config),encoding='utf-8')
    metadata=dict(format='deep-time-gospl-history-v1',version=VERSION,run_id=manifest['run_id'],
                  orientation=orientation,orientation_matrix=rotation.tolist(),
                  orientation_convention='Original row-vector XYZ @ matrix = exported XYZ; yaw Z, then pitch Y, then roll X, degrees.',
                  orientation_sha256=hashlib.sha256(ORIENTATION_SOURCE).hexdigest(),
                  ridge_interaction_sha256=hashlib.sha256(RIDGE_INTERACTION_SOURCE).hexdigest(),
                  native_sampling_sources_sha256={name:hashlib.sha256(payload).hexdigest()
                                                  for name,payload in NATIVE_SAMPLING_SOURCES.items()},
                  source_sampling=('native material triangle containment and native ocean reconstruction'
                                   if first_context is not None else 'legacy spherical raster interpolation'),
                  start_index=start_index,end_index=end_index,start_myr=times[0]/1e6,end_myr=times[-1]/1e6,
                  time_convention='goSPL year zero is the first selected epoch; source times are elapsed Myr, not Earth geological ages.',
                  subdivisions=int(subdivisions),node_count=len(vertices),triangle_count=20*4**int(subdivisions),
                  radius_m=RADIUS_M,source_width=first['width'],source_height=first['height'],
                  units=dict(v='m',z='m relative to model sea level',hdisp='Cartesian chord m/year',upsub='m/year',time='years'),
                  forcing_method='geometric surface change pulled back along recovered plate motion, plus recorded or estimated Deep Time net surface-process loss',
                  erosion_convention='net denudation after rebound' if first.get('structure_version') else 'signed relief relaxation',
                  limitations=['Not a complete physical uplift budget or an exact reproduction of saved epochs.',
                               'Three-neighbor goSPL remapping and coastal sampling alter results.',
                               'Motion between saved epochs is approximate where markers move non-rigidly or an ocean plate lacks markers.',
                               'Erosion correction uses representative material counters; unsupported new column histories use the saved last-step net rate, while legacy histories use bounded relief decay.',
                               'Rainfall and surface-process coefficients are editable starting assumptions, not calibrated climate.',
                               'Mesh spacing does not increase the source tectonic resolution.'],
                  source_engine_sha256=manifest.get('engine_sha256'),source_auxiliary_sources_sha256=manifest.get('auxiliary_sources_sha256',{}),
                  exporter_sha256=hashlib.sha256(EXPORTER_SOURCE).hexdigest(),
                  validator_sha256=hashlib.sha256(VALIDATOR_SOURCE).hexdigest(),
                  source_frames=fingerprints,intervals=diagnostics,
                  documented_goSPL_version='v2026.7.14',solver_run_here=False)
    if first.get('material_transport_version', 0) == 1:
        metadata.update(material_transport_version=1,
                        forcing_method='exact material root-chart transport; raw column and thermal change plus cumulative net erosion change; recorded Euler ocean motion')
        metadata['limitations'][2:4] = [
            'Material endpoints follow exact saved root-chart correspondence; paths within a saved interval and ocean Euler rotations remain approximations.',
            'Material forcing uses raw column and cumulative net-erosion changes. Continuous reconstruction changes caused by remeshing or exposure are diagnostic only; they are not uplift.',
            'Coarsening requires compatible physical columns and projective geometry; absent material descendants make export fail explicitly.']
        if first.get('collision_surface_version', 0) == 1:
            metadata['surface_erosion_version'] = first.get('surface_erosion_version', 0)
            if first.get('collision_coast_version', 0):
                metadata['collision_coast_version']=first['collision_coast_version']
            metadata.update(collision_surface_version=1,
                forcing_method='exact material root-chart transport; raw column, physical collision support and thermal changes plus cumulative net erosion change; recorded Euler ocean motion',
                collision_support_forcing='destination-minus-source material_collision_support_m on corresponding faces; collision_support_rate is included once in geometric_rate, not added again to upsub')
            metadata['units']['collision_support_rate']='m/year (component of geometric_rate)'
            metadata['limitations'][3]='Material forcing uses raw columns, explicit collision support and cumulative net-erosion changes. Continuous reconstruction changes caused by remeshing or exposure are diagnostic only; they are not uplift.'
    write_json(output_path/'export_metadata.json',metadata)
    write_json(source_path/'manifest.json',manifest)
    (source_path/'frame-index.csv').write_text('source_frame,elapsed_myr,gospl_year\n'+''.join(
        f'{start_index+i},{year/1e6:g},{year-times[0]}\n' for i,year in enumerate(times)),encoding='utf-8')
    model_dir=source_path/'model';model_dir.mkdir(exist_ok=True)
    for name,digest in [('engine.py',manifest.get('engine_sha256')),*manifest.get('auxiliary_sources_sha256',{}).items()]:
        if not re.fullmatch(r'[A-Za-z_][A-Za-z0-9_]*\.py',name):raise ValueError('Invalid recorded source filename.')
        source=Path(run_path)/name
        if digest:
            payload=source.read_bytes()
            if hashlib.sha256(payload).hexdigest()!=digest:raise ValueError('A saved engine source fingerprint does not match.')
            (model_dir/name).write_bytes(payload)
    (output_path/'README.md').write_text(package_readme(metadata,dt_years,rainfall_m_yr),encoding='utf-8')
    (output_path/'validate_export.py').write_bytes(VALIDATOR_SOURCE)
    (source_path/'validate_gospl.py').write_bytes(VALIDATOR_SOURCE)
    (source_path/'gospl_export.py').write_bytes(EXPORTER_SOURCE)
    (source_path/'orientation.py').write_bytes(ORIENTATION_SOURCE)
    (source_path/'ridge_interaction.py').write_bytes(RIDGE_INTERACTION_SOURCE)
    for name,payload in NATIVE_SAMPLING_SOURCES.items():
        (source_path/name).write_bytes(payload)
    check_cancel(cancel);update(.90,'Checking and packaging goSPL files')
    from validate_gospl import validate_package
    validation=validate_package(output_path,check=lambda:check_cancel(cancel),
                                progress=lambda value:update(.90+.06*value,'Validating mesh and forcing intervals'))
    write_json(output_path/'validation.json',validation)
    files=sorted(p for p in output_path.rglob('*') if p.is_file() and p.suffix not in ('.zip','.tmp') and p.name!='job.json')
    temporary=output_path/'gospl-history.zip.tmp';target=output_path/'gospl-history.zip'
    try:
        with zipfile.ZipFile(temporary,'w',compression=zipfile.ZIP_DEFLATED,compresslevel=3) as archive:
            for i,path in enumerate(files):
                check_cancel(cancel)
                archive.write(path,path.relative_to(output_path).as_posix(),compress_type=zipfile.ZIP_STORED if path.suffix=='.npz' else zipfile.ZIP_DEFLATED)
                update(.96+.039*(i+1)/len(files),'Packaging goSPL files')
        check_cancel(cancel);temporary.replace(target)
    finally:
        temporary.unlink(missing_ok=True)
    update(1.,'goSPL history package ready')
    return metadata


def package_readme(meta,dt,rainfall):
    native = meta.get('source_sampling', '').startswith('native')
    deforming = meta.get('material_transport_version', 0) == 1
    collision = deforming and meta.get('collision_surface_version', 0) == 1
    column_terms = ('raw column height, explicit physical collision support, and instantaneous thermal support'
                    if collision else 'raw column height plus instantaneous thermal support')
    sampling_text = (
        'This native history is sampled from its connected moving material triangles and spherical ocean control mesh. '
        'Actual triangle containment selects exposed crust, its owner and column elevation, including islands absent from the review map. '
        'Ocean age and relief use area-weighted native reconstruction restricted to the selected owner; coastlines are not reconstructed from display pixels.'
        if native else 'This legacy history is sampled from its saved spherical rasters with longitude wrapping and polar interpolation.')
    correction_text = (
        'For this native history, matched marker counters are keyed directly to persistent material face IDs. '
        'They are averaged only within the same source face, with its initial owner UID checked. There is no spatial spreading across adjacent material.'
        if native else 'For this legacy history, the correction is interpolated on a grid no finer than 256×128, restricted to the same motion plate and crust class.')
    thermal_text = (
        'For native histories, instantaneous thermal support is evaluated once at each query point for the exposed owner; '
        'saved material heights and ocean age/relief are its baseline. Native centroid elevations already containing heat are not interpolated or added again.'
        if native else 'For legacy histories, temporary thermal elevation is reconstructed on the original source grid and excluded from the relief-decay estimate.')
    motion_text = (
        'For each interval, actual source triangle containment and its homogeneous root chart locate the same material in a destination leaf triangle. '
        'The destination vertices include recorded deformation, so this follows nonrigid material motion and changing triangulation without fitting a plate rotation. '
        'Ocean points use recorded Euler rotations; finite motion within each saved interval remains approximate. '
        f'On material, `geometric_rate` is the change in {column_terms}, divided by interval years. '
        'Continuous initial and endpoint terrain reconstruction is retained for terrain and review, but a change in shared-vertex averaging caused only by remeshing is not imposed as uplift. '
        '`display_geometric_rate` records the displayed endpoint change along the same path. '
        '`reconstruction_difference_rate` is display minus physical geometric rate (m/year), including changed exposure or clipping as well as reconstruction.'
        if deforming else
        'For each interval, the exporter first estimates the finite plate rotation. It fits matching material markers where possible, reports non-rigid residuals, '
        'and uses recorded Euler motion for untracked plates (principally ocean). It samples the next saved surface at the advected position and subtracts the old surface '
        'at the original position. This geometric vertical change carries rift subsidence, thermal bathymetry, arc emergence and collision relief, along with sampling/overlap effects.')
    if collision:
        motion_text += (' The saved `material_collision_support_m` remains separate from `material_height_m`. '
            'Its destination-minus-source value on the exact corresponding faces contributes once to `geometric_rate`; '
            '`collision_support_rate` records that component in m/year, with zero for ocean. '
            'It is not added again to `upsub`. Constant inherited support contributes zero forcing during pure remeshing. '
            'Intervals mixing collision surface versions are rejected.')
    correction_paragraph = (
        'To avoid imposing Deep Time surface erosion twice, material forcing adds the destination-minus-source change in `material_erosion_total_m`, '
        'the cumulative denudation minus rebound ledger, sampled on the exact corresponding descendant faces. '
        'Refinement inherits these physical columns and counters, so refinement alone contributes zero vertical forcing. '
        'No representative marker or last-step rate fallback is used for this material-transport version. Missing material roots or uncovered reference-chart regions fail export explicitly. '
        '`correction_supported` is 1 for exact material correspondence and 0 for ocean, which receives no continental erosion correction.'
        if deforming else
        "To reduce applying Deep Time's surface processes twice, it adds back the change in recorded material `trace_erosion_m`. "
        'For crustal-column histories (`structure_version: 1`), this is net denudation after rebound: gross removal and its associated rebound are removed together. '
        f'Neither diagnostic is added a second time. {correction_text} Unsupported areas in new histories use the saved last-step net erosion rate across the interval, '
        'an approximation reported by the support fields and per-interval counts. Legacy histories retain signed relief relaxation and their bounded exponential fallback '
        '(−900…7,500 m internal relief); negative legacy correction represents relaxation of negative relief. Ocean targets receive no continental erosion correction. '
        'Rift cooling and foreland deflection are already in the geometric surface change, not extra forcing terms.')
    return f'''# Deep Time → goSPL evolving history

Source experiment: `{meta['run_id']}`. Selected epochs: **{meta['start_myr']:g}–{meta['end_myr']:g} elapsed Myr**.
Mesh: **{meta['node_count']:,} vertices**, {meta['triangle_count']:,} triangles, radius 6,371,000 metres.

Unzip into a new folder. From that folder, with goSPL installed, run:

```sh
python validate_export.py
gospl -i forcing-check.yml
gospl -i input.yml
```

The first goSPL configuration uses its `fast` mode to inspect the prescribed tectonics without surface processes. The second enables erosion and deposition. Both create numbered output folders (`makedir: true`). This package was validated against the documented goSPL v2026.7.14 input schema; the goSPL solver itself was not run by the exporter. Installing/running goSPL is separate from this NumPy-based exporter. Large meshes and long histories can require substantial compute and memory.

`input.yml` and `forcing-check.yml` use JSON syntax, which is valid YAML. Edit the **{dt:,}-year** goSPL step, **{rainfall:g} m/year** uniform rainfall, erodibility and diffusion parameters for your landscape experiment. Rainfall and coefficients are starting assumptions, not a climate reconstruction. The step must be a whole number of years dividing every tectonic interval exactly; goSPL's `interp` advection runs at interval ends. The original 2 Myr tectonic integration is not changed by a smaller goSPL surface-process step.

## Files and units

- `input/mesh.npz`: `v` (N,3) fixed Cartesian coordinates in metres; `c` (F,3) zero-based outward triangle indices; `z` (N,) initial elevation in metres relative to model sea level. Coordinates follow X = 0° longitude, Y = 90° E, Z = north pole. There are no duplicate seam or pole vertices.
- `input/forcing_####.npz`: `hdisp` (N,3) Cartesian **chord rate in m/year**, and `upsub` (N,) vertical rate in m/year. goSPL adds the chord displacement before interpolating the moved surface back onto its fixed mesh. The chord connects two points on the sphere; it is not an instantaneous tangent velocity.
- Additional per-node arrays retain `geometric_rate`, signed `relaxation_correction` (both m/year), and `correction_supported` (material support as described below). `upsub` is their sum. `reference_z_end`, `reference_crust_end` and `reference_boundary_end` describe the next source epoch at fixed mesh coordinates for review; goSPL does not apply these reference arrays as constraints.
- `export_metadata.json` records source frame hashes, engine fingerprints, motion fitting residuals, and erosion-correction support for each interval. `source/frame-index.csv` links goSPL years to the original history. Year zero is the selected starting epoch; no Earth geological ages are implied.

## Meaning of the forcing

{motion_text}

{sampling_text}

{correction_paragraph}

Saved slab-window episodes have a separate instantaneous surface contribution. {thermal_text} Heating and cooling remain in the geometric surface-change term; lasting volcanic construction is already included in the recorded material relief. The captured `source/ridge_interaction.py` and its metadata fingerprint identify the reconstruction used. The exporter and its native geometry/sampling helpers are included with hashes under `source/`.

This is **an approximate geometric forcing product, not a complete physical uplift budget**. Unresolved coasts, changes of ownership, clipping, mixed deposits, coarse marker support, and goSPL's three-neighbor remapping prevent exact reconstruction. goSPL's terrain should diverge as it develops drainage, erosion and sediment. No fixed-location height difference is being passed off as pure uplift, and no procedural 8K terrain noise is inserted into the tectonic rates. The export mesh samples the saved native material geometry or legacy grids; it does not recover missing fine-scale tectonic history.

The source traces and original full history remain in Deep Time. ROCKE-3D climate inputs still require a separate preparation stage.

Primary format and method references:
- [goSPL required inputs](https://gospl.readthedocs.io/en/latest/user_guide/inputfile.html)
- [goSPL tectonic forcing](https://gospl.readthedocs.io/en/latest/user_guide/optfile2.html)
- [goSPL geometric forcing example](https://github.com/Geodels/goSPL-examples/blob/main/shared_scripts/umeshFcts.py)
'''
