"""Import one completed goSPL MPI epoch as a closed, native spherical surface.

XDMF/XMF references are parsed as data inside the selected result directory.
The standard goSPL external DTD declaration is discarded, never fetched.
Only the chosen epoch's coordinates, connectivity and nodal elevation are read.
Local h5py is optional; Windows can use the existing DeepTime-goSPL WSL runtime.
"""
from __future__ import annotations
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path, PureWindowsPath
import re
import subprocess
import tempfile
import time
import xml.etree.ElementTree as ET
import numpy as np
import mesh_geometry
import gospl_hdf_reader

MAX_XML_BYTES=16*1024*1024
DEFAULT_MEMORY_BYTES=2*1024**3
WSL_PYTHON='/home/gospl/micromamba/envs/gospl/bin/python'
INCLUDE='{http://www.w3.org/2001/XInclude}include'


class ResultCancelled(RuntimeError):pass


def _cancel(cancel):
    if cancel and cancel():raise ResultCancelled('goSPL result import cancelled.')


def _hash(path):
    digest=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda:stream.read(1024*1024),b''):digest.update(block)
    return digest.hexdigest()


def _xml(path):
    path=Path(path)
    if path.stat().st_size>MAX_XML_BYTES:raise ValueError('The XDMF descriptor is too large.')
    data=path.read_text(encoding='utf-8-sig')
    if re.search(r'<!ENTITY',data,re.I):raise ValueError('XML entities are not allowed.')
    data=re.sub(r'<!DOCTYPE\s+Xdmf\s+SYSTEM\s+[\"\']Xdmf\.dtd[\"\']\s*>','',data,flags=re.I)
    if '<!DOCTYPE' in data.upper():raise ValueError('External or custom XML declarations are not allowed.')
    try:element=ET.fromstring(data)
    except ET.ParseError as exc:raise ValueError(f'Invalid XDMF descriptor: {path.name}.') from exc
    if element.tag!='Xdmf':raise ValueError('Select a goSPL XDMF/XMF descriptor.')
    return element


def _inside(path,root):
    try:path.resolve().relative_to(root.resolve())
    except ValueError:raise ValueError('A result reference escapes its selected output folder.') from None
    return path.resolve()


def _reference(text,root,descriptor):
    if (not text or '\\' in text or '\x00' in text or ':' in text or text.startswith('/')
            or PureWindowsPath(text).drive):
        raise ValueError('Result references must be local relative paths.')
    candidates=[];escaped=False
    for base in (root,descriptor.parent):
        try:candidate=_inside(base/text,root)
        except ValueError:escaped=True;continue
        if candidate.is_file() and candidate not in candidates:candidates.append(candidate)
    if len(candidates)>1:raise ValueError('Ambiguous result reference exists at two different paths.')
    if not candidates and escaped:raise ValueError('A result reference escapes its selected output folder.')
    if not candidates:raise ValueError(f'Incomplete result: missing referenced file {text!r}.')
    return candidates[0]


def _data(element,role,root,descriptor):
    items=element.findall('DataItem')
    if len(items)!=1 or items[0].get('Format','').upper()!='HDF' or list(items[0]):
        raise ValueError('Coordinates, triangles and elevation must directly reference numeric HDF5 datasets.')
    item=items[0];value=(item.text or '').strip()
    match=re.fullmatch(r'(.+\.(?:h5|hdf5)):(/[A-Za-z0-9_./-]+)',value,re.I)
    if not match or '..' in match[2].split('/'):raise ValueError('Invalid local HDF5 dataset reference.')
    try:shape=[int(n) for n in item.get('Dimensions','').split()]
    except ValueError:raise ValueError('Invalid XMF dataset dimensions.') from None
    if not shape or any(n<=0 for n in shape):raise ValueError('XMF datasets must have positive dimensions.')
    if role in ('coords','cells') and (len(shape)!=2 or shape[1]!=3):
        raise ValueError('goSPL coordinates and triangles require three columns.')
    if role=='elevation' and not (len(shape)==1 or (len(shape)==2 and shape[1]==1)):
        raise ValueError('goSPL elevation must be one scalar per node.')
    units=element.get('Units',element.get('Unit',item.get('Units',item.get('Unit'))))
    if units is not None and role!='cells' and units.strip().lower() not in ('m','metre','metres','meter','meters'):
        raise ValueError('Only metre coordinates and elevations are supported.')
    return dict(path=str(_reference(match[1],root,descriptor)),dataset=match[2],shape=shape,role=role)


def _epoch(descriptor,root,fallback_index):
    tree=_xml(descriptor)
    if tree.findall('.//'+INCLUDE):raise ValueError('Nested epoch includes are not supported.')
    grids=tree.findall('./Domain/Grid')
    if len(grids)!=1:raise ValueError('An epoch must have one spatial collection.')
    grid=grids[0];times=grid.findall('Time')
    if len(times)!=1:raise ValueError('A saved epoch needs one explicit time in years.')
    try:years=float(times[0].get('Value','nan'))
    except ValueError:raise ValueError('Invalid goSPL epoch time.') from None
    if not math.isfinite(years) or years<0:raise ValueError('goSPL epoch time must be finite and nonnegative.')
    blocks=grid.findall('Grid') or [grid]
    parts=[];names=set()
    for block in blocks:
        name=block.get('Name',f'Block.{len(parts)}')
        if name in names:raise ValueError('Duplicate MPI block names in a saved epoch.')
        names.add(name)
        topology=block.find('Topology');geometry=block.find('Geometry')
        if topology is None or topology.get('Type',topology.get('TopologyType'))!='Triangle':
            raise ValueError('Only triangular goSPL output is supported.')
        if geometry is None or geometry.get('Type',geometry.get('GeometryType'))!='XYZ':
            raise ValueError('goSPL output must contain Cartesian XYZ geometry.')
        try:offset=int(topology.get('BaseOffset','0'))
        except ValueError:raise ValueError('Invalid triangle index base.') from None
        if offset not in (0,1):raise ValueError('Triangle index base must be zero or one.')
        elevations=[a for a in block.findall('Attribute') if a.get('Name')=='Z']
        if len(elevations)!=1 or elevations[0].get('Center')!='Node':
            raise ValueError('A goSPL block must have one nodal Z elevation attribute.')
        requests=[_data(geometry,'coords',root,descriptor),_data(topology,'cells',root,descriptor),
                  _data(elevations[0],'elevation',root,descriptor)]
        if requests[0]['shape'][0]!=requests[2]['shape'][0]:raise ValueError('Coordinate/elevation node counts disagree.')
        if int(topology.get('NumberOfElements',requests[1]['shape'][0]))!=requests[1]['shape'][0]:
            raise ValueError('Declared triangle counts disagree.')
        parts.append(dict(name=name,index_base=offset,requests=requests))
    if not parts:raise ValueError('The epoch has no MPI blocks.')
    match=re.fullmatch(r'gospl(\d+)\.(?:xmf|xdmf)',descriptor.name,re.I)
    return dict(index=int(match[1]) if match else fallback_index,time_years=years,parts=len(parts),
                descriptor=str(descriptor),blocks=parts)


def inspect_result(path):
    """List complete declared epochs in a folder or a selected XMF/HDF5 file."""
    selected=Path(path).expanduser().resolve()
    if not selected.exists():raise ValueError('The selected goSPL result does not exist.')
    if selected.is_dir():
        root=selected
        master=root/'gospl.xdmf'
        if master.is_file():selected=master
        else:
            choices=sorted(root.glob('*.xdmf'))
            if len(choices)==1:selected=choices[0]
            elif len(choices)>1:raise ValueError('Choose a specific XDMF file; this folder contains multiple histories.')
            else:raise ValueError('Choose the goSPL output folder containing gospl.xdmf, or one epoch XMF file.')
    else:root=selected.parent.parent if selected.parent.name.lower() in ('xmf','h5') else selected.parent
    if selected.suffix.lower() in ('.h5','.hdf5'):
        match=re.fullmatch(r'gospl\.(\d+)\.p\d+\.(?:h5|hdf5)',selected.name,re.I)
        if not match:raise ValueError('Choose an epoch output file, not a topology-only HDF5 file.')
        selected=root/'xmf'/f'gospl{match[1]}.xmf'
        if not selected.is_file():raise ValueError('One HDF5 rank cannot identify a complete world; its epoch XMF is required.')
    if selected.suffix.lower() not in ('.xmf','.xdmf'):raise ValueError('Select a goSPL folder, XDMF/XMF or epoch HDF5 output.')
    _inside(selected,root);tree=_xml(selected)
    includes=tree.findall('.//'+INCLUDE)
    descriptors=[]
    if includes:
        for include in includes:
            if include.get('parse','xml')!='xml' or include.get('xpointer','') not in ('','xpointer(//Xdmf/Domain/Grid)'):
                raise ValueError('Unsupported XML include expression.')
            descriptor=_reference(include.get('href',''),root,selected)
            if descriptor.suffix.lower() not in ('.xmf','.xdmf'):raise ValueError('Epoch includes must name local XMF files.')
            descriptors.append(descriptor)
    else:descriptors=[selected]
    if len(set(descriptors))!=len(descriptors):raise ValueError('A history lists the same epoch more than once.')
    epochs=[_epoch(descriptor,root,i) for i,descriptor in enumerate(descriptors)]
    if len({e['index'] for e in epochs})!=len(epochs) or len({e['time_years'] for e in epochs})!=len(epochs):
        raise ValueError('Epoch indices and times must be unique.')
    epochs.sort(key=lambda e:e['time_years'])
    return dict(source_type='gospl_result',root=str(root),selection=str(selected),epochs=epochs,default_epoch=epochs[-1]['index'])


def _wsl_path(path):
    value=str(Path(path).resolve())
    match=re.fullmatch(r'([A-Za-z]):[\\/](.*)',value)
    if not match:raise ValueError('The WSL bridge needs a local Windows drive path.')
    return '/mnt/'+match[1].lower()+'/'+match[2].replace('\\','/')


def _read_hdf(requests,work_dir,cancel):
    if importlib.util.find_spec('h5py') is not None:
        _cancel(cancel);rows=gospl_hdf_reader.extract(requests,work_dir,check_cancel=lambda:_cancel(cancel))
        return rows,'local h5py'
    if os.name!='nt':raise ValueError('Reading goSPL HDF5 requires h5py in this Python environment.')
    converted=[dict(request,path=_wsl_path(request['path'])) for request in requests]
    request_path=work_dir/'requests.json';request_path.write_text(json.dumps(converted),encoding='utf-8')
    args=['wsl.exe','--distribution','DeepTime-goSPL','--user','gospl','--exec',WSL_PYTHON,
          _wsl_path(gospl_hdf_reader.__file__),_wsl_path(request_path),_wsl_path(work_dir)]
    log=work_dir/'bridge.log'
    with log.open('wb') as output:
        process=subprocess.Popen(args,stdout=output,stderr=subprocess.STDOUT,creationflags=getattr(subprocess,'CREATE_NO_WINDOW',0))
        started=time.monotonic()
        while process.poll() is None:
            if (cancel and cancel()) or time.monotonic()-started>600:
                (work_dir/'cancel').touch()
                try:process.wait(timeout=5)
                except subprocess.TimeoutExpired:process.terminate();process.wait(timeout=10)
                _cancel(cancel);raise ValueError('The HDF5 reader exceeded its ten-minute time limit.')
            time.sleep(.1)
    if process.returncode:
        message=log.read_bytes().decode('utf-8',errors='replace').replace('\x00','')[-2500:]
        raise ValueError('The installed DeepTime-goSPL HDF5 reader could not open this result. '+message)
    return json.loads((work_dir/'extracted.json').read_text(encoding='utf-8')),'DeepTime-goSPL WSL h5py'


def _closed(vertices,faces):
    if len(vertices)<4 or len(faces)<4:raise ValueError('A global result needs a closed spherical triangular mesh.')
    total=0.;flipped=0
    for start in range(0,len(faces),65536):
        section=faces[start:start+65536];tri=vertices[section];a,b,c=tri.transpose(1,0,2)
        determinant=np.einsum('ni,ni->n',a,np.cross(b,c))
        if np.any(abs(determinant)<1e-14):raise ValueError('Degenerate spherical triangles in the result.')
        denominator=1.+np.einsum('ni,ni->n',a,b)+np.einsum('ni,ni->n',b,c)+np.einsum('ni,ni->n',c,a)
        area=2*np.arctan2(abs(determinant),denominator)
        if np.any(area<=0)|np.any(area>=np.pi):raise ValueError('Invalid spherical triangle area.')
        total+=float(area.sum());flip=determinant<0;flipped+=int(flip.sum())
        section[flip]=section[flip][:,[0,2,1]]
    directed=faces[:,[[0,1],[1,2],[2,0]]].reshape(-1,2)
    edges=np.sort(directed,axis=1)
    unique,inverse,counts=np.unique(edges,axis=0,return_inverse=True,return_counts=True)
    if np.any(counts!=2):raise ValueError('MPI pieces do not reconstruct a closed mesh: missing, inconsistent or non-manifold edges.')
    if len(vertices)-len(unique)+len(faces)!=2:raise ValueError('The reconstructed mesh is not a single spherical surface.')
    direction=np.where(directed[:,0]<directed[:,1],1.,-1.)
    if np.any(np.bincount(inverse,weights=direction)!=0):
        raise ValueError('Adjacent spherical triangles overlap or have inconsistent orientation.')
    if not np.isclose(total,4*np.pi,rtol=2e-6,atol=1e-8):raise ValueError('The triangles do not cover one complete sphere.')
    return dict(closed=True,consistent_winding=True,spherical_area_steradians=total,edges=len(unique),reoriented_faces=flipped)


def _provenance(root):
    for folder in (root,root.parent):
        path=folder/'export_metadata.json'
        if path.is_file():
            if path.stat().st_size>MAX_XML_BYTES:raise ValueError('Export provenance metadata is too large.')
            data=json.loads(path.read_text(encoding='utf-8'))
            if not isinstance(data,dict):raise ValueError('Export provenance metadata must be a JSON object.')
            if not str(data.get('format','')).startswith('deep-time-gospl-history'):
                raise ValueError('Adjacent export metadata has an unsupported format.')
            names=('run_id','start_myr','end_myr','start_index','end_index','orientation','orientation_matrix',
                   'orientation_convention','radius_m','time_convention','source_engine_sha256','documented_goSPL_version')
            return {name:data[name] for name in names if name in data},dict(path=str(path),sha256=_hash(path))
    return {},None


def load_result(path,epoch_index=None,*,work_dir=None,cancel=None,progress=None,max_memory_bytes=DEFAULT_MEMORY_BYTES):
    """Read one complete epoch, weld exact MPI duplicates, and validate closure.

    Storage is bounded to one chosen epoch and a declared working-memory budget.
    HDF datasets are staged as NPY arrays one at a time, never as an entire run.
    The returned arrays are independent of the original solver directory.
    """
    report=inspect_result(path);epochs=report['epochs']
    selected=report['default_epoch'] if epoch_index is None else epoch_index
    if isinstance(selected,bool) or not isinstance(selected,(int,np.integer)):
        raise ValueError('Select an integer goSPL epoch index.')
    matches=[e for e in epochs if e['index']==selected]
    if len(matches)!=1:raise ValueError('The selected epoch is not declared by this result.')
    epoch=matches[0];requests=[r for block in epoch['blocks'] for r in block['requests']]
    nodes=sum(r['shape'][0] for r in requests if r['role']=='coords')
    triangles=sum(r['shape'][0] for r in requests if r['role']=='cells')
    estimated=nodes*192+triangles*256
    if nodes>10_000_000 or triangles>20_000_000 or estimated>max_memory_bytes:
        raise ValueError('This result exceeds the importer working-memory budget; choose a smaller mesh or raise the explicit budget.')
    root=Path(report['root']);files={Path(report['selection']),Path(epoch['descriptor'])}|{Path(r['path']) for r in requests}
    identities={str(p):(p.stat().st_size,p.stat().st_mtime_ns) for p in files}
    sources=[dict(path=str(p),relative_path=os.path.relpath(p,root),bytes=p.stat().st_size,sha256=_hash(p)) for p in sorted(files)]
    _cancel(cancel)
    if progress:progress(.08)
    if work_dir is not None:Path(work_dir).mkdir(parents=True,exist_ok=True)
    with tempfile.TemporaryDirectory(prefix='gospl-read-',dir=work_dir) as temporary:
        temporary=Path(temporary);rows,backend=_read_hdf(requests,temporary,cancel)
        if len(rows)!=len(requests):raise ValueError('The HDF5 reader did not return every requested dataset.')
        coordinates=[];heights=[];connectivity=[];offset=0
        for index,block in enumerate(epoch['blocks']):
            _cancel(cancel)
            arrays=[]
            for request,row in zip(block['requests'],rows[index*3:index*3+3]):
                units=row.get('units')
                if units is not None and request['role']!='cells' and units.strip().lower() not in ('m','metre','metres','meter','meters'):
                    raise ValueError('HDF5 units disagree with the goSPL metre convention.')
                filename=row['filename']
                if Path(filename).name!=filename:raise ValueError('Invalid HDF5 extraction filename.')
                arrays.append(np.load(temporary/filename,allow_pickle=False))
            xyz,cells,z=arrays;z=z.reshape(-1)
            if not np.isfinite(xyz).all() or not np.isfinite(z).all():raise ValueError('Non-finite goSPL coordinates or elevations.')
            local=cells.astype(np.int64)-block['index_base']
            if np.any(local<0) or np.any(local>=len(xyz)):raise ValueError('A local MPI triangle references a missing node.')
            coordinates.append(xyz);heights.append(z);connectivity.append(local+offset);offset+=len(xyz)
        xyz=np.concatenate(coordinates);z=np.concatenate(heights);cells=np.concatenate(connectivity)
        del coordinates,heights,connectivity
    if progress:progress(.45)
    for value in sources:
        source=Path(value['path'])
        if identities[str(source)]!=(source.stat().st_size,source.stat().st_mtime_ns) or _hash(source)!=value['sha256']:
            raise ValueError('The selected solver output changed during import; wait for the epoch to finish.')
    radius=np.linalg.norm(xyz.astype(np.float64),axis=1);radius_m=float(np.median(radius))
    if not 100_000.<=radius_m<=100_000_000. or np.max(abs(radius-radius_m))>max(2.,radius_m*2e-6):
        raise ValueError('Coordinates must lie on one planetary sphere in metres, with elevation stored separately.')
    provenance,provenance_file=_provenance(root)
    if 'radius_m' in provenance and not np.isclose(float(provenance['radius_m']),radius_m,rtol=2e-6):
        raise ValueError('Solver coordinates disagree with the source export radius.')
    # goSPL writes shared rank coordinates from the same global mesh. Exact
    # equality merges those copies without welding unrelated nearby vertices.
    unique,first,inverse=np.unique(xyz,axis=0,return_index=True,return_inverse=True)
    low=np.full(len(unique),np.inf);high=np.full(len(unique),-np.inf)
    np.minimum.at(low,inverse,z);np.maximum.at(high,inverse,z)
    tolerance=np.maximum(.01,np.maximum(abs(low),abs(high))*2e-6)
    if np.any(high-low>tolerance):raise ValueError('Shared MPI nodes contain inconsistent elevations; the epoch may be incomplete.')
    elevation=z[first].astype(np.float64);vertices=unique.astype(np.float64)
    vertices/=np.linalg.norm(vertices,axis=1)[:,None]
    remapped=inverse[cells]
    if np.any(np.diff(np.sort(remapped,axis=1),axis=1)==0):raise ValueError('Repeated node indices in a triangle.')
    _,chosen=np.unique(np.sort(remapped,axis=1),axis=0,return_index=True)
    faces=remapped[np.sort(chosen)].astype(np.int32)
    if len(np.unique(faces))!=len(vertices):raise ValueError('The reconstructed surface has unused/disconnected nodes.')
    _cancel(cancel)
    if progress:progress(.72)
    geometry=_closed(vertices,faces)
    metadata=dict(format='deep-time-gospl-result-v1',source_type='gospl_result',selected_epoch=int(selected),
        time_years=float(epoch['time_years']),time_convention='Elapsed goSPL solver years; no inferred Earth geological age.',
        source_root=str(root),descriptor=epoch['descriptor'],source_files=sources,reader_backend=backend,
        node_count=len(vertices),triangle_count=len(faces),mpi_parts=epoch['parts'],radius_m=radius_m,
        coordinate_units='m in source HDF5; normalized unit XYZ in canonical surface',elevation_units='m',
        units_basis='Documented goSPL coords/elev convention; available explicit unit attributes checked.',
        merged_duplicate_nodes=int(nodes-len(vertices)),merged_duplicate_triangles=int(triangles-len(faces)),
        maximum_shared_height_difference_m=float((high-low).max(initial=0.)),
        working_memory_estimate_bytes=int(estimated),geometry=geometry,source_export=provenance,
        orientation_in_coordinates=True,orientation_note='Solver XYZ already contains any exported-world rotation; do not apply source orientation a second time.',
        import_sources_sha256={Path(module.__file__).name:_hash(module.__file__) for module in (gospl_hdf_reader,mesh_geometry)},
        importer_sha256=_hash(__file__),fine_detail_default=0.,
        limitations=['Native nodal interpolation does not add landscape resolution.',
                     'Exact shared coordinates are required; inconsistent or near-miss MPI seams are refused, never filled.',
                     'Height units follow the goSPL output convention when unit attributes are absent.'])
    if provenance_file:metadata['source_files'].append(provenance_file)
    if 'start_myr' in provenance:
        origin=float(provenance['start_myr'])
        if not np.isfinite(origin):raise ValueError('Invalid source tectonic time origin.')
        metadata['tectonic_time_myr']=origin+epoch['time_years']/1e6
    try:json.dumps(metadata,allow_nan=False)
    except (TypeError,ValueError) as exc:raise ValueError('Result provenance must contain finite JSON values.') from exc
    if progress:progress(1.)
    return dict(source_type='gospl_result',vertices=vertices,faces=faces,elevation_m=elevation,metadata=metadata)


def prepare(source):
    """Build one bounded exact spherical locator for repeated output bands."""
    return dict(locator=mesh_geometry.build_locator(source['vertices'],source['faces']),
                faces=np.asarray(source['faces']),height=np.asarray(source['elevation_m'],float))


def sample_height(source,points,prepared=None):
    """Interpolate the actual containing triangle; never fill a missing mesh."""
    context=prepare(source) if prepared is None else prepared
    face,weights=mesh_geometry.locate_points(points,context['locator'])
    if np.any(face<0):raise ValueError('A goSPL query is outside the validated closed surface.')
    return np.sum(context['height'][context['faces'][face]]*weights,axis=1)
