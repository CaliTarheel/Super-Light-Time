"""Disk-backed terrain jobs and native-resolution map tiles."""
from __future__ import annotations

from copy import deepcopy
from datetime import datetime, timezone
import ast
import hashlib
import io
import json
import math
from pathlib import Path
import re
import shutil
import threading
import time
import uuid
import zipfile

import numpy as np

import terrain
import orientation as globe_orientation
from progress_timing import ProgressTiming
from terrain import build_terrain, color_tile, TerrainCancelled


BUILDER_SOURCE = Path(terrain.__file__).read_bytes()
ORIENTATION_SOURCE = Path(globe_orientation.__file__).read_bytes()
OUTPUT_WIDTHS = (2048, 4096, 8192, 16384)


def capture_builder_sources(root=None, entries=('terrain.py', 'gospl_results.py')):
    """Capture the recursive local import closure of the actual terrain reader.

    These sources interpret saved arrays today and are separate from the exact
    historical engine sources, which must never be replaced with current code.
    Standard-library and installed-package imports have no local .py candidate.
    """
    root = Path(root) if root is not None else Path(__file__).resolve().parent
    result, pending = {}, list(entries)
    while pending:
        name = pending.pop()
        if name in result or not re.fullmatch(r'[A-Za-z_][A-Za-z0-9_]*\.py', name):
            continue
        path = root/name
        if not path.is_file():
            continue
        payload = path.read_bytes()
        result[name] = payload
        for node in ast.walk(ast.parse(payload.decode('utf-8-sig'))):
            modules = ([entry.name for entry in node.names] if isinstance(node, ast.Import)
                       else [node.module] if isinstance(node, ast.ImportFrom) and node.module else [])
            pending.extend(module.split('.')[0]+'.py' for module in modules)
    return dict(sorted(result.items()))


BUILDER_SOURCES = capture_builder_sources()
BUILDER_SOURCE_HASHES = {name: hashlib.sha256(payload).hexdigest() for name, payload in BUILDER_SOURCES.items()}


def _file_hash(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024*1024), b''):
            digest.update(block)
    return digest.hexdigest()


def read_saved_auxiliary_sources(run_path, fingerprints):
    """Read only dependencies recorded by this run, checking exact source bytes.

    A legacy run's absent/empty mapping intentionally yields no helpers.  Never
    fill a missing saved dependency with the currently installed model source.
    """
    if not isinstance(fingerprints, dict):
        raise ValueError('Saved auxiliary model source fingerprints must be an object.')
    result = {}
    for name, fingerprint in fingerprints.items():
        if (not isinstance(name, str) or not re.fullmatch(r'[A-Za-z_][A-Za-z0-9_]*\.py', name)
                or name in ('engine.py', 'tectonics.py')
                or not isinstance(fingerprint, str) or not re.fullmatch(r'[0-9a-f]{64}', fingerprint)):
            raise ValueError('Saved auxiliary model source fingerprint is invalid.')
        path = Path(run_path) / name
        try:
            payload = path.read_bytes()
        except OSError as exc:
            raise ValueError(f'Saved model dependency {name} is missing or unreadable.') from exc
        if hashlib.sha256(payload).hexdigest() != fingerprint:
            raise ValueError(f'Saved model dependency {name} does not match its recorded fingerprint.')
        result[name] = payload
    return dict(sorted(result.items()))


def _write_json(path, value):
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, allow_nan=False, separators=(',', ':')), encoding='utf-8')
    temporary.replace(path)


def _integer(value, name):
    try:
        if isinstance(value, bool) or not math.isfinite(float(value)) or float(value) != int(value):
            raise ValueError()
        return int(value)
    except (ValueError, TypeError, OverflowError):
        raise ValueError(f'{name} must be an integer.') from None


class TerrainManager:
    def __init__(self, root, builder=None):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self.builder = builder or build_terrain
        self.lock = threading.RLock()
        self.stop = threading.Event()
        self.worker = None
        self.current = {'state': 'idle', 'progress': 0, 'job_id': None}
        self.timing = None
        jobs = []
        for path in self.root.glob('*/job.json'):
            try:
                job = json.loads(path.read_text(encoding='utf-8'))
                if job['state'] == 'running':
                    job.update(state='cancelled', eta_seconds=None, error='The app stopped before this terrain build completed.')
                    self.remove_incomplete(path.parent)
                    _write_json(path, job)
                jobs.append(job)
            except (OSError, ValueError, KeyError):
                continue
        if jobs:
            self.current = max(jobs, key=lambda job: job.get('created', ''))

    def path(self, job_id):
        if not isinstance(job_id, str) or not re.fullmatch(r'[A-Za-z0-9_-]{1,80}', job_id):
            raise ValueError('Invalid terrain job identifier.')
        return self.root / job_id

    def status(self, job_id=None):
        with self.lock:
            if job_id is None or job_id == self.current.get('job_id'):
                result = deepcopy(self.current)
                if self.timing:
                    result.update(self.timing.snapshot())
                if result.get('state') != 'running':
                    result['eta_seconds'] = 0. if result.get('state') == 'complete' else None
                else:
                    result.setdefault('eta_seconds', None)
                return result
        try:
            result = json.loads((self.path(job_id) / 'job.json').read_text(encoding='utf-8'))
            result['eta_seconds'] = 0. if result.get('state') == 'complete' else None
            return result
        except FileNotFoundError:
            raise ValueError('Terrain build not found.') from None

    def list_jobs(self):
        with self.lock:
            jobs = []
            for path in self.root.glob('*/job.json'):
                try:
                    job = json.loads(path.read_text(encoding='utf-8'))
                    if job.get('job_id') == self.current.get('job_id'):
                        job = self.status()
                    jobs.append(job)
                except (ValueError, OSError):
                    continue
            return sorted(jobs, key=lambda job: job.get('created', ''), reverse=True)

    def remove_incomplete(self, path):
        """Remove only this job's known derived products; retain its source and log."""
        if path.resolve().parent != self.root.resolve():
            raise ValueError('Incomplete terrain products must belong to this job directory.')
        for name in ('elevation_m.npy', 'heightmap_16bit.png', 'preview.png',
                     'terrain_metadata.json', 'terrain.zip', '.orientation_work.npy'):
            try:
                (path / name).unlink(missing_ok=True)
            except OSError:
                # Preserve the original job failure if Windows still holds a file.
                pass

    def start(self, simulation_manager, index, run_id=None, width=8192, detail=1., orientation=None,
              reconstruct_margins=False):
        orientation = globe_orientation.normalize_orientation(orientation)
        width = _integer(width, 'Terrain width')
        index = _integer(index, 'Frame')
        if width not in OUTPUT_WIDTHS:
            raise ValueError('Choose a terrain width of 2048, 4096, 8192 or 16384.')
        try:
            if isinstance(detail, bool):
                raise ValueError()
            detail = float(detail)
            if not math.isfinite(detail) or not 0 <= detail <= 2:
                raise ValueError()
        except (TypeError, ValueError):
            raise ValueError('Terrain detail must be between zero and two.') from None
        if not isinstance(reconstruct_margins, bool):
            raise ValueError('Margin reconstruction must be enabled or disabled explicitly.')
        run_path, manifest = simulation_manager._history_context(run_id)
        auxiliary_hashes = deepcopy(manifest.get('auxiliary_sources_sha256', {}))
        read_saved_auxiliary_sources(run_path, auxiliary_hashes)
        count = int(manifest.get('frame_count', 0))
        if index < 0:
            index += count
        if index < 0 or index >= count:
            raise ValueError('Select an available saved history frame.')
        frame = json.loads((run_path / f'frame_{index:04d}.json').read_text(encoding='utf-8'))
        if reconstruct_margins and not int(frame.get('mesh_version', 0)):
            raise ValueError('Reconstructing physical continental margins requires a native mesh history.')
        source_hashes = {suffix: _file_hash(run_path/f'frame_{index:04d}{suffix}') for suffix in ('.npz', '.json')}
        with self.lock:
            if self.worker and self.worker.is_alive():
                raise ValueError('A terrain build is already running. Stop it before starting another.')
            job_id = datetime.now().strftime('%Y%m%d-%H%M%S') + '-' + uuid.uuid4().hex[:8]
            path = self.path(job_id)
            path.mkdir()
            self.current = {
                'job_id': job_id, 'state': 'running', 'phase': 'synthesizing', 'progress': 0.,
                'created': datetime.now(timezone.utc).isoformat(),
                'run_id': manifest['run_id'], 'frame': index, 'time_myr': frame['time_myr'],
                'source_width': frame['width'], 'source_height': frame['height'],
                'width': width, 'height': width//2, 'detail': detail,
                'seed': int(manifest.get('config', {}).get('seed', 12)),
                'source_engine_sha256': manifest.get('engine_sha256'),
                'source_auxiliary_sources_sha256': auxiliary_hashes,
                'terrain_builder_sha256': hashlib.sha256(BUILDER_SOURCE).hexdigest(),
                'terrain_builder_sources_sha256': BUILDER_SOURCE_HASHES,
                'source_frame_sha256': source_hashes,
                'source_type': 'native_history' if int(frame.get('mesh_version', 0)) else 'raster_history',
                'reconstruct_margins': reconstruct_margins,
                'elapsed_seconds': 0.,
                'eta_seconds': None, 'orientation': orientation,
            }
            self.timing = ProgressTiming()
            _write_json(path / 'job.json', self.current)
            self.stop.clear()
            self.worker = threading.Thread(target=self._run, args=(run_path, deepcopy(self.current)), daemon=True)
            self.worker.start()
            return self.status()

    def start_gospl_result(self, input_path, epoch_index=None, width=8192, detail=0., orientation=None):
        """Build ordinary terrain products from one selected evolved solver epoch."""
        import gospl_results
        width = _integer(width, 'Terrain width')
        if width not in OUTPUT_WIDTHS:
            raise ValueError('Choose a terrain width of 2048, 4096, 8192 or 16384.')
        try:
            if isinstance(detail, bool):
                raise ValueError()
            detail = float(detail)
            if not math.isfinite(detail) or not 0 <= detail <= 2:
                raise ValueError()
        except (TypeError, ValueError):
            raise ValueError('Terrain detail must be between zero and two.') from None
        orientation = globe_orientation.normalize_orientation(orientation)
        inspected = gospl_results.inspect_result(input_path)
        selected = inspected['default_epoch'] if epoch_index is None else _integer(epoch_index, 'goSPL epoch')
        epochs = {int(row['index']): row for row in inspected['epochs']}
        if selected not in epochs:
            raise ValueError('Select an available goSPL result epoch.')
        with self.lock:
            if self.worker and self.worker.is_alive():
                raise ValueError('A terrain build is already running. Stop it before starting another.')
            job_id = datetime.now().strftime('%Y%m%d-%H%M%S')+'-'+uuid.uuid4().hex[:8]
            path = self.path(job_id)
            path.mkdir()
            self.current = dict(job_id=job_id, state='running', phase='reading goSPL result', progress=0.,
                created=datetime.now(timezone.utc).isoformat(), source_type='gospl_result',
                source_path=str(Path(input_path).resolve()), source_result_epoch=int(selected),
                source_result_inspection=inspected, run_id='gospl-result', frame=int(selected),
                time_myr=float(epochs[selected]['time_years'])/1e6, source_width=None, source_height=None,
                width=width, height=width//2, detail=detail, seed=12, orientation=orientation,
                terrain_builder_sha256=hashlib.sha256(BUILDER_SOURCE).hexdigest(),
                terrain_builder_sources_sha256=BUILDER_SOURCE_HASHES, elapsed_seconds=0., eta_seconds=None)
            self.timing = ProgressTiming()
            _write_json(path/'job.json', self.current)
            self.stop.clear()
            self.worker = threading.Thread(target=self._run_gospl_result, args=(deepcopy(self.current),), daemon=True)
            self.worker.start()
            return self.status()

    def _run_gospl_result(self, job):
        import gospl_results
        path = self.path(job['job_id'])
        source_path = path/'source'
        source_path.mkdir(exist_ok=True)
        started, last_save = time.perf_counter(), 0.

        def update_progress(value, phase):
            nonlocal last_save
            elapsed = time.perf_counter()-started
            with self.lock:
                self.current.update(progress=float(value), phase=phase, **self.timing.update(float(value)))
                if elapsed-last_save > .5:
                    _write_json(path/'job.json', self.current)
                    last_save = elapsed
        try:
            source = gospl_results.load_result(job['source_path'], job['source_result_epoch'], work_dir=source_path,
                cancel=self.stop.is_set, progress=lambda value: update_progress(.10*float(value), 'reading goSPL result'))
            if self.stop.is_set():
                raise TerrainCancelled()
            canonical = source_path/'gospl_result.npz'
            np.savez(canonical, **{key: source[key] for key in ('vertices', 'faces', 'elevation_m')})
            input_metadata = dict(source['metadata'], canonical_result_sha256=_file_hash(canonical))
            _write_json(source_path/'gospl_result.json', input_metadata)
            prepared = gospl_results.prepare(source)
            metadata_source = dict(input_metadata, source_type='gospl_result',
                source_result_epoch=job['source_result_epoch'], time_myr=job['time_myr'],
                vertices=len(source['vertices']), faces=len(source['faces']))
            terrain.build_sampled_terrain(lambda points: gospl_results.sample_height(source, points, prepared),
                path, source=metadata_source, width=job['width'], seed=job['seed'], detail=job['detail'],
                orientation=job['orientation'], cancel=self.stop.is_set,
                progress=lambda value: update_progress(.10+.84*float(value), 'sampling goSPL surface'))
            if self.stop.is_set():
                raise TerrainCancelled()
            metadata_path = path/'terrain_metadata.json'
            metadata = json.loads(metadata_path.read_text(encoding='utf-8'))
            metadata.update(source_type='gospl_result', source_result=input_metadata,
                terrain_builder_sha256=job['terrain_builder_sha256'],
                terrain_builder_sources_sha256=job['terrain_builder_sources_sha256'],
                orientation_sha256=hashlib.sha256(ORIENTATION_SOURCE).hexdigest())
            _write_json(metadata_path, metadata)
            (path/'terrain_builder.py').write_bytes(BUILDER_SOURCE)
            (path/'orientation.py').write_bytes(ORIENTATION_SOURCE)
            files = [(path/name, name) for name in ('elevation_m.npy', 'heightmap_16bit.png', 'preview.png',
                                                   'terrain_metadata.json', 'terrain_builder.py', 'orientation.py')]
            files += [(source_path/name, 'source/'+name) for name in ('gospl_result.npz', 'gospl_result.json')]
            files += self._preserve_builder_sources(path, job)
            self._write_package(path, files,
                progress=lambda value: update_progress(.95+.049*float(value), 'packaging'), cancel=self.stop.is_set)
            if self.stop.is_set():
                raise TerrainCancelled()
            with self.lock:
                self.current.update(state='complete', phase='complete', progress=1., metadata=metadata, **self.timing.finish())
                _write_json(path/'job.json', self.current)
        except Exception as exc:
            self.remove_incomplete(path)
            with self.lock:
                if self.stop.is_set() or isinstance(exc, TerrainCancelled):
                    self.current.update(state='cancelled', **self.timing.pause())
                else:
                    self.current.update(state='error', error=f'{type(exc).__name__}: {exc}', **self.timing.pause())
                _write_json(path/'job.json', self.current)

    def _run(self, run_path, job):
        path = self.path(job['job_id'])
        started = time.perf_counter()
        last_save = 0.

        def update_progress(value, phase):
            nonlocal last_save
            elapsed = time.perf_counter()-started
            with self.lock:
                self.current.update(progress=float(value), phase=phase, **self.timing.update(float(value)))
                if elapsed-last_save > .5:
                    _write_json(path / 'job.json', self.current)
                    last_save = elapsed
        try:
            index = job['frame']
            source = run_path / f'frame_{index:04d}.npz'
            frame = json.loads(source.with_suffix('.json').read_text(encoding='utf-8'))
            with np.load(source, allow_pickle=False) as archive:
                frame.update({key: archive[key] for key in archive.files})
            frame.update(run_id=job['run_id'], source_engine_sha256=job['source_engine_sha256'], index=index)
            frame['source_frame_sha256'] = job.get('source_frame_sha256', {})
            orientation_options = {'orientation': job['orientation']} if any(job['orientation'].values()) else {}
            if job.get('reconstruct_margins'):
                orientation_options['reconstruct_margins'] = True
            self.builder(frame, path, width=job['width'], seed=job['seed'], detail=job['detail'], **orientation_options,
                         progress=lambda value: update_progress(min(.95, float(value)*.95), 'synthesizing'),
                         cancel=self.stop.is_set)
            if self.stop.is_set():
                raise TerrainCancelled()
            update_progress(.95, 'packaging')
            self.package(path, run_path, job, progress=lambda value: update_progress(.95+.049*value, 'packaging'),
                         cancel=self.stop.is_set)
            if self.stop.is_set():
                raise TerrainCancelled()
            metadata = json.loads((path / 'terrain_metadata.json').read_text(encoding='utf-8'))
            with self.lock:
                self.current.update(state='complete', phase='complete', progress=1., metadata=metadata,
                                    **self.timing.finish())
                _write_json(path / 'job.json', self.current)
        except TerrainCancelled:
            self.remove_incomplete(path)
            with self.lock:
                self.current.update(state='cancelled', **self.timing.pause())
                _write_json(path / 'job.json', self.current)
        except Exception as exc:
            import traceback
            traceback.print_exc()
            self.remove_incomplete(path)
            with self.lock:
                self.current.update(state='error', error=f'{type(exc).__name__}: {exc}',
                                    **self.timing.pause())
                _write_json(path / 'job.json', self.current)

    def package(self, path, run_path, job, progress=None, cancel=None):
        """Keep the precise source epoch alongside the derived terrain."""
        metadata_path = path / 'terrain_metadata.json'
        metadata = json.loads(metadata_path.read_text(encoding='utf-8'))
        auxiliary_hashes = job.get('source_auxiliary_sources_sha256', {})
        auxiliary_sources = read_saved_auxiliary_sources(run_path, auxiliary_hashes)
        metadata.update(source_run_id=job['run_id'], source_frame=job['frame'],
                        source_engine_sha256=job.get('source_engine_sha256'),
                        source_auxiliary_sources_sha256=auxiliary_hashes,
                        terrain_builder_sha256=job['terrain_builder_sha256'],
                        terrain_builder_sources_sha256=job.get('terrain_builder_sources_sha256', BUILDER_SOURCE_HASHES),
                        source_frame_sha256=job.get('source_frame_sha256', {}),
                        reconstruct_margins=job.get('reconstruct_margins', False),
                        orientation=job.get('orientation', globe_orientation.normalize_orientation()),
                        orientation_sha256=hashlib.sha256(ORIENTATION_SOURCE).hexdigest())
        _write_json(metadata_path, metadata)
        (path / 'terrain_builder.py').write_bytes(BUILDER_SOURCE)
        (path / 'orientation.py').write_bytes(ORIENTATION_SOURCE)
        source_path = path / 'source'
        source_path.mkdir(exist_ok=True)
        source_files = []
        for suffix in ('.npz', '.json'):
            target = source_path / f'frame{suffix}'
            shutil.copyfile(run_path / f"frame_{job['frame']:04d}{suffix}", target)
            expected = job.get('source_frame_sha256', {}).get(suffix)
            if expected is not None and _file_hash(target) != expected:
                raise ValueError('The source epoch changed after this terrain job was selected.')
            source_files.append(target)
        for name in ('config.json', 'engine.py'):
            if (run_path / name).exists():
                target = source_path / name
                shutil.copyfile(run_path / name, target)
                source_files.append(target)
        for name, payload in auxiliary_sources.items():
            target = source_path / name
            target.write_bytes(payload)
            source_files.append(target)
        files = [(path / name, name) for name in
                 ('elevation_m.npy', 'heightmap_16bit.png', 'preview.png', 'terrain_metadata.json', 'terrain_builder.py', 'orientation.py')]
        files += [(source, 'source/' + source.name) for source in sorted(source_files)]
        files += self._preserve_builder_sources(path, job)
        self._write_package(path, files, progress, cancel)

    def _preserve_builder_sources(self, path, job):
        folder = path/'builder'
        folder.mkdir(exist_ok=True)
        hashes = job.get('terrain_builder_sources_sha256', BUILDER_SOURCE_HASHES)
        if hashes != BUILDER_SOURCE_HASHES:
            raise ValueError('Terrain builder helpers changed while the job was running.')
        files = []
        for name, payload in BUILDER_SOURCES.items():
            target = folder/name
            target.write_bytes(payload)
            files.append((target, 'builder/'+name))
        return files

    def _write_package(self, path, files, progress=None, cancel=None):
        total = sum(source.stat().st_size for source, _ in files)
        written = 0
        with zipfile.ZipFile(path / 'terrain.zip', 'w', compression=zipfile.ZIP_DEFLATED, compresslevel=1) as archive:
            for source, name in files:
                with source.open('rb') as incoming, archive.open(name, 'w', force_zip64=True) as outgoing:
                    while True:
                        if cancel and cancel():
                            raise TerrainCancelled()
                        chunk = incoming.read(1024*1024)
                        if not chunk:
                            break
                        outgoing.write(chunk)
                        written += len(chunk)
                        if progress:
                            progress(written / max(1, total))

    def cancel(self, job_id=None):
        with self.lock:
            if job_id is not None and job_id != self.current.get('job_id'):
                raise ValueError('That terrain build is no longer the active job.')
            self.stop.set()
            return self.status()

    def completed_path(self, job_id, filename):
        job = self.status(job_id)
        if job.get('state') != 'complete':
            raise ValueError('The terrain build is not complete yet.')
        return self.path(job['job_id']) / filename, job

    def download(self, job_id, format_name):
        filenames = {'png': 'heightmap_16bit.png', 'npy': 'elevation_m.npy', 'zip': 'terrain.zip'}
        if format_name not in filenames:
            raise ValueError('Choose PNG, NPY or ZIP for the terrain download.')
        path, _ = self.completed_path(job_id, filenames[format_name])
        return path

    def tile(self, job_id, x, y):
        x, y = _integer(x, 'Tile column'), _integer(y, 'Tile row')
        path, job = self.completed_path(job_id, 'elevation_m.npy')
        if x < 0 or y < 0 or x*512 >= job['width'] or y*512 >= job['height']:
            raise ValueError('Tile is outside the terrain map.')
        data = np.load(path, mmap_mode='r', allow_pickle=False)
        try:
            picture = color_tile(data, x*512, y*512, size=512)
            stream = io.BytesIO()
            picture.save(stream, format='PNG')
            return stream.getvalue()
        finally:
            del data

    def sample(self, job_id, x, y):
        x, y = _integer(x, 'Pixel column'), _integer(y, 'Pixel row')
        path, job = self.completed_path(job_id, 'elevation_m.npy')
        if not 0 <= x < job['width'] or not 0 <= y < job['height']:
            raise ValueError('Pixel is outside the terrain map.')
        data = np.load(path, mmap_mode='r', allow_pickle=False)
        try:
            return {'elevation_m': float(data[y, x]), 'lon': -180+(x+.5)*360/job['width'],
                    'lat': 90-(y+.5)*180/job['height'], 'x': x, 'y': y,
                    'run_id': job['run_id'], 'time_myr': job['time_myr']}
        finally:
            del data
