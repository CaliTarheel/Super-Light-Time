"""Independent background jobs for native evolving-history goSPL exports."""
from copy import deepcopy
from datetime import datetime, timezone
import json
import math
from pathlib import Path
import re
import threading
import time
import uuid

from gospl_export import build_history, checked_times, ExportCancelled
from orientation import normalize_orientation
from progress_timing import ProgressTiming


def _write(path,value):
    temporary=path.with_suffix('.json.tmp')
    temporary.write_text(json.dumps(value,allow_nan=False),encoding='utf-8')
    temporary.replace(path)


class GosplManager:
    def __init__(self,root,builder=build_history):
        self.root=Path(root);self.root.mkdir(parents=True,exist_ok=True)
        self.builder=builder;self.lock=threading.RLock();self.stop=threading.Event();self.worker=None
        self.current=dict(state='idle',job_id=None,progress=0.)
        self.timing=None
        jobs=self.list_jobs()
        for job in jobs:
            if job['state']=='running':
                job.update(state='error',eta_seconds=None,error='The app closed before this export finished. Build it again from the saved history.')
                _write(self.path(job['job_id'])/'job.json',job)
        if jobs:self.current=jobs[0]

    def path(self,job_id):
        if not isinstance(job_id,str) or not re.fullmatch(r'[A-Za-z0-9_-]{1,80}',job_id):
            raise ValueError('Invalid goSPL export identifier.')
        return self.root/job_id

    def status(self,job_id=None):
        with self.lock:
            if job_id is None or job_id==self.current.get('job_id'):
                result=deepcopy(self.current)
                if self.timing:result.update(self.timing.snapshot())
                if result.get('state')!='running':result['eta_seconds']=0. if result.get('state')=='complete' else None
                else:result.setdefault('eta_seconds',None)
                return result
        try:
            result=json.loads((self.path(job_id)/'job.json').read_text(encoding='utf-8'))
            result['eta_seconds']=0. if result.get('state')=='complete' else None
            return result
        except FileNotFoundError:raise ValueError('goSPL export not found.') from None

    def list_jobs(self):
        result=[]
        with self.lock:
            for path in self.root.glob('*/job.json'):
                try:
                    job=json.loads(path.read_text(encoding='utf-8'))
                    if job.get('job_id')==self.current.get('job_id'):job=self.status()
                    result.append(job)
                except (ValueError,OSError):pass
        return sorted(result,key=lambda item:item.get('created',''),reverse=True)

    def start(self,simulation_manager,run_id=None,start_index=0,end_index=None,
              subdivisions=7,dt_years=100_000,rainfall_m_yr=1.,orientation=None):
        orientation=normalize_orientation(orientation)
        if isinstance(subdivisions,bool) or int(subdivisions)!=subdivisions or subdivisions not in (6,7,8,9):
            raise ValueError('Choose one of the four supported goSPL mesh densities.')
        run_path,manifest=simulation_manager._history_context(run_id)
        if end_index is None:end_index=manifest.get('frame_count',0)-1
        times=checked_times(manifest,start_index,end_index,dt_years)
        rainfall=float(rainfall_m_yr)
        if isinstance(rainfall_m_yr,bool) or not math.isfinite(rainfall) or not 0<=rainfall<=20:
            raise ValueError('Rainfall must be between 0 and 20 metres per year.')
        # Read only the first frame header/archive index here. Large arrays are
        # loaded by the worker, so review and integration remain responsive.
        import numpy as np
        with np.load(run_path/f'frame_{int(start_index):04d}.npz',allow_pickle=False) as frame:
            if 'trace_erosion_m' not in frame.files:
                raise ValueError('This older history has no material erosion counters for evolving goSPL export.')
        with self.lock:
            if self.worker and self.worker.is_alive():raise ValueError('A goSPL export is already being built. Cancel it or let it finish.')
            job_id=datetime.now().strftime('%Y%m%d-%H%M%S')+'-'+uuid.uuid4().hex[:8]
            path=self.path(job_id);path.mkdir()
            self.current=dict(job_id=job_id,state='running',progress=0.,message='Preparing saved history',
                              created=datetime.now(timezone.utc).isoformat(),run_id=manifest['run_id'],
                              start_index=int(start_index),end_index=int(end_index),start_myr=float(times[0]/1e6),end_myr=float(times[-1]/1e6),
                              subdivisions=int(subdivisions),node_count=10*4**int(subdivisions)+2,
                              dt_years=int(dt_years),rainfall_m_yr=rainfall,filename='gospl-history.zip',elapsed_seconds=0.,
                              orientation=orientation,eta_seconds=None)
            self.timing=ProgressTiming()
            _write(path/'job.json',self.current);self.stop.clear()
            self.worker=threading.Thread(target=self._run,args=(run_path,deepcopy(manifest),deepcopy(self.current)),daemon=True)
            self.worker.start()
            return self.status()

    def _run(self,run_path,manifest,job):
        path=self.path(job['job_id']);started=time.perf_counter();last_update=0.
        def progress(value,message):
            nonlocal last_update
            with self.lock:
                self.current.update(progress=float(value),message=message,**self.timing.update(float(value)))
                if time.perf_counter()-last_update>.5:
                    _write(path/'job.json',self.current);last_update=time.perf_counter()
        try:
            metadata=self.builder(run_path,path,manifest,start_index=job['start_index'],end_index=job['end_index'],
                                  subdivisions=job['subdivisions'],dt_years=job['dt_years'],rainfall_m_yr=job['rainfall_m_yr'],
                                  progress=progress,cancel=self.stop.is_set,orientation=job['orientation'])
            with self.lock:
                if self.stop.is_set():raise ExportCancelled()
                self.current.update(state='complete',progress=1.,message='goSPL history package ready',
                                    bytes=(path/'gospl-history.zip').stat().st_size,
                                    **self.timing.finish(),
                                    correction_min_supported_fraction=min(i['relaxation_supported_fraction'] for i in metadata['intervals']))
                _write(path/'job.json',self.current)
        except Exception as exc:
            with self.lock:
                cancelled=isinstance(exc,ExportCancelled)
                self.current.update(state='cancelled' if cancelled else 'error',
                                    message='Export cancelled' if cancelled else 'Export failed',
                                    **self.timing.pause())
                if not cancelled:self.current['error']=f'{type(exc).__name__}: {exc}'
                # Remove only generated large products in this exact job folder.
                if path.resolve().parent != self.root.resolve():raise ValueError('Invalid job cleanup path.')
                for file in [path/'gospl-history.zip',path/'gospl-history.zip.tmp',path/'input'/'mesh.npz',*(path/'input').glob('forcing_*.npz')]:
                    file.unlink(missing_ok=True)
                _write(path/'job.json',self.current)

    def cancel(self,job_id=None):
        with self.lock:
            if job_id is not None and job_id!=self.current.get('job_id'):
                raise ValueError('That export is not the current goSPL build.')
            if self.current.get('state')=='running':
                self.stop.set();self.current['message']='Cancelling after the current mesh or interval operation'
                _write(self.path(self.current['job_id'])/'job.json',self.current)
            return self.status()

    def download(self,job_id):
        status=self.status(job_id)
        if status.get('state')!='complete':raise ValueError('This goSPL export is not complete.')
        path=self.path(job_id)/'gospl-history.zip'
        if not path.is_file():raise ValueError('The goSPL archive is missing.')
        return path
