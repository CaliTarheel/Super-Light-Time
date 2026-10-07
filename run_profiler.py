"""Bounded, read-only wall-stack sampling outside the scientific state.

Samples are observations, not committed simulation progress or CPU percentages.
Only scalar retry counters are inspected; never serialize arrays or checkpoints.
"""
from collections import Counter
from datetime import datetime, timezone
import json
import math
from pathlib import Path
import sys
import threading
import time
import uuid


def observe(frame):
    stack, retries, stages = [], [], []
    while frame is not None:
        name = Path(frame.f_code.co_filename).name
        function = frame.f_code.co_name
        stack.append(f'{name}:{function}:{frame.f_lineno}')
        if ((name == 'force_rifting.py' and function == 'update') or
                (name == 'plate_limit_analysis.py' and function == 'search_prepared') or
                (name == 'slab_tether_native.py' and function == 'advance')):
            stage = dict(module=name, function=function)
            for key in ('p', 'uid', 'round_index', 'count', 'epoch', 'h', 'target', 'solves'):
                value = frame.f_locals.get(key)
                if type(value) in (int, float) and math.isfinite(value):
                    stage[key] = value
                elif key == 'uid' and type(value) is str:
                    stage[key] = value[:80]
            stages.append(stage)
        if name == 'adaptive_timestepping.py' and function in ('attempt', 'advance'):
            values = frame.f_locals
            row = {'function': function}
            for key in ('interval', 'depth', 'old_time', 'start', 'dt', 'trials'):
                value = values.get(key)
                if type(value) in (int, float) and math.isfinite(value):
                    row[key] = value
            diagnostics = values.get('diagnostics')
            if type(diagnostics) is dict:
                for key in ('accepted_substeps', 'rejected_trials', 'minimum_dt_myr', 'last_rejection'):
                    value = diagnostics.get(key)
                    if type(value) is str:
                        row[key] = value[:1000]
                    elif type(value) in (int, float) and math.isfinite(value):
                        row[key] = value
            retries.append(row)
        frame = frame.f_back
    return dict(stack=stack[:80], adaptive_trials=retries, search_stages=stages,
                progress_scope='Tentative in-memory trials; only saved checkpoints are committed.')


class RunProfiler:
    """Sample the calling simulation thread, with one independent session file.

Failure of diagnostics must never reject a physical step. I/O runs in a daemon
thread. A blocked diagnostics disk cannot block simulation shutdown.
"""
    def __init__(self, directory, interval=2.):
        if not math.isfinite(interval) or interval < .01:
            raise ValueError('Profiler interval must be finite and at least 0.01 seconds.')
        self.directory = Path(directory)
        self.interval = interval
        self.stop = threading.Event()
        self.ready = threading.Event()
        self.thread = None
        self.error = None
        self.samples = 0
        self.leaves = Counter()
        self.session = uuid.uuid4().hex

    def __enter__(self):
        self.target = threading.get_ident()
        self.started = time.monotonic()
        try:
            self.thread = threading.Thread(target=self._run, name='run-profiler', daemon=True)
            self.thread.start()
        except Exception as error:
            self.error = f'{type(error).__name__}: {error}'
        return self

    def __exit__(self, *exception):
        self.stop.set()
        if self.thread is not None and self.thread.ident is not None:
            self.thread.join(timeout=.25)
        return False

    def _run(self):
        try:
            self.directory.mkdir(parents=True, exist_ok=True)
            # Each run/resume has its own bounded summary and sampled history.
            latest = self.directory / f'{self.session}.json'
            temporary = latest.with_suffix('.tmp')
            with (self.directory / f'{self.session}.jsonl').open('a', encoding='utf-8') as stream:
                while not self.stop.is_set():
                    frame = sys._current_frames().get(self.target)
                    record = observe(frame)
                    del frame  # Never retain a simulation frame between samples.
                    self.samples += 1
                    if record['stack']:
                        self.leaves[record['stack'][0]] += 1
                    record.update(version=1, session=self.session,
                        utc=datetime.now(timezone.utc).isoformat(),
                        elapsed_seconds=time.monotonic()-self.started, samples=self.samples,
                        interval_seconds=self.interval, metric='simulation-thread wall-stack samples',
                        top_locations=self.leaves.most_common(20))
                    payload = json.dumps(record, allow_nan=False)
                    stream.write(payload+'\n')
                    stream.flush()
                    temporary.write_text(payload+'\n', encoding='utf-8')
                    temporary.replace(latest)
                    self.ready.set()
                    if self.stop.wait(self.interval):
                        break
        except Exception as error:
            self.error = f'{type(error).__name__}: {error}'
            print(f'Run profiler disabled: {self.error}', file=sys.stderr, flush=True)
        finally:
            self.ready.set()
