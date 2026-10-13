"""Reproduce an isolated persistent-intake experiment from saved starting files.

This runs native engine diagnostics, saving source, inputs and every accepted
epoch for review. It neither supplies nor bypasses production guardian receipts.
An optional local viewer follows newly completed snapshots without owning the
simulation. The original dataset is only read; existing run IDs are refused.
"""
from contextlib import ExitStack
from datetime import datetime, timezone
import argparse
import hashlib
import json
from pathlib import Path
import sys
import time
import urllib.error
import urllib.request
from urllib.parse import urlparse

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import budget
import checkpoint
import effective_subduction
import flow_acceleration
import material_acceleration
import server


def publish(url, run_id, first=False):
    if not url:
        return
    try:
        with urllib.request.urlopen(url+'/api/status', timeout=10) as response:
            selected = json.load(response).get('run_id')
        if not first and selected != run_id:
            return  # Respect someone selecting another saved experiment.
        request = urllib.request.Request(url+'/api/load',
            json.dumps(dict(run_id=run_id)).encode(), {'Content-Type': 'application/json'})
        with urllib.request.urlopen(request, timeout=30) as response:
            response.read()
    except (OSError, urllib.error.URLError) as error:
        print('Viewer update unavailable: '+str(error), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', required=True, type=Path)
    parser.add_argument('--initial', required=True, type=Path)
    parser.add_argument('--run-id', required=True)
    parser.add_argument('--output-root', type=Path, default=ROOT/'output/runs')
    parser.add_argument('--end-myr', type=int, default=20)
    parser.add_argument('--workers', type=int, default=3)
    parser.add_argument('--viewer-url', default='')
    args = parser.parse_args()
    if args.end_myr <= 0:
        parser.error('--end-myr must be positive')
    if args.viewer_url:
        target = urlparse(args.viewer_url)
        if target.scheme != 'http' or target.hostname not in ('localhost', '127.0.0.1', '::1') or target.path not in ('', '/'):
            parser.error('--viewer-url must name a local HTTP viewer')
    url = args.viewer_url.rstrip('/')
    original = json.loads(args.config.read_bytes())
    initial_bytes = args.initial.read_bytes()
    config = json.loads(json.dumps(original))
    config['effective_subduction'] = dict(config.get('effective_subduction', {}), enabled=True,
        force_n_per_m=3e13, persistent_intake=dict(enabled=True, minimum_intake_cm_yr=.1,
                                                breakup_mode='on_conflict'))
    config['force_limit_rifting'] = dict(config.get('force_limit_rifting', {}), enabled=False)
    config['duration_myr'] = float(args.end_myr)
    with ExitStack() as stack:
        stack.enter_context(flow_acceleration.reference_backend())
        stack.enter_context(material_acceleration.reference_backend())
        manager = server.SimulationManager(root=args.output_root, workers=args.workers)
        stack.callback(manager.parallel.close)
        stack.enter_context(manager.parallel.activate())
        path = manager.path(args.run_id)
        path.mkdir(exist_ok=False)
        compatibility = manager.compatibility()
        manager.current = dict(run_id=args.run_id, title=args.run_id+' · Persistent subduction · Fracture only when needed',
            state='diagnostic-preview', can_resume=False, preview_only=True, preview_running=True,
            created=datetime.now(timezone.utc).isoformat(), config=config, duration_myr=args.end_myr,
            frames=[], frame_count=0, time_myr=0., progress=0., elapsed_seconds=0., **compatibility,
            note='Isolated native-engine experiment with cooperative step budgets; no production guardian acceptance.')
        server.write_json(path/'config.json', config)
        server.write_json(path/'original-config.json', original)
        (path/'initial.json').write_bytes(initial_bytes)
        (path/'engine.py').write_bytes(server.ENGINE_SOURCE)
        for name, payload in server.AUXILIARY_SOURCES.items():
            (path/name).write_bytes(payload)
        (path/'replay.py').write_bytes(Path(__file__).read_bytes())
        evidence = dict(initial_sha256=hashlib.sha256(initial_bytes).hexdigest(), seed=config['seed'],
            config=config, original_config=original, step_myr=1., compatibility=compatibility, epochs=[])
        started = time.monotonic()
        try:
            print('Constructing unchanged artwork at 0 MY.', flush=True)
            simulation = server.Simulation(config, json.loads(initial_bytes))
            for epoch in range(args.end_myr+1):
                with budget.step_budget(1.):
                    if epoch:
                        print(f'Computing {epoch-1} → {epoch} MY.', flush=True)
                        simulation.step(1.)
                    # Qualify the frame before making its manifest discoverable.
                    frame = simulation.snapshot()
                    server.mesh_history.arrays(frame)
                    effective_subduction.validate_frame(frame)
                    manager.current.update(preview_running=epoch < args.end_myr,
                        preview_message=f'Computing {epoch} → {epoch+1} MY' if epoch < args.end_myr else 'Experiment complete',
                        elapsed_seconds=time.monotonic()-started)
                    checkpoint.write_checkpoint(path/f'diagnostic-{epoch}myr.npz', simulation,
                                                manager.current, compatibility)
                    manager._append_snapshot(simulation)
                    row = dict(time_myr=float(simulation.t), stats=frame['stats'],
                        force_balance=server.native(simulation.plate_balance_diagnostics),
                        faults=server.native(getattr(simulation, 'persistent_intake_faults', [])),
                        elapsed_seconds=time.monotonic()-started)
                    evidence['epochs'].append(row)
                    evidence['completed'] = epoch == args.end_myr
                    server.write_json(path/'experiment.json', evidence)
                publish(url, args.run_id, first=epoch == 0)
                print(json.dumps(dict(time_myr=simulation.t,
                    mean_speed_cm_yr=frame['stats']['mean_plate_speed_cm_yr'],
                    ocean_created_km2=frame['stats']['ocean_created_km2'],
                    ocean_consumed_km2=frame['stats']['ocean_consumed_km2'],
                    faults=len(row['faults']), elapsed_seconds=row['elapsed_seconds'])), flush=True)
            server.write_json(path/'diagnostic-result.json', dict(completed=True, time_myr=simulation.t))
        except BaseException as error:
            manager.current.update(preview_running=False, preview_message='Experiment stopped: '+str(error))
            manager.persist()
            server.write_json(path/'diagnostic-result.json', dict(completed=False, error=str(error),
                                                                type=type(error).__name__))
            publish(url, args.run_id)
            raise


if __name__ == '__main__':
    main()
