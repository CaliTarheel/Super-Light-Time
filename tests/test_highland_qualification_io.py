"""Qualification control/IO tests; simulated engines are mocks, not physics evidence."""
from copy import deepcopy
import contextlib
import io
import json
import math
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
import highland_qualification as q
from tools import qualify_highland65 as cli


def sample(time=0.,**changes):
    row=dict(time_myr=time,ocean_fraction=.35,active_plates=8,
             largest_continent_fraction=.2,rift_breakthrough_count=1,
             ocean_created_km2=1e6,max_plate_speed_cm_yr=5.,
             stalled_trench_length_km=0.,boundary_length_km={'subduction':8000.},
             slab_ledger=dict(area_relative_residual=1e-12,mass_relative_residual=2e-12))
    row.update(changes)
    return row


def finished(seed=41,**changes):
    row=dict(version=1,seed=seed,status='completed',samples=[sample(500.)])
    row.update(changes)
    row['qualification']=q.evaluate(row['samples'],status=row['status'],requested_duration_myr=500.)
    return row


class SimulatedEngine:
    def __init__(self,action=None,mutate=False):
        self.action=action;self.mutate=mutate;self.calls=0;self.t=0.;self.config={}

    def Simulation(self,config,initial):
        if self.mutate:config['adopted_from_world']={'enabled':True}
        self.config=deepcopy(config)
        return self

    def make_initial(self,config,preset):
        assert preset=='highland65'
        return {}

    def step(self,dt):
        self.calls+=1
        if self.calls>4:
            raise RuntimeError('Test safety guard: repeated non-progressing engine call.')
        if self.action is not None:return self.action(self,dt)
        self.t+=dt
        return {'ruptures':[]}


class QualificationIOTests(unittest.TestCase):
    def run_cli(self,args,**kwargs):
        with contextlib.redirect_stdout(io.StringIO()),contextlib.redirect_stderr(io.StringIO()):
            return cli.main(args,**kwargs)

    def run_mock(self,engine,**kwargs):
        with patch.dict('sys.modules',{'native_engine':engine}),patch.object(
                q,'measure',side_effect=lambda s,**unused:sample(float(s.t))):
            return q.run_seed(41,duration_myr=4.,sample_myr=2.,**kwargs)

    def test_json_tags_nan_and_infinities_without_mutating_source(self):
        raw={'a':float('nan'),'b':np.array([float('inf'),-float('inf')]),'n':np.int64(7)}
        safe=cli.json_safe(raw)
        self.assertEqual(safe['a'],{'$nonfinite':'NaN'})
        self.assertEqual(safe['b'],[{'$nonfinite':'Infinity'},{'$nonfinite':'-Infinity'}])
        self.assertEqual(safe['n'],7)
        self.assertTrue(math.isnan(raw['a']))
        json.dumps(safe,allow_nan=False)

    def test_cli_writes_nonfinite_failed_seed(self):
        bad=finished(samples=[sample(500.,ocean_fraction=float('nan'))])
        with TemporaryDirectory() as d,patch.object(q,'run_seed',return_value=bad):
            path=Path(d)/'report.json'
            self.assertEqual(self.run_cli(['--seeds','41','--output',str(path)]),1)
            report=json.loads(path.read_text(),parse_constant=lambda value:self.fail(value))
            self.assertFalse(report['passed'])
            self.assertIn('nonfinite_metric',report['runs'][0]['qualification']['failures'])
            self.assertEqual(report['runs'][0]['samples'][0]['ocean_fraction'],{'$nonfinite':'NaN'})

    def test_atomic_replace_error_keeps_previous_evidence(self):
        with TemporaryDirectory() as d:
            path=Path(d)/'report.json';path.write_text('old evidence')
            with patch.object(cli.os,'replace',side_effect=OSError('disk failure')):
                with self.assertRaises(OSError):cli.write_report(path,{'new':True})
            self.assertEqual(path.read_text(),'old evidence')
            self.assertEqual(list(Path(d).iterdir()),[path])

    def test_serialization_error_keeps_previous_evidence(self):
        with TemporaryDirectory() as d:
            path=Path(d)/'report.json';path.write_text('old evidence')
            with self.assertRaises(TypeError):cli.write_report(path,{'bad':object()})
            self.assertEqual(path.read_text(),'old evidence')

    def test_atomic_report_replaces_successfully(self):
        with TemporaryDirectory() as d:
            path=Path(d)/'nested/report.json'
            cli.write_report(path,{'a':1})
            cli.write_report(path,{'b':2})
            self.assertEqual(json.loads(path.read_text()),{'b':2})
            self.assertEqual(list(path.parent.iterdir()),[path])

    def test_invalid_cli_schedule_rejected_without_run(self):
        for option,value in (('--duration-myr','nan'),('--dt-myr','inf'),
                             ('--sample-myr','-inf'),('--sample-myr','0'),('--dt-myr','0')):
            with self.subTest(option=option,value=value),patch.object(q,'run_seed') as run:
                with self.assertRaises(SystemExit) as error:self.run_cli([option+'='+value])
                self.assertEqual(error.exception.code,2)
                run.assert_not_called()

    def test_invalid_api_schedule_rejected(self):
        for name in ('duration_myr','sample_myr','dt_myr'):
            for value in (float('nan'),float('inf'),-1.,0.,True):
                with self.subTest(name=name,value=value),self.assertRaises(ValueError):
                    q.qualification_config(41,**{name:value})

    def test_invalid_seed_rejected(self):
        for value in (-1,2**32,1.2,True):
            with self.subTest(seed=value),self.assertRaises(ValueError):q.qualification_config(value)

    def test_seed_parser_bounds_range_before_expansion(self):
        for value in ('0-4294967295','4294967296','-1','','hello','1-2-3'):
            with self.subTest(value=value),self.assertRaises(cli.argparse.ArgumentTypeError):cli.seeds(value)
        self.assertEqual(cli.seeds('3-1,41'),[3,2,1,41])

    def test_completion_and_sampling_progress(self):
        updates=[];engine=SimulatedEngine()
        result=self.run_mock(engine,progress=updates.append)
        self.assertEqual(result['status'],'completed')
        self.assertTrue(result['qualification']['passed'])
        self.assertEqual([r['time_myr'] for r in result['samples']],[0.,2.,4.])
        self.assertEqual(engine.calls,2)
        self.assertEqual(updates[0]['samples'],[])
        self.assertEqual(updates[-1],result)

    def test_progress_payload_is_detached(self):
        def corrupt(row):row['requested_config']['seed']=999;row['samples'].clear()
        result=self.run_mock(SimulatedEngine(),progress=corrupt)
        self.assertEqual(result['requested_config']['seed'],41)
        self.assertEqual(len(result['samples']),3)

    def test_failed_seed_keeps_realized_config(self):
        def fail(s,dt):raise ValueError('geometry failed')
        result=self.run_mock(SimulatedEngine(fail,mutate=True))
        self.assertEqual(result['status'],'failed')
        self.assertNotIn('adopted_from_world',result['requested_config'])
        self.assertTrue(result['realized_config']['adopted_from_world']['enabled'])
        self.assertEqual(len(result['samples']),1)
        self.assertIn('geometry failed',result['error']['message'])
        self.assertIn('traceback',result['error'])

    def test_engine_import_failure_is_recorded(self):
        with patch.dict('sys.modules',{'native_engine':None}):result=q.run_seed(41)
        self.assertEqual(result['status'],'failed')
        self.assertEqual(result['error']['type'],'ModuleNotFoundError')
        self.assertIn('no_samples',result['qualification']['failures'])

    def test_nonprogress_and_nonfinite_time_fail_immediately(self):
        for time in (0.,-1.,float('nan'),float('inf')):
            def action(s,dt):s.t=time
            engine=SimulatedEngine(action)
            with self.subTest(time=time):
                result=self.run_mock(engine)
                self.assertEqual(engine.calls,1)
                self.assertEqual(result['status'],'failed')
                self.assertIn('did not advance finite',result['error']['message'])

    def test_overstepping_interval_fails(self):
        def action(s,dt):s.t+=dt+1.
        result=self.run_mock(SimulatedEngine(action))
        self.assertEqual(result['status'],'failed')
        self.assertIn('exceeded its requested interval',result['error']['message'])

    def test_keyboard_interrupt_retains_active_seed(self):
        def action(s,dt):raise KeyboardInterrupt
        result=self.run_mock(SimulatedEngine(action,mutate=True))
        self.assertEqual(result['status'],'interrupted')
        self.assertFalse(result['qualification']['passed'])
        self.assertEqual(len(result['samples']),1)
        self.assertIn('realized_config',result)

    def test_nonfinite_sample_is_preserved_and_stops_run(self):
        engine=SimulatedEngine();updates=[]
        with patch.dict('sys.modules',{'native_engine':engine}),patch.object(
                q,'measure',return_value=sample(ocean_fraction=float('nan'))):
            result=q.run_seed(41,progress=updates.append)
        self.assertEqual(engine.calls,0)
        self.assertEqual(result['status'],'failed')
        self.assertTrue(math.isnan(result['samples'][0]['ocean_fraction']))
        self.assertIn('nonfinite_metric',result['qualification']['failures'])
        self.assertGreaterEqual(len(updates),3)

    def test_writer_failure_is_not_a_simulation_failure(self):
        engine=SimulatedEngine()
        with self.assertRaises(q._ProgressError):
            self.run_mock(engine,progress=lambda row:(_ for _ in ()).throw(OSError('disk full')))
        self.assertEqual(engine.calls,0)

    def test_midrun_writer_failure_propagates(self):
        def writer(row):
            if row['samples']:raise OSError('disk full')
        with self.assertRaises(q._ProgressError):self.run_mock(SimulatedEngine(),progress=writer)

    def test_cli_preserves_completed_seed_during_next_seed(self):
        with TemporaryDirectory() as d:
            path=Path(d)/'report.json'
            def run(seed,**kwargs):
                kwargs['progress']({'seed':seed,'status':'running','samples':[sample()]})
                saved=json.loads(path.read_text())
                self.assertEqual(saved['active_run']['seed'],seed)
                self.assertIsNone(saved['passed'])
                self.assertFalse(saved['complete'])
                self.assertEqual(len(saved['runs']),0 if seed==12 else 1)
                return finished(seed)
            with patch.object(q,'run_seed',side_effect=run):
                self.assertEqual(self.run_cli(['--seeds','12,41','--output',str(path)]),0)
            saved=json.loads(path.read_text())
            self.assertEqual([r['seed'] for r in saved['runs']],[12,41])
            self.assertTrue(saved['passed']);self.assertTrue(saved['complete'])
            self.assertIsNone(saved['active_run'])

    def test_cli_interrupt_retains_previous_seed_and_active_samples(self):
        with TemporaryDirectory() as d:
            path=Path(d)/'report.json'
            def run(seed,**kwargs):
                kwargs['progress']({'seed':seed,'status':'running','samples':[sample()]})
                if seed==41:raise KeyboardInterrupt
                return finished(seed)
            with patch.object(q,'run_seed',side_effect=run):
                self.assertEqual(self.run_cli(['--seeds','12,41,81','--output',str(path)]),130)
            saved=json.loads(path.read_text())
            self.assertEqual(saved['status'],'interrupted');self.assertFalse(saved['complete'])
            self.assertEqual(len(saved['runs']),1)
            self.assertEqual(saved['active_run']['seed'],41)
            self.assertEqual(len(saved['active_run']['samples']),1)

    def test_cli_interrupted_result_stops_remaining_seeds(self):
        with TemporaryDirectory() as d,patch.object(q,'run_seed',return_value=finished(status='interrupted')) as run:
            path=Path(d)/'report.json'
            self.assertEqual(self.run_cli(['--seeds','12,41','--output',str(path)]),130)
            self.assertEqual(run.call_count,1)
            saved=json.loads(path.read_text())
            self.assertFalse(saved['passed']);self.assertFalse(saved['complete'])
            self.assertEqual(saved['status'],'interrupted')

    def test_failed_seed_is_not_dropped_from_campaign(self):
        results=[finished(12,status='failed'),finished(41)]
        with TemporaryDirectory() as d,patch.object(q,'run_seed',side_effect=results):
            path=Path(d)/'report.json'
            self.assertEqual(self.run_cli(['--seeds','12,41','--output',str(path)]),1)
            saved=json.loads(path.read_text())
            self.assertEqual(len(saved['runs']),2)
            self.assertTrue(saved['complete']);self.assertFalse(saved['passed'])

    def test_quick_mode_keeps_documented_settings(self):
        with TemporaryDirectory() as d,patch.object(q,'run_seed',return_value=finished()) as run:
            self.run_cli(['--quick','--output',str(Path(d)/'report.json')])
            args=run.call_args
            self.assertEqual(args.args,(41,))
            for key,value in dict(duration_myr=20.,sample_myr=5.,mesh_level=3,
                                  coast_geometry_level=3,mechanics_nodes=256).items():
                self.assertEqual(args.kwargs[key],value)

    def test_existing_report_requires_explicit_overwrite(self):
        with TemporaryDirectory() as d,patch.object(q,'run_seed') as run:
            path=Path(d)/'report.json';path.write_text('previous campaign')
            with self.assertRaises(SystemExit) as error:
                self.run_cli(['--seeds','41','--output',str(path)])
            self.assertEqual(error.exception.code,2)
            self.assertEqual(path.read_text(),'previous campaign')
            run.assert_not_called()

    def test_explicit_overwrite_is_supported(self):
        with TemporaryDirectory() as d,patch.object(q,'run_seed',return_value=finished()):
            path=Path(d)/'report.json';path.write_text('previous campaign')
            self.assertEqual(self.run_cli(['--seeds','41','--output',str(path),'--overwrite']),0)
            self.assertTrue(json.loads(path.read_text())['passed'])

    def test_policy_overrides_still_merge_with_defaults(self):
        result=q.evaluate([sample(500.,ocean_fraction=.1)],requested_duration_myr=500.,
                          policy={'minimum_ocean_fraction':.2})
        self.assertIn('ocean_floor_lost',result['failures'])
        self.assertEqual(result['policy']['minimum_active_plates'],2)


if __name__=='__main__':unittest.main()
