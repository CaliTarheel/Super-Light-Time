"""The read-only plausibility audit measures what the frames actually record."""
from pathlib import Path
import json
import tempfile
import unittest

import run_audit


def write_run(directory, frames, extra=None):
    """Write a minimal manifest and the JSON side of each frame."""
    directory = Path(directory)
    manifest = dict(run_id='test-run', state='paused',
                    frames=[dict(index=i, time_myr=row['time_myr'], stats=row['stats'])
                            for i, row in enumerate(frames)])
    (directory/'manifest.json').write_text(json.dumps(manifest), encoding='utf-8')
    for i, row in enumerate(frames):
        payload = dict(index=i, time_myr=row['time_myr'], stats=row['stats'], **(row.get('frame') or {}))
        (directory/f'frame_{i:04d}.json').write_text(json.dumps(payload), encoding='utf-8')
    if extra:
        (directory/extra).write_text('x', encoding='utf-8')
    return directory


def stats(**overrides):
    base = dict(mean_plate_speed_cm_yr=4.5, max_plate_speed_cm_yr=9., mean_ocean_age_myr=64.,
                active_plates=14, land_fraction=.29, max_elevation_m=8000.,
                ridge_length_km=60000., subduction_length_km=50000., transform_length_km=40000.,
                collision_length_km=8000., ocean_created_km2=10e6, ocean_consumed_km2=9.5e6,
                rift_events_breakup=0, collision_events=0, accreted_km2=0.)
    base.update(overrides)
    return base


def earthlike_frame(speeds=None, attached_uid=1):
    speeds = speeds or {'Ocean': 8., 'Continent': 2.}
    return dict(plate_balance_diagnostics=dict(mean_speed_cm_yr=speeds, net_rotation_deg_myr=.18,
                                               net_rotation_bound_deg_myr=.26),
                plates=[dict(uid=1, name='Ocean'), dict(uid=2, name='Continent')],
                trench_systems=[dict(phase='mature', downgoing_plate_uid=attached_uid,
                                     overriding_plate_uid=2, slab_retained_excess_mass_kg=4e18)])


class RunAuditTests(unittest.TestCase):
    def test_earthlike_run_passes_every_band(self):
        with tempfile.TemporaryDirectory() as tmp:
            run = write_run(tmp, [
                dict(time_myr=0., stats=stats(ocean_created_km2=0., ocean_consumed_km2=0.), frame=earthlike_frame()),
                dict(time_myr=100., stats=stats(rift_events_breakup=3, collision_events=6, accreted_km2=2e6),
                     frame=earthlike_frame())])
            result = run_audit.audit(run)
            self.assertEqual(result['summary']['failed'], 0, result['measures'])
            self.assertEqual(result['summary']['unavailable'], 0)
            measures = result['measures']
            self.assertEqual(measures['continental_breakups_per_100myr']['value'], 3.)
            self.assertEqual(measures['collisions_per_100myr']['value'], 6.)
            self.assertAlmostEqual(measures['accreted_mkm2_per_100myr']['value'], 2.)
            self.assertAlmostEqual(measures['subduction_over_ridge_length']['value'], 50000./60000.)
            self.assertAlmostEqual(measures['ocean_consumed_over_created']['value'], .95)
            self.assertAlmostEqual(measures['unexplained_ocean_fraction']['value'], .05)
            self.assertEqual(measures['net_rotation_deg_myr']['value'], .18)
            self.assertAlmostEqual(measures['slab_attached_speed_ratio']['value'], 4.)

    def test_the_live_run_symptoms_are_reported_as_failures(self):
        # The values that motivated this audit: slow plates, old ocean, ocean
        # created without a named sink, no accretion, net rotation far below.
        with tempfile.TemporaryDirectory() as tmp:
            slow = dict(plate_balance_diagnostics=dict(mean_speed_cm_yr={'Ocean': 1.1, 'Continent': 1.},
                                                       net_rotation_deg_myr=.006, net_rotation_bound_deg_myr=.26),
                        plates=[dict(uid=1, name='Ocean'), dict(uid=2, name='Continent')],
                        trench_systems=[dict(phase='mature', downgoing_plate_uid=1,
                                             overriding_plate_uid=2, slab_retained_excess_mass_kg=1e18)])
            run = write_run(tmp, [
                dict(time_myr=0., stats=stats(ocean_created_km2=0., ocean_consumed_km2=0.), frame=slow),
                dict(time_myr=100., stats=stats(mean_plate_speed_cm_yr=.98, max_plate_speed_cm_yr=4.6,
                                                mean_ocean_age_myr=170., ocean_created_km2=115e6,
                                                ocean_consumed_km2=35e6, accreted_km2=1500.), frame=slow)])
            measures = run_audit.audit(run)['measures']
            self.assertEqual(measures['mean_plate_speed_cm_yr']['verdict'], 'fail')
            self.assertEqual(measures['mean_ocean_age_myr']['verdict'], 'warn')
            self.assertEqual(measures['unexplained_ocean_fraction']['verdict'], 'fail')
            self.assertEqual(measures['accreted_mkm2_per_100myr']['verdict'], 'fail')
            self.assertEqual(measures['net_rotation_deg_myr']['verdict'], 'fail')
            self.assertEqual(measures['slab_attached_speed_ratio']['verdict'], 'warn')
            self.assertGreater(measures['unexplained_ocean_fraction']['detail']['unexplained_mkm2'], 79.)

    def test_verdict_bands_are_two_sided_with_a_warning_margin(self):
        band = (1., 2., 'reference')
        self.assertEqual(run_audit._verdict(1.5, band), 'pass')
        self.assertEqual(run_audit._verdict(.7, band), 'warn')     # within a factor of two below
        self.assertEqual(run_audit._verdict(.4, band), 'fail')
        self.assertEqual(run_audit._verdict(3.5, band), 'warn')    # within a factor of two above
        self.assertEqual(run_audit._verdict(5., band), 'fail')
        self.assertEqual(run_audit._verdict(None, band), 'unavailable')
        self.assertEqual(run_audit._verdict(float('nan'), band), 'unavailable')

    def test_missing_diagnostics_are_unavailable_not_invented(self):
        with tempfile.TemporaryDirectory() as tmp:
            bare = stats()
            for key in ('mean_ocean_age_myr', 'ridge_length_km', 'ocean_created_km2', 'land_fraction'):
                bare.pop(key)
            run = write_run(tmp, [dict(time_myr=0., stats=bare), dict(time_myr=10., stats=bare)])
            result = run_audit.audit(run)
            for name in ('mean_ocean_age_myr', 'subduction_over_ridge_length',
                         'ocean_consumed_over_created', 'land_fraction', 'net_rotation_deg_myr',
                         'slab_attached_speed_ratio'):
                self.assertEqual(result['measures'][name]['verdict'], 'unavailable', name)
            self.assertIn('unavailable', run_audit.format_report(result))

    def test_rates_use_the_requested_window_and_survive_a_short_history(self):
        with tempfile.TemporaryDirectory() as tmp:
            rows = [dict(time_myr=float(t), stats=stats(collision_events=t//10)) for t in range(0, 301, 10)]
            run = write_run(tmp, rows)
            # 1 event per 10 Myr = 10 per 100 Myr, whatever the window.
            self.assertAlmostEqual(run_audit.audit(run, window_myr=100.)['measures']['collisions_per_100myr']['value'], 10.)
            self.assertAlmostEqual(run_audit.audit(run, window_myr=50.)['measures']['collisions_per_100myr']['value'], 10.)
        with tempfile.TemporaryDirectory() as tmp:
            run = write_run(tmp, [dict(time_myr=0., stats=stats(collision_events=0)),
                                  dict(time_myr=2., stats=stats(collision_events=1))])
            self.assertAlmostEqual(run_audit.audit(run)['measures']['collisions_per_100myr']['value'], 50.)

    def test_plate_absorption_is_noticed_and_its_absence_warned(self):
        with tempfile.TemporaryDirectory() as tmp:
            run = write_run(tmp, [dict(time_myr=0., stats=stats(active_plates=12)),
                                  dict(time_myr=50., stats=stats(active_plates=15)),
                                  dict(time_myr=100., stats=stats(active_plates=13))])
            row = run_audit.audit(run)['measures']['plate_count_decreases']
            self.assertEqual((row['value'], row['verdict']), (1, 'pass'))
        with tempfile.TemporaryDirectory() as tmp:
            run = write_run(tmp, [dict(time_myr=0., stats=stats(active_plates=12)),
                                  dict(time_myr=100., stats=stats(active_plates=20))])
            row = run_audit.audit(run)['measures']['plate_count_decreases']
            self.assertEqual((row['value'], row['verdict']), (0, 'warn'))

    def test_audit_reads_and_never_writes(self):
        with tempfile.TemporaryDirectory() as tmp:
            run = write_run(tmp, [dict(time_myr=0., stats=stats(), frame=earthlike_frame()),
                                  dict(time_myr=100., stats=stats(), frame=earthlike_frame())])
            before = {p.name: p.stat().st_mtime_ns for p in run.iterdir()}
            run_audit.audit(run)
            after = {p.name: p.stat().st_mtime_ns for p in run.iterdir()}
            self.assertEqual(before, after)

    def test_frame_sampling_always_includes_the_last_frame(self):
        self.assertEqual(run_audit.sample_indices(5, 12), [0, 1, 2, 3, 4])
        for count, wanted in ((78, 12), (7, 3), (100, 1), (2, 5)):
            picked = run_audit.sample_indices(count, wanted)
            self.assertEqual(picked, sorted(set(picked)))
            self.assertIn(count-1, picked)
            self.assertTrue(all(0 <= i < count for i in picked))
        self.assertEqual(run_audit.sample_indices(0, 4), [])

    def test_a_run_with_no_frames_is_refused(self):
        with tempfile.TemporaryDirectory() as tmp:
            (Path(tmp)/'manifest.json').write_text(json.dumps(dict(run_id='empty', frames=[])), encoding='utf-8')
            with self.assertRaises(ValueError):
                run_audit.audit(tmp)


if __name__ == '__main__':
    unittest.main()
