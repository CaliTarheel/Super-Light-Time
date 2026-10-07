"""Reproducible component trajectories using the actual mechanical operators.

Run from a clean checkout: python -m benchmarks.collision_architecture.runner
--case symmetric --dt 1 --end 12 --output /chosen/directory

The manufactured basal/driver terms and omitted assemblies are explicit. A
small plate residual does NOT measure combined plate/sheet force closure.
"""
from __future__ import annotations

import argparse
from copy import deepcopy
import hashlib
import json
import os
from pathlib import Path
import platform
import subprocess
import time
import traceback
from types import SimpleNamespace

import numpy as np

import checkpoint
import collision_contacts as contacts
import collision_interface
import crustal_structure as columns
import enhanced_rifting
import material_surface as surface
import mesh_coverage
import native_material_evolution as evolution
import plate_balance as balance
import structure_engine

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
TOL = json.loads((HERE / 'tolerances.json').read_text())
MIRROR = np.diag([1., -1., -1.])  # Proper rotation, including triangle winding.
CASES = ('rest', 'symmetric', 'weak0', 'weak1')
POLICIES = dict(material_mechanics_version=1, native_gravity_constraint_version=1,
                native_contact_constraint_version=1, complete_contact_version=1,
                same_sheet_nonpenetration_version=1, plate_balance_version=1,
                suture_weld_version=1, suture_weld_coordinate_version=1)
EXPECTED_POLICIES = dict(POLICIES, collision_interface_version=1,
                         enhanced_rifting_version=1, plate_resistance_version=0)
SCOPE = dict(
    type='force-driven component trajectory; not Simulation.step',
    included=['contact history', 'rigid GPE share', 'local opening weld',
              'area interface shear', 'material sheet advection',
              'residual GPE relaxation', 'geometric columns and thermal height'],
    omitted=['ocean and trench lifecycle', 'slab pull and inherited polarity',
             'entry and rupture', 'consolidation and remeshing', 'dense phase',
             'erosion and exposed surface', 'rift lifecycle'],
    basal='manufactured isotropic generalized stiffness 1e9 W; fixed driver K*x_free',
    weld_strength='no control edges; production default boundary strength 1',
    strength_contrast='ordinary sheet viscosity only; not weld-strength mapping',
    combined_force_residual=None, combined_work_closure=None,
    missing='dimensionally calibrated reciprocal sheet reaction in plate solve')


def digest(array):
    a = np.ascontiguousarray(array)
    return hashlib.sha256(str((a.dtype.str, a.shape)).encode()+a.tobytes()).hexdigest()


def source_manifest():
    # Hash actual bytes, including uncommitted/untracked benchmark files.
    paths = sorted(set(ROOT.glob('*.py')) | set(HERE.glob('*.py')) | {HERE/'tolerances.json'})
    hashes = {p.relative_to(ROOT).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}
    def git(*args):
        try:
            return subprocess.check_output(['git', '-c', f'safe.directory={ROOT.as_posix()}',
                '-C', str(ROOT), *args], stderr=subprocess.DEVNULL).decode().strip()
        except (OSError, subprocess.CalledProcessError):
            return None
    closure = hashlib.sha256(json.dumps(hashes, sort_keys=True).encode()).hexdigest()
    return dict(commit=git('rev-parse', 'HEAD'), status=git('status', '--porcelain'),
                source_hashes=hashes, source_closure_sha256=closure,
                python=platform.python_version(), numpy=np.__version__,
                platform=platform.platform(), logical_cpus=os.cpu_count(),
                thread_environment={k: os.environ.get(k) for k in
                    ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS')})


def compatibility(manifest):
    return dict(engine_sha256=manifest['source_hashes']['native_material_evolution.py'],
                auxiliary_sources_sha256=manifest['source_closure_sha256'], numpy_version=np.__version__)


def fixture(case='symmetric', *, nx=5, ny=5, rotation=None):
    if case not in CASES or nx < 2 or ny < 2:
        raise ValueError('Choose a supported case and at least two nodes per axis.')
    rotation = np.eye(3) if rotation is None else np.asarray(rotation, float)
    if rotation.shape != (3, 3) or not np.allclose(rotation.T@rotation, np.eye(3), atol=1e-12) or not np.isclose(np.linalg.det(rotation), 1.):
        raise ValueError('Fixture orientation must be a proper rotation.')
    x, y = np.meshgrid(np.linspace(-900., -75., nx), np.linspace(-400., 400., ny))
    points = np.column_stack((np.ones(x.size), x.ravel()/6371., y.ravel()/6371.))
    points /= np.linalg.norm(points, axis=1)[:, None]
    faces = []
    for row in range(ny-1):
        for col in range(nx-1):
            a = row*nx+col
            faces.extend(((a, a+1, a+1+nx), (a, a+1+nx, a+nx)))
    faces = np.asarray(faces, int)
    mesh = surface.initialize_surface(points@rotation.T, faces, np.zeros(len(faces), int), np.ones(len(faces), int))
    surface.append_surface(mesh, points@MIRROR@rotation.T, faces, np.ones(len(faces), int), np.ones(len(faces), int))
    n = len(mesh['faces'])
    s = SimpleNamespace(material_surface=mesh, parcel_patch=mesh['face_id'].copy(),
        parcel_plate=mesh['face_owner'].copy(), kind=mesh['face_kind'].copy(),
        mass=mesh['reference_area_km2'].copy(), pos=surface.face_centres(mesh),
        relief=np.full(n, 780.), suture=np.zeros(n), plate_uid=np.array([101, 202], np.int64),
        t=0., omega=np.array([[0., 0., .003], [0., 0., -.003]])@rotation.T,
        config=dict(deforming_regions=1, deformation_width_km=450.,
                    enhanced_rifting=enhanced_rifting.normalize(dict(enabled=True, amplitude=0.))),
        bmid=np.empty((0, 3)), bn=np.empty((0, 3)), bl=np.empty(0), bp=np.empty(0, int), bq=np.empty(0, int),
        trace_xyz=np.empty((0, 3)), trace_patch=np.empty(0, np.int64),
        trace_rift_tangent=np.empty((0, 3)), rift_tangent=np.tile(np.array([0., 0., 1.])@rotation.T, (n, 1)),
        rng=np.random.default_rng(37), **POLICIES)
    if case == 'rest':
        s.omega[:] = 0.
    structure_engine.initialize_parcels(s)
    enhanced_rifting.initialize(s)
    if case.startswith('weak'):
        s.parcel_rift_seed_weakness[s.parcel_plate == int(case[-1])] = .6
    contacts.refresh(s)
    collision_interface.upgrade(s)
    s.benchmark = dict(case=case, free_omega=s.omega.copy(), initial_vertices=mesh['vertices'].copy(),
        initial_area=mesh['area_km2'].copy(), initial_thickness=s.structure['thickness_km'].copy(),
        initial_volume=mesh['area_km2']*s.structure['thickness_km'],
        initial_ids=s.parcel_patch.copy(), initial_owner=s.parcel_plate.copy(), initial_weights=s.mass.copy(),
        initial_viscosity=enhanced_rifting.material_viscosity(s).copy(),
        absolute_strain=np.zeros(n), last_step=None, rows=[], orientation=rotation.copy())
    return s


def assemble(s):
    """Partial Balance with declared manufactured loading; no hidden test imports."""
    model = balance.Balance.__new__(balance.Balance)
    model.s = s
    model.plates = [0, 1]
    model.slot = {0: 0, 1: 1}
    model.size = 6
    model.resistance_version = balance.resistance_version(s)
    model.bp = model.bq = np.empty(0, int)
    model.strength = model.length_m = np.empty(0)
    model.elements = []
    model.notes = {}
    model.stiffness = np.zeros((6, 6))
    model.hinge_coefficient = np.empty(0)
    model._weld()
    collision_interface.assemble(model)
    basal = np.eye(6)*1e9
    model.stiffness += basal
    scale = balance.CM_YR_M_S/balance.RADIUS_M*balance.SECONDS_PER_MYR
    external = basal@(s.benchmark['free_omega'].ravel()/scale)
    gpe = model._collision_torque()
    model.torque = external+gpe
    model.solve()
    s.omega = model.rotation()
    return model, external, gpe, basal


def gates(s, row):
    b = s.benchmark
    mesh = s.material_surface
    actual_area = surface.spherical_face_areas(mesh['vertices'], mesh['faces'], mesh['radius_km'])
    volumes = actual_area*s.structure['thickness_km']
    local = np.abs(volumes-b['initial_volume']) <= TOL['volume_atol_km3']+TOL['volume_rtol']*b['initial_volume']
    aggregates = [np.ones(len(volumes), bool), *(b['initial_owner'] == p for p in (0, 1))]
    inventory = all(abs((volumes-b['initial_volume'])[m].sum()) <= TOL['volume_atol_km3']+TOL['volume_rtol']*b['initial_volume'][m].sum() for m in aggregates)
    power = row['resisting_power_w']
    floor = TOL['passive_power_atol_w']+TOL['passive_power_rtol']*sum(abs(v) for v in power.values())
    sheet = (b.get('last_step') or {}).get('solver')
    out = dict(local_volume=bool(local.all()), aggregate_volume=bool(inventory),
        area_cache=bool(np.allclose(actual_area, mesh['area_km2'], rtol=TOL['volume_rtol'], atol=0.)),
        persistent_material=all(np.array_equal(a, c) for a, c in
            ((s.mass, b['initial_weights']), (s.parcel_patch, b['initial_ids']), (s.parcel_plate, b['initial_owner']))),
        plate_stationarity=row['plate_relative_residual'] <= balance.FORCE_RELATIVE_TOLERANCE,
        sheet_stationarity=(b['last_step'] is None and s.t == 0.) or
            (sheet is not None and bool(sheet['converged']) and sheet['relative_residual'] is not None and sheet['relative_residual'] <= TOL['sheet_residual']),
        policies=all(getattr(s, k, 0) == v for k, v in EXPECTED_POLICIES.items()),
        passivity=all(v >= -floor for v in power.values()) and row['minimum_local_resisting_power_w'] >= -floor,
        no_same_sheet_overlap=row['same_sheet_overlap_pairs'] == 0,
        positive_geometry=row['minimum_signed_triangle'] > 0 and row['minimum_area_km2'] > 0,
        column_bounds=bool(np.all((s.structure['thickness_km'] >= columns.MIN_THICKNESS_KM-1e-10) & (s.structure['thickness_km'] <= columns.MAX_THICKNESS_KM+1e-10))))
    if b['case'] == 'rest':
        out['rest'] = row['maximum_displacement_km']/6371. <= TOL['rest_position_rad'] and row['maximum_thickness_change_km'] <= TOL['rest_thickness_km'] and row['overlap_km2'] == 0.
    return out


def observe(s):
    contacts.refresh(s)
    model, external, gpe, basal = assemble(s)
    contacts.refresh(s)
    delta = balance.HUBER_CONTINUATION_KM_MYR[-1]*balance.KM_MYR_CM_YR
    gradient = model._evaluate(model.x, delta)[1]
    residual = model._relative_residual(gradient)  # Must use the fresh evaluator's reference.
    powers, passive = model._resistance_work(model.x, delta)
    powers.update(basal=float(model.x@basal@model.x), interface=float(model.x@model.interface_stiffness@model.x))
    mesh, b = s.material_surface, s.benchmark
    area = surface.spherical_face_areas(mesh['vertices'], mesh['faces'], mesh['radius_km'])
    h = s.structure['thickness_km']
    same = mesh_coverage.same_sheet_overlaps(mesh['vertices'], mesh['faces'], s.parcel_collision_sheet)
    triangles = mesh['vertices'][mesh['faces']]
    angular = np.arctan2(np.linalg.norm(np.cross(b['initial_vertices'], mesh['vertices']), axis=1), np.einsum('ij,ij->i', b['initial_vertices'], mesh['vertices']))
    vis = enhanced_rifting.material_viscosity(s)
    side = []
    for p in (0, 1):
        m = b['initial_owner'] == p
        w = b['initial_area'][m]
        side.append(dict(volume_km3=float(area[m]@h[m]),
            mean_thickness_km=float(np.average(h[m], weights=w)),
            signed_log_area=float(np.average(np.log(area[m]/w), weights=w)),
            accumulated_absolute_log_area=float(np.average(b['absolute_strain'][m], weights=w)),
            viscosity_min=float(vis[m].min()), viscosity_max=float(vis[m].max())))
    row = dict(time_myr=float(s.t), plate_relative_residual=residual,
        physical_torque_residual_n_m=(gradient.reshape(2, 3)*balance.RADIUS_M/balance.CM_YR_M_S).tolist(),
        omega_rad_myr=s.omega.tolist(), external_power_w=float(external@model.x), gpe_power_w=float(gpe@model.x),
        resisting_power_w=powers, minimum_local_resisting_power_w=passive['minimum_local_resisting_work_w'],
        overlap_km2=float(s.collision_diagnostics['pair_overlap_area_km2']),
        same_sheet_overlap_pairs=len(same['first']), maximum_displacement_km=float(angular.max()*6371.),
        maximum_thickness_change_km=float(np.max(np.abs(h-b['initial_thickness']))),
        maximum_local_volume_relative_error=float(np.max(np.abs(area*h/b['initial_volume']-1.))),
        minimum_area_km2=float(area.min()), minimum_signed_triangle=float(np.einsum('ij,ij->i', triangles[:, 0], np.cross(triangles[:, 1], triangles[:, 2])).min()),
        max_thickness_km=float(h.max()), sides=side, last_step=deepcopy(b['last_step']),
        front_normal_speed_km_myr=[float(f['normal_speed_km_myr']) for f in contacts.iter_motion_fronts(s.collision_contacts)],
        contact_order=[dict(id=int(r['id']), top_sheet=int(r['top_sheet']), under_sheet=int(r['under_sheet'])) for r in s.collision_contacts])
    row['gates'] = gates(s, row)
    return row


def advance(s, dt):
    started = time.perf_counter()
    evolution.advect(s, dt)
    changed, budget = columns.evolve_structure(s.structure, dt, geometric_log_area=s.geometric_log_area)
    s.structure = changed
    s.relief += budget['total_delta_m']
    s.benchmark['absolute_strain'] += np.abs(s.geometric_log_area)
    s.t += dt
    contacts.refresh(s, dt)
    diagnostics = s.deformation_diagnostics
    gravity = diagnostics.get('gravitational_relaxation', {})
    s.benchmark['last_step'] = dict(dt_myr=float(dt), material_stage_seconds=time.perf_counter()-started,
        solver=deepcopy(diagnostics.get('solver')), limited_faces=diagnostics.get('limited_faces', 0),
        same_sheet=deepcopy(diagnostics.get('same_sheet_nonpenetration')),
        finite_strain=deepcopy(diagnostics.get('finite_strain')),
        gravity={k: deepcopy(gravity[k]) for k in ('requested_dt_myr', 'completed_dt_myr', 'accepted_time_fraction',
            'energy_before_km4', 'energy_after_km4', 'gravitational_energy_change_j',
            'substeps', 'backtracks', 'energy_geometry_evaluations') if k in gravity},
        column_height_budget_m={k: [float(np.min(v)), float(np.max(v))] for k, v in budget.items() if k.endswith('_m')})


def atomic_json(path, data):
    temporary = path.with_suffix(path.suffix+'.tmp')
    temporary.write_text(json.dumps(data, indent=2, allow_nan=False)+'\n', encoding='utf-8')
    temporary.replace(path)


def run_case(s, *, dt, end, output):
    if not np.isfinite(dt) or dt <= 0 or not np.isfinite(end) or end < s.t:
        raise ValueError('Require finite positive dt and an end no earlier than the saved time.')
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    manifest = source_manifest()
    report = dict(schema=1, case=s.benchmark['case'], config=s.config, scope=SCOPE,
        policies={k: getattr(s, k, 0) for k in EXPECTED_POLICIES}, source=manifest, tolerances=TOL,
        dt_myr=dt, end_myr=end, initial_geometry_sha256=digest(s.benchmark['initial_vertices']),
        fixture=dict(vertices=len(s.material_surface['vertices']), faces=len(s.material_surface['faces']),
            faces_sha256=digest(s.material_surface['faces']),
            initial_columns_sha256=digest(s.benchmark['initial_thickness']),
            initial_area_sha256=digest(s.benchmark['initial_area']),
            free_omega_rad_myr=s.benchmark['free_omega'].tolist(),
            orientation=s.benchmark['orientation'].tolist()),
        initial_viscosity=s.benchmark['initial_viscosity'].tolist(),
        interface_parameters=s.collision_interface_parameters, rows=s.benchmark['rows'], complete=False, error=None)
    started = time.perf_counter()
    try:
        while True:
            current = observe(s)  # Rebuild transient solve after checkpoint load.
            if not report['rows'] or report['rows'][-1]['time_myr'] != s.t:
                report['rows'].append(current)
            else:
                report['rows'][-1] = current
            report['wall_seconds'] = time.perf_counter()-started
            atomic_json(output/'report.json', report)
            failed = [name for name, passed in current['gates'].items() if not passed]
            if failed:
                raise ValueError('Numerical benchmark gates failed: '+', '.join(failed))
            if s.t >= end-1e-12:
                break
            advance(s, min(dt, end-s.t))
        checkpoint.write_checkpoint(output/'state.npz', s, dict(config=s.config, benchmark=report['case']), compatibility(manifest))
        report['checkpoint_written'] = True
        report['complete'] = True
    except Exception as error:
        report['error'] = dict(type=type(error).__name__, message=str(error), traceback=traceback.format_exc())
    report['wall_seconds'] = time.perf_counter()-started
    report['numerical_gates_pass'] = report['complete'] and all(all(r['gates'].values()) for r in report['rows'])
    report['contact_coverage'] = any(r['overlap_km2'] > 0 for r in report['rows'])
    report['collision_validated'] = False
    atomic_json(output/'report.json', report)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--case', choices=CASES, default='symmetric')
    parser.add_argument('--dt', type=float, default=1.)
    parser.add_argument('--end', type=float, default=12.)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--resume', type=Path)
    args = parser.parse_args()
    s = checkpoint.read_checkpoint(args.resume, compatibility(source_manifest()), SimpleNamespace)[0] if args.resume else fixture(args.case)
    result = run_case(s, dt=args.dt, end=args.end, output=args.output)
    print(json.dumps({k: result[k] for k in ('case', 'complete', 'numerical_gates_pass', 'contact_coverage', 'wall_seconds', 'error')}))
    return 0 if result['numerical_gates_pass'] else 1


if __name__ == '__main__':
    raise SystemExit(main())
