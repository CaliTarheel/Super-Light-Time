"""Welded thrust stacks dock or stay as one body (welded-stack policy 1, N4).

Fixture, west to east along one latitude band 240 km wide: plate 0 carries a
sheet A ending at x = -100 km and a small terrane T (-150..-50 km) thrust over
A's east edge and B's west edge; plate 1 carries sheet B (-80..600 km). The
plates converge at about 38 km/Myr. Under the recorded policy 0, T alone docks
onto B, its weld with A becomes a new plate contact, and T returns about 20 Myr
later, then flickers every step on its stale welding memory.
"""
from copy import deepcopy
from pathlib import Path
import tempfile
from types import MethodType, SimpleNamespace
import unittest

import numpy as np

import checkpoint
import collision_contacts
import column_density
import local_accretion as local
import localized_accretion as welding
import material_surface
import mesh_geometry
from ridge_geometry import rotate
from tests.test_collision_contacts import rectangle
from tests.test_persistent_contact_accretion import world as free_world, mature

NEW_EVENT_FIELDS = {'accretion_welded_stack_version', 'welded_stack_sheets', 'released_welding_roots'}


def stack_world(a_left=-250., *, version=1, remote=False, width=120., abutting=None):
    """Sheets A and T on plate 0 (welded by overlap), B on plate 1.

    ``abutting`` adds a long plate-0 sheet D from x = -1100 km to that east
    edge, overlapping A's west edge (at ``a_left``) by the difference.
    """
    pieces = [rectangle(a_left, -100., 0, width=width, nx=6, ny=3),
              rectangle(-150., -50., 0, width=width, nx=3, ny=3),
              rectangle(-80., 600., 1, width=width, nx=8, ny=3)]
    if abutting is not None:
        pieces.append(rectangle(-1100., abutting, 0, width=width, nx=6, ny=3))
    if remote:
        far = list(rectangle(-200., 80., 0, width=width, nx=3, ny=3))
        far[0] = rotate(far[0], [0., 0., .7])
        pieces.append(tuple(far))
    surface = material_surface.initialize_surface(*pieces[0])
    for piece in pieces[1:]:
        material_surface.append_surface(surface, *piece)
    mesh = mesh_geometry.icosphere(2)
    count, n = len(surface['faces']), len(mesh['faces'])
    owner, kind = surface['face_owner'].copy(), surface['face_kind'].copy()
    pos, patches = material_surface.face_centres(surface), surface['face_id'].copy()
    locator = mesh_geometry.build_locator(mesh['vertices'], mesh['faces'])
    s = SimpleNamespace(native_mesh=mesh, material_surface=surface, n=n, xyz=mesh['xyz'].copy(),
        cell_area=mesh['area_km2'].copy(), pos=pos, parcel_patch=patches,
        parcel_plate=owner, mass=surface['reference_area_km2'].copy(), kind=kind,
        parcel_craton=np.full(count, -1, int), material_lineage={'root_id': patches.copy()},
        relief=np.full(count, 500.), suture=np.zeros(count),
        trace_patch=patches.copy(), trace_plate=owner.copy(), trace_xyz=pos.copy(),
        trace_kind=kind.copy(), trace_origin_kind=kind.copy(), trace_relief_m=np.full(count, 500.),
        trace_suture=np.zeros(count), trace_adjustment_m=np.zeros(count),
        structure={'thickness_km': np.full(count, 35.)}, trace_structure={'thickness_km': np.full(count, 35.)},
        plate=np.full(n, 2), crust=np.zeros(n, np.uint8),
        active=np.ones(3, bool), plate_uid=np.array([41, 52, 63]), born=np.zeros(3), t=100.,
        omega=np.array([[0., 0., .003], [0., 0., -.003], [0., 0., 0.]]),
        mantle=np.zeros((3, 3)), names=['West', 'East', 'Ocean'],
        events=[], ridge_episodes=[], backarc_basins=[], process_totals={'accreted_km2': 0.},
        ba=np.empty(0, int), bb=np.empty(0, int), bp=np.empty(0, int), bq=np.empty(0, int),
        bmid=np.empty((0, 3)), bn=np.empty((0, 3)), bcode=np.empty(0, np.uint8), bl=np.empty(0))
    attach_callables(s, locator)
    collision_contacts.refresh(s)
    hits = material_surface.sample_surface(surface, s.xyz)
    s.plate[hits['query_index']] = owner[hits['face_index']]
    s.crust[hits['query_index']] = kind[hits['face_index']]
    s.support = np.zeros((3, n))
    s.support[s.plate, np.arange(n)] = 1.
    welding.initialize(s)
    s.accretion_age_policy_version = local.AGE_POLICY_VERSION
    if version:
        s.accretion_welded_stack_version = version
    a_faces = np.arange(2*5*2)
    s.fixture = dict(A=a_faces, T=np.arange(len(a_faces), len(a_faces)+8),
                     B=np.arange(len(a_faces)+8, len(a_faces)+8+2*7*2))
    return s


def attach_callables(s, locator):
    """The engine methods the planner calls; a checkpoint never stores them."""
    def record(self, kind, description, key=None, *, plates=(), xyz=None, details=None):
        self.events.append(dict(type=kind, time_myr=float(self.t), details=deepcopy(details or {})))
    s._record = MethodType(record, s)
    s._indices = lambda points: mesh_geometry.locate_points(points, locator)[0]


def run(s, steps=70):
    """The engine's order: measure contacts, plan, apply at most one plan."""
    history = []
    for _ in range(steps):
        s.t += 2.
        collision_contacts.refresh(s, 2.)
        for plan in local.plan_accretions(s, 2.):
            local.apply_accretion(s, plan)
        history.append((float(s.t), s.parcel_plate.copy()))
    return history


def owner_changes(history, faces):
    owners = [int(plates[faces[0]]) for _, plates in history]
    return [(history[i][0], owners[i-1], owners[i]) for i in range(1, len(owners)) if owners[i] != owners[i-1]]


def observations(s):
    exact = local._persistent_contact_geometry(s)
    occupancy = local._native_occupancy(s)
    owners = {p for row in exact for p in (row['p'], row['q'])}
    geometry = {p: local._material_components(s, p, occupancy) for p in owners}
    return local._persistent_observations(s, exact, geometry), geometry


class WeldedStackAccretionTests(unittest.TestCase):
    def test_legacy_policy_reproduces_terrane_ping_pong(self):
        s = stack_world(-600., version=0)
        self.assertFalse(hasattr(s, 'accretion_welded_stack_version'))
        states = sorted(row['state'] for row in s.collision_contacts)
        self.assertEqual(states, ['accreted', 'active'])
        history = run(s)
        changes = owner_changes(history, s.fixture['T'])
        self.assertGreaterEqual(len(changes), 2)
        self.assertEqual(changes[0][1:], (0, 1))
        self.assertEqual(changes[1][1:], (1, 0))
        # 18 Myr to re-mature the new direction from zero, plus the 2 Myr step.
        self.assertTrue(18. <= changes[1][0]-changes[0][0] <= 22.)
        # A itself never moved: T was torn out of its stack every time.
        self.assertTrue(all(np.all(plates[s.fixture['A']] == 0) for _, plates in history))

    def test_small_welded_stack_docks_once_as_one_body(self):
        s = stack_world(-250.)
        history = run(s)
        A, T = s.fixture['A'], s.fixture['T']
        for _, plates in history:
            self.assertEqual(len(np.unique(plates[np.r_[A, T]])), 1)
        changes = owner_changes(history, T)
        self.assertEqual([change[1:] for change in changes], [(0, 1)])
        events = [event for event in s.events if event['type'] == 'accretion']
        self.assertEqual(len(events), 1)
        details = events[0]['details']
        self.assertEqual(details['welded_stack_sheets'], 2)
        self.assertEqual(details['accretion_welded_stack_version'], local.WELDED_STACK_VERSION)
        self.assertEqual(details['released_welding_roots'], len(np.r_[A, T]))
        # The one weld is T thrust over A: 50 km x 240 km of overlap.
        self.assertAlmostEqual(details['welded_stack_min_contact_km2'], 12000., delta=60.)
        self.assertTrue(details['local_welding_qualification']['eligible'])
        self.assertEqual(details['local_welding_qualification']['component_roots'], len(np.r_[A, T]))
        np.testing.assert_allclose(details['accreted_area_km2'], s.mass[np.r_[A, T]].sum())
        # Every contact left is internal to one plate.
        self.assertTrue(all(row['state'] != 'active' for row in s.collision_contacts))

    def test_stack_reaching_beyond_the_process_zone_keeps_its_terrane(self):
        s = stack_world(-900.)
        history = run(s)
        self.assertEqual(owner_changes(history, s.fixture['T']), [])
        self.assertEqual(owner_changes(history, s.fixture['B']), [])
        self.assertFalse(any(event['type'] == 'accretion' for event in s.events))
        # The same geometry under policy 0 tears T out of A.
        legacy = stack_world(-900., version=0)
        self.assertGreaterEqual(len(owner_changes(run(legacy, steps=15), legacy.fixture['T'])), 1)

    def test_free_terrane_docks_identically_under_both_policies(self):
        legacy = free_world(width=180.)
        welding.initialize(legacy)
        legacy.accretion_age_policy_version = local.AGE_POLICY_VERSION
        stacked = deepcopy(legacy)
        stacked.accretion_welded_stack_version = local.WELDED_STACK_VERSION
        self.assertEqual(len(collision_contacts.same_owner_overlap_pairs(stacked)), 0)
        plans = [mature(s, steps=40) for s in (legacy, stacked)]
        self.assertEqual(legacy.t, stacked.t)
        self.assertEqual(set(plans[0]), set(plans[1]))
        for name, value in plans[0].items():
            if isinstance(value, np.ndarray):
                np.testing.assert_array_equal(value, plans[1][name], err_msg=name)
            else:
                self.assertEqual(value, plans[1][name], name)
        for s, plan in zip((legacy, stacked), plans):
            self.assertTrue(local.apply_accretion(s, plan))
        np.testing.assert_array_equal(legacy.parcel_plate, stacked.parcel_plate)
        old, new = legacy.events[-1]['details'], stacked.events[-1]['details']
        self.assertEqual(set(new)-set(old), NEW_EVENT_FIELDS)
        self.assertEqual(old, {key: value for key, value in new.items() if key not in NEW_EVENT_FIELDS})
        self.assertEqual(new['welded_stack_sheets'], 1)

    def test_a_later_owner_split_releases_the_weld(self):
        s = stack_world(-250.)
        run(s, steps=25)
        A, T = s.fixture['A'], s.fixture['T']
        self.assertTrue(np.all(s.parcel_plate[np.r_[A, T]] == 1))
        docked = local._material_components(s, 1)
        self.assertEqual(len(np.unique(docked['face_component'][np.r_[A, T, s.fixture['B']]])), 1)
        # A rift gives T alone a different motion owner.
        s.parcel_plate[T] = 2
        collision_contacts.refresh(s)
        pairs = collision_contacts.same_owner_overlap_pairs(s)
        self.assertFalse(np.any(np.isin(pairs, T)))
        rest = local._material_components(s, 1)
        self.assertTrue(np.all(rest['face_component'][T] == -1))
        alone = local._material_components(s, 2)
        self.assertEqual(len(alone['components']), 1)
        np.testing.assert_array_equal(next(iter(alone['components'].values()))['parcel_indices'], T)
        self.assertTrue(any(row['state'] == 'active' for row in s.collision_contacts))

    def test_departed_rock_leaves_its_welding_memory_behind(self):
        s = stack_world(-250., remote=True)
        A, T = s.fixture['A'], s.fixture['T']
        remote = np.arange(len(s.mass)-8, len(s.mass))
        plan = None
        for _ in range(40):
            s.t += 2.
            collision_contacts.refresh(s, 2.)
            plans = local.plan_accretions(s, 2.)
            if plans:
                plan = plans[0]
                break
        self.assertIsNotNone(plan)
        np.testing.assert_array_equal(np.sort(plan['parcel_indices']), np.r_[A, T])
        # One transferred root also labels a remote face that stays behind.
        roots = s.material_lineage['root_id']
        shared = int(roots[T[0]])
        roots[remote[0]] = shared
        # Another labels a remote face that has lost all its mass (eroded or
        # foundered): no rock of that root is left on the source.
        eroded = int(roots[A[0]])
        roots[remote[1]] = eroded
        s.mass[remote[1]] = 0.
        state = s.accretion_welding_state
        source = int(plan['source_uid'])
        departed = set(map(int, roots[np.r_[A, T]])) - {shared}
        self.assertIn(eroded, departed)
        before = {int(r) for u, r in zip(state['source_uid'], state['root_id']) if u == source}
        self.assertTrue(departed <= before)
        self.assertIn(shared, before)
        self.assertTrue(local.apply_accretion(s, plan))
        state = s.accretion_welding_state
        after = {int(r) for u, r in zip(state['source_uid'], state['root_id']) if u == source}
        self.assertFalse(departed & after)
        self.assertIn(shared, after)
        self.assertTrue(set(map(int, roots[remote[2:]])) <= after)
        self.assertEqual(s.events[-1]['details']['released_welding_roots'], len(departed))
        frame = dict(welding.snapshot_fields(s), time_myr=float(s.t))
        self.assertEqual(frame['accretion_welded_stack_version'], 1)
        welding.validate_frame(frame)
        self.assertEqual(frame['native_accretion_diagnostics']['directed_root_records'], len(state['root_id']))

    def test_process_zone_crosses_the_weld_only_under_the_policy(self):
        s = stack_world(-250.)
        rows, geometry = observations(s)
        self.assertEqual(len(rows), 1)
        component = geometry[0]['components'][rows[0]['cp']]
        A, T = s.fixture['A'], s.fixture['T']
        np.testing.assert_array_equal(np.sort(component['parcel_indices']), np.r_[A, T])
        self.assertGreater(len(component['welded_pairs']), 0)
        weight, data = welding.process_weights(s, rows, component, 0)
        stacked = np.isin(data['material'], A)
        self.assertTrue(np.all(weight[stacked] > 0.))
        self.assertTrue(np.all(weight[np.isin(data['material'], T)] > 0.))
        s.accretion_welded_stack_version = 0
        weight, data = welding.process_weights(s, rows, component, 0)
        np.testing.assert_array_equal(weight[np.isin(data['material'], A)], 0.)
        legacy = local._material_components(s, 0)
        self.assertNotEqual(legacy['face_component'][A[0]], legacy['face_component'][T[0]])
        self.assertNotIn('welded_pairs', next(iter(legacy['components'].values())))

    def test_only_current_same_owner_overlap_welds(self):
        s = stack_world(-250.)
        overlap = s._collision_overlap
        pairs = collision_contacts.same_owner_overlap_pairs(s)
        self.assertGreater(len(pairs), 0)
        self.assertTrue(np.all(s.parcel_plate[pairs[:, 0]] == s.parcel_plate[pairs[:, 1]]))
        cross = s.parcel_plate[overlap['first']] != s.parcel_plate[overlap['second']]
        self.assertTrue(np.any(cross))
        self.assertEqual(len(pairs), np.count_nonzero(~cross))
        receiver = local._material_components(s, 1)
        self.assertTrue(np.all(receiver['face_component'][np.r_[s.fixture['A'], s.fixture['T']]] == -1))
        # Material with no mass cannot weld.
        empty = deepcopy(s)
        empty.mass[s.fixture['T']] = 0.
        self.assertEqual(len(collision_contacts.same_owner_overlap_pairs(empty)), 0)
        # A ledger measured on other owners is refused, never reused.
        stale = deepcopy(s)
        stale.parcel_plate[s.fixture['T'][0]] = 2
        with self.assertRaises(ValueError):
            collision_contacts.same_owner_overlap_pairs(stale)
        with self.assertRaises(ValueError):
            local._material_components(stale, 0)
        # No ledger, no welds.
        bare = deepcopy(s)
        del bare._collision_overlap
        self.assertEqual(collision_contacts.same_owner_overlap_pairs(bare).shape, (0, 2))

    def test_policy_selection_is_validated(self):
        s = stack_world(-250.)
        self.assertTrue(local.welded_stack_enabled(s))
        for invalid in (True, np.bool_(True), 2, '1', 1.0):
            s.accretion_welded_stack_version = invalid
            with self.subTest(value=invalid), self.assertRaises(ValueError):
                local.welded_stack_enabled(s)
        s.accretion_welded_stack_version = np.int64(1)
        self.assertTrue(local.welded_stack_enabled(s))
        without_welding = stack_world(-250.)
        without_welding.native_accretion_version = 0
        with self.assertRaises(ValueError):
            local.welded_stack_enabled(without_welding)
        raster = stack_world(-250.)
        raster.native_mesh = None
        with self.assertRaises(ValueError):
            local.welded_stack_enabled(raster)

    def test_roundoff_contact_does_not_weld_a_remote_sheet(self):
        # D reaches 1,100 km west and touches A's west edge by a 0.1 m strip,
        # 0.024 km2 in all: the roundoff arc emplacement leaves (section 17),
        # not a thrust. It must neither weld D into the stack nor stop {A, T}
        # docking.
        s = stack_world(-250., abutting=-250.+1e-4)
        D = np.arange(len(s.mass)-20, len(s.mass))
        overlap = s._collision_overlap
        touching = np.isin(overlap['first'], D) | np.isin(overlap['second'], D)
        self.assertTrue(0. < overlap['area_km2'][touching].sum() <= column_density.UNORDERED_OVERLAP_FLOOR_KM2)
        pairs = collision_contacts.same_owner_overlap_pairs(s)
        self.assertFalse(np.any(np.isin(pairs, D)))
        self.assertTrue(np.any(np.isin(pairs, s.fixture['A'])))
        history = run(s, steps=40)
        A, T = s.fixture['A'], s.fixture['T']
        self.assertEqual([change[1:] for change in owner_changes(history, T)], [(0, 1)])
        self.assertEqual(owner_changes(history, A), owner_changes(history, T))
        self.assertEqual(owner_changes(history, D), [])
        # Exactly as without D.
        alone = stack_world(-250.)
        self.assertEqual(owner_changes(run(alone, steps=40), alone.fixture['T']), owner_changes(history, T))

    def test_a_genuine_thrust_overlap_welds_however_narrow(self):
        # A 1 km overlap along the whole 240 km edge (240 km2) is a real thrust
        # contact: D joins the stack, which then reaches 1,100 km and stays.
        s = stack_world(-250., abutting=-249.)
        D = np.arange(len(s.mass)-20, len(s.mass))
        pairs, contact = collision_contacts.same_owner_overlap_pairs(s, with_contact_area=True)
        joined = np.any(np.isin(pairs, D), axis=1)
        self.assertTrue(np.any(joined))
        self.assertAlmostEqual(float(contact[joined].min()), 240., delta=2.)
        history = run(s, steps=40)
        self.assertEqual(owner_changes(history, s.fixture['T']), [])
        self.assertFalse(any(event['type'] == 'accretion' for event in s.events))

    def test_owner_split_through_a_sheet_releases_the_weld(self):
        s = stack_world(-250.)
        run(s, steps=25)
        A, T, B = s.fixture['A'], s.fixture['T'], s.fixture['B']
        self.assertTrue(np.all(s.parcel_plate[np.r_[A, T]] == 1))
        overlap = s._collision_overlap
        under_t = np.unique(np.r_[overlap['first'][np.isin(overlap['second'], T)],
                                  overlap['second'][np.isin(overlap['first'], T)]])
        east = np.intersect1d(A, under_t)
        west = np.setdiff1d(A, east)
        self.assertTrue(len(east) and len(west))
        sheet = int(s.parcel_collision_sheet[A[0]])
        self.assertTrue(np.all(s.parcel_collision_sheet[A] == sheet))
        # A fault through sheet A gives its eastern part, the part under T,
        # to another plate.
        s.parcel_plate[east] = 2
        collision_contacts.refresh(s)
        self.assertTrue(np.all(s.parcel_collision_sheet[west] == sheet))
        daughter = np.unique(s.parcel_collision_sheet[east])
        self.assertEqual(len(daughter), 1)
        self.assertNotEqual(int(daughter[0]), sheet)
        pairs = collision_contacts.same_owner_overlap_pairs(s)
        self.assertFalse(np.any(np.isin(pairs, east)))
        self.assertTrue(np.any(np.isin(pairs, T)))
        # T still welds to B on plate 1; A's western part is now a free body.
        receiver = local._material_components(s, 1)
        component = receiver['face_component']
        self.assertEqual(len(np.unique(component[np.r_[T, B]])), 1)
        self.assertTrue(np.all(component[west] == component[west[0]]))
        self.assertNotEqual(component[west[0]], component[T[0]])
        self.assertTrue(np.all(component[east] == -1))
        # The daughter now meets T across a plate contact.
        self.assertTrue(any(row['state'] == 'active'
                            and int(daughter[0]) in (row['top_sheet'], row['under_sheet'])
                            for row in s.collision_contacts))

    def test_checkpoint_round_trip_keeps_the_policy_and_its_course(self):
        s = stack_world(-250.)
        run(s, steps=8)
        self.assertTrue(s.accretion_welding_state['root_id'].size)
        saved = SimpleNamespace(**{k: v for k, v in vars(s).items() if k not in ('_record', '_indices')},
                                rng=np.random.default_rng(3), config={})
        tmp = Path(__file__).resolve().parents[1]/'tmp'
        tmp.mkdir(exist_ok=True)
        with tempfile.TemporaryDirectory(dir=tmp) as folder:
            path = Path(folder)/'checkpoint.npz'
            checkpoint.write_checkpoint(path, saved, dict(config={}), {})
            restored, _ = checkpoint.read_checkpoint(path, {}, SimpleNamespace)
        self.assertIs(type(restored.accretion_welded_stack_version), int)
        attach_callables(restored, mesh_geometry.build_locator(restored.native_mesh['vertices'],
                                                               restored.native_mesh['faces']))
        self.assertTrue(local.welded_stack_enabled(restored))
        original, resumed = run(s, steps=20), run(restored, steps=20)
        T = s.fixture['T']
        self.assertEqual(owner_changes(original, T), owner_changes(resumed, T))
        self.assertEqual(len(owner_changes(original, T)), 1)
        for (_, one), (_, two) in zip(original, resumed):
            np.testing.assert_array_equal(one, two)
        welding.validate_frame(dict(welding.snapshot_fields(restored), time_myr=float(restored.t)))

    def test_frame_schema_accepts_only_a_welded_policy_on_welding_frames(self):
        s = stack_world(-250.)
        frame = dict(welding.snapshot_fields(s), time_myr=float(s.t))
        self.assertEqual(frame['accretion_welded_stack_version'], 1)
        welding.validate_frame(frame)
        legacy = stack_world(-250., version=0)
        self.assertNotIn('accretion_welded_stack_version', welding.snapshot_fields(legacy))
        with self.assertRaises(ValueError):
            welding.validate_frame({'accretion_welded_stack_version': 1})
        welding.validate_frame({'accretion_welded_stack_version': 0})
        for invalid in (True, 2, 1.0, '1'):
            with self.subTest(value=invalid), self.assertRaises(ValueError):
                welding.validate_frame(dict(frame, accretion_welded_stack_version=invalid))


if __name__ == '__main__':
    unittest.main()
