"""Read recorded geological histories without reconstructing unrecorded motion."""
from __future__ import annotations

from bisect import bisect_right
import json
from pathlib import Path

import numpy as np


TRACE_DTYPES = {
    "trace_id": np.int64, "trace_xyz": np.float64,
    "trace_plate": np.int32, "trace_plate_uid": np.int64,
    "trace_kind": np.uint8, "trace_origin_kind": np.uint8,
    "trace_birth_myr": np.float64, "trace_relief_m": np.float64,
    "trace_uplift_m": np.float64, "trace_extension_m": np.float64,
    "trace_erosion_m": np.float64, "trace_adjustment_m": np.float64,
    "trace_suture": np.float64,
    "trace_patch": np.int64, "trace_craton": np.int32,
    "trace_ridge_uplift_m": np.float64, "trace_ridge_thermal_m": np.float64,
    "trace_rift_id": np.int64, "trace_rift_birth_myr": np.float64,
    "trace_rift_extension_m": np.float64, "trace_inversion_uplift_m": np.float64,
}
# New fields are appended LAST: the first five are the grid-sampled subset read
# through the [:5] slice below, and saved histories keep their field order.
STRUCTURE_TRACE_FIELDS = ('crustal_thickness_km', 'crustal_root_km',
    'rift_thermal_support_m', 'rift_cooling_age_myr', 'foreland_deflection_m',
    'denudation_m', 'rebound_m', 'thermal_subsidence_m', 'foreland_subsidence_m',
    'foundered_m')
TRACE_DTYPES.update({'trace_'+name: np.float64 for name in STRUCTURE_TRACE_FIELDS})


def history_schema():
    return {
        "version": 1,
        "time": "Elapsed million years since initial condition; no invented frames between saved times.",
        "grid_fields": ["elevation", "plate", "crust", "age", "boundary"],
        "optional_grid_fields": ["domain", "trench", "crustal_thickness_km", "crustal_root_km",
            "rift_thermal_support_m", "rift_cooling_age_myr", "foreland_deflection_m", "erosion_rate_m_myr",
            "rift_damage", "rift_strength_relative"],
        "domain": "Stable int32 connected-domain UID; optional and absent from older histories. Stored in frame NPZ, with domains metadata in frame JSON.",
        "domains": "Named connected-domain records: uid, name, plate_id, parent_plate_uid, area_km2, created_myr and source_domain_uid. Plate IDs still describe underlying kinematic motion; a distinct domain does not imply an independent rotation.",
        "trace_fields": list(TRACE_DTYPES),
        "optional_trace_fields": ["trace_ridge_uplift_m", "trace_ridge_thermal_m",
                                  "trace_rift_id", "trace_rift_birth_myr",
                                  "trace_rift_extension_m", "trace_inversion_uplift_m",
                                  *['trace_'+name for name in STRUCTURE_TRACE_FIELDS]],
        "trace_xyz": "N by 3 unit-sphere Cartesian positions; other trace arrays have length N.",
        "trace_id": "Stable identity of a representative buoyant-material marker within this run.",
        "trace_plate_uid": "Plate generation identity; numeric raster plate slots can be reused.",
        "trace_patch": "Enduring source material patch; original-cell samples remain coherent. Arc patches may coalesce.",
        "trace_craton": "Original connected craton material identity; -1 for other material.",
        "trace_relief_m": "Representative material relief; not the total mapped surface elevation.",
        "trace_uplift_m": "Cumulative positive model relief gain since marker birth, metres.",
        "trace_ridge_uplift_m": "Optional cumulative ridge-subduction volcanic relief gain, metres; a subset already included in trace_uplift_m, not an additional term.",
        "trace_ridge_thermal_m": "Optional current ridge-subduction thermal elevation offset, metres; temporary and additional to trace_relief_m, not a cumulative relief counter.",
        "trace_rift_id": "Optional int64 identity of the earliest recorded local continental rift scar; IDs start at 1 and -1 means no recorded scar.",
        "trace_rift_birth_myr": "Optional creation time of that structural scar, Myr elapsed; -1 means none. A scar seeded at time zero does not imply observed stretching.",
        "trace_rift_extension_m": "Optional cumulative observed rift depression, metres; a subset already included in trace_extension_m, not an additional loss.",
        "trace_inversion_uplift_m": "Optional cumulative realized rift-inversion contribution, metres; a subset already included in trace_uplift_m, not an additional gain.",
        "trace_extension_m": "Cumulative tectonic lowering since marker birth, metres: structure_version 1 includes thinning, cooling and foreland loading; legacy histories record extensional relief loss.",
        "trace_erosion_m": "Cumulative net denudation loss after rebound for structure_version 1; legacy histories record signed relief relaxation, which can be negative below zero relief.",
        "structure_version": "Version1 uses material crustal columns and net denudation loss after Airy rebound. New denudation/rebound counters are explanatory subsets, not extra height-budget terms. Legacy erosion retains its original signed relaxation meaning.",
        "crustal_thickness_km": "Compensated buoyant-column thickness; ocean cells use a constant7km reference. Scalar columns do not resize the transport footprint.",
        "crustal_root_km": "Positive thickness excess over each material column's neutral reference; not absolute Moho depth. It does NOT record eclogitic foundering: the neutral reference thins with the column, so a foundered root reads as neutral and only trace_foundered_m carries the loss.",
        "foundered_m": "Optional cumulative column thickness returned to the mantle by eclogitization, in metres of crust; foundering_version1 only. A mass outflow, not denudation: it produces no sediment and is excluded from trace_erosion_m. Its surface effect is Airy subsidence already inside trace_extension_m, so this is an explanatory subset, not an extra height-budget term.",
        "rift_thermal_support_m": "Current temporary uplift from rift heating; its later decline supplies post-rift subsidence.",
        "rift_cooling_age_myr": "Elapsed time since most recent modeled thinning; -1 means no recorded thinning.",
        "foreland_deflection_m": "Current positive downward displacement beside mountain loads, using a bounded flexure-inspired proxy.",
        "erosion_rate_m_myr": "Last step's net denudation minus associated rebound per Myr, material-weighted onto the saved grid; used only for approximate unsupported-marker goSPL correction.",
        "trench_systems": "Persistent local subduction identities, polarity, maturity, observed shortening and recorded lifecycle milestones. Geometry follows the overriding plate; shutdown does not imply the history record is deleted.",
        "effective_subduction_version": "Optional frame integer: 1 selects declared persistent incoming-plate traction without slab inventory; missing or 0 keeps the saved detailed law. It does not prescribe plate speed.",
        "effective_subduction_diagnostics": "Optional frame JSON: prescribed force_n_per_m in N/m, initial and current matched trench lengths in km, integrated incoming force in N, selection and omitted-reservoir limitations. The sum of force magnitudes is not net spherical torque or a closed mantle energy budget.",
        "rift_damage": "Optional float32 grid of local accumulated mechanical damage, bounded from zero to one. An approximate material weakening indicator, not physical stress or an observed fracture fraction.",
        "rift_strength_relative": "Optional positive float32 grid of relative lithospheric resistance; lower values mean weaker material. This is not surface rock hardness or a strength in pascals.",
        "rift_systems": "Optional frame JSON records of progressive continental rifting: id, plate_uid, phase (incipient, active, failed or broken_through), started_myr, last_active_myr, extension_km (cumulative modeled stretching, not present basin width), peak_damage, mean_strength, geometry_xyz (unordered current unit-sphere anchors, not a polyline), history entries with time_myr, phase and reason, and optional new_plate_uid. Missing means not recorded; an empty list means no systems at this time.",
        "rift_mechanics": "Optional frame JSON solver diagnostics: mesh_nodes, mesh_edges, target_nodes, solver_iterations, residual, converged and model. The approximate mechanical mesh budget is independent of the tectonic raster and final heightmap resolution. Rupture labels: damage_strain_measure always; rupture_strain_threshold (dimensionless, link-length dependent) under rupture criterion 0; under criterion 1 (frame key rupture_criterion_version = 1) instead rupture_criterion, breakup_extension_km (realized extension a link needs before breakup, km), realized_extension_fraction (1 for realized links, 0.25 for the loading proxy), rupture_extension_measure and max_bond_realized_extension_km. Commit labels only under rift commit policy 1 (frame key rift_commit_version = 1): commit_policy_version, commit_policy and commit_loading_unit ('connected material component' for realized links, 'plate' for the legacy membrane); on a step whose pending links met the breakup gate also commit_mesh_changed (whether any rift-mesh node key, link key or node's material changed between the loading update and the commit, normally by local accretion; a pure reordering of the arrays does not count), commit_changed_plate_uids (plates with such a change, including the donor of a partial node transfer), commit_carried_links (current links matched by key to the pending solve), commit_new_links (links without pending work, which cannot fail that step), commit_reassigned_patches (continental patches whose node key changed, from the material image the loading update records), commit_deferred_cuts and commit_deferred_plate_uids (cuts skipped because their own loading unit changed; they are re-solved next step).",
        "rift_commit_version": "Optional frame integer selecting how a continental breakthrough treats rift-mesh changes made after that step's loading update. Missing or 0: any node or link change anywhere refuses every cut of the step. 1 (fresh reviewed_v1 worlds): the pending solve is carried to the current mesh by persistent link keys and only cuts whose own loading unit changed are deferred.",
        "rupture_criterion_version": "Optional frame integer selecting the continental rupture gate. Missing or 0: the mode-specific accumulated-strain threshold (rift_mechanics.rupture_strain_threshold). 1 (fresh reviewed_v1 worlds): each removed link needs rift_mechanics.breakup_extension_km (100 km) of accumulated realized extension, independent of mechanics link length.",
        "rift_system_events": "Explicit rift_initiating, rift_active, rift_failed, rift_reactivated and rift_breakthrough events use details.rift_system_id to identify the local system and time_myr to locate its milestone. Damage or an incipient system alone does not establish continental breakup.",
        "trace_adjustment_m": "Cumulative signed relief gain from clipping or arc-floor resets, metres.",
        "trace_suture": "Dimensionless model deformation memory, zero to one.",
        "trace_birth_myr": "Marker birth in this simulation; zero does not date the original rocks.",
        "events": "Full catalogue in events.json; plate-associated events need not be local to a selected marker.",
        "ridge_episodes": "Optional frame JSON records of ridge-subduction episodes, with overriding/incoming plate UIDs, unit-XYZ center, radius_km, started_myr, peak_myr, end_myr and optional heat. The footprint is an approximate transient thermal region, not a plate boundary.",
        "rift_records": "Optional frame JSON continental-rift provenance records: id, started_myr, origin, center (unit XYZ at scar birth), parent_plate_uids and inversion status. Birth positions are not current material positions.",
        "backarc_basins": "Optional frame JSON records of back-arc basin evolution. Missing metadata means not recorded; an empty list means no basins recorded at that saved time. Records contain id, started_myr, phase (loading, rifting, spreading, quiet or closed), parent_plate_uid, arc_plate_uid (null before breakup), downgoing_plate_uid, center (current unit XYZ), opening_km (cumulative measured extension, not current basin width), extension_rate_km_myr (real opening rate, zero before rupture), loading_km, optional loading_rate_km_myr (measured loading rate before rupture, km/Myr: the swept trench retreat relative to the overriding plate under back-arc driver 0, or under driver 1 the overriding plate's stationary-mantle-frame trench-normal speed away from a slab anchored below its lithosphere), last_active_myr and optional rupture_myr. UIDs retain plate identity; centers rotate with an oriented export while scalar histories remain unchanged.",
        "backarc_driver_version": "Optional frame JSON integer naming the back-arc loading driver. Missing or 0 means swept trench retreat (the forearc sweep). 1, used by fresh moving-hinge (subduction_response_version 1) reviewed worlds, means the sea anchor: the overriding plate's mantle-frame motion away from its anchored slab. It never changes trench_retreat_speed or arc supply.",
        "backarc_driver_diagnostics": "Present only with backarc_driver_version 1: version, anchored_edges, anchored_length_km (attached subduction edges whose slab length x sin(50 deg) exceeds the overriding lithosphere thickness), loading_length_km (those moving away faster than 0.4 km/Myr), max_speed_km_myr, length-weighted mean_speed_km_myr over anchored edges, and the declared frame and law strings. Measured at the step's back-arc update, before that step's topology commit and post-split force re-solve, so in a rupture step they describe the pre-split plates and motion.",
        "backarc_events": "Explicit backarc_loading, backarc_rifting, backarc_spreading, backarc_quiet and backarc_closed events use details.backarc_id to link the basin record and time_myr to locate their recorded milestones. A loading record alone does not establish that a basin opened.",
        "limitations": [
            "Markers sample buoyant crust and do not represent every terrane or oceanic rock.",
            "A fixed-location inspection samples changing material at one geographic coordinate.",
            "Mapped elevation includes advection, relief mixing, overlap, bathymetry and roughness.",
            "Marker relief counters are not a complete uplift field or a GoSPL-ready forcing product.",
            "Legacy runs contain only the fields actually saved by their original engine.",
        ],
    }


def read_metadata(path: Path, index: int):
    return json.loads((path / f"frame_{index:04d}.json").read_text(encoding="utf-8"))


def read_record(path: Path, manifest: dict):
    count = int(manifest.get("frame_count", 0))
    latest = read_metadata(path, count - 1) if count else {}
    events = []
    for i, original in enumerate(latest.get("events", [])):
        event = dict(original)
        event.setdefault("id", i)
        event.setdefault("plate_ids", [])
        event.setdefault("plate_uids", [])
        events.append(event)
    return {
        "run_id": manifest.get("run_id"), "frame_count": count,
        "history_version": latest.get("history_version", 0),
        "frames": [{"index": int(f.get("index", i)), "time_myr": f["time_myr"]}
                   for i, f in enumerate(manifest.get("frames", [])[:count])],
        "events": events,
    }


def _coordinate(cell, width, height):
    return (-180 + (cell % width + .5) * 360 / width,
            90 - (cell // width + .5) * 180 / height)


def _xyz_coordinate(xyz):
    return (float(np.degrees(np.arctan2(xyz[1], xyz[0]))),
            float(np.degrees(np.arcsin(np.clip(xyz[2], -1, 1)))))


def _cell_at(lon, lat, width, height):
    x = int(np.floor((lon + 180) * width / 360)) % width
    y = int(np.clip(np.floor((90 - lat) * height / 180), 0, height - 1))
    return y * width + x


def read_history(path: Path, manifest: dict, index: int, cell: int):
    """Follow a saved marker where available; otherwise explicitly sample a site."""
    count = int(manifest.get("frame_count", 0))
    if index < 0:
        index += count
    if index < 0 or index >= count:
        raise ValueError("That history frame is not available yet.")
    selected = read_metadata(path, index)
    w, h = int(selected["width"]), int(selected["height"])
    if cell < 0 or cell >= w * h:
        raise ValueError("Choose a cell within the world map.")
    lon, lat = _coordinate(cell, w, h)
    selection = {"frame": index, "cell": cell, "lon": lon, "lat": lat}
    trace_id = None
    description = "Fixed geographic location: each point may contain different rock as plates pass through."
    with np.load(path / f"frame_{index:04d}.npz", allow_pickle=False) as data:
        if int(data["crust"].reshape(-1)[cell]) > 0 and "trace_id" in data.files:
            ids = data["trace_id"]
            xyz = data["trace_xyz"]
            owner = int(data["plate"].reshape(-1)[cell])
            candidates = np.flatnonzero(data["trace_plate"] == owner)
            if len(candidates):
                rlon, rlat = np.radians([lon, lat])
                query = np.array([np.cos(rlat)*np.cos(rlon), np.cos(rlat)*np.sin(rlon), np.sin(rlat)])
                dots = xyz[candidates] @ query
                nearest = int(candidates[int(np.argmax(dots))])
                angle = float(np.arccos(np.clip(np.max(dots), -1, 1)))
                # Limit association to two local cell diagonals. This does not
                # turn a distant surviving marker into a history for this place.
                reach = 2 * np.hypot(2*np.pi/w*np.cos(rlat), np.pi/h)
                if angle <= reach:
                    trace_id = int(ids[nearest])
                    selection["marker_distance_km"] = round(angle * 6371, 2)
                    description = ("Representative buoyant material followed through plate motion and ownership changes. "
                                   "Mapped elevation samples the surface at the marker; relief counters describe its model history.")
            if trace_id is None:
                description += " No recorded material marker is close enough to this selection."

    points = []
    for i in range(count):
        metadata = read_metadata(path, i)
        width, height = int(metadata["width"]), int(metadata["height"])
        sample_lon, sample_lat = lon, lat
        with np.load(path / f"frame_{i:04d}.npz", allow_pickle=False) as data:
            marker = None
            if trace_id is not None:
                if "trace_id" not in data.files:
                    continue
                matches = np.flatnonzero(data["trace_id"] == trace_id)
                if not len(matches):
                    continue  # A later-born arc has no invented earlier life.
                marker = int(matches[0])
                sample_lon, sample_lat = _xyz_coordinate(data["trace_xyz"][marker])
            sample_cell = _cell_at(sample_lon, sample_lat, width, height)
            plate_id = int(data["trace_plate"][marker] if marker is not None else data["plate"].reshape(-1)[sample_cell])
            plates = {int(p["id"]): p for p in metadata.get("plates", [])}
            plate = plates.get(plate_id, {})
            plate_uid = int(data["trace_plate_uid"][marker]) if marker is not None and "trace_plate_uid" in data.files else plate.get("uid")
            domain_uid, domain_name = None, None
            surface_plate = int(data["plate"].reshape(-1)[sample_cell])
            if "domain" in data.files and (marker is None or surface_plate == plate_id):
                candidate = int(data["domain"].reshape(-1)[sample_cell])
                domains = {int(row["uid"]): row for row in metadata.get("domains", [])}
                domain = domains.get(candidate)
                if domain is not None and int(domain["plate_id"]) == plate_id:
                    domain_uid, domain_name = candidate, domain.get("name")
            point = {
                "index": i, "time_myr": metadata["time_myr"],
                "lon": sample_lon, "lat": sample_lat,
                "elevation_m": float(data["elevation"].reshape(-1)[sample_cell]),
                "plate_id": plate_id, "plate_uid": plate_uid,
                "plate_name": plate.get("name", f"Plate {plate_id + 1:02d}"),
                "domain_uid": domain_uid, "domain_name": domain_name,
                "crust": int(data["crust"].reshape(-1)[sample_cell]),
                "boundary": int(data["boundary"].reshape(-1)[sample_cell]),
            }
            if marker is not None:
                for source, key in (("trace_kind", "material_kind"), ("trace_origin_kind", "origin_kind"),
                                    ("trace_birth_myr", "birth_myr"), ("trace_relief_m", "relief_m"),
                                    ("trace_uplift_m", "uplift_m"), ("trace_extension_m", "extension_m"),
                                    ("trace_ridge_uplift_m", "ridge_uplift_m"),
                                    ("trace_ridge_thermal_m", "ridge_thermal_m"),
                                    ("trace_rift_birth_myr", "rift_birth_myr"),
                                    ("trace_rift_extension_m", "rift_extension_m"),
                                    ("trace_inversion_uplift_m", "inversion_uplift_m"),
                                    ("trace_erosion_m", "erosion_m"), ("trace_adjustment_m", "adjustment_m"),
                                    ("trace_suture", "suture")):
                    if source in data.files:
                        point[key] = float(data[source][marker])
                if "trace_rift_id" in data.files:
                    point["rift_id"] = int(data["trace_rift_id"][marker])
                for name in STRUCTURE_TRACE_FIELDS:
                    if 'trace_'+name in data.files:
                        point[name] = float(data['trace_'+name][marker])
            if metadata.get('structure_version'):
                point['structure_version'] = metadata['structure_version']
                if marker is None:
                    for name in STRUCTURE_TRACE_FIELDS[:5]:
                        if name in data.files:
                            point[name] = float(data[name].reshape(-1)[sample_cell])
            if 'trench' in data.files:
                point['trench_id'] = int(data['trench'].reshape(-1)[sample_cell])
            points.append(point)

    catalogue = read_record(path, manifest)
    times = [p["time_myr"] for p in points]
    events = []
    for event in catalogue["events"]:
        # Related means its recorded host plate was involved, not that the
        # event necessarily happened at the selected location.
        at = bisect_right(times, float(event.get("time_myr", 0))) - 1
        if at >= 0:
            point = points[at]
            if point["plate_uid"] is not None and point["plate_uid"] in event.get("plate_uids", []):
                events.append(event)
    return {
        "run_id": manifest.get("run_id"), "mode": "material" if trace_id is not None else "location",
        "description": description, "trace_id": trace_id, "selection": selection,
        "points": points, "events": events, "events_scope": "host_plate",
        "units": {"time": "Myr elapsed", "elevation": "metres relative to model sea level",
                  "relief_counters": "metres accumulated since marker birth; not a complete elevation budget",
                  "ridge_uplift_m": "cumulative metres, already included in uplift_m",
                  "ridge_thermal_m": "current temporary metres, additional to relief_m",
                  "rift_birth_myr": "structural scar creation in elapsed Myr; -1 means none",
                  "rift_extension_m": "cumulative observed metres, already included in extension_m",
                  "inversion_uplift_m": "cumulative realized metres, already included in uplift_m",
                  "coordinates": "longitude/latitude degrees", "suture": "dimensionless 0–1"},
    }
