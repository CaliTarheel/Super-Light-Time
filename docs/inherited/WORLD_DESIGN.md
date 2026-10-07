# Regional world design, version 1

> **Documentation scope:** This retained design document includes historical development results. Refer to [KNOWN_LIMITATIONS.md](KNOWN_LIMITATIONS.md) and [VALIDATION.md](VALIDATION.md) for the initial GitHub snapshot. References to local `validation/`, `output/`, legacy copies, and handoff documents describe artifacts not included in this repository.

The starting-world editor now supports up to 32 authored geographic regions.
Open **Regional design controls**, select an effect, enter a reason and an active
time interval, then preview or pick the region's center on the map. **Add region**
enables the layer. **Save world** includes the full design; import, regridding,
whole-globe rotation, undo and redo retain it. The running and saved experiment
keeps its own recorded design.

For a compatible paused experiment, edit the regional design and choose
**Branch at … Myr**. A separate experiment starts paused at the exact preserved
checkpoint, with the parent's recorded frames retained and the new design
recorded for continuation. Its ordinary **Resume** action advances the branch.
New or changed active regions must start at or after that checkpoint; controls
that already affected the parent's recorded past cannot be rewritten or removed.
The branch uses checkpoint geometry and motion. Source frames and checkpoint
remain preserved. Old source-incompatible checkpoints continue in their archived
engine and cannot be branched into new mechanics by bypassing compatibility.

Each region is a spherical cap with longitude, latitude and a radius from 0.25°
to 45°. Caps cross the date line and poles using spherical geometry. Their
positions remain fixed in world coordinates as material moves through them.
The displayed footprint previews the geographic cap; fully protected material
includes entire triangles that touch it, so the effective boundary depends on
material resolution. Tiny protection caps inside a triangle are retained.

**Protect an interior** reduces its residual deformation response. Strength 1
fixes every vertex of touching material triangles to its exact rigid plate
rotation. **Prefer a weak region** increases the existing admitted deformation
response by at most 50%, tapering over the outer 20% of the cap. Overlapping
regions use their maximum strengths; protection takes precedence. Original
cratonic constraints, geometric quality limits and physical column bounds remain
in force. Quiet relative motion cannot acquire deformation from this layer.

These controls do not generate mantle forces, assign final plates, weaken the
separate internal-rift failure criterion, prescribe uplift/volcanism, or solve
crustal flow. Protected material can still change plate ownership through the
existing topology rules. No reference material or column budget is added or
removed. Geometric area changes feed the same reciprocal thickness update as
ordinary deformation. The feature is a bounded kinematic artistic control.

An interval includes its starting time and excludes its ending time. Integration
steps split at authored transitions, including transitions between saved epochs.
The history records supplied design, revision, reason, region geometry and
separate authored start/end events. Snapshot JSON includes `world_design` and
`world_design_diagnostics`; checkpoint state includes the active region IDs.
The UI's **Author's interventions** panel reads the saved snapshot, while the
editor changes a future experiment's draft.

Disabled layers, zero-strength regions and inactive intervals leave deformation
inputs unchanged. Historical events and portable metadata can still describe an
authored layer that was disabled. Unsupported schemas, duplicate IDs, missing
reasons, nonfinite values and conflicting config/input designs are rejected.

```json
{
  "version": 1,
  "enabled": true,
  "revision": 1,
  "interventions": [{
    "id": "northern-interior",
    "kind": "protected",
    "lon_deg": 179,
    "lat_deg": 70,
    "radius_deg": 8,
    "strength": 1,
    "start_myr": 50,
    "end_myr": 150,
    "reason": "Retain a coherent northern interior during convergence."
  }]
}
```

Focused validation: `tests/test_world_design.py` tests portable validation,
spherical footprints and rotation, exact disabled deformation, full protection,
weak-region constraints, interval events, restored provenance and the native
manager's saved JSON/source closure. `tests/test_editor_ui.cjs` exercises actual
UI handlers for preview, map picking, add/remove/toggle, import, undo, editing
locks and the saved authored/collision review panels. Existing deformation,
native material transport and orientation UI checks also pass.
