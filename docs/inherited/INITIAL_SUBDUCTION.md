# Selective inherited subduction

Deep Time can initialize a fresh reviewed-physics world with pre-existing slabs.
The initial condition may now choose a **subset of complete connected active
margin arcs** rather than declaring every eligible ocean/continent boundary to
be a mature trench.

A starting world may include:

```json
{
  "initial_subduction": {
    "enabled": true,
    "initial_slab_depth_km": 100,
    "target_margin_fraction": 0.45,
    "selection_seed": 41
  }
}
```

When the run uses `physics_profile="reviewed_v1"` and the run configuration
does not explicitly supply its own `primordial_subduction` settings, this
starting-world record becomes the primordial-subduction configuration.

Selection operates on connected local margin components after passive margins
and plate ownership are already established. Whole components are selected;
individual neighboring edges are never alternated on/off to hit an exact
length fraction. Longer arcs are preferred and a bounded seeded term resolves
near-equal geographic choices. The requested fraction is therefore a target,
not an exact equality.

Existing explicit `primordial_subduction` configuration takes precedence.
The historical default remains `target_margin_fraction=1`, so worlds that
already opted into inherited subduction continue to initialize every eligible
margin unless they request selective arcs.

For `highland65`, the preset records a 45% target with a 100 km inherited
vertical slab depth. These are worldbuilding initial-condition choices, not a
calibrated terrestrial reconstruction.

### Highland65 integration with fixed initial ownership

Passive margin aprons create mixed plates without changing their crust. Their
represented oceanic portions are valid incoming water carriers; initialization
records these carrier UIDs explicitly rather than assuming all ocean belongs to
one plate. This follows the guiding document's distinction between continental
material and plate ownership (see SCIENCE.md, Applying the guiding document).

When several exact source fronts map onto one control graph contact, ocean/ocean
pieces and the two coastal polarities receive separate force quadrature contacts.
They retain the original graph edge, exact source segments, and total length;
ownership, surface area, slab depth, and force coefficients do not change. This
is a computational representation correction, not a new geological mechanism
or a departure from Scotese's slab-driven motion principle. A finer graph can
still be needed if same-material fronts cancel their normals.

At the unchanged zero epoch, inherited trench matching is restricted to the
selected arcs. The ordinary moving-trace search radius must not spread the
declared initial slab inventory onto neighboring unselected margins. Subsequent
steps use the existing historical trench matching and lifecycle rules.
