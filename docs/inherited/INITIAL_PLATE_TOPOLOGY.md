# Initial plate topology

Starting crust and starting plate ownership are separate concepts.

A coast is a material boundary between continental and oceanic crust. It is not
necessarily a plate boundary. An oceanic strip can travel on the same rigid
plate as an adjacent continent and form a passive margin.

## Version 1: passive-margin aprons

A starting-world JSON may include:

```json
{
  "initial_plate_topology": {
    "version": 1,
    "mode": "passive_margin_aprons",
    "seed": 41,
    "passive_coast_fraction": 0.65,
    "apron_width_km": 650,
    "max_attached_ocean_fraction": 0.45
  }
}
```

The ordinary land-first plate seeding runs first. The topology helper then
selects broad, seeded sectors of those coasts and floods outward only through
existing oceanic cells. Reached ocean retains oceanic crust, cooling age and
bathymetry, but shares the adjacent continent's plate owner.

The helper:

- creates no crust;
- creates no additional plates;
- imposes no new velocity;
- never reclassifies ocean as continent;
- limits the attached share of the starting ocean;
- shrinks its physical apron width when necessary to leave one connected
  independent-ocean reservoir.

The resulting world therefore contains both passive coastlines, where crust
type changes inside one plate, and active coastlines, where crust type and plate
owner both change.

## Highland 65

The `highland65` preset carries a version-one topology seed by default. Its
65% target still refers to continental/cratonic material area, not emergent
surface above sea level.

The preset currently requests 65% of coastline as passive-margin candidates, a
650 km maximum oceanic apron and at most 45% of the starting ocean attached to
continental plates. Connectivity can force a narrower realized apron.

These are worldbuilding initial-condition choices, not calibrated terrestrial
constants.

## Interaction with primordial ocean initialization

Explicit `primordial_ocean` configuration takes precedence. When it is enabled,
the preset passive-margin seed is not applied before the authored primordial
ocean partition. This prevents two independent initial ownership systems from
reinterpreting the same water in sequence.

Existing starting worlds containing only `width`, `height` and `crust`
retain the historical crust-derived plate ownership unchanged.
