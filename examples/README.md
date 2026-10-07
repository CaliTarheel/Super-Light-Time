# Lite starter

[lite-highland65.json](lite-highland65.json) is a 20 Myr Highland 65 starter using
the app's full detail settings: a 5,120-cell tectonic mesh, level-4 coastlines,
adaptive detail level 1, 1,024 rift mechanics nodes, a 512 × 256 review map, and
saved frames every 2 Myr. Terrain and export quality controls remain available.
The shorter duration is for reviewing the new force law.

Create the matching initial map from the fork root:

```powershell
python -c "import json; from pathlib import Path; from native_engine import make_initial; from server import write_json; c=json.loads(Path('examples/lite-highland65.json').read_text()); p=Path('validation/lite-highland65'); p.mkdir(parents=True,exist_ok=True); write_json(p/'initial.json',make_initial(c,preset='highland65'))"
```

Then start the run:

```powershell
python run_simulation.py --config examples/lite-highland65.json --initial validation/lite-highland65/initial.json
```

The prepared local initial map uses the same full settings. Its startup resolves
six declared trenches and solves motion with no slab inventory or velocity kick.
Use the app to choose a different starting map or the same higher detail settings
used by another run. Lite does not reduce those settings to gain speed.
