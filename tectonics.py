"""Public entry point for the native spherical mesh edition of Deep Time."""
from native_engine import DEFAULT_CONFIG, Simulation, make_initial, validate_config
from raster_engine import RADIUS_KM, BOUNDARY_NAMES, ADVECTION_CHUNK_CELLS, _rotate, _unit, _xyz
