"""Read-only conservative buoyancy loads on projected entry/contact cells.

The supplied volume and mass belong to one uniform material-reference region.
Projection redistributes their force over a smaller horizontal footprint; it
does not create crust or buoyancy. Mixed phase histories require their own
regional inventories before using this calculation.
"""

import math

import numpy as np

import entry_contact_channel
import finite_entry_arc


def uniform_cell_buoyancy(volume_km3, mass_kg, face_reference_area_km2,
                          contact_cells):
    """Allocate one region's conserved buoyancy to matched contact cells.

    Returns mean pressure and extensive force for each cell. `contact_cells`
    carry material-reference and physical areas from a checked projection.
    This uniform approximation must not merge distinct local phase columns.
    """
    volume = float(volume_km3)
    mass = float(mass_kg)
    face_area = float(face_reference_area_km2)
    if (not np.isfinite([volume,mass,face_area]).all()
            or volume <= 0. or mass <= 0. or face_area <= 0.
            or not isinstance(contact_cells,list)):
        raise ValueError('Projected buoyancy needs positive uniform material inventory and contact cells.')
    area = []
    physical = []
    for cell in contact_cells:
        if not isinstance(cell,dict):
            raise ValueError('Projected buoyancy needs contact-cell area records.')
        a = float(cell.get('material_reference_area_km2',float('nan')))
        p = float(cell.get('physical_area_km2',float('nan')))
        if not np.isfinite([a,p]).all() or a <= 0. or p <= 0.:
            raise ValueError('Projected buoyancy cells need positive finite areas.')
        area.append(a)
        physical.append(p)
    covered = math.fsum(area)
    if covered > face_area + max(1e-7,face_area*2e-9):
        raise ValueError('Projected contact claims more material than its source region.')
    reference_pressure = float(entry_contact_channel.buoyancy_surface_load(
        volume,mass,face_area))
    cells = []
    for index,(a,p) in enumerate(zip(area,physical)):
        material_fraction = a/face_area
        force = reference_pressure*a*1e6
        pressure = force/(p*1e6)
        if not np.isfinite([force,pressure]).all():
            raise ValueError('Projected buoyancy exceeds the finite force range.')
        cells.append(dict(cell_index=index,material_fraction=material_fraction,
                          material_volume_km3=volume*material_fraction,
                          material_mass_kg=mass*material_fraction,
                          mean_buoyancy_pa=pressure,buoyancy_force_n=force,
                          physical_area_km2=p,material_reference_area_km2=a))
    total = math.fsum(cell['buoyancy_force_n'] for cell in cells)
    expected = reference_pressure*covered*1e6
    if not math.isclose(total,expected,rel_tol=2e-13,abs_tol=1e-4):
        raise ValueError('Projected buoyancy lost conserved material force.')
    return dict(cells=cells,contact_buoyancy_force_n=total,
                represented_material_area_km2=covered,
                represented_physical_area_km2=math.fsum(physical),
                source_reference_buoyancy_pa=reference_pressure,
                scope='read-only uniform-region load; no native force or phase-source commit')


def uniform_contact_entry_work(lower_triangle, hinge_normal, dip_degrees,
                               volume_km3, mass_kg, projected_pair,
                               *, radius_km=6371.):
    """Integrate the old affine entry potential over projected contact cells.

    This is the entry energy to remove locally before replacing that contact
    by a channel law. The depth is the existing nodal-affine conveyor depth,
    integrated in conserved reference area; physical pressure times physical
    area times reference-weighted mean depth gives the same work.
    """
    lower = np.asarray(lower_triangle,float)
    normal = np.asarray(hinge_normal,float)
    dip = float(dip_degrees)
    if (lower.shape != (3,3) or normal.shape != (3,)
            or not np.isfinite(lower).all() or not np.isfinite(normal).all()
            or abs(float(np.linalg.norm(normal))-1.) > 1e-10
            or not np.isfinite(dip) or not 0. < dip < 90.
            or not np.isfinite(radius_km) or radius_km <= 0.
            or not isinstance(projected_pair,dict)
            or projected_pair.get('contact_cells') is None):
        raise ValueError('Projected entry work needs checked geometry, dip and contact cells.')
    face_area = float(projected_pair['face_reference_area_km2'])
    load = uniform_cell_buoyancy(volume_km3,mass_kg,face_area,
                                 projected_pair['contact_cells'])
    q = float(radius_km)*1000.*np.arcsin(np.clip(lower@normal,-1.,1.))
    depth_factor = math.sin(math.radians(dip))
    result = []
    for polygon,cell_load in zip(projected_pair['contact_cells'],load['cells']):
        bary = np.asarray(polygon['material_barycentric_polygon'],float)
        if (bary.ndim != 2 or bary.shape[1] != 3 or len(bary) < 3
                or not np.isfinite(bary).all()
                or not np.allclose(bary.sum(axis=1),1.,rtol=0.,atol=1e-9)):
            raise ValueError('Projected entry work needs a material-barycentric polygon.')
        integral = 0.
        area = 0.
        for second in range(1,len(bary)-1):
            triangle = bary[[0,second,second+1]]
            a,b,c = triangle[:,1:]
            fraction = float((b[0]-a[0])*(c[1]-a[1])-
                             (b[1]-a[1])*(c[0]-a[0]))
            if fraction < -2e-12:
                raise ValueError('Projected entry polygon reversed material winding.')
            if fraction <= 0.:continue
            nodal_q = triangle@q
            local_mean, *_ = finite_entry_arc.integrate(
                nodal_q,np.ones(3),np.ones(3))
            area += face_area*fraction
            integral += face_area*fraction*local_mean
        expected_area = cell_load['material_reference_area_km2']
        if abs(area-expected_area) > max(1e-7,expected_area*2e-10):
            raise ValueError('Projected entry work lost its material area.')
        depth_integral = depth_factor*integral
        mean_depth = depth_integral/expected_area
        work = load['source_reference_buoyancy_pa']*depth_integral*1e6
        physical_work = (cell_load['mean_buoyancy_pa']*
                         cell_load['physical_area_km2']*1e6*mean_depth)
        if not math.isclose(work,physical_work,rel_tol=2e-12,abs_tol=1e-4):
            raise ValueError('Projected contact pressure and entry work disagree.')
        result.append(dict(cell_index=cell_load['cell_index'],
                           mean_entry_depth_m=mean_depth,
                           entry_work_j=work,
                           buoyancy_force_n=cell_load['buoyancy_force_n']))
    return dict(cells=result,contact_entry_work_j=math.fsum(
                    cell['entry_work_j'] for cell in result),
                contact_buoyancy_force_n=load['contact_buoyancy_force_n'],
                scope='read-only old-entry handoff work; no native force or source commit')
