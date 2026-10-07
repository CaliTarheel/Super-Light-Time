"""Read-only fixed-dip slab work with horizontal plate intake.

The native material mesh and Euler speeds describe horizontal surface
positions. For dip theta, inextensible slab travel u therefore projects to
horizontal intake u*cos(theta), and its vertical speed is u*sin(theta).
This candidate is not selected by the production plate balance.
"""

import math


def horizontal_intake_attachment(down_speed_m_s, hinge_speed_m_s,
                                 dip_degrees, mantle_drag_n_s_m,
                                 neck_drag_n_s_m, slab_weight_n):
    """Eliminate along-slab speed from one dissipative moving-hinge potential.

    ``d-h`` is horizontal intake. The neck pays ``(u cos(theta)-(d-h))²``;
    mantle drag pays both components of ``(h+u cos(theta), u sin(theta))``.
    The signed weight performs work at vertical speed ``u sin(theta)``.
    Returned plate forces are derivatives of the minimized potential.
    """
    values = tuple(map(float, (down_speed_m_s, hinge_speed_m_s,
                               dip_degrees, mantle_drag_n_s_m,
                               neck_drag_n_s_m, slab_weight_n)))
    d, h, angle, cs, cn, weight = values
    if (not all(math.isfinite(value) for value in values)
            or not 0. < angle < 90. or cs <= 0. or cn <= 0.):
        raise ValueError('Horizontal slab closure needs finite speeds, dip and positive drags.')
    sine, cosine = math.sin(math.radians(angle)), math.cos(math.radians(angle))
    intake = d - h
    denominator = cs + cn * cosine**2
    u = (weight * sine + cn * cosine * intake
         - cs * cosine * h) / denominator
    slip = cosine * u - intake
    horizontal = h + cosine * u
    vertical = sine * u
    down_force = cn * slip
    over_force = -cn * slip - cs * horizontal
    gravity_power = weight * vertical
    plate_power = down_force * d + over_force * h
    mantle_power = cs * (horizontal**2 + vertical**2)
    neck_power = cn * slip**2
    return dict(along_slab_speed_m_s=u, horizontal_intake_m_s=intake,
                mantle_horizontal_speed_m_s=horizontal,
                vertical_speed_m_s=vertical, neck_horizontal_slip_m_s=slip,
                down_plate_force_n=down_force, over_plate_force_n=over_force,
                gravity_power_w=gravity_power, plate_power_w=plate_power,
                mantle_dissipation_w=mantle_power,
                neck_dissipation_w=neck_power,
                power_residual_w=gravity_power - plate_power
                                 - mantle_power - neck_power,
                minimized_potential_w=.5 * (mantle_power + neck_power)
                                      - gravity_power,
                scope='read-only horizontal-intake alternative; native law unchanged')
