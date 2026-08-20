"""
release_velocity.py

Pure-function derivation of the release velocity needed to throw the box
from a release point p_rel to a landing point p_land at a fixed launch
angle theta, under simple projectile motion (constant gravity, no drag).

Zero MuJoCo dependency -- everything here is plain numpy, so this module is
independently unit-testable without a simulator.

Derivation (standard projectile-range equation solved for launch speed,
decomposed into horizontal range R and vertical drop/rise Δz):

    Δp = p_land - p_rel
    R  = ||Δp_xy||                                   (horizontal range)
    Δz = Δp_z                                        (can be negative)
    denom = R * tan(theta) - Δz
    v_rel = sqrt( GRAVITY * R^2 / (2 * cos(theta)^2 * denom) )
    t_f   = R / (v_rel * cos(theta))

Note on z_hat: the underlying paper uses ẑ = Δp_z / |Δp_z|, which is
undefined at Δz = 0. We instead always use the fixed world-up unit vector
[0, 0, 1] for z_hat, regardless of the sign of Δz -- the sign of Δz is
already fully accounted for by the denom term in the v_rel formula, so no
information is lost by fixing z_hat.
"""

import logging

import numpy as np

import config.throwing_config as config

logger = logging.getLogger(__name__)


def compute_release_velocity(p_rel: np.ndarray, p_land: np.ndarray, theta: float) -> dict:
    """
    Compute the release velocity (scalar, vector, time-of-flight) needed to
    throw an object from p_rel to p_land at launch angle theta, under
    simple projectile motion.

    Args:
        p_rel: (3,) release point, world frame.
        p_land: (3,) landing point, world frame.
        theta: launch angle in radians, measured from horizontal.

    Returns:
        dict with keys:
            'v_rel_scalar' -- float, release speed (m/s), clipped to
                config.MAX_OBJ_VEL if the unclipped solution exceeds it.
            'v_rel_vector' -- np.ndarray (3,), release velocity vector.
            't_f'          -- float, time of flight (s), computed using the
                (possibly clipped) v_rel_scalar.
            'R'            -- float, horizontal range (m).
            'e_h'          -- np.ndarray (2,), horizontal unit direction
                from p_rel to p_land (xy-plane).

    Raises:
        ValueError: if p_rel and p_land coincide horizontally (R == 0, so
            the horizontal direction e_h is undefined), or if
            `R * tan(theta) - Δz <= 0` (no real launch-speed solution
            exists for this angle/geometry -- the angle is too shallow to
            clear/reach the required Δz at this range).
    """
    p_rel = np.asarray(p_rel, dtype=np.float64)
    p_land = np.asarray(p_land, dtype=np.float64)

    delta_p = p_land - p_rel
    delta_p_xy = delta_p[:2]
    R = float(np.linalg.norm(delta_p_xy))

    if R <= 0.0:
        raise ValueError(
            "p_rel and p_land coincide in the horizontal (xy) plane; "
            "the horizontal direction e_h is undefined."
        )

    delta_z = float(delta_p[2])
    e_h = delta_p_xy / R
    z_hat = np.array([0.0, 0.0, 1.0])

    denom = R * np.tan(theta) - delta_z
    if denom <= 0:
        raise ValueError(
            "No real solution: angle too shallow for this Δz/R combination "
            f"(R={R:.4f} m, Δz={delta_z:.4f} m, theta={theta:.4f} rad, "
            f"denom={denom:.4f})."
        )

    v_rel = np.sqrt(config.GRAVITY * R ** 2 / (2 * np.cos(theta) ** 2 * denom))

    if v_rel > config.MAX_OBJ_VEL:
        logger.warning(
            "compute_release_velocity: unclipped v_rel=%.4f m/s exceeds "
            "MAX_OBJ_VEL=%.4f m/s; clipping. Note the underlying projectile "
            "formula ignores this limit, so the clipped throw will fall "
            "short of p_land.",
            v_rel, config.MAX_OBJ_VEL,
        )
        v_rel = config.MAX_OBJ_VEL

    t_f = R / (v_rel * np.cos(theta))

    v_rel_vector = (
        v_rel * np.cos(theta) * np.array([e_h[0], e_h[1], 0.0])
        + v_rel * np.sin(theta) * z_hat
    )

    return {
        "v_rel_scalar": float(v_rel),
        "v_rel_vector": v_rel_vector,
        "t_f": float(t_f),
        "R": R,
        "e_h": e_h,
    }


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    # Placeholder p_rel for standalone self-test only. At runtime the real
    # p_rel is the box's actual grasped pose, read from
    # MjInterface.get_object_state()['pos'] at throw-phase entry.
    p_rel = config.LEFT_GRASP_WORLD_POS
    p_land = config.LANDING_POINT
    theta = config.THROW_ANGLE

    print("=" * 70)
    print("release_velocity.py self-test")
    print("=" * 70)
    print(f"p_rel  (placeholder, LEFT_GRASP_WORLD_POS): {p_rel}")
    print(f"p_land (config.LANDING_POINT):               {p_land}")
    print(f"theta  (config.THROW_ANGLE):                 {theta:.4f} rad "
          f"({np.rad2deg(theta):.2f} deg)")

    result = compute_release_velocity(p_rel, p_land, theta)

    print()
    print(f"R (horizontal range)   : {result['R']:.4f} m")
    print(f"e_h (horizontal dir)   : {result['e_h']}")
    print(f"v_rel_scalar            : {result['v_rel_scalar']:.4f} m/s")
    print(f"v_rel_vector             : {result['v_rel_vector']}")
    print(f"t_f (time of flight)    : {result['t_f']:.4f} s")

    # Sanity check: integrate simple projectile motion forward using the
    # returned v_rel_vector and t_f, and confirm it lands near p_land.
    g_vec = np.array([0.0, 0.0, -config.GRAVITY])
    predicted_land = (
        p_rel + result["v_rel_vector"] * result["t_f"]
        + 0.5 * g_vec * result["t_f"] ** 2
    )
    print()
    print(f"predicted landing point (forward-integrated): {predicted_land}")
    print(f"p_land (target):                               {p_land}")
    print(f"landing position error: "
          f"{np.linalg.norm(predicted_land - p_land):.6f} m")
