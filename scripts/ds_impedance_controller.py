"""
ds_impedance_controller.py

Object-level control chain for the throw: a dynamical system (DS) that
drives the box toward the release pose/velocity, an object-level impedance
law that turns that desired acceleration into a desired object velocity
(blending in force feedback), and a grasp map that splits the resulting
object twist across both arms' admittance velocities into a single stacked
joint-velocity command.

Pure math on already-extracted state vectors -- zero MuJoCo dependency.
"""

import logging

import numpy as np

import config.throwing_config as config

logger = logging.getLogger(__name__)


# ----------------------------------------------------------------------
# 1. Dynamical system: drives the object toward the release pose/velocity.
# ----------------------------------------------------------------------

def compute_ds_accel(x_o: np.ndarray, xdot_o: np.ndarray,
                      x_rel: np.ndarray, xdot_rel: np.ndarray,
                      K_DS: np.ndarray = config.K_DS,
                      B_DS: np.ndarray = config.B_DS) -> np.ndarray:
    """
    Dynamical-system desired acceleration driving the object from its
    current pose/velocity toward the release pose/velocity:

        x_ddot_des = -K_DS @ (x_o - x_rel) - B_DS @ (xdot_o - xdot_rel)

    Args:
        x_o: (3,) current object position.
        xdot_o: (3,) current object linear velocity.
        x_rel: (3,) release-point target position.
        xdot_rel: (3,) release-point target velocity.
        K_DS: (3,3) DS stiffness gain.
        B_DS: (3,3) DS damping gain.

    Returns:
        np.ndarray, shape (3,): x_ddot_des.
    """
    return -K_DS @ (x_o - x_rel) - B_DS @ (xdot_o - xdot_rel)


# ----------------------------------------------------------------------
# 2. Object-level impedance: feedforward accel + impedance + force feedback
#    -> desired object velocity.
# ----------------------------------------------------------------------

def compute_desired_obj_vel(xdot_o: np.ndarray, x_ddot_des: np.ndarray,
                             w_fb_o: np.ndarray, w_hat_obj: np.ndarray,
                             x_rel: np.ndarray, x_o: np.ndarray,
                             D: np.ndarray = config.D_IMPEDANCE,
                             K1: np.ndarray = config.K_IMPEDANCE,
                             M_O: np.ndarray = config.M_O,
                             max_speed: float = config.MAX_OBJ_VEL) -> np.ndarray:
    """
    Object-level impedance law, blending feedforward acceleration, force
    feedback, and a positional correction term into a desired object
    velocity:

        x_dot_star = xdot_o + D^-1 (M_O @ x_ddot_des + w_fb_o - w_hat_obj
                                     + K1 @ (x_rel - x_o))

    result clipped to max_speed in norm.

    NOTE (simulation-only simplification): w_fb_o (momentum-observer
    feedback wrench) and w_hat_obj (estimated external wrench) are
    hardware state-estimation terms with no equivalent in a simulator that
    already has ground-truth state. Callers in sim should pass:
        w_fb_o   = np.zeros(3)                                (no observer needed)
        w_hat_obj = np.array([0, 0, -OBJECT_MASS * GRAVITY])  (gravity compensation)
    Both remain ordinary function parameters (not hardcoded here) so a real
    momentum observer / external-wrench estimator can be substituted for
    hardware transfer without touching this function.

    Args:
        xdot_o: (3,) current object linear velocity.
        x_ddot_des: (3,) desired acceleration, e.g. from compute_ds_accel.
        w_fb_o: (3,) momentum-observer feedback force term.
        w_hat_obj: (3,) estimated external force term (e.g. gravity).
        x_rel: (3,) release-point target position.
        x_o: (3,) current object position.
        D: (3,3) impedance damping gain.
        K1: (3,3) impedance stiffness gain.
        M_O: (3,3) object mass matrix.
        max_speed: hard clip on ||x_dot_star||.

    Returns:
        np.ndarray, shape (3,): x_dot_o_star, clipped to max_speed.
    """
    x_dot_star = xdot_o + np.linalg.solve(
        D, M_O @ x_ddot_des + w_fb_o - w_hat_obj + K1 @ (x_rel - x_o)
    )

    speed = np.linalg.norm(x_dot_star)
    if speed > max_speed and speed > 0.0:
        logger.warning(
            "compute_desired_obj_vel: unclipped |x_dot_star|=%.4f m/s exceeds "
            "max_speed=%.4f m/s; clipping direction-preserving.",
            speed, max_speed,
        )
        x_dot_star = x_dot_star * (max_speed / speed)

    return x_dot_star


# ----------------------------------------------------------------------
# 3. Grasp map and combined joint velocity.
# ----------------------------------------------------------------------

# Splits the object's 6D twist evenly across both arms' 6D admittance
# velocity commands: (12, 6).
G_T = np.vstack([0.5 * np.eye(6), 0.5 * np.eye(6)])


def _joint_limit_avoidance_gradient(q: np.ndarray, joint_ranges: np.ndarray) -> np.ndarray:
    """
    Liegeois (1977) joint-limit-avoidance potential gradient.

        H(q) = (1 / (2n)) * sum_i [ (q_i - qbar_i) / (q_i_max - q_i_min) ]^2
        grad H_i = (1/n) * (q_i - qbar_i) / (q_i_max - q_i_min)^2

    grad H points AWAY from mid-range as q_i approaches either limit;
    negating it (done in compute_dual_arm_qdot) yields a velocity that
    pulls q_i back toward mid-range -- the desired avoidance behavior.

    Args:
        q: (n,) current joint positions, rad.
        joint_ranges: (n, 2) array of [lower, upper] limits per joint,
            same row order as q (see config.JOINT_RANGES).

    Returns:
        np.ndarray, shape (n,): grad H(q).
    """
    q = np.asarray(q, dtype=np.float64)
    joint_ranges = np.asarray(joint_ranges, dtype=np.float64)
    q_min = joint_ranges[:, 0]
    q_max = joint_ranges[:, 1]
    q_mid = 0.5 * (q_min + q_max)
    n = q.shape[0]
    span = q_max - q_min
    return (1.0 / n) * (q - q_mid) / (span ** 2)


def compute_dual_arm_qdot(J_H_pinv: np.ndarray, xdot_stack: np.ndarray,
                           x_dot_obj_star_6d: np.ndarray,
                           J_H: np.ndarray = None,
                           q: np.ndarray = None,
                           joint_ranges: np.ndarray = None,
                           k_joint_limit: float = 0.0) -> np.ndarray:
    """
    Combined dual-arm joint velocity command, with an optional secondary
    (nullspace) joint-limit-avoidance task:

        q_dot_primary = J_H_pinv @ (xdot_stack + G_T @ x_dot_obj_star_6d)
        q_dot = q_dot_primary + N @ q_dot_avoid   (if nullspace args given)

    where N = I - J_H_pinv @ J_H is the (approximate, since J_H_pinv is a
    DAMPED pseudo-inverse -- see dual_arm_kinematics.damped_pinv) nullspace
    projector of the primary task, and q_dot_avoid = -k_joint_limit *
    grad(H(q)) is the Liegeois joint-limit-avoidance velocity (see
    _joint_limit_avoidance_gradient). Projecting through N means this
    secondary term acts (to first order) only in directions that don't
    disturb the primary Cartesian task -- letting a redundant joint (e.g.
    right_fr3_joint1) move to avoid its limit without changing where the
    box goes.

    The nullspace term is opt-in and backward-compatible: if J_H, q, or
    joint_ranges is None (the default), or k_joint_limit is 0.0 (the
    default), this reduces EXACTLY to the original primary-task-only
    behavior -- existing callers that don't pass the new arguments are
    unaffected.

    Args:
        J_H_pinv: (14, 12) pseudo-inverse of the stacked dual-arm Jacobian,
            from dual_arm_kinematics.py.
        xdot_stack: (12,) = concat(xdot_admittance_left, xdot_admittance_right),
            each a 6D admittance Cartesian velocity command.
        x_dot_obj_star_6d: (6,) = concat(x_dot_obj_star (3,), zeros(3)) --
            no commanded object angular velocity, since end-effector /
            object orientation must stay fixed throughout the throw.
        J_H: optional, (12, 14) stacked dual-arm Jacobian (NOT its pinv --
            needed separately to build the nullspace projector). Required,
            along with q and joint_ranges, to activate the nullspace term.
        q: optional, (14,) current joint positions [left(7), right(7)], rad.
        joint_ranges: optional, (14, 2) [lower, upper] joint limits, same
            [left(7), right(7)] row order as q (see config.JOINT_RANGES).
        k_joint_limit: gain on the nullspace joint-limit-avoidance term
            (see config.K_JOINT_LIMIT_AVOID). 0.0 (default) disables the
            term even if J_H/q/joint_ranges are supplied.

    Returns:
        np.ndarray, shape (14,): q_dot, the combined joint-velocity command
        for both arms.
    """
    qdot_primary = J_H_pinv @ (xdot_stack + G_T @ x_dot_obj_star_6d)

    if (
        k_joint_limit > 0.0
        and J_H is not None
        and q is not None
        and joint_ranges is not None
    ):
        grad_H = _joint_limit_avoidance_gradient(q, joint_ranges)
        qdot_avoid = -k_joint_limit * grad_H
        n = J_H_pinv.shape[0]
        N = np.eye(n) - J_H_pinv @ J_H
        qdot_primary = qdot_primary + N @ qdot_avoid

    return qdot_primary
