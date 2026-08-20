"""
dual_arm_kinematics.py

Builds the stacked, block-diagonal dual-arm Jacobian J_H (12x14) from the
two arms' individual 6x7 site Jacobians, and its damped least-squares
pseudo-inverse J_H_pinv (14x12), used by ds_impedance_controller.py's
compute_dual_arm_qdot to map the combined 12D admittance+object-twist
command back into a single 14D joint-velocity command.

Note: compute_dual_arm_qdot itself (and the G_T grasp map) live in
ds_impedance_controller.py, not here -- they're the final step of that
module's equation chain, not a kinematics utility.
"""

import numpy as np

import config.throwing_config as config
from mj_interface import MjInterface


def build_stacked_jacobian(interface: MjInterface) -> np.ndarray:
    """
    Stacked block-diagonal dual-arm Jacobian.

    Layout (12, 14): top-left 6x7 block is the left arm's EE-site Jacobian,
    bottom-right 6x7 block is the right arm's EE-site Jacobian, off-diagonal
    blocks are zero (neither arm's joints move the other arm's
    end-effector).

    Args:
        interface: MjInterface wrapping the live sim state.

    Returns:
        np.ndarray, shape (12, 14): J_H.
    """
    J_H = np.zeros((12, 14))
    J_H[0:6, 0:7] = interface.get_site_jacobian(config.LEFT_EE_SITE, "left")
    J_H[6:12, 7:14] = interface.get_site_jacobian(config.RIGHT_EE_SITE, "right")
    return J_H


def damped_pinv(J: np.ndarray, damping: float = 0.05) -> np.ndarray:
    """
    Damped least-squares pseudo-inverse of J: J.T @ (J @ J.T + damping^2 I)^-1.

    A larger damping is used here (default 0.05) than in ik_solver.py's IK
    loop (0.01): this runs live during the fast throw motion, which sweeps
    a large, fast workspace and is more prone to near-singular
    configurations than slow reaching, so robustness matters more than
    precision here. IK solving, by contrast, is offline/one-shot and can
    afford more iterations for accuracy instead of leaning on damping.

    Args:
        J: (m, n) Jacobian to invert (here, J_H with m=12, n=14).
        damping: damping factor (Levenberg-Marquardt lambda).

    Returns:
        np.ndarray, shape (n, m): the damped pseudo-inverse.
    """
    JJt = J @ J.T
    return J.T @ np.linalg.inv(JJt + damping ** 2 * np.eye(JJt.shape[0]))