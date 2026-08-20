"""
contact_admittance.py

Contact-level admittance control for grasp force closure:

    x_dot = K^-1 (F* - F_meas)

This is a 6D Cartesian velocity command per arm. Rows 0-2 (translational)
regulate the commanded squeeze force -- when the measured contact force
matches the target F*, x_dot's translational part goes to zero and the pad
holds position. Rows 3-5 (rotational) have F*_rot = 0 by construction (see
config.desired_wrench), so they actively resist any torque buildup that
would rotate the pad relative to the box; this is the mechanism that keeps
end-effector orientation fixed relative to the box and prevents slip during
the throw swing.

Zero MuJoCo dependency -- pure numpy over 6-vectors/6x6 matrices.
"""

import numpy as np

import config.throwing_config as config


def compute_admittance_velocity(F_star: np.ndarray, F_meas: np.ndarray,
                                 K: np.ndarray = config.ADMITTANCE_STIFFNESS,
                                 max_lin_vel: float = config.MAX_ADMITTANCE_VEL,
                                 max_ang_vel: float = 0.2,
                                 axis: np.ndarray | None = None) -> np.ndarray:
    """
    Contact-level admittance law: x_dot = K^-1 (F* - F_meas), with the
    translational and rotational parts of x_dot separately clipped.

    Args:
        F_star: (6,) desired wrench [Fx,Fy,Fz,Tx,Ty,Tz].
        F_meas: (6,) measured wrench [Fx,Fy,Fz,Tx,Ty,Tz]. Both F_star and
            F_meas must already be expressed in the same frame as `axis`
            (world frame) -- caller must rotate a sensor-local reading
            first (see MjInterface.get_wrench_world).
        K: (6,6) admittance stiffness matrix (diagonal). Defaults to
            config.ADMITTANCE_STIFFNESS.
        max_lin_vel: cap on the norm of x_dot[:3] (m/s). Defaults to
            config.MAX_ADMITTANCE_VEL.
        max_ang_vel: cap on the norm of x_dot[3:] (rad/s). Defaults to 0.2.
        axis: optional (3,) world-frame direction. When given, the law is
            restricted to that single translational axis: the wrench
            error's component along `axis` is kept, everything
            orthogonal to it (the other 2 translational directions) AND
            all 3 rotational components are zeroed BEFORE the solve --
            not after -- so sensor noise on those axes never turns into a
            nonzero commanded velocity there. When None (default),
            behavior is unchanged from before (full 6D law, used by
            THROW/RELEASE).

    Returns:
        np.ndarray, shape (6,): the (clipped) Cartesian admittance velocity
        command [vx, vy, vz, wx, wy, wz].
    """
    F_star = np.asarray(F_star, dtype=np.float64)
    F_meas = np.asarray(F_meas, dtype=np.float64)
    K = np.asarray(K, dtype=np.float64)

    F_err = F_star - F_meas

    if axis is not None:
        axis_unit = np.asarray(axis, dtype=np.float64)
        axis_unit = axis_unit / np.linalg.norm(axis_unit)
        lin_err_masked = np.dot(F_err[:3], axis_unit) * axis_unit
        F_err = np.concatenate([lin_err_masked, np.zeros(3)])

    x_dot = np.linalg.solve(K, F_err)

    lin = x_dot[:3]
    ang = x_dot[3:]

    lin_norm = np.linalg.norm(lin)
    if lin_norm > max_lin_vel and lin_norm > 0.0:
        lin = lin * (max_lin_vel / lin_norm)

    ang_norm = np.linalg.norm(ang)
    if ang_norm > max_ang_vel and ang_norm > 0.0:
        ang = ang * (max_ang_vel / ang_norm)

    return np.concatenate([lin, ang])


class GraspForceCloser:
    """
    Small state object gating the target grasp wrench fed to the admittance
    law. Starts active (squeezing at config.desired_wrench per side);
    state_machine.py calls deactivate() exactly once, at the release
    trigger, so F* snaps to zero for both arms and the admittance law stops
    actively squeezing.
    """

    def __init__(self):
        self.active = True

    def target_wrench(self, side: str) -> np.ndarray:
        """
        Desired grasp wrench for the given side, gated by self.active.

        Args:
            side: 'left' or 'right'.

        Returns:
            np.ndarray, shape (6,): config.desired_wrench(side) while
            active, else config.zero_wrench().
        """
        return config.desired_wrench(side) if self.active else config.zero_wrench()

    def deactivate(self):
        """Turn off force closure; subsequent target_wrench() calls return zero_wrench()."""
        self.active = False
