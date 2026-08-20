"""Experimental finite-time terminal-state trajectory and grasp kinematics."""

from __future__ import annotations

import numpy as np

import config.throwing_trajectory_config as config


class CubicTerminalTrajectory:
    """Cubic Hermite trajectory matching position and velocity endpoints."""

    def __init__(self, x0, v0, xf, vf, duration: float):
        if duration <= 0:
            raise ValueError("duration must be positive")
        self.x0 = np.asarray(x0, dtype=float)
        self.v0 = np.asarray(v0, dtype=float)
        self.xf = np.asarray(xf, dtype=float)
        self.vf = np.asarray(vf, dtype=float)
        self.duration = float(duration)

    def sample(self, elapsed: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        T = self.duration
        t = float(np.clip(elapsed, 0.0, T))
        s = t / T
        h00 = 2*s**3 - 3*s**2 + 1
        h10 = s**3 - 2*s**2 + s
        h01 = -2*s**3 + 3*s**2
        h11 = s**3 - s**2
        x = h00*self.x0 + h10*T*self.v0 + h01*self.xf + h11*T*self.vf

        dh00 = (6*s**2 - 6*s) / T
        dh10 = 3*s**2 - 4*s + 1
        dh01 = (-6*s**2 + 6*s) / T
        dh11 = 3*s**2 - 2*s
        v = dh00*self.x0 + dh10*self.v0 + dh01*self.xf + dh11*self.vf

        ddh00 = (12*s - 6) / T**2
        ddh10 = (6*s - 4) / T
        ddh01 = (-12*s + 6) / T**2
        ddh11 = (6*s - 2) / T
        a = ddh00*self.x0 + ddh10*self.v0 + ddh01*self.xf + ddh11*self.vf
        return x, v, a


def compute_terminal_velocity_command(x, v, x_ref, v_ref) -> np.ndarray:
    """Single-position-feedback resolved-rate terminal-state controller."""
    cmd = (
        v_ref
        + config.TRAJECTORY_K_POS @ (x_ref - x)
        + config.TRAJECTORY_K_VEL @ (v_ref - v)
    )
    speed = np.linalg.norm(cmd)
    if speed > config.MAX_OBJ_VEL:
        cmd *= config.MAX_OBJ_VEL / speed
    return cmd


def skew(r: np.ndarray) -> np.ndarray:
    x, y, z = np.asarray(r, dtype=float)
    return np.array([[0.0, -z, y], [z, 0.0, -x], [-y, x, 0.0]])


def rigid_contact_twist_map(r_left: np.ndarray, r_right: np.ndarray) -> np.ndarray:
    """Map object twist [v,omega] to both full contact twists.

    v_contact = v_object + omega x r = v_object - skew(r) omega.
    Unlike the baseline 0.5*I map, both contacts receive the complete rigid
    object motion; velocities are constraints and are not divided by two.
    """
    def block(r):
        return np.block([[np.eye(3), -skew(r)], [np.zeros((3, 3)), np.eye(3)]])
    return np.vstack([block(r_left), block(r_right)])

