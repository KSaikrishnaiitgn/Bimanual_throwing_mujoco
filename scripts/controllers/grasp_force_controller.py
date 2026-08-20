"""
Grasp Force Controller — Force-Closure Squeeze Grasp
Implements the admittance law:

    ẋ ← K⁻¹ (F⋆ − F_meas)

Two end-effectors squeeze the object from opposite sides along a grasp axis.
Target forces are equal and opposite (+f, −f) along that axis, so the net
force the squeeze applies to the object is ≈0 (force closure) — but each
contact individually presses inward with magnitude f, giving friction the
normal load it needs to resist gravity + the throw's inertial acceleration
without slipping.

At release, F⋆ is ramped to 0 over `ramp_steps` control cycles so the arms
relax and friction capacity decays to zero, letting the box separate
naturally — rather than being force-disconnected by disabling collision
geoms (which is not physical and can leave the box with whatever residual
squeeze-induced velocity it had).

This controller is deliberately independent of the throw-direction DS /
impedance controller: it only ever commands velocity along the grasp axis
(roughly orthogonal to the throw direction), so its output is meant to be
summed into `x_dot_ee` in DualArmJacobian.compute_joint_velocities, while
`x_dot_o_star` (from ThrowingDS + ThrowingImpedance) continues to drive the
throw-direction motion. The two do not fight each other.
"""
import numpy as np


class GraspForceController:
    def __init__(self, K_grasp: float, target_force: float,
                 ramp_steps: int = 10, max_vel: float = 0.3):
        """
        Args:
            K_grasp:      scalar admittance gain. ẋ = (F⋆ − F_meas) / K_grasp.
                          Larger K_grasp → softer / slower force response.
            target_force: nominal per-arm squeeze force magnitude f (N),
                          applied symmetrically (+f left, −f right along axis).
            ramp_steps:   number of control steps over which the target force
                          ramps from its current value down to 0 on release.
            max_vel:      safety clamp on commanded grasp-axis velocity (m/s).
        """
        self.K_inv = 1.0 / K_grasp
        self.nominal_force = target_force
        self.current_target_force = target_force
        self.ramp_steps = ramp_steps
        self.max_vel = max_vel

        self._ramp_counter = 0
        self._ramping = False

    # ──────────────────────────────────────────────────────────────────
    def reset(self, target_force: float = None):
        """Call when (re)entering the grasp / throw phase."""
        self.current_target_force = (self.nominal_force if target_force is None
                                      else target_force)
        self._ramp_counter = 0
        self._ramping = False

    def start_release_ramp(self):
        """Call once, exactly when release is triggered."""
        self._ramping = True
        self._ramp_counter = 0

    def is_ramp_complete(self) -> bool:
        return self._ramping and self._ramp_counter >= self.ramp_steps

    def _advance_force_ramp(self):
        if not self._ramping:
            return
        if self._ramp_counter < self.ramp_steps:
            frac = 1.0 - (self._ramp_counter + 1) / self.ramp_steps
            self.current_target_force = self.nominal_force * max(frac, 0.0)
            self._ramp_counter += 1
        else:
            self.current_target_force = 0.0

    # ──────────────────────────────────────────────────────────────────
    @staticmethod
    def compute_grasp_axis(left_ee_pos: np.ndarray, right_ee_pos: np.ndarray) -> np.ndarray:
        """Unit vector from left EE toward right EE ('inward' for the left arm)."""
        axis = right_ee_pos - left_ee_pos
        norm = np.linalg.norm(axis)
        if norm < 1e-9:
            return np.array([1.0, 0.0, 0.0])
        return axis / norm

    def compute_squeeze_velocities(self,
                                    left_ee_pos:  np.ndarray,
                                    right_ee_pos: np.ndarray,
                                    F_meas_left:  np.ndarray,
                                    F_meas_right: np.ndarray) -> np.ndarray:
        """
        ẋ ← K⁻¹ (F⋆ − F_meas), projected onto the grasp axis only.

        Args:
            left_ee_pos, right_ee_pos: world-frame EE positions (3,)
            F_meas_left, F_meas_right: world-frame contact force ON THE
                                        OBJECT from each arm (3,) — see
                                        ContactHandler.get_ee_contact_force

        Returns:
            x_dot_ee: (12,) stacked EE velocity command
                      [left_lin(3) left_ang(3) right_lin(3) right_ang(3)]
                      Angular components are always zero — this controller
                      only regulates linear squeeze along the grasp axis.
        """
        self._advance_force_ramp()

        axis = self.compute_grasp_axis(left_ee_pos, right_ee_pos)

        # Left arm target: +f along axis (toward the object / right arm).
        # Right arm target: −f along axis (toward the object / left arm).
        # Equal & opposite -> net squeeze force on object ≈ 0 (force closure).
        F_star_left  =  self.current_target_force * axis
        F_star_right = -self.current_target_force * axis

        # Only the axial component of measured force is force-controlled;
        # off-axis components are left to the throw DS / impedance path.
        F_meas_left_axis  = np.dot(F_meas_left,  axis) * axis
        F_meas_right_axis = np.dot(F_meas_right, axis) * axis

        v_left  = self.K_inv * (F_star_left  - F_meas_left_axis)
        v_right = self.K_inv * (F_star_right - F_meas_right_axis)

        v_left  = np.clip(v_left,  -self.max_vel, self.max_vel)
        v_right = np.clip(v_right, -self.max_vel, self.max_vel)

        x_dot_ee = np.zeros(12)
        x_dot_ee[0:3] = v_left
        x_dot_ee[6:9] = v_right
        return x_dot_ee

    def grasp_fully_released(self, F_meas_left: np.ndarray, F_meas_right: np.ndarray,
                              force_threshold: float = 0.5) -> bool:
        """
        True once the force ramp has finished AND measured contact force has
        actually decayed below `force_threshold` N on both arms — i.e.
        contact has physically broken, not just that the ramp counter expired.
        """
        return (self.is_ramp_complete() and
                np.linalg.norm(F_meas_left)  < force_threshold and
                np.linalg.norm(F_meas_right) < force_threshold)