"""
Admittance Grasp Controller — Force Closure
Implements:
    ẋ = K_adm^-1 (F* - F_meas)

Both end-effectors squeeze toward the object along the grasp axis (the line
connecting the two EE sites) until each senses the desired grasp force.
At equilibrium, both hands apply equal-magnitude, opposite-direction forces
along the grasp axis — force closure — which holds the object stable
without imparting any net translational force from the squeeze itself.

This output is the ẋ_ee term in:
    q̇ = J_H† [ ẋ_ee + G^T ẋ_o* ]

i.e. it is ADDED to the object-translation term (G^T ẋ_o*), not blended
through the grasp matrix — because a balanced squeeze nets to zero force on
the object, so it doesn't interfere with the commanded object motion.
"""
import numpy as np


class AdmittanceGraspController:
    """
    Per-hand admittance control toward a target grasp (squeeze) force.

    ẋ_left  = K_adm^-1 (F* - F_meas_left)  *  (+n̂)   (push toward object)
    ẋ_right = K_adm^-1 (F* - F_meas_right) *  (-n̂)   (push toward object,
                                                        opposite direction)

    where n̂ is the unit vector from the left EE site to the right EE site
    (recomputed every step from live EE positions, so it self-adjusts as the
    arms move — no fixed grasp-axis assumption required).
    """

    def __init__(self, K_adm: np.ndarray, desired_force: float,
                 max_vel: float = 0.3):
        """
        Args:
            K_adm: 3x3 admittance gain matrix (N·s/m). Larger K_adm ->
                smaller velocity per unit force error (stiffer/slower
                response); smaller K_adm -> faster/softer response.
                NOTE: this is applied as K_adm^-1, matching the slide
                equation ẋ = K^-1(F* - F_meas) — don't confuse this with
                a stiffness in the impedance sense (higher K_adm here means
                a SLOWER response, since it's inverted).
            desired_force: F*, target squeeze force magnitude (N), applied
                symmetrically by both hands toward the object.
            max_vel: safety clamp on the resulting squeeze velocity (m/s).
        """
        self.K_adm_inv = np.linalg.inv(K_adm)
        self.F_star = float(desired_force)
        self.max_vel = max_vel

    def grasp_axis(self, left_ee_pos: np.ndarray, right_ee_pos: np.ndarray) -> np.ndarray:
        """Unit vector from left EE to right EE. Recomputed every call so it
        tracks the live arm configuration rather than assuming a fixed axis."""
        d = right_ee_pos - left_ee_pos
        norm = np.linalg.norm(d)
        if norm < 1e-6:
            # Degenerate — hands coincident; fall back to a safe default
            # (world Y here matches this setup's throw axis; adjust if your
            # robot's default grasp axis differs).
            return np.array([0.0, 1.0, 0.0])
        return d / norm

    def compute_ee_velocities(self,
                               left_ee_pos:  np.ndarray,  # (3,)
                               right_ee_pos: np.ndarray,  # (3,)
                               left_force_meas:  np.ndarray,  # (3,) measured force at left EE
                               right_force_meas: np.ndarray   # (3,) measured force at right EE
                               ) -> np.ndarray:
        """
        Returns ẋ_ee, a stacked 12D vector matching DualArmJacobian's
        expected layout:
            [left_linear(3), left_angular(3), right_linear(3), right_angular(3)]
        Angular components are zero (linear-only grasp control, per spec).

        F_meas is projected onto the grasp axis to get the scalar squeeze
        force each hand currently senses (positive = pushing toward the
        object along the axis), then the admittance law is applied along
        that same axis so the resulting velocity only ever pushes/pulls
        along the grasp direction — it will not inject motion
        perpendicular to the squeeze axis.
        """
        n_hat = self.grasp_axis(left_ee_pos, right_ee_pos)

        # Scalar force each hand senses along the grasp axis.
        # Left hand pushes in +n̂ toward the object -> its applied/sensed
        # squeeze force is measured along +n̂.
        F_meas_left  = np.dot(left_force_meas,  n_hat)
        # Right hand pushes in -n̂ toward the object -> measure along -n̂
        # so a positive value also means "pushing toward the object".
        F_meas_right = np.dot(right_force_meas, -n_hat)

        # ẋ = K_adm^-1 (F* - F_meas), applied as a scalar along the grasp axis
        left_force_error  = self.F_star - F_meas_left
        right_force_error = self.F_star - F_meas_right

        # K_adm_inv is 3x3; since we're working with a scalar force error
        # along a single axis, use the axis-aligned component of K_adm_inv.
        # (For an isotropic/diagonal K_adm with equal entries, this reduces
        # to a simple scalar divide — kept general here in case K_adm is
        # later made anisotropic.)
        left_speed  = (self.K_adm_inv @ (n_hat * left_force_error))
        right_speed = (self.K_adm_inv @ (-n_hat * right_force_error))

        left_vel  = left_speed
        right_vel = right_speed

        # Safety clamp on squeeze speed
        l_norm = np.linalg.norm(left_vel)
        if l_norm > self.max_vel:
            left_vel = left_vel / l_norm * self.max_vel
        r_norm = np.linalg.norm(right_vel)
        if r_norm > self.max_vel:
            right_vel = right_vel / r_norm * self.max_vel

        x_dot_ee = np.zeros(12)
        x_dot_ee[0:3] = left_vel     # left linear
        x_dot_ee[3:6] = 0.0          # left angular (unused)
        x_dot_ee[6:9] = right_vel    # right linear
        x_dot_ee[9:12] = 0.0         # right angular (unused)

        return x_dot_ee, F_meas_left, F_meas_right