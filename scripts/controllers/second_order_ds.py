"""
Second-Order Dynamical System for Throwing
Implements the exact formulation from the mathematical specification:
    ẍ_des = -K_ds(x_o - x_rel) - B_ds(ẋ_o - ẋ_o^target)

where:
    x_o = current object position
    x_rel = release position (target)
    ẋ_o = current object velocity
    ẋ_o^target = desired release velocity (computed from ballistics)
    ẍ_des = desired object acceleration
"""
import numpy as np


class SecondOrderDS:
    """
    Second-order dynamical system for throwing motion.

    This computes the desired acceleration, not the velocity.
    The acceleration is then used in the impedance controller.
    """

    def __init__(self, K_ds, B_ds):
        """
        Initialize the second-order DS

        Args:
            K_ds: Position stiffness matrix (3x3 diagonal)
            B_ds: Velocity damping matrix (3x3 diagonal)
        """
        self.K_ds = K_ds
        self.B_ds = B_ds

    def compute_desired_acceleration(self, x_o, x_dot_o, x_rel, x_dot_o_target):
        """
        Compute desired object acceleration using second-order DS

        Mathematical formulation:
            ẍ_des = -K_ds(x_o - x_rel) - B_ds(ẋ_o - ẋ_o^target)

        Args:
            x_o: Current object position (3D vector)
            x_dot_o: Current object velocity (3D vector)
            x_rel: Release position / target position (3D vector)
            x_dot_o_target: Target release velocity (3D vector)

        Returns:
            x_ddot_des: Desired object acceleration (3D vector)
        """
        # Position error: (x_o - x_rel)
        position_error = x_o - x_rel

        # Velocity error: (ẋ_o - ẋ_o^target)
        velocity_error = x_dot_o - x_dot_o_target

        # Desired acceleration: ẍ_des = -K_ds(x_o - x_rel) - B_ds(ẋ_o - ẋ_o^target)
        x_ddot_des = -self.K_ds @ position_error - self.B_ds @ velocity_error

        return x_ddot_des
