"""
Dynamical System (DS) controller for throwing
Implements a second-order spring-damper system for smooth velocity convergence
"""
import numpy as np


class DynamicalSystem:
    """
    Second-order DS for throwing motion
    Equation: ẍ = -K(x - x*) - B(ẋ - ẋ*)
    """

    def __init__(self, K_gains, B_gains, max_velocity):
        """
        Initialize the dynamical system

        Args:
            K_gains: Position stiffness matrix (3x3 diagonal)
            B_gains: Velocity damping matrix (3x3 diagonal)
            max_velocity: Maximum allowed object velocity (m/s)
        """
        self.K = K_gains
        self.B = B_gains
        self.max_velocity = max_velocity
        self.integrated_velocity = None

    def reset(self):
        """Reset the integrated velocity state"""
        self.integrated_velocity = None

    def compute_desired_velocity(self, current_pos, current_vel, target_pos, target_vel, dt):
        """
        Compute desired object velocity using DS

        Args:
            current_pos: Current object position (3D)
            current_vel: Current object velocity (3D)
            target_pos: Target release position (3D)
            target_vel: Target release velocity (3D)
            dt: Time step

        Returns:
            Desired object velocity (3D)
        """
        # Initialize integrated velocity on first call
        if self.integrated_velocity is None:
            self.integrated_velocity = current_vel.copy()

        # Compute DS acceleration
        position_error = current_pos - target_pos
        velocity_error = current_vel - target_vel

        acceleration = (
            -self.K @ position_error
            -self.B @ velocity_error
        )

        # Integrate acceleration to get velocity
        self.integrated_velocity += acceleration * dt

        # Apply safety limits
        self.integrated_velocity = np.clip(
            self.integrated_velocity,
            -self.max_velocity,
            self.max_velocity
        )

        return self.integrated_velocity.copy()

    def get_convergence_error(self, current_vel, target_vel):
        """
        Compute velocity convergence error

        Args:
            current_vel: Current velocity (3D)
            target_vel: Target velocity (3D)

        Returns:
            Magnitude of velocity error
        """
        return np.linalg.norm(current_vel - target_vel)
