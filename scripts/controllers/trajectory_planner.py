"""
Trajectory planner for ballistic throwing
Computes release velocity for projectile motion
"""
import numpy as np


class TrajectoryPlanner:
    """Plans ballistic trajectories for throwing"""

    def __init__(self, gravity=9.81):
        """
        Initialize trajectory planner

        Args:
            gravity: Gravitational acceleration (m/s^2)
        """
        self.gravity = gravity

    def compute_release_velocity(self, release_pos, landing_pos, launch_angle):
        """
        Compute 3D release velocity for ballistic throw

        Args:
            release_pos: Release position (x, y, z) in world frame
            landing_pos: Landing position (x, y, z) in world frame
            launch_angle: Launch angle in radians

        Returns:
            Release velocity vector (3D)

        Raises:
            ValueError: If trajectory is not feasible
        """
        # Compute horizontal and vertical displacements
        delta_p = landing_pos - release_pos
        delta_xy = delta_p[:2]
        R = np.linalg.norm(delta_xy)  # Horizontal distance
        delta_z = delta_p[2]  # Vertical displacement

        if R < 1e-6:
            raise ValueError("Horizontal distance R ~ 0, cannot define throw direction.")

        # Compute required release speed
        cos_t = np.cos(launch_angle)
        sin_t = np.sin(launch_angle)
        denom = 2.0 * (cos_t**2) * (R * np.tan(launch_angle) - delta_z)

        if denom <= 0:
            raise ValueError("Invalid θ / landing point: need R*tan(theta) > Δz.")

        v_rel_sq = self.gravity * (R**2) / denom
        v_rel = np.sqrt(v_rel_sq)

        # Decompose into horizontal and vertical components
        e_h = delta_xy / R  # Horizontal direction unit vector
        v_horizontal = v_rel * cos_t * np.array([e_h[0], e_h[1], 0.0])
        v_vertical = v_rel * sin_t * np.array([0.0, 0.0, 1.0])

        return v_horizontal + v_vertical

    def predict_landing_point(self, release_pos, release_vel):
        """
        Predict where the object will land given release conditions

        Args:
            release_pos: Release position (x, y, z)
            release_vel: Release velocity (vx, vy, vz)

        Returns:
            Predicted landing position (x, y, z)
        """
        vx, vy, vz = release_vel
        x0, y0, z0 = release_pos

        # Time to hit ground (z = 0), solving: z0 + vz*t - 0.5*g*t^2 = 0
        if vz**2 + 2*self.gravity*z0 < 0:
            # Object won't reach ground
            return None

        t_land = (vz + np.sqrt(vz**2 + 2*self.gravity*z0)) / self.gravity

        # Compute landing position
        x_land = x0 + vx * t_land
        y_land = y0 + vy * t_land
        z_land = 0.0

        return np.array([x_land, y_land, z_land])

    def compute_flight_time(self, release_pos, release_vel):
        """
        Compute time of flight until object hits ground

        Args:
            release_pos: Release position (x, y, z)
            release_vel: Release velocity (vx, vy, vz)

        Returns:
            Time of flight (seconds)
        """
        vz = release_vel[2]
        z0 = release_pos[2]

        if vz**2 + 2*self.gravity*z0 < 0:
            return None

        t_flight = (vz + np.sqrt(vz**2 + 2*self.gravity*z0)) / self.gravity
        return t_flight
