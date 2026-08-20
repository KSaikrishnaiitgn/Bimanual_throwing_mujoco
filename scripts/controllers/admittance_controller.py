"""
Admittance controller for object manipulation
Computes desired object velocity based on contact forces/wrenches
"""
import numpy as np


class AdmittanceController:
    """
    Admittance control for compliant object manipulation
    Equation: ẋₒ* = ẋₒ + D⁻¹[W - W* - K(xₒ* - xₒ)]
    """

    def __init__(self, K_matrix, D_matrix, max_linear_vel, max_angular_vel,
                 upward_force_base, upward_force_scale):
        """
        Initialize admittance controller

        Args:
            K_matrix: Stiffness matrix (6x6 diagonal)
            D_matrix: Damping matrix (6x6 diagonal)
            max_linear_vel: Maximum linear velocity (m/s)
            max_angular_vel: Maximum angular velocity (rad/s)
            upward_force_base: Base upward force factor
            upward_force_scale: Additional upward force scaling
        """
        self.K = K_matrix
        self.D = D_matrix
        self.D_inv = np.linalg.inv(D_matrix)
        self.max_linear_vel = max_linear_vel
        self.max_angular_vel = max_angular_vel
        self.upward_force_base = upward_force_base
        self.upward_force_scale = upward_force_scale

    def compute_object_velocity(self, model, data, left_wrench, right_wrench,
                                object_body_id, desired_height):
        """
        Compute desired object velocity using admittance control

        Args:
            model: MuJoCo model
            data: MuJoCo data
            left_wrench: Wrench from left end-effector (6×1)
            right_wrench: Wrench from right end-effector (6×1)
            object_body_id: Body ID of the object
            desired_height: Target height for the object (meters)

        Returns:
            object_velocity: Desired object velocity (6×1)
            W: Combined wrench
            W_desired: Desired wrench
            pose_error: Pose error (6×1)
        """
        if object_body_id == -1:
            return np.zeros(6), np.zeros(6), np.zeros(6), np.zeros(6)

        # Get current object pose
        current_obj_pos = data.xpos[object_body_id].copy()
        current_obj_quat = data.xquat[object_body_id].copy()

        # Define desired object pose
        desired_obj_pos = current_obj_pos.copy()
        desired_obj_pos[2] = desired_height

        # Compute position error
        pos_error = desired_obj_pos - current_obj_pos
        height_error = desired_height - current_obj_pos[2]
        height_error_percentage = min(100, max(0, (height_error / (desired_height - 0.1)) * 100))

        # Get current object velocity
        current_obj_vel = np.zeros(6)
        if hasattr(data, 'cvel'):
            current_obj_vel[:3] = data.cvel[object_body_id, :3]  # Linear velocity
            current_obj_vel[3:] = data.cvel[object_body_id, 3:6]  # Angular velocity

        # Combine wrenches from both end-effectors
        W = left_wrench + right_wrench

        # Compute desired wrench (counteract gravity + lifting force)
        object_mass = model.body_mass[object_body_id]
        W_desired = np.zeros(6)

        weight_force = object_mass * 9.81
        upward_force_factor = self.upward_force_base + self.upward_force_scale * (height_error_percentage / 100)
        W_desired[2] = weight_force * 2.0 + upward_force_factor

        # Compute pose error vector
        pose_error = np.zeros(6)
        pose_error[:3] = pos_error

        # Admittance control law
        stiffness_term = self.K @ pose_error
        wrench_term = W - W_desired
        object_velocity = current_obj_vel + self.D_inv @ (wrench_term - stiffness_term)

        # Apply velocity limits
        object_velocity[:3] = np.clip(object_velocity[:3], -self.max_linear_vel, self.max_linear_vel)
        object_velocity[3:] = np.clip(object_velocity[3:], -self.max_angular_vel, self.max_angular_vel)

        # Ensure minimum upward velocity when lifting
        if height_error > 0.01:
            min_upward_vel = 0.2 * (height_error_percentage / 100 + 0.5)
            object_velocity[2] = max(object_velocity[2], min_upward_vel)
        else:
            object_velocity[2] = min(object_velocity[2], 0.05)

        return object_velocity, W, W_desired, pose_error
