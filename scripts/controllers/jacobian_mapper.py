"""
Jacobian-based velocity mapper
Maps object-level velocities to joint velocities using Jacobians
"""
import numpy as np
import mujoco


class JacobianMapper:
    """Maps object velocities to joint velocities for dual-arm manipulation"""

    def __init__(self, model, num_actuators):
        """
        Initialize Jacobian mapper

        Args:
            model: MuJoCo model
            num_actuators: Total number of actuators
        """
        self.model = model
        self.num_actuators = num_actuators
        self.actuators_per_robot = num_actuators // 2

    def get_arm_dofs(self):
        """
        Get DOF indices for left and right arms

        Returns:
            left_dofs: List of DOF indices for left arm
            right_dofs: List of DOF indices for right arm
        """
        left_dofs = []
        right_dofs = []

        # Left arm (first half of actuators)
        for i in range(self.actuators_per_robot):
            joint_id = self.model.actuator_trnid[i, 0]
            dof_adr = self.model.jnt_dofadr[joint_id]
            left_dofs.append(dof_adr)

        # Right arm (second half of actuators)
        for i in range(self.actuators_per_robot, self.num_actuators):
            joint_id = self.model.actuator_trnid[i, 0]
            dof_adr = self.model.jnt_dofadr[joint_id]
            right_dofs.append(dof_adr)

        return left_dofs, right_dofs

    def compute_jacobians(self, data, left_ee_site_id, right_ee_site_id):
        """
        Compute Jacobians for both end-effectors

        Args:
            data: MuJoCo data
            left_ee_site_id: Site ID for left end-effector
            right_ee_site_id: Site ID for right end-effector

        Returns:
            jac_left: Left arm Jacobian (6 x nv)
            jac_right: Right arm Jacobian (6 x nv)
        """
        jac_left = np.zeros((6, self.model.nv))
        jac_right = np.zeros((6, self.model.nv))

        mujoco.mj_jacSite(self.model, data, jac_left[:3], jac_left[3:], left_ee_site_id)
        mujoco.mj_jacSite(self.model, data, jac_right[:3], jac_right[3:], right_ee_site_id)

        return jac_left, jac_right

    def object_velocity_to_joint_velocities_lifting(self, data, object_velocity,
                                                     left_ee_site_id, right_ee_site_id):
        """
        Map object velocity to joint velocities for lifting (z-direction focus)

        Args:
            data: MuJoCo data
            object_velocity: Desired object velocity (6×1)
            left_ee_site_id: Site ID for left end-effector
            right_ee_site_id: Site ID for right end-effector

        Returns:
            Joint velocities for both arms
        """
        try:
            # Get Jacobians
            jac_left, jac_right = self.compute_jacobians(data, left_ee_site_id, right_ee_site_id)

            # Get arm DOFs
            left_dofs, right_dofs = self.get_arm_dofs()

            # Extract relevant columns
            jac_left_arm = jac_left[:, left_dofs]
            jac_right_arm = jac_right[:, right_dofs]

            # Focus on z-direction for lifting
            J_z = np.zeros((2, len(left_dofs) + len(right_dofs)))
            J_z[0, :len(left_dofs)] = jac_left_arm[2, :]
            J_z[1, len(left_dofs):] = jac_right_arm[2, :]

            # Compute pseudoinverse
            J_z_pinv = np.linalg.pinv(J_z)

            # Compute joint velocities
            v_z = np.array([object_velocity[2], object_velocity[2]])
            joint_velocities = J_z_pinv @ v_z

            # Scale and limit
            joint_velocities *= 2.0
            max_joint_vel = 1.0  # rad/s
            joint_velocities = np.clip(joint_velocities, -max_joint_vel, max_joint_vel)

            return joint_velocities

        except Exception as e:
            print(f"Error in jacobian mapping: {e}")
            return np.zeros(self.num_actuators)

    def object_velocity_to_joint_velocities_throwing(self, data, object_velocity,
                                                      left_ee_site_id, right_ee_site_id,
                                                      max_vel=3.0):
        """
        Map object velocity to joint velocities for throwing (3D motion)

        Args:
            data: MuJoCo data
            object_velocity: Desired object velocity (3D)
            left_ee_site_id: Site ID for left end-effector
            right_ee_site_id: Site ID for right end-effector
            max_vel: Maximum joint velocity (rad/s)

        Returns:
            Joint velocities for both arms
        """
        try:
            # Get Jacobians
            jac_left, jac_right = self.compute_jacobians(data, left_ee_site_id, right_ee_site_id)

            # Get arm DOFs
            left_dofs, right_dofs = self.get_arm_dofs()

            # Extract linear part of Jacobians
            Jl = jac_left[:3, left_dofs]
            Jr = jac_right[:3, right_dofs]

            # Split velocity between hands
            v_half = 0.5 * object_velocity

            # Compute joint velocities
            qdot = np.zeros(self.num_actuators)
            qdot[:self.actuators_per_robot] = np.linalg.pinv(Jl) @ v_half
            qdot[self.actuators_per_robot:] = np.linalg.pinv(Jr) @ v_half

            return np.clip(qdot, -max_vel, max_vel)

        except Exception as e:
            print(f"Error in throwing jacobian mapping: {e}")
            return np.zeros(self.num_actuators)
