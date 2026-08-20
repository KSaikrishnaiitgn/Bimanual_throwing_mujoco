"""
Dual-Arm Jacobian Mapping
Implements the exact formulation from the mathematical specification:

    q̇ = J_H† [ẋ + G^T · ẋ_o*]

where:
    q̇ = joint velocities
    J_H = stacked Jacobian for both arms
    ẋ = end-effector velocities (stacked for both arms)
    G = grasp matrix (maps object velocity to end-effector velocities)
    ẋ_o* = commanded object velocity
    † = pseudoinverse
"""
import numpy as np
import mujoco


class DualArmJacobian:
    """
    Dual-arm Jacobian mapping with grasp matrix integration
    """

    def __init__(self, model, num_actuators):
        """
        Initialize dual-arm Jacobian mapper

        Args:
            model: MuJoCo model
            num_actuators: Total number of actuators (both arms)
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
        Compute end-effector Jacobians for both arms

        Args:
            data: MuJoCo data
            left_ee_site_id: Site ID for left end-effector
            right_ee_site_id: Site ID for right end-effector

        Returns:
            J_left: Left arm Jacobian (6 x n_left) - full 6-DOF
            J_right: Right arm Jacobian (6 x n_right) - full 6-DOF
        """
        # Full Jacobians (6 x nv)
        jac_left_full = np.zeros((6, self.model.nv))
        jac_right_full = np.zeros((6, self.model.nv))

        mujoco.mj_jacSite(self.model, data, jac_left_full[:3], jac_left_full[3:], left_ee_site_id)
        mujoco.mj_jacSite(self.model, data, jac_right_full[:3], jac_right_full[3:], right_ee_site_id)

        # Get arm DOFs
        left_dofs, right_dofs = self.get_arm_dofs()

        # Extract full 6-DOF Jacobian for each arm
        J_left = jac_left_full[:, left_dofs]  # 6 x n_left
        J_right = jac_right_full[:, right_dofs]  # 6 x n_right

        return J_left, J_right

    def compute_stacked_jacobian(self, data, left_ee_site_id, right_ee_site_id):
        """
        Compute stacked Jacobian J_H for both arms

        Returns:
            J_H: Stacked Jacobian (12 x 12) for dual-arm system
                 [J_left   0    ]
                 [  0    J_right]
        """
        J_left, J_right = self.compute_jacobians(data, left_ee_site_id, right_ee_site_id)

        # Create stacked Jacobian (12 x 12)
        # Top 6 rows: left arm (6-DOF), bottom 6 rows: right arm (6-DOF)
        n_left = J_left.shape[1]
        n_right = J_right.shape[1]

        J_H = np.zeros((12, n_left + n_right))
        J_H[:6, :n_left] = J_left
        J_H[6:, n_left:] = J_right

        return J_H

    def compute_grasp_matrix(self, data, object_body_id, left_ee_site_id, right_ee_site_id):
        """
        Compute grasp matrix G that maps object velocity to end-effector velocities

        For a parallel grasp (simplified):
            G^T = [I_3  0  ]  (left hand linear follows object, no rotation)
                  [0    0  ]
                  [I_3  0  ]  (right hand linear follows object, no rotation)
                  [0    0  ]

        Args:
            data: MuJoCo data
            object_body_id: Body ID of the object
            left_ee_site_id: Site ID for left end-effector
            right_ee_site_id: Site ID for right end-effector

        Returns:
            G_transpose: Grasp matrix transpose (12 x 3)
        """
        # Grasp matrix: both hands' LINEAR velocities follow object
        # G^T maps object velocity (3D) to stacked hand velocities (12D = 6D left + 6D right)
        G_transpose = np.zeros((12, 3))
        G_transpose[:3, :] = np.eye(3)  # Left hand linear follows object
        # Rows 3:6 are zero (left hand rotational - no constraint)
        G_transpose[6:9, :] = np.eye(3)  # Right hand linear follows object
        # Rows 9:12 are zero (right hand rotational - no constraint)

        return G_transpose

    def compute_joint_velocities(self, data, x_dot_o_star, x_dot_ee,
                                  object_body_id, left_ee_site_id, right_ee_site_id):
        """
        Compute joint velocities using dual-arm formulation:
            q̇ = J_H† [ẋ + G^T · ẋ_o*]

        Args:
            data: MuJoCo data
            x_dot_o_star: Commanded object velocity (3D)
            x_dot_ee: Additional end-effector velocities (12D), usually zeros for throwing
            object_body_id: Body ID of the object
            left_ee_site_id: Site ID for left end-effector
            right_ee_site_id: Site ID for right end-effector

        Returns:
            q_dot: Joint velocities for both arms (12D)
        """
        try:
            # Compute stacked Jacobian J_H
            J_H = self.compute_stacked_jacobian(data, left_ee_site_id, right_ee_site_id)

            # Compute grasp matrix transpose G^T
            G_T = self.compute_grasp_matrix(data, object_body_id, left_ee_site_id, right_ee_site_id)

            # Compute desired end-effector velocity: ẋ + G^T · ẋ_o*
            x_dot_desired = x_dot_ee + G_T @ x_dot_o_star

            # Damped least-squares pseudoinverse: J^T (J J^T + λI)^{-1}
            # Avoids blow-up near kinematic singularities.
            lambda_reg = 0.01
            J_H_pinv = J_H.T @ np.linalg.inv(J_H @ J_H.T + lambda_reg * np.eye(12))
            q_dot = J_H_pinv @ x_dot_desired

            return q_dot

        except Exception as e:
            print(f"Error in dual-arm Jacobian mapping: {e}")
            return np.zeros(self.num_actuators)
