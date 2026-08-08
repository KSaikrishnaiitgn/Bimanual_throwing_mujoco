"""
Debug script to compare Jacobian behavior between kshitij_lifting.py and throwing_modular_2.py

This script will:
1. Load the MuJoCo model
2. Set up the robots in throwing configuration
3. Compute Jacobians using both methods
4. Test what happens when we command positive Y velocity
5. Identify coordinate frame issues
"""
import os
os.environ["MUJOCO_GL"] = "glfw"

import mujoco
import numpy as np
from config import throwing_config as config
from controllers.dual_arm_jacobian import DualArmJacobian


def find_end_effector_sites(model):
    """Find end-effector site IDs"""
    left_ee_id = -1
    right_ee_id = -1
    for i in range(model.nsite):
        site_name = model.site(i).name
        if "left" in site_name.lower() and left_ee_id == -1:
            left_ee_id = i
        elif "right" in site_name.lower() and right_ee_id == -1:
            right_ee_id = i
    return left_ee_id, right_ee_id


def compute_jacobian_kshitij_method(model, data, left_ee_id, right_ee_id):
    """
    Compute Jacobian using kshitij_lifting.py method
    (Lines 673-691 of kshitij_lifting.py)
    """
    # Get Jacobians for both end-effectors
    jac_left = np.zeros((6, model.nv))
    jac_right = np.zeros((6, model.nv))
    mujoco.mj_jacSite(model, data, jac_left[:3], jac_left[3:], left_ee_id)
    mujoco.mj_jacSite(model, data, jac_right[:3], jac_right[3:], right_ee_id)

    # Find joint DOFs for each arm
    num_actuators = 12
    actuators_per_robot = 6

    left_arm_dofs = []
    right_arm_dofs = []
    for i in range(actuators_per_robot):
        joint_id = model.actuator_trnid[i, 0]
        dof_adr = model.jnt_dofadr[joint_id]
        left_arm_dofs.append(dof_adr)
    for i in range(actuators_per_robot, num_actuators):
        joint_id = model.actuator_trnid[i, 0]
        dof_adr = model.jnt_dofadr[joint_id]
        right_arm_dofs.append(dof_adr)

    # Extract relevant Jacobian columns
    jac_left_arm = jac_left[:, left_arm_dofs]
    jac_right_arm = jac_right[:, right_arm_dofs]

    return jac_left_arm, jac_right_arm, jac_left, jac_right


def compute_jacobian_modular2_method(model, data, left_ee_id, right_ee_id):
    """
    Compute Jacobian using throwing_modular_2.py method
    (dual_arm_jacobian.py)
    """
    dual_arm_jac = DualArmJacobian(model, 12)

    # Get stacked Jacobian
    J_H = dual_arm_jac.compute_stacked_jacobian(data, left_ee_id, right_ee_id)

    return J_H


def test_velocity_command(model, data, method_name, jac_left=None, jac_right=None, J_H=None):
    """
    Test what happens when we command a specific velocity
    """
    print(f"\n{'='*60}")
    print(f"Testing {method_name}")
    print(f"{'='*60}")

    # Test velocity: +1 m/s in Y direction
    test_velocity = np.array([0.0, 1.0, 0.0, 0.0, 0.0, 0.0])  # [x, y, z, rx, ry, rz]

    if J_H is not None:
        # Modular2 method: stacked Jacobian
        print(f"\nCommanded velocity: +Y direction (world frame)")
        print(f"  test_velocity = {test_velocity[:3]}")

        J_H_pinv = np.linalg.pinv(J_H)
        q_dot = J_H_pinv @ np.concatenate([test_velocity, test_velocity])

        print(f"\nJoint velocities (12 joints):")
        print(f"  Left arm:  {q_dot[:6]}")
        print(f"  Right arm: {q_dot[6:]}")

        # Verify: what velocity would these joint velocities produce?
        ee_vel_check = J_H @ q_dot
        print(f"\nResulting end-effector velocities (verification):")
        print(f"  Left EE:  {ee_vel_check[:3]}")
        print(f"  Right EE: {ee_vel_check[6:9]}")

    else:
        # Kshitij method: separate pseudoinverses
        print(f"\nCommanded velocity: +Y direction (world frame)")
        print(f"  test_velocity = {test_velocity[:3]}")

        qdot_left = np.linalg.pinv(jac_left) @ test_velocity
        qdot_right = np.linalg.pinv(jac_right) @ test_velocity

        print(f"\nJoint velocities:")
        print(f"  Left arm:  {qdot_left}")
        print(f"  Right arm: {qdot_right}")

        # Verify: what velocity would these joint velocities produce?
        ee_vel_left_check = jac_left @ qdot_left
        ee_vel_right_check = jac_right @ qdot_right
        print(f"\nResulting end-effector velocities (verification):")
        print(f"  Left EE:  {ee_vel_left_check[:3]}")
        print(f"  Right EE: {ee_vel_right_check[:3]}")


def compare_jacobians(jac_left_k, jac_right_k, J_H_m2):
    """Compare Jacobians from both methods"""
    print(f"\n{'='*60}")
    print(f"Jacobian Comparison")
    print(f"{'='*60}")

    print(f"\nKshitij method:")
    print(f"  jac_left shape:  {jac_left_k.shape}")
    print(f"  jac_right shape: {jac_right_k.shape}")

    print(f"\nModular2 method:")
    print(f"  J_H stacked shape: {J_H_m2.shape}")

    # Extract corresponding blocks from J_H
    # J_H should be (12, 12) for 6-DOF left + 6-DOF right
    if J_H_m2.shape[0] == 12:
        # Top 6 rows are left EE Jacobian
        J_H_left = J_H_m2[:6, :6]
        J_H_right = J_H_m2[6:, 6:]

        print(f"\nExtracted from J_H:")
        print(f"  Left block:  {J_H_left.shape}")
        print(f"  Right block: {J_H_right.shape}")

        # Compare
        diff_left = np.linalg.norm(jac_left_k - J_H_left)
        diff_right = np.linalg.norm(jac_right_k - J_H_right)

        print(f"\nDifference (Frobenius norm):")
        print(f"  Left Jacobian:  {diff_left:.6f}")
        print(f"  Right Jacobian: {diff_right:.6f}")

        if diff_left < 1e-6 and diff_right < 1e-6:
            print(f"\n✅ Jacobians MATCH - no structural issue")
        else:
            print(f"\n⚠️ Jacobians DIFFER - potential coordinate frame issue")

            # Show sample differences
            print(f"\nSample left Jacobian difference (first 3x3 block):")
            print(jac_left_k[:3, :3] - J_H_left[:3, :3])


def main():
    """Main debug function"""
    print("="*60)
    print("JACOBIAN COMPARISON DEBUG SCRIPT")
    print("="*60)

    # Load model
    model = mujoco.MjModel.from_xml_path(config.XML_FILE_PATH)
    data = mujoco.MjData(model)

    # Reset and forward
    mujoco.mj_resetData(model, data)
    mujoco.mj_forward(model, data)

    # Find end-effector sites
    left_ee_id, right_ee_id = find_end_effector_sites(model)
    print(f"\nEnd-effector sites found:")
    print(f"  Left EE ID:  {left_ee_id}")
    print(f"  Right EE ID: {right_ee_id}")

    if left_ee_id == -1 or right_ee_id == -1:
        print("ERROR: Could not find end-effector sites!")
        return

    # Set robots to pre-throw configuration (second target positions)
    joint_positions = np.concatenate([
        config.SECOND_TARGET_POSITIONS_LEFT,
        config.SECOND_TARGET_POSITIONS_RIGHT
    ])

    for i in range(12):
        joint_id = model.actuator_trnid[i, 0]
        data.qpos[model.jnt_qposadr[joint_id]] = joint_positions[i]

    mujoco.mj_forward(model, data)
    print(f"\nRobots positioned at pre-throw configuration")

    # Compute Jacobians using both methods
    print(f"\n{'='*60}")
    print(f"Computing Jacobians...")
    print(f"{'='*60}")

    jac_left_k, jac_right_k, jac_left_full, jac_right_full = \
        compute_jacobian_kshitij_method(model, data, left_ee_id, right_ee_id)

    J_H_m2 = compute_jacobian_modular2_method(model, data, left_ee_id, right_ee_id)

    # Compare Jacobians
    compare_jacobians(jac_left_k, jac_right_k, J_H_m2)

    # Test velocity commands
    test_velocity_command(
        model, data, "Kshitij Method",
        jac_left=jac_left_k, jac_right=jac_right_k
    )

    test_velocity_command(
        model, data, "Modular2 Method",
        J_H=J_H_m2
    )

    # Additional coordinate frame checks
    print(f"\n{'='*60}")
    print(f"Coordinate Frame Checks")
    print(f"{'='*60}")

    print(f"\nEnd-effector world positions:")
    left_pos = data.site_xpos[left_ee_id]
    right_pos = data.site_xpos[right_ee_id]
    print(f"  Left EE:  {left_pos}")
    print(f"  Right EE: {right_pos}")

    print(f"\nEnd-effector world velocities:")
    left_vel = data.site_xvelp[left_ee_id]
    right_vel = data.site_xvelp[right_ee_id]
    print(f"  Left EE:  {left_vel}")
    print(f"  Right EE: {right_vel}")

    # Check box
    box_body_id = -1
    for i in range(model.nbody):
        if "box" in model.body(i).name:
            box_body_id = i
            break

    if box_body_id != -1:
        print(f"\nBox state:")
        print(f"  Position: {data.xpos[box_body_id]}")
        print(f"  Velocity: {data.cvel[box_body_id, :3]}")

    print(f"\n{'='*60}")
    print(f"Debug Complete")
    print(f"{'='*60}")
    print(f"\nIf commanded +Y velocity produces -Y motion in verification,")
    print(f"then there's a sign flip in the Jacobian or coordinate frame.")


if __name__ == "__main__":
    main()
