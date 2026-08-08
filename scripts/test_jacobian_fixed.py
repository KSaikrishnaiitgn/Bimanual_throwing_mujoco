"""
Quick test to verify the fixed dual-arm Jacobian works correctly
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


def main():
    print("Testing fixed dual-arm Jacobian...")

    # Load model
    model = mujoco.MjModel.from_xml_path(config.XML_FILE_PATH)
    data = mujoco.MjData(model)

    # Reset and forward
    mujoco.mj_resetData(model, data)
    mujoco.mj_forward(model, data)

    # Find sites
    left_ee_id, right_ee_id = find_end_effector_sites(model)
    print(f"Left EE: {left_ee_id}, Right EE: {right_ee_id}")

    # Find box
    box_body_id = -1
    for i in range(model.nbody):
        if "box" in model.body(i).name:
            box_body_id = i
            break
    print(f"Box body ID: {box_body_id}")

    # Create dual-arm Jacobian
    dual_jac = DualArmJacobian(model, 12)

    # Test 1: Compute stacked Jacobian
    print("\nTest 1: Stacked Jacobian shape")
    J_H = dual_jac.compute_stacked_jacobian(data, left_ee_id, right_ee_id)
    print(f"  J_H shape: {J_H.shape} (expected: (12, 12))")

    # Test 2: Compute grasp matrix
    print("\nTest 2: Grasp matrix")
    G_T = dual_jac.compute_grasp_matrix(data, box_body_id, left_ee_id, right_ee_id)
    print(f"  G^T shape: {G_T.shape} (expected: (12, 3))")

    # Test 3: Command a velocity
    print("\nTest 3: Compute joint velocities for +Y object motion")
    x_dot_o_star = np.array([0.0, 1.0, 0.0])  # 1 m/s in Y direction
    x_dot_ee = np.zeros(12)  # No additional EE velocity

    q_dot = dual_jac.compute_joint_velocities(
        data, x_dot_o_star, x_dot_ee,
        box_body_id, left_ee_id, right_ee_id
    )

    print(f"  Commanded object velocity: {x_dot_o_star}")
    print(f"  Computed joint velocities (12):")
    print(f"    Left arm:  {q_dot[:6]}")
    print(f"    Right arm: {q_dot[6:]}")

    # Verify
    ee_vel_desired = x_dot_ee + G_T @ x_dot_o_star
    ee_vel_achieved = J_H @ q_dot

    print(f"\n  Verification:")
    print(f"    Desired EE velocity (stacked): {ee_vel_desired[:3]} (left), {ee_vel_desired[6:9]} (right)")
    print(f"    Achieved EE velocity: {ee_vel_achieved[:3]} (left), {ee_vel_achieved[6:9]} (right)")

    error = np.linalg.norm(ee_vel_desired - ee_vel_achieved)
    print(f"    Error: {error:.6e}")

    if error < 1e-6:
        print("\n✅ Dual-arm Jacobian is working correctly!")
    else:
        print(f"\n❌ Dual-arm Jacobian has errors!")

    print("\nTest complete.")


if __name__ == "__main__":
    main()
