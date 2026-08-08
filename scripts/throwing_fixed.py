import os
from pyexpat import model
os.environ["MUJOCO_GL"] = "glfw"  # Must be before mujoco import
import mujoco
import numpy as np
import math  # if not already imported
from mujoco import viewer
import time

G = 9.81
LANDING_POINT = np.array([0.0, -2.5, 0.0])   # throw forward along negative Y axis
THROW_ANGLE = np.deg2rad(60.0)  # Higher angle for more visible arc
VELOCITY_SCALE = 1.4  # Scale the computed velocity to match target distance

# ================= DS GAINS =================
K_DS = np.diag([80.0, 80.0, 80.0])   # position stiffness
B_DS = np.diag([20.0, 20.0, 20.0])   # velocity damping
MAX_OBJ_VEL = 6.0                    # safety limit
# =================================================



# Import the controller class
from utils.mujoco_velocity_controller.samriddhi_dual_arm_velocity_controller import VelocityControllerGC

def compute_release_velocity(prel, pland, theta, g=G):
    """
    Compute 3D release velocity for a ballistic throw from prel to pland.
    prel, pland: 3D positions (x, y, z) in world frame.
    theta: launch angle (radians).
    """
    delta_p = pland - prel
    delta_xy = delta_p[:2]
    R = np.linalg.norm(delta_xy)
    delta_z = delta_p[2]

    if R < 1e-6:
        raise ValueError("Horizontal distance R ~ 0, cannot define throw direction.")

    cos_t = np.cos(theta)
    sin_t = np.sin(theta)
    denom = 2.0 * (cos_t**2) * (R * np.tan(theta) - delta_z)

    if denom <= 0:
        raise ValueError("Invalid θ / landing point: need R*tan(theta) > Δz.")

    v_rel_sq = g * (R**2) / denom
    v_rel = np.sqrt(v_rel_sq)

    e_h = delta_xy / R
    v_horizontal = v_rel * cos_t * np.array([e_h[0], e_h[1], 0.0])
    v_vertical   = v_rel * sin_t * np.array([0.0, 0.0, 1.0])

    return v_horizontal + v_vertical

def get_contact_wrenches(model, data, control_state):
    """
    Get wrenches (force/torque) at contact points between end-effectors and box.
    Prints the wrench values when contact is detected.
    """
    # Find box body ID

    box_body_id = control_state.get("box_body_id", -1)

    if box_body_id == -1:
        print(""
        " box_body_id not set in control_state")
        return np.zeros(6), np.zeros(6), False

    # Get all geom IDs for the box
    box_geom_ids = []
    for i in range(model.ngeom):
        if model.geom_bodyid[i] == box_body_id:
            box_geom_ids.append(i)

    # Initialize wrenches
    left_wrench = np.zeros(6)
    right_wrench = np.zeros(6)

    # Check all contacts
    contact_detected = False
    for i in range(data.ncon):
        contact = data.contact[i]

        # Check if this contact involves the box
        if contact.geom1 in box_geom_ids or contact.geom2 in box_geom_ids:
            # Get contact force
            contact_force = np.zeros(6)
            mujoco.mj_contactForce(model, data, i, contact_force)

            # Determine which end effector is involved
            other_geom = contact.geom2 if contact.geom1 in box_geom_ids else contact.geom1
            other_body = model.geom_bodyid[other_geom]
            body_name = model.body(other_body).name

            # Compute torque around box center
            contact_pos = contact.pos.copy()
            box_pos = data.xpos[box_body_id].copy()
            lever_arm = contact_pos - box_pos
            torque = np.cross(lever_arm, contact_force[:3])

            # Add to appropriate wrench
            if "left" in body_name.lower():
                left_wrench[:3] += contact_force[:3]
                left_wrench[3:] += torque
                contact_detected = True
            elif "right" in body_name.lower():
                right_wrench[:3] += contact_force[:3]
                right_wrench[3:] += torque
                contact_detected = True

    return left_wrench, right_wrench, contact_detected

def compute_object_admittance_control(model, data, left_wrench, right_wrench,control_state, desired_height=0.5):
    """
    Compute object admittance control using the equation:
    ẋₒ* = ẋₒ + D⁻¹[W - W* - K(xₒ* - xₒ)]

    Args:
        model: MuJoCo model
        data: MuJoCo data
        left_wrench: Wrench (force/torque) from left end-effector (6×1 vector)
        right_wrench: Wrench (force/torque) from right end-effector (6×1 vector)
        desired_height: Desired height for the object (in meters)

    Returns:
        object_velocity: Computed desired object velocity (6×1 vector)
    """
    # Find box body ID

    box_body_id = control_state.get("box_body_id", -1)

    if box_body_id == -1:
        return np.zeros(6),np.zeros(6),np.zeros(6),np.zeros(6)

    # 1. Get current object pose
    current_obj_pos = data.xpos[box_body_id].copy()
    current_obj_quat = data.xquat[box_body_id].copy()

    # 2. Define desired object pose (same position but higher z)
    desired_obj_pos = current_obj_pos.copy()
    desired_obj_pos[2] = desired_height  # Set desired height
    desired_obj_quat = current_obj_quat.copy()  # Keep same orientation

    # 3. Compute position error
    pos_error = desired_obj_pos - current_obj_pos

    # Calculate distance to target height
    height_error = desired_height - current_obj_pos[2]
    height_error_percentage = min(100, max(0, (height_error / (desired_height - 0.1)) * 100))

    current_obj_vel = np.zeros(6)
    if hasattr(data, 'cvel'):
        # Linear velocity is in the first 3 elements, angular in the last 3
        current_obj_vel[:3] = data.cvel[box_body_id, :3]
        current_obj_vel[3:] = data.cvel[box_body_id, 3:6]
    else:
        # If cvel not available, estimate from qvel
        # This is a simplified approximation
        current_obj_vel = np.zeros(6)

    # 5. Combine wrenches from both end-effectors
    W = left_wrench + right_wrench

    # 6. Define desired wrench (W*)
    # For lifting, we want to counteract gravity plus add upward force
    object_mass = model.body_mass[box_body_id]
    W_desired = np.zeros(6)

    # Calculate weight force
    weight_force = object_mass * 9.81  # N

    # Scale the upward force based on remaining distance to target
    # More force when far from target, less as we approach
    # Significantly increase the upward force factor to overcome weight
    upward_force_factor = 20.0 + 15.0 * (height_error_percentage / 100)

    # Set desired wrench to counteract gravity plus additional upward force
    # Multiply weight by a safety factor to ensure sufficient lifting force
    W_desired[2] = weight_force * 2.0 + upward_force_factor  # 2x gravity + additional force

    # 7. Define stiffness (K) and damping (D) matrices
    # These are diagonal matrices for simplicity
    # Increase stiffness for more aggressive position control
    K = np.diag([200.0, 200.0, 400.0, 20.0, 20.0, 20.0])  # Stiffness (increased for z-axis)
    D = np.diag([20.0, 20.0, 10.0, 3.0, 3.0, 3.0])        # Damping (decreased for faster response)

    # 8. Compute pose error vector (6×1)
    pose_error = np.zeros(6)
    pose_error[:3] = pos_error


    # 9. Compute the admittance control law
    # ẋₒ* = ẋₒ + D⁻¹[W - W* - K(xₒ* - xₒ)]
    D_inv = np.linalg.inv(D)
    stiffness_term = K @ pose_error
    wrench_term = W - W_desired

    # Complete admittance control law
    object_velocity = current_obj_vel + D_inv @ (wrench_term - stiffness_term)

    # 10. Apply velocity limits for safety
    max_lin_vel = 0.5  # m/s (increased for faster movement)
    max_ang_vel = 0.5  # rad/s
    object_velocity[:3] = np.clip(object_velocity[:3], -max_lin_vel, max_lin_vel)
    object_velocity[3:] = np.clip(object_velocity[3:], -max_ang_vel, max_ang_vel)

    # For lifting, ensure we have a minimum upward velocity if not at desired height
    if height_error > 0.01:  # If we need to move up
        # Scale minimum velocity based on distance to target
        min_upward_vel = 0.2 * (height_error_percentage / 100 + 0.5)  # At least 10-20 cm/s upward
        object_velocity[2] = max(object_velocity[2], min_upward_vel)
    else:
        # Near target, slow down to avoid overshooting
        object_velocity[2] = min(object_velocity[2], 0.05)

    return object_velocity, W, W_desired, pose_error

def compute_throwing_ds(model, data, control_state, dt):
    """
    Second-order object-level DS for throwing

    FIXED: Corrected velocity indexing and initialization
    """

    box_body_id = control_state["box_body_id"]

    # Current object position
    x_o = data.xpos[box_body_id].copy()

    # Current object linear velocity
    # FIX #1: Changed from [3:6] (angular) to [:3] (linear)
    if hasattr(data, "cvel"):
        xdot_o = data.cvel[box_body_id, :3].copy()  # ✅ FIXED: Linear velocity
    else:
        xdot_o = np.zeros(3)

    # DS attractors
    x_rel = control_state["throw_release_pos"]
    xdot_rel = control_state["throw_release_velocity"]

    # DS acceleration
    xddot = (
        -K_DS @ (x_o - x_rel)
        -B_DS @ (xdot_o - xdot_rel)
    )

    # Integrate acceleration → velocity
    # FIX #2: Initialize with current velocity for smooth transition
    if "ds_obj_vel" not in control_state:
        control_state["ds_obj_vel"] = xdot_o.copy()  # ✅ FIXED: Use current velocity

    control_state["ds_obj_vel"] += xddot * dt

    # Safety clamp
    control_state["ds_obj_vel"] = np.clip(
        control_state["ds_obj_vel"],
        -MAX_OBJ_VEL,
        MAX_OBJ_VEL
    )

    return control_state["ds_obj_vel"]


def object_velocity_to_joint_velocities(model, data, object_velocity, control_state):
    """
    Convert object velocity to joint velocities for dual-arm manipulation.

    Args:
        model: MuJoCo model
        data: MuJoCo data
        object_velocity: Desired object velocity (6×1 vector)

    Returns:
        joint_velocities: Joint velocities for both arms
    """
    try:
        # Find box body ID
        box_body_id = control_state.get("box_body_id", -1)

        if box_body_id == -1:
            print("⚠️ Could not find box body")
            return np.zeros(12)  # Assuming 6 DOFs per arm

        # Find end-effector site IDs - search more broadly for site names
        left_ee_site_id = -1
        right_ee_site_id = -1

        # If specific end-effector sites not found, use any left/right sites
        if left_ee_site_id == -1 or right_ee_site_id == -1:
            for i in range(model.nsite):
                site_name = model.site(i).name
                if "left" in site_name.lower() and left_ee_site_id == -1:
                    left_ee_site_id = i
                elif "right" in site_name.lower() and right_ee_site_id == -1:
                    right_ee_site_id = i

        if left_ee_site_id == -1 or right_ee_site_id == -1:
            return np.zeros(12)  # Assuming 6 DOFs per arm

        # Find joint IDs for each arm
        left_arm_dofs = []
        right_arm_dofs = []

        # Assuming the first half of actuators belong to the left arm and the second half to the right arm
        num_actuators = model.nu
        actuators_per_robot = num_actuators // 2

        # Get DOFs for left arm
        for i in range(actuators_per_robot):
            joint_id = model.actuator_trnid[i, 0]
            dof_adr = model.jnt_dofadr[joint_id]
            left_arm_dofs.append(dof_adr)

        # Get DOFs for right arm
        for i in range(actuators_per_robot, num_actuators):
            joint_id = model.actuator_trnid[i, 0]
            dof_adr = model.jnt_dofadr[joint_id]
            right_arm_dofs.append(dof_adr)

        # Get Jacobians for both end-effectors
        jac_left = np.zeros((6, model.nv))
        jac_right = np.zeros((6, model.nv))

        mujoco.mj_jacSite(model, data, jac_left[:3], jac_left[3:], left_ee_site_id)
        mujoco.mj_jacSite(model, data, jac_right[:3], jac_right[3:], right_ee_site_id)

        # Number of DOFs per arm
        n_dofs_per_arm = len(left_arm_dofs)
        total_dofs = n_dofs_per_arm * 2

        # Extract relevant columns from Jacobians
        jac_left_arm = jac_left[:, left_arm_dofs]
        jac_right_arm = jac_right[:, right_arm_dofs]

        # Combine Jacobians for both arms
        J_combined = np.zeros((12, total_dofs))
        J_combined[:6, :n_dofs_per_arm] = jac_left_arm
        J_combined[6:, n_dofs_per_arm:] = jac_right_arm

        # For lifting, we're mainly concerned with the z-direction
        # Extract z-rows from Jacobians
        J_z = np.zeros((2, total_dofs))
        J_z[0, :n_dofs_per_arm] = jac_left_arm[2, :]  # z-row for left arm
        J_z[1, n_dofs_per_arm:] = jac_right_arm[2, :]  # z-row for right arm

        # Compute pseudoinverse
        J_z_pinv = np.linalg.pinv(J_z)

        # Compute joint velocities for lifting
        v_z = np.array([object_velocity[2], object_velocity[2]])  # Same z-velocity for both contacts
        joint_velocities = J_z_pinv @ v_z

        # Scale for faster movement
        joint_velocities *= 2.0

        # Apply joint velocity limits
        max_joint_vel = 1  # rad/s
        joint_velocities = np.clip(joint_velocities, -max_joint_vel, max_joint_vel)

        return joint_velocities

    except Exception as e:
        print(f"Error in object_velocity_to_joint_velocities: {e}")
        import traceback
        traceback.print_exc()

        # # Return a simple velocity command as fallback
        # return np.array([
        #     # Left arm - adjust shoulder_lift and elbow joints for lifting
        #     0.0, 0.2, 0.0, 0.2, 0.0, 0.0,
        #     # Right arm - adjust shoulder_lift and elbow joints for lifting
        #     0.0, 0.2, 0.0, 0.2, 0.0, 0.0
        # ])

def main():
    global control_state
    control_state = {}
    # Path to your XML file
    xml_file_path = "/home/iitgn-robotics/Saikrishna/samriddhi_mujoco_bimanual_ds_dynamic_task/robot_description/kshitij_lifting.xml"
    model = mujoco.MjModel.from_xml_path(xml_file_path)

    # ✅ Disable gravity globally at the start
    model.opt.gravity[:] = [0.0, 0.0, 0.0]

    # ✅ Store original gravity to re-enable later
    original_gravity = np.array([0.0, 0.0, -9.81])


    # Load the model and data
    data = mujoco.MjData(model)

    # Reset the simulation to ensure clean initial state
    mujoco.mj_resetData(model, data)
    mujoco.mj_forward(model, data)

    # Find box body ID
    box_body_id = -1
    for i in range(model.nbody):
        if "box" in model.body(i).name:
            box_body_id = i
            break

    if box_body_id == -1:
        return

    target_lift_height = 0.3  # 50 cm from ground

    if box_body_id != -1:
        initial_box_height = data.xpos[box_body_id][2]
    else:
        initial_box_height = 0.1  # Default assumption

    # Define per-joint gains for the velocity controller
    # Format: [joint1, joint2, joint3, joint4, joint5, joint6] for each robot
    kd_gains = np.array([
        # Left robot - increased gains for better tracking
        60.0, 100.0, 70.0, 30.0, 15.0, 15.0,
        # Right robot - increased gains for better tracking
        60.0, 100.0, 70.0, 30.0, 15.0, 15.0
    ])

    ki_gains = np.array([
        # Left robot - increased integral gains
        0.01, 0.1, 0.01, 0.005, 0.002, 0.002,
        # Right robot - increased integral gains
        0.01, 0.1, 0.01, 0.005, 0.002, 0.002
    ])

    # Create the velocity controller with per-joint gains
    controller = VelocityControllerGC(model, data, kd=kd_gains, ki=ki_gains)

    # Get number of actuators per robot
    num_actuators = controller.num_actuators
    actuators_per_robot = num_actuators // 2


    # Define target joint positions for both robots
    # First target positions (approach positions)
    first_target_positions = np.zeros(num_actuators)

    # Left robot (first half of actuators)
    first_target_positions[:actuators_per_robot] = [-0.0805, 1.07, -0.126, 1.53, -0.00978, 0]

    # Right robot (second half of actuators)
    first_target_positions[actuators_per_robot:] = [0.105, 1.07, -0.126, -1.45, -0.00901, 0]

    # Second target positions (grasp positions)
    second_target_positions = np.zeros(num_actuators)

    # # Left robot (first half of actuators)
    # second_target_positions[:actuators_per_robot] = [0.221, 1.16, -0.0314, 1.79, -0.00978, 0]

    # # Right robot (second half of actuators)
    # second_target_positions[actuators_per_robot:] = [-0.221, 1.16, -0.0314, -1.79, -0.0314, 0]

    # Left robot (first half of actuators)
    second_target_positions[:actuators_per_robot] = [0.147, 1.16, -0.0314, 1.79, -0.00978, 0]

    # Right robot (second half of actuators)
    second_target_positions[actuators_per_robot:] = [-0.147, 1.16, -0.0314, -1.79, -0.0314, 0]




    # Initialize control state
    control_state.update({
        "phase": "reaching_first_target",
        "current_targets": first_target_positions.copy(),
        "phase_start_time": time.time(),
        "last_phase_change": time.time(),
        "last_print_time": time.time(),
        "initial_box_height": initial_box_height,
        "target_lift_height": target_lift_height,
        "lifting_complete": False,
        "box_body_id": box_body_id,
        "original_gravity": original_gravity,

        # throwing-related
        "throw_phase_started": False,
        "throw_release_time": None,
        "throw_velocity_applied": False,
        "landing_point": LANDING_POINT,
        "throw_angle": THROW_ANGLE,
        "throw_release_velocity": np.zeros(3),
    })

    # Store in control_state
    control_state["box_body_id"] = box_body_id


    # Define per-joint position control gains (kp)
    kp_gains = np.array([
        # Left robot - increased gains
        15.0, 60.0, 15.0, 5.0, 3.0, 3.0,
        # Right robot - increased gains
        15.0, 60.0, 15.0, 5.0, 3.0, 3.0
    ])

    # Maximum allowed velocity for each joint (rad/s)
    max_velocities = np.array([
        1.2, 1.2, 1.2, 1.5, 1.5, 1.5,
        1.2, 1.2, 1.2, 1.5, 1.5, 1.5
    ])


    # Position error threshold for considering a joint "reached" (radians)
    position_thresholds = np.array([
        # Left robot
        0.02, 0.02, 0.02, 0.03, 0.03, 0.03,
        # Right robot
        0.02, 0.02, 0.02, 0.03, 0.03, 0.03
    ])

    # Function to compute velocity commands based on position error
    def position_to_velocity_trajectory(t):
        """
        Compute velocity commands based on current state and time.

        Args:
            t: Current simulation time

        Returns:
            velocity_commands: Joint velocity commands for all actuators
        """

        # Get current joint positions
        joint_positions = np.zeros(num_actuators)
        for i in range(num_actuators):
            joint_id = model.actuator_trnid[i, 0]
            joint_positions[i] = data.qpos[model.jnt_qposadr[joint_id]]

        # Control logic based on current phase
        current_time = time.time()

        # Print current phase periodically
        if current_time - control_state.get("last_print_time", 0) > 1.0:
            print(f"📍 Phase: {control_state['phase']}")
            control_state["last_print_time"] = current_time

        # Store the active control strategy for logging
        if "active_control_strategy" not in control_state:
            control_state["active_control_strategy"] = "position_control"

        if control_state["phase"] == "reaching_first_target" or control_state["phase"] == "reaching_second_target":
            control_state["active_control_strategy"] = "position_control"
            # Compute position errors based on current targets
            position_errors = control_state["current_targets"] - joint_positions

            # Compute velocity commands using proportional control with per-joint gains
            velocity_commands = kp_gains * position_errors

            # Limit velocity commands using per-joint max velocities
            velocity_commands = np.clip(velocity_commands, -max_velocities, max_velocities)

            # Apply a small deadband to prevent tiny movements
            for i in range(num_actuators):
                if abs(position_errors[i]) < position_thresholds[i] * 0.5:
                    velocity_commands[i] = 0.0

            # Check if all joints have reached their targets
            all_reached = True
            for i in range(num_actuators):
                if abs(position_errors[i]) > position_thresholds[i]:
                    all_reached = False
                    break

            # If all joints have reached their targets
            if all_reached:
                if control_state["phase"] == "reaching_first_target":
                    # Switch to second target positions after reaching first targets
                    control_state["phase"] = "reaching_second_target"
                    control_state["current_targets"] = second_target_positions.copy()
                    control_state["phase_start_time"] = current_time
                    control_state["last_phase_change"] = current_time
                    return np.zeros(num_actuators)  # Momentarily stop before moving to next target
                elif control_state["phase"] == "reaching_second_target":
                    # After reaching second targets, switch to sensor reading phase
                    if current_time - control_state["last_phase_change"] > 0.1:
                        control_state["phase"] = "reading_sensors"
                        control_state["last_phase_change"] = current_time
                        return np.zeros(num_actuators)  # Maintain position

            return velocity_commands

        elif control_state["phase"] == "reading_sensors":
            control_state["active_control_strategy"] = "none"
            # After a few seconds in reading_sensors phase, switch to lifting phase
            if current_time - control_state["last_phase_change"] > 0.1:
                control_state["phase"] = "lifting_object"
                control_state["last_phase_change"] = current_time

            return np.zeros(num_actuators)  # Maintain position while reading sensors

        elif control_state["phase"] == "lifting_object":
            """
            Phase: lifting_object

            1. While box height < target -> use admittance control to lift.
            2. When box height >= target -> compute throw velocity, switch to 'throwing_object',
               and let that phase handle the actual release.
            """
            box_body_id = control_state["box_body_id"]
            current_box_height = data.xpos[box_body_id][2]
            height_threshold = 0.01  # 1 cm tolerance

            # ---------- 1) IF LIFT TARGET REACHED → PREP THROWING PHASE ----------
            if current_box_height >= control_state["target_lift_height"] - height_threshold:
                print(f"⚠️ LIFT TARGET REACHED! Box at {current_box_height:.3f}m >= target {control_state['target_lift_height']:.3f}m")
                # Only do this once, when we first hit the lift height
                if not control_state.get("throw_phase_started", False):
                    # Current release position (world frame)
                    prel = data.xpos[box_body_id].copy()

                    # Desired landing point & launch angle from control_state / globals
                    pland = control_state["landing_point"]
                    theta = control_state["throw_angle"]

                    # Compute release velocity using the projectile math
                    v_release = compute_release_velocity(prel, pland, theta, g=G)

                    control_state.update({
                        "throw_release_pos": prel,
                        "throw_release_velocity": v_release,
                        # FIX #3: Initialize ds_obj_vel with current velocity (removed, now done in compute_throwing_ds)
                        "throw_phase_started": True,
                    })

                    # Switch phase
                    control_state["phase"] = "throwing_object"

                    # Turn on gravity for ballistic motion
                    model.opt.gravity[:] = np.array([0.0, 0.0, -G])

                # Once we decided to throw, this phase no longer commands joints
                return np.zeros(num_actuators)

            # ---------- 2) IF STILL LIFTING → NORMAL ADMITTANCE-BASED LIFT ----------
            left_wrench, right_wrench, contact_detected = get_contact_wrenches(
                model, data, control_state
            )

            if "contact_retry_count" not in control_state:
                control_state["contact_retry_count"] = 0

            # Debug: Print lifting progress every 100 steps
            if "lift_debug_counter" not in control_state:
                control_state["lift_debug_counter"] = 0
            control_state["lift_debug_counter"] += 1
            if control_state["lift_debug_counter"] % 100 == 0:
                print(f"🔧 LIFTING: height={current_box_height:.3f}m, target={control_state['target_lift_height']:.3f}m, contact={contact_detected}")

            if contact_detected:
                control_state["contact_retry_count"] = 0

                # Compute desired object velocity from admittance
                object_velocity, W, W_desired, pose_error = compute_object_admittance_control(
                    model,
                    data,
                    left_wrench,
                    right_wrench,
                    control_state,
                    desired_height=control_state["target_lift_height"],
                )

                # Map object velocity → joint velocities for both arms
                joint_velocities = object_velocity_to_joint_velocities(
                    model, data, object_velocity, control_state
                )
                return joint_velocities

            else:
                # No contact detected: for now, just keep joints still
                control_state["contact_retry_count"] += 1
                return np.zeros(num_actuators)

        elif control_state["phase"] == "throwing_object":

            dt = model.opt.timestep

            # 1️⃣ Object-level DS (with fixes applied)
            obj_vel = compute_throwing_ds(model, data, control_state, dt)

            # ---- CHECK CONTACT BEFORE APPLYING DS ----
            left_w, right_w, contact = get_contact_wrenches(model, data, control_state)

            if not contact:
                # Object released → stop commanding arms
                return np.zeros(num_actuators)


            # 2️⃣ Map object velocity → joint velocity
            try:
                left_ee = right_ee = -1
                for i in range(model.nsite):
                    name = model.site(i).name.lower()
                    if "left" in name and left_ee == -1:
                        left_ee = i
                    elif "right" in name and right_ee == -1:
                        right_ee = i

                if left_ee == -1 or right_ee == -1:
                    return np.zeros(num_actuators)

                jac_l = np.zeros((6, model.nv))
                jac_r = np.zeros((6, model.nv))
                mujoco.mj_jacSite(model, data, jac_l[:3], jac_l[3:], left_ee)
                mujoco.mj_jacSite(model, data, jac_r[:3], jac_r[3:], right_ee)

                left_dofs, right_dofs = [], []
                for i in range(actuators_per_robot):
                    jid = model.actuator_trnid[i, 0]
                    left_dofs.append(model.jnt_dofadr[jid])
                for i in range(actuators_per_robot, num_actuators):
                    jid = model.actuator_trnid[i, 0]
                    right_dofs.append(model.jnt_dofadr[jid])

                Jl = jac_l[:3, left_dofs]
                Jr = jac_r[:3, right_dofs]

                # Split object velocity between hands
                v_half = 0.5 * obj_vel

                qdot = np.zeros(num_actuators)
                qdot[:actuators_per_robot] = np.linalg.pinv(Jl) @ v_half
                qdot[actuators_per_robot:] = np.linalg.pinv(Jr) @ v_half

                return np.clip(qdot, -3.0, 3.0)

            except Exception as e:
                print(f"DS throwing error: {e}")
                return np.zeros(num_actuators)


        else:
            # Unknown phase
            control_state["active_control_strategy"] = "unknown"
            return np.zeros(num_actuators)

    # Set the velocity trajectory function
    controller.set_velocity_trajectory(position_to_velocity_trajectory)

    # Register callback & launch viewer
    mujoco.set_mjcb_control(controller.control_callback)

    with viewer.launch_passive(model, data) as v:

        while True:
            mujoco.mj_step(model, data)

            box_body_id = control_state.get("box_body_id", -1)
            if box_body_id != -1:
                pos = data.xpos[box_body_id].copy()
                # FIX #4: Changed from [3:6] to [:3] for linear velocity
                if hasattr(data, "cvel"):
                    lin_vel = data.cvel[box_body_id, :3].copy()  # ✅ FIXED: Linear velocity
                else:
                    lin_vel = np.zeros(3)

                print(
                    f"box pos: [{pos[0]:.3f}, {pos[1]:.3f}, {pos[2]:.3f}], "
                    f"vel: [{lin_vel[0]:.3f}, {lin_vel[1]:.3f}, {lin_vel[2]:.3f}]"
                )

            v.sync()

            time.sleep(0.001)

if __name__ == "__main__":
    main()


