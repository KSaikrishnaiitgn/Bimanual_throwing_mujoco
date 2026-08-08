import os
import sys
os.environ["MUJOCO_GL"] = "glfw"

# Ensure the scripts directory is on the path so utils imports work regardless of cwd
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__))))

import mujoco
import mujoco.viewer
import numpy as np
import time

# ─────────────────────────────────────────────
#  GLOBAL CONSTANTS
# ─────────────────────────────────────────────
G             = 9.81
LANDING_POINT = np.array([0.0, -2.0, 0.0])
THROW_ANGLE   = np.deg2rad(60.0)

# Throw motion tuning
ACCELERATION_DURATION = 0.4    # seconds to accelerate
MAX_ARM_VEL = 8.0              

# Second-order dynamics parameters
K_DS = 60.0    
B_DS = 15.0    

from utils.mujoco_velocity_controller.samriddhi_dual_arm_velocity_controller import VelocityControllerGC


# ─────────────────────────────────────────────
#  PROJECTILE MATH
# ─────────────────────────────────────────────
def compute_release_velocity(prel, pland, theta, g=G):
    delta_p  = pland - prel
    delta_xy = delta_p[:2]
    R        = np.linalg.norm(delta_xy)
    delta_z  = delta_p[2]

    if R < 1e-6:
        raise ValueError("Horizontal distance R ~ 0.")

    cos_t = np.cos(theta)
    sin_t = np.sin(theta)
    denom = 2.0 * (cos_t ** 2) * (R * np.tan(theta) - delta_z)

    if denom <= 0:
        raise ValueError("Invalid angle / landing point.")

    v_mag        = np.sqrt(g * R ** 2 / denom)
    v_mag = np.clip(v_mag, 2.0, 6.0)
    
    e_h          = delta_xy / R
    v_horizontal = v_mag * cos_t * np.array([e_h[0], e_h[1], 0.0])
    v_vertical   = v_mag * sin_t * np.array([0.0, 0.0, 1.0])
    return v_horizontal + v_vertical


# ─────────────────────────────────────────────
#  SECOND ORDER DYNAMICAL SYSTEM
# ─────────────────────────────────────────────
class SecondOrderDynamics:
    def __init__(self, target_velocity, K_ds=K_DS, B_ds=B_DS):
        self.target_vel = target_velocity.copy()
        self.K_ds = K_ds
        self.B_ds = B_ds
        self.current_desired_vel = np.zeros(3)
        
    def update(self, current_velocity, dt):
        vel_error = current_velocity - self.target_vel
        desired_acc = -self.K_ds * vel_error - self.B_ds * current_velocity
        
        # Limit acceleration
        acc_mag = np.linalg.norm(desired_acc)
        if acc_mag > 100.0:
            desired_acc = desired_acc / acc_mag * 100.0
        
        self.current_desired_vel += desired_acc * dt
        
        # Don't overshoot
        target_mag = np.linalg.norm(self.target_vel)
        current_mag = np.linalg.norm(self.current_desired_vel)
        if current_mag > target_mag * 1.1:
            self.current_desired_vel = self.target_vel.copy()
            
        return self.current_desired_vel.copy()


# ─────────────────────────────────────────────
#  JACOBIAN VELOCITY MAPPING
# ─────────────────────────────────────────────
def compute_throw_joint_velocities(model, data, desired_velocity,
                                   actuators_per_robot, num_actuators):
    try:
        left_ee_site_id  = -1
        right_ee_site_id = -1
        for i in range(model.nsite):
            name = model.site(i).name
            if "left"  in name.lower() and left_ee_site_id  == -1:
                left_ee_site_id  = i
            elif "right" in name.lower() and right_ee_site_id == -1:
                right_ee_site_id = i

        if left_ee_site_id == -1 or right_ee_site_id == -1:
            return np.zeros(num_actuators)

        left_arm_dofs  = [model.jnt_dofadr[model.actuator_trnid[i, 0]]
                          for i in range(actuators_per_robot)]
        right_arm_dofs = [model.jnt_dofadr[model.actuator_trnid[i, 0]]
                          for i in range(actuators_per_robot, num_actuators)]

        jac_left  = np.zeros((6, model.nv))
        jac_right = np.zeros((6, model.nv))
        mujoco.mj_jacSite(model, data, jac_left[:3],  jac_left[3:],  left_ee_site_id)
        mujoco.mj_jacSite(model, data, jac_right[:3], jac_right[3:], right_ee_site_id)

        jac_left_arm  = jac_left[:,  left_arm_dofs]
        jac_right_arm = jac_right[:, right_arm_dofs]

        desired_vel_6d     = np.zeros(6)
        desired_vel_6d[:3] = desired_velocity

        lambda_reg = 0.01
        qdot_left  = jac_left_arm.T @ np.linalg.inv(jac_left_arm @ jac_left_arm.T + lambda_reg * np.eye(6)) @ desired_vel_6d
        qdot_right = jac_right_arm.T @ np.linalg.inv(jac_right_arm @ jac_right_arm.T + lambda_reg * np.eye(6)) @ desired_vel_6d

        throw_velocity = np.zeros(num_actuators)
        throw_velocity[:actuators_per_robot]  = qdot_left
        throw_velocity[actuators_per_robot:]  = qdot_right

        throw_velocity = np.clip(throw_velocity, -MAX_ARM_VEL, MAX_ARM_VEL)
        return throw_velocity

    except Exception as e:
        print(f"Error: {e}")
        return np.zeros(num_actuators)


# ─────────────────────────────────────────────
#  MAIN
# ─────────────────────────────────────────────
def main():
    control_state = {}

    xml_file_path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                 "..", "robot_description", "kshitij_lifting.xml")
    model = mujoco.MjModel.from_xml_path(xml_file_path)
    data  = mujoco.MjData(model)

    # Keep gravity OFF initially
    original_gravity = np.array([0.0, 0.0, -G])
    model.opt.gravity[:] = [0.0, 0.0, 0.0]

    mujoco.mj_resetData(model, data)
    mujoco.mj_forward(model, data)

    box_body_id = next((i for i in range(model.nbody)
                        if "box" in model.body(i).name), -1)
    if box_body_id == -1:
        print("❌  Cannot find box body.")
        return

    # Controller
    kd_gains = np.array([60.0, 100.0, 70.0, 30.0, 15.0, 15.0,
                         60.0, 100.0, 70.0, 30.0, 15.0, 15.0])
    ki_gains = np.array([0.01, 0.1, 0.01, 0.005, 0.002, 0.002,
                         0.01, 0.1, 0.01, 0.005, 0.002, 0.002])

    controller = VelocityControllerGC(model, data, kd=kd_gains, ki=ki_gains)
    num_actuators = controller.num_actuators
    actuators_per_robot = num_actuators // 2

    # Joint targets (working from original code)
    first_target = np.zeros(num_actuators)
    first_target[:actuators_per_robot] = [-0.0805, 1.07, -0.126, 1.53, -0.00978, 0]
    first_target[actuators_per_robot:] = [0.105, 1.07, -0.126, -1.45, -0.00901, 0]

    second_target = np.zeros(num_actuators)
    second_target[:actuators_per_robot] = [0.147, 1.16, -0.0314, 1.79, -0.00978, 0.5]
    second_target[actuators_per_robot:] = [-0.147, 1.16, -0.0314, -1.79, -0.0314, 0.5]

    # Position control gains
    kp_gains = np.array([15.0, 60.0, 15.0, 5.0, 3.0, 5.0,
                         15.0, 60.0, 15.0, 5.0, 3.0, 5.0])
    max_vels = np.array([1.2, 1.2, 1.2, 1.5, 1.5, 2.0,
                         1.2, 1.2, 1.2, 1.5, 1.5, 2.0])
    pos_thresh = np.array([0.02, 0.02, 0.02, 0.03, 0.03, 0.1,
                           0.02, 0.02, 0.02, 0.03, 0.03, 0.1])

    control_state.update({
        "phase": "reaching_first_target",
        "current_targets": first_target.copy(),
        "last_phase_change": time.time(),
        "last_print_time": time.time(),
        "box_body_id": box_body_id,
        "original_gravity": original_gravity,
        "landing_point": LANDING_POINT,
        "throw_angle": THROW_ANGLE,
        "acceleration_start_time": None,
        "acceleration_dynamics": None,
        "release_velocity": np.zeros(3),
        "released": False,
    })

    def position_to_velocity_trajectory(t):
        joint_positions = np.zeros(num_actuators)
        for i in range(num_actuators):
            jid = model.actuator_trnid[i, 0]
            joint_positions[i] = data.qpos[model.jnt_qposadr[jid]]

        current_time = time.time()
        phase = control_state["phase"]

        # Phase 1 & 2: Position control to grasp
        if phase in ("reaching_first_target", "reaching_second_target"):
            errors = control_state["current_targets"] - joint_positions
            cmds = kp_gains * errors
            cmds = np.clip(cmds, -max_vels, max_vels)
            for i in range(num_actuators):
                if abs(errors[i]) < pos_thresh[i] * 0.5:
                    cmds[i] = 0.0

            all_reached = all(abs(errors[i]) <= pos_thresh[i] for i in range(num_actuators))

            if all_reached:
                if phase == "reaching_first_target":
                    print("✅ First target reached — moving to grasp position.")
                    control_state["phase"] = "reaching_second_target"
                    control_state["current_targets"] = second_target.copy()
                    control_state["last_phase_change"] = current_time
                    return np.zeros(num_actuators)
                elif phase == "reaching_second_target":
                    if current_time - control_state["last_phase_change"] > 0.1:
                        print("✅ Grasp position reached — box grasped!")
                        print("🎯 Starting acceleration phase...")
                        
                        # Compute target release velocity
                        prel = data.xpos[box_body_id].copy()
                        v_release = compute_release_velocity(prel, LANDING_POINT, THROW_ANGLE, G)
                        
                        print(f"\n📊 Release calculation:")
                        print(f"    Current box position: {prel}")
                        print(f"    Target landing: {LANDING_POINT}")
                        print(f"    Target release velocity: {v_release}")
                        print(f"    Speed: {np.linalg.norm(v_release):.2f} m/s")
                        print(f"    Acceleration duration: {ACCELERATION_DURATION}s\n")
                        
                        # Initialize DS
                        dynamics = SecondOrderDynamics(v_release, K_DS, B_DS)
                        
                        control_state["acceleration_dynamics"] = dynamics
                        control_state["release_velocity"] = v_release
                        control_state["acceleration_start_time"] = data.time
                        control_state["released"] = False
                        control_state["phase"] = "accelerating"
                        
                        # CRITICAL: Keep gravity OFF during acceleration
                        model.opt.gravity[:] = [0.0, 0.0, 0.0]
                        
                        # DO NOT disable contacts! Keep them so arms can push box
                        # But ensure grippers are closed
                        return np.zeros(num_actuators)
            return cmds

        # Phase 3: Accelerate box (gravity OFF, contacts ON)
        elif phase == "accelerating":
            if control_state.get("released", False):
                return np.zeros(num_actuators)
            
            sim_time = data.time
            elapsed = sim_time - control_state["acceleration_start_time"]
            
            # Get current box velocity
            box_dof_adr = model.body_dofadr[box_body_id]
            current_box_vel = data.qvel[box_dof_adr:box_dof_adr + 3].copy()
            current_box_pos = data.xpos[box_body_id].copy()
            
            # Update DS
            dt = max(0.001, sim_time - control_state.get("last_time", sim_time))
            control_state["last_time"] = sim_time
            
            dynamics = control_state["acceleration_dynamics"]
            desired_vel = dynamics.update(current_box_vel, dt)
            
            target_mag = np.linalg.norm(control_state["release_velocity"])
            current_mag = np.linalg.norm(current_box_vel)
            
            # Print progress
            if int(elapsed * 20) % 20 == 0 and elapsed > 0.01:
                print(f"   t={elapsed:.2f}s | box vel: {current_mag:.2f}/{target_mag:.2f} m/s | height: {current_box_pos[2]:.3f}m")
            
            # Release condition
            if elapsed >= ACCELERATION_DURATION:
                control_state["released"] = True
                
                # Enable gravity at release
                model.opt.gravity[:] = control_state["original_gravity"]
                
                final_pos = current_box_pos.copy()
                final_vel = current_box_vel.copy()
                
                print(f"\n🚀 RELEASED at t={elapsed:.3f}s!")
                print(f"    Release position: {final_pos}")
                print(f"    Release velocity: {final_vel}")
                print(f"    Target velocity : {control_state['release_velocity']}")
                print(f"    Target landing  : {LANDING_POINT}\n")
                
                return np.zeros(num_actuators)
            
            # Command arms to move at DS velocity
            # Scale up to ensure arms lead the box
            cmd_vel = desired_vel * 1.3
            
            # Ensure upward motion during acceleration
            if cmd_vel[2] < 0.8 and elapsed < ACCELERATION_DURATION * 0.7:
                cmd_vel[2] = 0.8
            
            return compute_throw_joint_velocities(model, data, cmd_vel,
                                                   actuators_per_robot, num_actuators)

        return np.zeros(num_actuators)

    # Launch
    controller.set_velocity_trajectory(position_to_velocity_trajectory)
    mujoco.set_mjcb_control(controller.control_callback)

    with mujoco.viewer.launch_passive(model, data) as viewer:
        last_print_time = 0
        while viewer.is_running():
            mujoco.mj_step(model, data)

            current_time = time.time()
            if current_time - last_print_time > 0.1:
                bid = control_state.get("box_body_id", -1)
                if bid != -1:
                    pos = data.xpos[bid].copy()
                    dof = model.body_dofadr[bid]
                    vel = data.qvel[dof:dof + 3].copy()
                    print(f"box pos: [{pos[0]:.3f}, {pos[1]:.3f}, {pos[2]:.3f}],  "
                          f"vel: [{vel[0]:.3f}, {vel[1]:.3f}, {vel[2]:.3f}]")
                last_print_time = current_time

            viewer.sync()
            time.sleep(0.001)


if __name__ == "__main__":
    main()