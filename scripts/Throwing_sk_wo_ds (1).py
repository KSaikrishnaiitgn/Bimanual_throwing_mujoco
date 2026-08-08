import os
os.environ["MUJOCO_GL"] = "glfw"
import mujoco
import mujoco.viewer
import numpy as np
import time

# ─────────────────────────────────────────────
#  GLOBAL CONSTANTS
# ─────────────────────────────────────────────
G             = 9.81
LANDING_POINT = np.array([0.0, -2.5, 0.0])
THROW_ANGLE   = np.deg2rad(60.0)

# Throw motion tuning
THROW_DURATION = 0.20   # seconds of arm swing before release
THROW_ARM_SPEED = 3.0   # m/s end-effector speed during swing

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
    e_h          = delta_xy / R
    v_horizontal = v_mag * cos_t * np.array([e_h[0], e_h[1], 0.0])
    v_vertical   = v_mag * sin_t * np.array([0.0, 0.0, 1.0])
    return v_horizontal + v_vertical


# ─────────────────────────────────────────────
#  CONTACT WRENCH HELPER
# ─────────────────────────────────────────────
def get_contact_wrenches(model, data, control_state):
    box_body_id = control_state.get("box_body_id", -1)
    if box_body_id == -1:
        return np.zeros(6), np.zeros(6), False

    box_geom_ids = [i for i in range(model.ngeom)
                    if model.geom_bodyid[i] == box_body_id]

    left_wrench  = np.zeros(6)
    right_wrench = np.zeros(6)
    contact_detected = False

    for i in range(data.ncon):
        contact = data.contact[i]
        if contact.geom1 not in box_geom_ids and contact.geom2 not in box_geom_ids:
            continue

        contact_force = np.zeros(6)
        mujoco.mj_contactForce(model, data, i, contact_force)

        other_geom = contact.geom2 if contact.geom1 in box_geom_ids else contact.geom1
        other_body = model.geom_bodyid[other_geom]
        body_name  = model.body(other_body).name

        lever_arm = contact.pos.copy() - data.xpos[box_body_id].copy()
        torque    = np.cross(lever_arm, contact_force[:3])

        if "left" in body_name.lower():
            left_wrench[:3]  += contact_force[:3]
            left_wrench[3:]  += torque
            contact_detected  = True
        elif "right" in body_name.lower():
            right_wrench[:3] += contact_force[:3]
            right_wrench[3:] += torque
            contact_detected  = True

    return left_wrench, right_wrench, contact_detected


# ─────────────────────────────────────────────
#  ADMITTANCE CONTROL
# ─────────────────────────────────────────────
def compute_object_admittance_control(model, data, left_wrench, right_wrench,
                                      control_state, desired_height=0.5):
    box_body_id = control_state.get("box_body_id", -1)
    if box_body_id == -1:
        return np.zeros(6), np.zeros(6), np.zeros(6), np.zeros(6)

    current_obj_pos  = data.xpos[box_body_id].copy()
    current_obj_quat = data.xquat[box_body_id].copy()
    desired_obj_pos  = current_obj_pos.copy()
    desired_obj_pos[2] = desired_height

    pos_error        = desired_obj_pos - current_obj_pos
    height_error     = desired_height - current_obj_pos[2]
    height_error_pct = min(100, max(0,
                       (height_error / max(desired_height - 0.1, 1e-6)) * 100))

    current_obj_vel = np.zeros(6)
    if hasattr(data, 'cvel'):
        current_obj_vel[:3] = data.cvel[box_body_id, :3]
        current_obj_vel[3:] = data.cvel[box_body_id, 3:6]

    W = left_wrench + right_wrench

    object_mass   = model.body_mass[box_body_id]
    weight_force  = object_mass * G
    upward_factor = 20.0 + 15.0 * (height_error_pct / 100.0)
    W_desired     = np.zeros(6)
    W_desired[2]  = weight_force * 2.0 + upward_factor

    K     = np.diag([200.0, 200.0, 400.0, 20.0, 20.0, 20.0])
    D     = np.diag([ 20.0,  20.0,  10.0,  3.0,  3.0,  3.0])
    D_inv = np.linalg.inv(D)

    pose_error     = np.zeros(6)
    pose_error[:3] = pos_error

    object_velocity = current_obj_vel + D_inv @ (W - W_desired - K @ pose_error)
    object_velocity[:3] = np.clip(object_velocity[:3], -0.5, 0.5)
    object_velocity[3:] = np.clip(object_velocity[3:], -0.5, 0.5)

    if height_error > 0.01:
        min_up = 0.2 * (height_error_pct / 100.0 + 0.5)
        object_velocity[2] = max(object_velocity[2], min_up)
    else:
        object_velocity[2] = min(object_velocity[2], 0.05)

    return object_velocity, W, W_desired, pose_error


# ─────────────────────────────────────────────
#  JACOBIAN VELOCITY MAPPING (lifting)
# ─────────────────────────────────────────────
def object_velocity_to_joint_velocities(model, data, object_velocity, control_state):
    try:
        num_actuators       = model.nu
        actuators_per_robot = num_actuators // 2

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
        n_dofs        = len(left_arm_dofs)

        J_z = np.zeros((2, n_dofs * 2))
        J_z[0, :n_dofs]  = jac_left_arm[2, :]
        J_z[1,  n_dofs:] = jac_right_arm[2, :]

        v_z              = np.array([object_velocity[2], object_velocity[2]])
        joint_velocities = np.linalg.pinv(J_z) @ v_z
        joint_velocities *= 2.0
        joint_velocities  = np.clip(joint_velocities, -1.0, 1.0)
        return joint_velocities

    except Exception as e:
        print(f"Error in object_velocity_to_joint_velocities: {e}")
        return np.zeros(model.nu)


# ─────────────────────────────────────────────
#  JACOBIAN VELOCITY MAPPING (throwing)
#  — moves end-effectors in an arbitrary 3-D direction
# ─────────────────────────────────────────────
def compute_throw_joint_velocities(model, data, throw_dir_3d, ee_speed,
                                   actuators_per_robot, num_actuators):
    """
    Compute joint velocities that drive both end-effectors in throw_dir_3d
    at the requested ee_speed (m/s).
    Returns a (num_actuators,) array.
    """
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

        jac_left_arm  = jac_left[:,  left_arm_dofs]   # 6 × n
        jac_right_arm = jac_right[:, right_arm_dofs]  # 6 × n

        # desired 6-D ee velocity: only the linear part, in throw direction
        desired_vel_6d     = np.zeros(6)
        desired_vel_6d[:3] = throw_dir_3d * ee_speed

        # pseudoinverse for each arm separately
        qdot_left  = np.linalg.pinv(jac_left_arm)  @ desired_vel_6d
        qdot_right = np.linalg.pinv(jac_right_arm) @ desired_vel_6d

        throw_velocity = np.zeros(num_actuators)
        throw_velocity[:actuators_per_robot]  = qdot_left
        throw_velocity[actuators_per_robot:]  = qdot_right

        throw_velocity = np.clip(throw_velocity, -5.0, 5.0)
        return throw_velocity

    except Exception as e:
        print(f"Error in compute_throw_joint_velocities: {e}")
        return np.zeros(num_actuators)


# ─────────────────────────────────────────────
#  MAIN
# ─────────────────────────────────────────────
def main():
    global control_state
    control_state = {}

    xml_file_path = "/home/iitgn-robotics/Saikrishna/samriddhi_mujoco_bimanual_ds_dynamic_task/robot_description/kshitij_lifting.xml"
    model = mujoco.MjModel.from_xml_path(xml_file_path)
    data  = mujoco.MjData(model)

    # disable gravity so arms can settle before box falls
    model.opt.gravity[:] = [0.0, 0.0, 0.0]
    original_gravity      = np.array([0.0, 0.0, -G])

    mujoco.mj_resetData(model, data)
    mujoco.mj_forward(model, data)

    box_body_id = next((i for i in range(model.nbody)
                        if "box" in model.body(i).name), -1)
    if box_body_id == -1:
        print("❌  Cannot find box body.")
        return

    initial_box_height = data.xpos[box_body_id][2]
    target_lift_height = 0.3

    # ── controller ───────────────────────────────────────────────────────────
    kd_gains = np.array([60.0, 100.0, 70.0, 30.0, 15.0, 15.0,
                         60.0, 100.0, 70.0, 30.0, 15.0, 15.0])
    ki_gains = np.array([0.01, 0.1, 0.01, 0.005, 0.002, 0.002,
                         0.01, 0.1, 0.01, 0.005, 0.002, 0.002])

    controller          = VelocityControllerGC(model, data, kd=kd_gains, ki=ki_gains)
    num_actuators       = controller.num_actuators
    actuators_per_robot = num_actuators // 2

    # ── joint targets ─────────────────────────────────────────────────────────
    first_target  = np.zeros(num_actuators)
    first_target[:actuators_per_robot]  = [-0.0805, 1.07, -0.126,  1.53,  -0.00978, 0]
    first_target[actuators_per_robot:]  = [ 0.105,  1.07, -0.126, -1.45,  -0.00901, 0]

    second_target = np.zeros(num_actuators)
    second_target[:actuators_per_robot] = [ 0.147,  1.16, -0.0314,  1.79,  -0.00978, 0]
    second_target[actuators_per_robot:] = [-0.147,  1.16, -0.0314, -1.79,  -0.0314,  0]

    # ── position control gains ────────────────────────────────────────────────
    kp_gains = np.array([15.0, 60.0, 15.0, 5.0, 3.0, 3.0,
                         15.0, 60.0, 15.0, 5.0, 3.0, 3.0])
    max_vels  = np.array([1.2, 1.2, 1.2, 1.5, 1.5, 1.5,
                          1.2, 1.2, 1.2, 1.5, 1.5, 1.5])
    pos_thresh = np.array([0.02, 0.02, 0.02, 0.03, 0.03, 0.03,
                           0.02, 0.02, 0.02, 0.03, 0.03, 0.03])

    # ── state dict ────────────────────────────────────────────────────────────
    control_state.update({
        "phase":               "reaching_first_target",
        "current_targets":     first_target.copy(),
        "phase_start_time":    time.time(),
        "last_phase_change":   time.time(),
        "last_print_time":     time.time(),
        "box_body_id":         box_body_id,
        "target_lift_height":  target_lift_height,
        "original_gravity":    original_gravity,
        # throwing
        "throw_phase_started":    False,
        "throw_swing_started":    False,   # True once arm swing begins
        "throw_swing_start_time": None,    # sim time when swing began
        "throw_velocity_applied": False,
        "throw_landed":           False,
        "throw_start_pos":        None,
        "throw_release_velocity": np.zeros(3),
        "throw_direction":        np.zeros(3),
        "landing_point":          LANDING_POINT,
        "throw_angle":            THROW_ANGLE,
    })

    # ─────────────────────────────────────────────────────────────────────────
    #  TRAJECTORY FUNCTION
    # ─────────────────────────────────────────────────────────────────────────
    def position_to_velocity_trajectory(t):

        joint_positions = np.zeros(num_actuators)
        for i in range(num_actuators):
            jid = model.actuator_trnid[i, 0]
            joint_positions[i] = data.qpos[model.jnt_qposadr[jid]]

        current_time = time.time()
        if current_time - control_state.get("last_print_time", 0) > 1.0:
            print(f"📍 Phase: {control_state['phase']}")
            control_state["last_print_time"] = current_time

        phase = control_state["phase"]

        # ── PHASE 1 & 2: position control to grasp pose ───────────────────────
        if phase in ("reaching_first_target", "reaching_second_target"):
            errors   = control_state["current_targets"] - joint_positions
            cmds     = kp_gains * errors
            cmds     = np.clip(cmds, -max_vels, max_vels)
            for i in range(num_actuators):
                if abs(errors[i]) < pos_thresh[i] * 0.5:
                    cmds[i] = 0.0

            all_reached = all(abs(errors[i]) <= pos_thresh[i]
                              for i in range(num_actuators))

            if all_reached:
                if phase == "reaching_first_target":
                    print("✅  First target reached — moving to grasp position.")
                    control_state["phase"]           = "reaching_second_target"
                    control_state["current_targets"] = second_target.copy()
                    control_state["last_phase_change"] = current_time
                    return np.zeros(num_actuators)
                elif phase == "reaching_second_target":
                    if current_time - control_state["last_phase_change"] > 0.1:
                        print("✅  Grasp position reached — reading sensors.")
                        control_state["phase"]             = "reading_sensors"
                        control_state["last_phase_change"] = current_time
                        return np.zeros(num_actuators)
            return cmds

        # ── PHASE 3: short pause ──────────────────────────────────────────────
        elif phase == "reading_sensors":
            if current_time - control_state["last_phase_change"] > 0.1:
                print("✅  Sensors ready — starting lift.")
                control_state["phase"]             = "lifting_object"
                control_state["last_phase_change"] = current_time
            return np.zeros(num_actuators)

        # ── PHASE 4: admittance-controlled lifting ────────────────────────────
        elif phase == "lifting_object":
            box_body_id       = control_state["box_body_id"]
            current_box_height = data.xpos[box_body_id][2]

            if current_box_height >= control_state["target_lift_height"] - 0.01:

                if not control_state.get("throw_phase_started", False):
                    prel  = data.xpos[box_body_id].copy()
                    pland = control_state["landing_point"]
                    theta = control_state["throw_angle"]

                    try:
                        v_release = compute_release_velocity(prel, pland, theta, g=G)
                    except ValueError as e:
                        print(f"❌  compute_release_velocity failed: {e}")
                        return np.zeros(num_actuators)

                    # Normalised throw direction (used to drive the arm swing)
                    throw_dir = v_release / (np.linalg.norm(v_release) + 1e-9)

                    control_state["throw_release_velocity"] = v_release
                    control_state["throw_direction"]        = throw_dir
                    control_state["throw_phase_started"]    = True
                    control_state["throw_velocity_applied"] = False
                    control_state["throw_landed"]           = False
                    control_state["throw_swing_started"]    = False

                    # re-enable gravity for ballistic flight
                    model.opt.gravity[:] = original_gravity

                    print(f"\n🎯  Lift target reached at z={current_box_height:.3f} m")
                    print(f"    Release pos : {prel}")
                    print(f"    Target land : {pland}")
                    v = v_release
                    print(f"    v_release   : [{v[0]:.3f}, {v[1]:.3f}, {v[2]:.3f}] m/s")
                    print(f"    throw_dir   : [{throw_dir[0]:.3f}, {throw_dir[1]:.3f}, {throw_dir[2]:.3f}]")
                    print(f"    Arm swing starts now ({THROW_DURATION:.2f}s), then release.\n")

                    control_state["phase"]             = "throwing_object"
                    control_state["last_phase_change"] = current_time

                return np.zeros(num_actuators)

            # still lifting
            if "lift_dbg" not in control_state:
                control_state["lift_dbg"] = 0
            control_state["lift_dbg"] += 1
            if control_state["lift_dbg"] % 100 == 0:
                print(f"🔧  LIFTING  h={current_box_height:.3f}  target={control_state['target_lift_height']:.3f}")

            lw, rw, contact = get_contact_wrenches(model, data, control_state)
            if contact:
                ov, *_ = compute_object_admittance_control(
                    model, data, lw, rw, control_state,
                    desired_height=control_state["target_lift_height"])
                return object_velocity_to_joint_velocities(model, data, ov, control_state)
            return np.zeros(num_actuators)

        elif phase == "throwing_object":
            box_body_id = control_state["box_body_id"]
            box_dof_adr = model.body_dofadr[box_body_id]
            sim_time    = data.time

            # ── STEP 1: Arm swing (ramps up then ramps down) ─────────────────────
            if not control_state.get("throw_velocity_applied", False):

                if not control_state["throw_swing_started"]:
                    control_state["throw_swing_started"]    = True
                    control_state["throw_swing_start_time"] = sim_time
                    print(f"🏋️  Arm swing started at sim_time={sim_time:.3f}s")

                elapsed = sim_time - control_state["throw_swing_start_time"]

                # Phase A: swing arms forward for THROW_DURATION * 0.7
                swing_time   = THROW_DURATION * 0.7
                # Phase B: ramp down + disable contacts for remaining 0.3
                release_time = THROW_DURATION

                if elapsed < swing_time:
                    # Full speed swing in throw direction
                    throw_dir = control_state["throw_direction"]
                    return compute_throw_joint_velocities(
                        model, data, throw_dir, THROW_ARM_SPEED,
                        actuators_per_robot, num_actuators)

                elif elapsed < release_time:
                    # Ramp down arm speed to zero smoothly
                    # AND disable contacts here so box is free before velocity injection
                    if not control_state.get("contacts_disabled", False):
                        control_state["contacts_disabled"] = True
                        print("🔓  Contacts disabled — decoupling box from arms...")
                        for i in range(model.ngeom):
                            gn = model.geom(i).name
                            if (gn and
                                    "box"    not in gn.lower() and
                                    "ground" not in gn.lower() and
                                    "floor"  not in gn.lower()):
                                model.geom_contype[i]     = 0
                                model.geom_conaffinity[i] = 0

                    # Ramp factor: 1.0 → 0.0 over this sub-phase
                    ramp = 1.0 - (elapsed - swing_time) / (release_time - swing_time)
                    throw_dir = control_state["throw_direction"]
                    return compute_throw_joint_velocities(
                        model, data, throw_dir, THROW_ARM_SPEED * ramp,
                        actuators_per_robot, num_actuators)

                else:
                    # ── STEP 2: Swing complete, contacts already disabled.
                    #            Now safely inject velocity ───────────────────────
                    control_state["throw_velocity_applied"] = True

                    v_release = control_state["throw_release_velocity"]

                    # Zero out any residual box velocity first
                    data.qvel[box_dof_adr : box_dof_adr + 6] = 0.0

                    # Then set the precise release velocity
                    data.qvel[box_dof_adr + 0] = v_release[0]
                    data.qvel[box_dof_adr + 1] = v_release[1]
                    data.qvel[box_dof_adr + 2] = v_release[2]

                    control_state["throw_start_pos"] = data.xpos[box_body_id].copy()

                    print(f"\n🚀  RELEASED after {elapsed:.3f}s swing!")
                    v = v_release
                    print(f"    v_release : [{v[0]:.3f}, {v[1]:.3f}, {v[2]:.3f}] m/s")
                    print(f"    Release pos : {data.xpos[box_body_id]}")
                    print(f"    Target      : {LANDING_POINT}\n")

                    return np.zeros(num_actuators)

            # ── STEP 3: Flight monitoring ─────────────────────────────────────────
            box_pos = data.xpos[box_body_id]
            box_vel = data.qvel[box_dof_adr : box_dof_adr + 3]
            speed   = np.linalg.norm(box_vel)

            if (not control_state.get("throw_landed", False) and
                    speed < 0.05 and box_pos[2] < 0.11):

                control_state["throw_landed"] = True
                start = control_state["throw_start_pos"]
                dist  = np.linalg.norm(box_pos[:2] - start[:2])
                error = np.linalg.norm(box_pos[:2] - LANDING_POINT[:2])

                print(f"\n🎯  LANDED!")
                print(f"    Horizontal distance : {dist:.3f} m")
                print(f"    Start  : [{start[0]:.3f}, {start[1]:.3f}, {start[2]:.3f}]")
                print(f"    End    : [{box_pos[0]:.3f}, {box_pos[1]:.3f}, {box_pos[2]:.3f}]")
                print(f"    Target : [{LANDING_POINT[0]:.3f}, {LANDING_POINT[1]:.3f}, {LANDING_POINT[2]:.3f}]")
                print(f"    2-D landing error : {error:.3f} m\n")

            return np.zeros(num_actuators)

    # ─────────────────────────────────────────────────────────────────────────
    #  LAUNCH
    # ─────────────────────────────────────────────────────────────────────────
    controller.set_velocity_trajectory(position_to_velocity_trajectory)
    mujoco.set_mjcb_control(controller.control_callback)

    with mujoco.viewer.launch_passive(model, data) as viewer:
        while viewer.is_running():
            mujoco.mj_step(model, data)

            bid = control_state.get("box_body_id", -1)
            if bid != -1:
                pos = data.xpos[bid].copy()
                dof = model.body_dofadr[bid]
                vel = data.qvel[dof : dof + 3].copy()
                print(f"box pos: [{pos[0]:.3f}, {pos[1]:.3f}, {pos[2]:.3f}],  "
                      f"vel: [{vel[0]:.3f}, {vel[1]:.3f}, {vel[2]:.3f}]")

            viewer.sync()
            time.sleep(0.001)


if __name__ == "__main__":
    main()