import os
os.environ["MUJOCO_GL"] = "glfw"  # Must be before mujoco import
import mujoco
import numpy as np
from mujoco.viewer import launch
import time

# ─────────────────────────────────────────────
#  GLOBAL CONSTANTS  — edit these to change behaviour
# ─────────────────────────────────────────────
G             = 9.81
LANDING_POINT = np.array([0.0, -2.5, 0.0])   # where the box should land
THROW_ANGLE   = np.deg2rad(60.0)              # launch angle above horizontal

# Import the velocity controller
from utils.mujoco_velocity_controller.samriddhi_dual_arm_velocity_controller import VelocityControllerGC


# ─────────────────────────────────────────────
#  PROJECTILE MATH
# ─────────────────────────────────────────────
def compute_release_velocity(prel, pland, theta, g=G):
    """
    Compute the 3-D release velocity needed for a ballistic throw
    from position prel to position pland at launch angle theta.
    """
    delta_p  = pland - prel
    delta_xy = delta_p[:2]
    R        = np.linalg.norm(delta_xy)
    delta_z  = delta_p[2]

    if R < 1e-6:
        raise ValueError("Horizontal distance R ~ 0; cannot define throw direction.")

    cos_t = np.cos(theta)
    sin_t = np.sin(theta)
    denom = 2.0 * (cos_t ** 2) * (R * np.tan(theta) - delta_z)

    if denom <= 0:
        raise ValueError("Invalid angle / landing point: need R*tan(theta) > delta_z.")

    v_mag       = np.sqrt(g * R ** 2 / denom)
    e_h         = delta_xy / R
    v_horizontal = v_mag * cos_t * np.array([e_h[0], e_h[1], 0.0])
    v_vertical   = v_mag * sin_t * np.array([0.0,    0.0,    1.0])

    return v_horizontal + v_vertical


# ─────────────────────────────────────────────
#  CONTACT WRENCH HELPER
# ─────────────────────────────────────────────
def get_contact_wrenches(model, data, control_state):
    """
    Return (left_wrench, right_wrench, contact_detected).
    Each wrench is a 6-vector [fx,fy,fz, tx,ty,tz] in world frame.
    """
    box_body_id = control_state.get("box_body_id", -1)
    if box_body_id == -1:
        return np.zeros(6), np.zeros(6), False

    # collect all geom IDs that belong to the box body
    box_geom_ids = [i for i in range(model.ngeom)
                    if model.geom_bodyid[i] == box_body_id]

    left_wrench  = np.zeros(6)
    right_wrench = np.zeros(6)
    contact_detected = False

    for i in range(data.ncon):
        contact = data.contact[i]
        if contact.geom1 not in box_geom_ids and contact.geom2 not in box_geom_ids:
            continue

        # get the 6-D contact force
        contact_force = np.zeros(6)
        mujoco.mj_contactForce(model, data, i, contact_force)

        # identify which end-effector body is touching the box
        other_geom = contact.geom2 if contact.geom1 in box_geom_ids else contact.geom1
        other_body = model.geom_bodyid[other_geom]
        body_name  = model.body(other_body).name

        # torque = r × F  (lever arm from box centre to contact point)
        contact_pos = contact.pos.copy()
        box_pos     = data.xpos[box_body_id].copy()
        lever_arm   = contact_pos - box_pos
        torque      = np.cross(lever_arm, contact_force[:3])

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
#  ADMITTANCE CONTROL (lifting phase)
# ─────────────────────────────────────────────
def compute_object_admittance_control(model, data,
                                      left_wrench, right_wrench,
                                      control_state,
                                      desired_height=0.5):
    """
    Compute desired object velocity using:
        xdot_o* = xdot_o + D_inv @ (W - W* - K @ pose_error)
    Returns (object_velocity, W, W_desired, pose_error).
    """
    box_body_id = control_state.get("box_body_id", -1)
    if box_body_id == -1:
        return np.zeros(6), np.zeros(6), np.zeros(6), np.zeros(6)

    current_obj_pos  = data.xpos[box_body_id].copy()
    current_obj_quat = data.xquat[box_body_id].copy()

    desired_obj_pos      = current_obj_pos.copy()
    desired_obj_pos[2]   = desired_height
    desired_obj_quat     = current_obj_quat.copy()

    pos_error            = desired_obj_pos - current_obj_pos
    height_error         = desired_height - current_obj_pos[2]
    height_error_pct     = min(100, max(0,
                               (height_error / max(desired_height - 0.1, 1e-6)) * 100))

    # current object velocity
    current_obj_vel = np.zeros(6)
    if hasattr(data, 'cvel'):
        current_obj_vel[:3] = data.cvel[box_body_id, :3]
        current_obj_vel[3:] = data.cvel[box_body_id, 3:6]

    # combined wrench from both hands
    W = left_wrench + right_wrench

    # desired wrench: must exceed gravity to lift
    object_mass    = model.body_mass[box_body_id]
    weight_force   = object_mass * G
    upward_factor  = 20.0 + 15.0 * (height_error_pct / 100.0)
    W_desired      = np.zeros(6)
    W_desired[2]   = weight_force * 2.0 + upward_factor

    # stiffness and damping matrices (diagonal)
    K     = np.diag([200.0, 200.0, 400.0, 20.0, 20.0, 20.0])
    D     = np.diag([ 20.0,  20.0,  10.0,  3.0,  3.0,  3.0])
    D_inv = np.linalg.inv(D)

    pose_error      = np.zeros(6)
    pose_error[:3]  = pos_error

    object_velocity = current_obj_vel + D_inv @ (W - W_desired - K @ pose_error)

    # safety velocity limits
    object_velocity[:3] = np.clip(object_velocity[:3], -0.5,  0.5)
    object_velocity[3:] = np.clip(object_velocity[3:], -0.5,  0.5)

    # enforce minimum upward velocity when not yet at target height
    if height_error > 0.01:
        min_up = 0.2 * (height_error_pct / 100.0 + 0.5)
        object_velocity[2] = max(object_velocity[2], min_up)
    else:
        object_velocity[2] = min(object_velocity[2], 0.05)

    return object_velocity, W, W_desired, pose_error


# ─────────────────────────────────────────────
#  JACOBIAN-BASED JOINT VELOCITY MAPPING (lifting)
# ─────────────────────────────────────────────
def object_velocity_to_joint_velocities(model, data, object_velocity, control_state):
    """
    Map a desired object velocity (6-D) to joint velocities for both arms
    using the Jacobian pseudo-inverse (Z-axis only, for lifting).
    """
    try:
        box_body_id = control_state.get("box_body_id", -1)
        if box_body_id == -1:
            return np.zeros(model.nu)

        num_actuators      = model.nu
        actuators_per_robot = num_actuators // 2

        # find left / right end-effector sites by name
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

        # DOF address for each actuator
        left_arm_dofs  = [model.jnt_dofadr[model.actuator_trnid[i, 0]]
                          for i in range(actuators_per_robot)]
        right_arm_dofs = [model.jnt_dofadr[model.actuator_trnid[i, 0]]
                          for i in range(actuators_per_robot, num_actuators)]

        # full Jacobians (6 × nv)
        jac_left  = np.zeros((6, model.nv))
        jac_right = np.zeros((6, model.nv))
        mujoco.mj_jacSite(model, data, jac_left[:3],  jac_left[3:],  left_ee_site_id)
        mujoco.mj_jacSite(model, data, jac_right[:3], jac_right[3:], right_ee_site_id)

        # extract columns for each arm
        jac_left_arm  = jac_left[:,  left_arm_dofs]
        jac_right_arm = jac_right[:, right_arm_dofs]

        n_dofs = len(left_arm_dofs)

        # use only the Z row for lifting
        J_z = np.zeros((2, n_dofs * 2))
        J_z[0, :n_dofs]   = jac_left_arm[2, :]
        J_z[1,  n_dofs:]  = jac_right_arm[2, :]

        v_z              = np.array([object_velocity[2], object_velocity[2]])
        joint_velocities = np.linalg.pinv(J_z) @ v_z
        joint_velocities *= 2.0
        joint_velocities  = np.clip(joint_velocities, -1.0, 1.0)

        return joint_velocities

    except Exception as e:
        print(f"Error in object_velocity_to_joint_velocities: {e}")
        import traceback; traceback.print_exc()
        return np.zeros(model.nu)


# ─────────────────────────────────────────────
#  MAIN
# ─────────────────────────────────────────────
def main():
    global control_state
    control_state = {}

    xml_file_path = "/home/iitgn-robotics/Saikrishna/samriddhi_mujoco_bimanual_ds_dynamic_task/robot_description/kshitij_lifting.xml"
    model = mujoco.MjModel.from_xml_path(xml_file_path)
    data  = mujoco.MjData(model)

    # disable gravity initially so arms can reach & grasp before the box falls
    model.opt.gravity[:] = [0.0, 0.0, 0.0]
    original_gravity      = np.array([0.0, 0.0, -G])

    mujoco.mj_resetData(model, data)
    mujoco.mj_forward(model, data)

    # find the box body
    box_body_id = next((i for i in range(model.nbody)
                        if "box" in model.body(i).name), -1)
    if box_body_id == -1:
        print("❌  Could not find a body containing 'box' in the XML.")
        return

    initial_box_height = data.xpos[box_body_id][2]
    target_lift_height = 0.3   # metres

    # ── controller gains ──────────────────────────────────────────────────────
    kd_gains = np.array([
        60.0, 100.0, 70.0, 30.0, 15.0, 15.0,   # left arm
        60.0, 100.0, 70.0, 30.0, 15.0, 15.0,   # right arm
    ])
    ki_gains = np.array([
        0.01, 0.1, 0.01, 0.005, 0.002, 0.002,
        0.01, 0.1, 0.01, 0.005, 0.002, 0.002,
    ])
    controller          = VelocityControllerGC(model, data, kd=kd_gains, ki=ki_gains)
    num_actuators       = controller.num_actuators
    actuators_per_robot = num_actuators // 2

    # ── target joint positions ────────────────────────────────────────────────
    first_target_positions                         = np.zeros(num_actuators)
    first_target_positions[:actuators_per_robot]   = [-0.0805, 1.07, -0.126,  1.53,  -0.00978, 0]
    first_target_positions[actuators_per_robot:]   = [ 0.105,  1.07, -0.126, -1.45,  -0.00901, 0]

    second_target_positions                        = np.zeros(num_actuators)
    second_target_positions[:actuators_per_robot]  = [ 0.147,  1.16, -0.0314,  1.79,  -0.00978, 0]
    second_target_positions[actuators_per_robot:]  = [-0.147,  1.16, -0.0314, -1.79,  -0.0314,  0]

    # ── position control gains ────────────────────────────────────────────────
    kp_gains = np.array([
        15.0, 60.0, 15.0, 5.0, 3.0, 3.0,
        15.0, 60.0, 15.0, 5.0, 3.0, 3.0,
    ])
    max_velocities = np.array([
        1.2, 1.2, 1.2, 1.5, 1.5, 1.5,
        1.2, 1.2, 1.2, 1.5, 1.5, 1.5,
    ])
    position_thresholds = np.array([
        0.02, 0.02, 0.02, 0.03, 0.03, 0.03,
        0.02, 0.02, 0.02, 0.03, 0.03, 0.03,
    ])

    # ── shared state dict ─────────────────────────────────────────────────────
    control_state.update({
        "phase":               "reaching_first_target",
        "current_targets":     first_target_positions.copy(),
        "phase_start_time":    time.time(),
        "last_phase_change":   time.time(),
        "last_print_time":     time.time(),
        "initial_box_height":  initial_box_height,
        "target_lift_height":  target_lift_height,
        "box_body_id":         box_body_id,
        "original_gravity":    original_gravity,
        # throwing
        "throw_phase_started":    False,
        "throw_velocity_applied": False,
        "throw_landed":           False,
        "throw_start_pos":        None,
        "throw_release_velocity": np.zeros(3),
        "landing_point":          LANDING_POINT,
        "throw_angle":            THROW_ANGLE,
    })

    # ─────────────────────────────────────────────────────────────────────────
    #  TRAJECTORY / CONTROL FUNCTION  (called every sim step)
    # ─────────────────────────────────────────────────────────────────────────
    def position_to_velocity_trajectory(t):

        # read current joint positions
        joint_positions = np.zeros(num_actuators)
        for i in range(num_actuators):
            jid = model.actuator_trnid[i, 0]
            joint_positions[i] = data.qpos[model.jnt_qposadr[jid]]

        current_time = time.time()

        # periodic phase print
        if current_time - control_state.get("last_print_time", 0) > 1.0:
            print(f"📍 Phase: {control_state['phase']}")
            control_state["last_print_time"] = current_time

        phase = control_state["phase"]

        # ── PHASE 1 & 2 : move arms to grasp position ────────────────────────
        if phase in ("reaching_first_target", "reaching_second_target"):
            position_errors   = control_state["current_targets"] - joint_positions
            velocity_commands = kp_gains * position_errors
            velocity_commands = np.clip(velocity_commands, -max_velocities, max_velocities)

            # dead-band
            for i in range(num_actuators):
                if abs(position_errors[i]) < position_thresholds[i] * 0.5:
                    velocity_commands[i] = 0.0

            all_reached = all(abs(position_errors[i]) <= position_thresholds[i]
                              for i in range(num_actuators))

            if all_reached:
                if phase == "reaching_first_target":
                    print("✅  First target reached — moving to grasp position.")
                    control_state["phase"]           = "reaching_second_target"
                    control_state["current_targets"] = second_target_positions.copy()
                    control_state["last_phase_change"] = current_time
                    return np.zeros(num_actuators)

                elif phase == "reaching_second_target":
                    if current_time - control_state["last_phase_change"] > 0.1:
                        print("✅  Grasp position reached — reading sensors.")
                        control_state["phase"]             = "reading_sensors"
                        control_state["last_phase_change"] = current_time
                        return np.zeros(num_actuators)

            return velocity_commands

        # ── PHASE 3 : short pause to let contacts settle ──────────────────────
        elif phase == "reading_sensors":
            if current_time - control_state["last_phase_change"] > 0.1:
                print("✅  Sensors ready — starting lift.")
                control_state["phase"]             = "lifting_object"
                control_state["last_phase_change"] = current_time
            return np.zeros(num_actuators)

        # ── PHASE 4 : lift the box using admittance control ───────────────────
        elif phase == "lifting_object":
            box_body_id       = control_state["box_body_id"]
            current_box_height = data.xpos[box_body_id][2]
            height_threshold  = 0.01

            # ── lift target reached → prepare throw ──────────────────────────
            if current_box_height >= control_state["target_lift_height"] - height_threshold:

                if not control_state.get("throw_phase_started", False):
                    prel  = data.xpos[box_body_id].copy()
                    pland = control_state["landing_point"]
                    theta = control_state["throw_angle"]

                    try:
                        v_release = compute_release_velocity(prel, pland, theta, g=G)
                    except ValueError as e:
                        print(f"❌  compute_release_velocity failed: {e}")
                        return np.zeros(num_actuators)

                    control_state["throw_release_velocity"] = v_release
                    control_state["throw_phase_started"]    = True
                    control_state["throw_velocity_applied"] = False
                    control_state["throw_landed"]           = False

                    # re-enable gravity for realistic ballistic flight
                    model.opt.gravity[:] = original_gravity

                    print(f"\n🎯  Lift target reached at z={current_box_height:.3f} m")
                    print(f"    Release pos  : {prel}")
                    print(f"    Target land  : {pland}")
                    print(f"    v_release    : [{v_release[0]:.3f}, "
                          f"{v_release[1]:.3f}, {v_release[2]:.3f}] m/s\n")

                    control_state["phase"]             = "throwing_object"
                    control_state["last_phase_change"] = current_time

                return np.zeros(num_actuators)

            # ── still lifting ─────────────────────────────────────────────────
            if "lift_debug_counter" not in control_state:
                control_state["lift_debug_counter"] = 0
            control_state["lift_debug_counter"] += 1
            if control_state["lift_debug_counter"] % 100 == 0:
                print(f"🔧  LIFTING  height={current_box_height:.3f} m  "
                      f"target={control_state['target_lift_height']:.3f} m")

            left_wrench, right_wrench, contact_detected = \
                get_contact_wrenches(model, data, control_state)

            if contact_detected:
                object_velocity, *_ = compute_object_admittance_control(
                    model, data,
                    left_wrench, right_wrench,
                    control_state,
                    desired_height=control_state["target_lift_height"],
                )
                return object_velocity_to_joint_velocities(
                    model, data, object_velocity, control_state)
            else:
                return np.zeros(num_actuators)

        # ── PHASE 5 : throw ───────────────────────────────────────────────────
        elif phase == "throwing_object":
            box_body_id  = control_state["box_body_id"]
            box_dof_adr  = model.body_dofadr[box_body_id]

            # ── inject velocity ONCE, immediately when phase starts ───────────
            if not control_state.get("throw_velocity_applied", False):
                control_state["throw_velocity_applied"] = True

                v_release = control_state["throw_release_velocity"]

                # directly write to the free-joint DOFs
                data.qvel[box_dof_adr + 0] = v_release[0]   # vx
                data.qvel[box_dof_adr + 1] = v_release[1]   # vy
                data.qvel[box_dof_adr + 2] = v_release[2]   # vz (must be positive!)
                data.qvel[box_dof_adr + 3] = 0.0             # no angular spin
                data.qvel[box_dof_adr + 4] = 0.0
                data.qvel[box_dof_adr + 5] = 0.0

                # disable robot collision geometry so box flies free
                for i in range(model.ngeom):
                    gname = model.geom(i).name
                    if (gname and
                            "box"    not in gname.lower() and
                            "ground" not in gname.lower() and
                            "floor"  not in gname.lower()):
                        model.geom_contype[i]    = 0
                        model.geom_conaffinity[i] = 0

                control_state["throw_start_pos"] = data.xpos[box_body_id].copy()

                print(f"\n🚀  VELOCITY INJECTED!")
                print(f"    vx={v_release[0]:.3f}  "
                      f"vy={v_release[1]:.3f}  "
                      f"vz={v_release[2]:.3f}  m/s")
                print(f"    Release pos : {data.xpos[box_body_id]}")
                print(f"    Target      : {LANDING_POINT}\n")

            # ── monitor flight ────────────────────────────────────────────────
            box_pos = data.xpos[box_body_id]
            box_vel = data.qvel[box_dof_adr : box_dof_adr + 3]
            speed   = np.linalg.norm(box_vel)

            if (control_state.get("throw_velocity_applied") and
                    speed < 0.05 and box_pos[2] < 0.15 and
                    not control_state.get("throw_landed", False)):

                control_state["throw_landed"] = True
                start = control_state["throw_start_pos"]
                dist  = np.linalg.norm(box_pos[:2] - start[:2])

                print(f"\n🎯  LANDED!")
                print(f"    Horizontal distance travelled : {dist:.3f} m")
                print(f"    Start  : [{start[0]:.3f}, {start[1]:.3f}, {start[2]:.3f}]")
                print(f"    End    : [{box_pos[0]:.3f}, {box_pos[1]:.3f}, {box_pos[2]:.3f}]")
                print(f"    Target : [{LANDING_POINT[0]:.3f}, "
                      f"{LANDING_POINT[1]:.3f}, {LANDING_POINT[2]:.3f}]")
                error = np.linalg.norm(box_pos[:2] - LANDING_POINT[:2])
                print(f"    Landing error (2-D) : {error:.3f} m\n")

            # arms stay passive after release
            return np.zeros(num_actuators)

        # ── unknown phase ─────────────────────────────────────────────────────
        else:
            return np.zeros(num_actuators)

    # ─────────────────────────────────────────────────────────────────────────
    #  LAUNCH SIMULATION
    # ─────────────────────────────────────────────────────────────────────────
    controller.set_velocity_trajectory(position_to_velocity_trajectory)
    mujoco.set_mjcb_control(controller.control_callback)

    with mujoco.viewer.launch_passive(model, data) as viewer:
        while viewer.is_running():
            mujoco.mj_step(model, data)

            # live box status print
            box_body_id = control_state.get("box_body_id", -1)
            if box_body_id != -1:
                pos = data.xpos[box_body_id].copy()
                dof = model.body_dofadr[box_body_id]
                vel = data.qvel[dof : dof + 3].copy()
                print(f"box pos: [{pos[0]:.3f}, {pos[1]:.3f}, {pos[2]:.3f}],  "
                      f"vel: [{vel[0]:.3f}, {vel[1]:.3f}, {vel[2]:.3f}]")

            viewer.sync()
            time.sleep(0.001)


if __name__ == "__main__":
    main()