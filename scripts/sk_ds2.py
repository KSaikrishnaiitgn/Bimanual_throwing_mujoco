import os
import sys
os.environ["MUJOCO_GL"] = "glfw"
import mujoco
import mujoco.viewer
import numpy as np
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# ─────────────────────────────────────────────
#  GLOBAL CONSTANTS
# ─────────────────────────────────────────────
G             = 9.81
LANDING_POINT = np.array([0, -0.8, 0.0])
THROW_ANGLE   = np.deg2rad(60.0)

# Release point = where the box actually is after grasping (from logs)
RELEASE_POINT = np.array([0.0, -0.413, 0.110])

# Throw timeout
ACCELERATION_DURATION = 0.68   # safety timeout (s)
MAX_ARM_VEL           = 10.0

# DS gains
K_DS = 80.0
B_DS = 12.0

# KEY FIX: Jacobian pseudoinverse regularisation
# Old value 0.01 was too small → Jacobian saturated on Z axis, starved Y
# Higher lambda distributes effort more evenly across all joints/axes
LAMBDA_REG = 0.1

# Arms lead the box by this scale factor
CMD_SCALE = 1.2

# Release thresholds
VEL_MATCH_THRESH = 0.25   # m/s
POS_MATCH_THRESH = 0.12   # m

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

    v_mag = np.sqrt(g * R ** 2 / denom)
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
        self.K_ds       = K_ds
        self.B_ds       = B_ds
        self.vel        = np.zeros(3)

    def update(self, current_box_vel, dt):
        error     = self.target_vel - current_box_vel
        accel     = self.K_ds * error - self.B_ds * self.vel
        accel_mag = np.linalg.norm(accel)
        if accel_mag > 200.0:
            accel = accel / accel_mag * 200.0
        self.vel += accel * dt
        # Clamp: never exceed 110% of target magnitude
        target_mag  = np.linalg.norm(self.target_vel)
        current_mag = np.linalg.norm(self.vel)
        if current_mag > target_mag * 1.1:
            self.vel = self.target_vel * 1.1
        return self.vel.copy()


# ─────────────────────────────────────────────
#  JACOBIAN VELOCITY MAPPING
# ─────────────────────────────────────────────
def compute_throw_joint_velocities(model, data, desired_ee_velocity,
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
            print("⚠️  EE sites not found.")
            return np.zeros(num_actuators)

        left_arm_dofs  = [model.jnt_dofadr[model.actuator_trnid[i, 0]]
                          for i in range(actuators_per_robot)]
        right_arm_dofs = [model.jnt_dofadr[model.actuator_trnid[i, 0]]
                          for i in range(actuators_per_robot, num_actuators)]

        jac_left  = np.zeros((6, model.nv))
        jac_right = np.zeros((6, model.nv))
        mujoco.mj_jacSite(model, data, jac_left[:3],  jac_left[3:],  left_ee_site_id)
        mujoco.mj_jacSite(model, data, jac_right[:3], jac_right[3:], right_ee_site_id)

        # Translational Jacobian rows only
        Jl = jac_left[:3,  left_arm_dofs]   # 3 x n_arm
        Jr = jac_right[:3, right_arm_dofs]  # 3 x n_arm

        # Damped least-squares pseudoinverse with fixed LAMBDA_REG
        lI = LAMBDA_REG * np.eye(3)
        qdot_left  = Jl.T @ np.linalg.solve(Jl @ Jl.T + lI, desired_ee_velocity)
        qdot_right = Jr.T @ np.linalg.solve(Jr @ Jr.T + lI, desired_ee_velocity)

        throw_velocity = np.zeros(num_actuators)
        throw_velocity[:actuators_per_robot] = qdot_left
        throw_velocity[actuators_per_robot:] = qdot_right

        throw_velocity = np.clip(throw_velocity, -MAX_ARM_VEL, MAX_ARM_VEL)
        return throw_velocity

    except Exception as e:
        print(f"Jacobian error: {e}")
        return np.zeros(num_actuators)


# ─────────────────────────────────────────────
#  MAIN
# ─────────────────────────────────────────────
def main():
    control_state = {}

    xml_file_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "robot_description", "kshitij_lifting.xml")
    model = mujoco.MjModel.from_xml_path(xml_file_path)
    data  = mujoco.MjData(model)

    original_gravity = np.array([0.0, 0.0, -G])
    model.opt.gravity[:] = [0.0, 0.0, 0.0]

    mujoco.mj_resetData(model, data)
    mujoco.mj_forward(model, data)

    box_body_id = next((i for i in range(model.nbody)
                        if "box" in model.body(i).name), -1)
    if box_body_id == -1:
        print("❌  Cannot find box body.")
        return

    # Pre-compute release velocity
    print(f"\n📐 Pre-computing release velocity:")
    print(f"    Release point : {RELEASE_POINT}")
    print(f"    Landing point : {LANDING_POINT}")
    print(f"    Throw angle   : {np.rad2deg(THROW_ANGLE):.1f}°")
    v_release = compute_release_velocity(RELEASE_POINT, LANDING_POINT, THROW_ANGLE, G)
    print(f"    Release velocity : {np.round(v_release, 4)}")
    print(f"    Speed            : {np.linalg.norm(v_release):.3f} m/s")
    print(f"    vY={v_release[1]:.3f}  vZ={v_release[2]:.3f}\n")

    # Controller
    kd_gains = np.array([60.0, 100.0, 70.0, 30.0, 15.0, 15.0,
                         60.0, 100.0, 70.0, 30.0, 15.0, 15.0])
    ki_gains = np.array([0.01, 0.1, 0.01, 0.005, 0.002, 0.002,
                         0.01, 0.1, 0.01, 0.005, 0.002, 0.002])

    controller = VelocityControllerGC(model, data, kd=kd_gains, ki=ki_gains)
    num_actuators       = controller.num_actuators
    actuators_per_robot = num_actuators // 2

    first_target = np.zeros(num_actuators)
    first_target[:actuators_per_robot] = [-0.0805, 1.07, -0.126,  1.53, -0.00978, 0]
    first_target[actuators_per_robot:] = [ 0.105,  1.07, -0.126, -1.45, -0.00901, 0]

    second_target = np.zeros(num_actuators)
    second_target[:actuators_per_robot] = [ 0.147, 1.16, -0.0314,  1.79, -0.00978, 0.5]
    second_target[actuators_per_robot:] = [-0.147, 1.16, -0.0314, -1.79, -0.0314,  0.5]

    kp_gains  = np.array([15.0, 60.0, 15.0, 5.0, 3.0, 5.0,
                          15.0, 60.0, 15.0, 5.0, 3.0, 5.0])
    max_vels  = np.array([1.2, 1.2, 1.2, 1.5, 1.5, 2.0,
                          1.2, 1.2, 1.2, 1.5, 1.5, 2.0])
    pos_thresh = np.array([0.02, 0.02, 0.02, 0.03, 0.03, 0.1,
                           0.02, 0.02, 0.02, 0.03, 0.03, 0.1])

    control_state.update({
        "phase":                   "reaching_first_target",
        "current_targets":         first_target.copy(),
        "last_phase_change":       time.time(),
        "box_body_id":             box_body_id,
        "original_gravity":        original_gravity,
        "release_velocity":        v_release,
        "acceleration_start_time": None,
        "acceleration_dynamics":   None,
        "released":                False,
        "last_sim_time":           None,
    })

    def position_to_velocity_trajectory(t):
        joint_positions = np.zeros(num_actuators)
        for i in range(num_actuators):
            jid = model.actuator_trnid[i, 0]
            joint_positions[i] = data.qpos[model.jnt_qposadr[jid]]

        current_time = time.time()
        phase        = control_state["phase"]

        # ── Phases 1 & 2: position control to grasp ────────────────────────
        if phase in ("reaching_first_target", "reaching_second_target"):
            errors = control_state["current_targets"] - joint_positions
            cmds   = kp_gains * errors
            cmds   = np.clip(cmds, -max_vels, max_vels)
            for i in range(num_actuators):
                if abs(errors[i]) < pos_thresh[i] * 0.5:
                    cmds[i] = 0.0

            all_reached = all(abs(errors[i]) <= pos_thresh[i]
                              for i in range(num_actuators))

            if all_reached:
                if phase == "reaching_first_target":
                    print("✅ First target reached — moving to grasp position.")
                    control_state["phase"]             = "reaching_second_target"
                    control_state["current_targets"]   = second_target.copy()
                    control_state["last_phase_change"] = current_time
                    return np.zeros(num_actuators)

                elif phase == "reaching_second_target":
                    if current_time - control_state["last_phase_change"] > 0.1:
                        box_pos_now = data.xpos[box_body_id].copy()
                        print("✅ Grasp position reached — box grasped!")
                        print(f"   Actual box pos at grasp: {np.round(box_pos_now, 4)}")
                        print(f"   RELEASE_POINT set to   : {RELEASE_POINT}")
                        print("🎯 Starting throw phase...\n")
                        print(f"📊 Release plan:")
                        print(f"   velocity target : {np.round(control_state['release_velocity'], 4)}")
                        print(f"   speed           : {np.linalg.norm(control_state['release_velocity']):.2f} m/s")
                        print(f"   vY={control_state['release_velocity'][1]:.3f}  "
                              f"vZ={control_state['release_velocity'][2]:.3f}")
                        print(f"   timeout after   : {ACCELERATION_DURATION}s\n")

                        dynamics = SecondOrderDynamics(
                            control_state["release_velocity"], K_DS, B_DS)
                        control_state["acceleration_dynamics"]   = dynamics
                        control_state["acceleration_start_time"] = data.time
                        control_state["last_sim_time"]           = data.time
                        control_state["released"]                = False
                        control_state["phase"]                   = "accelerating"
                        model.opt.gravity[:] = [0.0, 0.0, 0.0]
                        return np.zeros(num_actuators)
            return cmds

        # ── Phase 3: throw — DS → Jacobian → joint velocities ──────────────
        elif phase == "accelerating":
            if control_state["released"]:
                return np.zeros(num_actuators)

            sim_time = data.time
            elapsed  = sim_time - control_state["acceleration_start_time"]
            dt       = max(0.001, sim_time - control_state["last_sim_time"])
            control_state["last_sim_time"] = sim_time

            box_dof_adr     = model.body_dofadr[box_body_id]
            current_box_vel = data.qvel[box_dof_adr:box_dof_adr + 3].copy()
            current_box_pos = data.xpos[box_body_id].copy()

            # DS computes desired EE velocity
            dynamics    = control_state["acceleration_dynamics"]
            desired_vel = dynamics.update(current_box_vel, dt)
            cmd_vel     = desired_vel * CMD_SCALE

            target_mag  = np.linalg.norm(control_state["release_velocity"])
            current_mag = np.linalg.norm(current_box_vel)
            vel_err     = np.linalg.norm(current_box_vel - control_state["release_velocity"])

            if int(elapsed * 20) % 4 == 0 and elapsed > 0.01:
                print(f"   t={elapsed:.3f}s | "
                      f"box_vel=[{current_box_vel[0]:.2f}, {current_box_vel[1]:.2f}, {current_box_vel[2]:.2f}] | "
                      f"speed={current_mag:.2f}/{target_mag:.2f} | "
                      f"vel_err={vel_err:.3f}")

            # Release condition
            vel_ok  = vel_err < VEL_MATCH_THRESH
            timeout = elapsed >= ACCELERATION_DURATION

            if vel_ok or timeout:
                control_state["released"] = True
                model.opt.gravity[:]      = control_state["original_gravity"]

                tag = "⚠️  Timeout" if timeout else "✅ Velocity matched"
                print(f"\n{tag} — RELEASED at t={elapsed:.3f}s!")
                print(f"   pos (actual)  : {np.round(current_box_pos, 4)}")
                print(f"   vel (actual)  : {np.round(current_box_vel, 4)}")
                print(f"   vel (target)  : {np.round(control_state['release_velocity'], 4)}")
                print(f"   vel_err={vel_err:.3f} m/s\n")
                return np.zeros(num_actuators)

            # Jacobian maps desired EE velocity → joint velocities
            return compute_throw_joint_velocities(
                model, data, cmd_vel,
                actuators_per_robot, num_actuators)

        return np.zeros(num_actuators)

    controller.set_velocity_trajectory(position_to_velocity_trajectory)
    mujoco.set_mjcb_control(controller.control_callback)

    with mujoco.viewer.launch_passive(model, data) as viewer:
        last_print_time = 0.0
        while viewer.is_running():
            mujoco.mj_step(model, data)
            current_time = time.time()
            if current_time - last_print_time > 0.1:
                bid = control_state.get("box_body_id", -1)
                if bid != -1:
                    pos = data.xpos[bid].copy()
                    dof = model.body_dofadr[bid]
                    vel = data.qvel[dof:dof + 3].copy()
                    print(f"box pos: [{pos[0]:.3f}, {pos[1]:.3f}, {pos[2]:.3f}]  "
                          f"vel: [{vel[0]:.3f}, {vel[1]:.3f}, {vel[2]:.3f}]")
                last_print_time = current_time
            viewer.sync()
            time.sleep(0.001)


if __name__ == "__main__":
    main()