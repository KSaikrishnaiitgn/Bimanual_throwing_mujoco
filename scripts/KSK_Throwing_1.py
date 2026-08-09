"""
Throwing with Grasp Admittance (force closure) + DS + Impedance Control

  Grasp / force closure:
      ẋ_ee = K_adm⁻¹ (F* − F_meas)     [per hand, along grasp axis]

  Step 1 — Release Velocity (projectile math), computed ONCE for the fixed,
           user-defined release point:
      R        = ||Δx_y||,   Δx_y = (RELEASE_POINT_land - RELEASE_POINT)[:2]
      v_rel²   = g R² / [2 cos²θ (R tanθ − Δz)]
      v⃗_rel   = v_rel cosθ ê_h  +  v_rel sinθ ẑ

  Step 2 — Modified DS for throwing:
      ẍ_des = −K_ds (x_o − x_rel)  −  B_ds (ẋ_o − v⃗_rel)

  Step 3 — Object-level impedance control (secondary correction only):
      ẋ_o* = ẋ_o + D⁻¹ [ M_o ẍ_des + w_mo^fb − ŵ_obj + K₁(x_rel − x_o) ]

  Step 4 — Dual-arm joint velocity, combining BOTH velocity streams:
      q̇ = J_H† [ ẋ_ee(grasp admittance) + G^T ẋ_o*(impedance) ]

  ── CHANGES IN THIS VERSION ──────────────────────────────────────────────
  1. FIX (crash): throw-phase debug print no longer references the locals
     f_left/f_right — reads self._last_grasp_force instead (see prior diff).
  2. FIX (misleading plot): the previously-logged "commanded EE velocity"
     was only the grasp-admittance squeeze term (x_dot_ee), NOT the full
     Jacobian target (x_dot_ee + G^T @ xdot_o_star). That's why it hugged
     ~0 regardless of throw speed. Now logs BOTH the squeeze component and
     the object-translation command (xdot_o_star, already available as
     commanded_vel) so the EE-velocity plot compares against the right
     reference.
  3. NEW — plot_release_diagnostics(): pos_error(t) vs its tolerance, and
     current_speed(t) vs the velocity-threshold speed, with the release
     instant marked. Shows directly whether/when the "near x_rel AND near
     v_rel" release condition is even reachable given the dynamics.
  4. NEW — plot_ds_ceiling_analysis(): open-loop simulation of the DS +
     impedance law alone (perfect tracking assumed, no robot/Jacobian/
     joint-limit effects) starting from the box's actual swing-start
     position. Shows the velocity ceiling the CONTROL LAW allows, isolated
     from execution — separates "the math can't do this" from "the robot
     isn't tracking what the math asked for".
"""

import os
os.environ["MUJOCO_GL"] = "glfw"

import mujoco
import mujoco.viewer
import numpy as np
import time
import matplotlib.pyplot as plt

# ── existing modular controllers
from config import throwing_config_1 as config
from controllers.contact_handler import ContactHandler
from controllers.admittance_controller import AdmittanceController
from controllers.trajectory_planner import TrajectoryPlanner
from controllers.jacobian_mapper import JacobianMapper
from controllers.dual_arm_jacobian import DualArmJacobian
from controllers.phase_controller import PhaseController
from controllers.KSK_Admittance_controller import AdmittanceGraspController
from utils.mujoco_velocity_controller.KSK_velocity_controller import VelocityControllerGC

G = config.GRAVITY


# ══════════════════════════════════════════════════════════════════════════════
#  MODIFIED DS — ẍ_des = −K_ds (x_o − x_rel) − B_ds (ẋ_o − v⃗_rel)
# ══════════════════════════════════════════════════════════════════════════════
class ThrowingDS:
    def __init__(self, K_ds: np.ndarray, B_ds: np.ndarray, max_vel: float = 8.0):
        self.K_ds   = K_ds
        self.B_ds   = B_ds
        self.max_vel = max_vel

    def compute_acceleration(self,
                             x_o:   np.ndarray,
                             xdot_o: np.ndarray,
                             x_rel:  np.ndarray,
                             v_rel:  np.ndarray
                             ) -> np.ndarray:
        pos_err = x_o   - x_rel
        vel_err = xdot_o - v_rel
        xddot_des = -self.K_ds @ pos_err - self.B_ds @ vel_err
        return xddot_des


# ══════════════════════════════════════════════════════════════════════════════
#  IMPEDANCE CONTROL — secondary correction on top of the DS
#  ẋ_o* = ẋ_o + D⁻¹ [ M_o ẍ_des + w_mo^fb − ŵ_obj + K₁(x_o* − x_o) ]
# ══════════════════════════════════════════════════════════════════════════════
class ThrowingImpedance:
    def __init__(self, M_o: np.ndarray, D: np.ndarray, K_1: np.ndarray,
                 max_vel: float = 5.0):
        self.M_o     = M_o
        self.D_inv   = np.linalg.inv(D)
        self.K_1     = K_1
        self.max_vel = max_vel

    def gravity_wrench(self, object_mass: float) -> np.ndarray:
        return np.array([0.0, 0.0, -object_mass * G])

    def compute_object_velocity(self,
                                xdot_o:    np.ndarray,
                                xddot_des: np.ndarray,
                                w_mo_fb:   np.ndarray,
                                w_obj_hat: np.ndarray,
                                x_o_star:  np.ndarray,
                                x_o:       np.ndarray
                                ) -> np.ndarray:
        inertial_term   = self.M_o @ xddot_des
        stiffness_term  = self.K_1 @ (x_o_star - x_o)
        bracket         = inertial_term + w_mo_fb - w_obj_hat + stiffness_term

        xdot_o_star = xdot_o + self.D_inv @ bracket
        xdot_o_star = np.clip(xdot_o_star, -self.max_vel, self.max_vel)
        return xdot_o_star


# ══════════════════════════════════════════════════════════════════════════════
#  MAIN SIMULATION
# ══════════════════════════════════════════════════════════════════════════════
class ThrowingDSImpedance:
    """
    Full pipeline per simulation step (throwing phase):
        0. AdmittanceGraspController        → ẋ_ee   (force-closure squeeze)
        1. compute_release_velocity (once)  → v⃗_rel  (for fixed RELEASE_POINT)
        2. ThrowingDS.compute_acceleration  → ẍ_des
        3. ContactHandler wrench            → w_mo^fb
        4. ThrowingImpedance                → ẋ_o*
        5. DualArmJacobian                  → q̇ = J_H† [ẋ_ee + G^T ẋ_o*]
        6. Joint-velocity clamp             → q̇ respects config.MAX_JOINT_VELOCITIES
    """

    def __init__(self):
        self.model = mujoco.MjModel.from_xml_path("/home/iitgn-robotics/Saikrishna/Bimanual_throwing_mujoco/robot_description/kshitij_lifting.xml")
        self.data  = mujoco.MjData(self.model)

        self.model.opt.gravity[:] = [0.0, 0.0, 0.0]
        self.original_gravity     = np.array([0.0, 0.0, -G])

        mujoco.mj_resetData(self.model, self.data)
        mujoco.mj_forward(self.model, self.data)

        # ── find box ──────────────────────────────────────────────────────────
        self.box_id = next(
            (i for i in range(self.model.nbody)
             if "box" in self.model.body(i).name), -1)
        if self.box_id == -1:
            raise RuntimeError("No body containing 'box' found in the XML.")

        self.object_mass = self.model.body_mass[self.box_id]

        # ── joint velocity limits (rad/s, per actuator, both arms) ────────────
        self.joint_vel_limits = np.asarray(config.MAX_JOINT_VELOCITIES, dtype=float)

        # ── velocity controller ────────────────────────────────────────────────
        self.vel_ctrl = VelocityControllerGC(
            self.model, self.data,
            kd=config.KD_GAINS, ki=config.KI_GAINS,
            joint_vel_limits=self.joint_vel_limits)
        self.n_act    = self.vel_ctrl.num_actuators
        self.n_per    = self.n_act // 2

        if self.joint_vel_limits.shape[0] != self.n_act:
            raise ValueError(
                f"config.MAX_JOINT_VELOCITIES has "
                f"{self.joint_vel_limits.shape[0]} entries but the model has "
                f"{self.n_act} actuators — these must match.")

        # ── sub-controllers ───────────────────────────────────────────────────
        self.contact_handler = ContactHandler(self.model, self.data)
        self.admittance_ctrl = AdmittanceController(
            config.ADMITTANCE_STIFFNESS, config.ADMITTANCE_DAMPING,
            config.MAX_LINEAR_VELOCITY, config.MAX_ANGULAR_VELOCITY,
            config.UPWARD_FORCE_BASE,   config.UPWARD_FORCE_SCALE)
        self.trajectory_planner = TrajectoryPlanner(G)
        self.jacobian_mapper    = JacobianMapper(self.model, self.n_act)
        self.dual_arm_jac       = DualArmJacobian(self.model, self.n_act)
        self.phase_ctrl         = PhaseController(config)
        self.phase_ctrl.initialize_targets(self.n_act)

        # ── grasp admittance (force closure) ───────────────────────────────────
        self.grasp_ctrl = AdmittanceGraspController(
            K_adm         = config.K_ADM,
            desired_force = config.DESIRED_GRASP_FORCE,
            max_vel       = config.MAX_GRASP_VELOCITY)

        # ── throwing DS (Step 2) ────────────────────────────────────────────────
        self.throwing_ds = ThrowingDS(
            K_ds   = config.K_DS,
            B_ds   = config.B_DS,
            max_vel= config.MAX_OBJ_VEL)

        # ── impedance controller (Step 3–4) ──────────────────────────────────
        self.impedance_ctrl = ThrowingImpedance(
            M_o    = config.M_O,
            D      = config.D_IMPEDANCE,
            K_1    = config.K_IMPEDANCE,
            max_vel= config.MAX_COMMANDED_VEL)

        # ── find EE sites ─────────────────────────────────────────────────────
        self.left_ee_id, self.right_ee_id = \
            self.contact_handler.find_end_effector_sites()

        # ── fixed, user-defined release point/velocity ───────────────────────
        self.x_rel = np.asarray(config.RELEASE_POINT, dtype=float).copy()
        v_scale = getattr(config, "VELOCITY_SCALE", 1.0)
        self.v_rel = self.trajectory_planner.compute_release_velocity(
            self.x_rel, config.LANDING_POINT, config.THROW_ANGLE) * v_scale
        self._throw_dir = self.v_rel / (np.linalg.norm(self.v_rel) + 1e-9)

        print(f"  [config] RELEASE_POINT = {self.x_rel}")
        print(f"  [config] LANDING_POINT = {config.LANDING_POINT}")
        print(f"  [config] v_rel (computed) = {self.v_rel}")

        # ── throw state ───────────────────────────────────────────────────────
        self.release_position_tolerance = getattr(
            config, "RELEASE_POSITION_TOLERANCE", 0.05)
        self.min_swing_time = getattr(config, "MIN_SWING_TIME", 0.15)
        self.min_release_speed = getattr(config, "MIN_RELEASE_SPEED", 0.15)

        # ── grasp force-closure deactivation near release ──────────────────────
        self.grasp_deactivation_tolerance = getattr(
            config, "GRASP_DEACTIVATION_TOLERANCE",
            self.release_position_tolerance * 2.0)
        self.grasp_ramp_down_time = getattr(config, "GRASP_RAMP_DOWN_TIME", 0.05)

        # ── data logger ───────────────────────────────────────────────────────
        self.log = {
            "time":            [],
            "achieved_pos":    [],
            "desired_pos":     [],
            "achieved_vel":    [],
            "commanded_vel":   [],
            "phase":           [],
            "joint_vel_cmd":   [],
            "joint_vel_scale": [],
            "grasp_force_left":  [],
            "grasp_force_right": [],
            "ee_left_vel_cmd":     [],   # grasp-squeeze component only
            "ee_right_vel_cmd":    [],   # grasp-squeeze component only
            "ee_left_vel_actual":  [],
            "ee_right_vel_actual": [],
            "grasp_active":        [],   # NEW — True/False each step
        }
        self._last_commanded_vel = np.full(3, np.nan)
        self._last_joint_vel_cmd = np.zeros(self.n_act)
        self._last_joint_vel_scale = 1.0
        self._last_grasp_force = (np.nan, np.nan)
        self._last_ee_vel_cmd = (np.zeros(3), np.zeros(3))
        self.release_log = None
        self.release_ee_log = None
        self._release_sim_time = None
        self._swing_start_box_pos = None   # NEW — captured in _prepare_throw

        print("✅  ThrowingDSImpedance initialised")
        print(f"    box id={self.box_id}  mass={self.object_mass:.3f} kg")
        print(f"    left EE site={self.left_ee_id}  right EE site={self.right_ee_id}")
        print(f"    joint velocity limits (rad/s): {self.joint_vel_limits}")
        print(f"    desired grasp force: {config.DESIRED_GRASP_FORCE} N per hand")

    # ──────────────────────────────────────────────────────────────────────────
    def _get_ee_velocities(self):
        """Actual (achieved) Cartesian EE linear velocities, world frame."""
        left_vel6  = np.zeros(6)
        right_vel6 = np.zeros(6)
        mujoco.mj_objectVelocity(self.model, self.data, mujoco.mjtObj.mjOBJ_SITE,
                                  self.left_ee_id, left_vel6, 0)
        mujoco.mj_objectVelocity(self.model, self.data, mujoco.mjtObj.mjOBJ_SITE,
                                  self.right_ee_id, right_vel6, 0)
        # mj_objectVelocity fills [angular(3), linear(3)]
        return left_vel6[3:6].copy(), right_vel6[3:6].copy()

    # ──────────────────────────────────────────────────────────────────────────
    def compute_velocity_commands(self, t: float) -> np.ndarray:
        joint_positions = self._read_joint_positions()
        phase = self.phase_ctrl.get_phase()

        if self.phase_ctrl.should_print_phase():
            print(f"  Phase: {phase}")

        if phase in ("reaching_first_target", "reaching_second_target"):
            cmds = self.phase_ctrl.compute_position_control(
                joint_positions, self.n_act)
            cmds = self._clip_joint_velocities(cmds)

            if self.phase_ctrl.check_reaching_complete(
                    joint_positions, self.n_act):
                if phase == "reaching_first_target":
                    print("✅  First target reached.")
                    self.phase_ctrl.state["current_targets"] = \
                        self.phase_ctrl.state["second_targets"].copy()
                    self.phase_ctrl.set_phase("reaching_second_target")
                    return np.zeros(self.n_act)
                elif phase == "reaching_second_target":
                    if time.time() - self.phase_ctrl.state["last_phase_change"] > 0.1:
                        print("✅  Grasp position reached.")
                        self.phase_ctrl.set_phase("reading_sensors")
                        return np.zeros(self.n_act)
            return cmds

        elif phase == "reading_sensors":
            if time.time() - self.phase_ctrl.state["last_phase_change"] \
                    > config.READING_SENSORS_DURATION:
                print("✅  Starting lift.")
                self.phase_ctrl.set_phase("lifting_object")
            return np.zeros(self.n_act)

        elif phase == "lifting_object":
            return self._lifting_phase()

        elif phase == "throwing_object":
            return self._throwing_phase()

        return np.zeros(self.n_act)

    # ──────────────────────────────────────────────────────────────────────────
    def _lifting_phase(self) -> np.ndarray:
        current_h = self.data.xpos[self.box_id][2]

        if self.phase_ctrl.check_lift_complete(
                current_h, config.TARGET_LIFT_HEIGHT):
            if not self.phase_ctrl.state.get("throw_phase_started", False):
                self._prepare_throw()
            return np.zeros(self.n_act)

        lw, rw, contact = self.contact_handler.get_object_contact_wrenches(
            self.box_id)
        if contact:
            obj_vel, *_ = self.admittance_ctrl.compute_object_velocity(
                self.model, self.data, lw, rw, self.box_id,
                config.TARGET_LIFT_HEIGHT)
            q_dot = self.jacobian_mapper.object_velocity_to_joint_velocities_lifting(
                self.data, obj_vel,
                self.left_ee_id, self.right_ee_id)
            return self._clip_joint_velocities(q_dot)
        return np.zeros(self.n_act)

    # ──────────────────────────────────────────────────────────────────────────
    def _prepare_throw(self):
        """
        x_rel / v_rel were already computed ONCE in __init__ from the fixed
        config.RELEASE_POINT — nothing to recompute here. This just starts
        the swing timer / phase transition.
        """
        self.phase_ctrl.state["throw_phase_started"] = True
        self.phase_ctrl.state["throw_released"]      = False
        self.phase_ctrl.state["flight_logged"]        = False
        self.phase_ctrl.state["swing_start_sim_time"] = self.data.time
        self.phase_ctrl.state["grasp_active"]          = True
        self.phase_ctrl.state["grasp_deactivate_time"] = None
        self.phase_ctrl.set_phase("throwing_object")

        box_pos_now = self.data.xpos[self.box_id].copy()
        self._swing_start_box_pos = box_pos_now.copy()   # NEW — for ceiling analysis

        print(f"\n  Lift target reached — arm swing starting")
        print(f"  Box pos @ swing start: [{box_pos_now[0]:.3f}, {box_pos_now[1]:.3f}, {box_pos_now[2]:.3f}]")
        print(f"  Release pos (fixed) : {self.x_rel}")
        print(f"  Target land          : {config.LANDING_POINT}")
        v = self.v_rel
        print(f"  v_rel                : [{v[0]:.3f}, {v[1]:.3f}, {v[2]:.3f}] m/s")
        d = self._throw_dir
        print(f"  throw_dir            : [{d[0]:.3f}, {d[1]:.3f}, {d[2]:.3f}]\n")

    # ──────────────────────────────────────────────────────────────────────────
    EE_SWING_SPEED     = 8.0   # m/s — cap on commanded object speed (Cartesian)
    VELOCITY_THRESHOLD = 0.88  # release when box reaches this fraction of target speed
    MAX_SWING_TIME     = 2.0   # s   — hard timeout fallback

    # ──────────────────────────────────────────────────────────────────────────
    def _clip_joint_velocities(self, q_dot: np.ndarray) -> np.ndarray:
        """Direction-preserving scale-down so every joint stays within
        config.MAX_JOINT_VELOCITIES."""
        limits = self.joint_vel_limits
        safe_limits = np.where(limits > 1e-9, limits, 1e-9)
        ratios = np.abs(q_dot) / safe_limits
        max_ratio = float(np.max(ratios)) if ratios.size else 1.0

        if max_ratio > 1.0:
            q_dot = q_dot / max_ratio
            self._last_joint_vel_scale = 1.0 / max_ratio
        else:
            self._last_joint_vel_scale = 1.0

        return q_dot

    # ──────────────────────────────────────────────────────────────────────────
    def _throwing_phase(self) -> np.ndarray:
        """
        Combines TWO velocity streams into the dual-arm Jacobian:

          ẋ_ee   = grasp admittance (force closure)   [Step 0]
          ẋ_o*   = DS + impedance (object translation) [Steps 1-4]
          q̇      = J_H† [ ẋ_ee + G^T ẋ_o* ]            [Step 5]
          q̇      = clip_to_joint_limits(q̇)             [Step 6 — hardware safety]
        """
        if self.phase_ctrl.state["throw_released"]:
            self._monitor_flight()
            return np.zeros(self.n_act)

        _box_dof = self.model.body_dofadr[self.box_id]
        xdot_o   = self.data.qvel[_box_dof:_box_dof + 3].copy()
        x_o      = self.data.xpos[self.box_id].copy()
        elapsed  = self.data.time - self.phase_ctrl.state["swing_start_sim_time"]

        target_speed  = np.linalg.norm(self.v_rel)
        current_speed = np.linalg.norm(xdot_o)
        vel_ratio     = current_speed / target_speed if target_speed > 1e-6 else 0.0
        pos_error     = np.linalg.norm(x_o - self.x_rel)

        release_ready = (
            elapsed >= self.min_swing_time and
            current_speed >= self.min_release_speed and
            (
                (vel_ratio >= self.VELOCITY_THRESHOLD and
                 pos_error <= self.release_position_tolerance) or
                elapsed >= self.MAX_SWING_TIME
            )
        )

        if release_ready:
            reason = "velocity+position threshold" if elapsed < self.MAX_SWING_TIME \
                     else "timeout"
            self._last_commanded_vel = np.full(3, np.nan)
            self._do_release(reason, xdot_o)
            return np.zeros(self.n_act)

        # ── Step 0: grasp admittance (force closure) → ẋ_ee ───────────────────
        f_left, f_right = self._last_grasp_force

        if self.phase_ctrl.state.get("grasp_active", True) and \
                pos_error <= self.grasp_deactivation_tolerance:
            self.phase_ctrl.state["grasp_active"] = False
            self.phase_ctrl.state["grasp_deactivate_time"] = self.data.time
            print(f"\n  Grasp force closure DEACTIVATED at t={self.data.time:.3f}s "
                  f"(pos_err={pos_error:.3f} m <= tol={self.grasp_deactivation_tolerance:.3f} m)\n")

        if self.phase_ctrl.state.get("grasp_active", True):
            left_ee_pos  = self.data.site_xpos[self.left_ee_id].copy()
            right_ee_pos = self.data.site_xpos[self.right_ee_id].copy()
            lw, rw, contact_ok = self.contact_handler.get_object_contact_wrenches(
                self.box_id)
            left_force_meas  = np.asarray(lw[:3]) if contact_ok else np.zeros(3)
            right_force_meas = np.asarray(rw[:3]) if contact_ok else np.zeros(3)

            x_dot_ee, f_left, f_right = self.grasp_ctrl.compute_ee_velocities(
                left_ee_pos, right_ee_pos, left_force_meas, right_force_meas)

            self._last_grasp_force = (f_left, f_right)
        else:
            deactivate_t = self.phase_ctrl.state.get("grasp_deactivate_time")
            if deactivate_t is not None and self.grasp_ramp_down_time > 1e-6:
                elapsed_since_deactivate = self.data.time - deactivate_t
                ramp = max(0.0, 1.0 - elapsed_since_deactivate / self.grasp_ramp_down_time)
            else:
                ramp = 0.0

            if ramp > 0.0:
                left_ee_pos  = self.data.site_xpos[self.left_ee_id].copy()
                right_ee_pos = self.data.site_xpos[self.right_ee_id].copy()
                lw, rw, contact_ok = self.contact_handler.get_object_contact_wrenches(
                    self.box_id)
                left_force_meas  = np.asarray(lw[:3]) if contact_ok else np.zeros(3)
                right_force_meas = np.asarray(rw[:3]) if contact_ok else np.zeros(3)
                x_dot_ee_raw, f_left, f_right = self.grasp_ctrl.compute_ee_velocities(
                    left_ee_pos, right_ee_pos, left_force_meas, right_force_meas)
                x_dot_ee = x_dot_ee_raw * ramp
                f_left, f_right = f_left * ramp, f_right * ramp
                self._last_grasp_force = (f_left, f_right)
            else:
                x_dot_ee = np.zeros(12)
                f_left, f_right = 0.0, 0.0
                self._last_grasp_force = (0.0, 0.0)

        # record grasp-squeeze command component (NOT the full EE target — see plot fix)
        self._last_ee_vel_cmd = (x_dot_ee[0:3].copy(), x_dot_ee[6:9].copy())

        # ── Step 2: DS desired acceleration ────────────────────────────────────
        xddot_des = self.throwing_ds.compute_acceleration(
            x_o, xdot_o, self.x_rel, self.v_rel)

        # ── Step 3: contact feedback / gravity (still zeroed during swing) ───
        w_mo_fb   = np.zeros(3)
        w_obj_hat = np.zeros(3)

        # ── Step 4: impedance → commanded object velocity ẋ_o* ───────────────
        xdot_o_star = self.impedance_ctrl.compute_object_velocity(
            xdot_o    = xdot_o,
            xddot_des = xddot_des,
            w_mo_fb   = w_mo_fb,
            w_obj_hat = w_obj_hat,
            x_o_star  = self.x_rel,
            x_o       = x_o)

        speed = np.linalg.norm(xdot_o_star)
        if speed > self.EE_SWING_SPEED:
            xdot_o_star = xdot_o_star / speed * self.EE_SWING_SPEED

        self._last_commanded_vel = xdot_o_star.copy()

        # ── Step 5: q̇ = J_H† [ ẋ_ee + G^T ẋ_o* ] ─────────────────────────────
        q_dot = self.dual_arm_jac.compute_joint_velocities(
            data             = self.data,
            x_dot_o_star     = xdot_o_star,
            x_dot_ee         = x_dot_ee,
            object_body_id   = self.box_id,
            left_ee_site_id  = self.left_ee_id,
            right_ee_site_id = self.right_ee_id)

        # ── Step 6: hardware safety clamp ─────────────────────────────────────
        q_dot = self._clip_joint_velocities(q_dot)
        self._last_joint_vel_cmd = q_dot.copy()

        if int(elapsed * 10) % 5 == 0:
            scale_note = ""
            if self._last_joint_vel_scale < 0.999:
                scale_note = f"  [joint-limit clamp: {self._last_joint_vel_scale*100:.0f}%]"
            gf_l, gf_r = self._last_grasp_force
            print(f"  swing t={elapsed:.2f}s | box {current_speed:.2f}/{target_speed:.2f} m/s"
                  f" ({vel_ratio*100:.0f}%) | pos_err={pos_error:.3f} m"
                  f" | grasp F: L={gf_l:.2f}N R={gf_r:.2f}N{scale_note}")

        return q_dot

    # ──────────────────────────────────────────────────────────────────────────
    def _do_release(self, reason: str, release_vel: np.ndarray):
        self.phase_ctrl.state["throw_released"] = True

        for i in range(self.model.ngeom):
            gn = self.model.geom(i).name
            if gn and "box" not in gn.lower() \
                   and "ground" not in gn.lower() \
                   and "floor"  not in gn.lower():
                self.model.geom_contype[i]     = 0
                self.model.geom_conaffinity[i] = 0

        self.model.opt.gravity[:] = self.original_gravity

        self._release_sim_time = self.data.time
        self.release_log = (self.v_rel.copy(), release_vel.copy())

        left_ee_actual, right_ee_actual = self._get_ee_velocities()
        self.release_ee_log = (left_ee_actual.copy(), right_ee_actual.copy())

        actual_pos  = self.data.xpos[self.box_id].copy()
        v           = release_vel
        speed_ratio = np.linalg.norm(v) / (np.linalg.norm(self.v_rel) + 1e-9)
        print(f"\n  RELEASED ({reason})")
        print(f"  Release pos : [{actual_pos[0]:.3f}, {actual_pos[1]:.3f}, {actual_pos[2]:.3f}]")
        print(f"  Release vel : [{v[0]:.3f}, {v[1]:.3f}, {v[2]:.3f}] m/s")
        print(f"  Target  vel : [{self.v_rel[0]:.3f}, {self.v_rel[1]:.3f}, {self.v_rel[2]:.3f}] m/s")
        print(f"  Speed ratio : {speed_ratio*100:.1f}%")
        print(f"  EE vel @ release  L=[{left_ee_actual[0]:.3f},{left_ee_actual[1]:.3f},{left_ee_actual[2]:.3f}]"
              f"  R=[{right_ee_actual[0]:.3f},{right_ee_actual[1]:.3f},{right_ee_actual[2]:.3f}] m/s")
        pred = self.trajectory_planner.predict_landing_point(actual_pos, v)
        if pred is not None:
            err = np.linalg.norm(pred[:2] - config.LANDING_POINT[:2])
            print(f"  Predicted landing: [{pred[0]:.3f}, {pred[1]:.3f}, {pred[2]:.3f}]"
                  f"  (target error ≈ {err:.3f} m)\n")

    # ──────────────────────────────────────────────────────────────────────────
    def _monitor_flight(self):
        if self.phase_ctrl.state.get("flight_logged", False):
            return
        box_dof = self.model.body_dofadr[self.box_id]
        box_pos = self.data.xpos[self.box_id]
        box_vel = self.data.qvel[box_dof: box_dof + 3]
        speed   = np.linalg.norm(box_vel)
        if speed < 0.1 and box_pos[2] < 0.15:
            self.phase_ctrl.state["flight_logged"] = True
            err = np.linalg.norm(box_pos[:2] - config.LANDING_POINT[:2])
            print(f"\n  LANDED!")
            print(f"  End   : [{box_pos[0]:.3f}, {box_pos[1]:.3f}, {box_pos[2]:.3f}]")
            print(f"  Target: [{config.LANDING_POINT[0]:.3f}, "
                  f"{config.LANDING_POINT[1]:.3f}, {config.LANDING_POINT[2]:.3f}]")
            print(f"  2-D landing error : {err:.3f} m\n")

    # ──────────────────────────────────────────────────────────────────────────
    def _get_desired_position(self) -> np.ndarray:
        phase    = self.phase_ctrl.get_phase()
        box_pos  = self.data.xpos[self.box_id].copy()

        if phase == "lifting_object":
            desired      = box_pos.copy()
            desired[2]   = config.TARGET_LIFT_HEIGHT
            return desired

        if phase == "throwing_object":
            released = self.phase_ctrl.state.get("throw_released", False)
            if not released:
                return self.x_rel.copy()
            else:
                if self._release_sim_time is not None:
                    t    = self.data.time - self._release_sim_time
                    gvec = np.array([0.0, 0.0, -G])
                    return self.x_rel + self.v_rel * t + 0.5 * gvec * t**2
        return box_pos.copy()

    # ──────────────────────────────────────────────────────────────────────────
    # NEW — open-loop "ceiling" simulation of the DS + impedance law alone.
    # Assumes PERFECT tracking (xdot_o each step = xdot_o_star commanded the
    # step before) — i.e. no Jacobian mapping loss, no joint-velocity clamp,
    # no contact/friction lag. This isolates what the CONTROL LAW allows,
    # independent of whether the robot can actually track it.
    # ──────────────────────────────────────────────────────────────────────────
    def simulate_ds_ceiling(self, box_start_pos: np.ndarray, dt: float = 0.001):
        if box_start_pos is None:
            return None
        x_o    = box_start_pos.copy()
        xdot_o = np.zeros(3)
        times, positions, velocities = [], [], []
        t = 0.0
        while t <= self.MAX_SWING_TIME:
            times.append(t)
            positions.append(x_o.copy())
            velocities.append(xdot_o.copy())

            pos_error = np.linalg.norm(x_o - self.x_rel)
            xddot_des = self.throwing_ds.compute_acceleration(
                x_o, xdot_o, self.x_rel, self.v_rel)
            xdot_o_star = self.impedance_ctrl.compute_object_velocity(
                xdot_o    = xdot_o,
                xddot_des = xddot_des,
                w_mo_fb   = np.zeros(3),
                w_obj_hat = np.zeros(3),
                x_o_star  = self.x_rel,
                x_o       = x_o)
            speed = np.linalg.norm(xdot_o_star)
            if speed > self.EE_SWING_SPEED:
                xdot_o_star = xdot_o_star / speed * self.EE_SWING_SPEED

            # perfect-tracking assumption: robot instantly achieves xdot_o_star
            xdot_o = xdot_o_star
            x_o    = x_o + xdot_o * dt
            t     += dt

            if pos_error <= self.release_position_tolerance and \
               np.linalg.norm(xdot_o) / (np.linalg.norm(self.v_rel) + 1e-9) >= self.VELOCITY_THRESHOLD:
                break  # would have released here under ideal tracking

        return {
            "time":     np.array(times),
            "position": np.array(positions),
            "velocity": np.array(velocities),
        }

    # ──────────────────────────────────────────────────────────────────────────
    def plot_results(self):
        """Four plots: Y position tracking, Y velocity, joint velocities vs
        limits, and grasp force tracking during the throw."""
        if not self.log["time"]:
            print("No data logged.")
            return

        times           = np.array(self.log["time"])
        achieved        = np.array(self.log["achieved_pos"])
        achieved_vel    = np.array(self.log["achieved_vel"])
        commanded_vel   = np.array(self.log["commanded_vel"])
        phases          = np.array(self.log["phase"])
        joint_vel_cmd   = np.array(self.log["joint_vel_cmd"])
        grasp_f_left    = np.array(self.log["grasp_force_left"])
        grasp_f_right   = np.array(self.log["grasp_force_right"])

        _LABEL_SIZE, _TICK_SIZE, _LEGEND_SIZE, _LW = 15, 13, 13, 2.5

        # ── Figure 1: Y Position ────────────────────────────────────────────
        fig1, ax1 = plt.subplots(figsize=(11, 5))
        ax1.plot(times, achieved[:, 1], color="#2166ac", linewidth=_LW,
                 label="Achieved object position")
        ax1.axhline(config.LANDING_POINT[1], color="#d6604d", linewidth=_LW,
                    linestyle="--", label="Desired object position")
        throw_mask = phases == "throwing_object"
        if throw_mask.any():
            ax1.axvspan(times[throw_mask][0], times[throw_mask][-1],
                        alpha=0.10, color="#4393c3", zorder=0)
        ax1.set_xlabel("Time (s)", fontsize=_LABEL_SIZE, fontweight="bold")
        ax1.set_ylabel("Y Position (m)", fontsize=_LABEL_SIZE, fontweight="bold")
        ax1.tick_params(labelsize=_TICK_SIZE)
        ax1.legend(fontsize=_LEGEND_SIZE, frameon=True, framealpha=0.9, edgecolor="0.7")
        ax1.grid(True, alpha=0.3, linestyle="--")
        ax1.spines[["top", "right"]].set_visible(False)
        plt.tight_layout()
        plt.savefig("position_tracking.png", dpi=180)
        print("  Saved: position_tracking.png")

        # ── Figure 2: Y Velocity during throw ───────────────────────────────
        fig2, ax2 = plt.subplots(figsize=(11, 5))
        cmd_finite = np.isfinite(commanded_vel[:, 1]) & throw_mask
        ax2.plot(times[throw_mask], achieved_vel[throw_mask, 1], color="#2166ac",
                 linewidth=_LW, label="Actual velocity")
        if cmd_finite.any():
            ax2.plot(times[cmd_finite], commanded_vel[cmd_finite, 1], color="#f4a582",
                     linewidth=_LW, label="Commanded velocity")
        ax2.axhline(self.v_rel[1], color="#d6604d", linewidth=_LW, linestyle="--",
                    label="Release velocity")
        ax2.set_xlabel("Time (s)", fontsize=_LABEL_SIZE, fontweight="bold")
        ax2.set_ylabel("Y Velocity (m/s)", fontsize=_LABEL_SIZE, fontweight="bold")
        ax2.tick_params(labelsize=_TICK_SIZE)
        ax2.legend(fontsize=_LEGEND_SIZE, frameon=True, framealpha=0.9, edgecolor="0.7")
        ax2.grid(True, alpha=0.3, linestyle="--")
        ax2.spines[["top", "right"]].set_visible(False)
        plt.tight_layout()
        plt.savefig("velocity_throw.png", dpi=180)
        print("  Saved: velocity_throw.png")

        # ── Figure 3: Joint velocities vs. limits ───────────────────────────
        if throw_mask.any() and joint_vel_cmd.shape[0] == times.shape[0]:
            fig3, ax3 = plt.subplots(figsize=(11, 5))
            n_act = joint_vel_cmd.shape[1]
            cmap = plt.get_cmap("tab20")
            for j in range(n_act):
                ax3.plot(times[throw_mask], joint_vel_cmd[throw_mask, j],
                         color=cmap(j / max(n_act - 1, 1)), linewidth=1.5,
                         label=f"joint {j}" if j < 6 else None)
                ax3.axhline(self.joint_vel_limits[j], color="0.5", linewidth=0.8,
                            linestyle=":", alpha=0.6)
                ax3.axhline(-self.joint_vel_limits[j], color="0.5", linewidth=0.8,
                            linestyle=":", alpha=0.6)
            ax3.set_xlabel("Time (s)", fontsize=_LABEL_SIZE, fontweight="bold")
            ax3.set_ylabel("Joint velocity (rad/s)", fontsize=_LABEL_SIZE, fontweight="bold")
            ax3.tick_params(labelsize=_TICK_SIZE)
            ax3.set_title("Commanded joint velocities vs. hardware limits (dotted)",
                          fontsize=_LABEL_SIZE)
            ax3.grid(True, alpha=0.3, linestyle="--")
            ax3.spines[["top", "right"]].set_visible(False)
            plt.tight_layout()
            plt.savefig("joint_velocity_limits.png", dpi=180)
            print("  Saved: joint_velocity_limits.png")

        # ── Figure 4: Grasp force tracking ───────────────────────────────────
        if throw_mask.any() and grasp_f_left.shape[0] == times.shape[0]:
            fig4, ax4 = plt.subplots(figsize=(11, 5))
            ax4.plot(times[throw_mask], grasp_f_left[throw_mask], color="#2166ac",
                     linewidth=_LW, label="Left EE grasp force")
            ax4.plot(times[throw_mask], grasp_f_right[throw_mask], color="#4393c3",
                     linewidth=_LW, label="Right EE grasp force")
            ax4.axhline(config.DESIRED_GRASP_FORCE, color="#d6604d", linewidth=_LW,
                        linestyle="--", label="Desired grasp force (F*)")
            ax4.set_xlabel("Time (s)", fontsize=_LABEL_SIZE, fontweight="bold")
            ax4.set_ylabel("Grasp force (N)", fontsize=_LABEL_SIZE, fontweight="bold")
            ax4.tick_params(labelsize=_TICK_SIZE)
            ax4.legend(fontsize=_LEGEND_SIZE, frameon=True, framealpha=0.9, edgecolor="0.7")
            ax4.grid(True, alpha=0.3, linestyle="--")
            ax4.spines[["top", "right"]].set_visible(False)
            plt.tight_layout()
            plt.savefig("grasp_force_tracking.png", dpi=180)
            print("  Saved: grasp_force_tracking.png")

        plt.show()

    # ──────────────────────────────────────────────────────────────────────────
    def plot_release_summary(self):
        """Bar chart: target v_rel vs actual release velocity of the box,
        per component + magnitude."""
        if self.release_log is None:
            print("No release occurred — skipping release summary plot.")
            return
        v_target, v_actual = self.release_log
        components  = ["Vx", "Vy", "Vz", "|V|"]
        target_vals = list(v_target) + [np.linalg.norm(v_target)]
        actual_vals = list(v_actual) + [np.linalg.norm(v_actual)]

        x, width = np.arange(len(components)), 0.35
        fig5, ax5 = plt.subplots(figsize=(9, 5))
        ax5.bar(x - width/2, target_vals, width, color="#d6604d", label="Target (v_rel)")
        ax5.bar(x + width/2, actual_vals, width, color="#2166ac", label="Actual (box, at release)")
        ax5.set_xticks(x)
        ax5.set_xticklabels(components, fontsize=13)
        ax5.set_ylabel("Velocity (m/s)", fontsize=15, fontweight="bold")
        ax5.set_title("Release velocity: target vs. actual (box)", fontsize=15)
        ax5.axhline(0, color="0.3", linewidth=0.8)
        ax5.legend(fontsize=13)
        ax5.grid(True, alpha=0.3, linestyle="--", axis="y")
        speed_ratio = np.linalg.norm(v_actual) / (np.linalg.norm(v_target) + 1e-9) * 100
        ax5.text(0.02, 0.95, f"Speed ratio: {speed_ratio:.1f}%", transform=ax5.transAxes,
                  fontsize=13, va="top", bbox=dict(boxstyle="round", facecolor="white", alpha=0.8))
        plt.tight_layout()
        plt.savefig("release_velocity_summary.png", dpi=180)
        print("  Saved: release_velocity_summary.png")
        plt.show()

    # ──────────────────────────────────────────────────────────────────────────
    def plot_ee_velocities(self):
        """Actual EE Vy vs the OBJECT TRANSLATION command (commanded_vel /
        xdot_o_star), which is the correct 'this is what the hand should be
        translating at' reference — NOT the grasp-squeeze term, which is a
        separate, small, local closing motion. Also shows the raw squeeze
        command for completeness, and the box's own actual velocity."""
        if not self.log["time"]:
            return
        times  = np.array(self.log["time"])
        phases = np.array(self.log["phase"])
        throw_mask = phases == "throwing_object"
        if not throw_mask.any():
            return

        ee_l_actual   = np.array(self.log["ee_left_vel_actual"])
        ee_r_actual   = np.array(self.log["ee_right_vel_actual"])
        squeeze_l_cmd = np.array(self.log["ee_left_vel_cmd"])
        squeeze_r_cmd = np.array(self.log["ee_right_vel_cmd"])
        obj_cmd       = np.array(self.log["commanded_vel"])   # xdot_o_star, same for both hands (rigid-grasp approx)
        box_actual    = np.array(self.log["achieved_vel"])

        fig6, (ax_l, ax_r) = plt.subplots(1, 2, figsize=(14, 5), sharey=True)
        for ax, ee_actual, squeeze_cmd, label in [
                (ax_l, ee_l_actual, squeeze_l_cmd, "Left EE"),
                (ax_r, ee_r_actual, squeeze_r_cmd, "Right EE")]:
            ax.plot(times[throw_mask], ee_actual[throw_mask, 1], color="#2166ac",
                    linewidth=2.5, label=f"{label} actual (measured)")
            cmd_finite = np.isfinite(obj_cmd[:, 1]) & throw_mask
            ax.plot(times[cmd_finite], obj_cmd[cmd_finite, 1], color="#f4a582",
                    linewidth=2.5, linestyle="--",
                    label="Object-translation target (xdot_o_star)")
            ax.plot(times[throw_mask], squeeze_cmd[throw_mask, 1], color="0.6",
                    linewidth=1.2, linestyle=":", label="Grasp-squeeze component only")
            ax.plot(times[throw_mask], box_actual[throw_mask, 1], color="0.3",
                    linewidth=1.5, linestyle="-.", label="Box actual")
            ax.axhline(self.v_rel[1], color="#d6604d", linewidth=2, linestyle="--",
                       label="Target release Vy")
            ax.set_xlabel("Time (s)", fontsize=13, fontweight="bold")
            ax.set_title(label, fontsize=14)
            ax.legend(fontsize=9)
            ax.grid(True, alpha=0.3, linestyle="--")
        ax_l.set_ylabel("Y Velocity (m/s)", fontsize=13, fontweight="bold")
        plt.tight_layout()
        plt.savefig("ee_velocities_throw.png", dpi=180)
        print("  Saved: ee_velocities_throw.png")
        plt.show()

    # ──────────────────────────────────────────────────────────────────────────
    def plot_release_diagnostics(self):
        """shows exactly why/whether the 'near x_rel AND near v_rel'
        release condition ever gets satisfied. Top: pos_error(t) vs its
        tolerance. Bottom: current_speed(t) vs the velocity-threshold speed.
        Grasp-active region shaded; actual release instant marked."""
        if not self.log["time"]:
            return
        times      = np.array(self.log["time"])
        phases     = np.array(self.log["phase"])
        achieved   = np.array(self.log["achieved_pos"])
        achieved_v = np.array(self.log["achieved_vel"])
        grasp_on   = np.array(self.log["grasp_active"], dtype=bool)
        throw_mask = phases == "throwing_object"
        if not throw_mask.any():
            return

        pos_error = np.linalg.norm(achieved[:, :3] - self.x_rel[None, :], axis=1)
        speed     = np.linalg.norm(achieved_v[:, :3], axis=1)
        target_speed = np.linalg.norm(self.v_rel)

        fig7, (axp, axv) = plt.subplots(2, 1, figsize=(11, 8), sharex=True)

        axp.plot(times[throw_mask], pos_error[throw_mask], color="#2166ac", linewidth=2.5,
                 label="pos_error = |x_o - x_rel|")
        axp.axhline(self.release_position_tolerance, color="#d6604d", linewidth=2, linestyle="--",
                    label=f"RELEASE_POSITION_TOLERANCE ({self.release_position_tolerance} m)")
        axp.axhline(self.grasp_deactivation_tolerance, color="#f4a582", linewidth=2, linestyle=":",
                    label=f"GRASP_DEACTIVATION_TOLERANCE ({self.grasp_deactivation_tolerance} m)")
        if grasp_on[throw_mask].any():
            axp.fill_between(times[throw_mask], 0, pos_error.max()*1.1,
                              where=grasp_on[throw_mask], color="#4393c3", alpha=0.08,
                              label="grasp active", step="pre")
        axp.set_ylabel("Position error (m)", fontsize=13, fontweight="bold")
        axp.legend(fontsize=10)
        axp.grid(True, alpha=0.3, linestyle="--")

        axv.plot(times[throw_mask], speed[throw_mask], color="#2166ac", linewidth=2.5,
                 label="|box velocity|")
        axv.axhline(target_speed * self.VELOCITY_THRESHOLD, color="#d6604d", linewidth=2, linestyle="--",
                    label=f"VELOCITY_THRESHOLD ({self.VELOCITY_THRESHOLD*100:.0f}% of |v_rel| = "
                          f"{target_speed*self.VELOCITY_THRESHOLD:.2f} m/s)")
        axv.axhline(target_speed, color="0.3", linewidth=2, linestyle=":", label="|v_rel| (full target)")
        if self.release_ee_log is not None and self._release_sim_time is not None:
            axv.axvline(self._release_sim_time, color="0.2", linewidth=1.5,
                        label=f"release @ t={self._release_sim_time:.2f}s")
            axp.axvline(self._release_sim_time, color="0.2", linewidth=1.5)
        axv.set_xlabel("Time (s)", fontsize=13, fontweight="bold")
        axv.set_ylabel("Speed (m/s)", fontsize=13, fontweight="bold")
        axv.legend(fontsize=10)
        axv.grid(True, alpha=0.3, linestyle="--")

        plt.tight_layout()
        plt.savefig("release_trigger_diagnostics.png", dpi=180)
        print("  Saved: release_trigger_diagnostics.png")
        plt.show()

    # ──────────────────────────────────────────────────────────────────────────
    def plot_ds_ceiling_analysis(self):
        """open-loop DS+impedance simulation (perfect tracking assumed)
        vs the actually-achieved box velocity, vs the target v_rel. Answers:
        does the control law even reach v_rel in principle, for these gains
        and this swing distance, if the robot tracked it perfectly?"""
        if self._swing_start_box_pos is None:
            print("Swing never started — skipping DS ceiling analysis.")
            return
        sim = self.simulate_ds_ceiling(self._swing_start_box_pos)
        if sim is None:
            return

        times      = np.array(self.log["time"])
        phases     = np.array(self.log["phase"])
        achieved_v = np.array(self.log["achieved_vel"])
        throw_mask = phases == "throwing_object"

        fig8, ax8 = plt.subplots(figsize=(11, 5))
        if throw_mask.any():
            t0 = times[throw_mask][0]
            ax8.plot(times[throw_mask] - t0, achieved_v[throw_mask, 1], color="#2166ac",
                     linewidth=2.5, label="Actual box Vy (real sim, real robot)")
        ax8.plot(sim["time"], sim["velocity"][:, 1], color="#4393c3", linewidth=2.5,
                 linestyle="--", label="Ideal DS+impedance ceiling Vy (perfect tracking, open-loop)")
        ax8.axhline(self.v_rel[1], color="#d6604d", linewidth=2, linestyle=":",
                    label="Target v_rel Y")
        ax8.set_xlabel("Time since swing start (s)", fontsize=13, fontweight="bold")
        ax8.set_ylabel("Y Velocity (m/s)", fontsize=13, fontweight="bold")
        ax8.set_title("Control-law ceiling vs. actual execution", fontsize=14)
        ax8.legend(fontsize=10)
        ax8.grid(True, alpha=0.3, linestyle="--")
        peak_ceiling = np.min(sim["velocity"][:, 1])  # most negative = fastest toward -Y
        note = (f"Ideal peak |Vy| ≈ {abs(peak_ceiling):.2f} m/s "
                f"vs target {abs(self.v_rel[1]):.2f} m/s "
                f"({abs(peak_ceiling)/abs(self.v_rel[1])*100:.0f}% of target)")
        ax8.text(0.02, 0.05, note, transform=ax8.transAxes, fontsize=10,
                 bbox=dict(boxstyle="round", facecolor="white", alpha=0.85))
        plt.tight_layout()
        plt.savefig("ds_ceiling_analysis.png", dpi=180)
        print("  Saved: ds_ceiling_analysis.png")
        print(f"  [DS ceiling] ideal peak Vy = {peak_ceiling:.3f} m/s "
              f"(target v_rel Y = {self.v_rel[1]:.3f} m/s, "
              f"{abs(peak_ceiling)/abs(self.v_rel[1])*100:.1f}% of target)")
        plt.show()

    # ──────────────────────────────────────────────────────────────────────────
    def _read_joint_positions(self) -> np.ndarray:
        q = np.zeros(self.n_act)
        for i in range(self.n_act):
            jid = self.model.actuator_trnid[i, 0]
            q[i] = self.data.qpos[self.model.jnt_qposadr[jid]]
        return q

    # ──────────────────────────────────────────────────────────────────────────
    def run(self):
        self.vel_ctrl.set_velocity_trajectory(self.compute_velocity_commands)
        mujoco.set_mjcb_control(self.vel_ctrl.control_callback)

        with mujoco.viewer.launch_passive(self.model, self.data) as viewer:
            while viewer.is_running():
                mujoco.mj_step(self.model, self.data)

                phase = self.phase_ctrl.get_phase()
                dof   = self.model.body_dofadr[self.box_id]
                pos   = self.data.xpos[self.box_id].copy()
                vel   = self.data.qvel[dof: dof + 3].copy()

                left_ee_actual, right_ee_actual = self._get_ee_velocities()

                self.log["time"].append(self.data.time)
                self.log["achieved_pos"].append(pos.copy())
                self.log["desired_pos"].append(self._get_desired_position())
                self.log["achieved_vel"].append(vel.copy())
                self.log["commanded_vel"].append(self._last_commanded_vel.copy())
                self.log["phase"].append(phase)
                self.log["joint_vel_cmd"].append(self._last_joint_vel_cmd.copy())
                self.log["joint_vel_scale"].append(self._last_joint_vel_scale)
                self.log["grasp_force_left"].append(self._last_grasp_force[0])
                self.log["grasp_force_right"].append(self._last_grasp_force[1])
                self.log["ee_left_vel_actual"].append(left_ee_actual)
                self.log["ee_right_vel_actual"].append(right_ee_actual)
                self.log["ee_left_vel_cmd"].append(self._last_ee_vel_cmd[0].copy())
                self.log["ee_right_vel_cmd"].append(self._last_ee_vel_cmd[1].copy())
                self.log["grasp_active"].append(
                    bool(self.phase_ctrl.state.get("grasp_active", True))
                    if phase == "throwing_object" else False)

                print(f"box pos: [{pos[0]:.3f}, {pos[1]:.3f}, {pos[2]:.3f}]  "
                      f"vel: [{vel[0]:.3f}, {vel[1]:.3f}, {vel[2]:.3f}]")

                viewer.sync()
                time.sleep(config.SLEEP_TIME)

        print("\nGenerating plots...")
        self.plot_results()
        self.plot_release_summary()
        self.plot_ee_velocities()
        self.plot_release_diagnostics()
        self.plot_ds_ceiling_analysis()


# ══════════════════════════════════════════════════════════════════════════════
def main():
    print("=" * 60)
    print("GRASP ADMITTANCE (force closure) + DS + IMPEDANCE THROWING")
    print("=" * 60)
    task = ThrowingDSImpedance()
    task.run()


if __name__ == "__main__":
    main()