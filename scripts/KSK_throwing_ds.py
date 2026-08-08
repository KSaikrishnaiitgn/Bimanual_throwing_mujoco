"""
Throwing with DS + Impedance Control
Implements the exact mathematical framework from the presentation slides:

  Step 1 — Release Velocity (projectile math)
      R        = ||Δx_y||,   Δx_y = (p_land - p_rel)[:2]
      v_rel²   = g R² / [2 cos²θ (R tanθ − Δz)]
      v⃗_rel   = v_rel cosθ ê_h  +  v_rel sinθ ẑ

  Step 2 — Modified DS for throwing
      ẍ_des = −K_ds (x_o − x_rel)  −  B_ds (ẋ_o − v⃗_rel)

  Step 3 — Object-level impedance control
      ẋ_o* = ẋ_o − D⁻¹ [ M_o ẍ_des + w_mo^fb − ŵ_obj + K₁(x_o* − x_o) ]

  Step 4 — Dual-arm joint velocity
      q̇ = J_H† [ ẋ_ee + G^T ẋ_o* ]

  Step 5 — Hardware safety: per-joint velocity clamp   <-- NEW
      q̇ scaled (direction-preserving) so every joint stays
      within config.MAX_JOINT_VELOCITIES before being sent
      to the velocity controller.
"""

import os
os.environ["MUJOCO_GL"] = "glfw"

import mujoco
import mujoco.viewer
import numpy as np
import time
import matplotlib.pyplot as plt

# ── existing modular controllers
from config import throwing_config as config
from controllers.contact_handler import ContactHandler
from controllers.admittance_controller import AdmittanceController
from controllers.trajectory_planner import TrajectoryPlanner
from controllers.jacobian_mapper import JacobianMapper
from controllers.dual_arm_jacobian import DualArmJacobian
from controllers.phase_controller import PhaseController
from utils.mujoco_velocity_controller.KSK_velocity_controller import VelocityControllerGC

G = config.GRAVITY


# ══════════════════════════════════════════════════════════════════════════════
#  MODIFIED DS — Slide: "Modified DS for Throwing"
#  ẍ_des = −K_ds (x_o − x_rel) − B_ds (ẋ_o − ẋ_o^target)
# ══════════════════════════════════════════════════════════════════════════════
class ThrowingDS:
    """
    Second-order DS whose equilibrium is (x_rel, v_rel).
    Implements slide equation:
        ẍ_des = −K_ds (x_o − x_rel)  −  B_ds (ẋ_o − v⃗_rel)
    """

    def __init__(self, K_ds: np.ndarray, B_ds: np.ndarray, max_vel: float = 8.0):
        self.K_ds   = K_ds          # 3×3 position stiffness
        self.B_ds   = B_ds          # 3×3 velocity damping
        self.max_vel = max_vel

    def compute_acceleration(self,
                             x_o:   np.ndarray,   # current object position  (3,)
                             xdot_o: np.ndarray,  # current object velocity   (3,)
                             x_rel:  np.ndarray,  # release position          (3,)
                             v_rel:  np.ndarray   # target release velocity   (3,)
                             ) -> np.ndarray:
        """
        Returns ẍ_des  (3,).
        Slide eq: ẍ_des = −K_ds(x_o − x_rel) − B_ds(ẋ_o − v⃗_rel)
        """
        pos_err = x_o   - x_rel    # drives position toward release point
        vel_err = xdot_o - v_rel   # drives velocity toward release velocity
        xddot_des = -self.K_ds @ pos_err - self.B_ds @ vel_err
        return xddot_des


# ══════════════════════════════════════════════════════════════════════════════
#  IMPEDANCE CONTROL — Slide: "Object-Level Impedance Control"
#  ẋ_o* = ẋ_o − D⁻¹ [ M_o ẍ_des + w_mo^fb − ŵ_obj + K₁(x_o* − x_o) ]
# ══════════════════════════════════════════════════════════════════════════════
class ThrowingImpedance:
    """
    Object-level impedance controller.
    Slide equation (corrected sign for feedforward control):
        ẋ_o* = ẋ_o + D⁻¹ [ M_o ẍ_des + w_mo^fb − ŵ_obj + K₁(x_o* − x_o) ]

    The slide uses '−D⁻¹' but that convention defines ẍ_des as the net force
    error, not the desired acceleration.  Here ẍ_des is a FORWARD desired
    acceleration (from the DS), so we need '+D⁻¹' so that a forward ẍ_des
    produces a forward velocity command.

    where:
        M_o     = virtual object inertia matrix          (3×3)
        D       = damping matrix                         (3×3)
        K₁      = impedance stiffness matrix             (3×3)
        w_mo^fb = linear force feedback from hands       (3,)
        ŵ_obj   = estimated object gravity wrench        (3,)
        x_o*    = desired object position (= x_rel)     (3,)
    """

    def __init__(self, M_o: np.ndarray, D: np.ndarray, K_1: np.ndarray,
                 max_vel: float = 5.0):
        self.M_o     = M_o
        self.D_inv   = np.linalg.inv(D)
        self.K_1     = K_1
        self.max_vel = max_vel

    def gravity_wrench(self, object_mass: float) -> np.ndarray:
        """ŵ_obj = [0, 0, −m·g]  (gravity acting on object)"""
        return np.array([0.0, 0.0, -object_mass * G])

    def compute_object_velocity(self,
                                xdot_o:    np.ndarray,  # current object vel   (3,)
                                xddot_des: np.ndarray,  # DS desired accel     (3,)
                                w_mo_fb:   np.ndarray,  # EE force feedback    (3,)
                                w_obj_hat: np.ndarray,  # gravity wrench       (3,)
                                x_o_star:  np.ndarray,  # desired position     (3,)
                                x_o:       np.ndarray   # current position     (3,)
                                ) -> np.ndarray:
        """
        Corrected equation ('+' so forward ẍ_des → forward ẋ_o*):
            ẋ_o* = ẋ_o + D⁻¹ [ M_o ẍ_des + w_mo^fb − ŵ_obj + K₁(x_o* − x_o) ]
        """
        inertial_term   = self.M_o @ xddot_des                  # M_o ẍ_des
        stiffness_term  = self.K_1 @ (x_o_star - x_o)           # K₁(x_o* − x_o)
        bracket         = inertial_term + w_mo_fb - w_obj_hat + stiffness_term

        xdot_o_star = xdot_o + self.D_inv @ bracket             # + not −

        # Safety clamp (Cartesian object velocity, m/s — component-wise)
        xdot_o_star = np.clip(xdot_o_star, -self.max_vel, self.max_vel)
        return xdot_o_star


# ══════════════════════════════════════════════════════════════════════════════
#  MAIN SIMULATION
# ══════════════════════════════════════════════════════════════════════════════
class ThrowingDSImpedance:
    """
    Full pipeline per simulation step (throwing phase):
        1. compute_release_velocity   → v⃗_rel
        2. ThrowingDS.compute_acceleration  → ẍ_des
        3. ContactHandler wrench      → w_mo^fb
        4. ThrowingImpedance          → ẋ_o*
        5. DualArmJacobian            → q̇ = J_H† [ẋ_ee + G^T ẋ_o*]
        6. Joint-velocity clamp       → q̇ respects config.MAX_JOINT_VELOCITIES
    """

    def __init__(self):
        self.model = mujoco.MjModel.from_xml_path("/home/iitgn-robotics/Saikrishna/samriddhi_mujoco_bimanual_ds_dynamic_task/robot_description/kshitij_lifting.xml")
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
        # config.MAX_JOINT_VELOCITIES is shape (12,) = 6 left + 6 right.
        # This is the ONLY place these hardware limits should ever be relaxed.
        self.joint_vel_limits = np.asarray(config.MAX_JOINT_VELOCITIES, dtype=float)

        # ── velocity controller (gravity compensation + joint-vel tracking) ──
        # NOTE: joint_vel_limits is now passed in as a second, independent
        # safety net enforced at the torque-command level (see
        # VelocityControllerGC.control_callback). This protects the hardware
        # even if some future change to the throwing phase forgets to clip.
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

        # ── throwing DS (Step 2) ──────────────────────────────────────────────
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

        # ── throw state ───────────────────────────────────────────────────────
        self.x_rel:  np.ndarray | None = None   # release position
        self.v_rel:  np.ndarray | None = None   # computed release velocity

        # ── data logger ───────────────────────────────────────────────────────
        self.log = {
            "time":          [],  # simulation time
            "achieved_pos":  [],  # actual box position  (3,)
            "desired_pos":   [],  # desired box position (3,)
            "achieved_vel":  [],  # actual box velocity  (3,)
            "commanded_vel": [],  # ẋ_o* from impedance  (3,) — NaN outside throw
            "phase":         [],  # phase string
            "joint_vel_cmd": [],  # q̇ commanded (post-clip), (n_act,)
            "joint_vel_scale": [],  # scale factor applied by the clamp (1.0 = no clip)
        }
        self._last_commanded_vel = np.full(3, np.nan)
        self._last_joint_vel_cmd = np.zeros(self.n_act)
        self._last_joint_vel_scale = 1.0
        self.release_log = None   # (desired_vel, achieved_vel) filled at release
        self._release_sim_time = None

        print("✅  ThrowingDSImpedance initialised")
        print(f"    box id={self.box_id}  mass={self.object_mass:.3f} kg")
        print(f"    left EE site={self.left_ee_id}  "
              f"right EE site={self.right_ee_id}")
        print(f"    joint velocity limits (rad/s): {self.joint_vel_limits}")

    # ──────────────────────────────────────────────────────────────────────────
    #  CONTROL CALLBACK
    # ──────────────────────────────────────────────────────────────────────────
    def compute_velocity_commands(self, t: float) -> np.ndarray:
        joint_positions = self._read_joint_positions()
        phase = self.phase_ctrl.get_phase()

        if self.phase_ctrl.should_print_phase():
            print(f"  Phase: {phase}")

        # ── PHASE 1 & 2: reach grasp pose ─────────────────────────────────────
        if phase in ("reaching_first_target", "reaching_second_target"):
            cmds = self.phase_ctrl.compute_position_control(
                joint_positions, self.n_act)
            # Reaching-phase commands are also hardware joint velocities —
            # clamp them too, in case the position controller's internal
            # gains ever produce something above the hardware limit.
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

        # ── PHASE 3: settle contacts ───────────────────────────────────────────
        elif phase == "reading_sensors":
            if time.time() - self.phase_ctrl.state["last_phase_change"] \
                    > config.READING_SENSORS_DURATION:
                print("✅  Starting lift.")
                self.phase_ctrl.set_phase("lifting_object")
            return np.zeros(self.n_act)

        # ── PHASE 4: admittance-controlled lift ────────────────────────────────
        elif phase == "lifting_object":
            return self._lifting_phase()

        # ── PHASE 5: DS + Impedance throw ─────────────────────────────────────
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
            # Lifting is slower than throwing, but clamp anyway for safety.
            return self._clip_joint_velocities(q_dot)
        return np.zeros(self.n_act)

    # ──────────────────────────────────────────────────────────────────────────
    def _prepare_throw(self):
        """Compute release velocity and transition to throwing phase."""
        self.x_rel = self.data.xpos[self.box_id].copy()
        self.v_rel = self.trajectory_planner.compute_release_velocity(
            self.x_rel, config.LANDING_POINT, config.THROW_ANGLE)

        # Optional scale factor from config (applied after the physics-based
        # release velocity solve — e.g. to compensate for real energy losses
        # at contact release). Uses config.VELOCITY_SCALE if present.
        v_scale = getattr(config, "VELOCITY_SCALE", 1.0)
        self.v_rel = self.v_rel * v_scale

        # Normalised throw direction for arm swing
        self._throw_dir = self.v_rel / (np.linalg.norm(self.v_rel) + 1e-9)

        # Gravity stays OFF during the arm swing; we re-enable it at release
        self.phase_ctrl.state["throw_phase_started"] = True
        self.phase_ctrl.state["throw_released"]      = False
        self.phase_ctrl.state["flight_logged"]        = False
        self.phase_ctrl.state["swing_start_sim_time"] = self.data.time
        self.phase_ctrl.set_phase("throwing_object")

        print(f"\n  Lift target reached — arm swing starting")
        print(f"  Release pos  : {self.x_rel}")
        print(f"  Target land  : {config.LANDING_POINT}")
        v = self.v_rel
        print(f"  v_rel        : [{v[0]:.3f}, {v[1]:.3f}, {v[2]:.3f}] m/s")
        d = self._throw_dir
        print(f"  throw_dir    : [{d[0]:.3f}, {d[1]:.3f}, {d[2]:.3f}]\n")

    # ──────────────────────────────────────────────────────────────────────────
    # Throw constants
    EE_SWING_SPEED     = 8.0   # m/s  — cap on commanded EE/object speed (Cartesian)
    VELOCITY_THRESHOLD = 0.88  # release when box reaches 88 % of target speed
    MAX_SWING_TIME     = 2.0   # s    — hard timeout fallback

    # ──────────────────────────────────────────────────────────────────────────
    def _clip_joint_velocities(self, q_dot: np.ndarray) -> np.ndarray:
        """
        Enforce config.MAX_JOINT_VELOCITIES (rad/s, per-joint) on a commanded
        joint-velocity vector.

        Uses a UNIFORM (direction-preserving) scale-down rather than a
        per-component np.clip. Independently clamping each joint distorts
        the direction of q̇ — some joints get clamped harder than others —
        which bends the actual swing/throw trajectory away from the one the
        Jacobian solved for. Scaling the whole vector by the single
        worst-offending joint's overshoot ratio keeps the motion's shape
        intact while guaranteeing every joint stays within its hardware
        limit. This is the behavior you want before deploying on real
        hardware where exceeding a joint's rated velocity can fault or
        damage the actuator.

        Returns the clipped q_dot and stores the applied scale factor
        (1.0 = no clipping needed) for logging/debugging.
        """
        limits = self.joint_vel_limits
        # Avoid divide-by-zero; limits should always be > 0, but guard anyway.
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
        Contact-based throw — box is accelerated purely by arm contact forces.

        Sub-phase A  (until velocity threshold or timeout):
          DS + Impedance + Jacobian drives the arms in the throw direction.
          Box velocity builds up through physical contact with the arms.
          No artificial forces applied to the box.

          ẍ_des  = −K_ds(x_o − x_rel) − B_ds(ẋ_o − v⃗_rel)          [Slide 7]
          ẋ_o*   = ẋ_o + D⁻¹[M_o ẍ_des + K₁(x_rel − x_o)]           [Slide 9]
          q̇      = J_H† [ẋ_ee + G^T ẋ_o*]                            [Slide 10]
          q̇      = clip_to_joint_limits(q̇)                          [NEW — hardware safety]

        Sub-phase B  (release):
          Disable arm–box contact, re-enable gravity.
          Box flies with the velocity contact forces actually gave it.
        """
        if self.phase_ctrl.state["throw_released"]:
            self._monitor_flight()
            return np.zeros(self.n_act)

        _box_dof = self.model.body_dofadr[self.box_id]
        xdot_o   = self.data.qvel[_box_dof:_box_dof + 3].copy()
        elapsed  = self.data.time - self.phase_ctrl.state["swing_start_sim_time"]

        # ── Velocity-based release condition ──────────────────────────────────
        target_speed  = np.linalg.norm(self.v_rel)
        current_speed = np.linalg.norm(xdot_o)
        vel_ratio     = current_speed / target_speed if target_speed > 1e-6 else 0.0

        if vel_ratio >= self.VELOCITY_THRESHOLD or elapsed >= self.MAX_SWING_TIME:
            reason = "velocity threshold" if vel_ratio >= self.VELOCITY_THRESHOLD \
                     else "timeout"
            self._last_commanded_vel = np.full(3, np.nan)
            self._do_release(reason, xdot_o)
            return np.zeros(self.n_act)

        # ── Step 2: DS desired acceleration  [Slide 7] ───────────────────────
        x_o       = self.data.xpos[self.box_id].copy()
        xddot_des = self.throwing_ds.compute_acceleration(
            x_o, xdot_o, self.x_rel, self.v_rel)

        # ── Step 3: no contact feedback; gravity OFF so w_obj_hat = 0 ─────────
        w_mo_fb   = np.zeros(3)
        w_obj_hat = np.zeros(3)

        # ── Step 4: impedance → commanded object velocity  [Slide 9] ─────────
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

        # ── Step 5: q̇ = J_H† [ẋ_ee + G^T ẋ_o*]  [Slide 10] ─────────────────
        q_dot = self.dual_arm_jac.compute_joint_velocities(
            data             = self.data,
            x_dot_o_star     = xdot_o_star,
            x_dot_ee         = np.zeros(12),
            object_body_id   = self.box_id,
            left_ee_site_id  = self.left_ee_id,
            right_ee_site_id = self.right_ee_id)

        # ── Step 6: HARDWARE SAFETY — clamp to per-joint velocity limits ─────
        # This replaces the old np.clip(q_dot, -MAX_COMMANDED_VEL, ...) call,
        # which incorrectly applied a Cartesian (m/s) bound to joint-space
        # (rad/s) values. MAX_COMMANDED_VEL is still used above to bound the
        # *object* velocity (correct units); this step bounds *joints*.
        q_dot = self._clip_joint_velocities(q_dot)
        self._last_joint_vel_cmd = q_dot.copy()

        if int(elapsed * 10) % 5 == 0:
            scale_note = ""
            if self._last_joint_vel_scale < 0.999:
                scale_note = f"  [joint-limit clamp active: {self._last_joint_vel_scale*100:.0f}% of solved q̇]"
            print(f"  swing t={elapsed:.2f}s | box {current_speed:.2f}/{target_speed:.2f} m/s"
                  f" ({vel_ratio*100:.0f}%){scale_note}")

        return q_dot

    # ──────────────────────────────────────────────────────────────────────────
    def _do_release(self, reason: str, release_vel: np.ndarray):
        """Disable arm–box contact and re-enable gravity. No velocity injection."""
        self.phase_ctrl.state["throw_released"] = True

        # Disable arm–box contact so box flies free
        for i in range(self.model.ngeom):
            gn = self.model.geom(i).name
            if gn and "box" not in gn.lower() \
                   and "ground" not in gn.lower() \
                   and "floor"  not in gn.lower():
                self.model.geom_contype[i]     = 0
                self.model.geom_conaffinity[i] = 0

        # Re-enable gravity — box flies with whatever velocity contact gave it
        self.model.opt.gravity[:] = self.original_gravity

        self._release_sim_time = self.data.time
        self.release_log = (self.v_rel.copy(), release_vel.copy())

        actual_pos  = self.data.xpos[self.box_id].copy()
        v           = release_vel
        speed_ratio = np.linalg.norm(v) / (np.linalg.norm(self.v_rel) + 1e-9)
        print(f"\n  RELEASED ({reason})")
        print(f"  Release pos : [{actual_pos[0]:.3f}, {actual_pos[1]:.3f}, {actual_pos[2]:.3f}]")
        print(f"  Release vel : [{v[0]:.3f}, {v[1]:.3f}, {v[2]:.3f}] m/s")
        print(f"  Target  vel : [{self.v_rel[0]:.3f}, {self.v_rel[1]:.3f}, {self.v_rel[2]:.3f}] m/s")
        print(f"  Speed ratio : {speed_ratio*100:.1f}%")
        pred = self.trajectory_planner.predict_landing_point(actual_pos, v)
        if pred is not None:
            err = np.linalg.norm(pred[:2] - config.LANDING_POINT[:2])
            print(f"  Predicted landing: [{pred[0]:.3f}, {pred[1]:.3f}, {pred[2]:.3f}]"
                  f"  (target error ≈ {err:.3f} m)\n")

    # ──────────────────────────────────────────────────────────────────────────
    def _monitor_flight(self):
        """Log landing once the box hits the ground."""
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
        """Return the desired box position for the current phase."""
        phase    = self.phase_ctrl.get_phase()
        box_pos  = self.data.xpos[self.box_id].copy()

        if phase == "lifting_object":
            desired      = box_pos.copy()
            desired[2]   = config.TARGET_LIFT_HEIGHT
            return desired

        if phase == "throwing_object":
            released = self.phase_ctrl.state.get("throw_released", False)
            if not released:
                # During swing: desired position = x_rel (DS equilibrium)
                return self.x_rel.copy() if self.x_rel is not None else box_pos
            else:
                # During flight: ideal ballistic from x_rel with v_rel
                if self.x_rel is not None and self._release_sim_time is not None:
                    t    = self.data.time - self._release_sim_time
                    gvec = np.array([0.0, 0.0, -G])
                    return self.x_rel + self.v_rel * t + 0.5 * gvec * t**2
        return box_pos.copy()

    # ──────────────────────────────────────────────────────────────────────────
    def plot_results(self):
        """Three plots: Y position tracking, Y velocity during throw, and
        joint velocities vs. their hardware limits during the throw."""
        if not self.log["time"]:
            print("No data logged.")
            return

        times           = np.array(self.log["time"])
        achieved        = np.array(self.log["achieved_pos"])      # (N, 3)
        achieved_vel    = np.array(self.log["achieved_vel"])      # (N, 3)
        commanded_vel   = np.array(self.log["commanded_vel"])     # (N, 3) — NaN outside throw
        phases          = np.array(self.log["phase"])
        joint_vel_cmd   = np.array(self.log["joint_vel_cmd"])     # (N, n_act)
        joint_vel_scale = np.array(self.log["joint_vel_scale"])   # (N,)

        _LABEL_SIZE  = 15
        _TICK_SIZE   = 13
        _LEGEND_SIZE = 13
        _LW          = 2.5

        # ── Figure 1: Y Position (throw direction) — full run ─────────────────
        fig1, ax1 = plt.subplots(figsize=(11, 5))

        ax1.plot(times, achieved[:, 1],
                 color="#2166ac", linewidth=_LW, label="Achieved object position")

        ax1.axhline(config.LANDING_POINT[1],
                    color="#d6604d", linewidth=_LW, linestyle="--",
                    label="Desired object position")

        throw_mask = phases == "throwing_object"
        if throw_mask.any():
            t_throw_start = times[throw_mask][0]
            t_throw_end   = times[throw_mask][-1]
            ax1.axvspan(t_throw_start, t_throw_end,
                        alpha=0.10, color="#4393c3", zorder=0)

        ax1.set_xlabel("Time (s)", fontsize=_LABEL_SIZE, fontweight="bold")
        ax1.set_ylabel("Y Position (m)", fontsize=_LABEL_SIZE, fontweight="bold")
        ax1.tick_params(labelsize=_TICK_SIZE)
        ax1.legend(fontsize=_LEGEND_SIZE, frameon=True, framealpha=0.9,
                   edgecolor="0.7")
        ax1.grid(True, alpha=0.3, linestyle="--")
        ax1.spines[["top", "right"]].set_visible(False)
        plt.tight_layout()
        plt.savefig("position_tracking.png", dpi=180)
        print("  Saved: position_tracking.png")

        # ── Figure 2: Y Velocity during throw — actual / commanded / release ──
        fig2, ax2 = plt.subplots(figsize=(11, 5))

        cmd_finite = np.isfinite(commanded_vel[:, 1]) & throw_mask

        ax2.plot(times[throw_mask], achieved_vel[throw_mask, 1],
                 color="#2166ac", linewidth=_LW, label="Actual velocity")

        if cmd_finite.any():
            ax2.plot(times[cmd_finite], commanded_vel[cmd_finite, 1],
                     color="#f4a582", linewidth=_LW, linestyle="-",
                     label="Commanded velocity")

        if self.v_rel is not None:
            ax2.axhline(self.v_rel[1],
                        color="#d6604d", linewidth=_LW, linestyle="--",
                        label="Release velocity")

        ax2.set_xlabel("Time (s)", fontsize=_LABEL_SIZE, fontweight="bold")
        ax2.set_ylabel("Y Velocity (m/s)", fontsize=_LABEL_SIZE, fontweight="bold")
        ax2.tick_params(labelsize=_TICK_SIZE)
        ax2.legend(fontsize=_LEGEND_SIZE, frameon=True, framealpha=0.9,
                   edgecolor="0.7")
        ax2.grid(True, alpha=0.3, linestyle="--")
        ax2.spines[["top", "right"]].set_visible(False)
        plt.tight_layout()
        plt.savefig("velocity_throw.png", dpi=180)
        print("  Saved: velocity_throw.png")

        # ── Figure 3: Joint velocities vs. limits during the throw ────────────
        if throw_mask.any() and joint_vel_cmd.shape[0] == times.shape[0]:
            fig3, ax3 = plt.subplots(figsize=(11, 5))
            n_act = joint_vel_cmd.shape[1]
            cmap = plt.get_cmap("tab20")
            for j in range(n_act):
                ax3.plot(times[throw_mask], joint_vel_cmd[throw_mask, j],
                         color=cmap(j / max(n_act - 1, 1)), linewidth=1.5,
                         label=f"joint {j}" if j < 6 else None)
                ax3.axhline(self.joint_vel_limits[j], color="0.5",
                            linewidth=0.8, linestyle=":", alpha=0.6)
                ax3.axhline(-self.joint_vel_limits[j], color="0.5",
                            linewidth=0.8, linestyle=":", alpha=0.6)

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

            frac_clamped = float(np.mean(joint_vel_scale[throw_mask] < 0.999)) * 100
            print(f"  Joint-limit clamp was active during {frac_clamped:.1f}% "
                  f"of the throw phase.")

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

                # Log from the very first step
                self.log["time"].append(self.data.time)
                self.log["achieved_pos"].append(pos.copy())
                self.log["desired_pos"].append(self._get_desired_position())
                self.log["achieved_vel"].append(vel.copy())
                self.log["commanded_vel"].append(self._last_commanded_vel.copy())
                self.log["phase"].append(phase)
                self.log["joint_vel_cmd"].append(self._last_joint_vel_cmd.copy())
                self.log["joint_vel_scale"].append(self._last_joint_vel_scale)

                print(f"box pos: [{pos[0]:.3f}, {pos[1]:.3f}, {pos[2]:.3f}]  "
                      f"vel: [{vel[0]:.3f}, {vel[1]:.3f}, {vel[2]:.3f}]")

                viewer.sync()
                time.sleep(config.SLEEP_TIME)

        print("\nGenerating plots...")
        self.plot_results()


# ══════════════════════════════════════════════════════════════════════════════
def main():
    print("=" * 60)
    print("DS + IMPEDANCE THROWING  (exact slide math, joint-limit safe)")
    print("=" * 60)
    task = ThrowingDSImpedance()
    task.run()


if __name__ == "__main__":
    main()