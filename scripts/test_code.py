"""
Throwing with DS + Impedance Control + Force-Closure Grasp
Implements the exact mathematical framework from the presentation slides:

  Step 1 — Release Velocity (projectile math)
      R        = ||Δx_y||,   Δx_y = (p_land - p_rel)[:2]
      v_rel²   = g R² / [2 cos²θ (R tanθ − Δz)]
      v⃗_rel   = v_rel cosθ ê_h  +  v_rel sinθ ẑ

  Step 2 — Modified DS for throwing
      ẍ_des = −K_ds (x_o − x_rel)  −  B_ds (ẋ_o − v⃗_rel)

  Step 3 — Object-level impedance control
      ẋ_o* = ẋ_o − D⁻¹ [ M_o ẍ_des + w_mo^fb − ŵ_obj + K₁(x_o* − x_o) ]

  Step 4 — Grasp force closure (NEW)
      ẋ_ee ← K_grasp⁻¹ (F⋆ − F_meas)
      Two arms squeeze the object from opposite sides with equal & opposite
      target force -> net force on object ≈0, but friction at each contact
      resists the throw's inertial/gravity loads without slipping.

  Step 5 — Dual-arm joint velocity
      q̇ = J_H† [ ẋ_ee + G^T ẋ_o* ]
      (squeeze command ẋ_ee summed with throw-direction G^T ẋ_o*)

NOTES (this version):
  - Gravity is enabled for the ENTIRE simulation (no zero-gravity swing phase).
  - The lifting phase has been removed. After the grasp settles, the box is
    accelerated directly from its initial (post-grasp) position straight into
    the throw — there is no intermediate "lift to target height" step.
  - AdmittanceController (which drove the lift) has been removed.
  - GraspForceController now provides real force-closure squeezing during
    the swing, and drives release via a physical force ramp-down instead of
    disabling collision geoms.
  - ContactHandler now correctly rotates contact forces into the world frame
    before they're used (the previous version summed raw contact-frame
    forces as if they were already world-frame, which is wrong whenever a
    contact normal isn't axis-aligned).
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
from controllers.Contact_handler import ContactHandler
from controllers.trajectory_planner import TrajectoryPlanner
from controllers.jacobian_mapper import JacobianMapper
from controllers.dual_arm_jacobian import DualArmJacobian
from controllers.phase_controller import PhaseController
from controllers.grasp_force_controller import GraspForceController
from utils.dual_arm_velocity_controller import VelocityControllerGC

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

        # Safety clamp
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
        3. ContactHandler wrench      → w_mo^fb  (now real, world-frame)
        4. ThrowingImpedance          → ẋ_o*
        5. GraspForceController       → ẋ_ee (force-closure squeeze)
        6. DualArmJacobian            → q̇ = J_H† [ẋ_ee + G^T ẋ_o*]
    """

    def __init__(self):
        self.model = mujoco.MjModel.from_xml_path("/home/iitgn-robotics/Saikrishna/samriddhi_mujoco_bimanual_ds_dynamic_task/robot_description/kshitij_lifting.xml")
        self.data  = mujoco.MjData(self.model)

        # Gravity is ON for the entire simulation (no swing-phase zeroing).
        self.model.opt.gravity[:] = [0.0, 0.0, -G]

        mujoco.mj_resetData(self.model, self.data)
        mujoco.mj_forward(self.model, self.data)

        # ── find box ──────────────────────────────────────────────────────────
        self.box_id = next(
            (i for i in range(self.model.nbody)
             if "box" in self.model.body(i).name), -1)
        if self.box_id == -1:
            raise RuntimeError("No body containing 'box' found in the XML.")

        self.object_mass = self.model.body_mass[self.box_id]

        # ── velocity controller (gravity compensation + joint-vel tracking) ──
        self.vel_ctrl = VelocityControllerGC(
            self.model, self.data,
            kd=config.KD_GAINS, ki=config.KI_GAINS)
        self.n_act    = self.vel_ctrl.num_actuators
        self.n_per    = self.n_act // 2

        # ── sub-controllers ───────────────────────────────────────────────────
        self.contact_handler = ContactHandler(self.model, self.data)
        self.trajectory_planner = TrajectoryPlanner(G)
        self.jacobian_mapper    = JacobianMapper(self.model, self.n_act)
        self.dual_arm_jac       = DualArmJacobian(self.model, self.n_act)
        self.phase_ctrl         = PhaseController(config)
        self.phase_ctrl.initialize_targets(self.n_act)

        # ── throwing DS (Step 2) ──────────────────────────────────────────────
        # K_ds = 0 by design: during the throw we want a PURE velocity-tracking
        # DS with no positional spring pulling toward x_rel (that pull was
        # fighting the velocity build-up and stalling the throw). Wired to
        # config so both gains are tunable in one place.
        self.throwing_ds = ThrowingDS(
            K_ds   = config.K_DS,
            B_ds   = config.B_DS,
            max_vel= config.MAX_OBJ_VEL)

        # ── impedance controller (Step 3–4) ──────────────────────────────────
        # K_1 = 0 by design for the same reason: the impedance's positional
        # stiffness term K1(x_o* - x_o) was competing with the DS's velocity
        # objective whenever x_rel wasn't where the box actually was. With
        # K_1 = 0 this becomes a pure velocity-tracking impedance controller
        # that only compensates gravity/force-feedback and follows ẍ_des.
        self.impedance_ctrl = ThrowingImpedance(
            M_o    = config.M_O,
            D      = config.D_IMPEDANCE,
            K_1    = config.K_IMPEDANCE,
            max_vel= config.MAX_COMMANDED_VEL)

        # ── grasp force controller (force-closure squeeze) ────────────────────
        # ẋ ← K⁻¹(F* − F_meas). Independent of the throw-direction DS —
        # squeezes along the grasp axis while the DS/impedance path drives
        # the object along the throw direction. Summed together in the
        # Jacobian mapping via x_dot_ee (grasp) + G^T x_dot_o_star (throw).
        self.grasp_ctrl = GraspForceController(
            K_grasp     = config.K_GRASP,
            target_force= config.GRASP_TARGET_FORCE,
            ramp_steps  = config.GRASP_RELEASE_RAMP_STEPS,
            max_vel     = config.K_GRASP_MAX_VEL)

        # ── find EE sites ─────────────────────────────────────────────────────
        self.left_ee_id, self.right_ee_id = \
            self.contact_handler.find_end_effector_sites()

        # ── throw state ───────────────────────────────────────────────────────
        self.x_rel:  np.ndarray | None = None   # release position
        self.v_rel:  np.ndarray | None = None   # computed release velocity
        self._release_reason = ""

        # ── data logger ───────────────────────────────────────────────────────
        self.log = {
            "time":          [],  # simulation time
            "achieved_pos":  [],  # actual box position  (3,)
            "desired_pos":   [],  # desired box position (3,)
            "achieved_vel":  [],  # actual box velocity  (3,)
            "commanded_vel": [],  # ẋ_o* from impedance  (3,) — NaN outside throw
            "phase":         [],  # phase string
        }
        self._last_commanded_vel = np.full(3, np.nan)
        self._cmd_vel_filtered   = np.zeros(3)   # low-pass state for commanded velocity
        self._release_hold_counter = 0            # debounce counter for release condition
        self.release_log = None   # (desired_vel, achieved_vel) filled at release
        self._release_sim_time = None

        print("✅  ThrowingDSImpedance initialised")
        print(f"    box id={self.box_id}  mass={self.object_mass:.3f} kg")
        print(f"    left EE site={self.left_ee_id}  "
              f"right EE site={self.right_ee_id}")
        print(f"    grasp: K_grasp={config.K_GRASP}  "
              f"target_force={config.GRASP_TARGET_FORCE} N/arm")

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

        # ── PHASE 3: settle contacts, then go straight into the throw ─────────
        elif phase == "reading_sensors":
            if time.time() - self.phase_ctrl.state["last_phase_change"] \
                    > config.READING_SENSORS_DURATION:
                print("✅  Grasp settled — accelerating directly into throw "
                      "(no lift phase).")
                self._prepare_throw()
            return np.zeros(self.n_act)

        # ── PHASE 4: DS + Impedance + Grasp-force throw ───────────────────────
        elif phase == "throwing_object":
            return self._throwing_phase()

        return np.zeros(self.n_act)

    # ──────────────────────────────────────────────────────────────────────────
    def _prepare_throw(self):
        """Compute release velocity and transition to throwing phase.

        x_rel comes from config.RELEASE_POINT if that's set to a fixed (3,)
        array; otherwise it defaults to the box's current (post-grasp)
        position — there is no lift step in between either way.
        """
        if getattr(config, "RELEASE_POINT", None) is not None:
            self.x_rel = np.array(config.RELEASE_POINT, dtype=float)
        else:
            self.x_rel = self.data.xpos[self.box_id].copy()

        self.v_rel = self.trajectory_planner.compute_release_velocity(
            self.x_rel, config.LANDING_POINT, config.THROW_ANGLE)

        # Normalised throw direction for arm swing
        self._throw_dir = self.v_rel / (np.linalg.norm(self.v_rel) + 1e-9)

        self.phase_ctrl.state["throw_phase_started"] = True
        self.phase_ctrl.state["throw_released"]      = False
        self.phase_ctrl.state["flight_logged"]        = False
        self.phase_ctrl.state["swing_start_sim_time"] = self.data.time
        self.phase_ctrl.set_phase("throwing_object")

        # Reset smoothing filter + release-debounce counter for this throw
        self._cmd_vel_filtered      = np.zeros(3)
        self._release_hold_counter  = 0
        self._release_reason        = ""

        # Reset the grasp-force controller: squeeze force back to nominal,
        # any leftover release ramp from a previous throw cleared.
        self.grasp_ctrl.reset()

        actual_box_pos = self.data.xpos[self.box_id].copy()
        print(f"\n  Grasp settled — arm swing starting from initial position")
        print(f"  Release pos (x_rel) : {self.x_rel}")
        if not np.allclose(self.x_rel, actual_box_pos, atol=1e-3):
            print(f"  NOTE: config.RELEASE_POINT differs from actual box "
                  f"position {actual_box_pos} — the impedance K1 term will "
                  f"pull the box toward x_rel.")
        print(f"  Target land  : {config.LANDING_POINT}")
        v = self.v_rel
        print(f"  v_rel        : [{v[0]:.3f}, {v[1]:.3f}, {v[2]:.3f}] m/s")
        d = self._throw_dir
        print(f"  throw_dir    : [{d[0]:.3f}, {d[1]:.3f}, {d[2]:.3f}]\n")

    # ──────────────────────────────────────────────────────────────────────────
    # Throw constants
    EE_SWING_SPEED     = 8.0   # m/s  — cap on commanded EE speed
    VELOCITY_THRESHOLD  = 0.90  # release ONLY when box reaches 90 % of target speed
    DIRECTION_THRESHOLD = 0.90  # cosine similarity between ẋ_o and v_rel required
                                 # to release — guards against releasing during a
                                 # contact-induced speed spike pointing the wrong way
    RELEASE_HOLD_STEPS  = 5     # consecutive steps the above must BOTH hold before
                                 # actually TRIGGERING the release ramp — filters
                                 # out single-step noise
    CMD_VEL_SMOOTHING_ALPHA = 0.25  # low-pass filter on commanded velocity, 0<a<=1
                                     # (smaller = smoother but slower to respond;
                                     # this directly targets the jerky/chattering
                                     # commanded-velocity behaviour seen in the plots)
    # NOTE: no time-based fallback anymore — release is gated purely on
    # velocity + direction (to TRIGGER the ramp) and on measured force decay
    # (to CONFIRM the box has actually separated). If the box can't reach
    # 90% of v_rel in the right direction, it will keep swinging indefinitely;
    # that's now a signal to retune K_DS/B_DS/D_IMPEDANCE/grasp force rather
    # than a state the sim silently papers over with a timeout.

    # ──────────────────────────────────────────────────────────────────────────
    def _throwing_phase(self) -> np.ndarray:
        """
        Throw driven by two things simultaneously, summed in the Jacobian
        mapping (Step 5):
          1. ThrowingDS + ThrowingImpedance -> x_dot_o_star, the commanded
             OBJECT velocity along the throw direction (mapped through G^T).
          2. GraspForceController -> x_dot_ee, per-arm squeeze velocity
             along the grasp axis, closing the force loop
             ẋ ← K⁻¹(F* − F_meas) so the arms hold the box via friction
             without applying any net force to it (force closure).

        Release sequencing:
          a) Velocity+direction condition holds for RELEASE_HOLD_STEPS
             consecutive steps  ->  triggers grasp_ctrl.start_release_ramp()
             (squeeze force F* ramps from nominal down to 0 over several
             steps — NOT an instantaneous cutoff).
          b) Once F* has ramped to 0 AND measured contact force on both arms
             has actually decayed below threshold (grasp_fully_released),
             the box is considered physically separated and we call
             _finish_release() to log the throw and switch to flight
             monitoring. No geom_contype/conaffinity hack is used — release
             is purely a consequence of friction capacity going to zero as
             the squeeze force ramps down.

        Pure velocity-tracking design (K_ds = 0, K_1 = 0):
          x_rel is only used to SOLVE for v_rel via the projectile equation —
          it no longer creates a positional pull during the swing. Both the
          DS and the impedance controller are pure velocity trackers here,
          so there's nothing fighting the velocity build-up.
        """
        if self.phase_ctrl.state["throw_released"]:
            self._monitor_flight()
            return np.zeros(self.n_act)

        _box_dof = self.model.body_dofadr[self.box_id]
        xdot_o   = self.data.qvel[_box_dof:_box_dof + 3].copy()
        elapsed  = self.data.time - self.phase_ctrl.state["swing_start_sim_time"]

        # ── measured per-arm contact force on the object (world frame) ───────
        F_meas_left, F_meas_right = self.contact_handler.get_ee_contact_force(self.box_id)

        # ── if a release ramp is already in progress, check whether contact
        #    has actually broken yet ────────────────────────────────────────────
        if self.grasp_ctrl._ramping:
            if self.grasp_ctrl.grasp_fully_released(
                    F_meas_left, F_meas_right,
                    force_threshold=config.GRASP_RELEASE_FORCE_THRESHOLD):
                self._last_commanded_vel = np.full(3, np.nan)
                self._finish_release(xdot_o)
                return np.zeros(self.n_act)

        # ── Velocity + DIRECTION release TRIGGER condition ────────────────────
        # Releasing on speed alone is unsafe: a contact-induced jerk/bounce can
        # momentarily spike the box's speed in the WRONG direction (e.g. mostly
        # Z instead of the intended Y-forward throw) and still cross the speed
        # threshold. We also require the velocity vector to be pointing close
        # to v_rel (cosine similarity), and require both conditions to hold
        # for several consecutive steps (debounce) before triggering release.
        target_speed  = np.linalg.norm(self.v_rel)
        current_speed = np.linalg.norm(xdot_o)
        vel_ratio     = current_speed / target_speed if target_speed > 1e-6 else 0.0

        if current_speed > 1e-6 and target_speed > 1e-6:
            direction_alignment = float(
                np.dot(xdot_o, self.v_rel) / (current_speed * target_speed))
        else:
            direction_alignment = 0.0

        release_ready = (vel_ratio >= self.VELOCITY_THRESHOLD and
                          direction_alignment >= self.DIRECTION_THRESHOLD)

        if release_ready:
            self._release_hold_counter += 1
        else:
            self._release_hold_counter = 0

        if (self._release_hold_counter >= self.RELEASE_HOLD_STEPS
                and not self.grasp_ctrl._ramping):
            self._release_reason = (
                f"velocity+direction threshold "
                f"({vel_ratio*100:.0f}% speed, {direction_alignment:.2f} alignment)")
            self.grasp_ctrl.start_release_ramp()
            print(f"\n  RELEASE RAMP STARTED ({self._release_reason})")

        # ── Step 2: DS desired acceleration  [Slide 7] ────────────────────────
        x_o       = self.data.xpos[self.box_id].copy()
        xddot_des = self.throwing_ds.compute_acceleration(
            x_o, xdot_o, self.x_rel, self.v_rel)

        # ── Step 3: real contact-force feedback (world-frame, from ContactHandler) ──
        w_mo_fb   = F_meas_left + F_meas_right
        w_obj_hat = self.impedance_ctrl.gravity_wrench(self.object_mass)

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

        # ── Low-pass filter the commanded velocity ────────────────────────────
        a = config.CMD_VEL_SMOOTHING_ALPHA if hasattr(config, "CMD_VEL_SMOOTHING_ALPHA") \
            else self.CMD_VEL_SMOOTHING_ALPHA
        self._cmd_vel_filtered = a * xdot_o_star + (1.0 - a) * self._cmd_vel_filtered
        xdot_o_star = self._cmd_vel_filtered

        self._last_commanded_vel = xdot_o_star.copy()

        # ── Grasp squeeze command (force closure)  ẋ ← K⁻¹(F* − F_meas) ──────
        left_ee_pos  = self.data.site_xpos[self.left_ee_id].copy()
        right_ee_pos = self.data.site_xpos[self.right_ee_id].copy()
        x_dot_ee = self.grasp_ctrl.compute_squeeze_velocities(
            left_ee_pos, right_ee_pos, F_meas_left, F_meas_right)

        # ── Step 5: q̇ = J_H† [ẋ_ee + G^T ẋ_o*]  [Slide 10] ─────────────────
        q_dot = self.dual_arm_jac.compute_joint_velocities(
            data             = self.data,
            x_dot_o_star     = xdot_o_star,
            x_dot_ee         = x_dot_ee,
            object_body_id   = self.box_id,
            left_ee_site_id  = self.left_ee_id,
            right_ee_site_id = self.right_ee_id)

        if int(elapsed * 10) % 5 == 0:
            print(f"  swing t={elapsed:.2f}s | box {current_speed:.2f}/{target_speed:.2f} m/s"
                  f" ({vel_ratio*100:.0f}%) | dir-align {direction_alignment:.2f}"
                  f" | hold {self._release_hold_counter}/{self.RELEASE_HOLD_STEPS}"
                  f" | F_L {np.linalg.norm(F_meas_left):.2f}N"
                  f" F_R {np.linalg.norm(F_meas_right):.2f}N"
                  f" | F* {self.grasp_ctrl.current_target_force:.2f}N")

        return np.clip(q_dot, -config.MAX_COMMANDED_VEL, config.MAX_COMMANDED_VEL)

    # ──────────────────────────────────────────────────────────────────────────
    def _finish_release(self, release_vel: np.ndarray):
        """
        Called once GraspForceController confirms measured contact force has
        decayed below threshold on both arms — i.e. the box has physically
        separated because friction capacity ran out as the squeeze force was
        ramped to zero, not because collision geoms were disabled.
        """
        self.phase_ctrl.state["throw_released"] = True

        self._release_sim_time = self.data.time
        self.release_log = (self.v_rel.copy(), release_vel.copy())

        actual_pos  = self.data.xpos[self.box_id].copy()
        v           = release_vel
        speed_ratio = np.linalg.norm(v) / (np.linalg.norm(self.v_rel) + 1e-9)
        print(f"\n  RELEASED ({self._release_reason or 'force decay'})")
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
        """Two professional plots: Y position tracking and Y velocity during throw."""
        if not self.log["time"]:
            print("No data logged.")
            return

        times         = np.array(self.log["time"])
        achieved      = np.array(self.log["achieved_pos"])    # (N, 3)
        achieved_vel  = np.array(self.log["achieved_vel"])    # (N, 3)
        commanded_vel = np.array(self.log["commanded_vel"])   # (N, 3) — NaN outside throw
        phases        = np.array(self.log["phase"])

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

        # Shade throwing window
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

        # commanded_vel is NaN after release — plot only where it is finite
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

                print(f"box pos: [{pos[0]:.3f}, {pos[1]:.3f}, {pos[2]:.3f}]  "
                      f"vel: [{vel[0]:.3f}, {vel[1]:.3f}, {vel[2]:.3f}]")

                viewer.sync()
                time.sleep(config.SLEEP_TIME)

        print("\nGenerating plots...")
        self.plot_results()


# ══════════════════════════════════════════════════════════════════════════════
def main():
    print("=" * 60)
    print("DS + IMPEDANCE + FORCE-CLOSURE GRASP THROWING")
    print("(direct grasp-to-throw, gravity always on)")
    print("=" * 60)
    task = ThrowingDSImpedance()
    task.run()


if __name__ == "__main__":
    main()