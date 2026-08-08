"""
Mathematically Correct Implementation of Dynamic Throwing
Follows the exact formulation from the LaTeX specification:

1. Release Velocity Computation (Ballistics):
   v_rel² = g·R² / [2·cos²θ·(R·tanθ - Δy)]
   v_rel = v_rel·cosθ·e_h + v_rel·sinθ·ẑ

2. Second-Order DS:
   ẍ_des = -K_ds(x_o - x_rel) - B_ds(ẋ_o - ẋ_o^target)

3. Impedance Control:
   ẋ_o* = ẋ_o - D^(-1)[M_o·ẍ_des + w_mo^fb - ŵ_obj + K_1(x_o* - x_o)]

4. Joint Velocity Mapping:
   q̇ = J_H† [ẋ + G^T · ẋ_o*]
"""
import os
os.environ["MUJOCO_GL"] = "glfw"

import mujoco
import numpy as np
from mujoco import viewer
import time

# Import configuration
from config import throwing_config as config

# Import controllers
from controllers.second_order_ds import SecondOrderDS
from controllers.impedance_controller import ImpedanceController
from controllers.dual_arm_jacobian import DualArmJacobian
from controllers.contact_handler import ContactHandler
from controllers.admittance_controller import AdmittanceController
from controllers.trajectory_planner import TrajectoryPlanner
from controllers.jacobian_mapper import JacobianMapper
from controllers.phase_controller import PhaseController

# Import velocity controller
from utils.mujoco_velocity_controller.samriddhi_dual_arm_velocity_controller import VelocityControllerGC


class ThrowingTaskV2:
    """
    Mathematically correct implementation of throwing task
    Uses exact formulations from the theoretical framework
    """

    def __init__(self):
        """Initialize the throwing task"""
        # Load MuJoCo model
        self.model = mujoco.MjModel.from_xml_path("/home/iitgn-robotics/Saikrishna/samriddhi_mujoco_bimanual_ds_dynamic_task/robot_description/kshitij_lifting.xml")
        self.data = mujoco.MjData(self.model)

        # Disable gravity initially (re-enable during throwing)
        self.model.opt.gravity[:] = [0.0, 0.0, 0.0]
        self.original_gravity = np.array([0.0, 0.0, -config.GRAVITY])

        # Reset simulation
        mujoco.mj_resetData(self.model, self.data)
        mujoco.mj_forward(self.model, self.data)

        # Find box body ID
        self.box_body_id = self._find_box_body_id()

        # Initialize mathematically correct controllers
        print("📐 Initializing mathematically correct controllers...")

        # Second-order DS for throwing
        self.ds_controller = SecondOrderDS(config.K_DS, config.B_DS)

        # Impedance controller for throwing
        self.impedance_controller = ImpedanceController(
            config.M_O,
            config.D_IMPEDANCE,
            config.K_IMPEDANCE
        )

        # Dual-arm Jacobian mapper
        self.dual_arm_jacobian = DualArmJacobian(
            self.model,
            12  # num_actuators
        )

        # Other controllers (for lifting phase)
        self.contact_handler = ContactHandler(self.model, self.data)
        self.admittance_controller = AdmittanceController(
            config.ADMITTANCE_STIFFNESS,
            config.ADMITTANCE_DAMPING,
            config.MAX_LINEAR_VELOCITY,
            config.MAX_ANGULAR_VELOCITY,
            config.UPWARD_FORCE_BASE,
            config.UPWARD_FORCE_SCALE
        )
        self.trajectory_planner = TrajectoryPlanner(config.GRAVITY)
        self.phase_controller = PhaseController(config)
        self.jacobian_mapper = JacobianMapper(self.model, 12)

        # Initialize velocity controller
        self.velocity_controller = VelocityControllerGC(
            self.model, self.data,
            kd=config.KD_GAINS,
            ki=config.KI_GAINS
        )

        # Initialize targets
        self.num_actuators = self.velocity_controller.num_actuators
        self.actuators_per_robot = self.num_actuators // 2
        self.phase_controller.initialize_targets(self.num_actuators)

        # Throwing state
        self.throw_release_pos = None
        self.throw_release_velocity = None

        # Find end-effector sites
        self.left_ee_id, self.right_ee_id = self.contact_handler.find_end_effector_sites()

        print("✅ ThrowingTaskV2 initialized successfully")
        print(f"   Box body ID: {self.box_body_id}")
        print(f"   Number of actuators: {self.num_actuators}")
        print(f"   Using mathematically correct formulation:")
        print(f"   - Second-order DS: ẍ_des = -K(x-x*) - B(ẋ-ẋ*)")
        print(f"   - Impedance: ẋ_o* = ẋ_o - D⁻¹[M·ẍ_des + w_fb - ŵ + K₁Δx]")
        print(f"   - Jacobian: q̇ = J_H†[ẋ + G^T·ẋ_o*]")

    def _find_box_body_id(self):
        """Find the body ID of the box"""
        for i in range(self.model.nbody):
            if "box" in self.model.body(i).name:
                return i
        return -1

    def compute_velocity_commands(self, t):
        """
        Main control function - computes velocity commands based on current phase

        Args:
            t: Current simulation time

        Returns:
            Joint velocity commands
        """
        # Get current state
        joint_positions = self._get_joint_positions()
        current_phase = self.phase_controller.get_phase()

        # Print phase periodically
        if self.phase_controller.should_print_phase():
            print(f"📍 Phase: {current_phase}")

        # ====================  PHASE: REACHING ====================
        if current_phase in ["reaching_first_target", "reaching_second_target"]:
            return self._handle_reaching_phase(joint_positions)

        # ==================== PHASE: READING SENSORS ====================
        elif current_phase == "reading_sensors":
            return self._handle_reading_sensors_phase()

        # ==================== PHASE: LIFTING ====================
        elif current_phase == "lifting_object":
            return self._handle_lifting_phase()

        # ==================== PHASE: THROWING ====================
        elif current_phase == "throwing_object":
            return self._handle_throwing_phase()

        # ==================== PHASE: IDLE ====================
        elif current_phase == "idle":
            # Robots completely stop moving after throwing (like kshitij_lifting.py)
            return np.zeros(self.num_actuators)

        else:
            return np.zeros(self.num_actuators)

    def _get_joint_positions(self):
        """Get current joint positions"""
        joint_positions = np.zeros(self.num_actuators)
        for i in range(self.num_actuators):
            joint_id = self.model.actuator_trnid[i, 0]
            joint_positions[i] = self.data.qpos[self.model.jnt_qposadr[joint_id]]
        return joint_positions

    def _handle_reaching_phase(self, joint_positions):
        """Handle reaching phase logic"""
        velocity_commands = self.phase_controller.compute_position_control(
            joint_positions, self.num_actuators
        )

        if self.phase_controller.check_reaching_complete(joint_positions, self.num_actuators):
            current_phase = self.phase_controller.get_phase()

            if current_phase == "reaching_first_target":
                self.phase_controller.state["current_targets"] = \
                    self.phase_controller.state["second_targets"].copy()
                self.phase_controller.set_phase("reaching_second_target")
                return np.zeros(self.num_actuators)

            elif current_phase == "reaching_second_target":
                current_time = time.time()
                if current_time - self.phase_controller.state["last_phase_change"] > 0.1:
                    self.phase_controller.set_phase("reading_sensors")
                    return np.zeros(self.num_actuators)

        return velocity_commands

    def _handle_reading_sensors_phase(self):
        """Handle reading sensors phase"""
        current_time = time.time()
        if current_time - self.phase_controller.state["last_phase_change"] > config.READING_SENSORS_DURATION:
            self.phase_controller.set_phase("lifting_object")
        return np.zeros(self.num_actuators)

    def _handle_lifting_phase(self):
        """Handle lifting phase logic"""
        current_box_height = self.data.xpos[self.box_body_id][2]

        # Check if lift complete
        if self.phase_controller.check_lift_complete(current_box_height, config.TARGET_LIFT_HEIGHT):
            print(f"⚠️ LIFT TARGET REACHED! Box at {current_box_height:.3f}m")

            if not self.phase_controller.state["throw_phase_started"]:
                # Compute throw trajectory
                self.throw_release_pos = self.data.xpos[self.box_body_id].copy()
                self.throw_release_velocity = self.trajectory_planner.compute_release_velocity(
                    self.throw_release_pos,
                    config.LANDING_POINT,
                    config.THROW_ANGLE
                )

                # Enable gravity
                self.model.opt.gravity[:] = self.original_gravity

                # Transition to throwing
                self.phase_controller.set_phase("throwing_object")

                print(f"🎯 Starting throw with mathematically correct implementation")
                print(f"   Release position: {self.throw_release_pos}")
                print(f"   Target velocity: {self.throw_release_velocity}")

            return np.zeros(self.num_actuators)

        # Continue lifting with admittance control
        left_wrench, right_wrench, contact = self.contact_handler.get_object_contact_wrenches(
            self.box_body_id
        )

        if contact:
            object_velocity, W, W_desired, pose_error = \
                self.admittance_controller.compute_object_velocity(
                    self.model, self.data,
                    left_wrench, right_wrench,
                    self.box_body_id,
                    config.TARGET_LIFT_HEIGHT
                )

            joint_velocities = self.jacobian_mapper.object_velocity_to_joint_velocities_lifting(
                self.data, object_velocity,
                self.left_ee_id, self.right_ee_id
            )

            return joint_velocities
        else:
            return np.zeros(self.num_actuators)

    def _handle_throwing_phase(self):
        """
        Handle throwing phase using mathematically correct formulation:

        1. DS computes acceleration: ẍ_des = -K_ds(x_o - x_rel) - B_ds(ẋ_o - ẋ_o^target)
        2. Impedance computes velocity: ẋ_o* = ẋ_o + D⁻¹[M_o·ẍ_des + w_mo^fb - ŵ_obj + K_1(x_o* - x_o)]
        3. Jacobian maps to joints: q̇ = J†·ẋ_o* (using proven kshitij method)
        """
        # Get current object state
        x_o = self.data.xpos[self.box_body_id].copy()  # Current position
        x_dot_o = self.data.cvel[self.box_body_id, :3].copy()  # Current velocity

        # Get force feedback from contacts
        left_wrench, right_wrench, contact = self.contact_handler.get_object_contact_wrenches(
            self.box_body_id
        )

        # Check if contact lost AND velocity is low - then stop
        velocity_magnitude = np.linalg.norm(x_dot_o)
        if not contact and velocity_magnitude < 0.5:
            print(f"🚀 Box released! Final velocity: {velocity_magnitude:.3f} m/s")

            # Disable robot-box collisions to cleanly release the box
            for i in range(self.model.ngeom):
                geom_name = self.model.geom(i).name
                if "left" in geom_name or "right" in geom_name:
                    self.model.geom_conaffinity[i] = 0

            self.phase_controller.set_phase("idle")
            # Reset velocity controller integral to stop robots immediately
            self.velocity_controller.reset_integral()
            return np.zeros(self.num_actuators)

        # Step 1: Compute desired acceleration using second-order DS
        x_ddot_des = self.ds_controller.compute_desired_acceleration(
            x_o=x_o,
            x_dot_o=x_dot_o,
            x_rel=self.throw_release_pos,
            x_dot_o_target=self.throw_release_velocity
        )

        # Print debug info periodically
        if hasattr(self, '_throw_debug_counter'):
            self._throw_debug_counter += 1
        else:
            self._throw_debug_counter = 0

        if self._throw_debug_counter % 50 == 0:
            print(f"🎯 Throwing: vel={velocity_magnitude:.3f}, contact={contact}, "
                  f"target_vel={np.linalg.norm(self.throw_release_velocity):.3f}")
            print(f"   x_ddot_des: {x_ddot_des}")
            print(f"   x_dot_o (current): {x_dot_o}")

        # Combined force feedback (only linear forces)
        w_mo_fb = (left_wrench[:3] + right_wrench[:3])

        # Estimate gravity wrench
        w_obj_hat = self.impedance_controller.estimate_gravity_wrench(
            config.OBJECT_MASS,
            config.GRAVITY
        )

        # Desired position (keep moving toward release point)
        x_o_star = self.throw_release_pos

        # Step 2: Compute commanded object velocity using impedance control
        x_dot_o_star = self.impedance_controller.compute_object_velocity(
            x_dot_o=x_dot_o,
            x_ddot_des=x_ddot_des,
            w_mo_fb=w_mo_fb,
            w_obj_hat=w_obj_hat,
            x_o_star=x_o_star,
            x_o=x_o
        )

        # Apply safety limits
        x_dot_o_star = np.clip(x_dot_o_star, -config.MAX_COMMANDED_VEL, config.MAX_COMMANDED_VEL)

        if self._throw_debug_counter % 50 == 0:
            print(f"   x_dot_o_star (commanded): {x_dot_o_star}")

        # Step 3: Map to joint velocities using dual-arm Jacobian formulation
        # q̇ = J_H† [ẋ + G^T · ẋ_o*]
        try:
            # Use dual-arm Jacobian with grasp matrix
            x_dot_ee = np.zeros(12)  # No additional end-effector velocity
            q_dot = self.dual_arm_jacobian.compute_joint_velocities(
                self.data,
                x_dot_o_star,  # 3D commanded object velocity
                x_dot_ee,  # 12D end-effector velocities (zeros)
                self.box_body_id,
                self.left_ee_id,
                self.right_ee_id
            )

            # Apply joint velocity limits
            max_vel = 3.0  # rad/s
            q_dot = np.clip(q_dot, -max_vel, max_vel)

            return q_dot

        except Exception as e:
            print(f"Error computing joint velocities: {e}")
            return np.zeros(self.num_actuators)

    def run(self):
        """Run the throwing task"""
        # Set velocity trajectory
        self.velocity_controller.set_velocity_trajectory(self.compute_velocity_commands)

        # Register control callback
        mujoco.set_mjcb_control(self.velocity_controller.control_callback)

        # Launch viewer
        with viewer.launch_passive(self.model, self.data) as v:
            while True:
                mujoco.mj_step(self.model, self.data)

                # Print box state
                if self.box_body_id != -1:
                    pos = self.data.xpos[self.box_body_id].copy()
                    if hasattr(self.data, "cvel"):
                        lin_vel = self.data.cvel[self.box_body_id, :3].copy()
                    else:
                        lin_vel = np.zeros(3)

                    print(f"box pos: [{pos[0]:.3f}, {pos[1]:.3f}, {pos[2]:.3f}], "
                          f"vel: [{lin_vel[0]:.3f}, {lin_vel[1]:.3f}, {lin_vel[2]:.3f}]")

                v.sync()
                time.sleep(config.SLEEP_TIME)


def main():
    """Main entry point"""
    print("=" * 60)
    print("MATHEMATICALLY CORRECT THROWING TASK (V2)")
    print("=" * 60)
    print()
    print("Implementation follows exact mathematical formulation:")
    print("1. Ballistic trajectory computation")
    print("2. Second-order DS: ẍ_des = -K(x-x*) - B(ẋ-ẋ*)")
    print("3. Impedance control: ẋ_o* = ẋ_o - D⁻¹[M·ẍ + forces]")
    print("4. Dual-arm Jacobian: q̇ = J_H†[ẋ + G^T·ẋ_o*]")
    print("=" * 60)
    print()

    task = ThrowingTaskV2()
    task.run()


if __name__ == "__main__":
    main()
