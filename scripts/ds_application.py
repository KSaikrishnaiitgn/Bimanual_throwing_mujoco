"""
This is the corrected version of the modular throwing task using MuJoCo.
It includes proper handling of the throwing phase to ensure the robots release the box
and transition to an idle state after the throw.
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
from controllers.dynamics_system import DynamicalSystem
from controllers.contact_handler import ContactHandler
from controllers.admittance_controller import AdmittanceController
from controllers.trajectory_planner import TrajectoryPlanner
from controllers.jacobian_mapper import JacobianMapper
from controllers.phase_controller import PhaseController

# Import velocity controller
from utils.mujoco_velocity_controller.samriddhi_dual_arm_velocity_controller import VelocityControllerGC


class ThrowingTask:
    """Main class orchestrating the throwing task"""

    def __init__(self):
        """Initialize the throwing task"""
        # Load MuJoCo model
        self.model = mujoco.MjModel.from_xml_path(config.XML_FILE_PATH)
        self.data = mujoco.MjData(self.model)

        # Disable gravity initially (re-enable during throwing)
        self.model.opt.gravity[:] = [0.0, 0.0, 0.0]
        self.original_gravity = np.array([0.0, 0.0, -config.GRAVITY])

        # Reset simulation
        mujoco.mj_resetData(self.model, self.data)
        mujoco.mj_forward(self.model, self.data)

        # Find box body ID
        self.box_body_id = self._find_box_body_id()

        # Initialize controllers
        self.ds_controller = DynamicalSystem(config.K_DS, config.B_DS, config.MAX_OBJ_VEL)
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

        # Initialize velocity controller
        self.velocity_controller = VelocityControllerGC(
            self.model, self.data,
            kd=config.KD_GAINS,
            ki=config.KI_GAINS
        )

        # Initialize Jacobian mapper
        self.jacobian_mapper = JacobianMapper(
            self.model,
            self.velocity_controller.num_actuators
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

        print("✅ ThrowingTask initialized successfully")
        print(f"   Box body ID: {self.box_body_id}")
        print(f"   Number of actuators: {self.num_actuators}")

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
            print("Robots in idle phase.")
            return np.zeros(self.num_actuators)  # Stop all motion

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
        # Compute position control
        velocity_commands = self.phase_controller.compute_position_control(
            joint_positions, self.num_actuators
        )

        # Check if reached
        if self.phase_controller.check_reaching_complete(joint_positions, self.num_actuators):
            current_phase = self.phase_controller.get_phase()

            if current_phase == "reaching_first_target":
                # Transition to second target
                self.phase_controller.state["current_targets"] = \
                    self.phase_controller.state["second_targets"].copy()
                self.phase_controller.set_phase("reaching_second_target")
                return np.zeros(self.num_actuators)

            elif current_phase == "reaching_second_target":
                # Transition to reading sensors
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

                # # NEGATE Y-COMPONENT: Make robots move forward instead of backward
                # self.throw_release_velocity[1] = -self.throw_release_velocity[1]

                # Reset DS controller
                self.ds_controller.reset()

                # Enable gravity
                self.model.opt.gravity[:] = self.original_gravity

                # Transition to throwing
                self.phase_controller.set_phase("throwing_object")

                print(f"🎯 Starting throw")
                print(f"   Target velocity: {self.throw_release_velocity}")

            return np.zeros(self.num_actuators)

        # Continue lifting with admittance control
        left_wrench, right_wrench, contact = self.contact_handler.get_object_contact_wrenches(
            self.box_body_id
        )

        if contact:
            # Compute desired object velocity
            object_velocity, W, W_desired, pose_error = \
                self.admittance_controller.compute_object_velocity(
                    self.model, self.data,
                    left_wrench, right_wrench,
                    self.box_body_id,
                    config.TARGET_LIFT_HEIGHT
                )

            # Map to joint velocities
            joint_velocities = self.jacobian_mapper.object_velocity_to_joint_velocities_lifting(
                self.data, object_velocity,
                self.left_ee_id, self.right_ee_id
            )

            return joint_velocities
        else:
            return np.zeros(self.num_actuators)

    def _handle_throwing_phase(self):
        """
        Handle throwing phase logic - improved to stop robots immediately after release.

        Strategy:
        1. Build up box velocity using DS controller until sufficient velocity is reached
        2. Once velocity > 1.5 m/s, release immediately and stop robots
        3. If contact is lost prematurely, stop
        """
        # Get current object state
        current_vel = self.data.cvel[self.box_body_id, :3].copy()
        vel_magnitude = np.linalg.norm(current_vel)

        # Check contact state
        _, _, contact = self.contact_handler.get_object_contact_wrenches(self.box_body_id)

        # Print velocity every 50 steps for debugging
        if not hasattr(self, '_throw_debug_counter'):
            self._throw_debug_counter = 0
        self._throw_debug_counter += 1

        if self._throw_debug_counter % 50 == 0:
            print(f"Throwing: vel={vel_magnitude:.3f} m/s, contact={contact}")

        # NEW: Check if box has achieved sufficient throwing velocity
        # Increased threshold to 1.5 m/s for a proper throw
        if vel_magnitude > 1.5:  # Box is moving fast enough (successful throw)
            print(f"🚀 Box released! Velocity: {vel_magnitude:.3f} m/s - Robots stopping immediately.")

            # Don't disable collisions - let box naturally separate
            # Box is moving fast enough that it will break contact naturally

            # Transition to idle - robots will stop moving
            self.phase_controller.set_phase("idle")
            return np.zeros(self.num_actuators)

        # If contact is lost but velocity is still low, also stop (failed throw scenario)
        if not contact and vel_magnitude < 1.5:
            print(f"⚠️ Contact lost but velocity low ({vel_magnitude:.3f} m/s). Stopping.")
            self.phase_controller.set_phase("idle")
            return np.zeros(self.num_actuators)

        # Continue throwing motion - compute DS velocity
        # Use release position for both current and target to nullify position error
        # This makes DS purely velocity-driven: acceleration = -B @ (vel - target_vel)
        obj_vel = self.ds_controller.compute_desired_velocity(
            self.throw_release_pos,  # Use fixed release position
            current_vel,
            self.throw_release_pos,  # Same as "current" position = zero position error
            self.throw_release_velocity,
            self.model.opt.timestep
        )
        

        # Map to joint velocities
        joint_velocities = self.jacobian_mapper.object_velocity_to_joint_velocities_throwing(
            self.data, obj_vel,
            self.left_ee_id, self.right_ee_id
        )

        return joint_velocities

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
    print("MODULAR THROWING TASK")
    print("=" * 60)

    task = ThrowingTask()
    task.run()


if __name__ == "__main__":
    main()