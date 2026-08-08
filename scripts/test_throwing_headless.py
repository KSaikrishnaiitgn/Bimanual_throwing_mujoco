"""
Test of throwing_modular_2.py behavior (non-interactive)
"""
import os
os.environ["MUJOCO_GL"] = "glfw"

import mujoco
import numpy as np
from config import throwing_config as config
from controllers.second_order_ds import SecondOrderDS
from controllers.impedance_controller import ImpedanceController
from controllers.dual_arm_jacobian import DualArmJacobian
from controllers.phase_controller import PhaseController
from utils.mujoco_velocity_controller.samriddhi_dual_arm_velocity_controller import VelocityControllerGC


class HeadlessThrowingTest:
    """Test throwing without viewer"""

    def __init__(self):
        # Load model
        self.model = mujoco.MjModel.from_xml_path(config.XML_FILE_PATH)
        self.data = mujoco.MjData(self.model)

        self.num_actuators = 12
        self.actuators_per_robot = 6

        # Find components
        self.left_ee_id = -1
        self.right_ee_id = -1
        for i in range(self.model.nsite):
            site_name = self.model.site(i).name
            if "left" in site_name.lower() and self.left_ee_id == -1:
                self.left_ee_id = i
            elif "right" in site_name.lower() and self.right_ee_id == -1:
                self.right_ee_id = i

        self.box_body_id = -1
        for i in range(self.model.nbody):
            if "box" in self.model.body(i).name:
                self.box_body_id = i
                break

        print(f"Found: left_ee={self.left_ee_id}, right_ee={self.right_ee_id}, box={self.box_body_id}")

        # Controllers
        self.phase_controller = PhaseController(config)
        self.ds_controller = SecondOrderDS(config.K_DS, config.B_DS)
        self.impedance_controller = ImpedanceController(
            config.M_O,  # Use the matrix, not the scalar
            config.D_IMPEDANCE,
            config.K_IMPEDANCE
        )
        self.dual_arm_jacobian = DualArmJacobian(self.model, 12)
        self.velocity_controller = VelocityControllerGC(
            self.model, self.data,
            config.KD_GAINS, config.KI_GAINS
        )

        # Compute throw parameters
        self.compute_throw_parameters()

        # Go to first target (initialization)
        self.go_to_position(
            np.concatenate([
                config.FIRST_TARGET_POSITIONS_LEFT,
                config.FIRST_TARGET_POSITIONS_RIGHT
            ])
        )
        print("Reached first target")

        # Go to second target (pre-throw)
        self.go_to_position(
            np.concatenate([
                config.SECOND_TARGET_POSITIONS_LEFT,
                config.SECOND_TARGET_POSITIONS_RIGHT
            ])
        )
        print("Reached second target (pre-throw configuration)")

        # Lift to target height
        self.lift_to_height()
        print("Lifted to target height")

        # Set to throwing phase
        self.phase_controller.set_phase("throwing")

    def compute_throw_parameters(self):
        """Compute ballistic throw parameters"""
        R = np.linalg.norm(config.LANDING_POINT[:2])
        delta_y = config.LANDING_POINT[2]
        theta = config.THROW_ANGLE
        g = config.GRAVITY

        v_rel_mag_sq = g * R**2 / (2 * np.cos(theta)**2 * (R * np.tan(theta) - delta_y))
        v_rel_mag = np.sqrt(v_rel_mag_sq) * config.VELOCITY_SCALE

        e_h = config.LANDING_POINT[:2] / np.linalg.norm(config.LANDING_POINT[:2])
        self.throw_release_velocity = np.zeros(3)
        self.throw_release_velocity[:2] = v_rel_mag * np.cos(theta) * e_h
        self.throw_release_velocity[2] = v_rel_mag * np.sin(theta)

        print(f"Release velocity: {self.throw_release_velocity}")

    def go_to_position(self, target_pos):
        """Move to target joint positions"""
        for _ in range(500):  # Enough steps to converge
            current_pos = np.zeros(12)
            for i in range(12):
                joint_id = self.model.actuator_trnid[i, 0]
                current_pos[i] = self.data.qpos[self.model.jnt_qposadr[joint_id]]

            error = target_pos - current_pos
            ctrl = config.KP_GAINS * error
            ctrl = np.clip(ctrl, -config.MAX_JOINT_VELOCITIES, config.MAX_JOINT_VELOCITIES)

            self.data.ctrl[:] = ctrl
            mujoco.mj_step(self.model, self.data)

            if np.all(np.abs(error) < config.POSITION_THRESHOLDS):
                break

    def lift_to_height(self):
        """Lift box to target height (simplified)"""
        for _ in range(200):
            mujoco.mj_step(self.model, self.data)

    def compute_throwing_velocity(self):
        """Compute joint velocities for throwing phase"""
        # Get object state
        x_o = self.data.xpos[self.box_body_id].copy()
        x_dot_o = self.data.cvel[self.box_body_id, :3].copy()

        # Set release position to current position
        x_rel = x_o.copy()

        # DS acceleration
        x_ddot_des = self.ds_controller.compute_desired_acceleration(
            x_o, x_dot_o, x_rel, self.throw_release_velocity
        )

        # Impedance control (simplified - no force feedback for headless test)
        x_dot_o_star = self.impedance_controller.compute_object_velocity(
            x_dot_o, x_ddot_des,
            w_mo_fb=np.zeros(3),
            w_obj_hat=np.zeros(3),
            x_o_star=x_rel,
            x_o=x_o
        )

        # Dual-arm Jacobian
        q_dot = self.dual_arm_jacobian.compute_joint_velocities(
            self.data, x_dot_o_star, np.zeros(12),
            self.box_body_id, self.left_ee_id, self.right_ee_id
        )

        return q_dot, x_o, x_dot_o, x_dot_o_star

    def run_throwing_test(self, num_steps=300):
        """Run throwing test for a fixed number of steps"""
        print("\n" + "="*60)
        print("THROWING PHASE TEST")
        print("="*60)

        for step in range(num_steps):
            q_dot, x_o, x_dot_o, x_dot_o_star = self.compute_throwing_velocity()

            # Apply velocities
            self.data.ctrl[:] = q_dot

            # Step simulation
            mujoco.mj_step(self.model, self.data)

            # Print status every 50 steps
            if step % 50 == 0:
                print(f"Step {step:3d}: pos=[{x_o[0]:6.3f}, {x_o[1]:6.3f}, {x_o[2]:6.3f}], "
                      f"vel=[{x_dot_o[0]:6.3f}, {x_dot_o[1]:6.3f}, {x_dot_o[2]:6.3f}], "
                      f"cmd=[{x_dot_o_star[0]:6.3f}, {x_dot_o_star[1]:6.3f}, {x_dot_o_star[2]:6.3f}]")

        print("\n" + "="*60)
        print("Final box state:")
        x_o = self.data.xpos[self.box_body_id].copy()
        x_dot_o = self.data.cvel[self.box_body_id, :3].copy()
        print(f"  Position: [{x_o[0]:.3f}, {x_o[1]:.3f}, {x_o[2]:.3f}]")
        print(f"  Velocity: [{x_dot_o[0]:.3f}, {x_dot_o[1]:.3f}, {x_dot_o[2]:.3f}]")
        print(f"  Target landing: {config.LANDING_POINT}")
        print("="*60)


def main():
    print("Headless Throwing Test")
    print("="*60)

    test = HeadlessThrowingTest()
    test.run_throwing_test(num_steps=300)

    print("\nTest complete!")


if __name__ == "__main__":
    main()
