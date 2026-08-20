"""Experimental trajectory-based throw state, leaving state_machine.py intact."""

from __future__ import annotations

import numpy as np

import config.throwing_config as base_config
import config.throwing_trajectory_config as exp_config
import release_velocity
from contact_admittance import compute_admittance_velocity
from dual_arm_kinematics import build_stacked_jacobian, damped_pinv
from state_machine import Phase, ThrowStateMachine, _quat_error_vec
from terminal_throw_controller import (
    CubicTerminalTrajectory,
    compute_terminal_velocity_command,
    rigid_contact_twist_map,
)


class TrajectoryThrowStateMachine(ThrowStateMachine):
    def __init__(self, interface):
        super().__init__(interface)
        self.release_trajectory = None
        self.terminal_position_error = None
        self.terminal_velocity_error = None
        self.actual_release_position = np.full(3, np.nan)
        self.actual_release_velocity = np.full(3, np.nan)
        # Public diagnostics consumed only by the experimental logger.
        nan3 = np.full(3, np.nan)
        nan6 = np.full(6, np.nan)
        self.diag_x_ref = nan3.copy()
        self.diag_v_ref = nan3.copy()
        self.diag_v_obj_cmd = nan3.copy()
        self.diag_hand_twist_left = nan6.copy()
        self.diag_hand_twist_right = nan6.copy()

    def _initialize_release_trajectory(self, t, x, v):
        self.p_rel = exp_config.RELEASE_POINT.copy()
        solution = release_velocity.compute_release_velocity(
            self.p_rel, exp_config.LANDING_POINT, exp_config.THROW_ANGLE
        )
        self.v_rel_vector = solution["v_rel_vector"]
        self.v_rel_scalar = solution["v_rel_scalar"]
        self.t_f = solution["t_f"]
        self.t_throw_start = t
        self.release_trajectory = CubicTerminalTrajectory(
            x, v, self.p_rel, self.v_rel_vector,
            exp_config.THROW_TRAJECTORY_DURATION,
        )
        print(
            f"[TRAJECTORY THROW] duration={exp_config.THROW_TRAJECTORY_DURATION:.3f}s "
            f"release={self.p_rel} velocity={self.v_rel_vector} "
            f"landing={exp_config.LANDING_POINT}"
        )

    def _step_throw(self, t: float, dt: float):
        state = self.interface.get_object_state()
        x, v = state["pos"], state["linvel"]
        if self.release_trajectory is None:
            self._initialize_release_trajectory(t, x, v)

        elapsed = t - self.t_throw_start
        x_ref, v_ref, _ = self.release_trajectory.sample(elapsed)
        v_obj_cmd = compute_terminal_velocity_command(x, v, x_ref, v_ref)
        self.diag_x_ref = x_ref.copy()
        self.diag_v_ref = v_ref.copy()
        self.diag_v_obj_cmd = v_obj_cmd.copy()
        object_twist_cmd = np.concatenate([v_obj_cmd, np.zeros(3)])

        F_left = self.interface.get_wrench_world(
            "left", base_config.LEFT_FORCE_SENSOR, base_config.LEFT_TORQUE_SENSOR
        )
        F_right = self.interface.get_wrench_world(
            "right", base_config.RIGHT_FORCE_SENSOR, base_config.RIGHT_TORQUE_SENSOR
        )
        adm_left = compute_admittance_velocity(
            self.force_closer.target_wrench("left"), F_left,
            axis=base_config.LEFT_PAD_APPROACH_AXIS_WORLD,
        )
        adm_right = compute_admittance_velocity(
            self.force_closer.target_wrench("right"), F_right,
            axis=base_config.RIGHT_PAD_APPROACH_AXIS_WORLD,
        )

        # Preserve the grasp orientations while the compliant squeeze acts only
        # along each pad normal.
        _, quat_left = self.interface.get_site_pose(base_config.LEFT_EE_SITE)
        _, quat_right = self.interface.get_site_pose(base_config.RIGHT_EE_SITE)
        adm_left[3:] = base_config.K_SQUEEZE_HOLD_ANG * _quat_error_vec(
            self.squeeze_target_quat_left, quat_left
        )
        adm_right[3:] = base_config.K_SQUEEZE_HOLD_ANG * _quat_error_vec(
            self.squeeze_target_quat_right, quat_right
        )

        box_pos = x
        left_pos, _ = self.interface.get_site_pose(base_config.LEFT_EE_SITE)
        right_pos, _ = self.interface.get_site_pose(base_config.RIGHT_EE_SITE)
        H = rigid_contact_twist_map(left_pos - box_pos, right_pos - box_pos)
        desired_hand_twists = np.concatenate([adm_left, adm_right]) + H @ object_twist_cmd
        self.diag_hand_twist_left = desired_hand_twists[:6].copy()
        self.diag_hand_twist_right = desired_hand_twists[6:].copy()

        J = build_stacked_jacobian(self.interface)
        qdot_cmd = damped_pinv(J) @ desired_hand_twists
        qdot_left, qdot_right = qdot_cmd[:7], qdot_cmd[7:]
        self._update_vel_clipped_flags(qdot_left, qdot_right)

        torque_left = self.pid_left.step(
            self.interface.get_qvel("left"), qdot_left, dt,
            bias_torque=self.interface.get_bias_torque("left"),
        )
        torque_right = self.pid_right.step(
            self.interface.get_qvel("right"), qdot_right, dt,
            bias_torque=self.interface.get_bias_torque("right"),
        )

        pos_error = float(np.linalg.norm(x - self.p_rel))
        vel_error = float(np.linalg.norm(v - self.v_rel_vector))
        self.terminal_position_error = pos_error
        self.terminal_velocity_error = vel_error
        in_terminal_window = elapsed >= exp_config.THROW_TRAJECTORY_DURATION
        if (
            in_terminal_window
            and pos_error < exp_config.RELEASE_POSITION_TOLERANCE
            and vel_error < exp_config.RELEASE_VELOCITY_TOLERANCE
        ):
            self.actual_release_position = x.copy()
            self.actual_release_velocity = v.copy()
            self.force_closer.deactivate()
            self.release_time = t
            self.phase = Phase.RELEASE
            self._reset_velocity_pid()
            print(
                f"[TERMINAL RELEASE t={t:.3f}] position_error={pos_error:.4f}m "
                f"velocity_error={vel_error:.4f}m/s actual_v={v}"
            )
        elif elapsed > exp_config.THROW_TRAJECTORY_DURATION + exp_config.RELEASE_WINDOW:
            # Do not release at the wrong terminal state. Continue tracking and
            # expose the miss in diagnostics instead of silently throwing badly.
            if int(round(t / dt)) % 250 == 0:
                print(
                    f"[TERMINAL MISS t={t:.3f}] position_error={pos_error:.4f}m "
                    f"velocity_error={vel_error:.4f}m/s"
                )
        return torque_left, torque_right

    def _step_release(self, t: float, dt: float):
        """Open the grasp without removing the box's launch momentum."""
        elapsed = t - self.release_time
        state = self.interface.get_object_state()
        box_pos = state["pos"]
        left_pos, _ = self.interface.get_site_pose(base_config.LEFT_EE_SITE)
        right_pos, _ = self.interface.get_site_pose(base_config.RIGHT_EE_SITE)
        H = rigid_contact_twist_map(left_pos - box_pos, right_pos - box_pos)
        release_twist = np.concatenate([self.v_rel_vector, np.zeros(3)])
        hand_twists = H @ release_twist

        # Add equal and opposite opening velocities while retaining the full
        # forward/upward release velocity at both hands.
        hand_twists[:3] += (
            -exp_config.RELEASE_OUTWARD_SPEED
            * base_config.LEFT_PAD_APPROACH_AXIS_WORLD
        )
        hand_twists[6:9] += (
            -exp_config.RELEASE_OUTWARD_SPEED
            * base_config.RIGHT_PAD_APPROACH_AXIS_WORLD
        )
        self.diag_x_ref = self.p_rel.copy()
        self.diag_v_ref = self.v_rel_vector.copy()
        self.diag_v_obj_cmd = self.v_rel_vector.copy()
        self.diag_hand_twist_left = hand_twists[:6].copy()
        self.diag_hand_twist_right = hand_twists[6:].copy()

        J = build_stacked_jacobian(self.interface)
        qdot = damped_pinv(J) @ hand_twists
        qdot_left, qdot_right = qdot[:7], qdot[7:]
        self._update_vel_clipped_flags(qdot_left, qdot_right)
        torque_left = self.pid_left.step(
            self.interface.get_qvel("left"), qdot_left, dt,
            bias_torque=self.interface.get_bias_torque("left"),
        )
        torque_right = self.pid_right.step(
            self.interface.get_qvel("right"), qdot_right, dt,
            bias_torque=self.interface.get_bias_torque("right"),
        )
        if elapsed >= exp_config.RELEASE_COMOVE_DURATION:
            self.phase = Phase.FOLLOW_THROUGH
            self.traj_left = None
            self.traj_right = None
            self.traj_start_time = None
            self._reset_velocity_pid()
        return torque_left, torque_right

    def _step_done(self):
        """Hold both arms stationary instead of dropping torque to zero."""
        dt = base_config.SIM_TIMESTEP
        torque_left = self.pid_left.step(
            self.interface.get_qvel("left"), np.zeros(7), dt,
            bias_torque=self.interface.get_bias_torque("left"),
        )
        torque_right = self.pid_right.step(
            self.interface.get_qvel("right"), np.zeros(7), dt,
            bias_torque=self.interface.get_bias_torque("right"),
        )
        return torque_left, torque_right
