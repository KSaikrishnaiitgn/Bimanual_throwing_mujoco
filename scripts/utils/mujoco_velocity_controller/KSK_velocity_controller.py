import mujoco
import numpy as np

class VelocityControllerGC:
    """
    A velocity controller for MuJoCo robots with gravity compensation.
    Based on the working example with additional filtering and control terms.

    NEW: accepts an optional `joint_vel_limits` array (rad/s, one per
    actuator). This is enforced INSIDE control_callback, independent of
    whatever upstream phase/controller produced the velocity target. This
    matters for hardware deployment: even if a bug, a bad gain, or a future
    change elsewhere in the pipeline produces an unsafe target, this
    controller will not command the joint past its rated velocity.
    """
    def __init__(self, model, data, kd=None, ki=0.05, joint_vel_limits=None):
        """
        Initialize the controller.

        Args:
            model: MuJoCo model
            data: MuJoCo data
            kd: Derivative gain (array of size num_actuators or scalar)
            ki: Integral gain (array of size num_actuators or scalar)
            joint_vel_limits: Optional array (num_actuators,) of maximum
                allowed |velocity| per joint, in rad/s. If provided, every
                velocity target passed to this controller (whether from a
                trajectory function or set_velocity_target) is clamped to
                these limits before any torque is computed. If None, no
                clamping is applied here (relies entirely on upstream
                clipping — NOT recommended for hardware).
        """
        self.model = model
        self.data = data
        self.num_actuators = model.nu

        # Set default gains if not provided
        if kd is None:
            self.kd = np.ones(self.num_actuators) * 100.0
        elif np.isscalar(kd):
            self.kd = np.ones(self.num_actuators) * kd
        else:
            self.kd = kd

        # Set integral gains
        if np.isscalar(ki):
            self.ki = np.ones(self.num_actuators) * ki
        else:
            self.ki = ki

        # ── Hardware joint velocity limits (rad/s) ─────────────────────────
        if joint_vel_limits is None:
            print("⚠️  WARNING: VelocityControllerGC initialised WITHOUT "
                  "joint_vel_limits. No hardware velocity safety net is "
                  "active in this controller. Strongly recommended to pass "
                  "config.MAX_JOINT_VELOCITIES when deploying on real "
                  "hardware.")
            self.joint_vel_limits = None
        else:
            limits = np.asarray(joint_vel_limits, dtype=float)
            if limits.shape[0] != self.num_actuators:
                raise ValueError(
                    f"joint_vel_limits has {limits.shape[0]} entries but "
                    f"model has {self.num_actuators} actuators.")
            if np.any(limits <= 0):
                raise ValueError("joint_vel_limits must be strictly positive.")
            self.joint_vel_limits = limits

        # Default velocity targets
        self.v_targets = np.zeros(self.num_actuators, dtype=float)

        # Integral of velocity error (for overcoming static friction)
        self.v_error_integral = np.zeros(self.num_actuators, dtype=float)

        # Maximum integral term to prevent windup
        self.integral_limit = 0.2

        # Trajectory function (optional)
        self.trajectory_function = None

        # For debugging
        self.debug = False

        # Filtered velocity (for noise reduction)
        self.filtered_velocity = np.zeros(self.num_actuators, dtype=float)
        self.filter_coeff = 0.7

        # Store DOF indices for efficiency
        joint_ids = model.actuator_trnid[:, 0]
        self.dof_indices = model.jnt_dofadr[joint_ids]

        # Tracks whether the limiter clipped anything on the last call —
        # useful for logging / debugging on hardware.
        self.last_clip_active = np.zeros(self.num_actuators, dtype=bool)

    def reset_integral(self):
        """Reset integral term to zero (useful when stopping or changing phases)"""
        self.v_error_integral = np.zeros(self.num_actuators, dtype=float)
        self.filtered_velocity = np.zeros(self.num_actuators, dtype=float)

    def set_velocity_target(self, v_targets):
        """Set target velocities for all actuators (clamped to joint limits if set)."""
        if len(v_targets) != self.num_actuators:
            raise ValueError(f"Expected {self.num_actuators} velocity targets, got {len(v_targets)}")
        self.v_targets = self._apply_joint_limits(np.array(v_targets, dtype=float))

    def set_velocity_trajectory(self, trajectory_function):
        """Set a time-varying velocity trajectory function."""
        self.trajectory_function = trajectory_function

    def _apply_joint_limits(self, v_targets: np.ndarray) -> np.ndarray:
        """
        Clamp a raw velocity-target vector to self.joint_vel_limits.

        Uses a simple per-joint np.clip here (NOT a direction-preserving
        scale). At this final safety-net stage, protecting each individual
        joint from exceeding its rated speed takes priority over preserving
        the exact direction of the commanded motion — by this point,
        upstream code (e.g. the throw controller's own uniform scale-down)
        should already have handled direction-preserving clamping for
        task-level motions. This layer is the last line of defense before
        actuator commands are computed.
        """
        if self.joint_vel_limits is None:
            self.last_clip_active = np.zeros(self.num_actuators, dtype=bool)
            return v_targets

        clipped = np.clip(v_targets, -self.joint_vel_limits, self.joint_vel_limits)
        self.last_clip_active = np.abs(v_targets) > self.joint_vel_limits
        if np.any(self.last_clip_active) and self.debug:
            over_idx = np.where(self.last_clip_active)[0]
            print(f"⚠️  Joint velocity limit clamp active on joints {list(over_idx)}: "
                  f"requested={v_targets[over_idx]}, limit={self.joint_vel_limits[over_idx]}")
        return clipped

    def control_callback(self, model, data):
        """MuJoCo control callback function with additional filtering and control terms."""
        # If a trajectory function is provided, update v_targets based on time
        if self.trajectory_function:
            try:
                new_targets = self.trajectory_function(data.time)
                if new_targets is not None:
                    self.v_targets = self._apply_joint_limits(
                        np.asarray(new_targets, dtype=float))
                elif self.v_targets is None:
                    # Initialize with zeros if targets are None
                    self.v_targets = np.zeros(self.num_actuators, dtype=float)
            except Exception as e:
                print(f"Error in trajectory function: {e}")
                # Ensure v_targets is never None
                if self.v_targets is None:
                    self.v_targets = np.zeros(self.num_actuators, dtype=float)
        elif self.v_targets is None:
            # Initialize with zeros if no trajectory function and targets are None
            self.v_targets = np.zeros(self.num_actuators, dtype=float)
        else:
            # Even if v_targets was set directly (not via a trajectory
            # function), re-apply the limit in case it was mutated elsewhere.
            self.v_targets = self._apply_joint_limits(self.v_targets)

        # Gravity compensation
        joint_ids = model.actuator_trnid[:, 0]
        dof_indices = model.jnt_dofadr[joint_ids]
        gravity_torques = data.qfrc_bias[dof_indices]
        data.ctrl[:] = gravity_torques

        # Control loop for each actuator
        for i in range(self.num_actuators):
            dof_i = dof_indices[i]
            v_actual = data.qvel[dof_i]

            # Apply low-pass filter to velocity measurements
            self.filtered_velocity[i] = self.filter_coeff * self.filtered_velocity[i] + \
                                       (1 - self.filter_coeff) * v_actual

            v_target = self.v_targets[i]

            # Use filtered velocity for error calculation
            v_error = self.filtered_velocity[i] - v_target

            # Update integral term (with anti-windup)
            if abs(v_error) < 0.1:  # Only integrate when error is small
                self.v_error_integral[i] += v_error * model.opt.timestep
                self.v_error_integral[i] = np.clip(self.v_error_integral[i],
                                                  -self.integral_limit,
                                                  self.integral_limit)
            else:
                # Decay integral when error is large
                self.v_error_integral[i] *= 0.95

            # D-control torque (like in the working example)
            torque_d = -self.kd[i] * v_error

            # I-control torque (small)
            torque_i = -self.ki[i] * self.v_error_integral[i]

            # Apply both terms
            data.ctrl[i] += torque_d + torque_i

            # Add extra torque for high-speed movements
            if abs(v_target) > 0.5:  # If we're commanding a high velocity
                # Add extra torque in the direction of motion to overcome friction
                extra_torque = np.sign(v_target) * 0.5
                data.ctrl[i] += extra_torque

            # Debug output
            if self.debug and i == 0 and data.time % 0.5 < model.opt.timestep:
                limit_note = ""
                if self.joint_vel_limits is not None:
                    limit_note = f", limit={self.joint_vel_limits[i]:.3f}" \
                                 f"{' [CLAMPED]' if self.last_clip_active[i] else ''}"
                print(f"Joint {i}: v_target={v_target:.3f}, v_actual={v_actual:.3f}, "
                      f"filtered_v={self.filtered_velocity[i]:.3f}, error={v_error:.3f}, "
                      f"integral={self.v_error_integral[i]:.3f}, torque_d={torque_d:.3f}, "
                      f"torque_i={torque_i:.3f}, gravity={gravity_torques[i]:.3f}, "
                      f"total={data.ctrl[i]:.3f}{limit_note}")