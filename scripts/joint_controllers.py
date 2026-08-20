"""
joint_controllers.py

Joint-space torque-output controllers for the dual-FR3 dynamic throwing system.

Both controllers here output raw joint torques (7,) intended for `<motor>`
actuators, i.e. `ctrl` IS torque directly -- there is no internal MuJoCo PD
to lean on. Neither controller clips against `ctrlrange`; that is the
responsibility of the caller (`MjInterface.set_ctrl`), which does the final
per-joint clip and is expected to log any saturation event.

Two controllers:
  1. `reaching_pd`   -- joint-space PD tracking controller (position +
                        velocity reference, + optional gravity/Coriolis
                        feedforward), used during APPROACH_PREGRASP /
                        APPROACH_GRASP / SQUEEZE_GRASP (holding) /
                        FOLLOW_THROUGH. Intended to track one sample of a
                        TrapezoidalJointTrajectory (trajectory_generator.py)
                        rather than a distant static goal directly -- PD-ing
                        straight to a far setpoint has no speed ceiling and
                        saturates the actuator into bang-bang motion.
  2. `VelocityPID`    -- joint-space PID velocity-tracking controller, used
                        during the throw phase to track `qdot_cmd` coming
                        from the inverse-kinematics/DS pipeline.

Depends on: config.py (gain/limit arrays), numpy.
"""

from __future__ import annotations

from typing import Optional

import numpy as np

from config.throwing_config import (
    KP_GAINS,
    KD_GAINS,
    KI_GAINS,
    MAX_JOINT_VELOCITIES,
    POSITION_THRESHOLDS,
    JOINT_FRICTIONLOSS,
    VELOCITY_KI_GAINS,
)


def _slice_for_arm(array: np.ndarray, arm: str) -> np.ndarray:
    """Return the 7-element slice of a 14-element per-arm gain/limit array.

    Both arms' arrays are laid out as [left(7), right(7)] throughout
    config.py (see KP_GAINS, KD_GAINS, KI_GAINS, MAX_JOINT_VELOCITIES,
    POSITION_THRESHOLDS). Centralizing the slice here avoids repeating the
    `[0:7] if arm == 'left' else [7:14]` ternary at every call site.
    """
    if arm == "left":
        return array[0:7]
    elif arm == "right":
        return array[7:14]
    else:
        raise ValueError(f"arm must be 'left' or 'right', got {arm!r}")


def reaching_pd(
    q: np.ndarray,
    qvel: np.ndarray,
    q_target: np.ndarray,
    arm: str,
    bias_torque: Optional[np.ndarray] = None,
    qdot_target: Optional[np.ndarray] = None,
) -> tuple[np.ndarray, bool]:
    """Joint-space PD tracking controller for the reaching phase.

    Tracks a (position, velocity) reference pair:

        torque = kp * (q_target - q) + kd * (qdot_target - qvel) + bias_torque

    `q_target`/`qdot_target` are intended to be a single sample from a
    TrapezoidalJointTrajectory (trajectory_generator.py), not the final
    phase goal directly. If `qdot_target` is None it defaults to zeros,
    which recovers pure position-PD behavior (used for the stationary hold
    in SQUEEZE_GRASP, where q_target IS the final goal and no motion is
    wanted).

    Args:
        q:            current joint positions, shape (7,), rad.
        qvel:         current joint velocities, shape (7,), rad/s.
        q_target:     reference joint positions, shape (7,), rad.
        arm:          'left' or 'right'.
        bias_torque:  optional, shape (7,), Nm. Gravity/Coriolis
                      feedforward (see MjInterface.get_bias_torque).
        qdot_target:  optional, shape (7,), rad/s. Reference joint
                      velocity; None => zeros (pure position hold).

    Returns:
        torque:  raw joint torques, shape (7,), Nm. Pre-clip.
        reached: True if every joint's absolute error vs. `q_target` is
                 below POSITION_THRESHOLDS. NOTE: when q_target is a moving
                 trajectory sample rather than the final goal, this flag is
                 not the right thing to gate a phase transition on -- check
                 against the trajectory's final target instead (see
                 state_machine.py).
    """
    kp = _slice_for_arm(KP_GAINS, arm)
    kd = _slice_for_arm(KD_GAINS, arm)
    pos_thresh = _slice_for_arm(POSITION_THRESHOLDS, arm)

    if qdot_target is None:
        qdot_target = np.zeros_like(q)

    q_err = q_target - q
    qvel_err = qdot_target - qvel
    torque = kp * q_err + kd * qvel_err
    if bias_torque is not None:
        torque = torque + bias_torque

    reached = bool(np.all(np.abs(q_err) < pos_thresh))

    return torque, reached


class VelocityPID:
    """Joint-space PID velocity-tracking controller for the throw phase.
    ...
    (docstring unchanged, but see step() for the added friction term)
    """

    # Deadband on the velocity error below which friction feedforward is
    # NOT applied. Without this, a joint sitting exactly at qdot_cmd would
    # get a friction-torque command that flips sign every step as err
    # jitters through zero from sensor/integration noise -- chatter, not
    # help. Needs to be well below the smallest error we actually want
    # compensated (e.g. the ~0.01-0.02 rad/s stiction-adjacent errors seen
    # in SQUEEZE_GRASP), and well above qvel sensor noise floor.
    FRICTION_ERR_DEADBAND = 1e-3  # rad/s

    def __init__(self, arm: str):
        if arm not in ("left", "right"):
            raise ValueError(f"arm must be 'left' or 'right', got {arm!r}")
        self.arm = arm
        self.integral = np.zeros(7)
        # Anti-windup clamp, per joint. Previously a single scalar (5.0)
        # applied uniformly -- fine as a *secondary* safety margin now that
        # friction feedforward (see step()) is the primary mechanism for
        # overcoming frictionloss, but the old value was sized with no
        # relationship to frictionloss at all and left every joint's
        # integral-torque ceiling (ki * integral_limit) far below its
        # breakaway torque. Bumped modestly here; friction feedforward
        # does the actual breakaway work, this just gives the integral
        # term room to trim out any remaining steady-state error friction
        # feedforward doesn't fully cancel (e.g. if the true frictionloss
        # differs slightly from the XML value under load).
        self.integral_limit = 10.0
        self._debug_step_count = 0
    def reset(self) -> None:
        """Zero the integral term. Call when re-entering the throw phase."""
        self.integral = np.zeros(7)

    def step(self, qdot: np.ndarray, qdot_cmd: np.ndarray, dt: float,
              bias_torque: Optional[np.ndarray] = None) -> np.ndarray:
        """Compute one torque command from a velocity-tracking error.

        torque = kd*err + ki*integral + friction_feedforward + bias_torque

        friction_feedforward compensates each joint's static friction
        (JOINT_FRICTIONLOSS, sourced from the model's <joint frictionloss>)
        in the direction of the velocity error, whenever |err| exceeds
        FRICTION_ERR_DEADBAND. This is what actually lets the controller
        break static friction promptly -- previously only the P and I
        terms fought frictionloss, and with a tightly-clamped integral the
        combined torque could sit below breakaway indefinitely for a small
        persistent error (see SQUEEZE_GRASP stiction bug).

        Args:
            qdot:     current joint velocities, shape (7,), rad/s.
            qdot_cmd: commanded joint velocities, shape (7,), rad/s, prior
                      to velocity-limit clipping (this function clips it).
            dt:       control-loop timestep, s.
            bias_torque: optional, shape (7,), Nm. Gravity/Coriolis
                      feedforward.

        Returns:
            torque: raw joint torques, shape (7,), Nm. Pre-clip -- caller
                     applies ctrlrange clipping.
        """
        max_vel = _slice_for_arm(MAX_JOINT_VELOCITIES, self.arm)
        qdot_cmd_clipped = np.clip(qdot_cmd, -max_vel, max_vel)

        if not np.allclose(qdot_cmd_clipped, qdot_cmd):
            # Caller should log this: the commanded joint velocity exceeded
            # MAX_JOINT_VELOCITIES and was clamped, so the Cartesian
            # release velocity v_rel may be under-delivered relative to
            # what the DS/throw planner requested.
            pass

        err = qdot_cmd_clipped - qdot
        self.integral = np.clip(
            self.integral + err * dt, -self.integral_limit, self.integral_limit
        )

        kd = _slice_for_arm(KD_GAINS, self.arm)
        ki = _slice_for_arm(VELOCITY_KI_GAINS, self.arm)
        friction_comp = _slice_for_arm(JOINT_FRICTIONLOSS, self.arm)

        torque = kd * err + ki * self.integral
        friction_ff = np.where(
            np.abs(err) > self.FRICTION_ERR_DEADBAND,
            np.sign(err) * friction_comp,
            0.0,
        )
        torque = torque + friction_ff

        if bias_torque is not None:
            torque = torque + bias_torque

        # ---- DEBUG: torque breakdown (remove after diagnosing) ----
        self._debug_step_count += 1
        if self.arm in ("left", "right") and self._debug_step_count % 250 == 0:
            j = 0  # joint1 -- carries the axial squeeze load
            approx_t = self._debug_step_count * dt
            print(
                f"[PID-{self.arm} j{j+1} t~{approx_t:.2f}] "
                f"err={err[j]:+.5f} kd*err={kd[j]*err[j]:+.4f} "
                f"ki*int={ki[j]*self.integral[j]:+.4f} "
                f"friction_ff={friction_ff[j]:+.4f} "
                f"bias={0.0 if bias_torque is None else bias_torque[j]:+.4f} "
                f"TOTAL={torque[j]:+.4f}Nm  qdot={qdot[j]:+.5f} qdot_cmd={qdot_cmd_clipped[j]:+.5f}"
            )
        # ---- END DEBUG ----
        return torque