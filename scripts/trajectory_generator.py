"""
trajectory_generator.py

Per-arm, multi-joint, velocity-and-acceleration-limited trajectory generator
using a synchronized trapezoidal ("bang-coast-bang") velocity profile.

Why this exists: reaching_pd (joint_controllers.py) is a pure position-PD
controller with no speed ceiling of its own. If it is asked to servo
directly to a distant setpoint, the commanded torque saturates the
ctrlrange immediately and stays saturated -- effectively bang-bang control
-- producing unrealistically fast, overshoot-prone motion with no
resemblance to how a real robot moves. On real Franka hardware, joint
motions are never commanded as a raw distant PD setpoint either: libfranka's
MotionGenerator (see the Franka `examples/generate_joint_pose_motion.cpp`
pattern) builds a smooth, velocity/acceleration-limited reference and PD/PID
controllers track *that*, not the final goal directly. This module is the
same idea: build the moving reference once per phase, then have
reaching_pd track it point-by-point.

Per joint:
  - If the joint can reach MAX_JOINT_VELOCITIES before needing to decelerate
    (i.e. the distance is large enough), it gets a full trapezoid: ramp up
    at max_accel, cruise at max_vel, ramp down at max_accel.
  - Otherwise it gets a triangular profile (never reaches max_vel).

All 7 joints are then time-synchronized to the SLOWEST joint's duration
(the standard multi-axis sync approach), by solving for a reduced peak
velocity per joint that exactly fits the shared duration at that joint's
own max_accel. This keeps all 7 joints arriving simultaneously, which is
what you want for both grasp alignment and predictable behavior.

No MuJoCo dependency -- pure numpy, unit-testable standalone.
"""

from __future__ import annotations

import numpy as np


class TrapezoidalJointTrajectory:
    """
    Synchronized trapezoidal-velocity trajectory for a 7-joint arm.

    Usage:
        traj = TrapezoidalJointTrajectory(q_start, q_target, max_vel, max_accel)
        q_ref, qdot_ref, finished = traj.sample(elapsed_time_since_start)
    """

    def __init__(
        self,
        q_start: np.ndarray,
        q_target: np.ndarray,
        max_vel: np.ndarray,
        max_accel: np.ndarray,
    ):
        self.q_start = np.asarray(q_start, dtype=np.float64).copy()
        self.q_target = np.asarray(q_target, dtype=np.float64).copy()
        max_vel = np.asarray(max_vel, dtype=np.float64)
        self.max_accel = np.asarray(max_accel, dtype=np.float64)

        delta = self.q_target - self.q_start
        self.direction = np.sign(delta)
        self.distance = np.abs(delta)  # (7,)

        n = self.distance.shape[0]

        # ---- Pass 1: each joint's own unconstrained-by-others duration ----
        t_acc_unsync = np.zeros(n)
        per_joint_duration = np.zeros(n)
        for i in range(n):
            if self.distance[i] < 1e-9:
                continue
            a = self.max_accel[i]
            v = max_vel[i]
            t_acc = v / a
            d_acc = 0.5 * a * t_acc**2
            if self.distance[i] >= 2.0 * d_acc:
                # full trapezoid: reaches max_vel
                t_flat = (self.distance[i] - 2.0 * d_acc) / v
                per_joint_duration[i] = 2.0 * t_acc + t_flat
            else:
                # triangular: never reaches max_vel
                t_acc = np.sqrt(self.distance[i] / a)
                per_joint_duration[i] = 2.0 * t_acc
            t_acc_unsync[i] = t_acc

        self.duration = float(np.max(per_joint_duration)) if n > 0 else 0.0

        # ---- Pass 2: rescale every joint's peak velocity so it takes
        # exactly self.duration, at that joint's own max_accel. Solving
        # d = v*T - v^2/a  (trapezoid distance covered in time T at accel a,
        # peak velocity v)  =>  v^2 - a*T*v + a*d = 0, take the smaller
        # (valid, v <= a*T/2) root. ----
        self.sync_peak_vel = np.zeros(n)
        self.sync_t_acc = np.zeros(n)
        self.sync_t_flat = np.zeros(n)
        for i in range(n):
            if self.distance[i] < 1e-9 or self.duration < 1e-9:
                continue
            a = self.max_accel[i]
            d = self.distance[i]
            T = self.duration
            disc = (a * T) ** 2 - 4.0 * a * d
            if disc < 0:
                # Shouldn't normally happen since T is this joint's own
                # worst case or looser; fall back to a saturating clip.
                v = min(max_vel[i], d / max(T, 1e-9))
            else:
                v = (a * T - np.sqrt(disc)) / 2.0
                v = min(v, max_vel[i])
            t_acc = v / a if a > 0 else 0.0
            t_flat = T - 2.0 * t_acc
            if t_flat < 0.0:
                t_flat = 0.0
            self.sync_peak_vel[i] = v
            self.sync_t_acc[i] = t_acc
            self.sync_t_flat[i] = t_flat

    def sample(self, t: float) -> tuple[np.ndarray, np.ndarray, bool]:
        """
        Sample the trajectory at elapsed time `t` (seconds since the
        trajectory's own start, i.e. since phase entry -- caller tracks
        that offset).

        Returns:
            q_ref:    np.ndarray (7,), reference joint position at time t.
            qdot_ref: np.ndarray (7,), reference joint velocity at time t.
            finished: True once t >= self.duration (all joints stationary
                      at q_target).
        """
        if self.duration <= 0.0:
            return self.q_target.copy(), np.zeros_like(self.q_target), True

        t_clamped = min(max(t, 0.0), self.duration)
        n = self.distance.shape[0]
        q_ref = self.q_start.copy()
        qdot_ref = np.zeros(n)

        for i in range(n):
            if self.distance[i] < 1e-9:
                q_ref[i] = self.q_target[i]
                continue
            a = self.max_accel[i]
            v = self.sync_peak_vel[i]
            t_acc = self.sync_t_acc[i]
            t_flat = self.sync_t_flat[i]
            d = self.direction[i]

            if t_clamped <= t_acc:
                pos = 0.5 * a * t_clamped**2
                vel = a * t_clamped
            elif t_clamped <= t_acc + t_flat:
                pos = 0.5 * a * t_acc**2 + v * (t_clamped - t_acc)
                vel = v
            else:
                t_dec = t_clamped - t_acc - t_flat
                pos = 0.5 * a * t_acc**2 + v * t_flat + v * t_dec - 0.5 * a * t_dec**2
                vel = v - a * t_dec

            q_ref[i] = self.q_start[i] + d * pos
            qdot_ref[i] = d * vel

        finished = t >= self.duration
        return q_ref, qdot_ref, finished