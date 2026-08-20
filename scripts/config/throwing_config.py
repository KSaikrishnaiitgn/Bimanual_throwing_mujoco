"""
config.py

Shared constants for the Dual-FR3 Dynamic Throwing task.

This module is pure configuration: robot/sim identifiers, object properties,
grasp geometry, throwing parameters, DS / impedance / admittance gains, and
joint-space controller gains, plus two tiny derived helpers for the desired
grasp wrench. No control logic lives here.

Import this module elsewhere as a flat module: `import config`. (Do not
import it as `config.throwing_config` -- this file is not inside a
package, it's a single top-level module.)
"""

import numpy as np

# ---- Robot / sim ----
XML_PATH = "/home/iitgn-robotics/Saikrishna/Bimanual_throwing_mujoco/robot_description/Dual_franka.xml"
SIM_TIMESTEP = 0.002  # s, must match <option timestep> in the XML

LEFT_JOINT_NAMES = [
    "left_fr3_joint1", "left_fr3_joint2", "left_fr3_joint3",
    "left_fr3_joint4", "left_fr3_joint5", "left_fr3_joint6", "left_fr3_joint7",
]
RIGHT_JOINT_NAMES = [
    "right_fr3_joint1", "right_fr3_joint2", "right_fr3_joint3",
    "right_fr3_joint4", "right_fr3_joint5", "right_fr3_joint6", "right_fr3_joint7",
]
LEFT_ACTUATOR_NAMES = LEFT_JOINT_NAMES   # actuator names == joint names in the XML
RIGHT_ACTUATOR_NAMES = RIGHT_JOINT_NAMES

LEFT_CTRLRANGE = [(-87, 87)] * 4 + [(-12, 12)] * 3   # per joint, matches <motor ctrlrange>
RIGHT_CTRLRANGE = [(-87, 87)] * 4 + [(-12, 12)] * 3

LEFT_EE_SITE = "left_center"
RIGHT_EE_SITE = "right_center"

LEFT_FORCE_SENSOR, LEFT_TORQUE_SENSOR = "left_force_sensor", "left_torque_sensor"
RIGHT_FORCE_SENSOR, RIGHT_TORQUE_SENSOR = "right_force_sensor", "right_torque_sensor"

BOX_BODY_NAME = "box"
BOX_FREE_JOINT_NAME = "box_joint"
LEFT_GRASP_SITE = "left_grasp_site"    # box-local (0.1, 0, 0)
RIGHT_GRASP_SITE = "right_grasp_site"  # box-local (-0.1, 0, 0)

# ---- Object ----
OBJECT_MASS = 0.36  # kg
M_O = np.diag([OBJECT_MASS, OBJECT_MASS, OBJECT_MASS])
GRAVITY = 9.81
PAD_HALF_THICKNESS = 0.015  # m, half the pad's thickness along its approach-axis normal
# ---- Grasp geometry (world frame, box at rest pose (-0.42, 0.25, 0.4), identity orientation) ----
LEFT_GRASP_WORLD_POS = np.array([-0.32, 0.25, 0.4])   # box's +x face center
RIGHT_GRASP_WORLD_POS = np.array([-0.52, 0.25, 0.4])  # box's -x face center
LEFT_PAD_APPROACH_AXIS_WORLD = np.array([-1.0, 0.0, 0.0])   # pad flat-normal target direction
RIGHT_PAD_APPROACH_AXIS_WORLD = np.array([1.0, 0.0, 0.0])
# Keep the wrists well clear of the box during the first approach.  The old
# 0.05 m stand-off allowed arm geometry to contact and disturb the free box
# before the controlled grasp phase began.
PREGRASP_STANDOFF = 0.10  # m, back off along the approach axis for the pregrasp waypoint

# ---- Throwing ----
# RELEASE_POINT is where the box should BE, mid-swing, at the moment of
# release -- it is a planned workspace target, exactly like LANDING_POINT,
# NOT the box's initial grasp position. The DS controller (see
# ds_impedance_controller.compute_ds_accel) drives the box FROM its grasp
# pose (LEFT_GRASP_WORLD_POS / RIGHT_GRASP_WORLD_POS midpoint, i.e. roughly
# the box's resting pose (-0.42, 0.4, 0.1)) TOWARD this point. If
# RELEASE_POINT were ever set equal to the box's starting position, the DS
# position error would be zero from the first step and the box would never
# accelerate -- this MUST be a genuinely different point ahead of the grasp
# pose along the intended swing direction.
#
# PLACEHOLDER -- chosen as "forward (+y) and up (+z) from the grasp pose",
# representing an arm-extended release partway through a forward swing.
# Verify this point is inside the dual-arm's combined reachable workspace
# (e.g. via ik_solver.py / frame_inspector.py) before trusting it.
RELEASE_POINT = np.array([-0.42, 0.6, 0.5])  # reduced from 0.9: right_fr3_joint1 saturated its -2.7437 limit swinging to y=0.9 (see debug trace, t~21.0s)

LANDING_POINT = np.array([-0.42, 2.5, 0.0])  # original baseline target
THROW_ANGLE = np.deg2rad(30.0)
MAX_OBJ_VEL = 8.0  # m/s, hard clip on |v_rel| and on x_dot_o*

# ---- DS (object-level, drives box from grasp pose to release pose/velocity) ----
K_DS = np.diag([18.0, 18.0, 18.0])
B_DS = np.diag([10.0, 10.0, 10.0])

# ---- Object-level impedance ----
D_IMPEDANCE = np.diag([20.0, 20.0, 20.0])
K_IMPEDANCE = np.diag([10.0, 10.0, 10.0])   # this is "K1" in the object-impedance equation

# ---- Contact-level admittance (grasp force closure) ----
ADMITTANCE_STIFFNESS = np.diag([200.0, 200.0, 400.0, 20.0, 20.0, 20.0])  # "K" in xdot = K^-1(F*-F_meas)
GRASP_FORCE_MAG = 3.0          # N, target squeeze force magnitude
GRASP_STABLE_FORCE_TOL = 0.5   # N
GRASP_STABLE_HOLD_TIME = 0.3   # s
MAX_ADMITTANCE_VEL = 0.05      # m/s, clip on the admittance-law Cartesian velocity output
# Sign of F* along the approach axis (+5N or -5N per side) is sensor-frame dependent;
# determine empirically per side using frame_inspector.py's sensor-sign test, then hardcode here.

# ---- Joint-space controller gains (7 per arm, 14 total) ----
# Franka's own joint-impedance reference-controller gains (validated on real
# Panda/FR3 hardware). Usable as-is now that reaching_pd cancels gravity +
# Coriolis/centrifugal torque via a separate bias_torque term
# (MjInterface.get_bias_torque, sourced from MuJoCo's qfrc_bias in sim, and
# from franka::Model::gravity()+coriolis() on hardware) -- so these gains
# only need to shape the transient response, not fight gravity.
#
# NOTE: VelocityPID (joint_controllers.py) reuses KD_GAINS/KI_GAINS as its
# velocity-error P/I gains during THROW. Watch the torque-saturation panel
# in throw_summary.png once THROW is reached; if joints 5-7 (kd=20/20/10)
# are chronically pinned at their +/-12 Nm ctrlrange, split VelocityPID off
# onto its own gain arrays instead of continuing to share KD_GAINS.
# KP_GAINS = np.array([600.0, 600.0, 600.0, 600.0, 250.0, 150.0, 50.0] * 2)
# KD_GAINS = np.array([50.0,  50.0,  50.0,  20.0,  20.0,  20.0,  10.0] * 2)
KP_GAINS = np.array([600.0, 600.0, 600.0, 600.0, 250.0, 150.0, 50.0] * 2)
KD_GAINS = np.array([150.0, 150.0, 150.0, 60.0,  60.0,  60.0,  30.0] * 2)
KI_GAINS = np.array([0.01, 0.1, 0.01, 0.01, 0.005, 0.002, 0.002] * 2)
# VELOCITY_KI_GAINS = np.array([8.0, 8.0, 8.0, 8.0, 8.0, 8.0, 8.0] * 2)  # used by VelocityPID during THROW
VELOCITY_KI_GAINS = np.array([30.0, 30.0, 30.0, 30.0, 30.0, 30.0, 30.0] * 2)  # used by VelocityPID during THROW
# Real FR3 joint velocity limits (rad/s), FCI-appropriate (note A6 is
# restricted to 239 deg/s / ~4.18 rad/s under real-time torque control,
# lower than its full-hardware 301 deg/s ceiling; A1-A4/A5/A7 use the
# datasheet's 150/301 deg/s figures). The XML has no native velocity-limit
# field (MuJoCo <joint>/<motor> don't carry one), so this array is sourced
# from Franka's published spec, not the model file.
#
# This is the layout for ONE arm's 7 joints; `* 2` on the Python list below
# duplicates it (not element-wise multiplication) to produce the 14-entry
# [left(7), right(7)] array that joint_controllers._slice_for_arm expects --
# valid because both arms are the same physical robot model.
MAX_JOINT_VELOCITIES = np.array([2.62, 2.62, 2.62, 2.62, 5.26, 4.18, 5.26] * 2)
JOINT_FRICTIONLOSS = np.array([1.137, 1.137, 1.137, 1.137, 0.763, 0.44, 0.248] * 2)

# Per-joint position limits, [left(7), right(7)] layout matching every
# other per-arm array in this module (see e.g. KP_GAINS / _slice_for_arm
# convention in joint_controllers.py). Sourced from Dual_franka.xml
# <joint range="..."> -- both arms share the same physical joint ranges,
# so this 7-row block is simply duplicated.
#
# Moved here from state_machine.py (previously module-local, used only for
# THROW-phase diagnostic instrumentation) so ds_impedance_controller.py's
# nullspace joint-limit-avoidance term can also import it as the single
# source of truth -- state_machine.py now imports it from here instead of
# defining its own copy.
JOINT_RANGES = np.array([
    [-2.7437,  2.7437],
    [-1.7837,  1.7837],
    [-2.9007,  2.9007],
    [-3.0421, -0.1518],
    [-2.8065,  2.8065],
    [ 0.5445,  4.5169],
    [-3.0159,  3.0159],
] * 2)

# Nullspace joint-limit-avoidance gain, used by
# ds_impedance_controller.compute_dual_arm_qdot's secondary (nullspace)
# task: qdot_avoid = -K_JOINT_LIMIT_AVOID * grad(Liegeois joint-limit
# potential), projected through (I - J_H_pinv @ J_H) before being summed
# with the primary task-space qdot_cmd.
#
# PLACEHOLDER starting value -- chosen to be small relative to the primary
# task's velocity scale (K_TRACK=4.0, DS gains K_DS/B_DS ~10-18) seen
# driving qdot_cmd to ~0.01-2.7 rad/s in the THROW-phase debug log, so this
# term nudges joints away from limits without fighting the throw swing.
# NOT YET TUNED against real sim behavior -- increase if right_fr3_joint1
# (or any joint) still saturates during THROW; decrease if the nullspace
# term visibly distorts the object's trajectory toward RELEASE_POINT.
K_JOINT_LIMIT_AVOID = 0.1
# ---- Trajectory generation (velocity-limited reaching phases) ----
# PLACEHOLDER, modeled on libfranka's default per-joint max acceleration
# values (kMaxJointAcceleration, rad/s^2) for Panda/FR3 -- reasonable
# starting point that is also representative of what real hardware
# actually enforces. Used by trajectory_generator.TrapezoidalJointTrajectory
# to build velocity-and-acceleration-limited reaching-phase references, so
# reaching_pd is tracking a moving setpoint instead of a distant static one.
MAX_JOINT_ACCELERATIONS = np.array([15.0, 7.5, 10.0, 12.5, 15.0, 20.0, 20.0] * 2)

POSITION_THRESHOLDS = np.array([0.02, 0.02, 0.02, 0.02, 0.03, 0.03, 0.03] * 2)
APPROACH_GRASP_POS_TOLERANCE = 0.05  # rad, looser fallback tolerance for
                                       # APPROACH_GRASP -> SQUEEZE_GRASP when
                                       # contact blocks exact convergence to
                                       # grasp_left/right
APPROACH_SETTLE_QVEL = 0.02           # rad/s, "stopped moving" threshold
K_TRACK = np.array([4.0, 4.0, 4.0, 4.0, 4.0, 4.0, 4.0] * 2)
K_SQUEEZE_HOLD_LIN = 20.0
K_SQUEEZE_HOLD_ANG = 10.0
# ---- Release trigger ----
RELEASE_POS_TOLERANCE = 0.01  # m, |x_o - RELEASE_POINT| below this triggers release


# ---- Derived helpers ----
"""
    Return the desired 6D grasp wrench [Fx, Fy, Fz, Tx, Ty, Tz] for the given
    arm side, used as F* in the contact-level admittance law xdot = K^-1(F* - F_meas).

    Fx = +GRASP_FORCE_MAG for 'left', -GRASP_FORCE_MAG for 'right'; all other
    components zero. The sign convention below is a placeholder and must be
    corrected after running frame_inspector.py's sensor-sign test.

    Args:
        side: 'left' or 'right'.

    Returns:
        np.ndarray of shape (6,).
"""
def desired_wrench(side: str) -> np.ndarray:
    """
    Return the desired 6D grasp wrench [Fx, Fy, Fz, Tx, Ty, Tz] for the given
    arm side, used as F* in the contact-level admittance law.

    Force is placed along that side's pad approach axis
    (LEFT_PAD_APPROACH_AXIS_WORLD / RIGHT_PAD_APPROACH_AXIS_WORLD) -- the
    two axes point opposite each other, so this naturally produces the two
    arms squeezing toward one another. All other components are zero.

    Sign along the approach axis (i.e. whether GRASP_FORCE_MAG is applied
    with the axis vector's sign as-is, or negated) is sensor-frame
    dependent and MUST be verified empirically per side using
    frame_inspector.py's sensor-sign test before trusting this on
    hardware or in a fresh sim -- flip the sign below if the measured
    contact force comes back negative of GRASP_FORCE_MAG at steady state.

    Args:
        side: 'left' or 'right'.

    Returns:
        np.ndarray of shape (6,).
    """
    axis = LEFT_PAD_APPROACH_AXIS_WORLD if side == "left" else RIGHT_PAD_APPROACH_AXIS_WORLD
    force = GRASP_FORCE_MAG * axis  # TODO_VERIFY_SIGN
    return np.concatenate([force, np.zeros(3)])


def zero_wrench() -> np.ndarray:
    """
    Return the zero 6D wrench, used to deactivate force-closure control
    (e.g. at release, when the grasp is opened and no squeeze force is commanded).

    Returns:
        np.ndarray of shape (6,), all zeros.
    """
    return np.zeros(6)
