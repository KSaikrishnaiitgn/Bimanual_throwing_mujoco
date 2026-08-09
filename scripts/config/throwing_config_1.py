"""
Configuration file for throwing task
Contains all constants, parameters, and settings
"""
import numpy as np

# ==================== PHYSICS CONSTANTS ====================
GRAVITY = 9.81  # m/s^2

# ==================== THROWING PARAMETERS ====================
LANDING_POINT = np.array([0.0, -2.5, 0.0])  # Target landing position (x, y, z)

# ── NEW: fixed release point ────────────────────────────────────────────────
# This is the position (in world frame, same frame as LANDING_POINT / box xpos)
# at which the box is intended to be released during the throw.
#
# PLACEHOLDER — you must tune this to your actual robot/box geometry.
# A reasonable starting guess: wherever the box naturally sits at the END of
# your lift phase, offset slightly forward in the throw direction (negative Y
# here, since LANDING_POINT is at -Y) to give the arms room to swing through
# before release. E.g. if your lift phase leaves the box around
# [0.0, 0.3, 0.9], you might start with:
RELEASE_POINT = np.array([0.0, -0.500, 0.400])  # <-- TUNE THIS in sim

THROW_ANGLE = np.deg2rad(30.0)  # Launch angle in radians
VELOCITY_SCALE = 1.4  # Scale factor for computed velocity
TARGET_LIFT_HEIGHT = 0.3  # Target height to lift box before throwing (meters)

# Release trigger tolerances (NEW — guards against premature/late release
# now that release is tied to a fixed spatial point, not just a velocity
# ratio from a snapshot position).
RELEASE_POSITION_TOLERANCE = 0.05   # m — box must be within this of RELEASE_POINT
MIN_SWING_TIME             = 0.15   # s — don't allow release before this elapses
MIN_RELEASE_SPEED          = 0.15   # m/s — absolute floor, avoids noise-triggered release

# ── NEW: grasp force-closure deactivation near release ──────────────────────
# Once the box is within this distance of RELEASE_POINT, the hands stop
# actively squeezing (the 7 N force closure is turned off) so the box can
# separate naturally at release instead of being held in a grip through it.
# Set slightly LARGER than RELEASE_POSITION_TOLERANCE so grasp force relaxes
# just before the release trigger itself fires.
GRASP_DEACTIVATION_TOLERANCE = 0.10  # m
GRASP_RAMP_DOWN_TIME         = 0.05  # s — smooth taper to zero force (0.0 = instant cutoff)

# ==================== DYNAMICAL SYSTEM GAINS ====================
# Spring-damper system for throwing acceleration
#   ẍ_des = -K_ds (x_o - x_rel) - B_ds (ẋ_o - v_rel)
#
# NOTE: K_ds is now NON-ZERO (previous version zeroed it out because x_rel
# was just wherever the box happened to be — with a real, fixed RELEASE_POINT
# defined above, K_ds now does meaningful work: it pulls the box toward that
# point while B_ds simultaneously pulls velocity toward v_rel.
#
# Tuning: natural frequency ω = sqrt(K_ds), damping ratio ζ = B_ds / (2ω).
# Values below give ω ≈ 3.87 rad/s, ζ ≈ 1.03 (just past critical damping —
# fast convergence with negligible overshoot). Increase K_ds for a snappier
# pull toward the release point; increase B_ds if you see oscillation.
K_DS = np.diag([15.0, 15.0, 15.0])   # Position stiffness (1/s^2)
B_DS = np.diag([8.0, 8.0, 8.0])      # Velocity damping (1/s)
MAX_OBJ_VEL = 8.0  # Maximum object velocity (m/s)

# ==================== ADMITTANCE CONTROL PARAMETERS (LIFTING) ====================
# For lifting phase (unchanged — separate from the new grasp admittance below)
ADMITTANCE_STIFFNESS = np.diag([200.0, 200.0, 400.0, 20.0, 20.0, 20.0])  # K matrix
ADMITTANCE_DAMPING = np.diag([20.0, 20.0, 10.0, 3.0, 3.0, 3.0])  # D matrix
MAX_LINEAR_VELOCITY = 0.5  # m/s
MAX_ANGULAR_VELOCITY = 0.5  # rad/s
UPWARD_FORCE_BASE = 20.0  # Base upward force factor
UPWARD_FORCE_SCALE = 15.0  # Additional upward force scaling

# ==================== GRASP / CONTACT ADMITTANCE (NEW — force closure) ====
# Implements:  ẋ = K_adm^-1 (F* - F_meas)
# Both end-effectors squeeze toward the box along the grasp axis (the line
# connecting the two EE sites) until each senses DESIRED_GRASP_FORCE. Equal
# and opposite forces at equilibrium => zero net force on the object (force
# closure) while holding it stable between the hands.
DESIRED_GRASP_FORCE = 7.0            # N — target squeeze force per hand
K_ADM = np.diag([25.0, 25.0, 25.0])  # N·s/m — admittance gain (force error -> velocity)
MAX_GRASP_VELOCITY = 0.3             # m/s — safety clamp on squeeze velocity

# ==================== IMPEDANCE CONTROL PARAMETERS (FOR THROWING) ====================
# Object-level impedance control during throwing phase
OBJECT_MASS = 0.5  # kg (estimated box mass)
M_O = np.diag([OBJECT_MASS, OBJECT_MASS, OBJECT_MASS])  # Virtual inertia (kg)
                                                          # matches real mass by default

# Damping matrix for impedance control
D_IMPEDANCE = np.diag([20.0, 20.0, 20.0])  # Damping (N·s/m)

# Impedance stiffness (position error term). REDUCED from the old value
# (was 10) because K_ds in the DS above now ALSO pulls toward the release
# point — K_1 here is meant as a light secondary correction, not the primary
# position authority, to avoid double-counting / over-stiffening the response.
K_IMPEDANCE = np.diag([4.0, 4.0, 4.0])  # Stiffness (N/m)

# Maximum commanded velocities for safety (Cartesian object velocity, m/s)
MAX_COMMANDED_VEL = 7.0  # m/s

# ==================== HARDWARE SAFETY — JOINT VELOCITY LIMITS ====================
# rad/s, per actuator: [left x6, right x6]. Enforced in both the throw
# controller (direction-preserving scale) and the low-level velocity
# controller (final per-joint clamp safety net).
MAX_JOINT_VELOCITIES = np.array([
    3.0, 1.0, 5.0, 1.0, 1.0, 1.0,
    3.0, 1.0, 5.0, 1.0, 1.0, 1.0
])

# ==================== VELOCITY CONTROLLER GAINS ====================
KD_GAINS = np.array([
    60.0, 100.0, 70.0, 30.0, 15.0, 15.0,
    60.0, 100.0, 70.0, 30.0, 15.0, 15.0
])

KI_GAINS = np.array([
    0.01, 0.1, 0.01, 0.005, 0.002, 0.002,
    0.01, 0.1, 0.01, 0.005, 0.002, 0.002
])

# ==================== POSITION CONTROLLER GAINS ====================
KP_GAINS = np.array([
    15.0, 60.0, 15.0, 5.0, 3.0, 3.0,
    15.0, 60.0, 15.0, 5.0, 3.0, 3.0
])

MAX_JOINT_VELOCITIES_REACH = np.array([   # kept separate name to avoid confusion
    3.0, 1.0, 5.0, 1.0, 1.0, 1.0,
    3.0, 1.0, 5.0, 1.0, 1.0, 1.0
])

POSITION_THRESHOLDS = np.array([
    0.02, 0.02, 0.02, 0.03, 0.03, 0.03,
    0.02, 0.02, 0.02, 0.03, 0.03, 0.03
])

# ==================== ROBOT CONFIGURATIONS ====================
FIRST_TARGET_POSITIONS_LEFT = np.array([-0.0805, 1.07, -0.126, 1.53, -0.00978, 0])
FIRST_TARGET_POSITIONS_RIGHT = np.array([0.105, 1.07, -0.126, -1.45, -0.00901, 0])

SECOND_TARGET_POSITIONS_LEFT = np.array([0.147, 1.16, -0.0314, 1.79, -0.00978, 0.5])
SECOND_TARGET_POSITIONS_RIGHT = np.array([-0.147, 1.16, -0.0314, -1.79, -0.0314, 0.5])

# ==================== SIMULATION SETTINGS ====================
XML_FILE_PATH = "/home/iitgn-robotics/Saikrishna/Bimanual_throwing_mujoco/robot_description/kshitij_lifting.xml"
SLEEP_TIME = 0.001  # seconds between simulation steps

# ==================== PHASE TIMINGS ====================
READING_SENSORS_DURATION = 0.1  # seconds
HEIGHT_THRESHOLD = 0.01  # meters

# ==================== DEBUG SETTINGS ====================
PHASE_PRINT_INTERVAL = 1.0  # seconds
LIFT_DEBUG_PRINT_INTERVAL = 100  # steps