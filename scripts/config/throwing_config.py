"""
Configuration file for throwing task
Contains all constants, parameters, and settings
"""
import numpy as np

# ==================== PHYSICS CONSTANTS ====================
GRAVITY = 9.81  # m/s^2

# ==================== THROWING PARAMETERS ====================
LANDING_POINT = np.array([0.0, -2.5, 0.0])  # Target landing position (x, y, z) - FORWARD throw (negative Y is forward)
THROW_ANGLE = np.deg2rad(30.0)  # Launch angle in radians
VELOCITY_SCALE = 1.4  # Scale factor for computed velocity
TARGET_LIFT_HEIGHT = 0.3  # Target height to lift box before throwing (meters)

# ==================== DYNAMICAL SYSTEM GAINS ====================
# Spring-damper system for throwing acceleration
# NOTE: Position term should be zero during throwing (we're already at release position)
# Velocity term drives convergence to target velocity
K_DS = np.diag([0.0, 0.0, 0.0])  # Position stiffness (zero - we're at release point)
B_DS = np.diag([10.0, 10.0, 10.0])  # Velocity damping (reduced to avoid over-control)
MAX_OBJ_VEL = 8.0  # Maximum object velocity (m/s)

# ==================== ADMITTANCE CONTROL PARAMETERS ====================
# For lifting phase
ADMITTANCE_STIFFNESS = np.diag([200.0, 200.0, 400.0, 20.0, 20.0, 20.0])  # K matrix
ADMITTANCE_DAMPING = np.diag([20.0, 20.0, 10.0, 3.0, 3.0, 3.0])  # D matrix
MAX_LINEAR_VELOCITY = 0.5  # m/s
MAX_ANGULAR_VELOCITY = 0.5  # rad/s
UPWARD_FORCE_BASE = 20.0  # Base upward force factor
UPWARD_FORCE_SCALE = 15.0  # Additional upward force scaling

# ==================== IMPEDANCE CONTROL PARAMETERS (FOR THROWING) ====================
# Object-level impedance control during throwing phase
# Object mass matrix (diagonal, translational only)
OBJECT_MASS = 0.5  # kg (estimated box mass)
M_O = np.diag([OBJECT_MASS, OBJECT_MASS, OBJECT_MASS])  # Inertia matrix (kg)

# Damping matrix for impedance control
# Higher damping = less amplification (D⁻¹ is smaller)
D_IMPEDANCE = np.diag([20.0, 20.0, 20.0])  # Damping (N·s/m) - increased to reduce gain

# Impedance stiffness (position error term - should be small during throwing)
K_IMPEDANCE = np.diag([10.0, 10.0, 10.0])  # Stiffness (N/m) - reduced

# Maximum commanded velocities for safety
MAX_COMMANDED_VEL = 7.0  # m/s

# ==================== THROWING SETTINGS ====================
# Release happens naturally when contact is lost (matching throwing_fixed.py)

# ==================== VELOCITY CONTROLLER GAINS ====================
# Per-joint PD gains for velocity tracking
KD_GAINS = np.array([
    # Left robot
    60.0, 100.0, 70.0, 30.0, 15.0, 15.0,
    # Right robot
    60.0, 100.0, 70.0, 30.0, 15.0, 15.0
])

KI_GAINS = np.array([
    # Left robot
    0.01, 0.1, 0.01, 0.005, 0.002, 0.002,
    # Right robot
    0.01, 0.1, 0.01, 0.005, 0.002, 0.002
])

# ==================== POSITION CONTROLLER GAINS ====================
# For reaching phase
KP_GAINS = np.array([
    # Left robot
    15.0, 60.0, 15.0, 5.0, 3.0, 3.0,
    # Right robot
    15.0, 60.0, 15.0, 5.0, 3.0, 3.0
])

MAX_JOINT_VELOCITIES = np.array([
    1.2, 1.2, 1.2, 1.5, 1.5, 1.5,
    1.2, 1.2, 1.2, 1.5, 1.5, 1.5
])

POSITION_THRESHOLDS = np.array([
    # Left robot
    0.02, 0.02, 0.02, 0.03, 0.03, 0.03,
    # Right robot
    0.02, 0.02, 0.02, 0.03, 0.03, 0.03
])

# ==================== ROBOT CONFIGURATIONS ====================
# Target joint positions for reaching phases
FIRST_TARGET_POSITIONS_LEFT = np.array([-0.0805, 1.07, -0.126, 1.53, -0.00978, 0])
FIRST_TARGET_POSITIONS_RIGHT = np.array([0.105, 1.07, -0.126, -1.45, -0.00901, 0])

SECOND_TARGET_POSITIONS_LEFT = np.array([0.147, 1.16, -0.0314, 1.79, -0.00978, 0.5])
SECOND_TARGET_POSITIONS_RIGHT = np.array([-0.147, 1.16, -0.0314, -1.79, -0.0314, 0.5])

# ==================== SIMULATION SETTINGS ====================
XML_FILE_PATH = "/home/samriddhi/samriddhi_mujoco_bimanual_ds_dynamic_task/robot_description/kshitij_lifting.xml"
SLEEP_TIME = 0.001  # seconds between simulation steps

# ==================== PHASE TIMINGS ====================
READING_SENSORS_DURATION = 0.1  # seconds
HEIGHT_THRESHOLD = 0.01  # meters (tolerance for reaching target height)

# ==================== DEBUG SETTINGS ====================
PHASE_PRINT_INTERVAL = 1.0  # seconds
LIFT_DEBUG_PRINT_INTERVAL = 100  # steps
