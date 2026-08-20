"""Configuration overrides for the experimental terminal-state controller.

The baseline config remains untouched; import this module only from the
experimental trajectory-based throw implementation.
"""

import numpy as np

from config.throwing_config import *  # noqa: F401,F403 - intentional baseline reuse


# Joint-limit-constrained range optimization at the fixed 30-degree launch
# angle selected this release pose.  It maximizes forward landing distance
# while retaining about 10 percent of the common end-effector speed capacity.
RELEASE_POINT = np.array([-0.42, 0.55, 0.50])

# Object-centre landing position.  The 0.2 m cube rests with its centre at
# z=0.1 m; using z=0 would incorrectly aim the centre through the floor.
LANDING_POINT = np.array([-0.42, 0.78, 0.10])

# Experimental override of the baseline launch angle.  Keeping it here means
# the original controller configuration is not modified.
THROW_ANGLE = np.deg2rad(35.0)

# Duration of the pre-release polynomial reference.  The reference is a
# cubic Hermite curve satisfying position and velocity at both endpoints.
THROW_TRAJECTORY_DURATION = 0.8

# Resolved-rate terminal trajectory feedback:
#   v_cmd = v_ref + Kx (x_ref-x) + Kv (v_ref-v)
TRAJECTORY_K_POS = np.diag([5.0, 5.0, 5.0])
TRAJECTORY_K_VEL = np.diag([0.15, 0.15, 0.15])

RELEASE_POSITION_TOLERANCE = 0.010  # m
RELEASE_VELOCITY_TOLERANCE = 0.08   # m/s
RELEASE_WINDOW = 0.20               # s after nominal terminal time

# During opening, keep both hands moving with the ballistic release velocity
# while separating quickly along the opposing pad normals.  The baseline
# stopped forward hand motion immediately and only opened 5 mm per side,
# braking the box before contact cleared.
RELEASE_COMOVE_DURATION = 0.08
RELEASE_OUTWARD_SPEED = 0.25
