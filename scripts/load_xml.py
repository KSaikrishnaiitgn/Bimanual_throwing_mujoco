import mujoco
from mujoco.viewer import launch_passive
import numpy as np

XML_PATH = "/home/iitgn-robotics/Saikrishna/Bimanual_throwing_mujoco/robot_description/Dual_franka.xml"

# Load model and data
model = mujoco.MjModel.from_xml_path(XML_PATH)
data = mujoco.MjData(model)

# Bodies whose local axes we want to see, world = fixed reference
BODIES_TO_AXES = ["left_base", "right_base", "box"]
body_ids = [mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, n) for n in BODIES_TO_AXES]

AXIS_LEN = 0.25   # m, make longer/shorter depending on how it looks at your scale
AXIS_RAD = 0.004
AXIS_COLORS = [
    np.array([1, 0, 0, 1], dtype=np.float64),  # x = red
    np.array([0, 1, 0, 1], dtype=np.float64),  # y = green
    np.array([0, 0, 1, 1], dtype=np.float64),  # z = blue
]

def draw_frame(scn, pos, mat, length=AXIS_LEN, radius=AXIS_RAD):
    """Draw one RGB triad at `pos` oriented by 3x3 rotation `mat` (world-frame axes = columns of mat)."""
    for axis_idx in range(3):
        if scn.ngeom >= scn.maxgeom:
            return
        direction = mat[:, axis_idx]
        endpoint = pos + direction * length
        g = scn.geoms[scn.ngeom]
        mujoco.mjv_initGeom(
            g, mujoco.mjtGeom.mjGEOM_ARROW, np.zeros(3),
            np.zeros(3), np.zeros(9), AXIS_COLORS[axis_idx],
        )
        mujoco.mjv_connector(g, mujoco.mjtGeom.mjGEOM_ARROW, radius, pos, endpoint)
        scn.ngeom += 1

mujoco.mj_step(model, data)

with launch_passive(model, data) as viewer:
    # also show MuJoCo's own body-frame overlay as a sanity cross-check
    viewer.opt.frame = mujoco.mjtFrame.mjFRAME_NONE  # set to mjFRAME_BODY to see ALL body frames instead

    while viewer.is_running():
        mujoco.mj_step(model, data)

        viewer.user_scn.ngeom = 0
        for bid in body_ids:
            pos = data.xpos[bid].copy()
            mat = data.xmat[bid].reshape(3, 3).copy()
            draw_frame(viewer.user_scn, pos, mat)

        viewer.sync()