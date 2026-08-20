"""
check_yaw_trajectory.py

For a single candidate right_base yaw (default: 180 deg, the sweep winner
from sweep_right_base_yaw.py), this does two checks the static
pregrasp/grasp/release sweep couldn't:

  1. TRAJECTORY MARGINS -- interpolates joint-space between the solved
     grasp_right and release_right configurations in N_STEPS and reports
     the min per-joint margin at EVERY step, not just the two endpoints.
     Caveat: this is a straight-line qpos interpolation, not the actual
     THROW-phase trajectory your DS/impedance/nullspace controllers will
     produce (see ds_impedance_controller.compute_ds_accel +
     compute_dual_arm_qdot) -- a real run could dip lower between these
     samples, or the controller could take a different path through
     joint space entirely. Treat this as a sanity check on the yaw choice,
     not a substitute for re-running THROW in sim.

  2. ARM-TO-ARM COLLISION -- at each interpolated step, also holds the
     LEFT arm at its own solved grasp configuration (left_base is
     unchanged) and checks MuJoCo's contact list for any contact between
     a left_* geom and a right_* geom. The XML's existing <contact
     exclude> pairs are same-arm only (link0/link1, hand/palm), so any
     left-vs-right contact reported here is a genuine potential collision,
     not a modeling artifact.

Usage:
    python check_yaw_trajectory.py [yaw_deg]

    (defaults to 180.0 if no argument given)
"""

import sys

import numpy as np
import mujoco

import config.throwing_config as config
from mj_interface import MjInterface
from ik_solver import (
    solve_ik_multistart,
    _joint_qpos_addresses,
    _joint_ranges,
    _quat_from_approach_axis,
)

BOX_REST_POS = np.array([-0.42, 0.4, 0.4])  # matches XML <body name="box" pos="...">
N_STEPS = 15  # interpolation resolution between grasp_right and release_right


def _set_right_base_yaw(model, yaw_rad: float) -> None:
    body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "right_base")
    quat = np.zeros(4)
    mujoco.mju_axisAngle2Quat(quat, np.array([0.0, 0.0, 1.0]), yaw_rad)
    model.body_quat[body_id] = quat


def _margins(q: np.ndarray, joint_range: np.ndarray) -> np.ndarray:
    return np.minimum(q - joint_range[:, 0], joint_range[:, 1] - q)


def _solve_arm_grasp(interface, model, data, arm: str, q_home: np.ndarray):
    site_name = config.LEFT_EE_SITE if arm == "left" else config.RIGHT_EE_SITE
    joint_names = config.LEFT_JOINT_NAMES if arm == "left" else config.RIGHT_JOINT_NAMES
    grasp_world = config.LEFT_GRASP_WORLD_POS if arm == "left" else config.RIGHT_GRASP_WORLD_POS
    approach_axis = config.LEFT_PAD_APPROACH_AXIS_WORLD if arm == "left" else config.RIGHT_PAD_APPROACH_AXIS_WORLD

    joint_range = _joint_ranges(model, joint_names)
    target_quat = _quat_from_approach_axis(approach_axis)
    pad_offset = config.PAD_HALF_THICKNESS * approach_axis
    grasp_target = grasp_world - pad_offset

    q, converged, err = solve_ik_multistart(
        interface, model, data, arm, site_name,
        grasp_target, target_quat, q_home, joint_range, n_restarts=8,
    )
    if not converged:
        print(f"  WARNING: {arm} grasp IK did not converge (err={err:.5f}) "
              f"-- collision/margin results below use a best-effort pose.")
    return q, joint_range


def _solve_right_release(interface, model, data, q_grasp_right: np.ndarray):
    joint_range = _joint_ranges(model, config.RIGHT_JOINT_NAMES)
    approach_axis = config.RIGHT_PAD_APPROACH_AXIS_WORLD
    target_quat = _quat_from_approach_axis(approach_axis)
    pad_offset = config.PAD_HALF_THICKNESS * approach_axis
    box_offset = config.RIGHT_GRASP_WORLD_POS - BOX_REST_POS
    release_target = config.RELEASE_POINT + box_offset - pad_offset

    q, converged, err = solve_ik_multistart(
        interface, model, data, "right", config.RIGHT_EE_SITE,
        release_target, target_quat, q_grasp_right, joint_range, n_restarts=8,
    )
    if not converged:
        print(f"  WARNING: right release IK did not converge (err={err:.5f}) "
              f"-- treat release-side results as approximate.")
    return q


def _arm_arm_contacts(model, data) -> list:
    """Return list of (geom1_name, geom2_name) for any contact where one
    geom belongs to the left arm and the other to the right arm."""
    hits = []
    for i in range(data.ncon):
        c = data.contact[i]
        n1 = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, c.geom1) or ""
        n2 = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, c.geom2) or ""
        sides = {n1[:5], n2[:5]}
        if "left_" in sides and "right" in sides:
            hits.append((n1, n2))
    return hits


def main():
    yaw_deg = float(sys.argv[1]) if len(sys.argv) > 1 else 180.0
    yaw_rad = np.deg2rad(yaw_deg)

    model = mujoco.MjModel.from_xml_path(config.XML_PATH)
    data = mujoco.MjData(model)
    home_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_KEY, "home")
    if home_id < 0:
        raise RuntimeError(f"No keyframe named 'home' found in {config.XML_PATH}.")
    data.qpos[:] = model.key_qpos[home_id]
    mujoco.mj_forward(model, data)
    interface = MjInterface(model, data)

    left_qpos_adr = _joint_qpos_addresses(model, config.LEFT_JOINT_NAMES)
    right_qpos_adr = _joint_qpos_addresses(model, config.RIGHT_JOINT_NAMES)
    q_home_left = data.qpos[left_qpos_adr].copy()
    q_home_right = data.qpos[right_qpos_adr].copy()

    # Left arm is unaffected by right_base's yaw -- solve once at the
    # model's default (still-home) right yaw, before we touch it.
    print(f"Solving left-arm grasp pose (held fixed for the whole check)...")
    q_grasp_left, left_joint_range = _solve_arm_grasp(interface, model, data, "left", q_home_left)

    print(f"\nSetting right_base yaw = {yaw_deg} deg and solving right grasp/release...")
    _set_right_base_yaw(model, yaw_rad)
    mujoco.mj_forward(model, data)
    q_grasp_right, right_joint_range = _solve_arm_grasp(interface, model, data, "right", q_home_right)
    q_release_right = _solve_right_release(interface, model, data, q_grasp_right)

    print(f"\n{'step':>4} | {'min margin(rad)':>16} | {'worst joint':<20} | arm-arm contacts")
    print("-" * 80)

    worst_margin_overall = np.inf
    worst_step = None
    any_collision = False

    for i in range(N_STEPS + 1):
        t = i / N_STEPS
        q_right_t = (1 - t) * q_grasp_right + t * q_release_right

        data.qpos[left_qpos_adr] = q_grasp_left
        data.qpos[right_qpos_adr] = q_right_t
        mujoco.mj_forward(model, data)

        margins = _margins(q_right_t, right_joint_range)
        worst_idx = int(np.argmin(margins))
        worst_val = margins[worst_idx]
        if worst_val < worst_margin_overall:
            worst_margin_overall = worst_val
            worst_step = i

        contacts = _arm_arm_contacts(model, data)
        contact_str = ""
        if contacts:
            any_collision = True
            contact_str = f"COLLISION: {contacts[0][0]} <-> {contacts[0][1]}" + \
                           (f" (+{len(contacts) - 1} more)" if len(contacts) > 1 else "")

        print(f"{i:>4} | {worst_val:>16.4f} | {config.RIGHT_JOINT_NAMES[worst_idx]:<20} | {contact_str}")

    print("-" * 80)
    print(f"Worst margin across the whole swing: {worst_margin_overall:.4f} rad "
          f"at step {worst_step}/{N_STEPS}")
    if any_collision:
        print("\n*** ARM-TO-ARM CONTACT DETECTED at one or more steps. ***")
        print("    This yaw is likely NOT usable as-is -- either the swing path")
        print("    needs to avoid this region, or this candidate should be rejected.")
    else:
        print("\nNo arm-to-arm contact detected along this straight-line interpolation.")
        print("(Reminder: the real controller won't move in a straight joint-space")
        print(" line -- this rules out the obvious case, not every possible path.)")


if __name__ == "__main__":
    main()
