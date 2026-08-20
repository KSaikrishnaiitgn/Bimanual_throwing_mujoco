"""
sweep_right_base_yaw.py

Explores candidate yaw angles for right_base and reports per-joint
position-limit margins for the right arm across pregrasp, grasp, and an
approximate release-swing target -- so a mirrored (or other) base
orientation can be picked based on measured margins rather than a
hand-guessed delta.

Why sweep instead of computing a single Delta-phi analytically:
The closed-form relationship (required_joint1 = target_azimuth - phi -
const) only holds for joint1. Joints 2-7's values for a given wrist pose
come out of the 7-DOF IK solver's redundancy resolution, which does not
shift in closed form when phi changes. So the only reliable way to know
whether a candidate yaw actually helps ALL seven joints (not just
joint1) is to re-solve IK and measure.

Approximations (read before trusting the numbers):
- Release-swing target for the right EE = RELEASE_POINT + (RIGHT_GRASP_
  WORLD_POS - BOX_REST_POS), i.e. the grasp-pose pad-to-box offset carried
  over unchanged to the release pose. This assumes the box does not
  rotate during the swing, consistent with the DS controller (see
  ds_impedance_controller.compute_ds_accel) driving position only. If
  that assumption doesn't hold, treat "release" margins here as
  approximate, not final -- cross-check against the real THROW-phase
  debug trace once you've picked a candidate.
- Only right_base's yaw is swept; left_base and all grasp-geometry
  constants (which are world-frame, not base-relative) are untouched.
- IK re-solves from scratch (multistart) per stage per candidate, so this
  can take a while. Trim CANDIDATE_YAWS_DEG or n_restarts if it's slow.

Usage:
    python sweep_right_base_yaw.py

After picking a yaw from the summary table:
  1. Hardcode it into Dual_franka.xml's right_base euler (still pure
     z-yaw, e.g. euler="0 0 <chosen_rad>").
  2. Re-run ik_solver.py's main() to regenerate pregrasp_right.npy /
     grasp_right.npy for real (this script does NOT overwrite those files).
  3. Re-check frame_inspector.py's sensor-sign test and RELEASE_POINT
     reachability per the existing config.py comments.
"""

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

# Matches the XML's <body name="box" pos="-0.42 0.4 0.4">, i.e. the box's
# rest pose that RIGHT_GRASP_WORLD_POS / LEFT_GRASP_WORLD_POS are offset from.
BOX_REST_POS = np.array([-0.42, 0.4, 0.4])

# Candidates to try. -90 deg (mirroring the left arm's +90 deg) is the
# principled first guess; the others are here so the sweep isn't just
# confirming one hypothesis.
CANDIDATE_YAWS_DEG = [90.0, -90.0, 0.0, 180.0, -135.0, 135.0]

N_RESTARTS = 8  # lower than ik_solver's default 12 to keep the sweep quicker


def _set_right_base_yaw(model, yaw_rad: float) -> None:
    """Overwrite right_base's world orientation in place. Valid without
    reloading the XML because right_base's parent is worldbody, so
    body_quat directly IS its world orientation -- mj_forward will
    propagate it through the whole subtree correctly."""
    body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "right_base")
    if body_id < 0:
        raise ValueError("Body 'right_base' not found in model.")
    quat = np.zeros(4)
    mujoco.mju_axisAngle2Quat(quat, np.array([0.0, 0.0, 1.0]), yaw_rad)
    model.body_quat[body_id] = quat


def _margins(q: np.ndarray, joint_range: np.ndarray) -> np.ndarray:
    """Distance from each joint's qpos to its NEAREST limit (rad, per joint).
    Smaller = closer to saturating; this is what actually matters, not
    just joint1's."""
    return np.minimum(q - joint_range[:, 0], joint_range[:, 1] - q)


def evaluate_yaw(model, data, interface, yaw_rad: float, q_home_right: np.ndarray):
    _set_right_base_yaw(model, yaw_rad)
    mujoco.mj_forward(model, data)

    joint_range = _joint_ranges(model, config.RIGHT_JOINT_NAMES)
    approach_axis = config.RIGHT_PAD_APPROACH_AXIS_WORLD
    target_quat = _quat_from_approach_axis(approach_axis)
    pad_offset = config.PAD_HALF_THICKNESS * approach_axis

    results = {}

    # --- Grasp pose ---
    grasp_target = config.RIGHT_GRASP_WORLD_POS - pad_offset
    q_grasp, conv_g, err_g = solve_ik_multistart(
        interface, model, data, "right", config.RIGHT_EE_SITE,
        grasp_target, target_quat, q_home_right, joint_range,
        n_restarts=N_RESTARTS,
    )
    results["grasp"] = (q_grasp, conv_g, err_g)

    # --- Pregrasp pose ---
    pregrasp_target = grasp_target - config.PREGRASP_STANDOFF * approach_axis
    q_pregrasp, conv_p, err_p = solve_ik_multistart(
        interface, model, data, "right", config.RIGHT_EE_SITE,
        pregrasp_target, target_quat, q_home_right, joint_range,
        n_restarts=N_RESTARTS,
    )
    results["pregrasp"] = (q_pregrasp, conv_p, err_p)

    # --- Approximate release-swing pose (see module docstring caveat) ---
    box_offset = config.RIGHT_GRASP_WORLD_POS - BOX_REST_POS
    release_target = config.RELEASE_POINT + box_offset - pad_offset
    q_release, conv_r, err_r = solve_ik_multistart(
        interface, model, data, "right", config.RIGHT_EE_SITE,
        release_target, target_quat, q_grasp, joint_range,  # warm-start from grasp
        n_restarts=N_RESTARTS,
    )
    results["release"] = (q_release, conv_r, err_r)

    return results, joint_range


def main():
    model = mujoco.MjModel.from_xml_path(config.XML_PATH)
    data = mujoco.MjData(model)
    home_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_KEY, "home")
    if home_id < 0:
        raise RuntimeError(f"No keyframe named 'home' found in {config.XML_PATH}.")
    data.qpos[:] = model.key_qpos[home_id]
    mujoco.mj_forward(model, data)
    interface = MjInterface(model, data)

    right_qpos_adr = _joint_qpos_addresses(model, config.RIGHT_JOINT_NAMES)
    q_home_right = data.qpos[right_qpos_adr].copy()

    print(f"{'yaw(deg)':>9} | {'stage':<9} | {'converged':<9} | "
          f"{'min margin(rad)':>16} | worst joint")
    print("-" * 78)

    summary = []
    for yaw_deg in CANDIDATE_YAWS_DEG:
        yaw_rad = np.deg2rad(yaw_deg)
        results, joint_range = evaluate_yaw(model, data, interface, yaw_rad, q_home_right)

        worst_overall = np.inf
        all_converged = True
        for stage, (q, converged, err) in results.items():
            margins = _margins(q, joint_range)
            worst_idx = int(np.argmin(margins))
            worst_val = margins[worst_idx]
            worst_overall = min(worst_overall, worst_val)
            all_converged = all_converged and converged
            flag = "" if converged else f"  <-- IK DID NOT CONVERGE (err={err:.4f})"
            print(f"{yaw_deg:>9.1f} | {stage:<9} | {str(converged):<9} | "
                  f"{worst_val:>16.4f} | {config.RIGHT_JOINT_NAMES[worst_idx]}{flag}")

        summary.append((yaw_deg, worst_overall, all_converged))
        print()

    print("=" * 78)
    print("Summary (worst-case margin across grasp/pregrasp/release; converged only):")
    feasible = [s for s in summary if s[2]]
    if not feasible:
        print("  No candidate converged on every stage -- widen CANDIDATE_YAWS_DEG,")
        print("  or check that RELEASE_POINT / grasp targets are sane to begin with.")
        return

    best = max(feasible, key=lambda s: s[1])
    for yaw_deg, worst, converged in sorted(summary, key=lambda s: -s[1]):
        marker = "  <== best" if (yaw_deg, worst, converged) == best else ""
        conv_str = "ok" if converged else "FAILED to converge on >=1 stage"
        print(f"  yaw={yaw_deg:>7.1f} deg : worst-case margin = {worst:.4f} rad  "
              f"[{conv_str}]{marker}")

    print(
        "\nNOTE: 'release' margins here use an approximated EE target (see module "
        "docstring). Before finalizing, re-verify the chosen yaw's margins against "
        "an actual THROW-phase debug trace, not just this static IK check."
    )


if __name__ == "__main__":
    main()
