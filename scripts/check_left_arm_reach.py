"""
check_left_arm_reach.py

Standalone diagnostic: is config.RELEASE_POINT actually well-conditioned
for the LEFT arm, or is the THROW-phase singularity (J_H_sv_min -> ~2e-5,
left_twist-only, per the SVD debug trace) a symptom of the target being
outside/at-the-edge of the left arm's reachable workspace?

Does NOT modify main_throw_sim.py, state_machine.py, or config.py.
Reuses ik_solver.solve_ik_multistart as-is (no changes to that file
either) -- this script only calls it with a different target.

Run from the scripts/ directory:
    python check_left_arm_reach.py

Assumption (stated explicitly -- verify if results look off): orientation
is held fixed throughout THROW, so the left EE's offset from the box
center stays close to its grasp-time offset. config.py's own comment
says LEFT_GRASP_SITE is box-local (0.1, 0, 0), so this probe targets the
left EE at RELEASE_POINT + [0.1, 0, 0] in world frame. If your actual
box orientation at release is far from identity, this offset assumption
breaks down -- rerun with the real offset if so.
"""

import numpy as np
import mujoco

import config.throwing_config as config
from mj_interface import MjInterface
from ik_solver import solve_ik_multistart, _joint_ranges, _quat_from_approach_axis


def main() -> None:
    model = mujoco.MjModel.from_xml_path(config.XML_PATH)
    data = mujoco.MjData(model)

    home_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_KEY, "home")
    if home_id < 0:
        raise RuntimeError(f"No keyframe named 'home' found in {config.XML_PATH}.")
    data.qpos[:] = model.key_qpos[home_id]
    mujoco.mj_forward(model, data)

    interface = MjInterface(model, data)

    # --- Target: left EE pose at the moment the box should be at RELEASE_POINT ---
    box_local_offset = np.array([0.1, 0.0, 0.0])  # per config.py comment on LEFT_GRASP_SITE
    target_pos = config.RELEASE_POINT + box_local_offset
    # Orientation held fixed since grasp -> reuse the same target orientation
    # ik_solver used to solve grasp_left.npy (pad approach axis convention).
    target_quat = _quat_from_approach_axis(config.LEFT_PAD_APPROACH_AXIS_WORLD)

    joint_range = _joint_ranges(model, config.LEFT_JOINT_NAMES)

    # Seed from the actual grasp configuration (more representative of the
    # THROW-phase starting point than the home keyframe).
    q_grasp_left = np.load("grasp_left.npy")

    print(f"Target left-EE position (RELEASE_POINT + box-local offset): {target_pos}")
    print(f"Seeding IK from grasp_left.npy: {q_grasp_left}")

    q_sol, converged, err = solve_ik_multistart(
        interface, model, data, "left", config.LEFT_EE_SITE,
        target_pos, target_quat, q_grasp_left, joint_range,
        n_restarts=20, seed=0,
    )

    print(f"\nConverged: {converged}, final error norm: {err:.6f}")
    print(f"Solution q_left: {np.round(q_sol, 4)}")

    # Set data to the solution and inspect the Jacobian condition there --
    # this is the number that actually answers "reachable but singular"
    # vs "genuinely out of reach".
    qpos_adr = np.array([
        int(model.jnt_qposadr[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, n)])
        for n in config.LEFT_JOINT_NAMES
    ])
    data.qpos[qpos_adr] = q_sol
    mujoco.mj_forward(model, data)
    J = interface.get_site_jacobian(config.LEFT_EE_SITE, "left")  # (6,7)
    svals = np.linalg.svd(J, compute_uv=False)
    print(f"\nSingular values of left-arm Jacobian at solution: {np.round(svals, 5)}")
    print(f"sv_min = {svals[-1]:.6f}, condition number = {svals[0]/svals[-1]:.2f}")

    margin_lo = q_sol - joint_range[:, 0]
    margin_hi = joint_range[:, 1] - q_sol
    margin = np.minimum(margin_lo, margin_hi)
    worst = int(np.argmin(margin))
    print(f"Tightest joint-limit margin: {config.LEFT_JOINT_NAMES[worst]} "
          f"= {margin[worst]:.4f} rad")

    print("\n--- Interpretation ---")
    if not converged:
        print("NOT REACHABLE (or not reachable with this orientation/seed): "
              "IK failed to converge even after multistart. This supports "
              "approach #2 (move RELEASE_POINT) over approach #1 (just add "
              "damping) -- damping can't fix a target that isn't actually there.")
    elif svals[-1] < 0.01:
        print("REACHABLE BUT NEAR-SINGULAR at this exact target: IK converges, "
              "but sv_min is small, consistent with the THROW-phase SVD trace. "
              "This is the 'full extension' case -- approach #2 (pull the "
              "target inward slightly) is likely still the better fix, since "
              "arm's margin for tracking a *moving* target through this pose "
              "is thin even if the static pose itself is technically reachable.")
    else:
        print("REACHABLE WITH GOOD CONDITIONING at the static target. If so, "
              "the singularity seen during THROW is more likely a *path* "
              "issue (the swing trajectory passes through a bad configuration "
              "even though start and end are fine) rather than RELEASE_POINT "
              "itself being unreachable -- worth sampling a few intermediate "
              "points along the expected swing (e.g. box position at t=13s, "
              "15s, 16.5s from your logged trace) through this same script.")


if __name__ == "__main__":
    main()
