"""
ik_solver.py

Arm-agnostic damped-least-squares differential inverse kinematics.

`solve_ik` takes an already-loaded (model, data, MjInterface) plus a target
Cartesian pose for a named site on a named arm, and iterates a standard
Jacobian-transpose-with-damping (Levenberg-Marquardt style) update until the
combined position+orientation error drops below `tol` or `max_iters` is
exhausted. It has no built-in notion of "grasp" or "pregrasp" -- those are
just two different target poses supplied by the caller.

`solve_ik_multistart` wraps `solve_ik`: it tries the supplied seed first,
and if that fails to converge, retries from several random seeds sampled
within the joint limits, keeping whichever attempt achieves the lowest
final error. This is a standard fix for local-minimum failures in
damped-least-squares IK, where a single fixed seed (e.g. the home
keyframe) can be a poor starting point for one arm's target even when a
solution exists elsewhere in configuration space.

Running this file directly solves IK for the four canonical waypoints used
by state_machine.py (pregrasp/grasp x left/right) and caches the resulting
7-vectors to disk as .npy files.

Note: this module is fully self-contained for joint addressing/ranges (see
_joint_qpos_addresses / _joint_ranges below) -- it does not require any
MjInterface method beyond get_site_pose and get_site_jacobian, both of
which already existed.
"""

from typing import Tuple

import numpy as np
import mujoco

import config.throwing_config as config
from mj_interface import MjInterface
from ds_impedance_controller import _joint_limit_avoidance_gradient


# ----------------------------------------------------------------------
# Internal helpers (self-contained -- does not reach into MjInterface's
# private state, only uses model/config to resolve joint addresses/ranges)
# ----------------------------------------------------------------------

def _joint_qpos_addresses(model, joint_names) -> np.ndarray:
    """qpos addresses for a list of (hinge) joint names, in the given order."""
    addrs = []
    for name in joint_names:
        jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
        if jid < 0:
            raise ValueError(f"Joint '{name}' not found in model.")
        addrs.append(int(model.jnt_qposadr[jid]))
    return np.array(addrs)


def _joint_ranges(model, joint_names) -> np.ndarray:
    """(7, 2) array of [lower, upper] limits for a list of joint names."""
    ranges = []
    for name in joint_names:
        jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
        if jid < 0:
            raise ValueError(f"Joint '{name}' not found in model.")
        ranges.append(model.jnt_range[jid])
    return np.array(ranges, dtype=np.float64)


def _rot_from_approach_axis(approach_axis: np.ndarray) -> np.ndarray:
    """
    Build a right-handed rotation matrix whose local +z axis equals the
    given (unit-normalized) approach_axis, with x/y chosen via Gram-Schmidt
    against a world-up reference (falling back to a different reference if
    approach_axis is nearly parallel to world-up, to avoid a degenerate
    cross product).
    """
    z = approach_axis / np.linalg.norm(approach_axis)
    world_up = np.array([0.0, 0.0, 1.0])
    if abs(np.dot(z, world_up)) > 0.99:
        world_up = np.array([0.0, 1.0, 0.0])
    x = np.cross(world_up, z)
    x = x / np.linalg.norm(x)
    y = np.cross(z, x)
    # Columns are the basis vectors expressed in world frame.
    return np.column_stack([x, y, z])


def _quat_from_approach_axis(approach_axis: np.ndarray) -> np.ndarray:
    """Rotation matrix -> quaternion (wxyz) for a pad-normal target orientation."""
    R = _rot_from_approach_axis(approach_axis)
    quat = np.zeros(4, dtype=np.float64)
    mujoco.mju_mat2Quat(quat, R.flatten())
    return quat


# ----------------------------------------------------------------------
# Core solver
# ----------------------------------------------------------------------

def solve_ik(interface: MjInterface, model, data, arm: str, site_name: str,
             target_pos: np.ndarray, target_quat: np.ndarray,
             q_init: np.ndarray, joint_range: np.ndarray,
             max_iters: int = 200, tol: float = 1e-4,
             damping: float = 0.01,
             nullspace_gain: float = 0.0) -> Tuple[np.ndarray, bool, float]:
    """
    Damped least-squares differential IK for a single arm/site.

    Args:
        interface: MjInterface wrapping (model, data).
        model, data: the MuJoCo model/data pair (data is mutated in place
            during the search -- caller should not rely on data.qpos for
            this arm being unchanged after calling this function).
        arm: 'left' or 'right' -- which arm's 7 joints to drive.
        site_name: the site whose pose should reach (target_pos, target_quat).
        target_pos: (3,) target world position.
        target_quat: (4,) target world orientation, wxyz.
        q_init: (7,) initial joint-angle guess.
        joint_range: (7, 2) array of [lower, upper] joint limits.
        max_iters: maximum DLS iterations.
        tol: convergence threshold on ||[e_pos; e_rot]||.
        damping: DLS damping factor (Levenberg-Marquardt lambda).
        nullspace_gain: gain on an optional secondary (nullspace) task that
            biases the solution toward joint mid-range, using the same
            Liegeois (1977) joint-limit-avoidance potential gradient as
            ds_impedance_controller.compute_dual_arm_qdot's runtime
            nullspace term (imported from there, not re-derived here, so
            there is a single source of truth for this potential).
            0.0 (the default) disables the term entirely and reduces this
            EXACTLY to the original primary-task-only behavior -- existing
            callers that don't pass this argument are unaffected.

            Motivation: solve_ik's plain DLS update has no preference among
            multiple valid IK solutions -- it converges to *a* solution and
            clips at limits, with no bias toward leaving joint-limit
            margin. A solution that converges with near-zero margin on some
            joint (e.g. a wrist joint sitting ~4 deg from its limit) is a
            legitimate IK solution but a poor *starting pose* for any
            downstream task that needs that joint to move further in that
            direction -- the manipulator Jacobian loses rank as that joint
            saturates, independent of how well-conditioned the target pose
            itself is. This term does not change what target_pos/target_quat
            is reached; it only picks a better-conditioned solution among
            the redundant set of joint angles that reach it.

    Returns:
        (q, converged, final_error_norm): q is the (7,) solution (or best
        effort if not converged), converged is True iff ||e|| < tol was
        reached within max_iters, final_error_norm is ||e|| at the last
        iteration -- always returned so a caller can tell "close but not
        quite" apart from "way off" even on failure. Callers MUST check
        the converged flag and not silently trust an unconverged result.
    """
    joint_names = config.LEFT_JOINT_NAMES if arm == "left" else config.RIGHT_JOINT_NAMES
    qpos_adr = _joint_qpos_addresses(model, joint_names)

    q = np.asarray(q_init, dtype=np.float64).copy()
    converged = False
    e = np.full(6, np.inf)

    for _ in range(max_iters):
        data.qpos[qpos_adr] = q
        mujoco.mj_forward(model, data)

        current_pos, current_quat = interface.get_site_pose(site_name)

        # Position error.
        e_pos = target_pos - current_pos

        # Orientation error: q_err = target_quat * current_quat^-1,
        # converted to a 3D rotation-vector error.
        current_quat_inv = np.zeros(4, dtype=np.float64)
        mujoco.mju_negQuat(current_quat_inv, current_quat)
        q_err = np.zeros(4, dtype=np.float64)
        mujoco.mju_mulQuat(q_err, target_quat, current_quat_inv)
        e_rot = np.zeros(3, dtype=np.float64)
        mujoco.mju_quat2Vel(e_rot, q_err, 1.0)

        e = np.concatenate([e_pos, e_rot])

        if np.linalg.norm(e) < tol:
            converged = True
            break

        J = interface.get_site_jacobian(site_name, arm)  # (6, 7)
        JJt = J @ J.T + (damping ** 2) * np.eye(6)
        J_pinv = J.T @ np.linalg.inv(JJt)  # damped pinv, (7, 6) -- reused
                                            # below for the nullspace
                                            # projector so the primary and
                                            # secondary tasks are consistent
                                            # about what "damped" means here.
        dq = J_pinv @ e

        if nullspace_gain > 0.0:
            grad_H = _joint_limit_avoidance_gradient(q, joint_range)
            dq_avoid = -nullspace_gain * grad_H
            N = np.eye(7) - J_pinv @ J
            dq = dq + N @ dq_avoid

        q = np.clip(q + dq, joint_range[:, 0], joint_range[:, 1])

    return q, converged, float(np.linalg.norm(e))


def solve_ik_multistart(interface: MjInterface, model, data, arm: str,
                         site_name: str, target_pos: np.ndarray,
                         target_quat: np.ndarray, q_init: np.ndarray,
                         joint_range: np.ndarray, n_restarts: int = 12,
                         seed: int = 0, **solve_kwargs) -> Tuple[np.ndarray, bool, float]:
    """
    Try solve_ik from q_init first; if it fails to converge, retry from
    n_restarts random seeds sampled uniformly within joint_range, keeping
    whichever attempt achieves the lowest final error norm. Returns as
    soon as any attempt converges.

    Args:
        (same as solve_ik, plus:)
        n_restarts: number of random-seed retries to attempt on failure.
        seed: RNG seed for reproducible restarts.
        **solve_kwargs: forwarded to solve_ik on every attempt -- this
            includes nullspace_gain, so passing nullspace_gain=... here
            applies it consistently across the seed attempt and all
            restarts.

    Returns:
        (q, converged, final_error_norm) -- same contract as solve_ik.
    """
    best_q, best_converged, best_err = solve_ik(
        interface, model, data, arm, site_name, target_pos, target_quat,
        q_init, joint_range, **solve_kwargs
    )
    if best_converged:
        return best_q, True, best_err

    rng = np.random.default_rng(seed)
    for i in range(n_restarts):
        q_rand = rng.uniform(joint_range[:, 0], joint_range[:, 1])
        q, converged, err = solve_ik(
            interface, model, data, arm, site_name, target_pos, target_quat,
            q_rand, joint_range, **solve_kwargs
        )
        print(f"  [{arm}] multistart attempt {i + 1}/{n_restarts}: "
              f"converged={converged}, err={err:.5f}")
        if converged:
            return q, True, err
        if err < best_err:
            best_q, best_converged, best_err = q, converged, err

    return best_q, best_converged, best_err

def _worst_case_margin(q: np.ndarray, joint_range: np.ndarray) -> float:
    """
    Smallest signed distance to either joint limit, across all joints.
    Positive means every joint has that much room on its tightest side;
    negative means at least one joint is already past its limit (shouldn't
    happen given solve_ik's np.clip, but kept signed so a caller can tell
    "barely inside" from "comfortably inside" rather than just "inside").
    """
    upper_margin = joint_range[:, 1] - q
    lower_margin = q - joint_range[:, 0]
    return float(np.minimum(upper_margin, lower_margin).min())


def solve_ik_best_margin(interface: MjInterface, model, data, arm: str,
                          site_name: str, target_pos: np.ndarray,
                          target_quat: np.ndarray, q_init: np.ndarray,
                          joint_range: np.ndarray, n_restarts: int = 12,
                          seed: int = 0, **solve_kwargs) -> Tuple[np.ndarray, bool, float]:
    """
    Same target/API contract as solve_ik_multistart, but a different
    selection strategy: ALWAYS runs q_init plus all n_restarts random
    seeds (no early return on first convergence), then among every
    attempt that converged, keeps the one with the best worst-case
    joint-limit margin (see _worst_case_margin) rather than whichever
    happened to converge first.

    Motivation: solve_ik_multistart's "first convergence wins" strategy
    means multistart's diversity is never exercised when the seed
    attempt (e.g. a warm-started or home seed) converges immediately --
    which is exactly what was happening at the grasp waypoints, so the
    IK-nullspace-bias term (see solve_ik's nullspace_gain) had only one
    local solution to nudge, and that solution's one redundant direction
    didn't happen to relieve the joint that was pinned. This function
    instead searches across the full set of restart solutions -- which
    can land on genuinely different branches of the redundant solution
    manifold, not just local perturbations of one branch -- and scores
    by margin directly, which is what we actually care about.

    Cost: this always runs 1 + n_restarts full solve_ik calls (each up
    to max_iters DLS iterations), where solve_ik_multistart runs as few
    as 1. For n_restarts=12 across 4 waypoints (2 arms x pregrasp/grasp)
    that's up to 52 solve_ik calls total when regenerating waypoints --
    still cheap in absolute terms (this only runs offline, not in the
    THROW control loop) but worth knowing if solve_ik's max_iters is
    large or n_restarts is increased further.

    Args:
        (identical to solve_ik_multistart)

    Returns:
        (q, converged, final_error_norm) -- same contract as solve_ik /
        solve_ik_multistart. If NO attempt converges, falls back to the
        lowest-error attempt (same fallback behavior as
        solve_ik_multistart), since "best margin among non-converged
        attempts" isn't a meaningful thing to optimize for.
    """
    candidates = []  # list of (q, converged, err)

    q0, converged0, err0 = solve_ik(
        interface, model, data, arm, site_name, target_pos, target_quat,
        q_init, joint_range, **solve_kwargs
    )
    candidates.append((q0, converged0, err0))
    print(f"  [{arm}] seed attempt: converged={converged0}, err={err0:.5f}, "
          f"margin={_worst_case_margin(q0, joint_range):.4f}")

    rng = np.random.default_rng(seed)
    for i in range(n_restarts):
        q_rand = rng.uniform(joint_range[:, 0], joint_range[:, 1])
        q_i, converged_i, err_i = solve_ik(
            interface, model, data, arm, site_name, target_pos, target_quat,
            q_rand, joint_range, **solve_kwargs
        )
        candidates.append((q_i, converged_i, err_i))
        print(f"  [{arm}] restart {i + 1}/{n_restarts}: converged={converged_i}, "
              f"err={err_i:.5f}, margin={_worst_case_margin(q_i, joint_range):.4f}")

    converged_candidates = [c for c in candidates if c[1]]

    if converged_candidates:
        best_q, best_converged, best_err = max(
            converged_candidates,
            key=lambda c: _worst_case_margin(c[0], joint_range),
        )
        print(f"  [{arm}] selected best-margin solution: "
              f"margin={_worst_case_margin(best_q, joint_range):.4f}, err={best_err:.5f} "
              f"(from {len(converged_candidates)}/{len(candidates)} converged attempts)")
        return best_q, True, best_err

    # Nothing converged -- fall back to lowest error, same as
    # solve_ik_multistart's failure-path behavior. Margin isn't a
    # meaningful selection criterion here since none of these are valid
    # IK solutions to begin with.
    best_q, best_converged, best_err = min(candidates, key=lambda c: c[2])
    print(f"  [{arm}] WARNING: no attempt converged; falling back to "
          f"lowest-error attempt (err={best_err:.5f})")
    return best_q, best_converged, best_err
# ----------------------------------------------------------------------
# Standalone waypoint-generation script
# ----------------------------------------------------------------------

# Nullspace joint-mid-range bias gain used when generating the cached
# pregrasp/grasp waypoints below. Picked small relative to the primary
# task's step scale (dq from the damped-pinv term is typically O(0.01-0.5)
# rad/iteration near convergence) so this nudges the solution toward
# mid-range without measurably slowing or destabilizing convergence to
# target_pos/target_quat -- NOT yet tuned beyond "small and same order as
# config.K_JOINT_LIMIT_AVOID used for the analogous runtime term".
# Increase if a cached waypoint still ends up within a few degrees of a
# joint limit; decrease if it visibly costs solve_ik extra iterations to
# converge.
IK_NULLSPACE_GAIN = 2.0


def _solve_pregrasp_and_grasp(interface, model, data, arm, site_name,
                               joint_names, grasp_pos, approach_axis,
                               q_home):
    """Solve pregrasp then grasp IK for one arm, warm-starting grasp from
    the pregrasp solution. Uses multistart retry on either stage failing
    to converge from its seed."""
    target_quat = _quat_from_approach_axis(approach_axis)
    joint_range = _joint_ranges(model, joint_names)

    pad_center_offset = config.PAD_HALF_THICKNESS * approach_axis
    grasp_pos_target = grasp_pos - pad_center_offset
    pregrasp_pos = grasp_pos_target - config.PREGRASP_STANDOFF * approach_axis

    print(f"[{arm}] solving pregrasp...")
    q_pregrasp, converged_pre, err_pre = solve_ik_best_margin(
        interface, model, data, arm, site_name,
        pregrasp_pos, target_quat, q_home, joint_range,
        nullspace_gain=IK_NULLSPACE_GAIN,
    )
    print(f"[{arm}] pregrasp IK converged: {converged_pre} (final error {err_pre:.5f})")

    print(f"[{arm}] solving grasp (continuity-prioritized from pregrasp)...")
    # The grasp target is only PREGRASP_STANDOFF away from pregrasp.  Preserve
    # that local IK branch instead of searching all branches and selecting a
    # slightly larger joint-limit margin.  Selecting grasp and pregrasp
    # independently previously caused >120-degree changes on several left-arm
    # joints (including a ~201-degree wrist flip) over this short Cartesian
    # move.  solve_ik_multistart returns immediately when the warm-started
    # pregrasp branch converges, and only explores alternatives if it does not.
    q_grasp, converged_grasp, err_grasp = solve_ik_multistart(
        interface, model, data, arm, site_name,
        grasp_pos_target, target_quat, q_pregrasp, joint_range,
        n_restarts=12,
        nullspace_gain=IK_NULLSPACE_GAIN,
    )
    print(f"[{arm}] grasp IK converged: {converged_grasp} (final error {err_grasp:.5f})")

    return q_pregrasp, converged_pre, q_grasp, converged_grasp


def main() -> None:
    model = mujoco.MjModel.from_xml_path(config.XML_PATH)
    data = mujoco.MjData(model)

    home_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_KEY, "home")
    if home_id < 0:
        raise RuntimeError(f"No keyframe named 'home' found in {config.XML_PATH}.")
    data.qpos[:] = model.key_qpos[home_id]
    mujoco.mj_forward(model, data)

    interface = MjInterface(model, data)

    arm_specs = [
        ("left", config.LEFT_GRASP_WORLD_POS, config.LEFT_PAD_APPROACH_AXIS_WORLD,
         config.LEFT_EE_SITE, config.LEFT_JOINT_NAMES),
        ("right", config.RIGHT_GRASP_WORLD_POS, config.RIGHT_PAD_APPROACH_AXIS_WORLD,
         config.RIGHT_EE_SITE, config.RIGHT_JOINT_NAMES),
    ]

    any_failed = False
    for arm, grasp_pos, approach_axis, site_name, joint_names in arm_specs:
        qpos_adr = _joint_qpos_addresses(model, joint_names)
        q_home = data.qpos[qpos_adr].copy()

        q_pregrasp, converged_pre, q_grasp, converged_grasp = _solve_pregrasp_and_grasp(
            interface, model, data, arm, site_name, joint_names,
            grasp_pos, approach_axis, q_home,
        )
        if not converged_pre or not converged_grasp:
            any_failed = True

        np.save(f"pregrasp_{arm}.npy", q_pregrasp)
        np.save(f"grasp_{arm}.npy", q_grasp)

    print(
        "Saved pregrasp_left.npy, grasp_left.npy, "
        "pregrasp_right.npy, grasp_right.npy"
    )

    if any_failed:
        print(
            "\nWARNING: at least one target did not converge even after "
            "multistart retries. Check the printed error norms above -- "
            "if they're still large (not just borderline over tol), the "
            "target pose may genuinely be outside that arm's reachable "
            "workspace given its base pose/orientation. Do not proceed "
            "to main_throw_sim.py until this is resolved."
        )


if __name__ == "__main__":
    main()
