"""Map the rigid dual-arm grasp workspace for the box.

The box orientation, both end-effector orientations, and both contact offsets
are copied from the cached grasp configuration.  A box-centre sample is marked
reachable only when both arms solve IK within their joint limits.

Example:
    python3 constrained_grasp_workspace.py --spacing 0.05
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib.pyplot as plt
import mujoco
import numpy as np

import config.throwing_config as config
from ik_solver import _joint_qpos_addresses, _joint_ranges, solve_ik, solve_ik_multistart
from mj_interface import MjInterface


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Sample the common dual-arm workspace at the grasp orientation."
    )
    parser.add_argument("--spacing", type=float, default=0.05, help="Grid spacing [m].")
    parser.add_argument("--x", nargs=2, type=float, default=(-0.75, -0.10), metavar=("MIN", "MAX"))
    parser.add_argument("--y", nargs=2, type=float, default=(0.15, 1.00), metavar=("MIN", "MAX"))
    parser.add_argument("--z", nargs=2, type=float, default=(0.15, 0.90), metavar=("MIN", "MAX"))
    parser.add_argument("--max-iters", type=int, default=160)
    parser.add_argument("--tol", type=float, default=2e-3, help="6D IK error tolerance.")
    parser.add_argument("--damping", type=float, default=0.02)
    parser.add_argument("--output-prefix", default="grasp_workspace")
    parser.add_argument(
        "--check-point", nargs=3, type=float, action="append", metavar=("X", "Y", "Z"),
        help="Check an exact box-centre point. May be supplied multiple times.",
    )
    parser.add_argument(
        "--check-only", action="store_true",
        help="Only check exact points; do not generate a workspace grid.",
    )
    return parser.parse_args()


def grid_axis(bounds: list[float] | tuple[float, float], spacing: float) -> np.ndarray:
    lo, hi = bounds
    if spacing <= 0 or hi < lo:
        raise ValueError("Require positive spacing and MIN <= MAX for every axis.")
    return np.arange(lo, hi + 0.5 * spacing, spacing)


def set_arm_q(data, addrs: np.ndarray, q: np.ndarray) -> None:
    data.qpos[addrs] = q


def site_pose(interface: MjInterface, site: str) -> tuple[np.ndarray, np.ndarray]:
    pos, quat = interface.get_site_pose(site)
    return pos.copy(), quat.copy()


def smallest_singular_value(interface: MjInterface, site: str, arm: str) -> float:
    return float(np.linalg.svd(interface.get_site_jacobian(site, arm), compute_uv=False)[-1])


def joint_margin(q: np.ndarray, ranges: np.ndarray) -> float:
    return float(np.minimum(q - ranges[:, 0], ranges[:, 1] - q).min())


def save_csv(path: Path, rows: list[dict[str, float | int]]) -> None:
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def plot_results(path: Path, rows: list[dict[str, float | int]], reference_box: np.ndarray) -> None:
    reachable = np.array(
        [[r["x"], r["y"], r["z"]] for r in rows if r["reachable"]], dtype=float
    )
    fig = plt.figure(figsize=(13, 10), constrained_layout=True)
    ax3 = fig.add_subplot(2, 2, 1, projection="3d")
    ax_xy = fig.add_subplot(2, 2, 2)
    ax_xz = fig.add_subplot(2, 2, 3)
    ax_yz = fig.add_subplot(2, 2, 4)

    if len(reachable):
        color = reachable[:, 2]
        ax3.scatter(reachable[:, 0], reachable[:, 1], reachable[:, 2], c=color,
                    cmap="viridis", s=9, alpha=0.7)
        ax_xy.scatter(reachable[:, 0], reachable[:, 1], c=reachable[:, 2],
                      cmap="viridis", s=9, alpha=0.65)
        ax_xz.scatter(reachable[:, 0], reachable[:, 2], c=reachable[:, 1],
                      cmap="plasma", s=9, alpha=0.65)
        ax_yz.scatter(reachable[:, 1], reachable[:, 2], c=reachable[:, 0],
                      cmap="cividis", s=9, alpha=0.65)

    points = [
        (reference_box, "grasp/reference", "black", "o"),
        (np.asarray(config.RELEASE_POINT), "configured release", "red", "*"),
    ]
    for point, label, color, marker in points:
        ax3.scatter(*point, color=color, marker=marker, s=100, label=label)
        ax_xy.scatter(point[0], point[1], color=color, marker=marker, s=70)
        ax_xz.scatter(point[0], point[2], color=color, marker=marker, s=70)
        ax_yz.scatter(point[1], point[2], color=color, marker=marker, s=70)

    ax3.set(xlabel="x [m]", ylabel="y [m]", zlabel="z [m]", title="Rigid dual-arm grasp workspace")
    ax3.legend(loc="best")
    for ax, xlabel, ylabel, title in (
        (ax_xy, "x [m]", "y [m]", "XY projection"),
        (ax_xz, "x [m]", "z [m]", "XZ projection"),
        (ax_yz, "y [m]", "z [m]", "YZ projection"),
    ):
        ax.set(xlabel=xlabel, ylabel=ylabel, title=title)
        ax.grid(True, alpha=0.25)
        ax.set_aspect("equal", adjustable="box")
    fig.savefig(path, dpi=180)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    model = mujoco.MjModel.from_xml_path(config.XML_PATH)
    data = mujoco.MjData(model)
    interface = MjInterface(model, data)

    home_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_KEY, "home")
    if home_id < 0:
        raise RuntimeError("The MuJoCo model has no 'home' keyframe.")
    data.qpos[:] = model.key_qpos[home_id]

    left_addrs = _joint_qpos_addresses(model, config.LEFT_JOINT_NAMES)
    right_addrs = _joint_qpos_addresses(model, config.RIGHT_JOINT_NAMES)
    left_ranges = _joint_ranges(model, config.LEFT_JOINT_NAMES)
    right_ranges = _joint_ranges(model, config.RIGHT_JOINT_NAMES)
    q_left_ref = np.load("grasp_left.npy")
    q_right_ref = np.load("grasp_right.npy")
    set_arm_q(data, left_addrs, q_left_ref)
    set_arm_q(data, right_addrs, q_right_ref)
    mujoco.mj_forward(model, data)

    box_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, config.BOX_BODY_NAME)
    reference_box = data.xpos[box_id].copy()
    left_ref_pos, left_ref_quat = site_pose(interface, config.LEFT_EE_SITE)
    right_ref_pos, right_ref_quat = site_pose(interface, config.RIGHT_EE_SITE)
    left_offset = left_ref_pos - reference_box
    right_offset = right_ref_pos - reference_box

    xs, ys, zs = (grid_axis(getattr(args, axis), args.spacing) for axis in ("x", "y", "z"))
    total = len(xs) * len(ys) * len(zs)
    print(f"Reference box centre: {reference_box}")
    print(f"Left/right EE offsets: {left_offset}, {right_offset}")

    # Check the two points that matter immediately using the exact coordinates
    # (they need not lie on the sampled grid).
    check_points = [
        ("grasp/reference", reference_box),
        ("configured release", np.asarray(config.RELEASE_POINT)),
    ]
    check_points.extend(
        (f"user point {index}", np.asarray(point, dtype=float))
        for index, point in enumerate(args.check_point or [], start=1)
    )
    for label, centre in check_points:
        _, left_ok, left_err = solve_ik_multistart(
            interface, model, data, "left", config.LEFT_EE_SITE,
            centre + left_offset, left_ref_quat, q_left_ref, left_ranges,
            n_restarts=12, seed=101,
            max_iters=max(args.max_iters, 500), tol=args.tol,
            damping=args.damping, nullspace_gain=0.03,
        )
        _, right_ok, right_err = solve_ik_multistart(
            interface, model, data, "right", config.RIGHT_EE_SITE,
            centre + right_offset, right_ref_quat, q_right_ref, right_ranges,
            n_restarts=12, seed=202,
            max_iters=max(args.max_iters, 500), tol=args.tol,
            damping=args.damping, nullspace_gain=0.03,
        )
        print(
            f"Exact point '{label}' {centre}: reachable={left_ok and right_ok} "
            f"(left_ok={left_ok}, error={left_err:.6f}; "
            f"right_ok={right_ok}, error={right_err:.6f})"
        )
    if args.check_only:
        return
    print(f"Sampling {total} box positions ({len(xs)} x {len(ys)} x {len(zs)})")

    rows: list[dict[str, float | int]] = []
    q_left_seed = q_left_ref.copy()
    q_right_seed = q_right_ref.copy()
    count = 0
    for iz, z in enumerate(zs):
        y_order = ys if iz % 2 == 0 else ys[::-1]
        for iy, y in enumerate(y_order):
            x_order = xs if iy % 2 == 0 else xs[::-1]
            for x in x_order:
                centre = np.array([x, y, z])
                ql, ok_l, err_l = solve_ik(
                    interface, model, data, "left", config.LEFT_EE_SITE,
                    centre + left_offset, left_ref_quat, q_left_seed, left_ranges,
                    max_iters=args.max_iters, tol=args.tol, damping=args.damping,
                    nullspace_gain=0.03,
                )
                if ok_l:
                    q_left_seed = ql
                qr, ok_r, err_r = solve_ik(
                    interface, model, data, "right", config.RIGHT_EE_SITE,
                    centre + right_offset, right_ref_quat, q_right_seed, right_ranges,
                    max_iters=args.max_iters, tol=args.tol, damping=args.damping,
                    nullspace_gain=0.03,
                )
                if ok_r:
                    q_right_seed = qr

                set_arm_q(data, left_addrs, ql)
                set_arm_q(data, right_addrs, qr)
                mujoco.mj_forward(model, data)
                reachable = bool(ok_l and ok_r)
                rows.append({
                    "x": float(x), "y": float(y), "z": float(z),
                    "reachable": int(reachable),
                    "left_ok": int(ok_l), "right_ok": int(ok_r),
                    "left_error": err_l, "right_error": err_r,
                    "left_joint_margin": joint_margin(ql, left_ranges),
                    "right_joint_margin": joint_margin(qr, right_ranges),
                    "left_sigma_min": smallest_singular_value(interface, config.LEFT_EE_SITE, "left"),
                    "right_sigma_min": smallest_singular_value(interface, config.RIGHT_EE_SITE, "right"),
                })
                count += 1
                if count % 250 == 0 or count == total:
                    print(f"  {count}/{total} samples")

    prefix = Path(args.output_prefix)
    csv_path = prefix.with_suffix(".csv")
    npz_path = prefix.with_suffix(".npz")
    png_path = prefix.with_suffix(".png")
    save_csv(csv_path, rows)
    np.savez_compressed(
        npz_path,
        samples=np.array([[r[k] for k in ("x", "y", "z", "reachable")]
                          for r in rows], dtype=float),
        reference_box=reference_box,
        left_offset=left_offset,
        right_offset=right_offset,
        left_quat=left_ref_quat,
        right_quat=right_ref_quat,
    )
    plot_results(png_path, rows, reference_box)

    reachable_count = sum(int(r["reachable"]) for r in rows)
    print(f"Reachable: {reachable_count}/{total} ({100 * reachable_count / total:.1f}%)")
    print(f"Saved {csv_path}, {npz_path}, and {png_path}")


if __name__ == "__main__":
    main()
