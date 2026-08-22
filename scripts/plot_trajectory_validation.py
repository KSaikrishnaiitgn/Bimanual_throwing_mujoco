"""Plot theoretical commands/references against achieved experimental motion."""

from __future__ import annotations

import argparse

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import config.throwing_trajectory_config as config


XYZ = ("x", "y", "z")
COLORS = ("tab:blue", "tab:orange", "tab:green")


def vec(df, prefix, labels=XYZ):
    return df[[f"{prefix}_{a}" for a in labels]].to_numpy(float)


def first_index(df, phase):
    idx = np.flatnonzero(df["phase"].to_numpy() == phase)
    if not len(idx):
        raise RuntimeError(f"phase {phase!r} is absent from the log")
    return int(idx[0])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--show-3d",
        action="store_true",
        help="Open a rotatable Matplotlib 3D trajectory window.",
    )
    args = parser.parse_args()

    df = pd.read_csv("throw_trajectory_log.csv")
    required = {"x_ref_x", "v_obj_cmd_x", "ee_cmd_left_vx", "ee_actual_left_vx"}
    missing = required.difference(df.columns)
    if missing:
        raise RuntimeError(
            "Log predates diagnostic logging. Run: python main_throw_trajectory.py --headless"
        )

    t = df["t"].to_numpy(float)
    p = vec(df, "object_pos")
    v = vec(df, "object_linvel")
    k_throw = first_index(df, "THROW")
    k_release = first_index(df, "RELEASE")
    t_release = t[k_release]
    p_release_actual = df.loc[k_release, [f"release_actual_pos_{a}" for a in XYZ]].to_numpy(float)
    v_release_actual = df.loc[k_release, [f"release_actual_vel_{a}" for a in XYZ]].to_numpy(float)

    # Planned curve: logged cubic reference through release, followed by the
    # ideal no-drag ballistic equation from the planned release state.
    theoretical = np.full_like(p, np.nan)
    ref = vec(df, "x_ref")
    theoretical[k_throw:k_release + 1] = ref[k_throw:k_release + 1]
    flight = np.arange(k_release, len(df))
    dt = t[flight] - t_release
    gravity = np.array([0.0, 0.0, -9.81])
    theoretical[flight] = (
        config.RELEASE_POINT[None, :]
        + df.loc[k_release, ["v_ref_x", "v_ref_y", "v_ref_z"]].to_numpy(float)[None, :] * dt[:, None]
        + 0.5 * gravity[None, :] * dt[:, None] ** 2
    )
    # Projectile motion is valid only until first contact with the table.
    # Do not extrapolate the ideal parabola through the floor for the rest of
    # the 8 s simulation, which would distort every plot and error scale.
    landed = np.flatnonzero(
        theoretical[k_release:, 2] <= config.LANDING_POINT[2]
    )
    if len(landed):
        k_theoretical_land = k_release + int(landed[0])
        theoretical[k_theoretical_land + 1:] = np.nan

    fig = plt.figure(figsize=(16, 8), constrained_layout=True)
    gs = fig.add_gridspec(1, 2)
    ax3 = fig.add_subplot(gs[0, 0], projection="3d")
    ax3.plot(*p.T, color="black", lw=2, label="achieved box path")
    valid = np.isfinite(theoretical[:, 0])
    ax3.plot(*theoretical[valid].T, color="tab:red", ls="--", lw=2, label="calculated path")
    points = [
        (p[0], "initial actual", "tab:blue", "o"),
        (config.RELEASE_POINT, "release calculated", "tab:orange", "^"),
        (p_release_actual, "release actual", "tab:red", "x"),
        (config.LANDING_POINT, "landing calculated", "tab:green", "^"),
        (p[-1], "landing actual", "purple", "x"),
    ]
    for point, label, color, marker in points:
        ax3.scatter(*point, color=color, marker=marker, s=80, label=label)
    ax3.set(xlabel="world x [m]", ylabel="world y [m]", zlabel="world z [m]", title="Calculated vs achieved 3D box trajectory")
    ax3.legend(fontsize=8)

    axp = fig.add_subplot(gs[0, 1])
    for j, (axis, color) in enumerate(zip(XYZ, COLORS)):
        axp.plot(t, p[:, j], color=color, label=f"actual {axis}")
        axp.plot(t[valid], theoretical[valid, j], color=color, ls="--", label=f"calculated {axis}")
    axp.axvline(t[k_throw], color="gray", ls=":", label="throw start")
    axp.axvline(t_release, color="black", ls=":", label="release")
    axp.set(title="Box position: calculated vs achieved", xlabel="time [s]", ylabel="position [m]")
    axp.grid(alpha=.25); axp.legend(ncol=2, fontsize=8)

    fig.savefig("trajectory_validation_3d.png", dpi=180)

    # A separate full-size 3D figure is easier to inspect than the 3D panel in
    # the summary. With --show-3d, Matplotlib opens an interactive window:
    # left-drag rotates, scroll zooms, and right/middle-drag pans.
    fig3d = plt.figure(figsize=(11, 9), constrained_layout=True)
    ax3d = fig3d.add_subplot(111, projection="3d")
    ax3d.plot(*p.T, color="black", lw=2.5, label="achieved box path")
    ax3d.plot(
        *theoretical[valid].T,
        color="tab:red",
        ls="--",
        lw=2.5,
        label="calculated path",
    )
    for point, label, color, marker in points:
        ax3d.scatter(*point, color=color, marker=marker, s=100, label=label)
        ax3d.text(*point, f"  {label}", color=color, fontsize=9)
    ax3d.set_xlabel("world x [m]")
    ax3d.set_ylabel("world y [m]")
    ax3d.set_zlabel("world z [m]")
    ax3d.set_title("Calculated vs achieved 3D box trajectory")
    ax3d.set_box_aspect(np.ptp(p, axis=0) + 1e-6)
    ax3d.view_init(elev=24, azim=-58)
    ax3d.legend(fontsize=9)
    fig3d.savefig("box_trajectory_3d.png", dpi=200)

    fig2, axes = plt.subplots(3, 1, figsize=(14, 12), sharex=True, constrained_layout=True)
    throw_mask = (np.arange(len(df)) >= k_throw) & (np.arange(len(df)) <= k_release)
    for j, (axis, color) in enumerate(zip(XYZ, COLORS)):
        axes[0].plot(t[throw_mask], vec(df, "v_ref")[throw_mask, j], ls="--", color=color, label=f"reference {axis}")
        axes[0].plot(t[throw_mask], vec(df, "v_obj_cmd")[throw_mask, j], color=color, label=f"commanded {axis}")
        axes[0].plot(t[throw_mask], v[throw_mask, j], color=color, alpha=.45, lw=2, label=f"actual {axis}")
    axes[0].set(title="Box velocity: trajectory reference, feedback command, and achieved", ylabel="velocity [m/s]")
    axes[0].legend(ncol=3, fontsize=8); axes[0].grid(alpha=.25)

    for row, side in enumerate(("left", "right"), start=1):
        cmd = vec(df, f"ee_cmd_{side}", ("vx", "vy", "vz"))
        actual = vec(df, f"ee_actual_{side}", ("vx", "vy", "vz"))
        for j, (axis, color) in enumerate(zip(XYZ, COLORS)):
            axes[row].plot(t[throw_mask], cmd[throw_mask, j], ls="--", color=color, label=f"required {axis}")
            axes[row].plot(t[throw_mask], actual[throw_mask, j], color=color, alpha=.65, label=f"actual {axis}")
        axes[row].set(title=f"{side.capitalize()} end-effector linear velocity", ylabel="velocity [m/s]")
        axes[row].legend(ncol=3, fontsize=8); axes[row].grid(alpha=.25)

    axes[2].set_xlabel("time [s]")
    fig2.savefig("trajectory_velocity_validation.png", dpi=180)

    print("MuJoCo gravity:", gravity)
    print("actual release position:", p_release_actual)
    print("calculated release position:", config.RELEASE_POINT)
    print("actual release velocity:", v_release_actual)
    print("calculated release velocity:", vec(df, "v_ref")[k_release])
    print("actual final position:", p[-1])
    print("calculated landing point:", config.LANDING_POINT)
    print(
        "saved box_trajectory_3d.png, trajectory_validation_3d.png, "
        "and trajectory_velocity_validation.png"
    )
    if args.show_3d:
        print("Interactive 3D window: drag to rotate and scroll to zoom.")
        plt.show()
    else:
        plt.close("all")


if __name__ == "__main__":
    main()
