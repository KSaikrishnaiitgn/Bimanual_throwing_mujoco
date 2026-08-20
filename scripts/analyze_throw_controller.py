"""Reproducible mathematical diagnostics for the current throw controller."""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np

import config.throwing_config as config
from release_velocity import compute_release_velocity


def simulate_original_ds(x0: np.ndarray, v0: np.ndarray, duration=3.0, dt=1e-4):
    sol = compute_release_velocity(
        config.RELEASE_POINT, config.LANDING_POINT, config.THROW_ANGLE
    )
    xr, vr = config.RELEASE_POINT, sol["v_rel_vector"]
    n = int(duration / dt) + 1
    t = np.arange(n) * dt
    x = np.empty((n, 3)); v = np.empty((n, 3))
    x[0], v[0] = x0, v0
    for k in range(n - 1):
        a = -config.K_DS @ (x[k] - xr) - config.B_DS @ (v[k] - vr)
        v[k + 1] = v[k] + dt * a
        x[k + 1] = x[k] + dt * v[k + 1]
    return t, x, v, xr, vr


def main():
    x0 = 0.5 * (config.LEFT_GRASP_WORLD_POS + config.RIGHT_GRASP_WORLD_POS)
    t, x, v, xr, vr = simulate_original_ds(x0, np.zeros(3))
    dist = np.linalg.norm(x - xr, axis=1)
    k = int(np.argmin(dist))

    # Best results from a reproducible 201-seed IK search at the release pose,
    # with a linear program maximizing speed along the exact desired 6D twist
    # under every per-joint speed bound. Ratio <= 1 is feasible.
    maximum_twist_speed = np.array([0.7812801534, 0.7786781153])
    best_limit_ratio = np.linalg.norm(vr) / maximum_twist_speed

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5), constrained_layout=True)
    labels = ("x", "y", "z")
    for j, label in enumerate(labels):
        axes[0].plot(t, x[:, j], label=f"{label}(t)")
        axes[0].axhline(xr[j], linestyle="--", alpha=0.65)
    axes[0].axvline(t[k], color="black", linestyle=":", label="closest position")
    axes[0].set(title="Original autonomous DS: position", xlabel="time [s]", ylabel="position [m]")
    axes[0].grid(alpha=0.25); axes[0].legend(fontsize=8)

    for j, label in enumerate(labels):
        axes[1].plot(t, v[:, j], label=f"v{label}(t)")
        axes[1].axhline(vr[j], linestyle="--", alpha=0.65)
    axes[1].axvline(t[k], color="black", linestyle=":")
    axes[1].set(
        title=f"At closest position: |e_p|={dist[k]:.3f} m, |e_v|={np.linalg.norm(v[k]-vr):.3f} m/s",
        xlabel="time [s]", ylabel="velocity [m/s]",
    )
    axes[1].grid(alpha=0.25); axes[1].legend(fontsize=8)

    bars = axes[2].bar(["left", "right"], best_limit_ratio)
    axes[2].axhline(1.0, color="black", linestyle="--", label="joint-speed feasibility limit")
    for bar, value in zip(bars, best_limit_ratio):
        axes[2].text(bar.get_x() + bar.get_width()/2, value, f"{value:.2f}x",
                     ha="center", va="bottom")
    axes[2].set(title="Best of 201 IK seeds + optimal velocity allocation", ylabel="required speed / feasible speed")
    axes[2].grid(axis="y", alpha=0.25); axes[2].legend(fontsize=8)
    fig.savefig("throw_controller_diagnostics.png", dpi=180)

    equilibrium = xr + np.linalg.solve(config.K_DS, config.B_DS @ vr)
    print("required_release_velocity", vr, "speed", np.linalg.norm(vr))
    print("original_DS_fixed_point_position", equilibrium)
    print("closest_position_time", t[k])
    print("closest_position_error", dist[k])
    print("velocity_error_at_closest_position", np.linalg.norm(v[k] - vr))
    print("best_joint_limit_ratios", best_limit_ratio)
    print("saved throw_controller_diagnostics.png")


if __name__ == "__main__":
    main()
