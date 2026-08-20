"""
logger.py

`SimLogger` records one row of scalars/arrays per sim step via `.record(...)`
and can later export the log as a pandas DataFrame (`.to_dataframe()`) or
render a 5-panel matplotlib summary figure (`.plot_summary(...)`).

NOTE ON INPUTS NOT COVERED BY `.record(...)`'s SIGNATURE:
`p_rel`, `LANDING_POINT`, and `v_rel_scalar` (used by panels 1 and 2) are
locked once at the SQUEEZE_GRASP -> THROW transition (see
`ThrowStateMachine`) and are not part of the per-step `record(...)` call.
`plot_summary` therefore accepts them as optional keyword arguments:
  - `landing_point` defaults to `config.LANDING_POINT` if not given.
  - `p_rel` / `v_rel_scalar` default to `None` and their markers are simply
    omitted from panels 1/2 if not supplied. Pass
    `state_machine.p_rel` / `np.linalg.norm(state_machine.v_rel_vector)`
    from the caller once the throw phase has been entered.

Depends on: numpy, pandas, matplotlib, config.py.
"""

from __future__ import annotations

from enum import Enum
from typing import Optional

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import config.throwing_config as config


_WRENCH_LABELS = ["Fx", "Fy", "Fz", "Tx", "Ty", "Tz"]
_JOINT_LABELS = [f"j{i + 1}" for i in range(7)]
_XYZ_LABELS = ["x", "y", "z"]
_QUAT_LABELS = ["w", "x", "y", "z"]


def _phase_str(phase) -> str:
    """Normalize a Phase enum member (or plain string) to a plot/DF-friendly str."""
    return phase.value if isinstance(phase, Enum) else str(phase)


class SimLogger:
    def __init__(self):
        self._t: list[float] = []
        self._phase: list[str] = []

        self._object_state: list[dict] = []

        self._F_meas_left: list[np.ndarray] = []
        self._F_meas_right: list[np.ndarray] = []
        self._F_star_left: list[np.ndarray] = []
        self._F_star_right: list[np.ndarray] = []

        self._qdot_left: list[np.ndarray] = []
        self._qdot_right: list[np.ndarray] = []

        self._torque_left: list[np.ndarray] = []
        self._torque_right: list[np.ndarray] = []

        self._vel_clipped_left: list[bool] = []
        self._vel_clipped_right: list[bool] = []

        self._torque_clipped_left: list[np.ndarray] = []
        self._torque_clipped_right: list[np.ndarray] = []

    # ------------------------------------------------------------------
    def record(
        self,
        t: float,
        phase,
        object_state: dict,
        F_meas_left: np.ndarray,
        F_meas_right: np.ndarray,
        F_star_left: np.ndarray,
        F_star_right: np.ndarray,
        qdot_left: np.ndarray,
        qdot_right: np.ndarray,
        torque_left: np.ndarray,
        torque_right: np.ndarray,
        vel_clipped_left: bool,
        vel_clipped_right: bool,
        torque_clipped_left: np.ndarray,
        torque_clipped_right: np.ndarray,
    ) -> None:
        """Append one step's worth of data. All array-likes are copied so
        later in-place mutation of the caller's buffers (e.g. MjInterface
        reusing a scratch array) can't retroactively corrupt the log."""
        self._t.append(float(t))
        self._phase.append(_phase_str(phase))

        # object_state is a dict of small arrays; copy it shallowly plus
        # copy each contained array so it's fully detached from the source.
        self._object_state.append({k: np.asarray(v).copy() for k, v in object_state.items()})

        self._F_meas_left.append(np.asarray(F_meas_left, dtype=float).copy())
        self._F_meas_right.append(np.asarray(F_meas_right, dtype=float).copy())
        self._F_star_left.append(np.asarray(F_star_left, dtype=float).copy())
        self._F_star_right.append(np.asarray(F_star_right, dtype=float).copy())

        self._qdot_left.append(np.asarray(qdot_left, dtype=float).copy())
        self._qdot_right.append(np.asarray(qdot_right, dtype=float).copy())

        self._torque_left.append(np.asarray(torque_left, dtype=float).copy())
        self._torque_right.append(np.asarray(torque_right, dtype=float).copy())

        self._vel_clipped_left.append(bool(vel_clipped_left))
        self._vel_clipped_right.append(bool(vel_clipped_right))

        self._torque_clipped_left.append(np.asarray(torque_clipped_left, dtype=bool).copy())
        self._torque_clipped_right.append(np.asarray(torque_clipped_right, dtype=bool).copy())

    # ------------------------------------------------------------------
    def to_dataframe(self) -> pd.DataFrame:
        """Flatten every recorded field into a wide DataFrame, one row per
        sim step. Vector fields are expanded into per-component columns
        (e.g. `F_meas_left_Fx`, `qdot_left_j3`, `torque_clipped_right_j7`)
        so everything downstream (plotting, CSV export, pandas filtering)
        works with plain scalar columns."""
        n = len(self._t)
        cols: dict[str, np.ndarray] = {
            "t": np.array(self._t),
            "phase": np.array(self._phase, dtype=object),
        }

        # object_state: pos(3,), quat(4,), linvel(3,), angvel(3,)
        obj_keys = self._object_state[0].keys() if n else []
        for key in obj_keys:
            stacked = np.stack([s[key] for s in self._object_state])  # (n, k)
            labels = _QUAT_LABELS if key == "quat" else _XYZ_LABELS
            for i, label in enumerate(labels[: stacked.shape[1]]):
                cols[f"object_{key}_{label}"] = stacked[:, i]

        def _expand_wrench(name: str, data: list[np.ndarray]) -> None:
            stacked = np.stack(data)  # (n, 6)
            for i, label in enumerate(_WRENCH_LABELS):
                cols[f"{name}_{label}"] = stacked[:, i]

        _expand_wrench("F_meas_left", self._F_meas_left)
        _expand_wrench("F_meas_right", self._F_meas_right)
        _expand_wrench("F_star_left", self._F_star_left)
        _expand_wrench("F_star_right", self._F_star_right)

        def _expand_joint(name: str, data: list[np.ndarray], dtype=float) -> None:
            stacked = np.stack(data).astype(dtype)  # (n, 7)
            for i, label in enumerate(_JOINT_LABELS):
                cols[f"{name}_{label}"] = stacked[:, i]

        _expand_joint("qdot_left", self._qdot_left)
        _expand_joint("qdot_right", self._qdot_right)
        _expand_joint("torque_left", self._torque_left)
        _expand_joint("torque_right", self._torque_right)
        _expand_joint("torque_clipped_left", self._torque_clipped_left, dtype=bool)
        _expand_joint("torque_clipped_right", self._torque_clipped_right, dtype=bool)

        cols["vel_clipped_left"] = np.array(self._vel_clipped_left, dtype=bool)
        cols["vel_clipped_right"] = np.array(self._vel_clipped_right, dtype=bool)

        return pd.DataFrame(cols)

    # ------------------------------------------------------------------
    def _phase_transitions(self, df: pd.DataFrame) -> list[tuple[float, str]]:
        """Return [(t, phase_name), ...] at each index where phase changes
        (including the very first row)."""
        transitions = []
        prev = None
        for t, phase in zip(df["t"], df["phase"]):
            if phase != prev:
                transitions.append((t, phase))
                prev = phase
        return transitions

    def _draw_phase_lines(self, ax, transitions: list[tuple[float, str]]) -> None:
        for t, phase in transitions:
            ax.axvline(t, color="gray", linestyle="--", linewidth=0.8, alpha=0.7)
        ylim = ax.get_ylim()
        for t, phase in transitions:
            ax.text(
                t,
                ylim[1],
                phase,
                rotation=90,
                fontsize=7,
                va="top",
                ha="right",
                color="gray",
            )

    # ------------------------------------------------------------------
    def plot_summary(
        self,
        save_path: Optional[str] = None,
        p_rel: Optional[np.ndarray] = None,
        landing_point: Optional[np.ndarray] = None,
        v_rel_scalar: Optional[float] = None,
    ):
        """Render the 5-panel summary figure. Returns the matplotlib Figure.

        Panels:
          1. object position (x/y/z) vs time, with p_rel and LANDING_POINT
             marked as horizontal reference lines (color-matched per axis).
          2. object speed (||linvel||) vs time, with v_rel_scalar marked
             as a horizontal dashed line.
          3. sensed vs. target grasp force magnitude, per side.
          4. per-arm worst-case |qdot| across the 7 joints vs time, against
             the two distinct MAX_JOINT_VELOCITIES ceilings (1.2 rad/s for
             joints 1-4, 1.5 rad/s for joints 5-7), with clipped timesteps
             highlighted.
          5. per-arm worst-case |torque| across the 7 joints vs time,
             against the two distinct ctrlrange magnitudes (87 Nm for
             joints 1-4, 12 Nm for joints 5-7), with saturated timesteps
             highlighted.

        Panels 4/5 summarize across all 7 joints per arm (max |value|)
        rather than drawing 14 individual traces + 14 individual limit
        lines, which becomes unreadable in one panel; the underlying
        per-joint columns are all still available via `.to_dataframe()`
        if a per-joint breakdown is needed.
        """
        df = self.to_dataframe()
        landing_point = config.LANDING_POINT if landing_point is None else landing_point
        transitions = self._phase_transitions(df)

        fig, axes = plt.subplots(5, 1, figsize=(12, 22), sharex=True)

        # --- Panel 1: object position ---
        ax = axes[0]
        colors = {"x": "tab:blue", "y": "tab:orange", "z": "tab:green"}
        for label in _XYZ_LABELS:
            ax.plot(df["t"].to_numpy(), df[f"object_pos_{label}"].to_numpy(), label=f"pos_{label}", color=colors[label])
        if p_rel is not None:
            for i, label in enumerate(_XYZ_LABELS):
                ax.axhline(
                    p_rel[i], color=colors[label], linestyle="--", alpha=0.6,
                    label=f"p_rel_{label}" if i == 0 else None,
                )
        for i, label in enumerate(_XYZ_LABELS):
            ax.axhline(
                landing_point[i], color=colors[label], linestyle=":", alpha=0.6,
                label="LANDING_POINT" if i == 0 else None,
            )
        ax.set_ylabel("object position (m)")
        ax.set_title("Object position vs. time")
        ax.legend(loc="upper right", fontsize=7, ncol=3)

        # --- Panel 2: object speed ---
        ax = axes[1]
        linvel_cols = [f"object_linvel_{l}" for l in _XYZ_LABELS]
        speed = np.linalg.norm(df[linvel_cols].to_numpy(), axis=1)
        ax.plot(df["t"].to_numpy(), speed, color="tab:purple", label="|object linvel|")
        if v_rel_scalar is not None:
            ax.axhline(v_rel_scalar, color="tab:red", linestyle="--", label="v_rel_scalar")
        ax.set_ylabel("speed (m/s)")
        ax.set_title("Object speed vs. time")
        ax.legend(loc="upper right", fontsize=7)

        # --- Panel 3: sensed vs. target grasp force ---
        ax = axes[2]
        f_meas_left_mag = np.linalg.norm(
            df[[f"F_meas_left_{l}" for l in _WRENCH_LABELS[:3]]].to_numpy(), axis=1
        )
        f_meas_right_mag = np.linalg.norm(
            df[[f"F_meas_right_{l}" for l in _WRENCH_LABELS[:3]]].to_numpy(), axis=1
        )
        f_star_left_mag = np.linalg.norm(
            df[[f"F_star_left_{l}" for l in _WRENCH_LABELS[:3]]].to_numpy(), axis=1
        )
        f_star_right_mag = np.linalg.norm(
            df[[f"F_star_right_{l}" for l in _WRENCH_LABELS[:3]]].to_numpy(), axis=1
        )
        ax.plot(df["t"].to_numpy(), f_meas_left_mag, color="tab:blue", label="|F_meas| left")
        ax.plot(df["t"].to_numpy(), f_star_left_mag, color="tab:blue", linestyle="--", label="|F_star| left")
        ax.plot(df["t"].to_numpy(), f_meas_right_mag, color="tab:orange", label="|F_meas| right")
        ax.plot(
            df["t"].to_numpy(), f_star_right_mag, color="tab:orange", linestyle="--", label="|F_star| right"
        )
        ax.set_ylabel("grasp force (N)")
        ax.set_title("Sensed vs. target grasp force")
        ax.legend(loc="upper right", fontsize=7, ncol=2)

        # --- Panel 4: joint velocities vs. MAX_JOINT_VELOCITIES ---
        ax = axes[3]
        qdot_left_cols = [f"qdot_left_{j}" for j in _JOINT_LABELS]
        qdot_right_cols = [f"qdot_right_{j}" for j in _JOINT_LABELS]
        max_abs_qdot_left = np.max(np.abs(df[qdot_left_cols].to_numpy()), axis=1)
        max_abs_qdot_right = np.max(np.abs(df[qdot_right_cols].to_numpy()), axis=1)
        ax.plot(df["t"].to_numpy(), max_abs_qdot_left, color="tab:blue", label="max|qdot| left")
        ax.plot(df["t"].to_numpy(), max_abs_qdot_right, color="tab:orange", label="max|qdot| right")
        # Two distinct limit tiers present in MAX_JOINT_VELOCITIES (joints 1-4 vs 5-7).
        for limit in sorted(set(config.MAX_JOINT_VELOCITIES.tolist())):
            ax.axhline(limit, color="black", linestyle=":", alpha=0.5)
        clipped_left_t = df.loc[df["vel_clipped_left"], "t"]
        clipped_right_t = df.loc[df["vel_clipped_right"], "t"]
        ax.scatter(
            clipped_left_t,
            max_abs_qdot_left[df["vel_clipped_left"].to_numpy()],
            color="tab:blue",
            marker="x",
            s=30,
            label="clipped (left)",
        )
        ax.scatter(
            clipped_right_t,
            max_abs_qdot_right[df["vel_clipped_right"].to_numpy()],
            color="tab:orange",
            marker="x",
            s=30,
            label="clipped (right)",
        )
        ax.set_ylabel("joint vel (rad/s)")
        ax.set_title("Max |joint velocity| per arm vs. MAX_JOINT_VELOCITIES")
        ax.legend(loc="upper right", fontsize=7, ncol=2)

        # --- Panel 5: torque vs. ctrlrange ---
        ax = axes[4]
        torque_left_cols = [f"torque_left_{j}" for j in _JOINT_LABELS]
        torque_right_cols = [f"torque_right_{j}" for j in _JOINT_LABELS]
        max_abs_torque_left = np.max(np.abs(df[torque_left_cols].to_numpy()), axis=1)
        max_abs_torque_right = np.max(np.abs(df[torque_right_cols].to_numpy()), axis=1)
        ax.plot(df["t"].to_numpy(), max_abs_torque_left, color="tab:blue", label="max|torque| left")
        ax.plot(df["t"].to_numpy(), max_abs_torque_right, color="tab:orange", label="max|torque| right")
        # Two distinct ctrlrange magnitudes: 87 Nm (joints 1-4), 12 Nm (joints 5-7).
        ctrl_mags = sorted({abs(lo) for lo, hi in config.LEFT_CTRLRANGE} |
                            {abs(hi) for lo, hi in config.LEFT_CTRLRANGE})
        for mag in ctrl_mags:
            ax.axhline(mag, color="black", linestyle=":", alpha=0.5)
        torque_clipped_left_cols = [f"torque_clipped_left_{j}" for j in _JOINT_LABELS]
        torque_clipped_right_cols = [f"torque_clipped_right_{j}" for j in _JOINT_LABELS]
        any_clipped_left = df[torque_clipped_left_cols].to_numpy().any(axis=1)
        any_clipped_right = df[torque_clipped_right_cols].to_numpy().any(axis=1)
        ax.scatter(
            df.loc[any_clipped_left, "t"],
            max_abs_torque_left[any_clipped_left],
            color="tab:blue",
            marker="x",
            s=30,
            label="saturated (left)",
        )
        ax.scatter(
            df.loc[any_clipped_right, "t"],
            max_abs_torque_right[any_clipped_right],
            color="tab:orange",
            marker="x",
            s=30,
            label="saturated (right)",
        )
        ax.set_xlabel("time (s)")
        ax.set_ylabel("torque (Nm)")
        ax.set_title("Max |torque| per arm vs. ctrlrange bounds")
        ax.legend(loc="upper right", fontsize=7, ncol=2)

        for ax in axes:
            self._draw_phase_lines(ax, transitions)

        fig.tight_layout()

        if save_path is not None:
            fig.savefig(save_path, dpi=150)

        return fig
