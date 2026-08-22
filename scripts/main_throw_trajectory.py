"""Run the experimental terminal-state throw without altering the baseline."""

from __future__ import annotations

import argparse
import contextlib
import os
import sys

# MuJoCo must choose its OpenGL backend before it is imported.  EGL enables
# off-screen recording on machines without a desktop/display server.
if "--headless" in sys.argv:
    os.environ.setdefault("MUJOCO_GL", "egl")

import mujoco
import mujoco.viewer
import numpy as np
import pandas as pd

import config.throwing_trajectory_config as config
from logger import SimLogger
from mj_interface import MjInterface
from state_machine import Phase
from state_machine_trajectory import TrajectoryThrowStateMachine
from video_recorder import CAMERA_PRESETS, POINT_STYLES, ThrowVideoRecorder, add_throw_markers


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--headless", action="store_true")
    parser.add_argument("--duration", type=float, default=8.0)
    parser.add_argument("--record", action="store_true",
                        help="record annotated videos from multiple camera angles")
    parser.add_argument("--video-dir", default="throw_videos")
    parser.add_argument("--video-fps", type=float, default=30.0)
    parser.add_argument("--video-width", type=int, default=1920)
    parser.add_argument("--video-height", type=int, default=1080)
    parser.add_argument(
        "--cameras", default="front,side,oblique",
        help=f"comma-separated camera presets: {','.join(CAMERA_PRESETS)}",
    )
    args = parser.parse_args()

    model = mujoco.MjModel.from_xml_path(config.XML_PATH)
    data = mujoco.MjData(model)
    home = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_KEY, "home")
    data.qpos[:] = model.key_qpos[home]
    mujoco.mj_forward(model, data)
    interface = MjInterface(model, data)
    machine = TrajectoryThrowStateMachine(interface)
    logger = SimLogger()
    trajectory_diagnostics = []
    previous_phase = None
    actual_landing_position = np.full(3, np.nan)
    recorder = None
    if args.record:
        camera_names = [name.strip() for name in args.cameras.split(",") if name.strip()]
        recorder = ThrowVideoRecorder(
            model, args.video_dir, camera_names, args.video_fps,
            args.video_width, args.video_height,
        )
    print("Point-marker colors:")
    for label, _, rgba, _ in POINT_STYLES:
        print(f"  {label}: RGBA {rgba}")

    viewer_context = (
        contextlib.nullcontext(None) if args.headless
        else mujoco.viewer.launch_passive(model, data)
    )
    with viewer_context as viewer:
        for step in range(int(round(args.duration / config.SIM_TIMESTEP))):
            t = step * config.SIM_TIMESTEP
            torque_left, torque_right = machine.step(t, config.SIM_TIMESTEP)
            clipped_left = interface.set_ctrl("left", torque_left)
            clipped_right = interface.set_ctrl("right", torque_right)
            mujoco.mj_step(model, data)
            if machine.phase == Phase.DONE:
                actual_landing_position = interface.get_object_state()["pos"].copy()
            marker_points = {
                "theoretical_release": config.RELEASE_POINT,
                "actual_release": machine.actual_release_position,
                "theoretical_landing": config.LANDING_POINT,
                "actual_landing": actual_landing_position,
            }
            if machine.phase != previous_phase:
                print(f"[t={t:.3f}s] phase -> {machine.phase.value}")
                previous_phase = machine.phase
            if step % 10 == 0:
                J_left = interface.get_site_jacobian(config.LEFT_EE_SITE, "left")
                J_right = interface.get_site_jacobian(config.RIGHT_EE_SITE, "right")
                ee_actual_left = J_left @ interface.get_qvel("left")
                ee_actual_right = J_right @ interface.get_qvel("right")
                row = {"t": t}
                row.update({f"release_actual_pos_{a}": value for a, value in zip(("x", "y", "z"), machine.actual_release_position)})
                row.update({f"release_actual_vel_{a}": value for a, value in zip(("x", "y", "z"), machine.actual_release_velocity)})
                for prefix, values in (
                    ("x_ref", machine.diag_x_ref),
                    ("v_ref", machine.diag_v_ref),
                    ("v_obj_cmd", machine.diag_v_obj_cmd),
                    ("ee_cmd_left", machine.diag_hand_twist_left),
                    ("ee_cmd_right", machine.diag_hand_twist_right),
                    ("ee_actual_left", ee_actual_left),
                    ("ee_actual_right", ee_actual_right),
                ):
                    labels = ("x", "y", "z") if len(values) == 3 else ("vx", "vy", "vz", "wx", "wy", "wz")
                    row.update({f"{prefix}_{label}": value for label, value in zip(labels, values)})
                trajectory_diagnostics.append(row)
                logger.record(
                    t, machine.phase, interface.get_object_state(),
                    interface.get_wrench_world(
                        "left", config.LEFT_FORCE_SENSOR, config.LEFT_TORQUE_SENSOR
                    ),
                    interface.get_wrench_world(
                        "right", config.RIGHT_FORCE_SENSOR, config.RIGHT_TORQUE_SENSOR
                    ),
                    machine.force_closer.target_wrench("left"),
                    machine.force_closer.target_wrench("right"),
                    interface.get_qvel("left"), interface.get_qvel("right"),
                    torque_left, torque_right,
                    machine.vel_clipped_left, machine.vel_clipped_right,
                    clipped_left, clipped_right,
                )
            if viewer is not None:
                with viewer.lock():
                    viewer.user_scn.ngeom = 0
                    add_throw_markers(viewer.user_scn, marker_points)
                viewer.sync()
                if not viewer.is_running():
                    break

            if recorder is not None:
                recorder.capture(
                    data, t, machine.phase.value, marker_points,
                )

    if recorder is not None:
        recorder.close()
        print("Saved videos:")
        for path in recorder.paths:
            print(f"  {path}")

    df = logger.to_dataframe()
    diag_df = pd.DataFrame(trajectory_diagnostics)
    df = df.merge(diag_df, on="t", how="left")
    df.to_csv("throw_trajectory_log.csv", index=False)
    logger.plot_summary(
        save_path="throw_trajectory_summary.png",
        p_rel=machine.p_rel,
        landing_point=config.LANDING_POINT,
        v_rel_scalar=machine.v_rel_scalar,
    )
    final = interface.get_object_state()
    print("Saved throw_trajectory_log.csv and throw_trajectory_summary.png")
    print(f"Final phase: {machine.phase.value}")
    print(f"Final object position: {final['pos']}")
    print(f"Final object velocity: {final['linvel']}")
    print(f"Landing target: {config.LANDING_POINT}")
    print(f"Landing error norm: {np.linalg.norm(final['pos'] - config.LANDING_POINT):.4f}m")
    if machine.terminal_position_error is not None:
        print(
            f"Last terminal errors: position={machine.terminal_position_error:.4f}m, "
            f"velocity={machine.terminal_velocity_error:.4f}m/s"
        )


if __name__ == "__main__":
    main()
