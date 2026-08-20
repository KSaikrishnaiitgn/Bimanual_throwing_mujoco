"""
main_throw_sim.py

Entry point for the dual-FR3 dynamic throwing simulation. Wires together
MjInterface, ThrowStateMachine, and SimLogger, and runs the main physics
loop either headless (fast, no rendering) or with a live passive viewer.

Usage:
    python main_throw_sim.py [--headless] [--duration 8.0]

Depends on: config.py, mj_interface.py, state_machine.py, logger.py,
mujoco, numpy, argparse. Requires pregrasp_left.npy / pregrasp_right.npy /
grasp_left.npy / grasp_right.npy to already exist in the working
directory (produced by `python ik_solver.py`) -- this script does not
solve IK inline.
"""

from __future__ import annotations

import argparse
import contextlib
import os
import sys

import mujoco
import mujoco.viewer
import numpy as np

import config.throwing_config as config
from mj_interface import MjInterface
from state_machine import Phase, ThrowStateMachine
from logger import SimLogger

REQUIRED_IK_FILES = [
    "pregrasp_left.npy",
    "pregrasp_right.npy",
    "grasp_left.npy",
    "grasp_right.npy",
]

LOG_EVERY_N_STEPS = 10
HOME_KEYFRAME_NAME = "home"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the dual-FR3 dynamic throw simulation.")
    parser.add_argument(
        "--headless",
        action="store_true",
        help="Run physics only, no viewer (faster).",
    )
    parser.add_argument(
        "--duration",
        type=float,
        default=8.0,
        help="Simulation duration in seconds (default: 8.0).",
    )
    return parser.parse_args()


def check_ik_cache_present() -> None:
    missing = [f for f in REQUIRED_IK_FILES if not os.path.isfile(f)]
    if missing:
        print("ERROR: missing IK cache file(s): " + ", ".join(missing))
        print("Run `python ik_solver.py` first to generate pregrasp/grasp joint "
              "configurations before running this script.")
        sys.exit(1)


def main() -> None:
    args = parse_args()

    check_ik_cache_present()

    model = mujoco.MjModel.from_xml_path(config.XML_PATH)
    data = mujoco.MjData(model)

    interface = MjInterface(model, data)

    home_key_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_KEY, HOME_KEYFRAME_NAME)
    if home_key_id == -1:
        print(f"ERROR: keyframe '{HOME_KEYFRAME_NAME}' not found in {config.XML_PATH}.")
        sys.exit(1)
    data.qpos[:] = model.key_qpos[home_key_id]
    mujoco.mj_forward(model, data)

    # ThrowStateMachine.__init__ calls interface.debug_ee_axes(), which reads
    # data.site_xmat -- must run AFTER mj_forward above, or it reads a
    # freshly-constructed MjData's zeroed-out site transforms.
    state_machine = ThrowStateMachine(interface)
    logger = SimLogger()
    n_steps = int(round(args.duration / config.SIM_TIMESTEP))
    prev_phase = None

    viewer_cm = (
        contextlib.nullcontext(None)
        if args.headless
        else mujoco.viewer.launch_passive(model, data)
    )

    with viewer_cm as viewer:
        for step_count in range(n_steps):
            t = step_count * config.SIM_TIMESTEP

            torque_left, torque_right = state_machine.step(t, config.SIM_TIMESTEP)
            clip_flags_left = interface.set_ctrl("left", torque_left)
            clip_flags_right = interface.set_ctrl("right", torque_right)
            if state_machine.phase == Phase.SQUEEZE_GRASP and step_count % 250 == 0:
                print(
                    f"[SQUEEZE TORQUE t={t:.3f}] "
                    f"L torque={np.round(torque_left, 3)} | "
                    f"L clip={clip_flags_left} | "
                    f"R torque={np.round(torque_right, 3)} | "
                    f"R clip={clip_flags_right}"
                )
            mujoco.mj_step(model, data)

            if state_machine.phase != prev_phase:
                print(f"[t={t:.3f}s] phase -> {state_machine.phase.value}")
                prev_phase = state_machine.phase

            if step_count % LOG_EVERY_N_STEPS == 0:
                obj_state = interface.get_object_state()
                qvel_left = interface.get_qvel("left")
                qvel_right = interface.get_qvel("right")
                F_meas_left = interface.get_wrench(
                    config.LEFT_FORCE_SENSOR, config.LEFT_TORQUE_SENSOR
                )
                F_meas_right = interface.get_wrench(
                    config.RIGHT_FORCE_SENSOR, config.RIGHT_TORQUE_SENSOR
                )
                F_star_left = state_machine.force_closer.target_wrench("left")
                F_star_right = state_machine.force_closer.target_wrench("right")

                logger.record(
                    t,
                    state_machine.phase,
                    obj_state,
                    F_meas_left,
                    F_meas_right,
                    F_star_left,
                    F_star_right,
                    qvel_left,
                    qvel_right,
                    torque_left,
                    torque_right,
                    state_machine.vel_clipped_left,
                    state_machine.vel_clipped_right,
                    clip_flags_left,
                    clip_flags_right,
                )

            if viewer is not None:
                viewer.sync()
                if not viewer.is_running():
                    break

    # ------------------------------------------------------------------
    # post-run diagnostics
    # ------------------------------------------------------------------
    p_rel = state_machine.p_rel
    v_rel_scalar = (
        float(np.linalg.norm(state_machine.v_rel_vector))
        if state_machine.v_rel_vector is not None
        else None
    )
    logger.plot_summary(
        save_path="throw_summary.png",
        p_rel=p_rel,
        v_rel_scalar=v_rel_scalar,
    )
    print("Saved throw_summary.png")

    final_obj_state = interface.get_object_state()
    final_pos = final_obj_state["pos"]
    landing_error = final_pos - config.LANDING_POINT
    print(f"Final object position: {final_pos}")
    print(f"Landing point target:  {config.LANDING_POINT}")
    print(
        f"Landing error (final - target): {landing_error}, "
        f"norm = {np.linalg.norm(landing_error):.4f} m"
    )

    released = state_machine.release_time is not None
    if state_machine.phase == Phase.THROW:
        print(
            "WARNING: simulation ended still in THROW phase -- release was never "
            "triggered (|x_o - p_rel| never dropped below RELEASE_POS_TOLERANCE). "
            "Consider retuning RELEASE_POS_TOLERANCE, K_DS/B_DS, or the joint-space "
            "gains (KP_GAINS/KD_GAINS/KI_GAINS)."
        )
    elif released:
        print(f"Release triggered at t={state_machine.release_time:.3f}s.")
    else:
        print(
            "Release never triggered and the sim did not end in THROW phase -- "
            "this shouldn't normally happen; check the phase-transition logic."
        )


if __name__ == "__main__":
    main()
