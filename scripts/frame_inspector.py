"""
frame_inspector.py

STANDALONE manual-verification script. Not imported by any other module.

Purpose: before trusting any downstream controller, visually/numerically
confirm that:
  1. The EE / pad sites are where we think they are, and their local-z axis
     (pad-normal) matches the configured approach axis once a grasp
     configuration is loaded.
  2. The grasp sites and the box body are where config.py claims they are.
  3. The force/torque sensors report force along the approach axis with a
     KNOWN sign, so config.desired_wrench's TODO_VERIFY_SIGN placeholder can
     be filled in with confidence instead of guessed.

Usage:
    python frame_inspector.py
    python frame_inspector.py --qpos-file my_grasp_qpos.txt

--qpos-file, if given, must contain a flat list of 14 numbers (whitespace or
newline separated, readable by np.loadtxt) -- the 7 left-arm joint angles
followed by the 7 right-arm joint angles (e.g. an IK solution produced by
ik_solver.py). These 14 values override the arm portion of the `home`
keyframe; everything else (notably the box's free joint) is left as defined
in the `home` keyframe.
"""

import argparse
import sys

import numpy as np
import mujoco

import config.throwing_config as config
from mj_interface import MjInterface


def _quat_local_z_in_world(quat_wxyz: np.ndarray) -> np.ndarray:
    """
    Given a world-frame orientation quaternion (wxyz), return the site's
    local +z axis expressed in the world frame -- i.e. the 3rd column of
    the 3x3 rotation matrix built from that quaternion.
    """
    mat = np.zeros(9, dtype=np.float64)
    mujoco.mju_quat2Mat(mat, quat_wxyz)
    mat = mat.reshape(3, 3)
    return mat[:, 2]


def _print_header(title: str) -> None:
    print()
    print("=" * 70)
    print(title)
    print("=" * 70)


def _fmt(vec: np.ndarray) -> str:
    return np.array2string(np.asarray(vec, dtype=np.float64), precision=4, suppress_small=True)


def load_model_and_data(qpos_file: str):
    """Build model/data, apply the `home` keyframe, optionally override the
    14 arm joint values from --qpos-file, then run mj_forward."""
    model = mujoco.MjModel.from_xml_path(config.XML_PATH)
    data = mujoco.MjData(model)

    home_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_KEY, "home")
    if home_id < 0:
        raise RuntimeError(
            f"No keyframe named 'home' found in {config.XML_PATH}; "
            "cannot establish a baseline qpos."
        )
    data.qpos[:] = model.key_qpos[home_id]
    if model.key_qvel is not None and model.key_qvel.shape[0] > home_id:
        data.qvel[:] = model.key_qvel[home_id]

    mujoco.mj_forward(model, data)

    iface = MjInterface(model, data)

    if qpos_file is not None:
        arm_qpos = np.loadtxt(qpos_file).reshape(-1)
        if arm_qpos.shape[0] != 14:
            raise ValueError(
                f"--qpos-file must contain exactly 14 values (7 left + 7 right), "
                f"got {arm_qpos.shape[0]}"
            )
        left_qpos, right_qpos = arm_qpos[:7], arm_qpos[7:]
        # Use the interface's cached joint qpos addresses so this stays
        # correct regardless of qpos layout elsewhere in the model.
        data.qpos[iface._left_qpos_adr] = left_qpos
        data.qpos[iface._right_qpos_adr] = right_qpos
        mujoco.mj_forward(model, data)

    return model, data, iface


def report_ee_frames(iface: MjInterface) -> None:
    _print_header("EE / PAD FRAME CHECK")
    for side, site_name, expected_axis in (
        ("left", config.LEFT_EE_SITE, config.LEFT_PAD_APPROACH_AXIS_WORLD),
        ("right", config.RIGHT_EE_SITE, config.RIGHT_PAD_APPROACH_AXIS_WORLD),
    ):
        pos, quat = iface.get_site_pose(site_name)
        local_z_world = _quat_local_z_in_world(quat)
        alignment = float(np.dot(local_z_world, expected_axis))
        print(f"[{side}] site '{site_name}'")
        print(f"    world pos          : {_fmt(pos)}")
        print(f"    world quat (wxyz)  : {_fmt(quat)}")
        print(f"    local +z in world  : {_fmt(local_z_world)}")
        print(f"    expected approach  : {_fmt(expected_axis)}")
        print(f"    dot(local_z, exp.) : {alignment:.4f}  "
              f"({'ALIGNED' if alignment > 0.95 else 'CHECK ALIGNMENT'})")


def report_grasp_geometry(iface: MjInterface) -> None:
    _print_header("BOX / GRASP-SITE GEOMETRY CHECK")

    obj_state = iface.get_object_state()
    print("box body")
    print(f"    world pos  : {_fmt(obj_state['pos'])}")
    print(f"    world quat : {_fmt(obj_state['quat'])}  (wxyz)")

    for side, site_name, expected_pos in (
        ("left", config.LEFT_GRASP_SITE, config.LEFT_GRASP_WORLD_POS),
        ("right", config.RIGHT_GRASP_SITE, config.RIGHT_GRASP_WORLD_POS),
    ):
        pos, _ = iface.get_site_pose(site_name)
        err = np.linalg.norm(pos - expected_pos)
        print(f"[{side}] site '{site_name}'")
        print(f"    world pos (actual)  : {_fmt(pos)}")
        print(f"    world pos (config)  : {_fmt(expected_pos)}")
        print(f"    position error      : {err:.5f} m  "
              f"({'OK' if err < 1e-3 else 'MISMATCH -- check config/XML'})")


def report_sensor_sign(model, data, iface: MjInterface, settle_steps: int = 200) -> None:
    _print_header("SENSOR-SIGN TEST (settle then read F/T sensors)")
    print(f"Stepping simulation {settle_steps} steps (dt={config.SIM_TIMESTEP}s) "
          "to let contact settle with gravity on...")
    for _ in range(settle_steps):
        mujoco.mj_step(model, data)

    for side, force_sensor, torque_sensor, approach_axis in (
        ("left", config.LEFT_FORCE_SENSOR, config.LEFT_TORQUE_SENSOR,
         config.LEFT_PAD_APPROACH_AXIS_WORLD),
        ("right", config.RIGHT_FORCE_SENSOR, config.RIGHT_TORQUE_SENSOR,
         config.RIGHT_PAD_APPROACH_AXIS_WORLD),
    ):
        wrench = iface.get_wrench(force_sensor, torque_sensor)
        force = wrench[0:3]
        torque = wrench[3:6]
        along_axis = float(np.dot(force, approach_axis))
        sign_str = "+" if along_axis >= 0 else "-"
        print(f"[{side}] sensors '{force_sensor}' / '{torque_sensor}'")
        print(f"    raw force  (sensor frame) : {_fmt(force)}")
        print(f"    raw torque (sensor frame) : {_fmt(torque)}")
        print(f"    approach axis (world)     : {_fmt(approach_axis)}")
        print(f"    force . approach_axis     : {along_axis:.4f} N  "
              f"(sign = '{sign_str}')")
        print(f"    ==> config.desired_wrench('{side}') should use "
              f"Fx = {sign_str}GRASP_FORCE_MAG for this side.")

    print()
    print("Manually transfer the printed signs above into config.py's")
    print("desired_wrench() function, replacing the '# TODO_VERIFY_SIGN' logic.")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Standalone frame/grasp/sensor-sign verification tool. "
                     "Console output only; nothing else imports this script."
    )
    parser.add_argument(
        "--qpos-file", type=str, default=None,
        help="Path to a text file with 14 whitespace/newline-separated "
             "numbers (7 left-arm + 7 right-arm joint angles), e.g. an IK "
             "solution from ik_solver.py. If omitted, the model's `home` "
             "keyframe is used as-is.",
    )
    args = parser.parse_args()

    model, data, iface = load_model_and_data(args.qpos_file)

    report_ee_frames(iface)
    report_grasp_geometry(iface)
    report_sensor_sign(model, data, iface)

    print()
    print("=" * 70)
    print("Inspection complete. Review the blocks above before trusting any "
          "downstream controller.")
    print("=" * 70)


if __name__ == "__main__":
    main()
