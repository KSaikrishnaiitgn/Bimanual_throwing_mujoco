"""
mj_interface.py

A typed, safe accessor layer around a MuJoCo (model, data) pair for the
Dual-FR3 Dynamic Throwing task. All name resolution happens once in the
constructor (cached ids/addresses); every method below is a pure read (or,
for `set_ctrl`, a clipped write) against `self.data`.

No control logic lives here -- this class only knows how to get/set state.
"""

from typing import Tuple

import numpy as np
import mujoco

import config.throwing_config as config


class MjInterface:
    """
    Accessor layer wrapping a loaded MuJoCo model/data pair.

    Call `mujoco.mj_forward(model, data)` (or `mj_step`) before reading any
    kinematic/dynamic quantities (poses, Jacobians, sensor data) so that
    `data` is up to date for the current `qpos`/`qvel`.
    """

    def __init__(self, model: mujoco.MjModel, data: mujoco.MjData):
        self.model = model
        self.data = data

        # ---- Joint qpos/qvel addresses for the 14 arm joints ----
        self._left_qpos_adr = np.array(
            [self._joint_qpos_adr(name) for name in config.LEFT_JOINT_NAMES]
        )
        self._left_qvel_adr = np.array(
            [self._joint_qvel_adr(name) for name in config.LEFT_JOINT_NAMES]
        )
        self._right_qpos_adr = np.array(
            [self._joint_qpos_adr(name) for name in config.RIGHT_JOINT_NAMES]
        )
        self._right_qvel_adr = np.array(
            [self._joint_qvel_adr(name) for name in config.RIGHT_JOINT_NAMES]
        )

        # ---- Actuator ids for the 14 actuators ----
        self._left_actuator_ids = np.array(
            [self._id(mujoco.mjtObj.mjOBJ_ACTUATOR, name)
             for name in config.LEFT_ACTUATOR_NAMES]
        )
        self._right_actuator_ids = np.array(
            [self._id(mujoco.mjtObj.mjOBJ_ACTUATOR, name)
             for name in config.RIGHT_ACTUATOR_NAMES]
        )

        # ---- Site ids ----
        self._site_ids = {
            config.LEFT_EE_SITE: self._id(mujoco.mjtObj.mjOBJ_SITE, config.LEFT_EE_SITE),
            config.RIGHT_EE_SITE: self._id(mujoco.mjtObj.mjOBJ_SITE, config.RIGHT_EE_SITE),
            config.LEFT_GRASP_SITE: self._id(mujoco.mjtObj.mjOBJ_SITE, config.LEFT_GRASP_SITE),
            config.RIGHT_GRASP_SITE: self._id(mujoco.mjtObj.mjOBJ_SITE, config.RIGHT_GRASP_SITE),
        }

        # ---- Sensor ids (store adr + dim so get_wrench can slice sensordata) ----
        self._sensor_info = {}
        for sensor_name in (
            config.LEFT_FORCE_SENSOR, config.LEFT_TORQUE_SENSOR,
            config.RIGHT_FORCE_SENSOR, config.RIGHT_TORQUE_SENSOR,
        ):
            sid = self._id(mujoco.mjtObj.mjOBJ_SENSOR, sensor_name)
            adr = self.model.sensor_adr[sid]
            dim = self.model.sensor_dim[sid]
            self._sensor_info[sensor_name] = (adr, dim)

        # ---- Box body id + free-joint qpos/qvel addresses ----
        self._box_body_id = self._id(mujoco.mjtObj.mjOBJ_BODY, config.BOX_BODY_NAME)
        box_qpos_adr = self._joint_qpos_adr(config.BOX_FREE_JOINT_NAME)
        box_qvel_adr = self._joint_qvel_adr(config.BOX_FREE_JOINT_NAME)
        # Free joint: 7 qpos (3 pos + 4 quat wxyz), 6 qvel (3 lin + 3 ang)
        self._box_qpos_slice = slice(box_qpos_adr, box_qpos_adr + 7)
        self._box_qvel_slice = slice(box_qvel_adr, box_qvel_adr + 6)

    # ------------------------------------------------------------------
    # Internal name-resolution helpers
    # ------------------------------------------------------------------

    def _id(self, obj_type: "mujoco.mjtObj", name: str) -> int:
        """Resolve a MuJoCo object id by type and name; raises if not found."""
        obj_id = mujoco.mj_name2id(self.model, obj_type, name)
        if obj_id < 0:
            raise ValueError(f"MuJoCo object '{name}' of type {obj_type} not found in model.")
        return obj_id

    def _joint_qpos_adr(self, joint_name: str) -> int:
        jid = self._id(mujoco.mjtObj.mjOBJ_JOINT, joint_name)
        return int(self.model.jnt_qposadr[jid])

    def _joint_qvel_adr(self, joint_name: str) -> int:
        jid = self._id(mujoco.mjtObj.mjOBJ_JOINT, joint_name)
        return int(self.model.jnt_dofadr[jid])

    def _arm_qpos_adr(self, arm: str) -> np.ndarray:
        if arm == "left":
            return self._left_qpos_adr
        elif arm == "right":
            return self._right_qpos_adr
        raise ValueError(f"arm must be 'left' or 'right', got '{arm}'")

    def _arm_qvel_adr(self, arm: str) -> np.ndarray:
        if arm == "left":
            return self._left_qvel_adr
        elif arm == "right":
            return self._right_qvel_adr
        raise ValueError(f"arm must be 'left' or 'right', got '{arm}'")

    def _arm_actuator_ids(self, arm: str) -> np.ndarray:
        if arm == "left":
            return self._left_actuator_ids
        elif arm == "right":
            return self._right_actuator_ids
        raise ValueError(f"arm must be 'left' or 'right', got '{arm}'")

    def _arm_ctrlrange(self, arm: str):
        if arm == "left":
            return config.LEFT_CTRLRANGE
        elif arm == "right":
            return config.RIGHT_CTRLRANGE
        raise ValueError(f"arm must be 'left' or 'right', got '{arm}'")

    # ------------------------------------------------------------------
    # Public accessors
    # ------------------------------------------------------------------

    def get_qpos(self, arm: str) -> np.ndarray:
        """
        Joint angles for the given arm, in the order of
        LEFT_JOINT_NAMES / RIGHT_JOINT_NAMES.

        Args:
            arm: 'left' or 'right'.

        Returns:
            np.ndarray, shape (7,).
        """
        adr = self._arm_qpos_adr(arm)
        return np.array(self.data.qpos[adr], dtype=np.float64)

    def get_qvel(self, arm: str) -> np.ndarray:
        """
        Joint velocities for the given arm, in the order of
        LEFT_JOINT_NAMES / RIGHT_JOINT_NAMES.

        Args:
            arm: 'left' or 'right'.

        Returns:
            np.ndarray, shape (7,).
        """
        adr = self._arm_qvel_adr(arm)
        return np.array(self.data.qvel[adr], dtype=np.float64)

    def get_site_pose(self, site_name: str) -> Tuple[np.ndarray, np.ndarray]:
        """
        World-frame position and orientation of a site.

        Args:
            site_name: one of LEFT_EE_SITE, RIGHT_EE_SITE, LEFT_GRASP_SITE,
                RIGHT_GRASP_SITE.

        Returns:
            (pos, quat): pos is shape (3,), quat is shape (4,) in wxyz order,
            derived from data.site_xmat via mujoco.mju_mat2Quat.
        """
        site_id = self._site_ids[site_name]
        pos = np.array(self.data.site_xpos[site_id], dtype=np.float64)
        mat = np.array(self.data.site_xmat[site_id], dtype=np.float64).reshape(9)
        quat = np.zeros(4, dtype=np.float64)
        mujoco.mju_mat2Quat(quat, mat)
        return pos, quat

    def get_site_jacobian(self, site_name: str, arm: str) -> np.ndarray:
        """
        6x7 Cartesian Jacobian of a site, restricted to one arm's 7 joints.

        Computes the full 6 x nv Jacobian via mujoco.mj_jacSite (3 rows
        linear from jacp, 3 rows angular from jacr, stacked as [jacp; jacr]),
        then slices the 7 columns corresponding to the given arm's cached
        qvel (dof) addresses.

        Args:
            site_name: one of LEFT_EE_SITE, RIGHT_EE_SITE, LEFT_GRASP_SITE,
                RIGHT_GRASP_SITE.
            arm: 'left' or 'right' -- which arm's 7 columns to return.

        Returns:
            np.ndarray, shape (6, 7).
        """
        site_id = self._site_ids[site_name]
        jacp = np.zeros((3, self.model.nv), dtype=np.float64)
        jacr = np.zeros((3, self.model.nv), dtype=np.float64)
        mujoco.mj_jacSite(self.model, self.data, jacp, jacr, site_id)
        jac_full = np.vstack([jacp, jacr])  # (6, nv)
        dof_adr = self._arm_qvel_adr(arm)
        return jac_full[:, dof_adr]

    def get_wrench(self, force_sensor: str, torque_sensor: str) -> np.ndarray:
        """
        Concatenated force/torque reading from a pair of sensors.

        Args:
            force_sensor: name of the 3D force sensor.
            torque_sensor: name of the 3D torque sensor.

        Returns:
            np.ndarray, shape (6,): [Fx, Fy, Fz, Tx, Ty, Tz].
        """
        f_adr, f_dim = self._sensor_info[force_sensor]
        t_adr, t_dim = self._sensor_info[torque_sensor]
        force = np.array(self.data.sensordata[f_adr:f_adr + f_dim], dtype=np.float64)
        torque = np.array(self.data.sensordata[t_adr:t_adr + t_dim], dtype=np.float64)
        return np.concatenate([force, torque])
    
    def get_site_rotation_matrix(self, site_name: str) -> np.ndarray:
        """
        World-frame 3x3 rotation matrix of a site (columns = the site's
        local x/y/z axes expressed in world frame). More direct than
        get_site_pose for this purpose since it skips the quaternion
        round-trip.

        Args:
            site_name: one of LEFT_EE_SITE, RIGHT_EE_SITE, LEFT_GRASP_SITE,
                RIGHT_GRASP_SITE.

        Returns:
            np.ndarray, shape (3, 3).
        """
        site_id = self._site_ids[site_name]
        return np.array(self.data.site_xmat[site_id], dtype=np.float64).reshape(3, 3)
    
    def get_wrench_world(self, arm: str, force_sensor: str, torque_sensor: str) -> np.ndarray:
        """
        Concatenated force/torque reading, rotated from the sensor site's
        local frame (which moves with the arm's joint configuration) into
        world frame.

        MuJoCo force/torque sensors report in the sensor site's own local
        frame -- for a sensor mounted deep in a 7-DOF chain, that frame
        rotates with every joint move, so it is NOT fixed relative to
        world or to the box even at a nominal grasp pose. Any caller
        comparing F_meas against a world-frame target wrench (e.g.
        config.desired_wrench, which is defined along
        LEFT_PAD_APPROACH_AXIS_WORLD / RIGHT_PAD_APPROACH_AXIS_WORLD) must
        use this method, not get_wrench directly.

        Args:
            arm: 'left' or 'right' -- selects which EE site's rotation to
                use (LEFT_EE_SITE / RIGHT_EE_SITE).
            force_sensor: name of the 3D force sensor.
            torque_sensor: name of the 3D torque sensor.

        Returns:
            np.ndarray, shape (6,): [Fx, Fy, Fz, Tx, Ty, Tz], world frame.
        """

        site_name = config.LEFT_EE_SITE if arm == "left" else config.RIGHT_EE_SITE
        F_local = self.get_wrench(force_sensor, torque_sensor)
        R = self.get_site_rotation_matrix(site_name)
        force_world = R @ F_local[0:3]
        torque_world = R @ F_local[3:6]
        return np.concatenate([force_world, torque_world])
    
    def debug_ee_axes(self) -> None:
        """
        Print each EE site's world-frame rotation matrix and local axes,
        plus the configured approach axes, for sanity-checking grasp
        geometry at startup. Read-only diagnostic; no state is modified.
        """
        print("\n========== EE AXIS DEBUG ==========")
        for side, site in [
            ("LEFT", config.LEFT_EE_SITE),
            ("RIGHT", config.RIGHT_EE_SITE),
        ]:
            R = self.get_site_rotation_matrix(site)
            print(f"{side} EE R:")
            print(R)
            print(f"{side} local +X in world:")
            print(R[:, 0])
            print(f"{side} local +Y in world:")
            print(R[:, 1])
            print(f"{side} local +Z in world:")
            print(R[:, 2])

        print("\nConfigured approach axes:")
        print("LEFT :", config.LEFT_PAD_APPROACH_AXIS_WORLD)
        print("RIGHT:", config.RIGHT_PAD_APPROACH_AXIS_WORLD)
        print("===================================\n")

    def get_bias_torque(self, arm: str) -> np.ndarray:
        """
        Gravity + Coriolis + centrifugal torque for the given arm's 7 joints,
        i.e. the torque a computed-torque controller must cancel to make the
        remaining PD/PID error dynamics behave like a simple double-integrator.

        In sim this reads MuJoCo's data.qfrc_bias (valid post mj_forward /
        mj_step). On hardware, this call site is the one place to swap for
        the real equivalent -- e.g. libfranka's franka::Model::gravity(state)
        + franka::Model::coriolis(state), or the robot's internal
        gravity-comp torque if running in torque-control mode with comp
        enabled. No caller of this method (reaching_pd, state_machine.py)
        needs to change when that swap happens.

        Args:
            arm: 'left' or 'right'.

        Returns:
            np.ndarray, shape (7,), Nm.
        """
        adr = self._arm_qvel_adr(arm)
        return np.array(self.data.qfrc_bias[adr], dtype=np.float64)

    def get_object_state(self) -> dict:
        """
        Box state read directly from its free-joint qpos/qvel slice.

        Returns:
            dict with keys:
                'pos'    -- np.ndarray (3,), world position.
                'quat'   -- np.ndarray (4,), world orientation, wxyz.
                'linvel' -- np.ndarray (3,), linear velocity.
                'angvel' -- np.ndarray (3,), angular velocity.
        """
        qpos = np.array(self.data.qpos[self._box_qpos_slice], dtype=np.float64)
        qvel = np.array(self.data.qvel[self._box_qvel_slice], dtype=np.float64)
        return {
            "pos": qpos[0:3],
            "quat": qpos[3:7],
            "linvel": qvel[0:3],
            "angvel": qvel[3:6],
        }

    def set_ctrl(self, arm: str, torques: np.ndarray) -> np.ndarray:
        """
        Write clipped joint torques into data.ctrl for the given arm.

        Each of the 7 torque values is clipped to that joint's entry in
        LEFT_CTRLRANGE / RIGHT_CTRLRANGE before being written.

        Args:
            arm: 'left' or 'right'.
            torques: np.ndarray, shape (7,), commanded torques in the order
                of LEFT_JOINT_NAMES / RIGHT_JOINT_NAMES.

        Returns:
            np.ndarray of bool, shape (7,): True where the corresponding
            joint's commanded torque was clipped, for logging.
        """
        torques = np.asarray(torques, dtype=np.float64)
        if torques.shape != (7,):
            raise ValueError(f"torques must have shape (7,), got {torques.shape}")

        actuator_ids = self._arm_actuator_ids(arm)
        ctrlrange = self._arm_ctrlrange(arm)

        clipped = np.zeros(7, dtype=bool)
        clipped_torques = np.empty(7, dtype=np.float64)
        for i in range(7):
            lo, hi = ctrlrange[i]
            val = torques[i]
            clamped = min(max(val, lo), hi)
            clipped[i] = clamped != val
            clipped_torques[i] = clamped

        self.data.ctrl[actuator_ids] = clipped_torques
        return clipped