"""
Contact force/wrench handler for dual-arm manipulation
Computes contact forces and wrenches between end-effectors and objects

NOTE on this version:
  mj_contactForce() returns the force in the CONTACT frame (normal along
  the contact's local x-axis), not the world frame. The original version of
  get_object_contact_wrenches() summed that raw contact-frame force directly
  into a "wrench" as if it were already in world coordinates — for a contact
  whose normal isn't aligned with a world axis this silently gives the wrong
  direction. Fixed here by rotating with contact.frame (a row-major 3x3
  whose rows are the contact frame's basis vectors expressed in world
  coordinates): F_world = R^T @ F_local.
"""
import numpy as np
import mujoco


class ContactHandler:
    """Handles contact detection and wrench computation"""

    def __init__(self, model, data):
        """
        Initialize contact handler

        Args:
            model: MuJoCo model
            data: MuJoCo data
        """
        self.model = model
        self.data = data

    # ──────────────────────────────────────────────────────────────────
    def _contact_force_world(self, contact_id: int):
        """
        Rotate a single contact's force/torque from the contact frame into
        the world frame.

        Returns:
            force_world:  (3,) world-frame force
            torque_world: (3,) world-frame torque
        """
        contact = self.data.contact[contact_id]
        force_local = np.zeros(6)
        mujoco.mj_contactForce(self.model, self.data, contact_id, force_local)

        R = contact.frame.reshape(3, 3)   # rows = contact-frame basis vectors in world coords
        force_world  = R.T @ force_local[:3]
        torque_world = R.T @ force_local[3:]
        return force_world, torque_world

    # ──────────────────────────────────────────────────────────────────
    def get_object_contact_wrenches(self, object_body_id):
        """
        Get world-frame wrenches at contact points between end-effectors and
        object.

        Args:
            object_body_id: Body ID of the object

        Returns:
            left_wrench:  6D wrench (world frame) from left end-effector
            right_wrench: 6D wrench (world frame) from right end-effector
            contact_detected: Boolean indicating if contact exists
        """
        if object_body_id == -1:
            return np.zeros(6), np.zeros(6), False

        object_geom_ids = [i for i in range(self.model.ngeom)
                            if self.model.geom_bodyid[i] == object_body_id]

        left_wrench  = np.zeros(6)
        right_wrench = np.zeros(6)
        contact_detected = False

        for i in range(self.data.ncon):
            contact = self.data.contact[i]

            if contact.geom1 in object_geom_ids or contact.geom2 in object_geom_ids:
                force_world, _ = self._contact_force_world(i)

                # Sign convention: mj_contactForce gives the force acting on
                # geom2 due to geom1 (MuJoCo convention). Flip if the object
                # is geom1 so we consistently get "force ON the object".
                if contact.geom1 in object_geom_ids:
                    force_on_object = -force_world
                else:
                    force_on_object = force_world

                other_geom = contact.geom2 if contact.geom1 in object_geom_ids else contact.geom1
                other_body = self.model.geom_bodyid[other_geom]
                body_name  = self.model.body(other_body).name

                contact_pos = contact.pos.copy()
                object_pos  = self.data.xpos[object_body_id].copy()
                lever_arm   = contact_pos - object_pos
                torque      = np.cross(lever_arm, force_on_object)

                if "left" in body_name.lower():
                    left_wrench[:3]  += force_on_object
                    left_wrench[3:]  += torque
                    contact_detected = True
                elif "right" in body_name.lower():
                    right_wrench[:3] += force_on_object
                    right_wrench[3:] += torque
                    contact_detected = True

        return left_wrench, right_wrench, contact_detected

    # ──────────────────────────────────────────────────────────────────
    def get_ee_contact_force(self, object_body_id) -> tuple:
        """
        Convenience wrapper for GraspForceController: returns just the
        world-frame linear force (3,) each arm applies ON the object,
        without the torque terms.

        Returns:
            F_left:  (3,) world-frame force on object from left arm
            F_right: (3,) world-frame force on object from right arm
        """
        left_wrench, right_wrench, _ = self.get_object_contact_wrenches(object_body_id)
        return left_wrench[:3].copy(), right_wrench[:3].copy()

    # ──────────────────────────────────────────────────────────────────
    def find_end_effector_sites(self):
        """
        Find end-effector site IDs by searching for 'left' and 'right' in names

        Returns:
            left_ee_id: Site ID for left end-effector
            right_ee_id: Site ID for right end-effector
        """
        left_ee_id = -1
        right_ee_id = -1

        for i in range(self.model.nsite):
            site_name = self.model.site(i).name.lower()
            if "left" in site_name and left_ee_id == -1:
                left_ee_id = i
            elif "right" in site_name and right_ee_id == -1:
                right_ee_id = i

        return left_ee_id, right_ee_id