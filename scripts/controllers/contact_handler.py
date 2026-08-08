"""
Contact force/wrench handler for dual-arm manipulation
Computes contact forces and wrenches between end-effectors and objects
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

    def get_object_contact_wrenches(self, object_body_id):
        """
        Get wrenches at contact points between end-effectors and object

        Args:
            object_body_id: Body ID of the object

        Returns:
            left_wrench: 6D wrench from left end-effector
            right_wrench: 6D wrench from right end-effector
            contact_detected: Boolean indicating if contact exists
        """
        if object_body_id == -1:
            return np.zeros(6), np.zeros(6), False

        # Get all geometry IDs for the object
        object_geom_ids = []
        for i in range(self.model.ngeom):
            if self.model.geom_bodyid[i] == object_body_id:
                object_geom_ids.append(i)

        # Initialize wrenches
        left_wrench = np.zeros(6)
        right_wrench = np.zeros(6)
        contact_detected = False

        # Check all contacts
        for i in range(self.data.ncon):
            contact = self.data.contact[i]

            # Check if this contact involves the object
            if contact.geom1 in object_geom_ids or contact.geom2 in object_geom_ids:
                # Get contact force
                contact_force = np.zeros(6)
                mujoco.mj_contactForce(self.model, self.data, i, contact_force)

                # Determine which end effector is involved
                other_geom = contact.geom2 if contact.geom1 in object_geom_ids else contact.geom1
                other_body = self.model.geom_bodyid[other_geom]
                body_name = self.model.body(other_body).name

                # Compute torque around object center
                contact_pos = contact.pos.copy()
                object_pos = self.data.xpos[object_body_id].copy()
                lever_arm = contact_pos - object_pos
                torque = np.cross(lever_arm, contact_force[:3])

                # Add to appropriate wrench
                if "left" in body_name.lower():
                    left_wrench[:3] += contact_force[:3]
                    left_wrench[3:] += torque
                    contact_detected = True
                elif "right" in body_name.lower():
                    right_wrench[:3] += contact_force[:3]
                    right_wrench[3:] += torque
                    contact_detected = True

        return left_wrench, right_wrench, contact_detected

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
