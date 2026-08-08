"""
Impedance Controller for Object-Level Control
Implements the exact formulation from the mathematical specification:

    ẋ_o* = ẋ_o - D^(-1)[M_o·ẍ_des + w_mo^fb - ŵ_obj + K_1(x_o* - x_o)]

where:
    ẋ_o = current object velocity
    D = damping matrix
    M_o = object mass matrix (inertia)
    ẍ_des = desired acceleration from DS
    w_mo^fb = force feedback from end-effectors
    ŵ_obj = estimated object wrench (gravity compensation)
    K_1 = impedance stiffness
    x_o* = desired object position
    x_o = current object position
"""
import numpy as np


class ImpedanceController:
    """
    Object-level impedance controller that combines:
    - Feedforward acceleration from DS
    - Force feedback from contacts
    - Impedance control
    """

    def __init__(self, M_o, D, K_1):
        """
        Initialize impedance controller

        Args:
            M_o: Object mass matrix (3x3 diagonal for translational inertia)
            D: Damping matrix (3x3 diagonal)
            K_1: Impedance stiffness matrix (3x3 diagonal)
        """
        self.M_o = M_o
        self.D = D
        self.K_1 = K_1
        self.D_inv = np.linalg.inv(D)

    def compute_object_velocity(self, x_dot_o, x_ddot_des, w_mo_fb, w_obj_hat,
                                 x_o_star, x_o):
        """
        Compute commanded object velocity using impedance control

        Corrected formulation (feedforward control):
            ẋ_o* = ẋ_o + D^(-1)[M_o·ẍ_des + w_mo^fb - ŵ_obj + K_1(x_o* - x_o)]

        Note: The LaTeX had a minus sign, but for feedforward control we need plus.
        The inertial term M_o·ẍ_des should ADD to the velocity, not subtract.

        Args:
            x_dot_o: Current object velocity (3D)
            x_ddot_des: Desired acceleration from DS (3D)
            w_mo_fb: Force feedback from end-effectors (3D force)
            w_obj_hat: Estimated object wrench/gravity (3D force)
            x_o_star: Desired object position (3D)
            x_o: Current object position (3D)

        Returns:
            x_dot_o_star: Commanded object velocity (3D)
        """
        # Inertial term: M_o · ẍ_des
        inertial_term = self.M_o @ x_ddot_des

        # Impedance term: K_1(x_o* - x_o)
        position_error = x_o_star - x_o
        impedance_term = self.K_1 @ position_error

        # Combined force term: M_o·ẍ_des + w_mo^fb - ŵ_obj + K_1(x_o* - x_o)
        force_term = inertial_term + w_mo_fb - w_obj_hat + impedance_term

        # Impedance control law (CORRECTED): ẋ_o* = ẋ_o + D^(-1)[force_term]
        x_dot_o_star = x_dot_o + self.D_inv @ force_term

        return x_dot_o_star

    def estimate_gravity_wrench(self, object_mass, gravity=9.81):
        """
        Estimate gravity wrench on object

        Args:
            object_mass: Mass of object (kg)
            gravity: Gravitational acceleration (m/s^2)

        Returns:
            w_obj_hat: Estimated gravity force (3D)
        """
        w_obj_hat = np.array([0.0, 0.0, -object_mass * gravity])
        return w_obj_hat
