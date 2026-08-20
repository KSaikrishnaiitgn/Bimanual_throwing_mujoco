"""
Phase controller for managing the throwing task state machine
Orchestrates the different phases: reaching, lifting, throwing
"""
import numpy as np
import time


class PhaseController:
    """Manages different phases of the throwing task"""

    def __init__(self, config):
        """
        Initialize phase controller

        Args:
            config: Configuration module with all parameters
        """
        self.config = config
        self.phase = "reaching_first_target"
        self.state = {
            "current_targets": None,
            "phase_start_time": time.time(),
            "last_phase_change": time.time(),
            "last_print_time": time.time(),
            "throw_phase_started": False,
            "throw_start_time": None,
            "release_announced": False,
            "natural_release_announced": False,
        }

    def get_phase(self):
        """Get current phase"""
        return self.phase

    def set_phase(self, new_phase):
        """Set new phase and update state"""
        self.phase = new_phase
        self.state["last_phase_change"] = time.time()
        if new_phase == "throwing_object" and not self.state["throw_phase_started"]:
            self.state["throw_start_time"] = time.time()
            self.state["throw_phase_started"] = True

    def should_print_phase(self, interval=1.0):
        """Check if we should print phase update"""
        current_time = time.time()
        if current_time - self.state["last_print_time"] > interval:
            self.state["last_print_time"] = current_time
            return True
        return False

    def initialize_targets(self, num_actuators):
        """
        Initialize target positions for both robots

        Args:
            num_actuators: Total number of actuators

        Returns:
            first_targets: First target positions
            second_targets: Second target positions
        """
        actuators_per_robot = num_actuators // 2

        first_targets = np.zeros(num_actuators)
        first_targets[:actuators_per_robot] = self.config.FIRST_TARGET_POSITIONS_LEFT
        first_targets[actuators_per_robot:] = self.config.FIRST_TARGET_POSITIONS_RIGHT

        second_targets = np.zeros(num_actuators)
        second_targets[:actuators_per_robot] = self.config.SECOND_TARGET_POSITIONS_LEFT
        second_targets[actuators_per_robot:] = self.config.SECOND_TARGET_POSITIONS_RIGHT

        self.state["current_targets"] = first_targets.copy()
        self.state["first_targets"] = first_targets
        self.state["second_targets"] = second_targets

        return first_targets, second_targets

    def check_reaching_complete(self, joint_positions, num_actuators):
        """
        Check if reaching phase is complete

        Args:
            joint_positions: Current joint positions
            num_actuators: Total number of actuators

        Returns:
            all_reached: Boolean indicating if all joints reached targets
        """
        position_errors = self.state["current_targets"] - joint_positions

        all_reached = True
        for i in range(num_actuators):
            if abs(position_errors[i]) > self.config.POSITION_THRESHOLDS[i]:
                all_reached = False
                break

        return all_reached

    def compute_position_control(self, joint_positions, num_actuators):
        """
        Compute velocity commands for position control

        Args:
            joint_positions: Current joint positions
            num_actuators: Total number of actuators

        Returns:
            velocity_commands: Joint velocity commands
        """
        position_errors = self.state["current_targets"] - joint_positions
        velocity_commands = self.config.KP_GAINS * position_errors
        velocity_commands = np.clip(velocity_commands, -self.config.MAX_JOINT_VELOCITIES,
                                   self.config.MAX_JOINT_VELOCITIES)

        # Apply deadband
        for i in range(num_actuators):
            if abs(position_errors[i]) < self.config.POSITION_THRESHOLDS[i] * 0.5:
                velocity_commands[i] = 0.0

        return velocity_commands

    def check_lift_complete(self, current_height, target_height):
        """
        Check if lifting phase is complete

        Args:
            current_height: Current object height
            target_height: Target height

        Returns:
            Boolean indicating if lift is complete
        """
        return current_height >= target_height - self.config.HEIGHT_THRESHOLD

    def check_release_condition(self, strategy, elapsed_time, current_vel, target_vel):
        """
        Check if release condition is met

        Args:
            strategy: Release strategy ("time", "velocity", or "retract")
            elapsed_time: Time elapsed in throwing phase
            current_vel: Current object velocity
            target_vel: Target release velocity

        Returns:
            should_release: Boolean
            release_reason: String describing reason
        """
        should_release = False
        release_reason = ""

        if strategy == "time":
            if elapsed_time > self.config.THROW_DURATION:
                should_release = True
                release_reason = f"Time limit reached ({elapsed_time:.3f}s > {self.config.THROW_DURATION}s)"

        elif strategy == "velocity":
            vel_error = np.linalg.norm(current_vel - target_vel)
            if vel_error < self.config.VELOCITY_ERROR_THRESHOLD:
                should_release = True
                release_reason = f"Target velocity reached (error: {vel_error:.3f} m/s)"

        elif strategy == "retract":
            if elapsed_time > self.config.THROW_DURATION:
                should_release = True
                release_reason = f"Retraction time reached ({elapsed_time:.3f}s)"

        return should_release, release_reason

    def get_retract_velocity(self):
        """Get retraction velocity command for arms"""
        retract_velocity = np.zeros(12)  # Assuming 6 DOF per arm
        retract_velocity[:6] = [0.0, -self.config.RETRACT_VELOCITY, 0.0,
                               -self.config.RETRACT_VELOCITY, 0.0, 0.0]
        retract_velocity[6:] = [0.0, -self.config.RETRACT_VELOCITY, 0.0,
                                self.config.RETRACT_VELOCITY, 0.0, 0.0]
        return retract_velocity
