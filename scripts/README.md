# Dual-FR3 Bimanual Throwing — Current Formulation

This directory contains the MuJoCo implementation of a bimanual throwing task
with two Franka FR3 robots. The **current experimental controller** is the
finite-time terminal-state trajectory controller launched by:

```bash
python3 main_throw_trajectory.py
```

The central idea is that the release state is **not an equilibrium** where the
box should stop. It is a state that the box must pass through at a specified
time with both the required position and the required velocity.

The older DS/impedance implementation remains in the repository for reference,
but it is not the formulation described below.

## 1. Current experiment parameters

The experimental overrides are defined in
`config/throwing_trajectory_config.py`.

| Parameter | Current value | Meaning |
|---|---:|---|
| Gravity | `[0, 0, -9.81] m/s²` | MuJoCo world gravity |
| Release point | `[-0.42, 0.55, 0.50] m` | Desired box-centre position at release |
| Landing point | `[-0.42, 0.78, 0.10] m` | Desired box-centre position after landing |
| Launch angle | `35°` | Angle used by the projectile calculation |
| Throw duration | `0.8 s` | Duration of the cubic pre-release trajectory |
| Position feedback | `diag(5, 5, 5)` | Terminal trajectory position correction |
| Velocity feedback | `diag(0.15, 0.15, 0.15)` | Terminal trajectory velocity correction |
| Release position tolerance | `0.010 m` | Maximum permitted position error |
| Release velocity tolerance | `0.08 m/s` | Maximum permitted velocity error |

The box landing height is `z = 0.10 m`, rather than zero, because the box is
0.20 m tall and its position is measured at its centre.

## 2. Controller flow

The simulation progresses through these states:

```text
APPROACH_PREGRASP
        ↓
APPROACH_GRASP
        ↓
SQUEEZE_GRASP
        ↓
THROW
        ↓
RELEASE
        ↓
FOLLOW_THROUGH
        ↓
DONE
```

- **APPROACH_PREGRASP:** move both arms to collision-safe pregrasp poses.
- **APPROACH_GRASP:** move the pads onto the two opposite box faces.
- **SQUEEZE_GRASP:** regulate the opposing grasp forces and hold pose.
- **THROW:** track the cubic terminal-state trajectory.
- **RELEASE:** open both hands while retaining the ballistic box velocity.
- **FOLLOW_THROUGH:** move the arms away safely.
- **DONE:** hold the arms stationary while the box remains at rest.

The state implementation is in `state_machine_trajectory.py`, which subclasses
the shared phase machinery in `state_machine.py`.

## 3. Ballistic release velocity

`release_velocity.py` calculates the release velocity required to travel from
the theoretical release point to the theoretical landing point at a selected
launch angle.

For

```text
Δp = p_land - p_release
R  = ||Δp_xy||
Δz = Δp_z
```

the required speed is

```text
v_release = sqrt(g R² / (2 cos²(θ) (R tan(θ) - Δz)))
```

The horizontal direction and three-dimensional release velocity are

```text
e_h = Δp_xy / R

v_release_vector =
    v_release cos(θ) [e_hx, e_hy, 0]
    + v_release sin(θ) [0, 0, 1]
```

For the current points and angle, the target is approximately:

```text
release speed    = 0.83020 m/s
release velocity = [0.00000, 0.68006, 0.47618] m/s
time of flight   = 0.33821 s
```

Small code representation:

```python
solution = release_velocity.compute_release_velocity(
    RELEASE_POINT, LANDING_POINT, THROW_ANGLE
)
v_release = solution["v_rel_vector"]  # desired velocity at release
```

The projectile model assumes constant gravity and no aerodynamic drag:

```text
p(t) = p_release + v_release t + 0.5 g t²
```

## 4. Finite-time cubic trajectory

At the instant the controller enters `THROW`, it reads the real box state:

```text
x(0) = x₀
v(0) = v₀
```

It then constructs a cubic Hermite trajectory that reaches

```text
x(T) = x_release
v(T) = v_release
```

where `T = 0.8 s`.

With normalized time `s = t/T`, the reference is

```text
x_ref(s) = h00(s)x₀ + h10(s)Tv₀ + h01(s)x_release + h11(s)Tv_release
```

using

```text
h00 =  2s³ - 3s² + 1
h10 =   s³ - 2s² + s
h01 = -2s³ + 3s²
h11 =   s³ - s²
```

`terminal_throw_controller.py` implements the polynomial and analytically
calculates its position, velocity, and acceleration.

Small code representation:

```python
trajectory = CubicTerminalTrajectory(x0, v0, x_release, v_release, 0.8)
x_ref, v_ref, a_ref = trajectory.sample(elapsed)
```

The important point is that both endpoint position and endpoint velocity are
built directly into the reference. The box is not asked to stop at the release
point.

## 5. Closed-loop trajectory correction

The cubic is the feedforward reference. Feedback corrects tracking errors:

```text
v_cmd = v_ref + Kx(x_ref - x) + Kv(v_ref - v)
```

This is implemented by `compute_terminal_velocity_command()` in
`terminal_throw_controller.py`:

```python
cmd = (
    v_ref
    + TRAJECTORY_K_POS @ (x_ref - x)
    + TRAJECTORY_K_VEL @ (v_ref - v)
)
```

The command is norm-limited by `MAX_OBJ_VEL` for safety. The feedback does not
generate the throw trajectory; it only corrects deviations from the cubic
reference.

## 6. Mapping box motion to both hands

The box command is converted to a rigid-body twist:

```text
V_object = [v_object, ω_object]
```

For contact point `i`, with offset `r_i` from the box centre:

```text
v_contact_i = v_object + ω_object × r_i
```

The corresponding matrix block is

```text
[ I  -skew(r_i) ]
[ 0       I     ]
```

`rigid_contact_twist_map()` in `terminal_throw_controller.py` stacks the left
and right blocks. Both hands receive the **complete** rigid-body velocity. The
velocity is not divided by two because contact velocity is a kinematic
constraint, not a force allocation.

A small grasp-admittance correction is added along each pad normal:

```python
desired_hand_twists = grasp_admittance + H @ object_twist_cmd
```

This lets the hands maintain squeeze force while carrying the box along the
throwing trajectory.

## 7. Cartesian-to-joint conversion

`dual_arm_kinematics.py` builds the block-diagonal stacked Jacobian:

```text
J = [ J_left     0    ]
    [    0    J_right ]
```

The commanded joint velocity is obtained with a damped pseudoinverse:

```text
q̇_cmd = J# V_hands

J# = Jᵀ(JJᵀ + λ²I)⁻¹
```

The damping reduces excessive joint velocities near singular
configurations. The current default damping is `0.05`.

`VelocityPID` in `joint_controllers.py` converts the joint-velocity command to
joint torque. It includes:

- velocity-error feedback;
- integral correction;
- joint-friction feedforward;
- MuJoCo bias-torque compensation for gravity and Coriolis effects;
- joint-velocity and actuator-torque limiting.

## 8. Release decision

The controller does not release solely because `0.8 s` has elapsed. At or
after the nominal terminal time, it evaluates

```text
position_error = ||x - x_release||
velocity_error = ||v - v_release||
```

Release occurs only when

```text
position_error < 0.010 m
velocity_error < 0.08 m/s
```

This logic is in `TrajectoryThrowStateMachine._step_throw()`.

During `RELEASE`, both hands retain the full forward/upward release velocity
and simultaneously separate along their opposing pad normals. This avoids
braking the box while the contacts are opening.

## 9. Gravity

Gravity remains active throughout the complete experiment. The loaded model is
`robot_description/Dual_franka.xml`, which contains:

```xml
<option gravity="0 0 -9.81" timestep="0.002" integrator="implicitfast"/>
```

Consequently, gravity acts during grasping, throwing, free flight, and
landing. The joint controller uses MuJoCo bias forces to compensate for robot
gravity while the free box still follows normal projectile dynamics after
release.

## 10. Main source files

| File | Purpose |
|---|---|
| `main_throw_trajectory.py` | Main simulation, logging, plotting and video entry point |
| `config/throwing_trajectory_config.py` | Current experimental release/landing points, duration, gains and tolerances |
| `terminal_throw_controller.py` | Cubic Hermite trajectory, feedback law and rigid-contact twist map |
| `state_machine_trajectory.py` | Current THROW and RELEASE phase implementation |
| `release_velocity.py` | Projectile calculation for release speed, direction and flight time |
| `dual_arm_kinematics.py` | Stacked Jacobian and damped pseudoinverse |
| `contact_admittance.py` | Grasp-force regulation along the pad normals |
| `joint_controllers.py` | Joint-velocity PID producing actuator torques |
| `mj_interface.py` | Safe access to MuJoCo state, sensors, Jacobians and controls |
| `logger.py` | CSV logging and summary plots |
| `plot_trajectory_validation.py` | Calculated-versus-achieved trajectory validation plots |
| `video_recorder.py` | Multi-camera Full-HD recording and release/landing markers |

`config/throwing_config.py` contains shared robot, actuator, sensor and
controller parameters. The trajectory-specific file imports that baseline and
overrides only the experimental throwing values.

## 11. Running the simulation

Interactive MuJoCo window:

```bash
python3 main_throw_trajectory.py
```

Headless simulation:

```bash
python3 main_throw_trajectory.py --headless
```

The standard run writes:

```text
throw_trajectory_log.csv
throw_trajectory_summary.png
```

## 12. Validation plots

Generate the calculated-versus-achieved plots:

```bash
python3 plot_trajectory_validation.py
```

Open the rotatable three-dimensional plot as well:

```bash
python3 plot_trajectory_validation.py --show-3d
```

Generated files:

```text
trajectory_validation_3d.png
box_trajectory_3d.png
trajectory_velocity_validation.png
```

The deliberately removed error-only panels are not generated; the remaining
position, velocity, end-effector and 3D trajectory comparisons are preserved.

## 13. High-quality annotated videos

Record the supplied user camera pose together with side and top views:

```bash
python3 main_throw_trajectory.py \
  --headless \
  --record \
  --cameras user_pose,side,top \
  --video-fps 30 \
  --video-width 1920 \
  --video-height 1080
```

The current custom camera is equivalent to:

```xml
<camera pos="-0.435 2.678 1.346"
        xyaxes="-0.999 0.052 -0.000 -0.022 -0.429 0.903"/>
```

Videos are written under `throw_videos/`. Available presets are:

```text
user_pose, front, side, oblique, top
```

The point colors are:

| Marker | Color |
|---|---|
| Theoretical release point | Cyan |
| Actual release point | Red |
| Theoretical landing point | Green |
| Actual landing point | Magenta |

The theoretical points exist from the start. The actual release point appears
when release is triggered, and the actual landing point appears once the box
has settled and the state reaches `DONE`.

The recorder automatically expands MuJoCo's off-screen framebuffer to the
requested resolution and strengthens the camera-following fill light so that
the Frankas remain visible from low and side camera angles.

## 14. Interpreting logged errors

Use a consistent signed component error:

```text
error = actual - theoretical
```

- A positive `y` error means the box travelled farther forward than planned.
- A negative `y` error means it fell short.
- A positive `z` error means it is above the planned point.
- A negative `z` error means it is below the planned point.

The Euclidean error reported by the controller is

```text
||error|| = sqrt(error_x² + error_y² + error_z²)
```

Actual results should be read from the newest `throw_trajectory_log.csv`, since
they depend on the simulation run and controller tuning.

## 15. Important implementation notes

1. Run `main_throw_trajectory.py` for the current formulation. Running another
   older entry point may invoke the baseline DS/impedance controller instead.
2. Release and landing coordinates describe the **box centre** in the MuJoCo
   world frame.
3. The ballistic model neglects drag and assumes constant gravity.
4. Torque, joint-velocity, workspace and singularity limits can prevent exact
   tracking even when the mathematical reference is feasible.
5. The cubic reference is rebuilt from the measured box state whenever the
   throw begins, avoiding a discontinuity caused by assuming an ideal initial
   state.
6. The source tree contains historical and experimental scripts. The file map
   above identifies the path used by the current trajectory-based run.
