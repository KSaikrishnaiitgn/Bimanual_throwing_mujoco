# Sign Fix Summary

## Current Status After Impedance Fix

✅ **Fixed**: Impedance controller sign (changed from minus to plus)
❌ **New Issue**: Box throws in **opposite Y direction**

## Observed Behavior

### Expected:
- Target landing point: `[0.0, -2.5, 0.0]`
- Starting position: `[0.001, -0.498, 0.291]`
- Should move: **-Y direction** (from -0.5 to -2.5)

### Actual:
- Final position: `[-0.681, 1.047, 0.159]`
- Box moved in **+Y direction** (from -0.5 to +1.0)
- **Wrong direction!**

## Analysis

### Target Velocity Computation
```
Target velocity: [-6.71e-04, -2.287, 3.962]
```
- X: ~0 ✅
- Y: **-2.287** (negative, correct for going -0.5 → -2.5) ✅
- Z: +3.962 (upward, correct) ✅

**Conclusion**: Ballistic calculation is CORRECT

### DS Acceleration
```
x_ddot_des: [3.34e-02, -45.33, 78.20]
```
- Y: **-45.33** (large negative acceleration) ✅

**Conclusion**: DS is computing correct direction

### Final Motion
- Box ends at Y = +1.047 (moved in **+Y** direction)
- This is **opposite** of desired -Y direction

## Hypothesis

The problem is likely in **one of these**:

### 1. Force Feedback Sign ⚠️
In [throwing_modular_2.py:309](throwing_modular_2.py#L309):
```python
w_mo_fb = (left_wrench[:3] + right_wrench[:3])
```

If the wrenches have wrong signs, this could flip the direction.

### 2. Jacobian Mapping Sign ⚠️
In [dual_arm_jacobian.py](controllers/dual_arm_jacobian.py), the pseudoinverse might need sign correction.

### 3. Coordinate Frame Mismatch ⚠️
MuJoCo might use a different Y-axis convention than expected.

## Debugging Steps

### Step 1: Check Force Magnitudes
Add print statements in throwing phase:
```python
print(f"   w_mo_fb: {w_mo_fb}")
print(f"   w_obj_hat: {w_obj_hat}")
print(f"   x_dot_o_star: {x_dot_o_star}")
```

### Step 2: Check Jacobian Output
Print the computed joint velocities:
```python
print(f"   q_dot: {q_dot}")
```

### Step 3: Verify Coordinate System
Check if MuJoCo Y-axis points forward or backward:
- Standard robotics: **+Y = forward**
- Some simulators: **-Y = forward**

## Quick Fix: Try Negating Y Components

If the issue is just Y-axis convention, try in `trajectory_planner.py`:

```python
# After computing v_rel vector
# Flip Y component
v_horizontal[1] = -v_horizontal[1]  # Negate Y
```

Or in the config, set landing point with opposite sign:
```python
LANDING_POINT = np.array([0.0, +2.5, 0.0])  # Try positive instead
```

## Recommended Action

1. **First**: Check MuJoCo coordinate system convention
   - Look at the XML file to see which direction is "forward"
   - The target `Y = -2.5` should mean "forward" or "backward"?

2. **Then**: Add debug prints to track the sign through the pipeline:
   - Target velocity Y component
   - DS acceleration Y component
   - Impedance output Y component
   - Joint velocities

3. **Finally**: Apply fix at the correct location once root cause is found
