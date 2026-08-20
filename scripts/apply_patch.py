#!/usr/bin/env python3
"""
apply_patch.py

Run this from your project root, the directory containing the config/
package and state_machine.py:
    python3 apply_patch.py

Fixes the APPROACH_GRASP -> SQUEEZE_GRASP deadlock: the phase never
transitions once the arm's pad contacts the box, because the exact
joint-space position match can never be satisfied while contact blocks
convergence. Adds a settle+tolerance fallback.
"""
import re
import sys

CONFIG_FILE = "config/throwing_config.py"
STATE_MACHINE_FILE = "state_machine.py"

CONFIG_ANCHOR = "POSITION_THRESHOLDS = np.array([0.02, 0.02, 0.02, 0.02, 0.03, 0.03, 0.03] * 2)"
CONFIG_ADDITION = '''POSITION_THRESHOLDS = np.array([0.02, 0.02, 0.02, 0.02, 0.03, 0.03, 0.03] * 2)
APPROACH_GRASP_POS_TOLERANCE = 0.05  # rad, looser fallback tolerance for
                                       # APPROACH_GRASP -> SQUEEZE_GRASP when
                                       # contact blocks exact convergence to
                                       # grasp_left/right
APPROACH_SETTLE_QVEL = 0.02           # rad/s, "stopped moving" threshold'''

SM_OLD = '''        pos_thresh_left = config.POSITION_THRESHOLDS[0:7]
        pos_thresh_right = config.POSITION_THRESHOLDS[7:14]
        at_goal_left = np.all(np.abs(self.grasp_left - q_left) < pos_thresh_left)
        at_goal_right = np.all(np.abs(self.grasp_right - q_right) < pos_thresh_right)

        if traj_done_left and traj_done_right and at_goal_left and at_goal_right:'''

SM_NEW = '''        pos_thresh_left = config.POSITION_THRESHOLDS[0:7]
        pos_thresh_right = config.POSITION_THRESHOLDS[7:14]
        at_goal_left = np.all(np.abs(self.grasp_left - q_left) < pos_thresh_left)
        at_goal_right = np.all(np.abs(self.grasp_right - q_right) < pos_thresh_right)

        # Fallback: if the arm has stopped moving (stalled against contact
        # with the box) and is within a looser tolerance of the grasp
        # target, treat that as "reached" too. The tight at_goal check
        # above can permanently deadlock this phase once the pad contacts
        # the box, since the rigid joint-space target assumes free-space
        # reachability.
        settled_left = np.all(np.abs(qvel_left) < config.APPROACH_SETTLE_QVEL)
        settled_right = np.all(np.abs(qvel_right) < config.APPROACH_SETTLE_QVEL)
        close_left = np.all(np.abs(self.grasp_left - q_left) < config.APPROACH_GRASP_POS_TOLERANCE)
        close_right = np.all(np.abs(self.grasp_right - q_right) < config.APPROACH_GRASP_POS_TOLERANCE)
        reached_grasp = (at_goal_left and at_goal_right) or (
            settled_left and settled_right and close_left and close_right
        )

        if traj_done_left and traj_done_right and reached_grasp:'''


def patch_file(path, old, new, label):
    with open(path, "r") as f:
        content = f.read()
    if new in content:
        print(f"[{label}] already patched, skipping.")
        return
    count = content.count(old)
    if count == 0:
        print(f"[{label}] ERROR: anchor text not found. File may already differ "
              f"from the version this patch expects -- apply manually.")
        sys.exit(1)
    if count > 1:
        print(f"[{label}] ERROR: anchor text found {count} times, expected 1 "
              f"(not unique). Apply manually to avoid patching the wrong spot.")
        sys.exit(1)
    content = content.replace(old, new)
    with open(path, "w") as f:
        f.write(content)
    print(f"[{label}] patched.")


if __name__ == "__main__":
    patch_file(CONFIG_FILE, CONFIG_ANCHOR, CONFIG_ADDITION, "config/throwing_config.py")
    patch_file(STATE_MACHINE_FILE, SM_OLD, SM_NEW, "state_machine.py")
    print("Done. Re-run: python3 main_throw_sim.py --headless")