"""
Main entry point: loads the MuJoCo model, resolves the object/EE ids,
builds the DSThrowingController, and drives the physics + viewer loop.

This is the piece that was missing before — config.py and
ds_throwing_controller.py only *define* the controller; nothing was
actually stepping the simulation or opening a viewer.

Run:
    python3 main.py
"""
import sys
import time

import numpy as np
import mujoco
import mujoco.viewer

from config import throwing_config as config
from controllers.Contact_handler import ContactHandler
from ds_throwing_controller import DSThrowingController


def resolve_object_body_id(model, name_substring):
    """
    Find the body id whose name contains `name_substring` (case-insensitive).
    Prints all body names if nothing matches, so you can fix
    config.OBJECT_BODY_NAME instead of guessing.
    """
    name_substring = name_substring.lower()
    matches = []
    for i in range(model.nbody):
        body_name = model.body(i).name
        if name_substring in body_name.lower():
            matches.append((i, body_name))

    if not matches:
        all_names = [model.body(i).name for i in range(model.nbody)]
        raise ValueError(
            f"No body name containing '{name_substring}' found.\n"
            f"Available body names in the XML:\n  " + "\n  ".join(all_names) +
            f"\n\nSet config.OBJECT_BODY_NAME to the correct one."
        )
    if len(matches) > 1:
        print(f"[main] Warning: multiple bodies match '{name_substring}': "
              f"{matches}. Using the first: '{matches[0][1]}'.")
    return matches[0][0]


def resolve_ee_sites(model):
    """
    Use ContactHandler's existing left/right site finder. Raises with the
    full site list if either side can't be found, instead of silently
    returning -1 and letting the Jacobian code fail downstream.
    """
    dummy_data = mujoco.MjData(model)
    finder = ContactHandler(model, dummy_data)
    left_id, right_id = finder.find_end_effector_sites()

    if left_id == -1 or right_id == -1:
        all_sites = [model.site(i).name for i in range(model.nsite)]
        raise ValueError(
            f"Could not find both EE sites (left_id={left_id}, right_id={right_id}).\n"
            f"Available site names in the XML:\n  " + "\n  ".join(all_sites) +
            f"\n\nSite names need 'left' / 'right' in them, or edit "
            f"resolve_ee_sites() in main.py to match your naming."
        )
    return left_id, right_id


def describe_object(model, data, object_body_id):
    """
    Print the object's actual world position and the size of every geom
    attached to it. Handy sanity check even though REACHING now targets
    fixed joint configs rather than a Cartesian point near the object —
    e.g. to confirm the box is actually where GRASP_JOINT_TARGET_LEFT/RIGHT
    are expected to leave the hands.
    """
    mujoco.mj_forward(model, data)
    obj_pos = data.xpos[object_body_id].copy()
    print(f"[main] object '{model.body(object_body_id).name}' world position = {obj_pos}")

    geom_ids = [i for i in range(model.ngeom) if model.geom_bodyid[i] == object_body_id]
    for gid in geom_ids:
        gtype = model.geom_type[gid]
        gsize = model.geom_size[gid].copy()
        type_name = mujoco.mjtGeom(gtype).name
        print(f"[main]   geom '{model.geom(gid).name or gid}' type={type_name} size={gsize}")
    return obj_pos, geom_ids


def debug_print_raw_contacts(model, data, object_body_id):
    """
    Diagnostic: lists EVERY contact touching the object's geoms, regardless
    of whether ContactHandler could attribute it to a "left"/"right" body
    name. If this shows real contacts while controller.last_log still says
    contacted=False, the bug is in ContactHandler's body-name matching, not
    in the reach/grasp targets — check your gripper body names against
    what's printed here.
    """
    object_geom_ids = {i for i in range(model.ngeom) if model.geom_bodyid[i] == object_body_id}
    if data.ncon == 0:
        print("[main][contact-debug] ncon=0 (no contacts anywhere in the scene)")
        return
    hits = []
    for i in range(data.ncon):
        c = data.contact[i]
        if c.geom1 in object_geom_ids or c.geom2 in object_geom_ids:
            g1_body = model.body(model.geom_bodyid[c.geom1]).name
            g2_body = model.body(model.geom_bodyid[c.geom2]).name
            hits.append(f"{model.geom(c.geom1).name}({g1_body}) <-> {model.geom(c.geom2).name}({g2_body})")
    if hits:
        print(f"[main][contact-debug] {len(hits)} contact(s) involving the object:")
        for h in hits:
            print(f"[main][contact-debug]   {h}")
    else:
        print(f"[main][contact-debug] ncon={data.ncon} total, but NONE involve the object's geoms")


def build_release_pos_dynamic(model, data, object_body_id, lift_height=0.3):
    """
    OPTIONAL alternative to config.RELEASE_POS: compute the release point
    at runtime as directly above the object's current resting position,
    lifted by `lift_height`. Not used by default — main() uses
    config.RELEASE_POS. Switch to this (see USE_DYNAMIC_RELEASE_POS below)
    if you'd rather the release point track wherever the object actually
    starts each run instead of a fixed world-frame constant.

    NOTE: this is NOT a scripted lift phase — the controller has no such
    phase. It only decides where THROWING's DS+impedance law aims for; the
    object's actual path there is whatever the DS produces, not a separate
    joint-space waypoint sequence.
    """
    mujoco.mj_forward(model, data)  # populate xpos before we read it
    obj_pos = data.xpos[object_body_id].copy()
    release_pos = obj_pos.copy()
    release_pos[2] += lift_height
    return release_pos


# Set True to compute release_pos from the object's live position instead
# of using the fixed config.RELEASE_POS constant.
USE_DYNAMIC_RELEASE_POS = False

# How long (s) to stay in `grasping` with no contact before dumping raw
# contact diagnostics, to help pin down a ContactHandler naming mismatch.
GRASP_STALL_DEBUG_AFTER = 2.0


def main():
    # ---- Load model ----
    xml_path = config.XML_FILE_PATH
    print(f"[main] Loading model from: {xml_path}")
    try:
        model = mujoco.MjModel.from_xml_path(xml_path)
    except Exception as e:
        print(f"[main] Failed to load XML at '{xml_path}': {e}")
        sys.exit(1)
    data = mujoco.MjData(model)

    # ---- Resolve ids (prints available names on failure, doesn't guess silently) ----
    object_body_id = resolve_object_body_id(model, config.OBJECT_BODY_NAME)
    left_ee_site_id, right_ee_site_id = resolve_ee_sites(model)

    print(f"[main] object_body_id = {object_body_id} "
          f"('{model.body(object_body_id).name}')")
    print(f"[main] left_ee_site_id = {left_ee_site_id} "
          f"('{model.site(left_ee_site_id).name}')")
    print(f"[main] right_ee_site_id = {right_ee_site_id} "
          f"('{model.site(right_ee_site_id).name}')")

    # ---- Object geometry (informational — REACHING targets joints now, not this) ----
    describe_object(model, data, object_body_id)
    print(f"[main] GRASP_JOINT_TARGET_LEFT  = {config.GRASP_JOINT_TARGET_LEFT}")
    print(f"[main] GRASP_JOINT_TARGET_RIGHT = {config.GRASP_JOINT_TARGET_RIGHT}")

    # ---- Release point ----
    # By default this comes straight from config.RELEASE_POS (set it there
    # to tune the throw). Flip USE_DYNAMIC_RELEASE_POS above to instead
    # compute it from the object's live starting position each run.
    if USE_DYNAMIC_RELEASE_POS:
        release_pos = build_release_pos_dynamic(model, data, object_body_id)
    else:
        release_pos = np.asarray(config.RELEASE_POS, dtype=float)
    print(f"[main] release_pos = {release_pos}  "
          f"(source: {'dynamic' if USE_DYNAMIC_RELEASE_POS else 'config.RELEASE_POS'})")

    # ---- Build controller (imports config.py itself; see ds_throwing_controller.py) ----
    controller = DSThrowingController(
        model, data, object_body_id,
        left_ee_site_id, right_ee_site_id,
        release_pos=release_pos,
    )
    print(f"[main] v_release = {controller.v_release}")

    # ---- Sim + viewer loop ----
    last_phase = None
    last_print_time = 0.0
    grasping_start_time = None
    grasp_debug_fired = False

    with mujoco.viewer.launch_passive(model, data) as viewer:
        while viewer.is_running():
            step_start = time.time()

            controller.control_callback(model, data)
            mujoco.mj_step(model, data)

            # Phase-change / periodic logging
            if controller.phase != last_phase:
                print(f"[main] t={data.time:.3f}s  phase -> {controller.phase}")
                last_phase = controller.phase
                if controller.phase == DSThrowingController.PHASE_GRASPING:
                    grasping_start_time = data.time
                    grasp_debug_fired = False
            elif data.time - last_print_time > config.PHASE_PRINT_INTERVAL:
                log = controller.last_log
                log_summary = {
                    k: (np.round(v, 3) if isinstance(v, np.ndarray) else v)
                    for k, v in log.items() if k != 'phase'
                }
                print(f"[main] t={data.time:.3f}s  phase={controller.phase}  "
                      f"log={log_summary}")
                last_print_time = data.time

            # Stuck in grasping with no contact for a while -> dump raw contacts
            if (controller.phase == DSThrowingController.PHASE_GRASPING
                    and not grasp_debug_fired
                    and grasping_start_time is not None
                    and data.time - grasping_start_time > GRASP_STALL_DEBUG_AFTER
                    and not controller.last_log.get("contacted", False)):
                print(f"[main] Still not contacted {GRASP_STALL_DEBUG_AFTER}s into grasping — "
                      f"dumping raw contacts:")
                debug_print_raw_contacts(model, data, object_body_id)
                grasp_debug_fired = True

            viewer.sync()

            # Real-time pacing
            time_until_next_step = model.opt.timestep - (time.time() - step_start)
            if time_until_next_step > 0:
                time.sleep(time_until_next_step)


if __name__ == "__main__":
    main()