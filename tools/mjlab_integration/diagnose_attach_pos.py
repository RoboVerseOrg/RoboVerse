"""Step into the exact dm_control attach behavior for the cartpole MJCF.

We want to verify whether `robot_attached.pos = child_body.pos` actually
sticks — the exported XML shows cartpole/ at pos=(0,0,0) which causes the
cart to sit inside the ground plane.
"""

from __future__ import annotations

import os

from dm_control import mjcf

_MJLAB = os.environ.get("MJLAB_DIR", os.path.expanduser("~/projects/mjlab"))


def main():
    os.environ.setdefault("MUJOCO_GL", "egl")
    raw_xml = os.path.join(_MJLAB, "src/mjlab/tasks/cartpole/cartpole.xml")

    # Mirror MetaSim's _add_robots_to_model logic step-by-step
    parent = mjcf.RootElement()
    robot_xml = mjcf.from_path(raw_xml)
    robot_attached = parent.attach(robot_xml)

    print(f"robot_attached type: {type(robot_attached).__name__}")
    print(f"robot_attached.tag: {robot_attached.tag}")
    print(f"robot_attached.pos BEFORE assignment: {robot_attached.pos!r}")

    child_bodies = robot_attached.find_all("body")
    print(f"len(child_bodies) = {len(child_bodies)}")
    for i, b in enumerate(child_bodies):
        print(f"  child[{i}]: name={b.name!r}, pos={b.pos!r}")

    child_body = child_bodies[0]
    print(f"\nFIRST child_body.pos (=cart.pos): {child_body.pos!r}")

    saved_pos = child_body.pos
    robot_attached.pos = child_body.pos
    child_body.pos = "0 0 0"

    print(f"After assignment, robot_attached.pos: {robot_attached.pos!r}")
    print(f"After assignment, child_body.pos: {child_body.pos!r}")

    # Inject a dummy inertial like MetaSim does
    robot_attached.add("inertial", mass="1e-9", diaginertia="1e-9 1e-9 1e-9", pos=(0.0, 0.0, 0.0))

    xml = parent.to_xml_string()
    print("\n--- Resulting attached body block (first 800 chars) ---")
    # Print just the worldbody region for brevity
    import re

    m = re.search(r"<worldbody>.*?</worldbody>", xml, flags=re.DOTALL)
    if m:
        block = m.group(0)
        print(block[:1500])

    # Verify by parsing the xml back
    import mujoco

    model = mujoco.MjModel.from_xml_string(xml)
    print("\n=== Loaded model body positions ===")
    for i in range(model.nbody):
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, i) or "<unnamed>"
        print(f"  body[{i}] name={name!r} pos={model.body_pos[i]}")


if __name__ == "__main__":
    main()
