"""Catalog of RoboTwin v2 tasks (50 total).

For now we only enumerate them so the report can list coverage; per-task
asset/scene/reward wiring is a separate (large) workstream.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class RoboTwinTask:
    """One RoboTwin task — name maps to envs/<name>.py."""

    name: str
    description: str = ""
    needs_objaverse: bool = False  # objaverse bundle is 3.74 GB


# Hand-curated grouping; the full catalog is `ls ~/projects/robotwin/envs/`.
# `needs_objaverse=True` flags tasks whose scene includes generic
# objaverse clutter alongside the primary actor — we cannot load those
# without the 3.74 GB objaverse bundle.

ALL_TASKS = [
    RoboTwinTask("adjust_bottle", "Rotate a bottle on the table to a target yaw."),
    RoboTwinTask("beat_block_hammer", "Pick the hammer and hit a small red cube."),
    RoboTwinTask("blocks_ranking_rgb", "Sort three colored blocks left-to-right by RGB target."),
    RoboTwinTask("blocks_ranking_size", "Sort three blocks by size."),
    RoboTwinTask("click_alarmclock", "Press the top button on an alarm clock."),
    RoboTwinTask("click_bell", "Tap a desk bell."),
    RoboTwinTask("dump_bin_bigbin", "Empty a small bin into a larger bin.", needs_objaverse=True),
    RoboTwinTask("grab_roller", "Pick up a rolling pin."),
    RoboTwinTask("handover_block", "Pass a block from one arm to the other."),
    RoboTwinTask("handover_mic", "Pass a microphone between arms."),
    RoboTwinTask("hanging_mug", "Hang a mug on a peg by its handle."),
    RoboTwinTask("lift_pot", "Two arms lift a pot together."),
    RoboTwinTask("move_can_pot", "Move a can next to a pot."),
    RoboTwinTask("move_pillbottle_pad", "Move a pill bottle onto a pad."),
    RoboTwinTask("move_playingcard_away", "Move a playing card off the workspace."),
    RoboTwinTask("move_stapler_pad", "Move a stapler onto a pad."),
    RoboTwinTask("open_laptop", "Open a closed laptop with both arms."),
    RoboTwinTask("open_microwave", "Open the microwave door."),
    RoboTwinTask("pick_diverse_bottles", "Pick one of several different bottles."),
    RoboTwinTask("pick_dual_bottles", "Pick two bottles, one per arm."),
    RoboTwinTask("place_a2b_left", "Place object A onto/near object B (left)."),
    RoboTwinTask("place_a2b_right", "Place object A onto/near object B (right)."),
    RoboTwinTask("place_bread_basket", "Place bread into a basket."),
    RoboTwinTask("place_bread_skillet", "Place bread into a skillet."),
    RoboTwinTask("place_burger_fries", "Plate a burger + fries."),
    RoboTwinTask("place_can_basket", "Place a can into a basket."),
    RoboTwinTask("place_cans_plasticbox", "Place cans into a plastic box."),
    RoboTwinTask("place_container_plate", "Place a container on a plate."),
    RoboTwinTask("place_dual_shoes", "Place a pair of shoes."),
    RoboTwinTask("place_empty_cup", "Place an empty cup down."),
    RoboTwinTask("place_fan", "Place a desk fan."),
    RoboTwinTask("place_mouse_pad", "Place a mouse on a pad."),
    RoboTwinTask("place_object_basket", "Place an object into a basket."),
    RoboTwinTask("place_object_scale", "Place an object on a scale."),
    RoboTwinTask("place_object_stand", "Place an object on a stand."),
    RoboTwinTask("place_phone_stand", "Place a phone on a stand."),
    RoboTwinTask("place_shoe", "Place a single shoe."),
    RoboTwinTask("press_stapler", "Press a stapler."),
    RoboTwinTask("put_bottles_dustbin", "Put bottles into a dustbin."),
    RoboTwinTask("put_object_cabinet", "Put an object into a cabinet."),
    RoboTwinTask("rotate_qrcode", "Rotate a card to align a QR code."),
    RoboTwinTask("scan_object", "Scan an object with the wrist camera.", needs_objaverse=True),
    RoboTwinTask("shake_bottle", "Shake a bottle."),
    RoboTwinTask("shake_bottle_horizontally", "Shake a bottle horizontally."),
    RoboTwinTask("stack_blocks_three", "Stack three blocks."),
    RoboTwinTask("stack_blocks_two", "Stack two blocks."),
    RoboTwinTask("stack_bowls_three", "Stack three bowls."),
    RoboTwinTask("stack_bowls_two", "Stack two bowls."),
    RoboTwinTask("stamp_seal", "Press a stamp on a paper."),
    RoboTwinTask("turn_switch", "Flip a light switch."),
]


def task_by_name(name: str) -> RoboTwinTask:
    for t in ALL_TASKS:
        if t.name == name:
            return t
    raise KeyError(name)
