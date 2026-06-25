"""Backend-free conformance guardrail: every task ``reset`` must accept ``seed``.

The gym bridge (``metasim.task.gym_registration``) forwards ``env.reset(seed=)``
to a task only when the task's ``reset`` signature accepts it — tasks that
override ``reset`` without a ``seed`` parameter silently drop the seed, so
rollouts are not reproducible on the simulator side.

This test statically (no sim backend, no instantiation) scans every task file
and asserts:

* no *new* ``reset`` override lacks ``seed`` beyond the known P1-cleanup
  allowlist (ratchet: the set may only shrink);
* the allowlist has no stale entries (forces it to shrink as P1 lands);
* the unified family base classes already honor the contract.

When you fix a file's ``reset`` to accept ``seed`` (or stop overriding
``reset``), delete it from ``KNOWN_NO_SEED_RESET`` below.
"""

from __future__ import annotations

import ast
import pathlib

import pytest

_TASKS = pathlib.Path(__file__).resolve().parents[1] / "roboverse_pack" / "tasks"

# Family base classes fixed in the reset(seed) unification — must stay conformant.
_FIXED_BASES = {
    "libero/libero_base.py",
    "libero_90/libero_90_base.py",
    "maniskill/maniskill_base.py",
    "rlbench/rl_bench.py",
    "embodiedgen/base.py",
    "calvin/base_table.py",
    "pick_place/base.py",
    "humanoid/base/base_legged_robot.py",
}

# P1 cleanup targets: task files whose ``reset`` override still drops ``seed``.
# This list may only SHRINK. Remove an entry when you fix that file.
KNOWN_NO_SEED_RESET = frozenset({
    "benchmark/base.py",
    "beyondmimic/isaaclab/robots/actuator.py",
    "beyondmimic/metasim/envs/base_legged_robot.py",
    "beyondmimic/metasim/mdp/commands.py",
    "libero/native_libero.py",
    "libero_90/libero_kitchen_scene10_close_the_top_drawer_of_the_cabinet.py",
    "libero_90/libero_kitchen_scene10_close_the_top_drawer_of_the_cabinet_and_put_the_black_bowl_on_top_of_it.py",
    "libero_90/libero_kitchen_scene10_put_the_black_bowl_in_the_top_drawer_of_the_cabinet.py",
    "libero_90/libero_kitchen_scene10_put_the_butter_at_the_back_in_the_top_drawer_of_the_cabinet_and_close_it.py",
    "libero_90/libero_kitchen_scene10_put_the_butter_at_the_front_in_the_top_drawer_of_the_cabinet_and_close_it.py",
    "libero_90/libero_kitchen_scene10_put_the_chocolate_pudding_in_the_top_drawer_of_the_cabinet_and_close_it.py",
    "libero_90/libero_kitchen_scene1_open_drawer_put_bowl.py",
    "libero_90/libero_kitchen_scene1_put_the_black_bowl_on_the_plate.py",
    "libero_90/libero_kitchen_scene1_put_the_black_bowl_on_top_of_the_cabinet.py",
    "libero_90/libero_kitchen_scene2_put_the_black_bowl_at_the_back_on_the_plate.py",
    "libero_90/libero_kitchen_scene2_put_the_black_bowl_at_the_front_on_the_plate.py",
    "libero_90/libero_kitchen_scene2_put_the_black_bowl_in_the_middle_on_the_plate.py",
    "libero_90/libero_kitchen_scene2_put_the_middle_black_bowl_on_top_of_the_cabinet.py",
    "libero_90/libero_kitchen_scene2_stack_the_black_bowl_at_the_front_on_the_black_bowl_in_the_middle.py",
    "libero_90/libero_kitchen_scene2_stack_the_middle_black_bowl_on_the_back_black_bowl.py",
    "libero_90/libero_kitchen_scene3_put_the_frying_pan_on_the_stove.py",
    "libero_90/libero_kitchen_scene3_put_the_moka_pot_on_the_stove.py",
    "libero_90/libero_kitchen_scene3_turn_on_the_stove.py",
    "libero_90/libero_kitchen_scene3_turn_on_the_stove_and_put_the_frying_pan_on_it.py",
    "libero_90/libero_kitchen_scene4_close_the_bottom_drawer_of_the_cabinet.py",
    "libero_90/libero_kitchen_scene4_close_the_bottom_drawer_of_the_cabinet_and_open_the_top_drawer.py",
    "libero_90/libero_kitchen_scene4_put_the_black_bowl_in_the_bottom_drawer_of_the_cabinet.py",
    "libero_90/libero_kitchen_scene4_put_the_black_bowl_on_top_of_the_cabinet.py",
    "libero_90/libero_kitchen_scene4_put_the_wine_bottle_in_the_bottom_drawer_of_the_cabinet.py",
    "libero_90/libero_kitchen_scene4_put_the_wine_bottle_on_the_wine_rack.py",
    "libero_90/libero_kitchen_scene5_close_the_top_drawer_of_the_cabinet.py",
    "libero_90/libero_kitchen_scene5_put_the_black_bowl_in_the_top_drawer_of_the_cabinet.py",
    "libero_90/libero_kitchen_scene5_put_the_black_bowl_on_the_plate.py",
    "libero_90/libero_kitchen_scene5_put_the_black_bowl_on_top_of_the_cabinet.py",
    "libero_90/libero_kitchen_scene5_put_the_ketchup_in_the_top_drawer_of_the_cabinet.py",
    "libero_90/libero_kitchen_scene6_close_the_microwave.py",
    "libero_90/libero_kitchen_scene6_put_the_yellow_and_white_mug_to_the_front_of_the_white_mug.py",
    "libero_90/libero_kitchen_scene7_open_the_microwave.py",
    "libero_90/libero_kitchen_scene7_put_the_white_bowl_on_the_plate.py",
    "libero_90/libero_kitchen_scene7_put_the_white_bowl_to_the_right_of_the_plate.py",
    "libero_90/libero_kitchen_scene8_put_the_right_moka_pot_on_the_stove.py",
    "libero_90/libero_kitchen_scene8_turn_off_the_stove.py",
    "libero_90/libero_kitchen_scene9_put_the_frying_pan_on_the_cabinet_shelf.py",
    "libero_90/libero_kitchen_scene9_put_the_frying_pan_on_top_of_the_cabinet.py",
    "libero_90/libero_kitchen_scene9_put_the_frying_pan_under_the_cabinet_shelf.py",
    "libero_90/libero_kitchen_scene9_put_the_white_bowl_on_top_of_the_cabinet.py",
    "libero_90/libero_kitchen_scene9_turn_on_the_stove.py",
    "libero_90/libero_kitchen_scene9_turn_on_the_stove_and_put_the_frying_pan_on_it.py",
    "libero_90/living_room_scene1_pick_up_the_alphabet_soup_and_put_it_in_the_basket.py",
    "libero_90/living_room_scene1_pick_up_the_cream_cheese_box_and_put_it_in_the_basket.py",
    "libero_90/living_room_scene1_pick_up_the_ketchup_and_put_it_in_the_basket.py",
    "libero_90/living_room_scene1_pick_up_the_tomato_sauce_and_put_it_in_the_basket.py",
    "libero_90/living_room_scene2_pick_up_the_alphabet_soup_and_put_it_in_the_basket.py",
    "libero_90/living_room_scene2_pick_up_the_butter_and_put_it_in_the_basket.py",
    "libero_90/living_room_scene2_pick_up_the_milk_and_put_it_in_the_basket.py",
    "libero_90/living_room_scene2_pick_up_the_orange_juice_and_put_it_in_the_basket.py",
    "libero_90/living_room_scene2_pick_up_the_tomato_sauce_and_put_it_in_the_basket.py",
    "libero_90/living_room_scene3_pick_up_the_alphabet_soup_and_put_it_in_the_tray.py",
    "libero_90/living_room_scene3_pick_up_the_butter_and_put_it_in_the_tray.py",
    "libero_90/living_room_scene3_pick_up_the_cream_cheese_and_put_it_in_the_tray.py",
    "libero_90/living_room_scene3_pick_up_the_ketchup_and_put_it_in_the_tray.py",
    "libero_90/living_room_scene3_pick_up_the_tomato_sauce_and_put_it_in_the_tray.py",
    "libero_90/living_room_scene4_pick_up_the_black_bowl_on_the_left_and_put_it_in_the_tray.py",
    "libero_90/living_room_scene4_pick_up_the_chocolate_pudding_and_put_it_in_the_tray.py",
    "libero_90/living_room_scene4_pick_up_the_salad_dressing_and_put_it_in_the_tray.py",
    "libero_90/living_room_scene4_stack_the_left_bowl_on_the_right_bowl_and_place_them_in_the_tray.py",
    "libero_90/living_room_scene4_stack_the_right_bowl_on_the_left_bowl_and_place_them_in_the_tray.py",
    "maniskill/draw_svg.py",
    "maniskill/draw_triangle.py",
    "maniskill/lift_peg_upright.py",
    "maniskill/roll_ball.py",
    "mjlab/cartpole_train.py",
    "mjlab/mdp/commands.py",
    "mjlab/mdp/sensors.py",
    "mujoco_playground/handover.py",
    "mujoco_playground/open_cabinet.py",
    "mujoco_playground/pick.py",
    "pick_place/approach_grasp.py",
    "pick_place/approach_grasp_ceramic_teapot.py",
    "pick_place/approach_grasp_knife.py",
    "pick_place/hand_trajectory.py",
    "pick_place/track_banana.py",
    "pick_place/track_screwdriver.py",
    "pick_place/track_spoon.py",
    "robosuite/native.py",
    "robosuite/robosuite_env.py",
    "simpler_env/_native/control/base_controller.py",
    "simpler_env/_native/control/pd_ee_pose.py",
    "simpler_env/_native/control/pd_joint_pos.py",
    "simpler_env/_native/grasp.py",
})


def _reset_methods_without_seed(path: pathlib.Path) -> bool:
    """True if the file defines a class method ``reset`` lacking a ``seed`` arg."""
    try:
        tree = ast.parse(path.read_text(encoding="utf-8"))
    except (SyntaxError, UnicodeDecodeError):
        return False
    for node in ast.walk(tree):
        if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) or node.name != "reset":
            continue
        a = node.args
        names = {p.arg for p in (*a.posonlyargs, *a.args, *a.kwonlyargs)}
        if "seed" in names or a.kwarg is not None:  # explicit seed= or **kwargs
            continue
        return True
    return False


def _all_violators() -> set[str]:
    out = set()
    for path in _TASKS.rglob("*.py"):
        if _reset_methods_without_seed(path):
            out.add(path.relative_to(_TASKS).as_posix())
    return out


@pytest.mark.general
def test_no_new_reset_without_seed():
    """No task may add a ``reset`` override that drops ``seed`` (ratchet)."""
    new = sorted(_all_violators() - KNOWN_NO_SEED_RESET)
    assert not new, (
        "These task files override reset() without a seed parameter, breaking the "
        "env.reset(seed=) contract. Add seed=None and forward it to super().reset:\n  " + "\n  ".join(new)
    )


@pytest.mark.general
def test_reset_seed_allowlist_has_no_stale_entries():
    """Allowlisted files must still be violators — forces the list to shrink."""
    stale = sorted(KNOWN_NO_SEED_RESET - _all_violators())
    assert not stale, (
        "These files no longer violate the reset(seed) contract — remove them from "
        "KNOWN_NO_SEED_RESET:\n  " + "\n  ".join(stale)
    )


@pytest.mark.general
def test_unified_bases_accept_seed():
    """The fixed family base classes must keep accepting seed (regression guard)."""
    regressed = sorted(f for f in _FIXED_BASES if _reset_methods_without_seed(_TASKS / f))
    assert not regressed, f"Base classes regressed to reset() without seed: {regressed}"
