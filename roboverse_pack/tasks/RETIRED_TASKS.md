# Retired tasks

Tasks removed from `roboverse_pack` on 2026-11-27 because they failed the release gate (`python -m metasim bench health --floor --repeat 2`: demos replay at or above 0.80 on their own backend, the hold policy scores 0, two replays agree; demo-less RL tasks must build, step and reset). Nothing here is lost: every file is in git at commit `56a0e67b`.

Restore one task:

```bash
git checkout 56a0e67b -- roboverse_pack/tasks/<family>/<file>.py   # then make it pass the gate before registering it again
```

3022 classes retired. Machine-readable copy: `retired_tasks.csv` next to this file.

| family | classes | why |
|---|---|---|
| maniskill | 2666 | 1588 no demos or assets on the Hub; 999 USD assets absent from the Hub; 56 asset or demo missing on the Hub; 15 replays at 0 on Isaac Sim 6 (USD-only); 5 demos replay below 0.80; 3 no class-level scenario |
| libero | 178 | 169 no class-level scenario; 9 demos replay below 0.80 |
| rlbench | 78 | 73 replays at 0 on Isaac Sim 6 (USD-only); 5 demos replay below 0.80 |
| libero_90 | 65 | 58 demos replay below 0.80; 7 asset or demo missing on the Hub |
| pick_place | 14 | 9 errors when built or stepped; 4 asset or demo missing on the Hub; 1 replays at 0 on Isaac Sim 6 (USD-only) |
| calvin | 5 | 5 replays at 0 on Isaac Sim 6 (USD-only) |
| robosuite | 5 | 5 no class-level scenario |
| mujoco_playground | 3 | 2 errors when built or stepped; 1 replays at 0 on Isaac Sim 6 (USD-only) |
| beyondmimic | 2 | 2 errors when built or stepped |
| mjlab | 2 | 2 superseded by the _v2 port |
| bimanual | 1 | 1 demos replay below 0.80 |
| box_task | 1 | 1 asset or demo missing on the Hub |
| embodiedgen | 1 | 1 errors when built or stepped |
| humanoid | 1 | 1 asset or demo missing on the Hub |

## Every retired class

### never importable

- `simpler.google_robot_pick_coke_can`, `simpler.google_robot_pick_horizontal_coke_can`, `simpler.google_robot_pick_standing_coke_can`, `simpler.google_robot_pick_vertical_coke_can` (`roboverse_pack/tasks/simpler_env/`): the SIMPLER port needs `sapien` (not a Tier-1 backend) and never imported in this environment; the whole package is retired.
- `motion-tracking`, `motion-tracking-isaaclab`, `motion-tracking-isaaclab-deploy` (`roboverse_pack/tasks/beyondmimic/`): Isaac-Lab-only training environments that assert on any other backend; the registrations are removed, the training code stays.


### beyondmimic

- `motion-tracking-isaaclab` (`roboverse_pack/tasks/beyondmimic/isaaclab/envs/tracking_rl_env.py`): error: AssertionError: Only Isaac Lab is supported for training of motion tracking task
- `motion-tracking-isaaclab-deploy` (`roboverse_pack/tasks/beyondmimic/isaaclab/envs/tracking_rl_env.py`): error: AssertionError: Only Isaac Lab is supported for training of motion tracking task

### bimanual

- `bimanual.franka_handover` (`roboverse_pack/tasks/bimanual/franka_handover.py`): low: replay 0.00

### box_task

- `box_task.replay` (`roboverse_pack/tasks/box_task/box_task_replay.py`): error: Exception: File roboverse_data/assets/box_task/local_pack_box/cardboard_box/cardboard_box.xml neither exists in the local directory n…

### calvin

- `calvin.base_table` (`roboverse_pack/tasks/calvin/base_table.py`): MuJoCo: USD-only asset; Isaac 6: error Isaac Sim 6 child: AttributeError: 'str' object has no attribute 'values'
- `calvin.base_table_a` (`roboverse_pack/tasks/calvin/scene_A.py`): MuJoCo: USD-only asset; Isaac 6: error Isaac Sim 6 child: AttributeError: 'str' object has no attribute 'values'
- `calvin.base_table_b` (`roboverse_pack/tasks/calvin/scene_B.py`): MuJoCo: USD-only asset; Isaac 6: error Isaac Sim 6 child: AttributeError: 'str' object has no attribute 'values'
- `calvin.base_table_c` (`roboverse_pack/tasks/calvin/scene_C.py`): MuJoCo: USD-only asset; Isaac 6: error Isaac Sim 6 child: AttributeError: 'str' object has no attribute 'values'
- `calvin.base_table_d` (`roboverse_pack/tasks/calvin/scene_D.py`): MuJoCo: USD-only asset; Isaac 6: error Isaac Sim 6 child: AttributeError: 'str' object has no attribute 'values'

### embodiedgen

- `embodiedgen.put_banana` (`roboverse_pack/tasks/embodiedgen/put_banana.py`): error: KeyError: "Invalid name 'box/box_joint'. Valid names: ['banana/', 'book/', 'box_base/box_joint', 'franka/panda_finger_joint1', 'frank…

### humanoid

- `agibot_a2.walk_agibot_a2_dof12` (`roboverse_pack/tasks/humanoid/locomotion/walk_agibot_a2_dof12.py`): error: Exception: File roboverse_data/robots/agibot_a2/mjcf/agibot_a2_dof12.mjcf neither exists in the local directory nor exists in the Hug…

### libero

- `libero.pick_alphabet_soup` (`roboverse_pack/tasks/libero/libero_pick_alphabet_soup.py`): low: replay 0.00
- `libero.pick_bbq_sauce` (`roboverse_pack/tasks/libero/libero_pick_bbq_sauce.py`): low:  [20 episodes]
- `libero.pick_chocolate_pudding` (`roboverse_pack/tasks/libero/libero_pick_chocolate_pudding.py`): low: replay 0.00
- `libero.pick_cream_cheese` (`roboverse_pack/tasks/libero/libero_pick_cream_cheese.py`): low:  [20 episodes]
- `libero.pick_ketchup` (`roboverse_pack/tasks/libero/libero_pick_ketchup.py`): low: replay 0.00
- `libero.pick_milk` (`roboverse_pack/tasks/libero/libero_pick_milk.py`): low: replay 0.00
- `libero.pick_orange_juice` (`roboverse_pack/tasks/libero/libero_pick_orange_juice.py`): low:  [20 episodes]
- `libero.pick_salad_dressing` (`roboverse_pack/tasks/libero/libero_pick_salad_dressing.py`): low:  [20 episodes]
- `libero.pick_tomato_sauce` (`roboverse_pack/tasks/libero/libero_pick_tomato_sauce.py`): low:  [20 episodes]
- `libero_native.10__kitchen_scene3_turn_on_the_stove_and_put_the_moka_pot_on_it` (`roboverse_pack/tasks/libero/native_libero.py`): no class-level scenario: cannot be built by the harness
- `libero_native.10__kitchen_scene4_put_the_black_bowl_in_the_bottom_drawer_of_the_cabinet_and_close_it` (`roboverse_pack/tasks/libero/native_libero.py`): no class-level scenario: cannot be built by the harness
- `libero_native.10__kitchen_scene6_put_the_yellow_and_white_mug_in_the_microwave_and_close_it` (`roboverse_pack/tasks/libero/native_libero.py`): no class-level scenario: cannot be built by the harness
- `libero_native.10__kitchen_scene8_put_both_moka_pots_on_the_stove` (`roboverse_pack/tasks/libero/native_libero.py`): no class-level scenario: cannot be built by the harness
- `libero_native.10__living_room_scene1_put_both_the_alphabet_soup_and_the_cream_cheese_box_in_the_basket` (`roboverse_pack/tasks/libero/native_libero.py`): no class-level scenario: cannot be built by the harness
- `libero_native.10__living_room_scene2_put_both_the_alphabet_soup_and_the_tomato_sauce_in_the_basket` (`roboverse_pack/tasks/libero/native_libero.py`): no class-level scenario: cannot be built by the harness
- `libero_native.10__living_room_scene2_put_both_the_cream_cheese_box_and_the_butter_in_the_basket` (`roboverse_pack/tasks/libero/native_libero.py`): no class-level scenario: cannot be built by the harness
- `libero_native.10__living_room_scene5_put_the_white_mug_on_the_left_plate_and_put_the_yellow_and_white_mug_on_the_right_plate` (`roboverse_pack/tasks/libero/native_libero.py`): no class-level scenario: cannot be built by the harness
- `libero_native.10__living_room_scene6_put_the_white_mug_on_the_plate_and_put_the_chocolate_pudding_to_the_right_of_the_plate` (`roboverse_pack/tasks/libero/native_libero.py`): no class-level scenario: cannot be built by the harness
- `libero_native.10__study_scene1_pick_up_the_book_and_place_it_in_the_back_compartment_of_the_caddy` (`roboverse_pack/tasks/libero/native_libero.py`): no class-level scenario: cannot be built by the harness
- `libero_native.90__kitchen_scene10_close_the_top_drawer_of_the_cabinet` (`roboverse_pack/tasks/libero/native_libero.py`): no class-level scenario: cannot be built by the harness
- `libero_native.90__kitchen_scene10_close_the_top_drawer_of_the_cabinet_and_put_the_black_bowl_on_top_of_it` (`roboverse_pack/tasks/libero/native_libero.py`): no class-level scenario: cannot be built by the harness
- `libero_native.90__kitchen_scene10_put_the_black_bowl_in_the_top_drawer_of_the_cabinet` (`roboverse_pack/tasks/libero/native_libero.py`): no class-level scenario: cannot be built by the harness
- `libero_native.90__kitchen_scene10_put_the_butter_at_the_back_in_the_top_drawer_of_the_cabinet_and_close_it` (`roboverse_pack/tasks/libero/native_libero.py`): no class-level scenario: cannot be built by the harness
- `libero_native.90__kitchen_scene10_put_the_butter_at_the_front_in_the_top_drawer_of_the_cabinet_and_close_it` (`roboverse_pack/tasks/libero/native_libero.py`): no class-level scenario: cannot be built by the harness
- `libero_native.90__kitchen_scene10_put_the_chocolate_pudding_in_the_top_drawer_of_the_cabinet_and_close_it` (`roboverse_pack/tasks/libero/native_libero.py`): no class-level scenario: cannot be built by the harness
- `libero_native.90__kitchen_scene1_open_the_bottom_drawer_of_the_cabinet` (`roboverse_pack/tasks/libero/native_libero.py`): no class-level scenario: cannot be built by the harness
- `libero_native.90__kitchen_scene1_open_the_top_drawer_of_the_cabinet` (`roboverse_pack/tasks/libero/native_libero.py`): no class-level scenario: cannot be built by the harness
- `libero_native.90__kitchen_scene1_open_the_top_drawer_of_the_cabinet_and_put_the_bowl_in_it` (`roboverse_pack/tasks/libero/native_libero.py`): no class-level scenario: cannot be built by the harness
- `libero_native.90__kitchen_scene1_put_the_black_bowl_on_the_plate` (`roboverse_pack/tasks/libero/native_libero.py`): no class-level scenario: cannot be built by the harness
- `libero_native.90__kitchen_scene1_put_the_black_bowl_on_top_of_the_cabinet` (`roboverse_pack/tasks/libero/native_libero.py`): no class-level scenario: cannot be built by the harness
- `libero_native.90__kitchen_scene2_open_the_top_drawer_of_the_cabinet` (`roboverse_pack/tasks/libero/native_libero.py`): no class-level scenario: cannot be built by the harness
- `libero_native.90__kitchen_scene2_put_the_black_bowl_at_the_back_on_the_plate` (`roboverse_pack/tasks/libero/native_libero.py`): no class-level scenario: cannot be built by the harness
- `libero_native.90__kitchen_scene2_put_the_black_bowl_at_the_front_on_the_plate` (`roboverse_pack/tasks/libero/native_libero.py`): no class-level scenario: cannot be built by the harness
- `libero_native.90__kitchen_scene2_put_the_middle_black_bowl_on_the_plate` (`roboverse_pack/tasks/libero/native_libero.py`): no class-level scenario: cannot be built by the harness
- `libero_native.90__kitchen_scene2_put_the_middle_black_bowl_on_top_of_the_cabinet` (`roboverse_pack/tasks/libero/native_libero.py`): no class-level scenario: cannot be built by the harness
- `libero_native.90__kitchen_scene2_stack_the_black_bowl_at_the_front_on_the_black_bowl_in_the_middle` (`roboverse_pack/tasks/libero/native_libero.py`): no class-level scenario: cannot be built by the harness
- `libero_native.90__kitchen_scene2_stack_the_middle_black_bowl_on_the_back_black_bowl` (`roboverse_pack/tasks/libero/native_libero.py`): no class-level scenario: cannot be built by the harness
- `libero_native.90__kitchen_scene3_put_the_frying_pan_on_the_stove` (`roboverse_pack/tasks/libero/native_libero.py`): no class-level scenario: cannot be built by the harness
- `libero_native.90__kitchen_scene3_put_the_moka_pot_on_the_stove` (`roboverse_pack/tasks/libero/native_libero.py`): no class-level scenario: cannot be built by the harness
- `libero_native.90__kitchen_scene3_turn_on_the_stove` (`roboverse_pack/tasks/libero/native_libero.py`): no class-level scenario: cannot be built by the harness
- `libero_native.90__kitchen_scene3_turn_on_the_stove_and_put_the_frying_pan_on_it` (`roboverse_pack/tasks/libero/native_libero.py`): no class-level scenario: cannot be built by the harness
- `libero_native.90__kitchen_scene4_close_the_bottom_drawer_of_the_cabinet` (`roboverse_pack/tasks/libero/native_libero.py`): no class-level scenario: cannot be built by the harness
- `libero_native.90__kitchen_scene4_close_the_bottom_drawer_of_the_cabinet_and_open_the_top_drawer` (`roboverse_pack/tasks/libero/native_libero.py`): no class-level scenario: cannot be built by the harness
- `libero_native.90__kitchen_scene4_put_the_black_bowl_in_the_bottom_drawer_of_the_cabinet` (`roboverse_pack/tasks/libero/native_libero.py`): no class-level scenario: cannot be built by the harness
- `libero_native.90__kitchen_scene4_put_the_black_bowl_on_top_of_the_cabinet` (`roboverse_pack/tasks/libero/native_libero.py`): no class-level scenario: cannot be built by the harness
- `libero_native.90__kitchen_scene4_put_the_wine_bottle_in_the_bottom_drawer_of_the_cabinet` (`roboverse_pack/tasks/libero/native_libero.py`): no class-level scenario: cannot be built by the harness
- `libero_native.90__kitchen_scene4_put_the_wine_bottle_on_the_wine_rack` (`roboverse_pack/tasks/libero/native_libero.py`): no class-level scenario: cannot be built by the harness
- `libero_native.90__kitchen_scene5_close_the_top_drawer_of_the_cabinet` (`roboverse_pack/tasks/libero/native_libero.py`): no class-level scenario: cannot be built by the harness
- `libero_native.90__kitchen_scene5_put_the_black_bowl_in_the_top_drawer_of_the_cabinet` (`roboverse_pack/tasks/libero/native_libero.py`): no class-level scenario: cannot be built by the harness
- `libero_native.90__kitchen_scene5_put_the_black_bowl_on_the_plate` (`roboverse_pack/tasks/libero/native_libero.py`): no class-level scenario: cannot be built by the harness
- `libero_native.90__kitchen_scene5_put_the_black_bowl_on_top_of_the_cabinet` (`roboverse_pack/tasks/libero/native_libero.py`): no class-level scenario: cannot be built by the harness
- `libero_native.90__kitchen_scene5_put_the_ketchup_in_the_top_drawer_of_the_cabinet` (`roboverse_pack/tasks/libero/native_libero.py`): no class-level scenario: cannot be built by the harness
- `libero_native.90__kitchen_scene6_close_the_microwave` (`roboverse_pack/tasks/libero/native_libero.py`): no class-level scenario: cannot be built by the harness
- `libero_native.90__kitchen_scene6_put_the_yellow_and_white_mug_to_the_front_of_the_white_mug` (`roboverse_pack/tasks/libero/native_libero.py`): no class-level scenario: cannot be built by the harness
- `libero_native.90__kitchen_scene7_open_the_microwave` (`roboverse_pack/tasks/libero/native_libero.py`): no class-level scenario: cannot be built by the harness
- `libero_native.90__kitchen_scene7_put_the_white_bowl_on_the_plate` (`roboverse_pack/tasks/libero/native_libero.py`): no class-level scenario: cannot be built by the harness
- `libero_native.90__kitchen_scene7_put_the_white_bowl_to_the_right_of_the_plate` (`roboverse_pack/tasks/libero/native_libero.py`): no class-level scenario: cannot be built by the harness
- `libero_native.90__kitchen_scene8_put_the_right_moka_pot_on_the_stove` (`roboverse_pack/tasks/libero/native_libero.py`): no class-level scenario: cannot be built by the harness
- `libero_native.90__kitchen_scene8_turn_off_the_stove` (`roboverse_pack/tasks/libero/native_libero.py`): no class-level scenario: cannot be built by the harness
- `libero_native.90__kitchen_scene9_put_the_frying_pan_on_the_cabinet_shelf` (`roboverse_pack/tasks/libero/native_libero.py`): no class-level scenario: cannot be built by the harness
- `libero_native.90__kitchen_scene9_put_the_frying_pan_on_top_of_the_cabinet` (`roboverse_pack/tasks/libero/native_libero.py`): no class-level scenario: cannot be built by the harness
- `libero_native.90__kitchen_scene9_put_the_frying_pan_under_the_cabinet_shelf` (`roboverse_pack/tasks/libero/native_libero.py`): no class-level scenario: cannot be built by the harness
- `libero_native.90__kitchen_scene9_put_the_white_bowl_on_top_of_the_cabinet` (`roboverse_pack/tasks/libero/native_libero.py`): no class-level scenario: cannot be built by the harness
- `libero_native.90__kitchen_scene9_turn_on_the_stove` (`roboverse_pack/tasks/libero/native_libero.py`): no class-level scenario: cannot be built by the harness
- `libero_native.90__kitchen_scene9_turn_on_the_stove_and_put_the_frying_pan_on_it` (`roboverse_pack/tasks/libero/native_libero.py`): no class-level scenario: cannot be built by the harness
- `libero_native.90__living_room_scene1_pick_up_the_alphabet_soup_and_put_it_in_the_basket` (`roboverse_pack/tasks/libero/native_libero.py`): no class-level scenario: cannot be built by the harness
- `libero_native.90__living_room_scene1_pick_up_the_cream_cheese_box_and_put_it_in_the_basket` (`roboverse_pack/tasks/libero/native_libero.py`): no class-level scenario: cannot be built by the harness
- `libero_native.90__living_room_scene1_pick_up_the_ketchup_and_put_it_in_the_basket` (`roboverse_pack/tasks/libero/native_libero.py`): no class-level scenario: cannot be built by the harness
- `libero_native.90__living_room_scene1_pick_up_the_tomato_sauce_and_put_it_in_the_basket` (`roboverse_pack/tasks/libero/native_libero.py`): no class-level scenario: cannot be built by the harness
- `libero_native.90__living_room_scene2_pick_up_the_alphabet_soup_and_put_it_in_the_basket` (`roboverse_pack/tasks/libero/native_libero.py`): no class-level scenario: cannot be built by the harness
- `libero_native.90__living_room_scene2_pick_up_the_butter_and_put_it_in_the_basket` (`roboverse_pack/tasks/libero/native_libero.py`): no class-level scenario: cannot be built by the harness
- `libero_native.90__living_room_scene2_pick_up_the_milk_and_put_it_in_the_basket` (`roboverse_pack/tasks/libero/native_libero.py`): no class-level scenario: cannot be built by the harness
- `libero_native.90__living_room_scene2_pick_up_the_orange_juice_and_put_it_in_the_basket` (`roboverse_pack/tasks/libero/native_libero.py`): no class-level scenario: cannot be built by the harness
- `libero_native.90__living_room_scene2_pick_up_the_tomato_sauce_and_put_it_in_the_basket` (`roboverse_pack/tasks/libero/native_libero.py`): no class-level scenario: cannot be built by the harness
- `libero_native.90__living_room_scene3_pick_up_the_alphabet_soup_and_put_it_in_the_tray` (`roboverse_pack/tasks/libero/native_libero.py`): no class-level scenario: cannot be built by the harness
- `libero_native.90__living_room_scene3_pick_up_the_butter_and_put_it_in_the_tray` (`roboverse_pack/tasks/libero/native_libero.py`): no class-level scenario: cannot be built by the harness
- `libero_native.90__living_room_scene3_pick_up_the_cream_cheese_and_put_it_in_the_tray` (`roboverse_pack/tasks/libero/native_libero.py`): no class-level scenario: cannot be built by the harness
- `libero_native.90__living_room_scene3_pick_up_the_ketchup_and_put_it_in_the_tray` (`roboverse_pack/tasks/libero/native_libero.py`): no class-level scenario: cannot be built by the harness
- `libero_native.90__living_room_scene3_pick_up_the_tomato_sauce_and_put_it_in_the_tray` (`roboverse_pack/tasks/libero/native_libero.py`): no class-level scenario: cannot be built by the harness
- `libero_native.90__living_room_scene4_pick_up_the_black_bowl_on_the_left_and_put_it_in_the_tray` (`roboverse_pack/tasks/libero/native_libero.py`): no class-level scenario: cannot be built by the harness
- `libero_native.90__living_room_scene4_pick_up_the_chocolate_pudding_and_put_it_in_the_tray` (`roboverse_pack/tasks/libero/native_libero.py`): no class-level scenario: cannot be built by the harness
- `libero_native.90__living_room_scene4_pick_up_the_salad_dressing_and_put_it_in_the_tray` (`roboverse_pack/tasks/libero/native_libero.py`): no class-level scenario: cannot be built by the harness
- `libero_native.90__living_room_scene4_stack_the_left_bowl_on_the_right_bowl_and_place_them_in_the_tray` (`roboverse_pack/tasks/libero/native_libero.py`): no class-level scenario: cannot be built by the harness
- `libero_native.90__living_room_scene4_stack_the_right_bowl_on_the_left_bowl_and_place_them_in_the_tray` (`roboverse_pack/tasks/libero/native_libero.py`): no class-level scenario: cannot be built by the harness
- `libero_native.90__living_room_scene5_put_the_red_mug_on_the_left_plate` (`roboverse_pack/tasks/libero/native_libero.py`): no class-level scenario: cannot be built by the harness
- `libero_native.90__living_room_scene5_put_the_red_mug_on_the_right_plate` (`roboverse_pack/tasks/libero/native_libero.py`): no class-level scenario: cannot be built by the harness
- `libero_native.90__living_room_scene5_put_the_white_mug_on_the_left_plate` (`roboverse_pack/tasks/libero/native_libero.py`): no class-level scenario: cannot be built by the harness
- `libero_native.90__living_room_scene5_put_the_yellow_and_white_mug_on_the_right_plate` (`roboverse_pack/tasks/libero/native_libero.py`): no class-level scenario: cannot be built by the harness
- `libero_native.90__living_room_scene6_put_the_chocolate_pudding_to_the_left_of_the_plate` (`roboverse_pack/tasks/libero/native_libero.py`): no class-level scenario: cannot be built by the harness
- `libero_native.90__living_room_scene6_put_the_chocolate_pudding_to_the_right_of_the_plate` (`roboverse_pack/tasks/libero/native_libero.py`): no class-level scenario: cannot be built by the harness
- `libero_native.90__living_room_scene6_put_the_red_mug_on_the_plate` (`roboverse_pack/tasks/libero/native_libero.py`): no class-level scenario: cannot be built by the harness
- `libero_native.90__living_room_scene6_put_the_white_mug_on_the_plate` (`roboverse_pack/tasks/libero/native_libero.py`): no class-level scenario: cannot be built by the harness
- `libero_native.90__study_scene1_pick_up_the_book_and_place_it_in_the_front_compartment_of_the_caddy` (`roboverse_pack/tasks/libero/native_libero.py`): no class-level scenario: cannot be built by the harness
- `libero_native.90__study_scene1_pick_up_the_book_and_place_it_in_the_left_compartment_of_the_caddy` (`roboverse_pack/tasks/libero/native_libero.py`): no class-level scenario: cannot be built by the harness
- `libero_native.90__study_scene1_pick_up_the_book_and_place_it_in_the_right_compartment_of_the_caddy` (`roboverse_pack/tasks/libero/native_libero.py`): no class-level scenario: cannot be built by the harness
- `libero_native.90__study_scene1_pick_up_the_yellow_and_white_mug_and_place_it_to_the_right_of_the_caddy` (`roboverse_pack/tasks/libero/native_libero.py`): no class-level scenario: cannot be built by the harness
- `libero_native.90__study_scene2_pick_up_the_book_and_place_it_in_the_back_compartment_of_the_caddy` (`roboverse_pack/tasks/libero/native_libero.py`): no class-level scenario: cannot be built by the harness
- `libero_native.90__study_scene2_pick_up_the_book_and_place_it_in_the_front_compartment_of_the_caddy` (`roboverse_pack/tasks/libero/native_libero.py`): no class-level scenario: cannot be built by the harness
- `libero_native.90__study_scene2_pick_up_the_book_and_place_it_in_the_left_compartment_of_the_caddy` (`roboverse_pack/tasks/libero/native_libero.py`): no class-level scenario: cannot be built by the harness
- `libero_native.90__study_scene2_pick_up_the_book_and_place_it_in_the_right_compartment_of_the_caddy` (`roboverse_pack/tasks/libero/native_libero.py`): no class-level scenario: cannot be built by the harness
- `libero_native.90__study_scene3_pick_up_the_book_and_place_it_in_the_front_compartment_of_the_caddy` (`roboverse_pack/tasks/libero/native_libero.py`): no class-level scenario: cannot be built by the harness
- `libero_native.90__study_scene3_pick_up_the_book_and_place_it_in_the_left_compartment_of_the_caddy` (`roboverse_pack/tasks/libero/native_libero.py`): no class-level scenario: cannot be built by the harness
- `libero_native.90__study_scene3_pick_up_the_book_and_place_it_in_the_right_compartment_of_the_caddy` (`roboverse_pack/tasks/libero/native_libero.py`): no class-level scenario: cannot be built by the harness
- `libero_native.90__study_scene3_pick_up_the_red_mug_and_place_it_to_the_right_of_the_caddy` (`roboverse_pack/tasks/libero/native_libero.py`): no class-level scenario: cannot be built by the harness
- `libero_native.90__study_scene3_pick_up_the_white_mug_and_place_it_to_the_right_of_the_caddy` (`roboverse_pack/tasks/libero/native_libero.py`): no class-level scenario: cannot be built by the harness
- `libero_native.90__study_scene4_pick_up_the_book_in_the_middle_and_place_it_on_the_cabinet_shelf` (`roboverse_pack/tasks/libero/native_libero.py`): no class-level scenario: cannot be built by the harness
- `libero_native.90__study_scene4_pick_up_the_book_on_the_left_and_place_it_on_top_of_the_shelf` (`roboverse_pack/tasks/libero/native_libero.py`): no class-level scenario: cannot be built by the harness
- `libero_native.90__study_scene4_pick_up_the_book_on_the_right_and_place_it_on_the_cabinet_shelf` (`roboverse_pack/tasks/libero/native_libero.py`): no class-level scenario: cannot be built by the harness
- `libero_native.90__study_scene4_pick_up_the_book_on_the_right_and_place_it_under_the_cabinet_shelf` (`roboverse_pack/tasks/libero/native_libero.py`): no class-level scenario: cannot be built by the harness
- `libero_native.goal__open_the_middle_drawer_of_the_cabinet` (`roboverse_pack/tasks/libero/native_libero.py`): no class-level scenario: cannot be built by the harness
- `libero_native.goal__open_the_top_drawer_and_put_the_bowl_inside` (`roboverse_pack/tasks/libero/native_libero.py`): no class-level scenario: cannot be built by the harness
- `libero_native.goal__push_the_plate_to_the_front_of_the_stove` (`roboverse_pack/tasks/libero/native_libero.py`): no class-level scenario: cannot be built by the harness
- `libero_native.goal__put_the_bowl_on_the_plate` (`roboverse_pack/tasks/libero/native_libero.py`): no class-level scenario: cannot be built by the harness
- `libero_native.goal__put_the_bowl_on_the_stove` (`roboverse_pack/tasks/libero/native_libero.py`): no class-level scenario: cannot be built by the harness
- `libero_native.goal__put_the_bowl_on_top_of_the_cabinet` (`roboverse_pack/tasks/libero/native_libero.py`): no class-level scenario: cannot be built by the harness
- `libero_native.goal__put_the_cream_cheese_in_the_bowl` (`roboverse_pack/tasks/libero/native_libero.py`): no class-level scenario: cannot be built by the harness
- `libero_native.goal__put_the_wine_bottle_on_the_rack` (`roboverse_pack/tasks/libero/native_libero.py`): no class-level scenario: cannot be built by the harness
- `libero_native.goal__put_the_wine_bottle_on_top_of_the_cabinet` (`roboverse_pack/tasks/libero/native_libero.py`): no class-level scenario: cannot be built by the harness
- `libero_native.goal__turn_on_the_stove` (`roboverse_pack/tasks/libero/native_libero.py`): no class-level scenario: cannot be built by the harness
- `libero_native.object__pick_up_the_alphabet_soup_and_place_it_in_the_basket` (`roboverse_pack/tasks/libero/native_libero.py`): no class-level scenario: cannot be built by the harness
- `libero_native.object__pick_up_the_bbq_sauce_and_place_it_in_the_basket` (`roboverse_pack/tasks/libero/native_libero.py`): no class-level scenario: cannot be built by the harness
- `libero_native.object__pick_up_the_butter_and_place_it_in_the_basket` (`roboverse_pack/tasks/libero/native_libero.py`): no class-level scenario: cannot be built by the harness
- `libero_native.object__pick_up_the_chocolate_pudding_and_place_it_in_the_basket` (`roboverse_pack/tasks/libero/native_libero.py`): no class-level scenario: cannot be built by the harness
- `libero_native.object__pick_up_the_cream_cheese_and_place_it_in_the_basket` (`roboverse_pack/tasks/libero/native_libero.py`): no class-level scenario: cannot be built by the harness
- `libero_native.object__pick_up_the_ketchup_and_place_it_in_the_basket` (`roboverse_pack/tasks/libero/native_libero.py`): no class-level scenario: cannot be built by the harness
- `libero_native.object__pick_up_the_milk_and_place_it_in_the_basket` (`roboverse_pack/tasks/libero/native_libero.py`): no class-level scenario: cannot be built by the harness
- `libero_native.object__pick_up_the_orange_juice_and_place_it_in_the_basket` (`roboverse_pack/tasks/libero/native_libero.py`): no class-level scenario: cannot be built by the harness
- `libero_native.object__pick_up_the_salad_dressing_and_place_it_in_the_basket` (`roboverse_pack/tasks/libero/native_libero.py`): no class-level scenario: cannot be built by the harness
- `libero_native.object__pick_up_the_tomato_sauce_and_place_it_in_the_basket` (`roboverse_pack/tasks/libero/native_libero.py`): no class-level scenario: cannot be built by the harness
- `libero_native.spatial__pick_up_the_black_bowl_between_the_plate_and_the_ramekin_and_place_it_on_the_plate` (`roboverse_pack/tasks/libero/native_libero.py`): no class-level scenario: cannot be built by the harness
- `libero_native.spatial__pick_up_the_black_bowl_from_table_center_and_place_it_on_the_plate` (`roboverse_pack/tasks/libero/native_libero.py`): no class-level scenario: cannot be built by the harness
- `libero_native.spatial__pick_up_the_black_bowl_in_the_top_drawer_of_the_wooden_cabinet_and_place_it_on_the_plate` (`roboverse_pack/tasks/libero/native_libero.py`): no class-level scenario: cannot be built by the harness
- `libero_native.spatial__pick_up_the_black_bowl_next_to_the_cookie_box_and_place_it_on_the_plate` (`roboverse_pack/tasks/libero/native_libero.py`): no class-level scenario: cannot be built by the harness
- `libero_native.spatial__pick_up_the_black_bowl_next_to_the_plate_and_place_it_on_the_plate` (`roboverse_pack/tasks/libero/native_libero.py`): no class-level scenario: cannot be built by the harness
- `libero_native.spatial__pick_up_the_black_bowl_next_to_the_ramekin_and_place_it_on_the_plate` (`roboverse_pack/tasks/libero/native_libero.py`): no class-level scenario: cannot be built by the harness
- `libero_native.spatial__pick_up_the_black_bowl_on_the_cookie_box_and_place_it_on_the_plate` (`roboverse_pack/tasks/libero/native_libero.py`): no class-level scenario: cannot be built by the harness
- `libero_native.spatial__pick_up_the_black_bowl_on_the_ramekin_and_place_it_on_the_plate` (`roboverse_pack/tasks/libero/native_libero.py`): no class-level scenario: cannot be built by the harness
- `libero_native.spatial__pick_up_the_black_bowl_on_the_stove_and_place_it_on_the_plate` (`roboverse_pack/tasks/libero/native_libero.py`): no class-level scenario: cannot be built by the harness
- `libero_native.spatial__pick_up_the_black_bowl_on_the_wooden_cabinet_and_place_it_on_the_plate` (`roboverse_pack/tasks/libero/native_libero.py`): no class-level scenario: cannot be built by the harness
- `libero_plus_native.add__10__kitchen_scene3_turn_on_the_stove_and_put_the_moka_pot_on_it_add_1` (`roboverse_pack/tasks/libero/native_liberoplus.py`): no class-level scenario: cannot be built by the harness
- `libero_plus_native.add__10__kitchen_scene3_turn_on_the_stove_and_put_the_moka_pot_on_it_add_10` (`roboverse_pack/tasks/libero/native_liberoplus.py`): no class-level scenario: cannot be built by the harness
- `libero_plus_native.add__goal__open_the_middle_drawer_of_the_cabinet_add_1` (`roboverse_pack/tasks/libero/native_liberoplus.py`): no class-level scenario: cannot be built by the harness
- `libero_plus_native.add__object__pick_up_the_alphabet_soup_and_place_it_in_the_basket_add_1` (`roboverse_pack/tasks/libero/native_liberoplus.py`): no class-level scenario: cannot be built by the harness
- `libero_plus_native.add__object__pick_up_the_alphabet_soup_and_place_it_in_the_basket_add_10` (`roboverse_pack/tasks/libero/native_liberoplus.py`): no class-level scenario: cannot be built by the harness
- `libero_plus_native.add__spatial__pick_up_the_black_bowl_between_the_plate_and_the_ramekin_and_place_it_on_the_plate_add_1` (`roboverse_pack/tasks/libero/native_liberoplus.py`): no class-level scenario: cannot be built by the harness
- `libero_plus_native.add__spatial__pick_up_the_black_bowl_between_the_plate_and_the_ramekin_and_place_it_on_the_plate_add_10` (`roboverse_pack/tasks/libero/native_liberoplus.py`): no class-level scenario: cannot be built by the harness
- `libero_plus_native.language__10__kitchen_scene3_turn_on_the_stove_and_put_the_moka_pot_on_it_language_1` (`roboverse_pack/tasks/libero/native_liberoplus.py`): no class-level scenario: cannot be built by the harness
- `libero_plus_native.language__10__kitchen_scene3_turn_on_the_stove_and_put_the_moka_pot_on_it_language_10` (`roboverse_pack/tasks/libero/native_liberoplus.py`): no class-level scenario: cannot be built by the harness
- `libero_plus_native.language__goal__open_the_middle_drawer_of_the_cabinet_language_1` (`roboverse_pack/tasks/libero/native_liberoplus.py`): no class-level scenario: cannot be built by the harness
- `libero_plus_native.language__goal__open_the_middle_drawer_of_the_cabinet_language_10` (`roboverse_pack/tasks/libero/native_liberoplus.py`): no class-level scenario: cannot be built by the harness
- `libero_plus_native.language__object__pick_up_the_alphabet_soup_and_place_it_in_the_basket_language_1` (`roboverse_pack/tasks/libero/native_liberoplus.py`): no class-level scenario: cannot be built by the harness
- `libero_plus_native.language__object__pick_up_the_alphabet_soup_and_place_it_in_the_basket_language_10` (`roboverse_pack/tasks/libero/native_liberoplus.py`): no class-level scenario: cannot be built by the harness
- `libero_plus_native.language__spatial__pick_up_the_black_bowl_between_the_plate_and_the_ramekin_and_place_it_on_the_plate_language_1` (`roboverse_pack/tasks/libero/native_liberoplus.py`): no class-level scenario: cannot be built by the harness
- `libero_plus_native.language__spatial__pick_up_the_black_bowl_between_the_plate_and_the_ramekin_and_place_it_on_the_plate_language_10` (`roboverse_pack/tasks/libero/native_liberoplus.py`): no class-level scenario: cannot be built by the harness
- `libero_plus_native.light__10__kitchen_scene3_turn_on_the_stove_and_put_the_moka_pot_on_it_light_1` (`roboverse_pack/tasks/libero/native_liberoplus.py`): no class-level scenario: cannot be built by the harness
- `libero_plus_native.light__10__kitchen_scene3_turn_on_the_stove_and_put_the_moka_pot_on_it_light_10` (`roboverse_pack/tasks/libero/native_liberoplus.py`): no class-level scenario: cannot be built by the harness
- `libero_plus_native.light__goal__open_the_middle_drawer_of_the_cabinet_light_1` (`roboverse_pack/tasks/libero/native_liberoplus.py`): no class-level scenario: cannot be built by the harness
- `libero_plus_native.light__goal__open_the_middle_drawer_of_the_cabinet_light_10` (`roboverse_pack/tasks/libero/native_liberoplus.py`): no class-level scenario: cannot be built by the harness
- `libero_plus_native.light__object__pick_up_the_alphabet_soup_and_place_it_in_the_basket_light_1` (`roboverse_pack/tasks/libero/native_liberoplus.py`): no class-level scenario: cannot be built by the harness
- `libero_plus_native.light__object__pick_up_the_alphabet_soup_and_place_it_in_the_basket_light_10` (`roboverse_pack/tasks/libero/native_liberoplus.py`): no class-level scenario: cannot be built by the harness
- `libero_plus_native.light__spatial__pick_up_the_black_bowl_between_the_plate_and_the_ramekin_and_place_it_on_the_plate_light_1` (`roboverse_pack/tasks/libero/native_liberoplus.py`): no class-level scenario: cannot be built by the harness
- `libero_plus_native.light__spatial__pick_up_the_black_bowl_between_the_plate_and_the_ramekin_and_place_it_on_the_plate_light_10` (`roboverse_pack/tasks/libero/native_liberoplus.py`): no class-level scenario: cannot be built by the harness
- `libero_plus_native.table__10__kitchen_scene3_turn_on_the_stove_and_put_the_moka_pot_on_it_table_1` (`roboverse_pack/tasks/libero/native_liberoplus.py`): no class-level scenario: cannot be built by the harness
- `libero_plus_native.table__10__kitchen_scene3_turn_on_the_stove_and_put_the_moka_pot_on_it_table_10` (`roboverse_pack/tasks/libero/native_liberoplus.py`): no class-level scenario: cannot be built by the harness
- `libero_plus_native.table__goal__open_the_middle_drawer_of_the_cabinet_table_1` (`roboverse_pack/tasks/libero/native_liberoplus.py`): no class-level scenario: cannot be built by the harness
- `libero_plus_native.table__goal__open_the_middle_drawer_of_the_cabinet_table_10` (`roboverse_pack/tasks/libero/native_liberoplus.py`): no class-level scenario: cannot be built by the harness
- `libero_plus_native.table__object__pick_up_the_alphabet_soup_and_place_it_in_the_basket_table_1` (`roboverse_pack/tasks/libero/native_liberoplus.py`): no class-level scenario: cannot be built by the harness
- `libero_plus_native.table__object__pick_up_the_alphabet_soup_and_place_it_in_the_basket_table_10` (`roboverse_pack/tasks/libero/native_liberoplus.py`): no class-level scenario: cannot be built by the harness
- `libero_plus_native.table__spatial__pick_up_the_black_bowl_between_the_plate_and_the_ramekin_and_place_it_on_the_plate_table_1` (`roboverse_pack/tasks/libero/native_liberoplus.py`): no class-level scenario: cannot be built by the harness
- `libero_plus_native.table__spatial__pick_up_the_black_bowl_between_the_plate_and_the_ramekin_and_place_it_on_the_plate_table_10` (`roboverse_pack/tasks/libero/native_liberoplus.py`): no class-level scenario: cannot be built by the harness
- `libero_plus_native.tb__10__kitchen_scene3_turn_on_the_stove_and_put_the_moka_pot_on_it_tb_1` (`roboverse_pack/tasks/libero/native_liberoplus.py`): no class-level scenario: cannot be built by the harness
- `libero_plus_native.tb__10__kitchen_scene3_turn_on_the_stove_and_put_the_moka_pot_on_it_tb_10` (`roboverse_pack/tasks/libero/native_liberoplus.py`): no class-level scenario: cannot be built by the harness
- `libero_plus_native.tb__goal__open_the_middle_drawer_of_the_cabinet_tb_1` (`roboverse_pack/tasks/libero/native_liberoplus.py`): no class-level scenario: cannot be built by the harness
- `libero_plus_native.tb__goal__open_the_middle_drawer_of_the_cabinet_tb_10` (`roboverse_pack/tasks/libero/native_liberoplus.py`): no class-level scenario: cannot be built by the harness
- `libero_plus_native.tb__object__pick_up_the_alphabet_soup_and_place_it_in_the_basket_tb_1` (`roboverse_pack/tasks/libero/native_liberoplus.py`): no class-level scenario: cannot be built by the harness
- `libero_plus_native.tb__object__pick_up_the_alphabet_soup_and_place_it_in_the_basket_tb_10` (`roboverse_pack/tasks/libero/native_liberoplus.py`): no class-level scenario: cannot be built by the harness
- `libero_plus_native.tb__spatial__pick_up_the_black_bowl_between_the_plate_and_the_ramekin_and_place_it_on_the_plate_tb_1` (`roboverse_pack/tasks/libero/native_liberoplus.py`): no class-level scenario: cannot be built by the harness
- `libero_plus_native.tb__spatial__pick_up_the_black_bowl_between_the_plate_and_the_ramekin_and_place_it_on_the_plate_tb_10` (`roboverse_pack/tasks/libero/native_liberoplus.py`): no class-level scenario: cannot be built by the harness

### libero_90

- `libero_90.kitchen_scene10_close_the_top_drawer_of_the_cabinet` (`roboverse_pack/tasks/libero_90/libero_kitchen_scene10_close_the_top_drawer_of_the_cabinet.py`): low:  [20 episodes]
- `libero_90.kitchen_scene10_close_the_top_drawer_of_the_cabinet_and_put_the_black_bowl_on_top_of_it` (`roboverse_pack/tasks/libero_90/libero_kitchen_scene10_close_the_top_drawer_of_the_cabinet_and_put_the_black_bowl_on_top_of_it.py`): low: replay 0.00
- `libero_90.kitchen_scene10_put_the_black_bowl_in_the_top_drawer_of_the_cabinet` (`roboverse_pack/tasks/libero_90/libero_kitchen_scene10_put_the_black_bowl_in_the_top_drawer_of_the_cabinet.py`): low: replay 0.00
- `libero_90.kitchen_scene10_put_the_butter_at_the_back_in_the_top_drawer_of_the_cabinet_and_close_it` (`roboverse_pack/tasks/libero_90/libero_kitchen_scene10_put_the_butter_at_the_back_in_the_top_drawer_of_the_cabinet_and_close_it.py`): low: replay 0.00
- `libero_90.kitchen_scene10_put_the_butter_at_the_front_in_the_top_drawer_of_the_cabinet_and_close_it` (`roboverse_pack/tasks/libero_90/libero_kitchen_scene10_put_the_butter_at_the_front_in_the_top_drawer_of_the_cabinet_and_close_it.py`): low: replay 0.00
- `libero_90.kitchen_scene10_put_the_chocolate_pudding_in_the_top_drawer_of_the_cabinet_and_close_it` (`roboverse_pack/tasks/libero_90/libero_kitchen_scene10_put_the_chocolate_pudding_in_the_top_drawer_of_the_cabinet_and_close_it.py`): low: replay 0.00
- `libero_90.kitchen_scene1_open_bottom_drawer` (`roboverse_pack/tasks/libero_90/libero_kitchen_scene1_open_bottom_drawer.py`): low:  [20 episodes]
- `libero_90.kitchen_scene1_open_drawer_put_bowl` (`roboverse_pack/tasks/libero_90/libero_kitchen_scene1_open_drawer_put_bowl.py`): low: replay 0.00
- `libero_90.kitchen_scene1_open_top_drawer` (`roboverse_pack/tasks/libero_90/libero_kitchen_scene1_open_top_drawer.py`): low:  [20 episodes]
- `libero_90.kitchen_scene1_put_the_black_bowl_on_the_plate` (`roboverse_pack/tasks/libero_90/libero_kitchen_scene1_put_the_black_bowl_on_the_plate.py`): low: replay 0.00
- `libero_90.kitchen_scene1_put_the_black_bowl_on_top_of_the_cabinet` (`roboverse_pack/tasks/libero_90/libero_kitchen_scene1_put_the_black_bowl_on_top_of_the_cabinet.py`): low: replay 0.00
- `libero_90.kitchen_scene2_open_the_top_drawer_of_the_cabinet` (`roboverse_pack/tasks/libero_90/libero_kitchen_scene2_open_the_top_drawer_of_the_cabinet.py`): low:  [20 episodes]
- `libero_90.kitchen_scene2_put_the_black_bowl_at_the_back_on_the_plate` (`roboverse_pack/tasks/libero_90/libero_kitchen_scene2_put_the_black_bowl_at_the_back_on_the_plate.py`): low: replay 0.00
- `libero_90.kitchen_scene2_put_the_black_bowl_at_the_front_on_the_plate` (`roboverse_pack/tasks/libero_90/libero_kitchen_scene2_put_the_black_bowl_at_the_front_on_the_plate.py`): low: replay 0.00
- `libero_90.kitchen_scene2_put_the_black_bowl_in_the_middle_on_the_plate` (`roboverse_pack/tasks/libero_90/libero_kitchen_scene2_put_the_black_bowl_in_the_middle_on_the_plate.py`): low: replay 0.00
- `libero_90.kitchen_scene2_put_the_middle_black_bowl_on_top_of_the_cabinet` (`roboverse_pack/tasks/libero_90/libero_kitchen_scene2_put_the_middle_black_bowl_on_top_of_the_cabinet.py`): low: replay 0.00
- `libero_90.kitchen_scene2_stack_the_black_bowl_at_the_front_on_the_black_bowl_in_the_middle` (`roboverse_pack/tasks/libero_90/libero_kitchen_scene2_stack_the_black_bowl_at_the_front_on_the_black_bowl_in_the_middle.py`): low:  [20 episodes]
- `libero_90.kitchen_scene2_stack_the_middle_black_bowl_on_the_back_black_bowl` (`roboverse_pack/tasks/libero_90/libero_kitchen_scene2_stack_the_middle_black_bowl_on_the_back_black_bowl.py`): low: replay 0.00
- `libero_90.kitchen_scene3_put_the_frying_pan_on_the_stove` (`roboverse_pack/tasks/libero_90/libero_kitchen_scene3_put_the_frying_pan_on_the_stove.py`): low: replay 0.00
- `libero_90.kitchen_scene3_put_the_moka_pot_on_the_stove` (`roboverse_pack/tasks/libero_90/libero_kitchen_scene3_put_the_moka_pot_on_the_stove.py`): low:  [20 episodes]
- `libero_90.kitchen_scene3_turn_on_the_stove` (`roboverse_pack/tasks/libero_90/libero_kitchen_scene3_turn_on_the_stove.py`): low:  [20 episodes]
- `libero_90.kitchen_scene3_turn_on_the_stove_and_put_the_frying_pan_on_it` (`roboverse_pack/tasks/libero_90/libero_kitchen_scene3_turn_on_the_stove_and_put_the_frying_pan_on_it.py`): low: replay 0.00
- `libero_90.kitchen_scene4_close_the_bottom_drawer_of_the_cabinet_and_open_the_top_drawer` (`roboverse_pack/tasks/libero_90/libero_kitchen_scene4_close_the_bottom_drawer_of_the_cabinet_and_open_the_top_drawer.py`): low:  [20 episodes]
- `libero_90.kitchen_scene4_put_the_black_bowl_in_the_bottom_drawer_of_the_cabinet` (`roboverse_pack/tasks/libero_90/libero_kitchen_scene4_put_the_black_bowl_in_the_bottom_drawer_of_the_cabinet.py`): low: replay 0.00
- `libero_90.kitchen_scene4_put_the_black_bowl_on_top_of_the_cabinet` (`roboverse_pack/tasks/libero_90/libero_kitchen_scene4_put_the_black_bowl_on_top_of_the_cabinet.py`): low: replay 0.00
- `libero_90.kitchen_scene4_put_the_wine_bottle_in_the_bottom_drawer_of_the_cabinet` (`roboverse_pack/tasks/libero_90/libero_kitchen_scene4_put_the_wine_bottle_in_the_bottom_drawer_of_the_cabinet.py`): low: replay 0.00
- `libero_90.kitchen_scene4_put_the_wine_bottle_on_the_wine_rack` (`roboverse_pack/tasks/libero_90/libero_kitchen_scene4_put_the_wine_bottle_on_the_wine_rack.py`): low: replay 0.00
- `libero_90.kitchen_scene5_close_the_top_drawer_of_the_cabinet` (`roboverse_pack/tasks/libero_90/libero_kitchen_scene5_close_the_top_drawer_of_the_cabinet.py`): low:  [20 episodes]
- `libero_90.kitchen_scene5_put_the_black_bowl_in_the_top_drawer_of_the_cabinet` (`roboverse_pack/tasks/libero_90/libero_kitchen_scene5_put_the_black_bowl_in_the_top_drawer_of_the_cabinet.py`): low: replay 0.00
- `libero_90.kitchen_scene5_put_the_black_bowl_on_the_plate` (`roboverse_pack/tasks/libero_90/libero_kitchen_scene5_put_the_black_bowl_on_the_plate.py`): low: replay 0.00
- `libero_90.kitchen_scene5_put_the_black_bowl_on_top_of_the_cabinet` (`roboverse_pack/tasks/libero_90/libero_kitchen_scene5_put_the_black_bowl_on_top_of_the_cabinet.py`): low:  [20 episodes]
- `libero_90.kitchen_scene5_put_the_ketchup_in_the_top_drawer_of_the_cabinet` (`roboverse_pack/tasks/libero_90/libero_kitchen_scene5_put_the_ketchup_in_the_top_drawer_of_the_cabinet.py`): low: replay 0.00
- `libero_90.kitchen_scene6_close_the_microwave` (`roboverse_pack/tasks/libero_90/libero_kitchen_scene6_close_the_microwave.py`): error: Exception: File roboverse_data/assets/libero/COMMON/turbosquid_objects/porcelain_mug/mjcf/porcelain_mug.xml neither exists in the loc…
- `libero_90.kitchen_scene6_put_the_yellow_and_white_mug_to_the_front_of_the_white_mug` (`roboverse_pack/tasks/libero_90/libero_kitchen_scene6_put_the_yellow_and_white_mug_to_the_front_of_the_white_mug.py`): error: Exception: File roboverse_data/assets/libero/COMMON/turbosquid_objects/porcelain_mug/mjcf/porcelain_mug.xml neither exists in the loc…
- `libero_90.kitchen_scene7_open_the_microwave` (`roboverse_pack/tasks/libero_90/libero_kitchen_scene7_open_the_microwave.py`): low: replay 0.00
- `libero_90.kitchen_scene7_put_the_white_bowl_on_the_plate` (`roboverse_pack/tasks/libero_90/libero_kitchen_scene7_put_the_white_bowl_on_the_plate.py`): low: replay 0.00
- `libero_90.kitchen_scene7_put_the_white_bowl_to_the_right_of_the_plate` (`roboverse_pack/tasks/libero_90/libero_kitchen_scene7_put_the_white_bowl_to_the_right_of_the_plate.py`): low: replay 0.00
- `libero_90.kitchen_scene8_put_the_right_moka_pot_on_the_stove` (`roboverse_pack/tasks/libero_90/libero_kitchen_scene8_put_the_right_moka_pot_on_the_stove.py`): low: replay 0.00
- `libero_90.kitchen_scene8_turn_off_the_stove` (`roboverse_pack/tasks/libero_90/libero_kitchen_scene8_turn_off_the_stove.py`): low:  [20 episodes]
- `libero_90.kitchen_scene9_put_the_frying_pan_on_the_cabinet_shelf` (`roboverse_pack/tasks/libero_90/libero_kitchen_scene9_put_the_frying_pan_on_the_cabinet_shelf.py`): error: Exception: File roboverse_data/assets/libero/COMMON/stable_scanned_objects/white_bowl/mjcf/white_bowl.xml neither exists in the local…
- `libero_90.kitchen_scene9_put_the_frying_pan_on_top_of_the_cabinet` (`roboverse_pack/tasks/libero_90/libero_kitchen_scene9_put_the_frying_pan_on_top_of_the_cabinet.py`): low: replay 0.00
- `libero_90.kitchen_scene9_put_the_frying_pan_under_the_cabinet_shelf` (`roboverse_pack/tasks/libero_90/libero_kitchen_scene9_put_the_frying_pan_under_the_cabinet_shelf.py`): error: Exception: File roboverse_data/assets/libero/COMMON/stable_scanned_objects/white_bowl/mjcf/white_bowl.xml neither exists in the local…
- `libero_90.kitchen_scene9_put_the_white_bowl_on_top_of_the_cabinet` (`roboverse_pack/tasks/libero_90/libero_kitchen_scene9_put_the_white_bowl_on_top_of_the_cabinet.py`): low: replay 0.00
- `libero_90.kitchen_scene9_turn_on_the_stove` (`roboverse_pack/tasks/libero_90/libero_kitchen_scene9_turn_on_the_stove.py`): error: Exception: File roboverse_data/assets/libero/COMMON/stable_scanned_objects/white_bowl/mjcf/white_bowl.xml neither exists in the local…
- `libero_90.kitchen_scene9_turn_on_the_stove_and_put_the_frying_pan_on_it` (`roboverse_pack/tasks/libero_90/libero_kitchen_scene9_turn_on_the_stove_and_put_the_frying_pan_on_it.py`): error: Exception: File roboverse_data/assets/libero/COMMON/stable_scanned_objects/white_bowl/mjcf/white_bowl.xml neither exists in the local…
- `libero_90.living_room_scene1_pick_up_the_alphabet_soup_and_put_it_in_the_basket` (`roboverse_pack/tasks/libero_90/living_room_scene1_pick_up_the_alphabet_soup_and_put_it_in_the_basket.py`): low: replay 0.00
- `libero_90.living_room_scene1_pick_up_the_cream_cheese_box_and_put_it_in_the_basket` (`roboverse_pack/tasks/libero_90/living_room_scene1_pick_up_the_cream_cheese_box_and_put_it_in_the_basket.py`): low: replay 0.00
- `libero_90.living_room_scene1_pick_up_the_ketchup_and_put_it_in_the_basket` (`roboverse_pack/tasks/libero_90/living_room_scene1_pick_up_the_ketchup_and_put_it_in_the_basket.py`): low: replay 0.00
- `libero_90.living_room_scene1_pick_up_the_tomato_sauce_and_put_it_in_the_basket` (`roboverse_pack/tasks/libero_90/living_room_scene1_pick_up_the_tomato_sauce_and_put_it_in_the_basket.py`): low: replay 0.00
- `libero_90.living_room_scene2_pick_up_the_alphabet_soup_and_put_it_in_the_basket` (`roboverse_pack/tasks/libero_90/living_room_scene2_pick_up_the_alphabet_soup_and_put_it_in_the_basket.py`): low: replay 0.00
- `libero_90.living_room_scene2_pick_up_the_butter_and_put_it_in_the_basket` (`roboverse_pack/tasks/libero_90/living_room_scene2_pick_up_the_butter_and_put_it_in_the_basket.py`): low: replay 0.00
- `libero_90.living_room_scene2_pick_up_the_milk_and_put_it_in_the_basket` (`roboverse_pack/tasks/libero_90/living_room_scene2_pick_up_the_milk_and_put_it_in_the_basket.py`): low: replay 0.00
- `libero_90.living_room_scene2_pick_up_the_orange_juice_and_put_it_in_the_basket` (`roboverse_pack/tasks/libero_90/living_room_scene2_pick_up_the_orange_juice_and_put_it_in_the_basket.py`): low: replay 0.00
- `libero_90.living_room_scene2_pick_up_the_tomato_sauce_and_put_it_in_the_basket` (`roboverse_pack/tasks/libero_90/living_room_scene2_pick_up_the_tomato_sauce_and_put_it_in_the_basket.py`): low: replay 0.00
- `libero_90.living_room_scene3_pick_up_the_alphabet_soup_and_put_it_in_the_tray` (`roboverse_pack/tasks/libero_90/living_room_scene3_pick_up_the_alphabet_soup_and_put_it_in_the_tray.py`): low: replay 0.00
- `libero_90.living_room_scene3_pick_up_the_butter_and_put_it_in_the_tray` (`roboverse_pack/tasks/libero_90/living_room_scene3_pick_up_the_butter_and_put_it_in_the_tray.py`): low: replay 0.00
- `libero_90.living_room_scene3_pick_up_the_cream_cheese_and_put_it_in_the_tray` (`roboverse_pack/tasks/libero_90/living_room_scene3_pick_up_the_cream_cheese_and_put_it_in_the_tray.py`): low: replay 0.00
- `libero_90.living_room_scene3_pick_up_the_ketchup_and_put_it_in_the_tray` (`roboverse_pack/tasks/libero_90/living_room_scene3_pick_up_the_ketchup_and_put_it_in_the_tray.py`): low: replay 0.00
- `libero_90.living_room_scene3_pick_up_the_tomato_sauce_and_put_it_in_the_tray` (`roboverse_pack/tasks/libero_90/living_room_scene3_pick_up_the_tomato_sauce_and_put_it_in_the_tray.py`): low: replay 0.00
- `libero_90.living_room_scene4_pick_up_the_black_bowl_on_the_left_and_put_it_in_the_tray` (`roboverse_pack/tasks/libero_90/living_room_scene4_pick_up_the_black_bowl_on_the_left_and_put_it_in_the_tray.py`): low: replay 0.00
- `libero_90.living_room_scene4_pick_up_the_chocolate_pudding_and_put_it_in_the_tray` (`roboverse_pack/tasks/libero_90/living_room_scene4_pick_up_the_chocolate_pudding_and_put_it_in_the_tray.py`): low: replay 0.00
- `libero_90.living_room_scene4_pick_up_the_salad_dressing_and_put_it_in_the_tray` (`roboverse_pack/tasks/libero_90/living_room_scene4_pick_up_the_salad_dressing_and_put_it_in_the_tray.py`): low: replay 0.00
- `libero_90.living_room_scene4_stack_the_left_bowl_on_the_right_bowl_and_place_them_in_the_tray` (`roboverse_pack/tasks/libero_90/living_room_scene4_stack_the_left_bowl_on_the_right_bowl_and_place_them_in_the_tray.py`): low: replay 0.00
- `libero_90.living_room_scene4_stack_the_right_bowl_on_the_left_bowl_and_place_them_in_the_tray` (`roboverse_pack/tasks/libero_90/living_room_scene4_stack_the_right_bowl_on_the_left_bowl_and_place_them_in_the_tray.py`): no-demos: Exception: File roboverse_data/trajs/libero90/libero_90_living_room_scene4_stack_the_right_bowl_on_the_left_bowl_and_place_them_in…

### maniskill

- `maniskill.peg_insertion_side_*` (999 generated variants, `roboverse_pack/tasks/maniskill/peg_insertion_side.py`): USD assets absent from the Hub
- `maniskill.pick_single_egad_*` (1588 generated variants, `roboverse_pack/tasks/maniskill/pick_single_egad.py`): no demos or assets on the Hub
- `maniskill.draw_svg` (`roboverse_pack/tasks/maniskill/draw_svg.py`): low: replay 0.00
- `maniskill.draw_triangle` (`roboverse_pack/tasks/maniskill/draw_triangle.py`): low: replay 0.00
- `maniskill.draw_triangle_native` (`roboverse_pack/tasks/maniskill/native_tasks.py`): MuJoCo: USD-only asset; Isaac 6: error Isaac Sim 6 child: RuntimeError: Failed to remove '/World/envs/env_0/panda/Geometry/panda_link0.newt
- `maniskill.lift_peg_upright_native` (`roboverse_pack/tasks/maniskill/native_tasks.py`): MuJoCo: USD-only asset; Isaac 6: error Isaac Sim 6 child: RuntimeError: Failed to remove '/World/envs/env_0/panda/Geometry/panda_link0.newt
- `maniskill.peg_insertion_side_base` (`roboverse_pack/tasks/maniskill/peg_insertion_side.py`): no class-level scenario: cannot be built by the harness
- `maniskill.peg_insertion_side_native` (`roboverse_pack/tasks/maniskill/native_tasks.py`): MuJoCo: USD-only asset; Isaac 6: error Isaac Sim 6 child: RuntimeError: Failed to remove '/World/envs/env_0/panda/Geometry/panda_link0.newt
- `maniskill.pick_cube_native` (`roboverse_pack/tasks/maniskill/native_tasks.py`): MuJoCo: USD-only asset; Isaac 6: error Isaac Sim 6 child: RuntimeError: Failed to remove '/World/envs/env_0/panda/Geometry/panda_link0.newt
- `maniskill.pick_cube_v1` (`roboverse_pack/tasks/maniskill/pick_cube_v1.py`): low: replay 0.00
- `maniskill.pick_single_egad_base` (`roboverse_pack/tasks/maniskill/pick_single_egad.py`): no class-level scenario: cannot be built by the harness
- `maniskill.pick_single_ycb_adjustable_wrench` (`roboverse_pack/tasks/maniskill/pick_single_ycb.py`): no-demos: Exception: File roboverse_data/trajs/maniskill/pick_single_ycb/trajectory-franka-042_adjustable_wrench_v2.pkl neither exists in th…
- `maniskill.pick_single_ycb_apple` (`roboverse_pack/tasks/maniskill/pick_single_ycb.py`): no-demos: Exception: File roboverse_data/trajs/maniskill/pick_single_ycb/trajectory-franka-013_apple_v2.pkl neither exists in the local dire…
- `maniskill.pick_single_ycb_banana` (`roboverse_pack/tasks/maniskill/pick_single_ycb.py`): no-demos: Exception: File roboverse_data/trajs/maniskill/pick_single_ycb/trajectory-franka-011_banana_v2.pkl neither exists in the local dir…
- `maniskill.pick_single_ycb_base` (`roboverse_pack/tasks/maniskill/pick_single_ycb.py`): no class-level scenario: cannot be built by the harness
- `maniskill.pick_single_ycb_baseball` (`roboverse_pack/tasks/maniskill/pick_single_ycb.py`): no-demos: Exception: File roboverse_data/trajs/maniskill/pick_single_ycb/trajectory-franka-055_baseball_v2.pkl neither exists in the local d…
- `maniskill.pick_single_ycb_bleach_cleanser` (`roboverse_pack/tasks/maniskill/pick_single_ycb.py`): no-demos: Exception: File roboverse_data/trajs/maniskill/pick_single_ycb/trajectory-franka-021_bleach_cleanser_v2.pkl neither exists in the …
- `maniskill.pick_single_ycb_bowl` (`roboverse_pack/tasks/maniskill/pick_single_ycb.py`): no-demos: Exception: File roboverse_data/trajs/maniskill/pick_single_ycb/trajectory-franka-024_bowl_v2.pkl neither exists in the local direc…
- `maniskill.pick_single_ycb_colored_wood_blocks` (`roboverse_pack/tasks/maniskill/pick_single_ycb.py`): no-demos: Exception: File roboverse_data/trajs/maniskill/pick_single_ycb/trajectory-franka-070-b_colored_wood_blocks_v2.pkl neither exists i…
- `maniskill.pick_single_ycb_cracker_box` (`roboverse_pack/tasks/maniskill/pick_single_ycb.py`): no-demos: Exception: File roboverse_data/trajs/maniskill/pick_single_ycb/trajectory-franka-003_cracker_box_v2.pkl neither exists in the loca…
- `maniskill.pick_single_ycb_cups` (`roboverse_pack/tasks/maniskill/pick_single_ycb.py`): no-demos: Exception: File roboverse_data/trajs/maniskill/pick_single_ycb/trajectory-franka-065-j_cups_v2.pkl neither exists in the local dir…
- `maniskill.pick_single_ycb_dice` (`roboverse_pack/tasks/maniskill/pick_single_ycb.py`): no-demos: Exception: File roboverse_data/trajs/maniskill/pick_single_ycb/trajectory-franka-062_dice_v2.pkl neither exists in the local direc…
- `maniskill.pick_single_ycb_extra_large_clamp` (`roboverse_pack/tasks/maniskill/pick_single_ycb.py`): no-demos: Exception: File roboverse_data/trajs/maniskill/pick_single_ycb/trajectory-franka-052_extra_large_clamp_v2.pkl neither exists in th…
- `maniskill.pick_single_ycb_flat_screwdriver` (`roboverse_pack/tasks/maniskill/pick_single_ycb.py`): no-demos: Exception: File roboverse_data/trajs/maniskill/pick_single_ycb/trajectory-franka-044_flat_screwdriver_v2.pkl neither exists in the…
- `maniskill.pick_single_ycb_foam_brick` (`roboverse_pack/tasks/maniskill/pick_single_ycb.py`): no-demos: Exception: File roboverse_data/trajs/maniskill/pick_single_ycb/trajectory-franka-061_foam_brick_v2.pkl neither exists in the local…
- `maniskill.pick_single_ycb_fork` (`roboverse_pack/tasks/maniskill/pick_single_ycb.py`): no-demos: Exception: File roboverse_data/trajs/maniskill/pick_single_ycb/trajectory-franka-030_fork_v2.pkl neither exists in the local direc…
- `maniskill.pick_single_ycb_gelatin_box` (`roboverse_pack/tasks/maniskill/pick_single_ycb.py`): no-demos: Exception: File roboverse_data/trajs/maniskill/pick_single_ycb/trajectory-franka-009_gelatin_box_v2.pkl neither exists in the loca…
- `maniskill.pick_single_ycb_golf_ball` (`roboverse_pack/tasks/maniskill/pick_single_ycb.py`): no-demos: Exception: File roboverse_data/trajs/maniskill/pick_single_ycb/trajectory-franka-058_golf_ball_v2.pkl neither exists in the local …
- `maniskill.pick_single_ycb_hammer` (`roboverse_pack/tasks/maniskill/pick_single_ycb.py`): no-demos: Exception: File roboverse_data/trajs/maniskill/pick_single_ycb/trajectory-franka-048_hammer_v2.pkl neither exists in the local dir…
- `maniskill.pick_single_ycb_knife` (`roboverse_pack/tasks/maniskill/pick_single_ycb.py`): no-demos: Exception: File roboverse_data/trajs/maniskill/pick_single_ycb/trajectory-franka-032_knife_v2.pkl neither exists in the local dire…
- `maniskill.pick_single_ycb_large_clamp` (`roboverse_pack/tasks/maniskill/pick_single_ycb.py`): no-demos: Exception: File roboverse_data/trajs/maniskill/pick_single_ycb/trajectory-franka-051_large_clamp_v2.pkl neither exists in the loca…
- `maniskill.pick_single_ycb_large_marker` (`roboverse_pack/tasks/maniskill/pick_single_ycb.py`): no-demos: Exception: File roboverse_data/trajs/maniskill/pick_single_ycb/trajectory-franka-040_large_marker_v2.pkl neither exists in the loc…
- `maniskill.pick_single_ycb_lego_duplo` (`roboverse_pack/tasks/maniskill/pick_single_ycb.py`): no-demos: Exception: File roboverse_data/trajs/maniskill/pick_single_ycb/trajectory-franka-073-g_lego_duplo_v2.pkl neither exists in the loc…
- `maniskill.pick_single_ycb_lemon` (`roboverse_pack/tasks/maniskill/pick_single_ycb.py`): no-demos: Exception: File roboverse_data/trajs/maniskill/pick_single_ycb/trajectory-franka-014_lemon_v2.pkl neither exists in the local dire…
- `maniskill.pick_single_ycb_marbles` (`roboverse_pack/tasks/maniskill/pick_single_ycb.py`): no-demos: Exception: File roboverse_data/trajs/maniskill/pick_single_ycb/trajectory-franka-063-b_marbles_v2.pkl neither exists in the local …
- `maniskill.pick_single_ycb_master_chef_can` (`roboverse_pack/tasks/maniskill/pick_single_ycb.py`): no-demos: Exception: File roboverse_data/trajs/maniskill/pick_single_ycb/trajectory-franka-002_master_chef_can_v2.pkl neither exists in the …
- `maniskill.pick_single_ycb_medium_clamp` (`roboverse_pack/tasks/maniskill/pick_single_ycb.py`): no-demos: Exception: File roboverse_data/trajs/maniskill/pick_single_ycb/trajectory-franka-050_medium_clamp_v2.pkl neither exists in the loc…
- `maniskill.pick_single_ycb_mini_soccer_ball` (`roboverse_pack/tasks/maniskill/pick_single_ycb.py`): no-demos: Exception: File roboverse_data/trajs/maniskill/pick_single_ycb/trajectory-franka-053_mini_soccer_ball_v2.pkl neither exists in the…
- `maniskill.pick_single_ycb_mug` (`roboverse_pack/tasks/maniskill/pick_single_ycb.py`): no-demos: Exception: File roboverse_data/trajs/maniskill/pick_single_ycb/trajectory-franka-025_mug_v2.pkl neither exists in the local direct…
- `maniskill.pick_single_ycb_mustard_bottle` (`roboverse_pack/tasks/maniskill/pick_single_ycb.py`): no-demos: Exception: File roboverse_data/trajs/maniskill/pick_single_ycb/trajectory-franka-006_mustard_bottle_v2.pkl neither exists in the l…
- `maniskill.pick_single_ycb_nine_hole_peg_test` (`roboverse_pack/tasks/maniskill/pick_single_ycb.py`): no-demos: Exception: File roboverse_data/trajs/maniskill/pick_single_ycb/trajectory-franka-071_nine_hole_peg_test_v2.pkl neither exists in t…
- `maniskill.pick_single_ycb_orange` (`roboverse_pack/tasks/maniskill/pick_single_ycb.py`): no-demos: Exception: File roboverse_data/trajs/maniskill/pick_single_ycb/trajectory-franka-017_orange_v2.pkl neither exists in the local dir…
- `maniskill.pick_single_ycb_padlock` (`roboverse_pack/tasks/maniskill/pick_single_ycb.py`): no-demos: Exception: File roboverse_data/trajs/maniskill/pick_single_ycb/trajectory-franka-038_padlock_v2.pkl neither exists in the local di…
- `maniskill.pick_single_ycb_peach` (`roboverse_pack/tasks/maniskill/pick_single_ycb.py`): no-demos: Exception: File roboverse_data/trajs/maniskill/pick_single_ycb/trajectory-franka-015_peach_v2.pkl neither exists in the local dire…
- `maniskill.pick_single_ycb_pear` (`roboverse_pack/tasks/maniskill/pick_single_ycb.py`): no-demos: Exception: File roboverse_data/trajs/maniskill/pick_single_ycb/trajectory-franka-016_pear_v2.pkl neither exists in the local direc…
- `maniskill.pick_single_ycb_phillips_screwdriver` (`roboverse_pack/tasks/maniskill/pick_single_ycb.py`): no-demos: Exception: File roboverse_data/trajs/maniskill/pick_single_ycb/trajectory-franka-043_phillips_screwdriver_v2.pkl neither exists in…
- `maniskill.pick_single_ycb_pitcher_base` (`roboverse_pack/tasks/maniskill/pick_single_ycb.py`): no-demos: Exception: File roboverse_data/trajs/maniskill/pick_single_ycb/trajectory-franka-019_pitcher_base_v2.pkl neither exists in the loc…
- `maniskill.pick_single_ycb_plum` (`roboverse_pack/tasks/maniskill/pick_single_ycb.py`): no-demos: Exception: File roboverse_data/trajs/maniskill/pick_single_ycb/trajectory-franka-018_plum_v2.pkl neither exists in the local direc…
- `maniskill.pick_single_ycb_potted_meat_can` (`roboverse_pack/tasks/maniskill/pick_single_ycb.py`): no-demos: Exception: File roboverse_data/trajs/maniskill/pick_single_ycb/trajectory-franka-010_potted_meat_can_v2.pkl neither exists in the …
- `maniskill.pick_single_ycb_power_drill` (`roboverse_pack/tasks/maniskill/pick_single_ycb.py`): no-demos: Exception: File roboverse_data/trajs/maniskill/pick_single_ycb/trajectory-franka-035_power_drill_v2.pkl neither exists in the loca…
- `maniskill.pick_single_ycb_pudding_box` (`roboverse_pack/tasks/maniskill/pick_single_ycb.py`): no-demos: Exception: File roboverse_data/trajs/maniskill/pick_single_ycb/trajectory-franka-008_pudding_box_v2.pkl neither exists in the loca…
- `maniskill.pick_single_ycb_racquetball` (`roboverse_pack/tasks/maniskill/pick_single_ycb.py`): no-demos: Exception: File roboverse_data/trajs/maniskill/pick_single_ycb/trajectory-franka-057_racquetball_v2.pkl neither exists in the loca…
- `maniskill.pick_single_ycb_rubiks_cube` (`roboverse_pack/tasks/maniskill/pick_single_ycb.py`): no-demos: Exception: File roboverse_data/trajs/maniskill/pick_single_ycb/trajectory-franka-077_rubiks_cube_v2.pkl neither exists in the loca…
- `maniskill.pick_single_ycb_scissors` (`roboverse_pack/tasks/maniskill/pick_single_ycb.py`): no-demos: Exception: File roboverse_data/trajs/maniskill/pick_single_ycb/trajectory-franka-037_scissors_v2.pkl neither exists in the local d…
- `maniskill.pick_single_ycb_softball` (`roboverse_pack/tasks/maniskill/pick_single_ycb.py`): no-demos: Exception: File roboverse_data/trajs/maniskill/pick_single_ycb/trajectory-franka-054_softball_v2.pkl neither exists in the local d…
- `maniskill.pick_single_ycb_spatula` (`roboverse_pack/tasks/maniskill/pick_single_ycb.py`): no-demos: Exception: File roboverse_data/trajs/maniskill/pick_single_ycb/trajectory-franka-033_spatula_v2.pkl neither exists in the local di…
- `maniskill.pick_single_ycb_sponge` (`roboverse_pack/tasks/maniskill/pick_single_ycb.py`): no-demos: Exception: File roboverse_data/trajs/maniskill/pick_single_ycb/trajectory-franka-026_sponge_v2.pkl neither exists in the local dir…
- `maniskill.pick_single_ycb_spoon` (`roboverse_pack/tasks/maniskill/pick_single_ycb.py`): no-demos: Exception: File roboverse_data/trajs/maniskill/pick_single_ycb/trajectory-franka-031_spoon_v2.pkl neither exists in the local dire…
- `maniskill.pick_single_ycb_strawberry` (`roboverse_pack/tasks/maniskill/pick_single_ycb.py`): no-demos: Exception: File roboverse_data/trajs/maniskill/pick_single_ycb/trajectory-franka-012_strawberry_v2.pkl neither exists in the local…
- `maniskill.pick_single_ycb_sugar_box` (`roboverse_pack/tasks/maniskill/pick_single_ycb.py`): no-demos: Exception: File roboverse_data/trajs/maniskill/pick_single_ycb/trajectory-franka-004_sugar_box_v2.pkl neither exists in the local …
- `maniskill.pick_single_ycb_tennis_ball` (`roboverse_pack/tasks/maniskill/pick_single_ycb.py`): no-demos: Exception: File roboverse_data/trajs/maniskill/pick_single_ycb/trajectory-franka-056_tennis_ball_v2.pkl neither exists in the loca…
- `maniskill.pick_single_ycb_tomato_soup_can` (`roboverse_pack/tasks/maniskill/pick_single_ycb.py`): no-demos: Exception: File roboverse_data/trajs/maniskill/pick_single_ycb/trajectory-franka-005_tomato_soup_can_v2.pkl neither exists in the …
- `maniskill.pick_single_ycb_toy_airplane` (`roboverse_pack/tasks/maniskill/pick_single_ycb.py`): no-demos: Exception: File roboverse_data/trajs/maniskill/pick_single_ycb/trajectory-franka-072-e_toy_airplane_v2.pkl neither exists in the l…
- `maniskill.pick_single_ycb_tuna_fish_can` (`roboverse_pack/tasks/maniskill/pick_single_ycb.py`): no-demos: Exception: File roboverse_data/trajs/maniskill/pick_single_ycb/trajectory-franka-007_tuna_fish_can_v2.pkl neither exists in the lo…
- `maniskill.pick_single_ycb_wood_block` (`roboverse_pack/tasks/maniskill/pick_single_ycb.py`): no-demos: Exception: File roboverse_data/trajs/maniskill/pick_single_ycb/trajectory-franka-036_wood_block_v2.pkl neither exists in the local…
- `maniskill.place_sphere` (`roboverse_pack/tasks/maniskill/place_sphere.py`): error: ValueError: Cannot fetch '/home/ntu/Desktop/ant/RoboVerse/roboverse_data/assets/maniskill/PlaceSphere/bin/mjcf/bin.stl': an asset fil…
- `maniskill.place_sphere_native` (`roboverse_pack/tasks/maniskill/native_tasks.py`): MuJoCo: USD-only asset; Isaac 6: error Isaac Sim 6 child: RuntimeError: Failed to remove '/World/envs/env_0/panda/Geometry/panda_link0.newt
- `maniskill.plug_charger_native` (`roboverse_pack/tasks/maniskill/native_tasks.py`): MuJoCo: USD-only asset; Isaac 6: error Isaac Sim 6 child: RuntimeError: Failed to remove '/World/envs/env_0/panda/Geometry/panda_link0.newt
- `maniskill.poke_cube` (`roboverse_pack/tasks/maniskill/poke_cube.py`): error: ValueError: Cannot fetch '/home/ntu/Desktop/ant/RoboVerse/roboverse_data/assets/maniskill/PokeCube/peg/mjcf/stick.stl': an asset file…
- `maniskill.poke_cube_native` (`roboverse_pack/tasks/maniskill/native_tasks.py`): MuJoCo: USD-only asset; Isaac 6: error Isaac Sim 6 child: RuntimeError: Failed to remove '/World/envs/env_0/panda/Geometry/panda_link0.newt
- `maniskill.pull_cube_native` (`roboverse_pack/tasks/maniskill/native_tasks.py`): MuJoCo: USD-only asset; Isaac 6: error Isaac Sim 6 child: RuntimeError: Failed to remove '/World/envs/env_0/panda/Geometry/panda_link0.newt
- `maniskill.pull_cube_tool` (`roboverse_pack/tasks/maniskill/pull_cube_tool.py`): error: ValueError: Cannot fetch '/home/ntu/Desktop/ant/RoboVerse/roboverse_data/assets/maniskill/PullCubeTool/tool/mjcf/tool.stl': an asset …
- `maniskill.pull_cube_tool_native` (`roboverse_pack/tasks/maniskill/native_tasks.py`): MuJoCo: USD-only asset; Isaac 6: error Isaac Sim 6 child: RuntimeError: Failed to remove '/World/envs/env_0/panda/Geometry/panda_link0.newt
- `maniskill.push_cube_native` (`roboverse_pack/tasks/maniskill/native_tasks.py`): MuJoCo: USD-only asset; Isaac 6: error Isaac Sim 6 child: RuntimeError: Failed to remove '/World/envs/env_0/panda/Geometry/panda_link0.newt
- `maniskill.push_t_native` (`roboverse_pack/tasks/maniskill/native_tasks.py`): MuJoCo: USD-only asset; Isaac 6: error Isaac Sim 6 child: RuntimeError: Failed to remove '/World/envs/env_0/panda/Geometry/panda_link0.newt
- `maniskill.roll_ball` (`roboverse_pack/tasks/maniskill/roll_ball.py`): low: replay 0.00
- `maniskill.roll_ball_native` (`roboverse_pack/tasks/maniskill/native_tasks.py`): MuJoCo: USD-only asset; Isaac 6: error Isaac Sim 6 child: RuntimeError: Failed to remove '/World/envs/env_0/panda/Geometry/panda_link0.newt
- `maniskill.stack_cube_native` (`roboverse_pack/tasks/maniskill/native_tasks.py`): MuJoCo: USD-only asset; Isaac 6: error Isaac Sim 6 child: RuntimeError: Failed to remove '/World/envs/env_0/panda/Geometry/panda_link0.newt
- `maniskill.stack_pyramid` (`roboverse_pack/tasks/maniskill/stack_pyramid.py`): low: replay 0.00
- `maniskill.stack_pyramid_native` (`roboverse_pack/tasks/maniskill/native_tasks.py`): MuJoCo: USD-only asset; Isaac 6: error Isaac Sim 6 child: RuntimeError: Failed to remove '/World/envs/env_0/panda/Geometry/panda_link0.newt
- `maniskill.two_robot_pick_cube_native` (`roboverse_pack/tasks/maniskill/two_robot_pick_cube.py`): MuJoCo: USD-only asset; Isaac 6: error Isaac Sim 6 child: pxr.Tf.ErrorException: 

### mjlab

- `mjlab.cartpole_balance` (`roboverse_pack/tasks/mjlab/cartpole.py`): scene-only task: no robot, zero action slots, so the harness cannot drive it; mjlab.cartpole_balance_v2 is the usable port
- `mjlab.cartpole_swingup` (`roboverse_pack/tasks/mjlab/cartpole.py`): scene-only task: no robot, zero action slots; mjlab.cartpole_swingup_v2 is the usable port

### mujoco_playground

- `mujoco_playground.handover` (`roboverse_pack/tasks/mujoco_playground/handover.py`): MuJoCo: USD-only asset; Isaac 6: unsupported ValueError: Object 'aloha' has no usd asset path set for the 'isaacsim' simulator. Please set '…
- `mujoco_playground.open_cabinet` (`roboverse_pack/tasks/mujoco_playground/open_cabinet.py`): error: KeyError: "Invalid name 'franka/left_finger_pad'. Valid names: ['//unnamed_geom_0', 'barrier/barrier_geom', 'franka//unnamed_geom_0',…
- `mujoco_playground.pick` (`roboverse_pack/tasks/mujoco_playground/pick.py`): error: KeyError: "Invalid name 'franka/left_finger_pad'. Valid names: ['//unnamed_geom_0', 'box_model/box_geom', 'franka//unnamed_geom_0', '…

### pick_place

- `pick_place.approach_grasp_hu` (`roboverse_pack/tasks/pick_place/approach_grasp_ceramic_teapot.py`): error: Exception: File roboverse_data/EmbodiedGenData/all_asset/table/mjcf/table.xml neither exists in the local directory nor exists in the…
- `pick_place.approach_grasp_knife` (`roboverse_pack/tasks/pick_place/approach_grasp_knife.py`): error: Exception: File roboverse_data/EmbodiedGenData/all_asset/table/mjcf/table.xml neither exists in the local directory nor exists in the…
- `pick_place.approach_grasp_simple_banana` (`roboverse_pack/tasks/pick_place/approach_grasp_banana.py`): error: Exception: File roboverse_data/assets/EmbodiedGenData/new_assets/lighting_fixtures/1/mjcf/0a4489b1a2875c82a580f8b62d346e08.xml neithe…
- `pick_place.approach_grasp_simple_bowl` (`roboverse_pack/tasks/pick_place/approach_grasp_bowl.py`): error: Exception: File roboverse_data/assets/EmbodiedGenData/assets/bowl/0f296af3df66565c9e1a7c2bc7b35d72/mjcf/0f296af3df66565c9e1a7c2bc7b35…
- `pick_place.approach_grasp_simple_screwdriver` (`roboverse_pack/tasks/pick_place/approach_grasp_screwdriver.py`): error: Exception: File roboverse_data/assets/EmbodiedGenData/new_assets/lighting_fixtures/1/mjcf/0a4489b1a2875c82a580f8b62d346e08.xml neithe…
- `pick_place.approach_grasp_simple_spoon` (`roboverse_pack/tasks/pick_place/approach_grasp_spoon.py`): error: Exception: File roboverse_data/assets/EmbodiedGenData/new_assets/basket/1/mjcf/663158968e3f5900af1f6e7cecef24c7.xml neither exists in…
- `pick_place.approach_grasp_simple_spoon2` (`roboverse_pack/tasks/pick_place/approach_grasp_spoon.py`): error: Exception: File roboverse_data/assets/EmbodiedGenData/new_assets/basket/1/mjcf/663158968e3f5900af1f6e7cecef24c7.xml neither exists in…
- `pick_place.hand_trajectory` (`roboverse_pack/tasks/pick_place/hand_trajectory.py`): MuJoCo: USD-only asset; Isaac 6: error ValueError: Refusing to fetch 'roboverse_pack/tasks/pick_place/marker/marker/marker.usd': resolves o
- `pick_place.track_banana` (`roboverse_pack/tasks/pick_place/track_banana.py`): error: FileNotFoundError: State file not found: eval_states/pick_place.approach_grasp_banana_franka_lift_states_100states_20251210_153518.pk…
- `pick_place.track_bowl` (`roboverse_pack/tasks/pick_place/track_bowl.py`): error: Exception: File roboverse_data/assets/EmbodiedGenData/assets/bowl/0f296af3df66565c9e1a7c2bc7b35d72/mjcf/0f296af3df66565c9e1a7c2bc7b35…
- `pick_place.track_hu` (`roboverse_pack/tasks/pick_place/track_ceramic_teapot.py`): error: FileNotFoundError: State file not found: /usr1/home/s125mdg56_03/RoboVerse/eval_states/pick_place.approach_grasp_hu_franka_lift_state…
- `pick_place.track_knife` (`roboverse_pack/tasks/pick_place/track_knife.py`): error: FileNotFoundError: State file not found: /usr1/home/s125mdg56_03/RoboVerse/eval_states/pick_place.approach_grasp_knife_franka_lift_st…
- `pick_place.track_screwdriver` (`roboverse_pack/tasks/pick_place/track_screwdriver.py`): error: FileNotFoundError: State file not found: eval_states/pick_place.approach_grasp_screwdriver_franka_lift_states_100states_20251210_1549…
- `pick_place.track_spoon` (`roboverse_pack/tasks/pick_place/track_spoon.py`): error: FileNotFoundError: State file not found: eval_states/pick_place.approach_grasp_simple_franka_lift_states_154states_20251207_171537.pk…

### rlbench

- `franka.basketball_in_hoop` (`roboverse_pack/tasks/rlbench/basketball_in_hoop.py`): MuJoCo: USD-only asset; Isaac 6: low 
- `franka.beat_the_buzz` (`roboverse_pack/tasks/rlbench/beat_the_buzz.py`): MuJoCo: USD-only asset; Isaac 6: low 
- `franka.block_pyramid` (`roboverse_pack/tasks/rlbench/block_pyramid.py`): low: replay 0.00
- `franka.change_channel` (`roboverse_pack/tasks/rlbench/change_channel.py`): low: replay 0.00
- `franka.change_clock` (`roboverse_pack/tasks/rlbench/change_clock.py`): MuJoCo: USD-only asset; Isaac 6: error Isaac Sim 6 child: RuntimeError: Failed to create articulation at: /World/envs/env_[^/]+/clock/root_
- `franka.close_door` (`roboverse_pack/tasks/rlbench/close_door.py`): MuJoCo: USD-only asset; Isaac 6: error Isaac Sim 6 child: RuntimeError: Failed to create articulation at: /World/envs/env_[^/]+/door_frame/
- `franka.close_drawer` (`roboverse_pack/tasks/rlbench/close_drawer.py`): MuJoCo: USD-only asset; Isaac 6: error Isaac Sim 6 child: RuntimeError: Failed to create articulation at: /World/envs/env_[^/]+/drawer_fram
- `franka.close_fridge` (`roboverse_pack/tasks/rlbench/close_fridge.py`): MuJoCo: USD-only asset; Isaac 6: error Isaac Sim 6 child: RuntimeError: Failed to create articulation at: /World/envs/env_[^/]+/fridge_base
- `franka.close_grill` (`roboverse_pack/tasks/rlbench/close_grill.py`): MuJoCo: USD-only asset; Isaac 6: error Isaac Sim 6 child: RuntimeError: Failed to create articulation at: /World/envs/env_[^/]+/grill/root_
- `franka.close_jar` (`roboverse_pack/tasks/rlbench/close_jar.py`): MuJoCo: USD-only asset; Isaac 6: low 
- `franka.close_laptop_lid` (`roboverse_pack/tasks/rlbench/close_laptop_lid.py`): MuJoCo: USD-only asset; Isaac 6: error Isaac Sim 6 child: RuntimeError: Failed to create articulation at: /World/envs/env_[^/]+/base/root_j
- `franka.close_microwave` (`roboverse_pack/tasks/rlbench/close_microwave.py`): MuJoCo: USD-only asset; Isaac 6: error Isaac Sim 6 child: RuntimeError: Failed to create articulation at: /World/envs/env_[^/]+/microwave_f
- `franka.empty_dishwasher` (`roboverse_pack/tasks/rlbench/empty_dishwasher.py`): MuJoCo: USD-only asset; Isaac 6: error Isaac Sim 6 child: RuntimeError: Failed to create articulation at: /World/envs/env_[^/]+/dishwasher/
- `franka.get_ice_from_fridge` (`roboverse_pack/tasks/rlbench/get_ice_from_fridge.py`): MuJoCo: USD-only asset; Isaac 6: error Isaac Sim 6 child: RuntimeError: Failed to create articulation at: /World/envs/env_[^/]+/fridge_base
- `franka.hit_ball_with_queue` (`roboverse_pack/tasks/rlbench/hit_ball_with_queue.py`): MuJoCo: USD-only asset; Isaac 6: low 
- `franka.hockey` (`roboverse_pack/tasks/rlbench/hockey.py`): MuJoCo: USD-only asset; Isaac 6: low 
- `franka.insert_onto_square_peg` (`roboverse_pack/tasks/rlbench/insert_onto_square_peg.py`): MuJoCo: USD-only asset; Isaac 6: low 
- `franka.lamp_off` (`roboverse_pack/tasks/rlbench/lamp_off.py`): MuJoCo: USD-only asset; Isaac 6: error Isaac Sim 6 child: RuntimeError: Failed to create articulation at: /World/envs/env_[^/]+/push_button
- `franka.lamp_on` (`roboverse_pack/tasks/rlbench/lamp_off.py`): MuJoCo: USD-only asset; Isaac 6: error Isaac Sim 6 child: RuntimeError: Failed to create articulation at: /World/envs/env_[^/]+/push_button
- `franka.lift_numbered_block` (`roboverse_pack/tasks/rlbench/lift_numbered_block.py`): MuJoCo: USD-only asset; Isaac 6: low 
- `franka.light_bulb_in` (`roboverse_pack/tasks/rlbench/light_bulb_in.py`): MuJoCo: USD-only asset; Isaac 6: low 
- `franka.light_bulb_out` (`roboverse_pack/tasks/rlbench/light_bulb_out.py`): MuJoCo: USD-only asset; Isaac 6: low 
- `franka.meat_off_grill` (`roboverse_pack/tasks/rlbench/meat_off_grill.py`): MuJoCo: USD-only asset; Isaac 6: low 
- `franka.meat_on_grill` (`roboverse_pack/tasks/rlbench/meat_off_grill.py`): MuJoCo: USD-only asset; Isaac 6: low 
- `franka.open_box` (`roboverse_pack/tasks/rlbench/open_box.py`): low: replay 0.00
- `franka.open_door` (`roboverse_pack/tasks/rlbench/close_door.py`): MuJoCo: USD-only asset; Isaac 6: error Isaac Sim 6 child: RuntimeError: Failed to create articulation at: /World/envs/env_[^/]+/door_frame/
- `franka.open_drawer` (`roboverse_pack/tasks/rlbench/close_drawer.py`): MuJoCo: USD-only asset; Isaac 6: error Isaac Sim 6 child: RuntimeError: Failed to create articulation at: /World/envs/env_[^/]+/drawer_fram
- `franka.open_fridge` (`roboverse_pack/tasks/rlbench/close_fridge.py`): MuJoCo: USD-only asset; Isaac 6: error Isaac Sim 6 child: RuntimeError: Failed to create articulation at: /World/envs/env_[^/]+/fridge_base
- `franka.open_grill` (`roboverse_pack/tasks/rlbench/open_grill.py`): MuJoCo: USD-only asset; Isaac 6: error Isaac Sim 6 child: RuntimeError: Failed to create articulation at: /World/envs/env_[^/]+/grill/root_
- `franka.open_jar` (`roboverse_pack/tasks/rlbench/open_jar.py`): MuJoCo: USD-only asset; Isaac 6: low 
- `franka.open_microwave` (`roboverse_pack/tasks/rlbench/close_microwave.py`): MuJoCo: USD-only asset; Isaac 6: error Isaac Sim 6 child: RuntimeError: Failed to create articulation at: /World/envs/env_[^/]+/microwave_f
- `franka.open_oven` (`roboverse_pack/tasks/rlbench/open_oven.py`): MuJoCo: USD-only asset; Isaac 6: error Isaac Sim 6 child: RuntimeError: Failed to create articulation at: /World/envs/env_[^/]+/oven_base/r
- `franka.open_washing_machine` (`roboverse_pack/tasks/rlbench/open_washing_machine.py`): MuJoCo: USD-only asset; Isaac 6: error Isaac Sim 6 child: RuntimeError: Failed to create articulation at: /World/envs/env_[^/]+/washer/root
- `franka.open_window` (`roboverse_pack/tasks/rlbench/open_window.py`): MuJoCo: USD-only asset; Isaac 6: error Isaac Sim 6 child: RuntimeError: Failed to create articulation at: /World/envs/env_[^/]+/window_main
- `franka.open_wine_bottle` (`roboverse_pack/tasks/rlbench/open_wine_bottle.py`): MuJoCo: USD-only asset; Isaac 6: error Isaac Sim 6 child: RuntimeError: Failed to create articulation at: /World/envs/env_[^/]+/bottle/root
- `franka.phone_on_base` (`roboverse_pack/tasks/rlbench/phone_on_base.py`): MuJoCo: USD-only asset; Isaac 6: low 
- `franka.pick_and_lift` (`roboverse_pack/tasks/rlbench/pick_and_lift.py`): low: replay 0.00
- `franka.pick_and_lift_small` (`roboverse_pack/tasks/rlbench/pick_and_lift_small.py`): MuJoCo: USD-only asset; Isaac 6: low 
- `franka.pick_up_cup` (`roboverse_pack/tasks/rlbench/pick_up_cup.py`): MuJoCo: USD-only asset; Isaac 6: low 
- `franka.place_cups` (`roboverse_pack/tasks/rlbench/place_cups.py`): MuJoCo: USD-only asset; Isaac 6: low 
- `franka.place_shape_in_shape_sorter` (`roboverse_pack/tasks/rlbench/place_shape_in_shape_sorter.py`): MuJoCo: USD-only asset; Isaac 6: low 
- `franka.play_jenga` (`roboverse_pack/tasks/rlbench/play_jenga.py`): MuJoCo: USD-only asset; Isaac 6: low 
- `franka.plug_charger_in_power_supply` (`roboverse_pack/tasks/rlbench/plug_charger_in_power_supply.py`): MuJoCo: USD-only asset; Isaac 6: low 
- `franka.pour_from_cup_to_cup` (`roboverse_pack/tasks/rlbench/pour_from_cup_to_cup.py`): MuJoCo: USD-only asset; Isaac 6: low 
- `franka.press_switch` (`roboverse_pack/tasks/rlbench/press_switch.py`): MuJoCo: USD-only asset; Isaac 6: error Isaac Sim 6 child: RuntimeError: Failed to create articulation at: /World/envs/env_[^/]+/switch_main
- `franka.push_button` (`roboverse_pack/tasks/rlbench/push_button.py`): MuJoCo: USD-only asset; Isaac 6: error Isaac Sim 6 child: RuntimeError: Failed to create articulation at: /World/envs/env_[^/]+/push_button
- `franka.push_buttons` (`roboverse_pack/tasks/rlbench/push_buttons.py`): MuJoCo: USD-only asset; Isaac 6: error Isaac Sim 6 child: RuntimeError: Failed to create articulation at: /World/envs/env_[^/]+/push_button
- `franka.put_all_groceries_in_cupboard` (`roboverse_pack/tasks/rlbench/put_groceries_in_cupboard.py`): MuJoCo: USD-only asset; Isaac 6: low 
- `franka.put_books_on_bookshelf` (`roboverse_pack/tasks/rlbench/put_books_on_bookshelf.py`): MuJoCo: USD-only asset; Isaac 6: low 
- `franka.put_bottle_in_fridge` (`roboverse_pack/tasks/rlbench/put_bottle_in_fridge.py`): MuJoCo: USD-only asset; Isaac 6: error Isaac Sim 6 child: RuntimeError: Failed to create articulation at: /World/envs/env_[^/]+/fridge_base
- `franka.put_groceries_in_cupboard` (`roboverse_pack/tasks/rlbench/put_groceries_in_cupboard.py`): MuJoCo: USD-only asset; Isaac 6: low 
- `franka.put_item_in_drawer` (`roboverse_pack/tasks/rlbench/put_item_in_drawer.py`): MuJoCo: USD-only asset; Isaac 6: error Isaac Sim 6 child: RuntimeError: Failed to create articulation at: /World/envs/env_[^/]+/drawer_fram
- `franka.put_knife_in_knife_block` (`roboverse_pack/tasks/rlbench/put_knife_in_knife_block.py`): MuJoCo: USD-only asset; Isaac 6: low 
- `franka.put_knife_on_chopping_board` (`roboverse_pack/tasks/rlbench/put_knife_in_knife_block.py`): MuJoCo: USD-only asset; Isaac 6: low 
- `franka.put_money_in_safe` (`roboverse_pack/tasks/rlbench/put_money_in_safe.py`): MuJoCo: USD-only asset; Isaac 6: error Isaac Sim 6 child: RuntimeError: Failed to create articulation at: /World/envs/env_[^/]+/safe_body/r
- `franka.put_plate_in_colored_dish_rack` (`roboverse_pack/tasks/rlbench/put_plate_in_colored_dish_rack.py`): MuJoCo: USD-only asset; Isaac 6: low 
- `franka.put_rubbish_in_bin` (`roboverse_pack/tasks/rlbench/put_rubbish_in_bin.py`): MuJoCo: USD-only asset; Isaac 6: low 
- `franka.put_shoes_in_box` (`roboverse_pack/tasks/rlbench/put_shoes_in_box.py`): MuJoCo: USD-only asset; Isaac 6: error Isaac Sim 6 child: RuntimeError: Failed to create articulation at: /World/envs/env_[^/]+/box_base/ro
- `franka.put_toilet_roll_on_stand` (`roboverse_pack/tasks/rlbench/put_toilet_roll_on_stand.py`): MuJoCo: USD-only asset; Isaac 6: low 
- `franka.put_tray_in_oven` (`roboverse_pack/tasks/rlbench/open_oven.py`): MuJoCo: USD-only asset; Isaac 6: error Isaac Sim 6 child: RuntimeError: Failed to create articulation at: /World/envs/env_[^/]+/oven_base/r
- `franka.put_umbrella_in_umbrella_stand` (`roboverse_pack/tasks/rlbench/put_umbrella_in_umbrella_stand.py`): MuJoCo: USD-only asset; Isaac 6: low 
- `franka.reach_and_drag` (`roboverse_pack/tasks/rlbench/reach_and_drag.py`): MuJoCo: USD-only asset; Isaac 6: low 
- `franka.reach_target` (`roboverse_pack/tasks/rlbench/reach_target.py`): low: replay 0.00
- `franka.remove_cups` (`roboverse_pack/tasks/rlbench/place_cups.py`): MuJoCo: USD-only asset; Isaac 6: low 
- `franka.scoop_with_spatula` (`roboverse_pack/tasks/rlbench/scoop_with_spatula.py`): MuJoCo: USD-only asset; Isaac 6: low 
- `franka.screw_nail` (`roboverse_pack/tasks/rlbench/screw_nail.py`): MuJoCo: USD-only asset; Isaac 6: error Isaac Sim 6 child: RuntimeError: Failed to create articulation at: /World/envs/env_[^/]+/block/root_
- `franka.set_the_table` (`roboverse_pack/tasks/rlbench/set_the_table.py`): MuJoCo: USD-only asset; Isaac 6: low 
- `franka.setup_checkers` (`roboverse_pack/tasks/rlbench/setup_checkers.py`): MuJoCo: USD-only asset; Isaac 6: low 
- `franka.setup_chess` (`roboverse_pack/tasks/rlbench/setup_chess.py`): MuJoCo: USD-only asset; Isaac 6: low 
- `franka.slide_block_to_target` (`roboverse_pack/tasks/rlbench/slide_block_to_target.py`): MuJoCo: USD-only asset; Isaac 6: low 
- `franka.slide_cabinet_open_and_place_cups` (`roboverse_pack/tasks/rlbench/slide_cabinet_open_and_place_cups.py`): MuJoCo: USD-only asset; Isaac 6: error Isaac Sim 6 child: RuntimeError: Failed to create articulation at: /World/envs/env_[^/]+/cabinet_bas
- `franka.stack_blocks` (`roboverse_pack/tasks/rlbench/stack_blocks.py`): MuJoCo: USD-only asset; Isaac 6: low 
- `franka.stack_chairs` (`roboverse_pack/tasks/rlbench/stack_chairs.py`): MuJoCo: USD-only asset; Isaac 6: low 
- `franka.stack_cups` (`roboverse_pack/tasks/rlbench/stack_cups.py`): MuJoCo: USD-only asset; Isaac 6: low 
- `franka.stack_wine` (`roboverse_pack/tasks/rlbench/stack_wine.py`): MuJoCo: USD-only asset; Isaac 6: low 
- `franka.sweep_to_dustpan` (`roboverse_pack/tasks/rlbench/sweep_to_dustpan.py`): MuJoCo: USD-only asset; Isaac 6: low 
- `franka.take_cup_out_from_cabinet` (`roboverse_pack/tasks/rlbench/slide_cabinet_open_and_place_cups.py`): MuJoCo: USD-only asset; Isaac 6: error Isaac Sim 6 child: RuntimeError: Failed to create articulation at: /World/envs/env_[^/]+/cabinet_bas
- `franka.take_tray_out_of_oven` (`roboverse_pack/tasks/rlbench/open_oven.py`): MuJoCo: USD-only asset; Isaac 6: error Isaac Sim 6 child: RuntimeError: Failed to create articulation at: /World/envs/env_[^/]+/oven_base/r

### robosuite

- `robosuite.door` (`roboverse_pack/tasks/robosuite/robosuite_env.py`): no class-level scenario: cannot be built by the harness
- `robosuite.lift` (`roboverse_pack/tasks/robosuite/robosuite_env.py`): no class-level scenario: cannot be built by the harness
- `robosuite.nut_assembly_square` (`roboverse_pack/tasks/robosuite/robosuite_env.py`): no class-level scenario: cannot be built by the harness
- `robosuite.pick_place_can` (`roboverse_pack/tasks/robosuite/robosuite_env.py`): no class-level scenario: cannot be built by the harness
- `robosuite.stack` (`roboverse_pack/tasks/robosuite/robosuite_env.py`): no class-level scenario: cannot be built by the harness

## Retired robot configs (2026-11-28)

Robot configs whose asset files exist neither on the Hub nor in a checkout, so no backend could ever load them (`tools/assets/check_manifest.py`). Restore from `56a0e67b` together with the assets.

- `roboverse_pack/robots/agibot_a2_cfg.py` (AgibotA2Dof12Cfg)
- `roboverse_pack/robots/allegrohand_cfg.py` (AllegroHandCfg)
- `roboverse_pack/robots/ant_cfg.py` (AntCfg)
- `roboverse_pack/robots/anymal_cfg.py` (AnymalCfg)
- `roboverse_pack/robots/brainco_hand_left_cfg.py` (BraincoHandLeftCfg)
- `roboverse_pack/robots/brainco_hand_right_cfg.py` (BraincoHandRightCfg)
- `roboverse_pack/robots/cartpole_cfg.py` (CartpoleCfg)
- `roboverse_pack/robots/go2_cfg.py` (Go2Cfg)
- `roboverse_pack/robots/inspire_hand_left_cfg.py` (InspireHandLeftCfg)
- `roboverse_pack/robots/inspire_hand_right_cfg.py` (InspireHandRightCfg)
- `roboverse_pack/robots/kinova_gen3_cfg.py` (KinovaGen3Cfg)
- `roboverse_pack/robots/psihand_left_cfg.py` (PsihandLeftCfg)
- `roboverse_pack/robots/psihand_right_cfg.py` (PsihandRightCfg)
- `roboverse_pack/robots/walker_cfg.py` (WalkerCfg)
- `roboverse_pack/robots/g1_feet.py` ()

Also dropped: the `mjx_mjcf_path` of `aloha.py` and `h1_body_collision.py` (files absent).

Second batch, 2026-11-28 evening (`robot check --all` config row): configs whose asset paths point outside the Hub tree
(`data_isaaclab/...`, `roboverse_data_release/...`, a directory that is not in the repository) or declare no asset at
all, so nothing can fetch them. Restore from `56a0e67b` together with the assets, uploaded under `roboverse_data/`.

- `roboverse_pack/robots/aloha.py` (AlohaCfg: no asset path; its bimanual tasks were retired on 2026-11-27)
- `roboverse_pack/robots/iiwa_cfg.py` (IiwaCfg: `data_isaaclab/robots/iiwa/iiwa.usd`; `iiwa14_cfg.py` is the KUKA on the Hub)
- `roboverse_pack/robots/ur5e_2f85_cfg.py` (Ur5E2F85Cfg: `data_isaaclab/robots/UniversalRobots/ur5e/ur5e_2f85_fix.usd`)
- `roboverse_pack/robots/h1_2_without_hand_cfg.py` (H12WithoutHandCfg: `roboverse_data_release/...`)
- `roboverse_pack/robots/vega_cfg.py` (VegaCfg: `roboverse_pack/robots/robots_vega/...`, not in the repository; its
  `torso_j1` default also sat outside the single-point limit)

## Retired on 2026-11-28: the last libero_90 task

`libero_90.kitchen_scene4_close_the_bottom_drawer_of_the_cabinet` (`roboverse_pack/tasks/libero_90/`, restore from
`56a0e67b`) passed the 2026-11-27 gate with an unphysical start: its demo file stores `panda_finger_joint2` in
robosuite's sign (-0.034, range [0, 0.04]), which the gate now reports as `invalid`. With the sign mirrored into range
(the corrected file is in `scratch_poc/demos_fixed/libero90/` on the maintainer's machine, not on the Hub) the demos
replay at 0.78 over 50 episodes, under the 0.80 threshold, so the task and the pack (the BDDL goal predicates it
carried) are retired together. The LIBERO goal machinery stays recoverable with it.

## Retired packs (2026-11-28)

Packs that registered nothing in the task registry and bridged no data: thin wrappers that re-registered another
library's `gym` ids under a RoboVerse prefix (`MetaWorld/`, `MyoSuite/`, `GymRobotics/`, `DMControl/`, `Robosuite/`),
with no tests, no docs page and no suite that scored them. A pack earns its place by registering MetaSim tasks the
gate runs, or by bridging an external benchmark's data or environments into MetaSim formats with tests and a docs
page (the RoboTwin, LIBERO-plus, ManiSkill and mjlab integrations do). Restore from `56a0e67b`.

- `roboverse_pack/tasks/dm_control/`
- `roboverse_pack/tasks/gym_robotics/`
- `roboverse_pack/tasks/metaworld/`
- `roboverse_pack/tasks/myosuite/`
- `roboverse_pack/tasks/robosuite/` (its five native ports were retired on 2026-11-27; the passthrough and the
  `tools/robosuite_integration/` parity harness that served them go with them)
