#!/usr/bin/env python3

import json
import os
import pickle
import re

import emoji
import openai
from colorama import Fore, Style

# ======================================
# 1. Configuration & Setup
# ======================================

openai.api_key = ""

OBJECT_LIST_JSON = "metasim/cfg/tasks/gpt/config/rigid_objects_init_list.json"
ROBOT_LIST_JSON = "metasim/cfg/tasks/gpt/config/robots_init_list.json"
TASKS_OUTPUT_FOLDER = "metasim/cfg/tasks/gpt/config/tasks"

# Where to save final PKLs
PKL_OUTPUT_BASE = "roboverse_data/trajs/gpt"

# Where to save the metacfg .py files
METACFG_OUTPUT_FOLDER = "metasim/cfg/tasks/gpt/metacfg"


# -------------- JSON LOADING UTILS --------------


def load_all_objects_data():
    """Load the entire objects_init_list.json and return as a dict."""
    if not os.path.isfile(OBJECT_LIST_JSON):
        raise FileNotFoundError(f"Cannot find {OBJECT_LIST_JSON}")
    with open(OBJECT_LIST_JSON, encoding="utf-8") as f:
        data = json.load(f)
    return data


def load_all_robots_data():
    """Load the entire robots_init_list.json and return as a dict."""
    if not os.path.isfile(ROBOT_LIST_JSON):
        raise FileNotFoundError(f"Cannot find {ROBOT_LIST_JSON}")
    with open(ROBOT_LIST_JSON, encoding="utf-8") as f:
        data = json.load(f)
    return data


def load_available_objects():
    """Return list of object names (keys) from object_init_list.json."""
    return list(load_all_objects_data().keys())


def load_available_robots():
    """Return list of robot names (keys) from robot_init_list.json."""
    return list(load_all_robots_data().keys())


# ======================================
# Utility: Conversions of naming
# ======================================


def to_snake_case(s: str) -> str:
    """Convert a string to snake_case. E.g., 'Sauce Pyramid' -> 'sauce_pyramid'."""
    s = re.sub(r"[^0-9a-zA-Z]+", " ", s)
    return "_".join(s.lower().split())


def to_camel_case(s: str) -> str:
    """Convert a string to CamelCase. E.g., 'Sauce Pyramid' -> 'SaucePyramid'."""
    s = re.sub(r"[^0-9a-zA-Z]+", " ", s)
    return "".join(word.capitalize() for word in s.split())


# ======================================
# 2. First GPT Call (Partial Task JSON)
# ======================================


def call_gpt_to_generate_task(user_prompt, object_list, robot_list):
    """
    GPT must output strictly valid JSON:
      {
        "task_name": "...",
        "task_language_instruction": "...",
        "robot_involved": [...],
        "objects_involved": [...]
      }
    """
    system_instructions = (
        "You are a helpful assistant that invents an interesting tabletop-manipulation task.\n"
        "We have the following objects:\n"
        f"{object_list}\n\n"
        "We have the following robots:\n"
        f"{robot_list}\n\n"
        "You must:\n"
        "1) Pick exactly one or more robots from the above.\n"
        "2) Pick zero or more objects from the above.\n"
        "3) Invent a short, unique task name.\n"
        "4) Write a one or two sentence 'task_language_instruction' describing an unusual or amusing task.\n"
        "   - Example_1: 'Put the basket upside down to cover the orange juice like a hat.'\n"
        "   - Example_2: 'Place butter, chocolate pudding and milk in a line and knock them over like dominoes.'\n"
        "   - Keep it short, no more than 2 sentences.\n"
        "5) Output strictly valid JSON **only** in the following format:\n"
        "{\n"
        '  "task_name": "...",\n'
        '  "task_language_instruction": "...",\n'
        '  "robot_involved": [...],\n'
        '  "objects_involved": [...]\n'
        "}\n\n"
        "Constraints:\n"
        "- No extraneous keys.\n"
        "- Do not wrap in triple backticks.\n"
        "- The 'task_name' must be unique.\n"
        "- The 'task_language_instruction' must be 1-2 sentences at most.\n"
    )

    messages = [
        {"role": "system", "content": system_instructions},
        {"role": "user", "content": user_prompt},
    ]

    response = openai.ChatCompletion.create(
        model="gpt-4o",
        messages=messages,
        temperature=0.7,
        max_tokens=15000,
    )
    content = response.choices[0].message["content"].strip()

    # Remove triple backticks if GPT includes them
    if content.startswith("```"):
        if content.endswith("```"):
            content = content[3:-3]
        else:
            content = content[3:]
        content = content.replace("json", "").strip()

    # Convert to dict
    data = json.loads(content)
    return data


# ======================================
# 3. Second GPT Call: Append "init_state" to Partial JSON
# ======================================


def call_gpt_to_get_init_state(partial_task_json, all_objects_data, all_robots_data):
    """
    GPT merges:
      - The chosen robot(s) (with known pos, rot, dof_pos from all_robots_data).
      - The chosen objects (plus any decorative objects, if GPT desires).
        For each object: x,y in [-0.5, 0.5], z and rot from the library.

    GPT outputs strictly valid JSON:
    {
      "init_state": {
        "robotName": {
          "pos": [...],
          "rot": [...],
          "dof_pos": {...}
        },
        "object1": {
          "pos": [...],
          "rot": [...]
        },
        ...
      }
    }
    """
    task_name = partial_task_json["task_name"]
    robot_involved = partial_task_json["robot_involved"]
    objects_involved = partial_task_json["objects_involved"]
    task_instr = partial_task_json["task_language_instruction"]

    # Condense objects
    condensed_objs = {}
    for name, data in all_objects_data.items():
        condensed_objs[name] = {
            "z": data["init_state"]["pos"][2],
            "rot": data["init_state"]["rot"],
            "filepath": data["filepath"],
        }

    # Condense robots
    condensed_robs = {}
    for rname, rdata in all_robots_data.items():
        condensed_robs[rname] = {"pos": rdata["pos"], "rot": rdata["rot"], "dof_pos": rdata["dof_pos"]}

    # GPT Prompt
    system_instructions = (
        "You are a helpful assistant that finalizes the 'init_state' for a RoboVerse-like scene.\n"
        "We have a partial task specification:\n"
        f'  task_name: "{task_name}"\n'
        f'  task_language_instruction: "{task_instr}"\n'
        f"  robot_involved: {robot_involved}\n"
        f"  objects_involved: {objects_involved}\n\n"
        "We have a library of all possible objects with fixed z & rot:\n"
        f"{json.dumps(condensed_objs, indent=2)}\n\n"
        "We have these possible robots:\n"
        f"{json.dumps(condensed_robs, indent=2)}\n\n"
        "Your job:\n"
        "1) Possibly add more 'decorative' objects (any from the object library) if you want.\n"
        "2) For each object (required + decorative), pick x,y in [-0.5, 0.5].\n"
        "   Then combine that with the known z,rot from the library.\n"
        "3) For each chosen robot, just copy its pos,rot,dof_pos from the library exactly.\n"
        "4) Output strictly valid JSON with exactly one top-level key:\n"
        '   "init_state" : { ... }.\n'
        '5) Inside "init_state", each key is either a robot name or object name.\n'
        '   The value is { "pos": [x,y,z], "rot": [w,x,y,z], (optionally "dof_pos" if it is the robot) }.\n'
        "6) No extra keys.\n"
        "7) Do NOT wrap in triple backticks.\n"
        '8) No actions, no states, no extra in this JSON—only the "init_state" dict.\n'
    )

    messages = [
        {"role": "system", "content": system_instructions},
    ]

    response = openai.ChatCompletion.create(
        model="gpt-4o",
        messages=messages,
        temperature=0.7,
        max_tokens=15000,
    )
    content = response.choices[0].message["content"].strip()

    if content.startswith("```"):
        if content.endswith("```"):
            content = content[3:-3]
        else:
            content = content[3:]
        content = content.replace("json", "").strip()

    data = json.loads(content)
    return data


# ======================================
# 4. Write Cfg .py
# ======================================
def write_cfg_file(final_json, object_library):
    """
    Generate a Python config file in metasim/cfg/tasks/gpt/metacfg/{snake_task_name}.py
    Class name is {CamelCaseTaskName}Cfg, but the final run command will just
    use {CamelCaseTaskName} (without "Cfg").
    """
    task_name = final_json["task_name"]
    # We'll keep the class name as {CamelCase}+Cfg
    snake_task_name = to_snake_case(task_name)
    camel_task_name = to_camel_case(task_name)
    metacfg_class_name = camel_task_name + "Cfg"

    # The .py file we want to write
    out_py = os.path.join(METACFG_OUTPUT_FOLDER, f"{snake_task_name}.py")

    # The objects we want to define in 'objects = [...]':
    robots_involved = final_json.get("robot_involved", [])
    init_state_data = final_json["init_state"]

    object_list_entries = []
    for name in init_state_data.keys():
        # skip if the key is a robot
        if name in robots_involved:
            continue
        if name not in object_library:
            print(f"Warning: object '{name}' not found in library; skipping in metacfg.")
            continue
        filepath = object_library[name]["filepath"]
        entry = (
            "RigidObjCfg(\n"
            f'            name="{name}",\n'
            "            physics=PhysicStateType.RIGIDBODY,\n"
            f'            usd_path="{filepath}",\n'
            "        )"
        )
        object_list_entries.append(entry)

    objects_block = "    objects = [\n"
    for idx, e in enumerate(object_list_entries):
        if idx == 0:
            objects_block += f"        {e}"
        else:
            objects_block += f",\n        {e}"
    objects_block += "\n    ]\n"

    # The trajectory file path
    snake_name = snake_task_name
    traj_path = f"roboverse_data/trajs/gpt/{snake_name}/franka_v2.pkl"

    # The .py file content
    py_content = f"""import math
from metasim.cfg.checkers import DetectedChecker, RelativeBboxDetector
from metasim.cfg.objects import ArticulationObjCfg, RigidObjCfg
from metasim.constants import BenchmarkType, PhysicStateType, TaskType
from metasim.utils import configclass

from metasim.cfg.tasks.base_task_cfg import BaseTaskCfg


@configclass
class {metacfg_class_name}(BaseTaskCfg):
    source_benchmark = BenchmarkType.GPT
    task_type = TaskType.TABLETOP_MANIPULATION
    episode_length = 250

{objects_block}
    traj_filepath = "{traj_path}"
"""

    os.makedirs(METACFG_OUTPUT_FOLDER, exist_ok=True)
    with open(out_py, "w", encoding="utf-8") as f:
        f.write(py_content)

    return out_py, snake_task_name, metacfg_class_name, camel_task_name


# ======================================
# 5. Main Workflow
# ======================================


def main():
    # Step A: Greet the user.
    print(Fore.YELLOW + emoji.emojize("🔥 What can I help you with today? ✨") + Style.RESET_ALL)
    # print("What can I help you with today?")
    user_prompt = input("> ").strip()
    if not user_prompt:
        user_prompt = "Please generate an interesting task for me."

    # Step B: Load available objects/robots (names) for GPT's first call
    available_objects = load_available_objects()
    available_robots = load_available_robots()

    # Step C: Call GPT => partial task JSON
    partial_task = call_gpt_to_generate_task(user_prompt, available_objects, available_robots)

    # Validate partial JSON
    for key in ["task_name", "task_language_instruction", "robot_involved", "objects_involved"]:
        if key not in partial_task:
            raise ValueError(f"GPT missing key '{key}' in partial_task.")

    # Step D: Append "init_state" by calling GPT again
    all_objs = load_all_objects_data()
    all_robs = load_all_robots_data()
    init_state_data = call_gpt_to_get_init_state(partial_task, all_objs, all_robs)
    if "init_state" not in init_state_data:
        raise ValueError("GPT did not provide 'init_state' top-level in second call.")

    partial_task["init_state"] = init_state_data["init_state"]

    # Step E: Write final JSON
    snake_task_name = to_snake_case(partial_task["task_name"])
    final_json_path = os.path.join(TASKS_OUTPUT_FOLDER, f"{snake_task_name}.json")
    with open(final_json_path, "w", encoding="utf-8") as f:
        json.dump(partial_task, f, indent=2, ensure_ascii=False)

    # Step F: Convert to PKL with RoboVerse structure
    roverse_data = {}
    for robot_name in partial_task["robot_involved"]:
        if robot_name not in partial_task["init_state"]:
            raise ValueError(f"Robot '{robot_name}' not found in final init_state JSON.")
        dof_pos_dict = partial_task["init_state"][robot_name].get("dof_pos", {})
        zero_dof_target = {joint: 0.0 for joint in dof_pos_dict.keys()}

        roverse_data[robot_name] = [
            {
                "actions": [{"dof_pos_target": zero_dof_target}],
                "init_state": partial_task["init_state"],
                "states": [],
                "extra": None,
            }
        ]

    pkl_folder = os.path.join(PKL_OUTPUT_BASE, snake_task_name)
    os.makedirs(pkl_folder, exist_ok=True)
    pkl_path = os.path.join(pkl_folder, "franka_v2.pkl")

    with open(pkl_path, "wb") as f:
        pickle.dump(roverse_data, f)

    # Step G: Write metacfg
    metacfg_path, snake_task_name_out, metacfg_class_name, camel_task_name = write_cfg_file(partial_task, all_objs)

    # Step H: Print final info
    print("\n" + Fore.GREEN + emoji.emojize("🚀 The task has been generated! 🎉") + Style.RESET_ALL)
    # print("\nThe task has been generated!\n")

    # print(f"Task Name: {partial_task['task_name']}")

    # print(f"Task Language Instruction: {partial_task['task_language_instruction']}\n")

    # print(f"Task JSON saved to:\n  {final_json_path}")

    # print(f"Task PKL saved to:\n  {pkl_path}")

    # print(f"Task Cfg Python file saved to:\n  {metacfg_path}\n")

    # print("Please add this line in metasim/cfg/tasks/__init__.py to import the new metacfg class:")

    # print(f"  from .gpt.metacfg.{snake_task_name_out} import {metacfg_class_name}\n")

    # # Here: the user wants to run with the CamelCase name WITHOUT "Cfg" appended
    # # So the final run command is: --task=SOME_CAMEL_CASE, not --task=...Cfg
    # print("You can replay your task by running this command line (no 'Cfg' in the name):")
    # print(f"  python metasim/scripts/replay_demo.py --sim=isaaclab --task={camel_task_name} --num_envs 1\n")

    print(Fore.CYAN + "🔹 Task Name: " + Style.BRIGHT + partial_task["task_name"] + Style.RESET_ALL)

    print(
        Fore.MAGENTA
        + "📝 Task Language Instruction: "
        + Style.BRIGHT
        + partial_task["task_language_instruction"]
        + Style.RESET_ALL
        + "\n"
    )

    print(Fore.BLUE + "📁 Task JSON saved to:" + Style.RESET_ALL)
    print(Fore.YELLOW + f"  {final_json_path}" + Style.RESET_ALL)

    print(Fore.BLUE + "📦 Task PKL saved to:" + Style.RESET_ALL)
    print(Fore.YELLOW + f"  {pkl_path}" + Style.RESET_ALL)

    print(Fore.BLUE + "🔧 Task Cfg Python file saved to:" + Style.RESET_ALL)
    print(Fore.YELLOW + f"  {metacfg_path}" + Style.RESET_ALL + "\n")

    print(
        Fore.RED
        + emoji.emojize("⚠️ Please add this line in metasim/cfg/tasks/gpt/__init__.py to import the new metacfg class:")
        + Style.RESET_ALL
    )
    print(Fore.WHITE + f"  from .metacfg.{snake_task_name_out} import {metacfg_class_name}" + Style.RESET_ALL + "\n")

    print(
        Fore.GREEN
        + emoji.emojize("🎮 You can replay your task by running this command (no 'Cfg' in the name):")
        + Style.RESET_ALL
    )
    print(
        Fore.WHITE
        + f"  python metasim/scripts/replay_demo.py --sim=isaaclab --task=gpt:{camel_task_name} --num_envs 1"
        + Style.RESET_ALL
        + "\n"
    )


if __name__ == "__main__":
    main()
