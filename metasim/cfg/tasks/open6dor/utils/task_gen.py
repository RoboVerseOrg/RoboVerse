import glob
import json
import pickle


def dict_to_pkl(data, filename):
    with open(filename, "wb") as f:
        pickle.dump(data, f)


CAT = "task_refine_rot_only"
ROOT = f"/home/haoran/Project/RoboVerse/RoboVerse/Open6DOR/Open6DOR/assets/tasks/{CAT}"
"/home/haoran/Project/RoboVerse/RoboVerse/Open6DOR/Open6DOR/assets/tasks/task_refine_6dof/behind/Place_the_apple_behind_the_bottle_on_the_table.__upright/20240824-165044_no_interaction/task_config_new5.json"
tasks = glob.glob(f"{ROOT}/*/*/*/task_config_new5.json")

current = 0
total = len(tasks)
for task in tasks:
    current += 1
    print(f"{current}/{total}")
    with open(task) as f:
        config = json.load(f)
    obj_names = config["selected_obj_names"]
    obj_urdfs = config["selected_urdfs"]
    instruction = config["instruction"]
    init_obj_pos = config["init_obj_pos"]

    init_state = {}

    init_state["table"] = {
        "pos": [0.5, 0, 0.15],
        "rot": [1, 0, 0, 0],
    }
    for obj_i in range(len(obj_names)):
        rot_ori = init_obj_pos[obj_i][3:7]
        rot = [rot_ori[3], rot_ori[0], rot_ori[1], rot_ori[2]]
        init_state[obj_names[obj_i]] = {
            "pos": init_obj_pos[obj_i][:3],
            "rot": rot,
        }
    init_state["franka"] = {
        "pos": [-0.26765137910842896, -0.0052999998442828655, 0.00031763582956045866],
        "rot": [0.999606728553772, 0.0002215634740423411, 0.0028197357896715403, -0.0061978623270988464],
        "dof_pos": {
            "panda_joint1": 9.572522685630247e-07,
            "panda_joint2": 0.1750054657459259,
            "panda_joint3": 2.8940730771864764e-06,
            "panda_joint4": -0.8729366660118103,
            "panda_joint5": -9.889257853501476e-06,
            "panda_joint6": 1.2216076850891113,
            "panda_joint7": 0.7853957414627075,
            "panda_finger_joint1": 0.0399974063038826,
            "panda_finger_joint2": 0.04000125825405121,
        },
    }
    extra = config
    data = {
        "franka": [
            {
                "actions": [
                    # action 0
                    {
                        "dof_pos_target": {
                            "panda_joint1": -2.3806078388588503e-06,
                            "panda_joint2": 0.1750042736530304,
                            "panda_joint3": 9.092956133827101e-06,
                            "panda_joint4": -0.8729740977287292,
                            "panda_joint5": 5.131112629896961e-06,
                            "panda_joint6": 1.221634864807129,
                            "panda_joint7": 0.7853959798812866,
                            "panda_finger_joint1": 0.03999824821949005,
                            "panda_finger_joint2": 0.04000125080347061,
                        }
                    }
                    # action 1
                    # ...
                ],
                "init_state": init_state,
                "states": [
                    # state 0
                    # state 1
                    # ...
                ],
                "extra": extra,
            }
            # 'actions', 'init_state', 'states'
        ]
    }
    tag4 = task.split("/")[-4]
    tag3 = task.split("/")[-3]
    tag2 = task.split("/")[-2]
    root_new = f"tmp/{CAT}/{tag4}/{tag3}/{tag2}"
    import os

    os.makedirs(root_new, exist_ok=True)
    dict_to_pkl(data, f"{root_new}/trajectory-unified_wo_traj_v2.pkl")


# # take close box as an example
# data_template = {
#     "franka": [
#         {
#             "action": [
#                 # action 0
#                 {
#                     "dof_pos_target": {
#                         "panda_joint1": -2.3806078388588503e-06,
#                         "panda_joint2": 0.1750042736530304,
#                         "panda_joint3": 9.092956133827101e-06,
#                         "panda_joint4": -0.8729740977287292,
#                         "panda_joint5": 5.131112629896961e-06,
#                         "panda_joint6": 1.221634864807129,
#                         "panda_joint7": 0.7853959798812866,
#                         "panda_finger_joint1": 0.03999824821949005,
#                         "panda_finger_joint2": 0.04000125080347061,
#                     }
#                 }
#                 # action 1
#                 # ...
#             ],
#             "init_state": {
#                 "box_base": {
#                     "pos": [0.2565089166164398, 0.1984991729259491, 0.07473278045654297],
#                     "rot": [0.7069969773292542, 0.012462549842894077, -0.7069969773292542, 0.012462549842894077],
#                     "dof_pos": {"box_joint": 2.3649210929870605},
#                 },
#                 "franka": {
#                     "pos": [-0.26765137910842896, -0.0052999998442828655, 0.00031763582956045866],
#                     "rot": [0.999606728553772, 0.0002215634740423411, 0.0028197357896715403, -0.0061978623270988464],
#                     "dof_pos": {
#                         "panda_joint1": 9.572522685630247e-07,
#                         "panda_joint2": 0.1750054657459259,
#                         "panda_joint3": 2.8940730771864764e-06,
#                         "panda_joint4": -0.8729366660118103,
#                         "panda_joint5": -9.889257853501476e-06,
#                         "panda_joint6": 1.2216076850891113,
#                         "panda_joint7": 0.7853957414627075,
#                         "panda_finger_joint1": 0.0399974063038826,
#                         "panda_finger_joint2": 0.04000125825405121,
#                     },
#                 },
#             },
#             "states": [
#                 # state 0
#                 {
#                     "box_base": {
#                         "pos": [0.2565089166164398, 0.1984991729259491, 0.07473278045654297],
#                         "rot": [0.7069969773292542, 0.012462549842894077, -0.7069969773292542, 0.012462549842894077],
#                         "dof_pos": {"box_joint": 2.3649210929870605},
#                     },
#                     "franka": {
#                         "pos": [-0.26765137910842896, -0.0052999998442828655, 0.00031763582956045866],
#                         "rot": [
#                             0.999606728553772,
#                             0.0002215634740423411,
#                             0.0028197357896715403,
#                             -0.0061978623270988464,
#                         ],
#                         "dof_pos": {
#                             "panda_joint1": -2.3806078388588503e-06,
#                             "panda_joint2": 0.1750042736530304,
#                             "panda_joint3": 9.092956133827101e-06,
#                             "panda_joint4": -0.8729740977287292,
#                             "panda_joint5": 5.131112629896961e-06,
#                             "panda_joint6": 1.221634864807129,
#                             "panda_joint7": 0.7853959798812866,
#                             "panda_finger_joint1": 0.03999824821949005,
#                             "panda_finger_joint2": 0.04000125080347061,
#                         },
#                     },
#                 }
#                 # state 1
#                 # ...
#             ],
#         }
#         #'actions', 'init_state', 'states'
#     ]
# }


# dict_to_pkl(data_template, "data_template.pkl")
