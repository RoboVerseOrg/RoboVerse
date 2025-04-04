import os
import pickle as pkl
from collections import defaultdict

from tqdm.rich import tqdm_rich as tqdm

# input_pkl = "data/source_data/maniskill2/rigid_body/PickSingleYCB-v0/trajectory-unified.pkl"
# input_pkl = "data/source_data/maniskill2/rigid_body/PlugCharger-v0/trajectory-unified.pkl"
input_pkl = "data/source_data/maniskill2/rigid_body/PegInsertionSide-v0/trajectory-unified.pkl"

output_pkl = os.path.join(os.path.dirname(input_pkl), "trajs")
os.makedirs(output_pkl, exist_ok=True)

input_data = pkl.load(open(input_pkl, "rb"))

output_data = defaultdict(list)

franka_dofs = [f"panda_joint{i + 1}" for i in range(7)] + ["panda_finger_joint1", "panda_finger_joint2"]

for data in tqdm(input_data["demos"]["franka"], desc="Processing data"):
    asset_name = data["asset_name"]
    output_data[asset_name].append({
        "actions": [
            {"dof_pos_target": dict(zip(franka_dofs, [float(aa) for aa in a]))} for a in data["robot_traj"]["q"]
        ],
        "init_state": {
            # Charger
            # "base": {"pos": data["env_setup"]["init_base_pos"], "rot": data["env_setup"]["init_base_quat"]},
            # "charger": {"pos": data["env_setup"]["init_charger_pos"], "rot": data["env_setup"]["init_charger_quat"]},
            # Peg
            "base": {"pos": data["env_setup"]["init_base_pos"], "rot": data["env_setup"]["init_base_quat"]},
            "stick": {"pos": data["env_setup"]["init_charger_pos"], "rot": data["env_setup"]["init_charger_quat"]},
            # EGAD, YCB
            # "obj": {"pos": data["env_setup"]["init_object_pos"], "rot": data["env_setup"]["init_object_quat"]},
            "franka": {
                "pos": data["env_setup"]["init_robot_pos"],
                "rot": data["env_setup"]["init_robot_quat"],
                "dof_pos": dict(zip(franka_dofs, [float(aa) for aa in data["robot_traj"]["q"][0]])),
            },
        },
    })

for asset_name, data in tqdm(output_data.items(), desc="Saving data"):
    tqdm.write(f"Dumped {len(data)} trajectories for {asset_name}")
    pkl.dump({"franka": data}, open(os.path.join(output_pkl, f"trajectory-franka-{asset_name}_v2.pkl"), "wb"))
