import os
import pickle as pkl
from collections import defaultdict

from tqdm.rich import tqdm_rich as tqdm

input_pkl = "data/source_data/maniskill2/rigid_body/PegInsertionSide-v0/trajectory-unified.pkl"

output_pkl = os.path.join(os.path.dirname(input_pkl), "trajs")
os.makedirs(output_pkl, exist_ok=True)

input_data = pkl.load(open(input_pkl, "rb"))

output_data = defaultdict(list)

franka_dofs = [f"panda_joint{i + 1}" for i in range(7)] + ["panda_finger_joint1", "panda_finger_joint2"]

for i_data, data in enumerate(tqdm(input_data["demos"]["franka"], desc="Processing data")):
    asset_name = str(i_data)  # data["asset_name"]
    output_data[asset_name].append({
        "actions": [
            {"dof_pos_target": dict(zip(franka_dofs, [float(aa) for aa in a]))} for a in data["robot_traj"]["q"]
        ],
        "init_state": {
            # Peg
            "box": {"pos": data["env_setup"]["init_box_pos"], "rot": data["env_setup"]["init_box_quat"]},
            "stick": {"pos": data["env_setup"]["init_stick_pos"], "rot": data["env_setup"]["init_stick_quat"]},
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
