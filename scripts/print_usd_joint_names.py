import argparse

from omni.isaac.lab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--usd_path", type=str, required=True)
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
args.headless = True
app_launcher = AppLauncher(args)
simulation_app = app_launcher.app

###########################################################
import omni.isaac.lab.sim as sim_utils
from omni.isaac.lab.assets import Articulation, ArticulationCfg

obj = Articulation(
    ArticulationCfg(
        prim_path="/World/test",
        spawn=sim_utils.UsdFileCfg(usd_path=args.usd_path),
        actuators={},
    ),
)
sim_cfg = sim_utils.SimulationCfg(dt=0.01, device="cpu")
sim = sim_utils.SimulationContext(sim_cfg)
sim.reset()

print("=" * 100)
print(obj.joint_names)
print("=" * 100)
joint_limits = obj.root_physx_view.get_dof_limits().squeeze(0).tolist()
# joint_qpos = obj.root_physx_view.get_dof_positions()
# print(joint_qpos)
# print(obj.root_physx_view.get_dof_position_targets())
for joint_name, joint_limit in zip(obj.joint_names, joint_limits):
    print(f"{joint_name}: ({joint_limit[0]:.4f}, {joint_limit[1]:.4f})")

print("=" * 100)
print(obj.body_names)
print("=" * 100)
