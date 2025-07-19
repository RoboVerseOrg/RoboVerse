import time
import torch
import phc.env.tasks.humanoid_im as humanoid_im

from isaacgym.torch_utils import *
from phc.utils.flags import flags
from rl_games.algos_torch import torch_ext
import torch.nn as nn
from phc.learning.pnn import PNN
from collections import deque
from phc.learning.network_loader import load_mcp_mlp, load_pnn

class HumanoidImMCP(humanoid_im.HumanoidIm):

    def __init__(self, cfg, sim_params, physics_engine, device_type, device_id, headless):
        self.num_prim = cfg["env"].get("num_prim", 3)
        self.discrete_mcp = cfg["env"].get("discrete_moe", False)
        self.has_pnn = cfg["env"].get("has_pnn", False)
        self.has_lateral = cfg["env"].get("has_lateral", False)
        self.z_activation = cfg["env"].get("z_activation", "relu")

        super().__init__(cfg=cfg, sim_params=sim_params, physics_engine=physics_engine, device_type=device_type, device_id=device_id, headless=headless)

        if self.has_pnn:
            assert (len(self.models_path) == 1)
            pnn_ck = torch_ext.load_checkpoint(self.models_path[0])
            self.pnn = load_pnn(pnn_ck, num_prim = self.num_prim, has_lateral = self.has_lateral, activation = self.z_activation, device = self.device)
            self.running_mean, self.running_var = pnn_ck['running_mean_std']['running_mean'], pnn_ck['running_mean_std']['running_var']
        
        self.fps = deque(maxlen=90)
        
        return

    def _setup_character_props(self, key_bodies):
        super()._setup_character_props(key_bodies)
        self._num_actions = self.num_prim
        return

    def get_task_obs_size_detail(self):
        task_obs_detail = super().get_task_obs_size_detail()
        task_obs_detail['num_prim'] = self.num_prim
        return task_obs_detail

    def step(self, weights):

        # if self.dr_randomizations.get('actions', None):
        #     actions = self.dr_randomizations['actions']['noise_lambda'](actions)
        # if flags.server_mode:
            # t_s = time.time()
        
        with torch.no_grad():
            # Apply trained Model.
            curr_obs = ((self.obs_buf - self.running_mean.float()) / torch.sqrt(self.running_var.float() + 1e-05))
            
            curr_obs = torch.clamp(curr_obs, min=-5.0, max=5.0)
            if self.discrete_mcp:
                max_idx = torch.argmax(weights, dim=1)
                weights = torch.nn.functional.one_hot(max_idx, num_classes=self.num_prim).float()
            
            if self.has_pnn:
                _, actions = self.pnn(curr_obs)
                
                x_all = torch.stack(actions, dim=1)
            else:
                x_all = torch.stack([net(curr_obs) for net in self.actors], dim=1)
            # print(weights)
            actions = torch.sum(weights[:, :, None] * x_all, dim=1)
                
        # actions = x_all[:, 3]  # Debugging
        # apply actions
        self.pre_physics_step(actions)

        # step physics and render each frame
        self._physics_step()

        # to fix!
        if self.device == 'cpu':
            self.gym.fetch_results(self.sim, True)

        # compute observations, rewards, resets, ...
        self.post_physics_step()
        # if flags.server_mode:
        #     dt = time.time() - t_s
        #     print(f'\r {1/dt:.2f} fps', end='')
            
        # dt = time.time() - t_s
        # self.fps.append(1/dt)
        # print(f'\r {np.mean(self.fps):.2f} fps', end='')
        

        if self.dr_randomizations.get('observations', None):
            self.obs_buf = self.dr_randomizations['observations']['noise_lambda'](self.obs_buf)
