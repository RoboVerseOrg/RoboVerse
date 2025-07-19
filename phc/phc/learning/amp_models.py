# This is the overall forward pass of the model. 

import torch.nn as nn
from rl_games.algos_torch.models import ModelA2CContinuousLogStd
import torch
class ModelAMPContinuous(ModelA2CContinuousLogStd):
    def __init__(self, network):
        super().__init__(network)
        return

    def build(self, config):
        net = self.network_builder.build('amp', **config)
        for name, _ in net.named_parameters():
            print(name)
        return ModelAMPContinuous.Network(net)

    class Network(ModelA2CContinuousLogStd.Network):
        def __init__(self, a2c_network):
            super().__init__(a2c_network)

            return

        def forward(self, input_dict):
            is_train = input_dict.get('is_train', True)
            amp_steps = input_dict.get("amp_steps", 2)
            
            
            result = super().forward(input_dict)
            
            if (is_train):
                amp_obs, amp_obs_replay, amp_demo_obs = input_dict['amp_obs'],  input_dict['amp_obs_replay'], input_dict['amp_obs_demo']
                
                disc_agent_logit = self.a2c_network.eval_disc(amp_obs)
                result["disc_agent_logit"] = disc_agent_logit

                disc_agent_replay_logit = self.a2c_network.eval_disc(amp_obs_replay)
                result["disc_agent_replay_logit"] = disc_agent_replay_logit

                disc_demo_logit = self.a2c_network.eval_disc(amp_demo_obs)
                result["disc_demo_logit"] = disc_demo_logit

                # # HACK....
                # if input_dict.get("compute_direct_logit", False):
                #     from phc.utils.torch_utils import project_to_norm
                #     import ipdb; ipdb.set_trace()
                #     mus = project_to_norm(result['mus'], input_dict.get("embedding_norm", 1.0))
                #     mus = mus.view(-1, 32, 64)
                #     mus = mus.reshape(-1, 2048)
                #     result['disc_direct_logit'] = self.a2c_network.eval_disc(mus)
                
                
                # amp_obs.requires_grad_(True)
                # disc_agent_logit = self.a2c_network.eval_disc(amp_obs)
                # import ipdb; ipdb.set_trace()
                # torch.autograd.grad(disc_agent_logit, amp_obs, grad_outputs=torch.ones_like(disc_agent_logit), create_graph=False, retain_graph=True, only_inputs=True)
                # torch.autograd.grad(disc_agent_replay_logit, amp_obs_replay, grad_outputs=torch.ones_like(disc_agent_replay_logit), create_graph=False, retain_graph=True, only_inputs=True)
                # torch.autograd.grad(disc_demo_logit, amp_demo_obs, grad_outputs=torch.ones_like(disc_demo_logit), create_graph=False, retain_graph=True, only_inputs=True)
                # (1 / (1 + torch.exp(-disc_demo_logit)))[:50]

            return result

        def dropout_amp_obs(self, amp_obs, dropout_mask):
            return amp_obs * dropout_mask

        def get_dropout_mask(self,
                             amp_obs,
                             steps,
                             num_masks=3,
                             dropout_rate=0.3):
            # ZL Hack: amp_obs_dims, should drop out whole joints
            # [root_rot 6, root_vel 3, root_ang_vel 3, dof_pos 23 * 6 - 4 * 6, dof_vel 69 - 12, key_body_pos 3 * 4, shape_obs_disc 11]
            # [root_rot 6, root_vel 3, root_ang_vel 3, dof_pos 23 * 6 - 4 * 6, dof_vel 69 - 12, key_body_pos 3 * 4, shape_obs_disc 47]
            # 6 + 3 + 3 + 19 * 6 + 19 * 3 + 3 * 4 + 11 = 206
            # 6 + 3 + 3 + 19 * 6 + 19 * 3 + 3 * 4  = 195 # mean body
            # 6 + 3 + 3 + 19 * 6 + 19 * 3 + 3 * 4  = 196 # mean body + height
            # 1 + 6 + 3 + 3 + 19 * 6 + 19 * 3 + 3 * 4  + 11 = 207 # shape body + height
            # 6 + 3 + 3 + 19 * 6 + 19 * 3 + 3 * 4 + 10 = 205 # concise limb weight
            # 6 + 3 + 3 + 19 * 6 + 19 * 3 + 3 * 4 + 47 = 242 # full limb weight
            # 6 + 3 + 3 + 19 * 6 + 19 * 3 + 3 * 4 + 59 = 254 - masterfoot
            B, F = amp_obs.shape
            B, _, amp_f = amp_obs.view(B, steps, -1).shape
            try:
                assert (F / steps == 205 or F / steps == 254 or F / steps == 242 or F / steps == 206 or F / steps == 197 or F / steps == 188 or F / steps == 195 or F / steps == 196 or F / steps == 207)
            except:
                print(F/steps)
                import ipdb; ipdb.set_trace()
                print(F/steps)

            dof_joints_offset = 12 # 6 + 3 + 3
            num_joints = 19

            if F / steps == 197: # Remove neck
                num_joints = 18
            elif F / steps == 188:  # Remove hands
                num_joints = 17
            elif F / steps == 196 or F / steps == 207:
                dof_joints_offset = 13 # 1 + 6 + 3 + 3

            dof_vel_offsets = dof_joints_offset + num_joints * 6  # 12 + 19 * 6

            dropout_mask = torch.ones([B, amp_f, num_masks])

            for idx_joint in range(num_joints):
                has_drop_out = torch.rand(B, num_masks) > dropout_rate
                dropout_mask[:, dof_joints_offset + idx_joint * 6 : dof_joints_offset + idx_joint * 6 + 6, :] = has_drop_out[:, None]
                dropout_mask[:, dof_vel_offsets + idx_joint * 3 : dof_vel_offsets + idx_joint * 3 + 3, :] = has_drop_out[:, None]
            return dropout_mask.repeat(1, steps, 1).to(amp_obs)


