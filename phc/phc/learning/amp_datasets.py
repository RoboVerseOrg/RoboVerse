import torch
from rl_games.common import datasets

class AMPDataset(datasets.PPODataset):
    def __init__(self, batch_size, minibatch_size, is_discrete, is_rnn, device, seq_len):
        super().__init__(batch_size, minibatch_size, is_discrete, is_rnn, device, seq_len)
        self._idx_buf = torch.randperm(self.batch_size)
        
        
        
        return
    
    def update_mu_sigma(self, mu, sigma):	  
        raise NotImplementedError()
        return
    
    # def _get_item_rnn(self, idx):
    #     gstart = idx * self.num_games_batch
    #     gend = (idx + 1) * self.num_games_batch
    #     start = gstart * self.seq_len
    #     end = gend * self.seq_len
    #     self.last_range = (start, end)   
    #     input_dict = {}
    #     for k,v in self.values_dict.items():
    #         if k not in self.special_names:
    #             if v is dict:
    #                 v_dict = { kd:vd[start:end] for kd, vd in v.items() }
    #                 input_dict[k] = v_dict
    #             else:
    #                 input_dict[k] = v[start:end]
        
    #     rnn_states = self.values_dict['rnn_states']
    #     input_dict['rnn_states'] = [s[:,gstart:gend,:] for s in rnn_states]
    #     return input_dict
    
    def update_values_dict(self, values_dict, rnn_format = False, horizon_length = 1, num_envs = 1):
        self.values_dict = values_dict     
        self.horizon_length = horizon_length
        self.num_envs = num_envs
        
        if rnn_format and self.is_rnn:
            for k,v in self.values_dict.items():
                if k not in self.special_names and v is not None:
                    self.values_dict[k] = self.values_dict[k].view(self.num_envs,  self.horizon_length, -1).squeeze() # Actions are already swapped to the correct format. 
            if not self.values_dict['rnn_states'] is None:
                self.values_dict['rnn_states'] = [s.reshape(self.num_envs,  self.horizon_length, -1) for s in self.values_dict['rnn_states']] # rnn_states are not swapped in AMP, so do not swap it here. 
            self._idx_buf = torch.randperm(self.num_envs) # Update to only shuffle the envs.
            
    # def _get_item_rnn(self, idx):
    #     data = super()._get_item_rnn(idx)
    #     import ipdb; ipdb.set_trace()
    #     return data
            
    def _get_item_rnn(self, idx):
        # ZL: I am doubling the get_item_rnn function to in a way also get the sequential data. Pretty hacky. 
        # BPTT, input dict is [batch, seqlen, features]. This function return the sequences that are from the same episide and enviornment in sequentila mannar. Not used at the moment since seq_len is set to 1 for RNN right now. 
        step_size = int(self.minibatch_size/self.horizon_length)
        
        start = idx * step_size
        end = (idx + 1) * step_size
        sample_idx = self._idx_buf[start:end]
        
        input_dict = {}
        
        for k,v in self.values_dict.items():
            if k not in self.special_names and v is not None:
                input_dict[k] = v[sample_idx, :].view(step_size * self.horizon_length, -1).squeeze() # flatten to batch size 
        
        input_dict['old_values'] = input_dict['old_values'][:, None] # ZL Hack: following compute assumes that the old_values is [batch, 1], so has to change this back. Otherwise, the loss will be wrong.
        input_dict['returns'] = input_dict['returns'][:, None] # ZL Hack: following compute assumes that the old_values is [batch, 1], so has to change this back. Otherwise, the loss will be wrong.
        
        if not self.values_dict['rnn_states'] is None:
            input_dict['rnn_states'] = [s[sample_idx, :].view(step_size * self.horizon_length, -1) for s in self.values_dict["rnn_states"]]
        
        if (end >= self.batch_size):
            self._shuffle_idx_buf()
            
        
        return input_dict

    def _get_item(self, idx):
        start = idx * self.minibatch_size
        end = (idx + 1) * self.minibatch_size
        sample_idx = self._idx_buf[start:end]
        
        input_dict = {}
        for k,v in self.values_dict.items():
            if k not in self.special_names and v is not None:
                input_dict[k] = v[sample_idx]
                
        if (end >= self.batch_size):
            self._shuffle_idx_buf()

        return input_dict

    def _shuffle_idx_buf(self):
        if self.is_rnn:
            self._idx_buf = torch.randperm(self.num_envs)
        else:
            self._idx_buf[:] = torch.randperm(self.batch_size)
        return