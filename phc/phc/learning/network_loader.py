import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from phc.utils import torch_utils

from easydict import EasyDict as edict
from phc.learning.vq_quantizer import EMAVectorQuantizer, Quantizer
from phc.learning.pnn import PNN

def load_mcp_mlp(checkpoint, activation = "relu", device = "cpu", mlp_name = "actor_mlp"):
    actvation_func = torch_utils.activation_facotry(activation)
    key_name = f"a2c_network.{mlp_name}"
    
    loading_keys = [k for k in checkpoint['model'].keys() if k.startswith(key_name)]
    if not mlp_name == "composer":
        loading_keys += ["a2c_network.mu.weight", 'a2c_network.mu.bias']
        
    loading_keys_linear = [k for k in loading_keys if k.endswith('weight')]
    
    nn_modules = []
    
    for idx, key in enumerate(loading_keys_linear):
        if len(checkpoint['model'][key].shape) == 1: # layernorm
            layer = torch.nn.LayerNorm(*checkpoint['model'][key].shape[::-1])
            nn_modules.append(layer)
        elif len(checkpoint['model'][key].shape) == 2: # nn
            layer = nn.Linear(*checkpoint['model'][key].shape[::-1])
            nn_modules.append(layer)
            if idx < len(loading_keys_linear) - 1:
                nn_modules.append(actvation_func())
        else:
            raise NotImplementedError
        
    mlp = nn.Sequential(*nn_modules)
    
    if mlp_name == "composer":
        # ZL: shouldn't really have this here, but it's a quick fix for now. 
        mlp.append(actvation_func())
    
    state_dict = mlp.state_dict()
    
    for idx, key_affix in enumerate(state_dict.keys()):
        state_dict[key_affix].copy_(checkpoint['model'][loading_keys[idx]])
    
    for param in mlp.parameters():
        param.requires_grad = False
        
    mlp.to(device)
    mlp.eval()
    
    return mlp

def load_pnn(checkpoint, num_prim, has_lateral, activation = "relu", device = "cpu"):
    state_dict_load = checkpoint['model']
    
    net_key_name = "a2c_network.pnn.actors.0"
    loading_keys = [k for k in checkpoint['model'].keys() if k.startswith(net_key_name) and k.endswith('bias')]
    layer_size = []
    for idx, key in enumerate(loading_keys): 
        layer_size.append(checkpoint['model'][key].shape[::-1][0])
    
    mlp_args = {'input_size': state_dict_load['a2c_network.pnn.actors.0.0.weight'].shape[1], 'units':layer_size[:-1], 'activation': activation, 'dense_func': torch.nn.Linear}
    pnn = PNN(mlp_args, output_size=checkpoint['model']['a2c_network.mu.bias'].shape[0], numCols=num_prim, has_lateral=has_lateral)
    state_dict = pnn.state_dict()
    for k in state_dict_load.keys():
        if "pnn" in k:
            pnn_dict_key = k.split("pnn.")[1]
            state_dict[pnn_dict_key].copy_(state_dict_load[k])
                
    pnn.freeze_pnn(num_prim)
    pnn.to(device)
    return pnn
    

def load_z_encoder(checkpoint, activation = "relu", z_type = "sphere", device = "cpu"):
    net_dict = edict()
    
    actvation_func = torch_utils.activation_facotry(activation)
    if z_type == "sphere" or z_type == "uniform" or z_type == "vq_vae" or z_type == "vae":
        net_key_name = "a2c_network._task_mlp" if "a2c_network._task_mlp.0.weight" in checkpoint['model'].keys() else "a2c_network.z_mlp"
    elif z_type == "hyper":
        net_key_name = "a2c_network.z_mlp"
    else:
        raise NotImplementedError

    loading_keys = [k for k in checkpoint['model'].keys() if k.startswith(net_key_name)]
    actor = load_mlp(loading_keys, checkpoint, actvation_func)
    
    actor.to(device)
    actor.eval()
    
    net_dict.encoder= actor
    if "a2c_network.z_logvar.weight" in checkpoint['model'].keys():
        z_logvar = load_linear('a2c_network.z_logvar', checkpoint=checkpoint)
        z_mu = load_linear('a2c_network.z_mu', checkpoint=checkpoint)
        z_logvar.eval(); z_mu.eval()
        net_dict.z_mu = z_mu.to(device)
        net_dict.z_logvar = z_logvar.to(device)

    return net_dict

def load_mlp(loading_keys, checkpoint, actvation_func, model_key="model"):
    
    loading_keys_linear = [k for k in loading_keys if k.endswith('weight')]
    nn_modules = []
    for idx, key in enumerate(loading_keys_linear):
        if len(checkpoint[model_key][key].shape) == 1: # layernorm
            layer = torch.nn.LayerNorm(*checkpoint[model_key][key].shape[::-1])
            nn_modules.append(layer)
        elif len(checkpoint[model_key][key].shape) == 2: # nn
            layer = nn.Linear(*checkpoint[model_key][key].shape[::-1])
            nn_modules.append(layer)
            if idx < len(loading_keys_linear) - 1:
                nn_modules.append(actvation_func())
        else:
            raise NotImplementedError
        
    net = nn.Sequential(*nn_modules)
    
    state_dict = net.state_dict()
    
    for idx, key_affix in enumerate(state_dict.keys()):
        state_dict[key_affix].copy_(checkpoint[model_key][loading_keys[idx]])
        
    for param in net.parameters():
        param.requires_grad = False

    return net

def load_linear(net_name, checkpoint):
    net = nn.Linear(checkpoint['model'][net_name + '.weight'].shape[1], checkpoint['model'][net_name + '.weight'].shape[0])
    state_dict = net.state_dict()
    state_dict['weight'].copy_(checkpoint['model'][net_name + '.weight'])
    state_dict['bias'].copy_(checkpoint['model'][net_name + '.bias'])
    
    return net

def load_z_decoder(checkpoint, activation = "relu", z_type = "sphere", device = "cpu"):
    actvation_func = torch_utils.activation_facotry(activation)
    key_name = "a2c_network.actor_mlp"
    loading_keys = [k for k in checkpoint['model'].keys() if k.startswith(key_name)] + ["a2c_network.mu.weight", 'a2c_network.mu.bias']
        
    actor = load_mlp(loading_keys, checkpoint, actvation_func)
    
    actor.to(device)
    actor.eval()
    net_dict = edict()
    
    net_dict.decoder= actor
    if z_type == "vq_vae":
        quantizer_weights = checkpoint['model']['a2c_network.quantizer.embedding.weight']
        quantizer = Quantizer(quantizer_weights.shape[0], quantizer_weights.shape[1], beta = 0.25)
        state_dict = quantizer.state_dict()
        state_dict['embedding.weight'].copy_(quantizer_weights)
        
        quantizer.to(device)
        quantizer.eval()
        net_dict.quantizer = quantizer
        
    elif z_type == "vae" and "a2c_network.z_prior.0.weight" in checkpoint['model'].keys():
        prior_loading_keys = [k for k in checkpoint['model'].keys() if k.startswith("a2c_network.z_prior.")]
        z_prior = load_mlp(prior_loading_keys, checkpoint, actvation_func)
        z_prior.append(actvation_func())
        z_prior_mu = load_linear('a2c_network.z_prior_mu', checkpoint=checkpoint)
        
        z_prior.eval(); z_prior_mu.eval()
        net_dict.z_prior = z_prior.to(device)
        net_dict.z_prior_mu = z_prior_mu.to(device)

        if "a2c_network.z_prior_logvar.weight" in checkpoint['model'].keys():
            z_prior_logvar = load_linear('a2c_network.z_prior_logvar', checkpoint=checkpoint)
            z_prior_logvar.eval()
            net_dict.z_prior_logvar = z_prior_logvar.to(device)
    
    return net_dict