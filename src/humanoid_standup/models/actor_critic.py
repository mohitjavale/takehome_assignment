import torch
import torch.nn as nn
from skrl.models.torch import Model, GaussianMixin, DeterministicMixin

def get_activation(name):
    if name is None or str(name).lower() == "none":
        return None
        
    name_lower = str(name).lower()
    if name_lower == "elu": return nn.ELU()
    elif name_lower == "relu": return nn.ReLU()
    elif name_lower == "tanh": return nn.Tanh()
    else: raise ValueError(f"Unsupported activation: {name}")

class Policy(GaussianMixin, Model):
    def __init__(self, observation_space, action_space, device, model_cfg, clip_actions=True):
        Model.__init__(self, observation_space, action_space, device)
        GaussianMixin.__init__(self, clip_actions)
        
        layers = []
        curr_dim = self.num_observations
        for h in model_cfg.layers:
            layers.append(nn.Linear(curr_dim, h))
            hidden_act = get_activation(model_cfg.hidden_activation)
            if hidden_act:
                layers.append(hidden_act)
            curr_dim = h
            
        layers.append(nn.Linear(curr_dim, self.num_actions))
        
        output_act = get_activation(model_cfg.get("output_activation", "none"))
        if output_act:
            layers.append(output_act)
            
        self.net = nn.Sequential(*layers)
        self.log_std_parameter = nn.Parameter(torch.zeros(self.num_actions))

    def compute(self, inputs, role):
        return self.net(inputs["states"]), self.log_std_parameter, {}

class Value(DeterministicMixin, Model):
    def __init__(self, observation_space, action_space, device, model_cfg):
        Model.__init__(self, observation_space, action_space, device)
        DeterministicMixin.__init__(self)
        
        layers = []
        curr_dim = self.num_observations
        for h in model_cfg.layers:
            layers.append(nn.Linear(curr_dim, h))
            hidden_act = get_activation(model_cfg.hidden_activation)
            if hidden_act:
                layers.append(hidden_act)
            curr_dim = h
            
        layers.append(nn.Linear(curr_dim, 1))
        
        output_act = get_activation(model_cfg.get("output_activation", "none"))
        if output_act:
            layers.append(output_act)

        self.net = nn.Sequential(*layers)

    def compute(self, inputs, role):
        return self.net(inputs["states"]), {}