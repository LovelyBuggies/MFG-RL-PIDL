import numpy as np
from torch import nn
import torch


class Multiply(nn.Module):
    def __init__(self, scale):
        super(Multiply, self).__init__()
        self.scale = scale

    def forward(self, tensors):
        return self.scale * tensors

class Normalization(nn.Module):
    def __init__(self, mean, std, device):
        super(Normalization, self).__init__()
        mean = np.array(mean)
        std = np.array(std)
        self.device = device
        self.mean = torch.from_numpy(mean).float().to(self.device)
        self.std = torch.from_numpy(std).float().to(self.device)


    def forward(self, tensors):
        # TODO NEED TO MODIFY
        norm_tensor = (tensors - self.mean)/ self.std

        return norm_tensor

def instantiate_activation_function(function_name):
    function_dict = {
        "leaky_relu": nn.LeakyReLU(),
        "tanh": nn.Tanh(),
        "sigmoid": nn.Sigmoid(),
        "none": None
    }
    return function_dict[function_name]


def get_fully_connected_layer(input_dim, output_dim, n_hidden, hidden_dim,
                              activation_type="leaky_relu",
                              last_activation_type="tanh",
                              device=None,
                              mean=0,
                              std=1):
    modules = [ Normalization(mean, std, device), nn.Linear(input_dim, hidden_dim, device=device)]
    activation = instantiate_activation_function(activation_type)
    if activation is not None:
        modules.append(activation)

    # add hidden layers
    if n_hidden > 1:
        for l in range(n_hidden-1):
            modules.append(nn.Linear(hidden_dim, hidden_dim, device=device))
            activation = instantiate_activation_function(activation_type)
            if activation is not None:
                modules.append(activation)

    # add the last layer

    modules.append(nn.Linear(hidden_dim, output_dim, device=device))
    last_activation = instantiate_activation_function(last_activation_type)

    modules.append(Multiply(0.5))
    if last_activation_type == "none":
        pass
    else:
        modules.append(last_activation)
    modules.append(Multiply(2))
    # the "mulltiply 2" is to stretch the range of the activation funtion (e.g. sigmoid) for 2 times long
    # the "multiply 0.5" is the stretch the x-axis accordingly.



    return nn.Sequential(*modules)