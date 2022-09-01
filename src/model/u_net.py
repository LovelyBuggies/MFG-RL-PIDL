from torch import nn
import torch
import math
import numpy as np
from torch._C import device
from src.layers.fully_connected import get_fully_connected_layer

class u_net(nn.Module):
    def __init__(self, data_loader, device,
                net_args, net_kwargs):
        super(u_net, self).__init__()

        # hardcode: force the first s-net to have a tanh activation function
        #s_kwargs["last_activation_type"] = "tanh"
        #s[0] = get_fully_connected_layer(*s_args, **s_kwargs)
        self.device = device

        self.x_t = np.stack((data_loader.test_data["x_of_u"], data_loader.test_data["t_of_u"]), axis=1)

        self.mean = np.mean(self.x_t, axis=0)
        self.std = np.std(self.x_t, axis=0)
        net_kwargs["mean"] = self.mean
        net_kwargs["std"] = self.std
        self.net_forward = torch.nn.ModuleList([get_fully_connected_layer(*net_args, **net_kwargs)])
        # self.mean_rho = torch.from_numpy(np.asarray(self.batch["mean_rho"])).float().to(self.device)
        # self.std_rho = torch.from_numpy(np.asarray(self.batch["std_rho"])).float().to(self.device)
        # self.mean_V =  torch.from_numpy(np.asarray(self.batch["mean_V"])).float().to(self.device)
        # self.std_V = torch.from_numpy(np.asarray(self.batch["std_V"])).float().to(self.device)
        # self.net_forward = torch.nn.ModuleList([get_fully_connected_layer(*net_args, **net_kwargs, mean = self.mean, std = self.std)])

    def f(self, x):
        # normalization already done in function get_fully_connected_layer.
        y = self.net_forward[0](x)
        # y[:, 0] = y[:, 0] * self.std_rho + self.mean_rho
        # y[:, 1] = y[:, 1] * self.std_V + self.mean_V
        return y



