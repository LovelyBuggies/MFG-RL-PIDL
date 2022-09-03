import numpy as np
import torch
import pandas as pd
import copy

from value_iteration import value_iteration
from value_iteration_ddpg import train_ddpg
from utils import get_rho_from_u, plot_3d
from get_rho_network import get_rho_network
from torch import nn

class Actor(nn.Module):
    def __init__(self, state_dim):
        super(Actor, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(state_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, 8),
            nn.ReLU(),
            nn.Linear(8, 1)
        )

    def forward(self, x):
        x = self.model(torch.from_numpy(x).float())
        x = torch.tanh(x)
        x = (x + 1) / 2
        return x

if __name__ == '__main__':
    data = pd.read_csv('data_rho_lwr_new.csv')
    rho = data.iloc[:, 1:len(data.iloc[0, :])]
    d = np.array(data['0.1'])
    rho = np.array(rho)
    for _ in range(1):
        actor = train_ddpg(rho, d, 5000)
        torch.save(actor.state_dict(), f"./u_model/actor.pt")
        model = get_rho_network("./u_model/actor.pt")
        pidl = np.zeros((32, 32))
        for i in range(32):
            for t in range(32):
                a = torch.tensor([i / 32]).float().to(model.device)
                b = torch.tensor([t / 32]).float().to(model.device)
                a = torch.unsqueeze(a, dim=-1)
                b = torch.unsqueeze(b, dim=-1)
                pidl[i, t] = float(model.f(torch.cat((a, b), 1))[:, 0])


        plot_3d(32, 1, pidl, "pidl.png")



    # rho_network = pidl(random actor)
    # for i in 10000:
    #     actor = train_ddpg(rho_network, iterations)
    #     rho_network = pidl(actor)


