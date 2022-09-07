import numpy as np
import torch
import pandas as pd
import copy

from value_iteration import value_iteration
from value_iteration_ddpg import train_ddpg
from utils import get_rho_from_u, plot_3d
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
    n_cell = 32
    T_terminal = 1
    T = n_cell * T_terminal
    u = 0.5 * np.ones((n_cell, T))
    d = np.array([
        0.39999,
        0.39998,
        0.39993,
        0.39977,
        0.39936,
        0.39833,
        0.39602,
        0.39131,
        0.38259,
        0.36804,
        0.34619,
        0.31695,
        0.28248,
        0.24752,
        0.21861,
        0.20216,
        0.20216,
        0.21861,
        0.24752,
        0.28248,
        0.31695,
        0.34619,
        0.36804,
        0.38259,
        0.39131,
        0.39602,
        0.39833,
        0.39936,
        0.39977,
        0.39993,
        0.39998,
        0.39999
    ])
    rho = get_rho_from_u(u, d)
    u = train_ddpg(rho, d, 100000)


