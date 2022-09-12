import numpy as np
import torch
from torch import nn
from utils import plot_3d
import csv
import copy
import pandas as pd
from value_iteration import value_iteration
from utils import get_rho_from_u, plot_3d


# class Critic(nn.Module):
#     def __init__(self, state_dim):
#         super(Critic, self).__init__()
#         self.model = nn.Sequential(
#             nn.Linear(state_dim, 64),
#             nn.ReLU(),
#             nn.Linear(64, 32),
#             nn.ReLU(),
#             nn.Linear(32, 16),
#             nn.ReLU(),
#             nn.Linear(16, 8),
#             nn.ReLU(),
#             nn.Linear(8, 1)
#         )
#
#     def forward(self, x):
#         x = self.model(torch.from_numpy(x).float())
#         x = torch.tanh(x)
#         x = x / 2
#         return x


# class RhoNetwork(nn.Module):
#     def __init__(self, state_dim):
#         super().__init__()
#         self.model = nn.Sequential(
#             nn.Linear(state_dim, 64),
#             nn.ReLU(),
#             nn.Linear(64, 32),
#             nn.ReLU(),
#             nn.Linear(32, 16),
#             nn.ReLU(),
#             nn.Linear(16, 8),
#             nn.ReLU(),
#             nn.Linear(8, 1)
#         )
#
#     def forward(self, x):
#         return self.model(torch.from_numpy(x).float())


# def train_rho_network(n_cell, T_terminal, rho, rho_network, rho_optimizer):
#     truths, keys = list(), list()
#     for i in range(len(rho)):
#         for j in range(len(rho[0])):
#             truths.append(rho[i, j])
#             keys.append(np.array([i, j]) / n_cell)
#
#     for _ in range(2000):
#         truths = torch.tensor(truths, requires_grad=True)
#         preds = torch.reshape(rho_network(np.array(keys)), (1, len(truths)))
#         loss = (truths - preds).abs().mean()
#         rho_optimizer.zero_grad()
#         loss.backward()
#         rho_optimizer.step()
#
#     return rho_network


def train_ddpg(n_cell, T_terminal, d, iterations):
    delta_T = 1 / n_cell
    T = int(T_terminal / delta_T)
    u_hist = list()

    u = 0.5 * np.ones((n_cell, T))
    V = np.zeros((n_cell + 1, T + 1), dtype=np.float64)
    rho = get_rho_from_u(u, d)

    # rho_network = RhoNetwork(2)
    # rho_optimizer = torch.optim.Adam(rho_network.parameters(), lr=1e-3)
    # rho_network = train_rho_network(n_cell, T_terminal, rho, rho_network, rho_optimizer)

    for it in range(iterations):
        # train V
        for t in range(T):
            for i in range(n_cell):
                rho_i_t = rho[i, t]
                u[i, t] = (V[i, t + 1] - V[i + 1, t + 1]) / delta_T + 1 - rho_i_t
                u[i, t] = min(max(u[i, t], 0), 1)
                V[i, t] = delta_T * (0.5 * u[i, t] ** 2 + rho_i_t * u[i, t] - u[i, t]) + (1 - u[i, t]) * V[
                    i, t + 1] + u[i, t] * V[i + 1, t + 1]

        V[-1, :] = V[0, :].copy()

        u_hist.append(u)
        u = np.array(u_hist).mean(axis=0)
        rho = get_rho_from_u(u, d)
        # rho_network = train_rho_network(n_cell, T_terminal, rho, rho_network, rho_optimizer)

        # if it % 10 == 0 and it != 0:
            # plot_3d(32, 1, u, f"./fig/u/{it}.png")  # show without fp
            # plot_3d(32, 1, rho, f"./fig/rho/{it}.png")  # show with fp on rho
    return u, rho
