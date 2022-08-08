"""
Basically changing the value array to a critic network. But this env is deterministic, V->Q, critic->DQN.
"""

import numpy as np
import torch
from torch import nn
import random
import time


class DQN(nn.Module):
    def __init__(self, state_dim):
        super().__init__()
        random.seed(time.time())
        self.model = nn.Sequential(
            nn.Linear(state_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 16),
            nn.ReLU(),
            nn.Linear(16, 1),
        )

        def init_weights(m):
            if isinstance(m, nn.Linear):
                torch.nn.init.xavier_uniform(m.weight)
                m.bias.data.fill_(0.01)

        self.model.apply(init_weights)

    def forward(self, x):
        return self.model(torch.from_numpy(x).float())


def value_iteration_dqn(rho, u_max):
    iteration = 36
    n_cell = rho.shape[0]
    T_terminal = int(rho.shape[1] / rho.shape[0])
    delta_T = 1 / n_cell
    T = int(T_terminal / delta_T)
    u = dict()
    V = dict()
    for i in range(n_cell + 1):
        for t in range(T + 1):
            if i < n_cell and t < T:
                u[(i, t)] = 0

            V[(i, t)] = 0

    dqn = DQN(2)
    dqn_optimizer = torch.optim.Adam(dqn.parameters(), lr=1e-3)

    for v_it in range(iteration):
        bootstrap = int(iteration * 5 / 6) + 1
        for i in range(n_cell):
            for t in range(T):
                u[(i, t)] = (V[(i, t + 1)] - V[(i + 1, t + 1)]) / delta_T + 1 - rho[i, t]
                u[(i, t)] = min(max(u[(i, t)], 0), 1)
                if v_it <= bootstrap:
                    V[(i, t)] = delta_T * (0.5 * u[(i, t)] ** 2 + rho[i, t] * u[(i, t)] - u[(i, t)]) + (1 - u[(i, t)]) * V[(i, t + 1)] + u[(i, t)] * V[(i + 1, t + 1)]
                else:
                    V[(i, t)] = delta_T * (0.5 * u[(i, t)] ** 2 + rho[i, t] * u[(i, t)] - u[(i, t)]) + (1 - u[(i, t)]) * dqn(np.array([i, t + 1])) + u[(i, t)] * dqn(np.array([i + 1, t + 1]))

            for t in range(T + 1):
                V[(n_cell, t)] = V[(0, t)]

        # update network if not in bootstrap
        if v_it >= bootstrap - 1:
            for shuo in range(1000):
                truths = torch.tensor(list(V.values()), requires_grad=True)
                preds = torch.reshape(dqn(np.array(list(V.keys()), dtype=float)), (1, len(V)))
                while float(torch.count_nonzero(preds)) == 0:  # to avoid zeros, else while -> if and break
                    dqn = DQN(2)
                    dqn_optimizer = torch.optim.Adam(dqn.parameters(), lr=1e-3)
                    preds = torch.reshape(dqn(np.array(list(V.keys()), dtype=float)), (1, len(V)))

                dqn_loss = (truths - preds).abs().mean()
                dqn_optimizer.zero_grad()
                dqn_loss.backward()
                dqn_optimizer.step()

    u_new = np.zeros((n_cell, T))
    V_new = np.zeros((n_cell + 1, T + 1), dtype=np.float64)
    for i in range(n_cell + 1):
        for t in range(T + 1):
            if i < n_cell and t < T:
                u_new[i, t] = u[(i, t)]

            V_new[i, t] = dqn(np.array([i, t]))

    return u_new, V_new