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
            nn.Linear(64, 16),
            nn.ReLU(),
            nn.Linear(16, 1),
            nn.ReLU()
        )

        def init_weights(m):
            if isinstance(m, nn.Linear):
                torch.nn.init.xavier_uniform(m.weight)
                m.bias.data.fill_(0.01)

        self.model.apply(init_weights)

    def forward(self, x):
        return self.model(torch.from_numpy(x).float())


def value_iteration_dqn(rho, u_max, n_action):
    iteration = 36
    n_cell = rho.shape[0]
    delta_u = u_max / n_action
    u_action = np.arange(0, u_max + delta_u, delta_u)
    T_terminal = int(rho.shape[1] / rho.shape[0])
    delta_T = 1 / n_cell
    T = int(T_terminal / delta_T)
    u = dict()
    V = dict()
    dqn = DQN(2)
    dqn_optimizer = torch.optim.Adam(dqn.parameters(), lr=1e-3)
    u_new = np.zeros((n_cell, T))
    V_new = np.zeros((n_cell + 1, T + 1), dtype=np.float64)

    for v_it in range(iteration):
        bootstrap = int(iteration * 3 / 4) + 1
        for i in range(n_cell * n_action):
            for t in range(T):
                min_value = np.float('inf')
                for j in np.arange(n_action + 1):
                    speed = u_action[j]
                    new_i = int(i + speed / delta_u)
                    rho_i = int(i / n_action)
                    # bootstrap
                    if new_i <= n_cell * n_action:
                        if v_it <= bootstrap:
                            value = delta_T * (0.5 * speed ** 2 + rho[rho_i, t] + 1) + V[(new_i, t + 1)] \
                                if (new_i, t + 1) in V else delta_T * (0.5 * speed ** 2 + rho[rho_i, t] + 1)
                        else:
                            value = delta_T * (0.5 * speed ** 2 + rho[rho_i, t] + 1) + dqn(np.array([new_i, t + 1]))
                    else:
                        time = delta_u * delta_T * (n_cell * n_action - i) / speed
                        value = time * (0.5 * speed ** 2 + rho[rho_i, t] + 1)

                    if min_value > value:
                        min_value = value
                        u[(i, t)] = speed
                        V[(i, t)] = min_value

        # update network if not in bootstrap
        if v_it >= bootstrap - 1:
            for shuo in range(1000):
                truths = torch.tensor(list(V.values()), requires_grad=True)
                preds = torch.reshape(dqn(np.array(list(V.keys()), dtype=float)), (1, len(V)))
                while float(torch.count_nonzero(preds)) == 0:  # to avoid zeros, else while -> if and break
                    dqn = DQN(2)
                    dqn_optimizer = torch.optim.Adam(dqn.parameters(), lr=1e-3)
                    preds = torch.reshape(dqn(np.array(list(V.keys()), dtype=float)), (1, len(V)))

                loss = truths - preds
                dqn_loss = loss.abs().mean()
                dqn_optimizer.zero_grad()
                dqn_loss.backward()
                dqn_optimizer.step()

    # return value for checking
    for i in range(n_cell):
        for t in range(T):
            u_new[i, t] = u[(i * n_action, t)]
            V_new[i, t] = dqn(np.array([i * n_action, t]))

    return u_new, V_new