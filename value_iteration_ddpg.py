import numpy as np
import torch
from torch import nn
import random
import time


class Actor(nn.Module):
    def __init__(self, state_dim):
        super().__init__()
        random.seed(time.time())
        self.model = nn.Sequential(
            nn.Linear(state_dim, 16),
            nn.ReLU(),
            nn.Linear(16, 8),
            nn.ReLU(),
            nn.Linear(8, 1),
        )

        def init_weights(m):
            if isinstance(m, nn.Linear):
                torch.nn.init.xavier_uniform(m.weight)
                m.bias.data.fill_(0.01)

        self.model.apply(init_weights)

    def forward(self, x):
        return self.model(torch.from_numpy(x).float())


class Critic(nn.Module):
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


def value_iteration_ddpg(rho, u_max):
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

    actor = Actor(2)
    actor_optimizer = torch.optim.Adam(actor.parameters(), lr=1e-3)
    critic = Critic(2)
    critic_optimizer = torch.optim.Adam(critic.parameters(), lr=1e-3)

    for v_it in range(iteration):
        bootstrap = int(iteration * 5 / 6) + 1
        for i in range(n_cell):
            for t in range(T):
                u[(i, t)] = (V[(i, t + 1)] - V[(i + 1, t + 1)]) / delta_T + 1 - rho[i, t]
                u[(i, t)] = min(max(u[(i, t)], 0), 1)
                if v_it <= bootstrap:
                    V[(i, t)] = delta_T * (0.5 * u[(i, t)] ** 2 + rho[i, t] * u[(i, t)] - u[(i, t)]) + (1 - u[(i, t)]) * V[(i, t + 1)] + u[(i, t)] * V[(i + 1, t + 1)]
                else:
                    V[(i, t)] = delta_T * (0.5 * u[(i, t)] ** 2 + rho[i, t] * u[(i, t)] - u[(i, t)]) + (1 - u[(i, t)]) * critic(np.array([i, t + 1])) + u[(i, t)] * critic(np.array([i + 1, t + 1]))

            for t in range(T + 1):
                V[(n_cell, t)] = V[(0, t)]

        # update critic network if not in bootstrap
        if v_it >= bootstrap - 1:
            for shuo in range(1000):
                truths = torch.tensor(list(V.values()), requires_grad=True)
                preds = torch.reshape(critic(np.array(list(V.keys()), dtype=float)), (1, len(V)))
                while float(torch.count_nonzero(preds)) == 0:  # to avoid zeros, else while -> if and break
                    critic = Critic(2)
                    critic_optimizer = torch.optim.Adam(critic.parameters(), lr=1e-3)
                    preds = torch.reshape(critic(np.array(list(V.keys()), dtype=float)), (1, len(V)))

                critic_loss = (truths - preds).abs().mean()
                critic_optimizer.zero_grad()
                critic_loss.backward()
                critic_optimizer.step()

            # update actor network
            for liu in range(1000):
                truths = torch.tensor(list(u.values()), requires_grad=True)
                preds = torch.reshape(actor(np.array(list(u.keys()), dtype=float)), (1, len(u)))
                while float(torch.count_nonzero(preds)) == 0:  # to avoid zeros, else while -> if and break
                    actor = Actor(2)
                    actor_optimizer = torch.optim.Adam(critic.parameters(), lr=1e-3)
                    preds = torch.reshape(actor(np.array(list(u.keys()), dtype=float)), (1, len(u)))

                actor_loss = (truths - preds).abs().mean()
                print(actor_loss)
                actor_optimizer.zero_grad()
                actor_loss.backward()
                actor_optimizer.step()

    u_new = np.zeros((n_cell, T))
    V_new = np.zeros((n_cell + 1, T + 1), dtype=np.float64)
    for i in range(n_cell + 1):
        for t in range(T + 1):
            if i < n_cell and t < T:
                u_new[i, t] = actor(np.array([i, t]))

            V_new[i, t] = critic(np.array([i, t]))

    return u_new, V_new
