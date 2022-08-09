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


def value_iteration_ddpg(rho, u_max):
    iteration = 36
    n_cell = rho.shape[0]
    T_terminal = int(rho.shape[1] / rho.shape[0])
    delta_T = 1 / n_cell
    T = int(T_terminal / delta_T)
    u = np.zeros((n_cell, T))
    V = np.zeros((n_cell + 1, T + 1))
    states = list()

    actor = Actor(2)
    actor_optimizer = torch.optim.Adam(actor.parameters(), lr=1e-3)

    # use value iteration to get the expected V table
    for _ in range(30):
        for i in range(n_cell):
            for t in range(T):
                states.append([i ,t])
                u[i, t] = (V[i, t + 1] - V[i + 1, t + 1]) / delta_T + 1 - rho[i, t]
                u[i, t] = min(max(u[i, t], 0), 1)
                V[i, t] = delta_T * (0.5 * u[i, t] ** 2 + rho[i, t] * u[i, t] - u[i, t]) + (1 - u[i, t]) * V[i, t + 1] + u[i, t] * V[i + 1, t + 1]

            for t in range(T + 1):
                V[(n_cell, t)] = V[(0, t)]

    # use policy gradient to train the network
    for v_it in range(iteration):
        advantages = list()
        for i in range(n_cell):
            for t in range(T):
                speed = actor.forward(np.array([i, t])) # todo: min max
                advantages.append(delta_T * (0.5 * speed ** 2 + rho[i, t] * speed - speed) + (1 - speed) * V[i, t + 1] + speed * V[i + 1, t + 1])

        policy_loss = torch.tensor(advantages).mean()
        print(policy_loss)
        actor_optimizer.zero_grad()
        policy_loss.backward()
        actor_optimizer.step()


    u_new = np.zeros((n_cell, T))
    V_new = np.zeros((n_cell + 1, T + 1), dtype=np.float64)
    for i in range(n_cell + 1):
        for t in range(T + 1):
            if i < n_cell and t < T:
                u_new[i, t] = u[i, t]

            V_new[i, t] = V[i, t]

    return u_new, V_new
