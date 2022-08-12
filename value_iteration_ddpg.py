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

        def init_weights(m):
            if isinstance(m, nn.Linear):
                torch.nn.init.xavier_uniform(m.weight)
                m.bias.data.fill_(0.01)

        self.model.apply(init_weights)

    def forward(self, x):
        return self.model(torch.from_numpy(x).float())


def value_iteration_ddpg(rho, critic):
    n_cell = rho.shape[0]
    T_terminal = int(rho.shape[1] / rho.shape[0])
    delta_T = 1 / n_cell
    T = int(T_terminal / delta_T)
    states = list()
    rhos = list()

    actor = Actor(2)
    actor_optimizer = torch.optim.Adam(actor.parameters(), lr=1e-3)

    for i in range(n_cell):
        for t in range(T):
            states.append(np.array([i, t]) / n_cell)
            rhos.append(rho[i, t])


    states = np.array(states)
    rhos = torch.tensor(np.reshape(np.array(rhos), (n_cell * T, 1)))
    for _ in range(10000):
        speeds = actor.forward(states)
        next_xs = np.reshape(states[:, 0], (n_cell * T, 1)) + speeds.detach().numpy() / n_cell
        next_ts = np.reshape(states[:, 1], (n_cell * T, 1)) + np.ones((n_cell * T, 1)) / n_cell
        next_states = np.append(next_xs, next_ts, axis=1)
        advantages = delta_T * (0.5 * speeds ** 2 + rhos * speeds - speeds) + critic.forward(next_states) - critic(states)
        policy_loss = advantages.mean()
        print(policy_loss)
        actor_optimizer.zero_grad()
        policy_loss.backward()
        actor_optimizer.step()


    u_new = np.zeros((n_cell, T))
    V_new = np.zeros((n_cell + 1, T + 1), dtype=np.float64)
    for i in range(n_cell + 1):
        for t in range(T + 1):
            if i < n_cell and t < T:
                u_new[i, t] = actor(np.array([i / n_cell, t / n_cell]))

            V_new[i, t] = critic(np.array([i / n_cell, t / n_cell]))

    return u_new, V_new
