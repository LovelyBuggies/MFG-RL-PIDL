import numpy as np
import torch
from torch import nn
from torch.distributions import Categorical
from utils import update_density
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter


class Actor(nn.Module):
    def __init__(self, state_dim, action_dim):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(state_dim, 64),
            nn.Tanh(),
            nn.Linear(64, 32),
            nn.Tanh(),
            nn.Linear(32, action_dim)
        )

    def forward(self, X):
        return self.model(X)


# Critic module
class Critic(nn.Module):
    def __init__(self, state_dim):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(state_dim, 64),
            nn.ReLU(inplace=False),
            nn.Linear(64, 32),
            nn.ReLU(inplace=False),
            nn.Linear(32, 1)
        )

    def forward(self, X):
        return self.model(X)


cell = 2
granularity = 10
n_actions = granularity
delta_t = cell / granularity
state_dim = 2
action_dim = cell * granularity
gamma = 0.99

actor = Actor(state_dim, action_dim)
critic = Critic(state_dim)
optimizer_actor = torch.optim.Adam(actor.parameters(), lr=1e-3)
optimizer_critic = torch.optim.Adam(critic.parameters(), lr=1e-3)
history = dict()

for episode in range(1):
    done = False
    state = torch.tensor([0., 0.])
    density = torch.zeros(cell, granularity)
    density[0][1] = 2
    total_reward = 0
    time_stamp = 0

    while not done:
        represent_x = int(np.floor(state[0]))
        action = actor(torch.tensor([represent_x, state[1]]))
        for i, _ in enumerate(action):
            action[i] = min(max(action[i], 0), granularity)

        action = action.reshape(cell, granularity)
        u = action[represent_x][int(state[1] / delta_t)]
        state_next = torch.tensor([state[0] + u * delta_t, state[1] + delta_t])
        done = int(state_next[0]) >= 2
        density = update_density(density, action)
        # rho = density[represent_x][int(state[1] / delta_t)]
        rho = 2
        reward = (u * u / 2 + rho + 1) * delta_t
        total_reward += reward

        advantage = reward + (1 - done) * gamma * critic(state_next) - critic(state)
        state = state_next
        critic_loss = advantage.pow(2).mean()
        optimizer_critic.zero_grad()
        critic_loss.backward()
        optimizer_critic.step()

        actor_loss = -distribution.log_prob(action) * advantage.detach()
        optimizer_actor.zero_grad()
        actor_loss.backward()
        optimizer_actor.step()
        time_stamp += 1

    history[episode] = total_reward / time_stamp

plt.figure()
plt.plot(history.keys(), savgol_filter(list(history.values()), 51, 3))
plt.show()
