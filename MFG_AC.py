import numpy as np
import torch
from torch import nn
from torch.distributions import Categorical
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter
from utils import restrict_u, plot_rho


class Actor(nn.Module):
    def __init__(self, state_dim, n_actions):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(state_dim, 64),
            nn.Tanh(),
            nn.Linear(64, 32),
            nn.Tanh(),
            nn.Linear(32, n_actions),
            nn.Softmax(),
        )

    def forward(self, X):
        return self.model(X)


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
granularity = 4
delta_t = cell / granularity
delta_x = 1 / cell
u_max = int(delta_x / delta_t)
state_dim = 2
gamma = 0.99
epsilon = 0.3

actor = Actor(state_dim, u_max * 5)
critic = Critic(state_dim)
optimizer_actor = torch.optim.Adam(actor.parameters(), lr=1e-3)
optimizer_critic = torch.optim.Adam(critic.parameters(), lr=1e-3)
history = dict()
density = torch.zeros(cell, 40)
density[0][1] = 2

for loop in range(15):
    history[loop] = dict()

    for episode in range(100):
        done = False
        state = torch.tensor([np.random.uniform(low=0, high=2), 0.])
        total_reward = 0
        time_stamp = 0

        while not done:
            represent_x = int(np.floor(state[0]))
            distribution = Categorical(probs=actor(torch.tensor([represent_x, state[1]])))
            action = max(distribution.sample(), torch.tensor(1.))
            u = action / 5
            print(u)
            state_next = torch.tensor([state[0] + u * delta_t, state[1] + delta_t])
            done = int(state_next[0]) >= 2

            rho = density[represent_x][int(state[1] / delta_t)]
            reward = (u * u / 2 + rho + 1) * delta_t
            total_reward += int(reward)

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

        history[loop][episode] = total_reward / time_stamp

    for x in range(cell):
        for t in range(1, granularity):
            u_last_cell = Categorical(probs=actor(torch.tensor([x - 1, delta_t * (t - 1)]))).sample() / 5
            u_curr_cell = Categorical(probs=actor(torch.tensor([x, delta_t * (t - 1)]))).sample() / 5
            density[x][t] = density[x][t - 1] + density[x - 1][t - 1] * u_last_cell - density[x][t - 1] * u_curr_cell

    if loop == 14:
        plot_rho(cell, int(cell / delta_t), density.numpy(), None)

# plt.figure()
# for i, h in enumerate(history.values()):
#     if not i % 3:
#         plt.plot(h.keys(), savgol_filter(list(h.values()), 101, 7), label=i)
#
# plt.legend()
# plt.show()

torch.save(actor.state_dict(), "actor.pt")
torch.save(critic.state_dict(), "critic.pt")