import numpy as np
import torch
from torch import nn
from utils import plot_3d
import csv
import copy
from value_iteration import value_iteration
from utils import get_rho_from_u, plot_3d

class Critic(nn.Module):
    def __init__(self, state_dim):
        super().__init__()
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
        x = x / 2
        return x


class Actor(nn.Module):
    def __init__(self, state_dim):
        super().__init__()
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


def train_critic_fake(n_cell, T_terminal, V_array):
    T = n_cell * T_terminal
    truths = []
    keys = []

    liu = Critic(2)
    liu_optimizer = torch.optim.Adam(liu.parameters(), lr=1e-3)

    for i in range(n_cell + 1):
        for t in range(T + 1):
            truths.append(V_array[i, t])
            keys.append(np.array([i, t]))

    for _ in range(1000):
        truths = torch.tensor(truths, requires_grad=True)
        preds = torch.reshape(liu(np.array(keys)), (1, len(truths)))
        loss = (truths - preds).abs().mean()
        # print(loss)
        liu_optimizer.zero_grad()
        loss.backward()
        liu_optimizer.step()

    pred_V = np.zeros((n_cell + 1, T + 1))
    for i in range(n_cell + 1):
        for t in range(T + 1):
            pred_V[i, t] = liu(np.array([i, t]))

    return liu


def train_ddpg(rho, d, iterations):
    n_cell = rho.shape[0]
    T_terminal = int(rho.shape[1] / rho.shape[0])
    delta_T = 1 / n_cell
    T = int(T_terminal / delta_T)
    V = np.ones((n_cell + 1, T + 1))
    states = list()
    rhos = list()
    rho_hist = list()

    actor = Actor(2)
    actor_optimizer = torch.optim.Adam(actor.parameters(), lr=1e-3)

    fake_critic = train_critic_fake(n_cell, T_terminal, value_iteration(n_cell, T_terminal, rho, fake=True)[1])
    critic = Critic(2)
    critic_optimizer = torch.optim.Adam(critic.parameters(), lr=1e-3)

    # use value iteration to get the expected V table
    for i in range(n_cell):
        for t in range(T):
            states.append([i, t])
            rhos.append(rho[i, t])

    states = np.array(states)
    rhos = torch.tensor(np.reshape(np.array(rhos), (n_cell * T, 1)))

    for a_it in range(iterations):
        # train critic
        if a_it % 50 == 0:
            keys = list()
            truths = list()
            for i in range(n_cell + 1):
                for t in range(T + 1):
                    keys.append([i, t])
                    speed = float(actor.forward(np.array([i, t])))
                    if t != T:
                        if i != n_cell:
                            truths.append(delta_T * (0.5 * speed ** 2 + rho[i, t] * speed - speed) + fake_critic(
                            np.array([i + speed, t + 1])))
                        else:
                            truths.append(delta_T * (0.5 * speed ** 2 + rho[0, t] * speed - speed) + fake_critic(
                                np.array([speed, t + 1])))
                    else:
                        if i != n_cell:
                            truths.append(delta_T * (0.5 * speed ** 2 + rho[i, t - 1] * speed - speed))
                        else:
                            truths.append(delta_T * (0.5 * speed ** 2 + rho[0, t - 1] * speed - speed))

            truths = torch.tensor(truths, requires_grad=True)
            for c_it in range(5000):
                preds = torch.reshape(critic(np.array(keys)), (1, len(truths)))
                critic_loss = (truths - preds).abs().mean()
                if c_it % 100 == 0:
                    print(f"{c_it} critic loss", float(critic_loss))
                critic_optimizer.zero_grad()
                critic_loss.backward()
                critic_optimizer.step()
                if critic_loss < 1e-4:
                    break

            fake_critic = critic

        # train actor
        speeds = actor.forward(states)
        next_xs_1 = np.reshape(states[:, 0], (n_cell * T, 1))
        next_xs_2 = np.reshape(states[:, 0], (n_cell * T, 1)) + np.ones((n_cell * T, 1))
        next_xs = torch.reshape(torch.tensor(states[:, 0]), (n_cell * T, 1)) + speeds.detach().numpy()
        next_ts = np.reshape(states[:, 1], (n_cell * T, 1)) + np.ones((n_cell * T, 1))
        next_states_1 = np.append(next_xs_1, next_ts, axis=1)
        next_states_2 = np.append(next_xs_2, next_ts, axis=1)
        next_states = np.append(next_xs, next_ts, axis=1)
        interp_V_next_state = (torch.ones((n_cell * T, 1)) - speeds) * critic.forward(
            next_states_1) + speeds * critic.forward(next_states_2)
        advantages = delta_T * (0.5 * speeds ** 2 + rhos * speeds - speeds) + critic.forward(next_states) - critic(states)
        policy_loss = advantages.mean()
        if a_it % 5 == 0:
            # print(max(critic.forward(next_states) - interp_V_next_state))
            print(f"{a_it} policy loss", float(policy_loss))

        actor_optimizer.zero_grad()
        policy_loss.backward()
        actor_optimizer.step()

        u_new = np.zeros((n_cell, T))
        V_new = np.zeros((n_cell + 1, T + 1), dtype=np.float64)
        for i in range(n_cell + 1):
            for t in range(T + 1):
                if i < n_cell and t < T:
                    u_new[i, t] = actor(np.array([i, t]))

                V_new[i, t] = V[i, t]

        rho_hist.append(get_rho_from_u(u_new, d))
        if a_it % 50 == 0 and a_it != 0:
            plot_3d(32, 1, u_new, f"./fig/u/{a_it}.png")
            plot_3d(32, 1, np.array(rho_hist[:-20]).mean(axis=0),  f"./fig/rho/{a_it}.png")