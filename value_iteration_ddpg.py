import numpy as np
import torch
from torch import nn
from utils import get_rho_from_u, plot_3d


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

    def f(self, x):
        x = self.model(x)
        x = torch.tanh(x)
        x = (x + 1) / 2
        return x


class Critic(nn.Module):
    def __init__(self, state_dim):
        super(Critic, self).__init__()
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


class RhoNetwork(nn.Module):
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
        return self.model(torch.from_numpy(x).float())


def train_critic_fake(n_cell, T_terminal, V_array):
    T = n_cell * T_terminal
    truths = []
    keys = []

    liu = Critic(2)
    liu_optimizer = torch.optim.Adam(liu.parameters(), lr=1e-3)

    for i in range(n_cell + 1):
        for t in range(T + 1):
            truths.append(V_array[i, t])
            keys.append(np.array([i, t]) / n_cell)

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
            pred_V[i, t] = liu(np.array([i, t]) / n_cell)

    return liu


def train_rho_network_one_step(n_cell, T_terminal, rho, rho_network, rho_optimizer):
    truths, keys = list(), list()
    for i in range(len(rho)):
        for j in range(len(rho[0])):
            truths.append(rho[i, j])
            keys.append(np.array([i, j]) / n_cell)

    for _ in range(1):
        truths = torch.tensor(truths, requires_grad=True)
        preds = torch.reshape(rho_network(np.array(keys)), (1, len(truths)))
        loss = (truths - preds).abs().mean()
        rho_optimizer.zero_grad()
        loss.backward()
        rho_optimizer.step()

    return rho_network


def train_ddpg(n_cell, T_terminal, d, iterations):
    delta_T = 1 / n_cell
    T = int(T_terminal / delta_T)
    u_hist = list()

    u_init = 0.5 * np.ones((n_cell, T))
    rho = get_rho_from_u(u_init, d)

    actor = Actor(2)
    actor_optimizer = torch.optim.Adam(actor.parameters(), lr=1e-3)

    fake_critic = train_critic_fake(n_cell, T_terminal, np.zeros((n_cell + 1, T + 1)))
    critic = Critic(2)
    critic_optimizer = torch.optim.Adam(critic.parameters(), lr=1e-3)

    rho_network = RhoNetwork(2)
    rho_optimizer = torch.optim.Adam(rho_network.parameters(), lr=1e-3)
    rho_network = train_rho_network_one_step(n_cell, T_terminal, rho, rho_network, rho_optimizer)

    for it in range(iterations):
        # train critic
        if it % 1 == 0:
            keys, truths = list(), list()
            for i in range(n_cell + 1):
                for t in range(T + 1):
                    keys.append([i / n_cell, t / n_cell])
                    if t != T:
                        if i != n_cell:
                            speed = float(actor.forward(np.array([i, t]) / n_cell))
                            truths.append(delta_T * (0.5 * speed ** 2 + rho[i, t] * speed - speed) + fake_critic(
                                np.array([i + speed, t + 1]) / n_cell))
                        else:
                            speed = float(actor.forward(np.array([0, t]) / n_cell))
                            truths.append(delta_T * (0.5 * speed ** 2 + rho[0, t] * speed - speed) + fake_critic(
                                np.array([speed, t + 1]) / n_cell))
                    else:
                        truths.append(0)

            truths = torch.tensor(truths, requires_grad=True)
            for _ in range(100):
                preds = torch.reshape(critic(np.array(keys)), (1, len(truths)))
                critic_loss = (truths - preds).abs().mean()
                critic_optimizer.zero_grad()
                critic_loss.backward()
                critic_optimizer.step()

            fake_critic = critic

        # train actor
        states, rhos, Vs, Vus = list(), list(), list(), list()
        for i in range(n_cell):
            for t in range(T):
                rho_i_t = float(rho_network.forward(np.array([i, t]) / n_cell))
                states.append([i / n_cell, t / n_cell])
                rhos.append(rho_i_t)

        for i in range(T):
            for t in range(n_cell):
                Vs.append(float(critic(np.array([i, t + 1]) / n_cell) - critic(np.array([i, t]) / n_cell)))
                Vus.append(float(critic(np.array([i + 1, t + 1]) / n_cell) - critic(np.array([i, t + 1]) / n_cell)))

        states = np.array(states)
        rhos = torch.tensor(np.reshape(np.array(rhos), (n_cell * T, 1)))
        Vs = torch.tensor(np.reshape(np.array(Vs), (n_cell * T, 1)))
        Vus = torch.tensor(np.reshape(np.array(Vus), (n_cell * T, 1)))
        for _ in range(1):
            speeds = actor.forward(states)
            advantages = delta_T * (0.5 * speeds ** 2 + rhos * speeds - speeds) + Vus * speeds + Vs
            policy_loss = advantages.mean()
            actor_optimizer.zero_grad()
            policy_loss.backward()
            actor_optimizer.step()

        # train rho net
        u = np.zeros((n_cell, T))
        for i in range(n_cell):
            for t in range(T):
                u[i, t] = actor(np.array([i, t]) / n_cell)

        u_hist.append(u)
        u = np.array(u_hist).mean(axis=0)
        rho = get_rho_from_u(u, d)
        rho_network = train_rho_network_one_step(n_cell, T_terminal, rho, rho_network, rho_optimizer)

        if it % 20 == 0 and it != 0:
            plot_3d(n_cell, T_terminal, u, f"./fig/u/{it}.png")  # show without fp
            plot_3d(n_cell, T_terminal, rho, f"./fig/rho/{it}.png")  # show with fp on rho

    return u, rho
