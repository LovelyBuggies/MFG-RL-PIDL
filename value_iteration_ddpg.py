import numpy as np
import torch
from torch import nn
from utils import plot_3d
import csv

"""
Actor part ***********************************
"""


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

        def init_weights(m):
            if isinstance(m, nn.Linear):
                torch.nn.init.xavier_uniform(m.weight)
                m.bias.data.fill_(0.01)

        self.model.apply(init_weights)

    def forward(self, x):
        return self.model(torch.from_numpy(x).float())


def train_actor(n_cell, T_terminal, rho, critic):
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
    for _ in range(1000):
        speeds = actor.forward(states)
        next_xs = np.reshape(states[:, 0], (n_cell * T, 1)) + speeds.detach().numpy() / n_cell
        next_ts = np.reshape(states[:, 1], (n_cell * T, 1)) + np.ones((n_cell * T, 1)) / n_cell
        next_states = np.append(next_xs, next_ts, axis=1)
        advantages = delta_T * (0.5 * speeds ** 2 + rhos * speeds - speeds) + critic.forward(next_states) - critic(
            states)
        policy_loss = advantages.mean()
        # print(policy_loss)
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

    return u_new, V_new, actor


"""
Critic part ***********************************
"""


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

        def init_weights(m):
            if isinstance(m, nn.Linear):
                torch.nn.init.xavier_uniform(m.weight)
                m.bias.data.fill_(0.01)

        self.model.apply(init_weights)

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


def train_critic(n_cell, T_terminal, rho, actor, critic):
    T = n_cell * T_terminal
    delta_T = 1 / n_cell

    shuo = Critic(2)
    shuo_optimizer = torch.optim.Adam(shuo.parameters(), lr=1e-3)

    for _ in range(100):
        states, truths = list(), list()
        for i in range(n_cell):
            for t in range(T):
                states.append(np.array([i, t]) / n_cell)
                speed = float(actor.forward(np.array([i, t]) / n_cell))
                if t < T - 1:
                    truths.append(delta_T * (0.5 * speed ** 2 + rho[i, t] * speed - speed) + critic(
                        np.array([i + speed, t + 1]) / n_cell))
                else:
                    truths.append(delta_T * (0.5 * speed ** 2 + rho[i, t] * speed - speed))

        truths = torch.tensor(truths, requires_grad=True)
        preds = torch.reshape(shuo(np.array(states)), (1, len(truths)))
        loss = (truths - preds).abs().mean()
        # print(loss)
        shuo_optimizer.zero_grad()
        loss.backward()
        shuo_optimizer.step()

    pred_V = np.zeros((n_cell + 1, T + 1))
    for i in range(n_cell + 1):
        for t in range(T + 1):
            pred_V[i, t] = shuo(np.array([i, t]) / n_cell)

    plot_3d(n_cell, T_terminal, pred_V[:-1, :-1], None)
    return shuo



"""
Rho part ***********************************
"""
class RhoNetwork(nn.Module):
    def __init__(self, state_dim):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(state_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 8),
            nn.ReLU(),
            nn.Linear(8, 1),
            nn.ReLU(),
        )

        def init_weights(m):
            if isinstance(m, nn.Linear):
                torch.nn.init.xavier_uniform(m.weight)
                m.bias.data.fill_(0.01)

        self.model.apply(init_weights)

    def forward(self, x):
        return self.model(torch.from_numpy(x).float())


def train_rho(n_cell, T_terminal):
    T = n_cell * T_terminal
    csv_list = []
    with open("./csv/data_rho_non_sep.csv") as file:
        reader = csv.reader(file, delimiter=',')
        for i, row in enumerate(reader):
            csv_list.append(row)

    csv_array = np.array(csv_list, dtype=float)
    t_array = csv_array[0, 1:]
    x_array = csv_array[1:, 0]
    rho_array = csv_array[1:, 1:]
    truths = []
    keys = []
    for i, x in enumerate(x_array):
        for j, t in enumerate(t_array):
            truths.append(rho_array[i, j])
            keys.append(np.array([x, t]))

    rho = RhoNetwork(2)
    rho_optimizer = torch.optim.Adam(rho.parameters(), lr=1e-3)
    for _ in range(2000):
        truths = torch.tensor(truths, requires_grad=True)
        preds = torch.reshape(rho(np.array(keys)), (1, len(truths)))
        loss = (truths - preds).abs().mean()
        # print(loss)
        rho_optimizer.zero_grad()
        loss.backward()
        rho_optimizer.step()

    # pred_rho = np.zeros((len(x_array), len(t_array)))
    # for i, x in enumerate(x_array):
    #     for j, t in enumerate(t_array):
    #         pred_rho[i, j] = rho(np.array([x, t]))
    #
    # plot_3d(n_cell, T_terminal, rho_array, None)
    # plot_3d(n_cell, T_terminal, pred_rho, None)
    return rho