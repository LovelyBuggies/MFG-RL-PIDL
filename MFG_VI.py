import numpy as np
import pandas as pd
import csv
import os
import torch
from torch import nn

from value_iteration import value_iteration
from value_iteration_ddpg import value_iteration_ddpg
from utils import get_rho_from_u, plot_rho


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

critic = Critic(2)
critic_optimizer = torch.optim.Adam(critic.parameters(), lr=1e-3)

def train_critic(n_cell, T_terminal):
    T = n_cell * T_terminal
    csv_list = []
    with open("data_V_non_sep.csv") as file:
        reader = csv.reader(file, delimiter=',')
        for i, row in enumerate(reader):
            csv_list.append(row)

    csv_array = np.array(csv_list, dtype=float)
    t_array = csv_array[0, 1:]
    x_array = csv_array[1:, 0]
    V_array = csv_array[1:, 1:]
    truths = []
    keys = []
    for i, x in enumerate(x_array):
        for j, t in enumerate(t_array):
            truths.append(V_array[i, j])
            keys.append(np.array([x, t]))

    for _ in range(2000):
        truths = torch.tensor(truths, requires_grad=True)
        preds = torch.reshape(critic(np.array(keys)), (1, len(truths)))
        loss = (truths - preds).abs().mean()
        print(loss)
        critic_optimizer.zero_grad()
        loss.backward()
        critic_optimizer.step()

    pred_u = np.zeros((len(x_array), len(t_array)))
    for i, x in enumerate(x_array):
        for j, t in enumerate(t_array):
            pred_u[i, j] = critic(np.array([x, t]))

    plot_rho(n_cell, T_terminal, V_array[:, :-1], None)
    plot_rho(n_cell, T_terminal, pred_u[:, :-1], None)
    torch.save(critic, "./critic.pt")
    return critic


if __name__ == '__main__':
    n_cell = 32
    T_terminal = 1
    u = 0.5 * np.ones((n_cell, n_cell * T_terminal), dtype=np.float64)
    d = np.array([
        0.39999,
        0.39998,
        0.39993,
        0.39977,
        0.39936,
        0.39833,
        0.39602,
        0.39131,
        0.38259,
        0.36804,
        0.34619,
        0.31695,
        0.28248,
        0.24752,
        0.21861,
        0.20216,
        0.20216,
        0.21861,
        0.24752,
        0.28248,
        0.31695,
        0.34619,
        0.36804,
        0.38259,
        0.39131,
        0.39602,
        0.39833,
        0.39936,
        0.39977,
        0.39993,
        0.39998,
        0.39999
    ])


    data_rho = pd.read_csv('data_rho_non_sep.csv')
    rho = np.array(data_rho.iloc[:, 1:len(data_rho.iloc[0, :])])
    if not os.path.exists("./critic.pt"):
        critic = train_critic(n_cell, T_terminal)
    else:
        critic = torch.load("./critic.pt")

    u_ddpg, V_ddpg = value_iteration_ddpg(rho, critic)
    rho_ddpg = get_rho_from_u(u_ddpg, d)
    plot_rho(n_cell, T_terminal, u_ddpg, f"./fig/ddpg.png")
    plot_rho(n_cell, T_terminal, rho_ddpg, f"./fig_rho/ddpg.png")
