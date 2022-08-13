import numpy as np
import pandas as pd
import copy

from value_iteration import value_iteration
from value_iteration_ddpg import train_actor, train_critic, train_critic_fake, train_rho
from utils import get_rho_from_u, plot_3d



if __name__ == '__main__':
    n_cell = 32
    T_terminal = 1
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


    data_rho = pd.read_csv('./csv/data_rho_non_sep.csv')
    rho = np.array(data_rho.iloc[:, 1:len(data_rho.iloc[0, :])])
    # rho = train_rho(n_cell, T_terminal)
    critic = train_critic_fake(n_cell, T_terminal, value_iteration(n_cell, T_terminal, rho, fake=True)[1])
    for episode in range(10):
        u_ddpg, V_ddpg, actor = train_actor(n_cell, T_terminal, rho, critic)
        plot_3d(n_cell, T_terminal, u_ddpg, f"./fig/ddpg_{episode}.png")
        critic = train_critic(n_cell, T_terminal, rho, actor, copy.deepcopy(critic))
