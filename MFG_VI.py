import numpy as np
import torch
import pandas as pd
import copy

from value_iteration import value_iteration
from value_iteration_ddpg import train_ddpg
from utils import get_rho_from_u, plot_3d



if __name__ == '__main__':
    data = pd.read_csv('data_rho_non_sep.csv')
    rho = data.iloc[:, 1:len(data.iloc[0, :])]
    d = np.array(data['0.1'])
    rho = np.array(rho)
    train_ddpg(rho, d, 3000)
