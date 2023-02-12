import numpy as np
import torch

from value_iteration import value_iteration_routing
from load_model import load_model_braess
from model import RhoNetwork
from utils import network_loading, plot_4d, network_loading_from_rho_network, get_rho_from_net


if __name__ == '__main__':
    n_cell = 8
    T_terminal = 6
    T = n_cell * T_terminal
    model = load_model_braess()
    u = 0.5 * np.ones((model.n_edge, n_cell, T), dtype=float)
    beta = np.zeros((model.n_edge, T + 1), dtype=float)
    u_hist, beta_hist = list(), list()
    demand = np.zeros((n_cell * T, 1))
    demand[0] = 2
    for node in range(model.n_node):
        out_link = list()
        for out_node in range(model.n_node):
            k = model.adjacency[node, out_node]
            if k > -1:
                out_link.append(k)

        for t in range(T + 1):
            for j in out_link:
                beta[j, t] = 1 / len(out_link)

    rho = network_loading(n_cell, T_terminal, model, u, beta, demand)
    # rho_network = RhoNetwork(3)
    # rho_optimizer = torch.optim.Adam(rho_network.parameters(), lr=1e-3)
    # rho_network = network_loading_from_rho_network(n_cell, T_terminal, model, u, beta, demand, rho_network, rho_optimizer)

    for loop in range(50):
        print(loop)
        u, critic, beta, pi = value_iteration_routing(model, n_cell, T, rho)
        # u, V, beta, pi = value_iteration_routing(model, n_cell, T, get_rho_from_net(n_cell, T_terminal, model, rho_network))
        u_hist.append(u)
        beta_hist.append(beta)
        u = np.array(u_hist).mean(axis=0)
        beta = np.array(beta_hist).mean(axis=0)
        rho = network_loading(n_cell, T_terminal, model, u, beta, demand)
        # rho_network = network_loading_from_rho_network(n_cell, T_terminal, model, u, beta, demand, rho_network, rho_optimizer)

        if loop % 5 == 0:
            # rho = get_rho_from_net(n_cell, T_terminal, model, rho_network)
            plot_4d(n_cell, T_terminal, rho, (0, 4, 3), f"./fig/rho/{loop}.pdf")
