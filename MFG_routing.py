import numpy as np

from value_iteration import value_iteration_routing
from load_model import load_model_braess
from utils import network_loading, plot_4d


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

    rho = network_loading(model, u, beta, demand, n_cell, T)

    for loop in range(300):
        print(loop)
        u, V, beta, pi = value_iteration_routing(model, n_cell, T, rho)
        u_hist.append(u)
        beta_hist.append(beta)
        u = np.array(u_hist).mean(axis=0)
        beta = np.array(beta_hist).mean(axis=0)
        rho = network_loading(model, u, beta, demand, n_cell, T)

    plot_4d(n_cell, T_terminal, rho, (0, 4, 3), None)
