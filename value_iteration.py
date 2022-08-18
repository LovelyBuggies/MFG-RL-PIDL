import numpy as np

def value_iteration_ring_road(n_cell, T_terminal, rho_network, fake=False):
    iteration = 36
    delta_T = 1 / n_cell
    T = int(T_terminal / delta_T)
    u = np.zeros((n_cell, T))
    V = np.zeros((n_cell + 1, T + 1), dtype=np.float64)

    for _ in range(iteration):
        for i in range(n_cell):
            for t in range(T):
                rho_i_t = float(rho_network(np.array([i, t])) / n_cell)
                u[i, t] = (V[i, t + 1] - V[i + 1, t + 1]) / delta_T + 1 - rho_i_t
                u[i, t] = 0.5 if fake else min(max(u[i, t], 0), 1)
                V[i, t] = delta_T * (0.5 * u[i, t] ** 2 + rho_i_t * u[i, t] - u[i, t]) + (1 - u[i, t]) * V[i, t + 1] + u[i, t] * V[i + 1, t + 1]

        V[-1, :] = V[0, :].copy()

    return u, V


def value_iteration_routing(model, n_cell, T, rho):
    delta_t = 1 / n_cell
    u = np.zeros((model.n_edge, n_cell, T), dtype=float)
    V = np.zeros((model.n_edge, n_cell, T + 1), dtype=float)
    pi = np.zeros((model.n_node, T + 1), dtype=float)
    c = np.array([1, 1, 1, 1, 0])
    for l in range(model.n_edge):
        start_node, end_node = model.edges[l, 0], model.edges[l, 1]
        for i in range(n_cell):
            V[l, i, T] = model.terminal[start_node] + (model.terminal[end_node] - model.terminal[start_node]) * i / n_cell

    for node in range(model.n_node):
        pi[node, T] = model.terminal[node]

    for iter in range(200):
        for l in range(model.n_edge):
            for t in range(T):
                for i in range(n_cell):
                    if i < n_cell - 1:
                        tmp = (V[l, i, t + 1] - V[l, i + 1, t + 1]) / delta_t - rho[l, i, t]
                        u[l, i, t] = min(max(0, tmp), 1)
                        V[l, i, t] = delta_t * (0.5 * u[l, i, t] **2 + rho[l, i, t] * u[l, i, t] + c[l]) + \
                                     (1 - u[l, i, t]) * V[l, i, t + 1] + u[l, i, t] * V[l, i + 1, t + 1]
                    else:
                        end_node = model.edges[l, 1]
                        tmp = (V[l, i, t + 1] - pi[end_node, t + 1]) / delta_t - rho[l, i, t]
                        u[l, i, t] = min(max(0, tmp), 1)
                        V[l, i, t] = delta_t * (0.5 * u[l, i, t] ** 2 + rho[l, i, t] * u[l, i, t] + c[l]) + \
                                     (1 - u[l, i, t]) * V[l, i, t + 1] + u[l, i, t] * pi[end_node, t + 1]

        beta = np.zeros((model.n_edge, T + 1), dtype=float)
        for node in range(model.n_node):
            for t in range(T):
                out_link = list()
                min_cost = float('inf')
                for out_node in range(model.n_node):
                    k = model.adjacency[node, out_node]
                    if k > -1:
                        out_link.append(k)
                        min_cost = min(min_cost, V[k, 0, t])

                min_link = [link for link in out_link if V[link, 0, t] == min_cost]
                for k in min_link:
                    beta[k, t] = 1 / len(min_link)

                pi[node, t] = 0
                for link in out_link:
                    pi[node, t] = pi[node, t] + beta[link, t] * V[link, 0, t]


    return u, V, beta, pi