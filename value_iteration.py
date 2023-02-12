import numpy as np
import torch

# from model import Critic


def value_iteration_routing(model, n_cell, T, rho):
    delta_t = 1 / n_cell
    u = np.zeros((model.n_edge, n_cell, T), dtype=float)
    V = np.zeros((model.n_edge, n_cell, T + 1), dtype=float)
    # critic = Critic(3)
    # critic_optimizer = torch.optim.Adam(critic.parameters(), lr=1e-3)
    # states, critic_values = list(), list()
    pi = np.zeros((model.n_node, T + 1), dtype=float)
    c = np.array([1, 3, 3, 1, 1])
    for l in range(model.n_edge):
        start_node, end_node = model.edges[l, 0], model.edges[l, 1]
        for i in range(n_cell):
            # states.append(np.array([l, i / n_cell, T / n_cell]))
            # critic_values.append(model.terminal[start_node] + (model.terminal[end_node] - model.terminal[start_node]) * i / n_cell)
            V[l, i, T] = model.terminal[start_node] + (model.terminal[end_node] - model.terminal[start_node]) * i / n_cell

    for node in range(model.n_node):
        pi[node, T] = model.terminal[node]

    for iter in range(200):
        for l in range(model.n_edge):
            for t in range(T):
                for i in range(n_cell):
                    # states.append(np.array([l, i / n_cell, t / n_cell]))
                    if i < n_cell - 1:
                        # tmp = (critic(np.array([l, i / n_cell, (t + 1) / n_cell])) - critic(np.array([l, (i + 1) / n_cell, (t + 1) / n_cell]))) / delta_t - rho[l, i, t]
                        tmp = (V[l, i, t + 1] - V[l, i + 1, t + 1]) / delta_t - rho[l, i, t]
                        u[l, i, t] = min(max(0, tmp), 1)
                        # critic_values.append(delta_t * (0.5 * u[l, i, t] **2 + rho[l, i, t] * u[l, i, t] + c[l]) + (1 - u[l, i, t]) * critic(np.array([l, i / n_cell, (t + 1) / n_cell])) + u[l, i, t] * critic(np.array([l, (i + 1) / n_cell, (t + 1) / n_cell])))
                        V[l, i, t] = delta_t * (0.5 * u[l, i, t] ** 2 + rho[l, i, t] * u[l, i, t] + c[l]) + \
                                     (1 - u[l, i, t]) * V[l, i, t + 1] + u[l, i, t] * V[l, i + 1, t + 1]
                    else:
                        end_node = model.edges[l, 1]
                        # tmp = (critic(np.array([l, i / n_cell, (t + 1) / n_cell])) - pi[end_node, t + 1]) / delta_t - rho[l, i, t]
                        tmp = (V[l, i, t + 1] - pi[end_node, t + 1]) / delta_t - rho[l, i, t]
                        u[l, i, t] = min(max(0, tmp), 1)
                        # critic_values.append(delta_t * (0.5 * u[l, i, t] ** 2 + rho[l, i, t] * u[l, i, t] + c[l]) + (1 - u[l, i, t]) * critic(np.array([l, i / n_cell, (t + 1) / n_cell])) + u[l, i, t] * pi[end_node, t + 1])
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
                        # min_cost = min(min_cost, critic(np.array([k, 0, t / n_cell])))
                        min_cost = min(min_cost, V[k, 0, t])

                min_link = [link for link in out_link if V[link, 0, t] == min_cost]
                # min_link = [link for link in out_link if critic(np.array([link, 0, t / n_cell])) == min_cost]
                for k in min_link:
                    beta[k, t] = 1 / len(min_link)

                pi[node, t] = 0
                for link in out_link:
                    pi[node, t] = pi[node, t] + beta[link, t] * V[link, 0, t]
                    # pi[node, t] = pi[node, t] + beta[link, t] * critic(np.array([link, 0, t / n_cell]))

    # critic_values = torch.tensor(critic_values)
    # for _ in range(100):
    #     preds = torch.reshape(critic(np.array(states)), (1, len(critic_values)))
    #     critic_loss = (critic_values - preds).abs().mean()
    #     critic_optimizer.zero_grad()
    #     critic_loss.backward()
    #     critic_optimizer.step()

    return u, V, beta, pi