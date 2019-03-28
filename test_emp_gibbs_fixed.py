import pickle
from my_code.generate_synthetic_data import *
from my_code.empirical_gibbs_fixed_prec import *

import numpy as np

'''
Test on simulated data from a mixture of normals
'''

N = 50
alpha = 0.5
psi = N / 25

dim = 2

compute = True

file_name = "first_fixed_trial"


if compute:

    x, y = generate_dataset(N) # mixture of normals
    x = x[:, 1:3]

    hypers = {"beta_0": np.zeros((dim)), "V_beta_0": np.linalg.inv(x.reshape(dim, N) @ x.reshape(N, dim)) / N,
              "nu_0": dim, "Sigma_0": np.identity((dim)), "a_tau": 1, "b_tau": 0.5}

    trace = gibbs_mean(y, x, 10000, 1000, hypers, alpha, psi)
    results = {"x": x, "y": y, "trace": trace}
    pickle.dump(results, open(file_name, "wb"))
else:
    res = pickle.load(open(file_name, "rb"))
    trace = res["trace"]
    x = res["x"]
    y = res["y"]


S = np.array(trace['S'])
S = np.array([np.array([int(si) for si in Si]) for Si in S])
k = np.array(trace['k'])
theta = np.array(trace['theta'])
tau = np.array(trace['tau'])
beta = np.array(trace['beta'])
S_beta_rec = np.array(trace["Sigma_beta^(-1)"])

# INFERENCE

grid_size = 100 # resolution of the density prediction
y_grid = np.linspace(-0.5, 1.5, grid_size) # domain of the density prediction

n_xnew = 100 # number of test points

x1 = np.linspace(x[:, 0].min(), x[:, 0].max(), n_xnew)
x2 = np.linspace(x[:, 1].min(), x[:, 1].max(), n_xnew)


pp_x = np.array([np.array([x1[i], x2[i]]) for i in range(0, n_xnew)]) # test set
T = 500 # number of iterations to average over


perc_bhv = True
expected_predictive_density = True

if True: #extra analysis on weight behaviour
    t = -10
    k_t = int(k[t])
    S_1hot = np.eye(k_t + 1)[S[t]][:, 1:]

    plt.scatter(x[:, 1], S[t])
    plt.show()

    W_new_plot = [w_kernel(xi, [0.5, 1], psi) for xi in x]
    plt.scatter(x[:, 0], W_new_plot)
    plt.show()

    w_plot = []
    for i, x_new in enumerate(pp_x):
        # W_new = [w_kernel_f(kernel, xi, x_new)[0] for xi in x]
        W_new = [w_kernel(xi, x_new, psi, prec) for xi in x]
        norm = alpha + np.sum(W_new)
        w_tmp = np.hstack((alpha, np.dot(S_1hot.T, W_new))) / norm
        w_plot.append(w_tmp)
    w_plot = np.array(w_plot)
    for h in range(k_t + 1):
        plt.plot(pp_x[:, 0], w_plot[:, h])
    plt.show()

if expected_predictive_density:

    true_exp = lambda x: x * np.exp(-2 * x) + (1 - np.exp(-2 * x)) * x ** 4

    expect_density = np.zeros(n_xnew)
    for i in range(pp_x.shape[0]):
        expect_density[i] = expected_pred_density(x, pp_x[i, :], T, k, S, theta, beta, alpha, psi)

    plt.plot(pp_x[:, 0], expect_density, 'r', label="estim")
    plt.scatter(x[:, 0], y, label="training_points")
    plt.plot(pp_x[:, 0], true_exp(pp_x[:, 0]), 'g--', label="true")
    plt.legend()
    plt.show()


if perc_bhv:

    percentiles = [99, 90, 75, 50, 25, 10]
    for perc in percentiles:
        xp = np.percentile(pp_x, perc, axis=0)

        mu = [xp[0], xp[0] ** 4]  # means
        V = [0.01, 0.04]  # variances
        p = [(np.exp(-2 * xp[0])), (1 - np.exp(-2 * xp[0]))]  # proportions
        model_p = MM([gauss(mu0, V0) for mu0, V0 in zip(mu, V)], p)
        # Plot the generative mixture model
        f = plt.figure(figsize=(8, 6))
        ax = f.add_subplot(111)
        model_p.plot(axis=ax)
        yp = predictive_estimator(x, xp, y_grid, T, k, S, theta, tau, S_beta_rec, beta, alpha, psi)
        exp_value = np.dot(yp, y_grid) / np.sum(yp)
        plt.plot(y_grid, yp)
        plt.vlines(exp_value, ymin=0, ymax=1)
        plt.title("Percentile = {0}".format(perc))

        plt.show()
