import pickle
from my_code.generate_synthetic_data import *
from my_code.empirical_gibbs_varying_prec import *

import numpy as np

'''
Test on simulated data from a mixture of normals
'''

N = 500
alpha = 0.5
psi = N / 25

dim = 1

compute = True

file_name = "first_fixed_trial"


if compute:

    x, y = generate_dataset(N) # mixture of normals
    x = x[:, 1:2]

    hypers = {"beta_0": np.zeros((dim)), "V_0": N*np.linalg.inv(x.reshape(dim, N) @ x.reshape(N, dim)), "a_tau": 1, "b_tau": 0.5, "a_k": 1, "b_k": 0.5}

    trace = gibbs_mean_scale(y, x, 10000, 1000, hypers, alpha, psi)
    results = {"x": x, "y": y, "trace": trace}
    pickle.dump(results, open(file_name, "wb"))
else:
    res = pickle.load(open(file_name, "rb"))
    trace = res["trace"]
    x = res["x"]
    y = res["y"]


S = np.array(trace['S'])
S = np.array([np.array([int(si) for si in Si]) for Si in S])
nclusters = np.array(trace['nclusters'])
theta = np.array(trace['theta'])
V = np.array(trace['V'])
beta = np.array(trace['beta'])
kS_beta_rec = np.array(trace["k"])

# INFERENCE

grid_size = 50 # resolution of the density prediction
y_grid = np.linspace(-0.5, 1.5, grid_size) # domain of the density prediction

n_xnew = 100 # number of test points

x1 = np.linspace(x[:, 0].min(), x[:, 0].max(), n_xnew)
#x2 = np.linspace(x[:, 1].min(), x[:, 1].max(), n_xnew)
#pp_x = np.array([np.array([x1[i], x2[i]]) for i in range(0, n_xnew)]) # test set

pp_x = x1

T = 100 # number of iterations to average over

perc_bhv = True
expected_predictive_density = True

if expected_predictive_density:

    true_exp = lambda x: x * np.exp(-2 * x) + (1 - np.exp(-2 * x)) * x ** 4

    expect_density = np.zeros(n_xnew)
    for i in range(pp_x.shape[0]):
        expect_density[i] = expected_pred_density(x, pp_x[i], T, nclusters, S, theta, beta, alpha, psi)

    plt.plot(pp_x, expect_density, 'r', label="estim")
    plt.scatter(x, y, label="training_points")
    plt.plot(pp_x, true_exp(pp_x), 'g--', label="true")
    plt.legend()
    plt.show()


if perc_bhv:

    percentiles = [99, 90, 75, 50, 25, 10]
    for perc in percentiles:
        xp = np.percentile(pp_x, perc, axis=0)

        mu = [xp, xp ** 4]  # means
        V = [0.01, 0.04]  # variances
        p = [(np.exp(-2 * xp)), (1 - np.exp(-2 * xp))]  # proportions
        model_p = MM([gauss(mu0, V0) for mu0, V0 in zip(mu, V)], p)
        # Plot the generative mixture model
        f = plt.figure(figsize=(8, 6))
        ax = f.add_subplot(111)
        model_p.plot(axis=ax)
        yp = predictive_estimator(x, xp, y_grid, T, nclusters, S, theta, beta, V, alpha, psi)
        exp_value = np.dot(yp, y_grid) / np.sum(yp)
        plt.plot(y_grid, yp)
        plt.vlines(exp_value, ymin=0, ymax=1)
        plt.title("Percentile = {0}".format(perc))

        plt.show()
