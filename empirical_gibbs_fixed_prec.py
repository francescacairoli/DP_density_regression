'''
Gibbs 3 step algorithm
'''

from numpy import random
import time
import scipy.stats

from my_code.fixed_precision_distribs import *


def w_kernel(x_1, x_2, psi):
    '''
    :param psi: smoothing parameter
    '''
    return np.exp(-(psi) * (np.linalg.norm(x_1 - x_2, 2)) ** 2)



def gibbs_mean(y, x, n_iter, burn_in, hypers, alpha, psi):
    '''
    n_iter: numbero iterationi
    dim = dimension of the predictors' space
    N = number of observations
    alpha: innovation parameter
    psi: smoothing parameter
    '''

    dim = x.shape[1]
    N = len(y) # number of subjects

    start = time.time()
    '''
    Initialize parameters
    '''
    tau = np.random.gamma(hypers["a_tau"], scale=1 / hypers["b_tau"])  # inverso della varianza (precisione)

    # pre-computations of constant quantities

    invVb0 = np.linalg.inv(hypers["V_beta_0"])
    invVbo_b0 = np.dot(invVb0, hypers["beta_0"])
    nu0_Sigma0 = hypers["nu_0"] * hypers["Sigma_0"]

    beta = np.random.multivariate_normal(hypers["beta_0"], hypers["V_beta_0"])  # media della misura di base
    Sigma_beta_rec = wishart.rvs(df=hypers["nu_0"], scale=np.linalg.inv(nu0_Sigma0))  # inverso di Sigma (matrice di covarianza)

    '''
    Initialize the information for clusters' containers
    '''
    nclusters = 1  # number of cluster (componenti della mistura)
    S = np.array([1] * N)  # vettore che indica a quale cluster sono assegnati i soggetti

    theta = np.copy(beta)  # parametri dei cluster (medie delle normali) UNIQUE VALUEs
    phi = beta*np.ones((N, dim))  # media a cui è associato ogni soggetto

    # containers for saving parameter values at each iterate
    theta_f = []
    K_f = []
    S_f = []
    tau_f = []
    Sigma_beta_rec_f = []
    beta_f = []

    # pre-computation of the weights between subjects of the training set
    W = np.zeros((N, N))
    for i in range(N):
        for j in range(N):
            if (i != j):
                W[i, j] = w_kernel(x[i], x[j], psi)

    # pre-computations
    XXt = np.array([np.dot(xj.reshape(dim, 1), xj.reshape(1, dim)) for xj in x])
    Xy = x * y.reshape(N, 1)

    for iter in range(n_iter):
        t0 = time.time()
        print("iter n. = ", iter)

        # 1 Aggiorno locazione punti nei cluster
        for i in range(N):

            S_1hot = np.eye(nclusters + 1)[S][:, 1:]
            PDF = norm.pdf(np.ones(nclusters) * y[i], loc=np.dot(theta, x[i]), scale=np.ones(nclusters) * np.sqrt(1 / tau))
            qih = np.hstack((alpha * marginal(y[i], x[i], beta, Sigma_beta_rec, tau), np.dot(S_1hot.T, W[i]) * PDF))
            qih = qih / np.sum(qih)

            s = random.choice(np.arange(nclusters + 1), size=None, p=qih)  # alloco x_i in un cluster con prob qih

            if (s == 0):  # controllo se va creato nuovo cluster
                # creo nuovo valore di teta
                phi[i] = sample_new_phi(XXt[i], Xy[i], beta, Sigma_beta_rec, tau)

                theta, ind_s = np.unique(phi, return_inverse=True, axis=0)
                S = ind_s + 1  # aggiorno valori di S in modo che non ci siano cluster vuoti

                nclusters = int(np.amax(S))


            else:  # associo l'osservazione i al cluster s

                phi[i] = phi[(S == s)][0]
                theta, ind_s = np.unique(phi, return_inverse=True, axis=0)
                S = ind_s + 1
                nclusters = int(np.amax(S))

                assert len(theta) == nclusters

        # 2 aggiorno valore di teta in ogni cluster
        b = np.dot(Sigma_beta_rec, beta)

        for h in range(1, nclusters + 1):
            h_mask = (S == h)

            c_h = (Xy[h_mask]).T
            d_h = XXt[h_mask]

            V_betah = np.linalg.inv(Sigma_beta_rec + (tau * np.sum(d_h, axis=0)))
            beta_h = np.dot(V_betah, b + tau * np.sum(c_h, axis=1))

            theta[h - 1] = np.random.multivariate_normal(beta_h, V_betah)
            phi[h_mask] = theta[h - 1]

        # 3 aggiorno gli altri parametri
        tau = sample_tau(N, x, y, hypers["a_tau"], hypers["b_tau"], phi)
        beta = sample_beta(theta.shape[0], hypers["V_beta_0"], Sigma_beta_rec, hypers["beta_0"], theta, invVb0, invVbo_b0)
        Sigma_beta_rec = sample_Sigma_beta_rec(theta, beta, hypers["nu_0"], hypers["Sigma_0"], theta.shape[0], nu0_Sigma0)

        if iter >= burn_in:
            K_f.append(nclusters)
            theta_f.append(np.array(theta))
            S_f.append(np.array(S))
            tau_f.append(tau)
            Sigma_beta_rec_f.append(np.array(Sigma_beta_rec))
            beta_f.append(np.array(beta))

        print(time.time() - t0)

    trace = ({"k": K_f, "theta": theta_f, "S": S_f, "tau": tau_f, "Sigma_beta^(-1)": Sigma_beta_rec_f, "beta": beta_f})
    end = time.time()
    print("overall time ", (end - start)/3600, "hours")

    return trace




def predictive_estimator(x, x_new, y_grid, T, k, S, theta, tau, S_beta_rec, beta, alpha, psi):
    grid_size = y_grid.shape[0]

    W_new = [w_kernel(xi, x_new, psi) for xi in x]

    y_density = np.zeros(grid_size)
    for i in range(grid_size):
        y = np.zeros(T)
        for t in range(-T, -1):
            k_t = int(k[t])
            S_1hot = np.eye(k_t + 1)[S[t]][:, 1:]

            den = alpha+np.sum(W_new)
            weight = np.dot(S_1hot.T, W_new)

            normal_0 = scipy.stats.norm(np.dot(x_new.T, beta[t]), np.sqrt(1 / tau[t]))
            y[t] = alpha * normal_0.pdf(y_grid[i]) / den
            for h in range(k_t):
                normal_h = scipy.stats.norm(np.dot(x_new.T, theta[t][h]), np.sqrt(1 / tau[t]))
                y[t] += weight[h] * normal_h.pdf(y_grid[i]) / den
        y_density[i] = np.mean(y)

    return y_density



def expected_pred_density(x, x_new, T, k, S, theta, beta, alpha, psi):
    exp_y = np.zeros(T)

    W_new = [w_kernel(xi, x_new, psi) for xi in x]

    for t in range(-T, -1):

        k_t = int(k[t])
        S_1hot = np.eye(k_t + 1)[S[t]][:, 1:]

        weight = np.dot(S_1hot.T, W_new)

        norm = alpha + np.sum(W_new)

        y = alpha * np.dot(x_new, beta[t])
        for h in range(k_t):
            y += weight[h] * np.dot(x_new, theta[t][h, :])

        exp_y[t] = y / norm

    return np.mean(exp_y)

