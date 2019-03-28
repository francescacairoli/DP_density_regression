'''
Gibbs 3 step algorithm
'''

from numpy import random
import time
import scipy.stats

from my_code.vatying_prec_distribs import *

def w_kernel(x_1, x_2, psi):
    '''
    :param psi: smoothing parameter
    '''
    return np.exp(-(psi) * (np.linalg.norm(x_1 - x_2, 2)) ** 2)



def gibbs_mean_scale(y, x, n_iter, burn_in, hypers, alpha, psi):
    '''
    n_iter: numbero iterationi
    dim = dimension of the predictors' space
    N = number of observations
    alpha: innovation parameter
    psi: smoothing parameter
    '''

    dim = x.shape[1]
    dim=1
    N = len(y) # number of subjects

    start = time.time()
    '''
    Initialize parameters
    '''
    # pre-computations of constant quantities

    invV0 = np.linalg.inv(hypers["V_0"])
    k = np.random.gamma(hypers["a_k"],1/hypers["b_k"])
    beta = hypers["beta_0"]


    '''
    Initialize the information for clusters' containers
    '''
    nclusters = 1  # number of cluster (componenti della mistura)
    S = np.array([1] * N)  # vettore che indica a quale cluster sono assegnati i soggetti
    theta = np.zeros((1,dim+1))
    theta[0,:] = np.hstack((beta, 1))  # parametri dei cluster (medie delle normali) UNIQUE VALUEs
    phi = theta*np.ones((N, dim+1))  # media a cui è associato ogni soggetto

    # containers for saving parameter values at each iterate
    theta_f = []
    nclusters_f = []
    S_f = []
    k_f = []
    V_f = []
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
    XtX = np.dot(x.reshape(dim, N), x.reshape(N, dim))

    V_tmp = N*np.linalg.inv(XtX)
    V = V_tmp/k
    V = np.identity(dim)
    for iter in range(n_iter):
        t0 = time.time()
        print("iter n. = ", iter)


        # 1 Aggiorno locazione punti nei cluster
        for i in range(N):

            S_1hot = np.eye(nclusters + 1)[S][:, 1:]
            PDF = norm.pdf(np.ones(nclusters) * y[i], loc=np.dot(theta[:,:dim], x[i]), scale=np.sqrt(1 / theta[:,-1]))

            qih = np.hstack((alpha * marginal(y[i], x[i], beta, V, hypers["a_tau"], hypers["b_tau"]), np.dot(S_1hot.T, W[i]) * PDF))
            qih = qih / np.sum(qih)

            s = random.choice(np.arange(nclusters + 1), size=None, p=qih)  # alloco x_i in un cluster con prob qih

            if (s == 0):  # controllo se va creato nuovo cluster
                # creo nuovo valore di teta
                phi[i] = update_theta(y[i], XXt[i], Xy[i], beta, V, hypers["a_tau"], hypers["b_tau"])
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

        for h in range(1, nclusters + 1):
            h_mask = (S == h)

            theta_h = update_theta(y[h_mask], XXt[h_mask], Xy[h_mask], beta, V, hypers["a_tau"], hypers["b_tau"])
            phi[h_mask] = theta_h

        # 3 aggiorno gli altri parametri
        beta, k = update_beta_k(N, theta, XtX, hypers["beta_0"], invV0, hypers["a_k"], hypers["b_k"])
        V = V_tmp/k
        if iter >= burn_in:
            nclusters_f.append(nclusters)
            theta_f.append(np.array(theta))
            S_f.append(np.array(S))
            k_f.append(k)
            V_f.append(V)
            beta_f.append(np.array(beta))

        print(time.time() - t0)

    trace = ({"nclusters": nclusters_f, "theta": theta_f, "S": S_f, "k": k_f, "V": V_f, "beta": beta_f})
    end = time.time()
    print("overall time ", (end - start)/3600, "hours")

    return trace




def predictive_estimator(x, x_new, y_grid, T, nclusters, S, theta, beta, V, alpha, psi):
    grid_size = y_grid.shape[0]
    dim = 1
    W_new = [w_kernel(xi, x_new, psi) for xi in x]

    y_density = np.zeros(grid_size)
    for i in range(grid_size):
        y = np.zeros(T)
        for t in range(-T, -1):
            nclus_t = int(nclusters[t])
            S_1hot = np.eye(nclus_t + 1)[S[t]][:, 1:]

            den = np.sum(W_new)
            weight = np.dot(S_1hot.T, W_new)

            for h in range(nclus_t):
                normal_h = scipy.stats.norm(np.dot(x_new.T, theta[t][h,:dim]), np.sqrt(1 / theta[t][h,-1]))
                y[t] += weight[h] * normal_h.pdf(y_grid[i]) / den
        y_density[i] = np.mean(y)

    return y_density



def expected_pred_density(x, x_new, T, nclusters, S, theta, beta, alpha, psi):
    exp_y = np.zeros(T)
    dim = 1
    W_new = [w_kernel(xi, x_new, psi) for xi in x]

    for t in range(-T, -1):

        nclus_t = int(nclusters[t])
        S_1hot = np.eye(nclus_t + 1)[S[t]][:, 1:]

        weight = np.dot(S_1hot.T, W_new)

        den = alpha + np.sum(W_new)

        y = alpha * np.dot(x_new, beta[t])
        for h in range(nclus_t):
            y += weight[h] * np.dot(x_new, theta[t][h, :dim])

        exp_y[t] = y / den

    return np.mean(exp_y)

