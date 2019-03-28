import numpy as np

from scipy.stats import wishart
from scipy.stats import norm

'''
Update the hyperparameters
FIXED PRECISION tau
'''

def sample_tau(N, x, y, a_tau, b_tau, phi):
    a = a_tau + N/2
    c = np.sum((y-np.diag(np.dot(x,phi.T)))**2)
    b = b_tau + (1/2) * c
    return np.random.gamma(a,scale = 1/b)


def sample_beta(k, V_beta_0, Sigma_beta_rec, beta_0, teta, invVb0, invVbo_b0):
    V_beta_hat = np.linalg.inv(invVb0 + k * (Sigma_beta_rec))

    c = np.array([np.dot(Sigma_beta_rec, tetaj) for tetaj in teta])

    beta_hat = np.dot(V_beta_hat, invVbo_b0 + np.sum(c, axis=0))
    return np.random.multivariate_normal(beta_hat, V_beta_hat)


# campiona da Sigma_beta**(-1)
def sample_Sigma_beta_rec(teta, beta, nu_0, Sigma_0, k, nu0_Sigma0):
    tb = (teta - beta)
    dim = teta.shape[1]
    c = np.array([np.dot(tbj.reshape((dim, 1)), tbj.reshape((1, dim))) for tbj in tb])  # da sistemare

    mean = np.linalg.inv(np.sum(c, axis=0) + nu0_Sigma0)

    var = k + nu_0
    return wishart.rvs(df=var, scale=mean)


'''
Needed to compute the probability of generating new clusters
'''

def marginal(yi, xi, beta, Sigma_beta_rec, tau):
    M = np.dot(xi, beta)
    dim = xi.shape[0]
    V = (1 / tau) + xi.reshape(1, dim) @ (np.linalg.inv(Sigma_beta_rec) @ xi.reshape(dim, 1))

    return norm.pdf(yi, loc=M, scale=np.sqrt(V[0]))

# Sample a new value from the posterior Gi0
def sample_new_phi(xtx_i, xy_i, beta, Sigma_beta_rec, tau):
    V_hat = np.linalg.inv(Sigma_beta_rec + tau * xtx_i)
    b = np.dot(Sigma_beta_rec, beta)
    c = tau * xy_i + b
    beta_hat = np.dot(V_hat, c)
    return(np.random.multivariate_normal(beta_hat,V_hat))