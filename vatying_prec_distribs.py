import numpy as np
import math
from scipy.stats import norm
import scipy.special
'''
Update the hyperparameters
VARYING MEAN AND PRECISION
'''


def update_beta_k(N, theta, XtX, beta_0, invV_0, a_k, b_k):

    dim = len(beta_0)
    nclusters = theta.shape[0]

    invV_hat = invV_0+(XtX/N)*np.sum(theta[:,-1])
    V_hat = np.linalg.inv(invV_hat)
    cs = 0
    for h in range(nclusters):
        cs += np.dot(XtX,theta[h,:dim])*theta[h,-1]/N
    beta_hat = np.dot(V_hat, np.dot(invV_0,beta_0) + cs )

    a_hat = a_k+0.5*nclusters*dim

    scs = 0
    for h in range(nclusters):
        scs += np.dot(theta[h,:dim],np.dot(XtX,theta[h,:dim])*theta[h,-1])/N
    b_hat = b_k+0.5*(scs + np.dot(beta_0,np.dot(invV_0,beta_0))-np.dot(beta_hat,np.dot(invV_hat,beta_hat)))

    k = np.random.gamma(a_hat,1/b_hat)
    beta = np.random.multivariate_normal(beta_hat,V_hat/k)
    return beta, k



'''
Needed to compute the probability of generating new clusters
'''

def marginal(yi, xi, beta, V, a_tau, b_tau):
    dim = len(xi)

    invV = np.linalg.inv(V)

    C = (np.power(b_tau,a_tau))/scipy.special.gamma(a_tau)

    V_i = invV+np.dot(xi.reshape(dim,1),xi.reshape(1,dim))
    beta_i = np.dot(V_i, np.dot(invV,beta)+xi*yi)

    invV_i = np.linalg.inv(V_i)

    ai = a_tau+0.5*(dim+1)
    bi = b_tau+0.5*(yi**2+np.dot(beta,np.dot(invV,beta))-np.dot(beta_i,np.dot(invV_i,beta_i)))

    C_i = (np.power(bi,ai)) / scipy.special.gamma(ai)

    c_factor = C/(np.sqrt(2*math.pi)*C_i)

    return c_factor*((np.linalg.det(V_i)/np.linalg.det(V))**(dim/2)) #double_check dell'esponente


def update_theta(y_h, XX_h, Xy_h, beta, V, a_tau, b_tau):
    N_h = XX_h.shape[0]
    invV = np.linalg.inv(V)
    invV_h = invV+np.sum(XX_h,axis=0)
    V_h = np.linalg.inv(invV_h)
    beta_h = np.dot(V_h, np.dot(invV,beta)+np.sum(Xy_h,axis=0))

    a_h = a_tau+0.5*N_h

    b_tmp = np.sum(y_h**2)+np.dot(beta,np.dot(invV,beta))-np.dot(beta_h,np.dot(invV_h,beta_h))

    b_h = b_tau+0.5*b_tmp

    tau_h = np.random.gamma(a_h,1/b_h)
    beta_h = np.random.multivariate_normal(beta_h,V_h/tau_h)
    return np.hstack((beta_h, tau_h))