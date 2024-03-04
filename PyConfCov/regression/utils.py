import numpy as np
from scipy.stats import multivariate_t
# generate different types of data

def gen_gaussian(n, m, k, l, mean, cov, nugget = 0):
    """
    Generates iid Gaussian data with specified mean and covariance.
    :param n: (int) number of examples in full calibration set
    :param m: (int) number of examples in test set
    :param k: (int) dimension of x
    :param l: (int) dimension of y
    :param mean: (str) name of mean type for data
    :param cov: (str) name of covariance type for data
    :return: data_calib, data_test
    """

    d = k + l
    # compute mean vector

    if mean == 'zero':
        mean = np.zeros(d)
        mean = mean.tolist()

    # compute covariance matrix
    if cov == 'matern-1/2':
        cov_mat = np.zeros((d, d))
        rho = 0.1
        sig = 2
        # lazy loop
        for i in range(d):
            for j in range(i+1):
                exp_h = (i - j)
                cov_mat[i, j] = rho ** exp_h
    elif cov == 'matern-3/2':
        cov_mat = np.zeros((d, d))
        rho = 0.9
        sig = 1
        S = -np.log(rho)
        # lazy loop
        for i in range(d):
            for j in range(i+1):
                exp_h = (i - j)
                cov_mat[i, j] = (1+S*exp_h) * rho ** exp_h

    cov_mat = sig * (cov_mat + cov_mat.transpose() - np.eye(d))
    eig_h = np.linalg.eigh(cov_mat)
    data = np.random.multivariate_normal(mean, cov_mat, n + m)

    data_calib = data[0:n, :]
    data_test = data[n:(n + m), :]

    am = cov_mat[k:(k+l), k:(k+l)] + nugget * np.eye(l) - np.transpose(cov_mat[0:k, k:(k+l)]) @ np.linalg.solve(cov_mat[0:k, 0:k] + nugget * np.eye(k), cov_mat[0:k, k:(k+l)])
    A_theoretic = np.linalg.inv(am)

    return data_calib, data_test, A_theoretic

def gen_cauchy(n, m, k, l, loc, shape):
    """
    Generates iid Cauchy data with specified position and scattter matrix
    :param n: (int) number of examples in full calibration set
    :param m: (int) number of examples in test set
    :param k: (int) dimension of x
    :param l: (int) dimension of y
    :param loc: (str) name of mean type for data
    :param shape: (str) name of covariance type for data
    :return: data_calib, data_test
    """
    d = k + l
    # compute mean vector

    if loc == 'zero':
        loc = np.zeros(d)
        loc = loc.tolist()

    # compute covariance matrix
    if shape == 'matern-1/2':
        shape_mat = np.zeros((d, d))
        rho = 0.1
        sig = 2
        # lazy loop
        for i in range(d):
            for j in range(i+1):
                exp_h = (i - j)
                shape_mat[i, j] = rho ** exp_h
    elif shape == 'matern-3/2':
        shape_mat = np.zeros((d, d))
        rho = 0.8
        sig = 2
        S = -np.log(rho)
        # lazy loop
        for i in range(d):
            for j in range(i+1):
                exp_h = (i - j)
                shape_mat[i, j] = (1+S*exp_h) * rho ** exp_h

    shape_mat = sig * (shape_mat + shape_mat.transpose() - np.eye(d))

    rv = multivariate_t(loc, shape_mat, df=1)
    data = rv.rvs(size=n+m)

    data_calib = data[0:n, :]
    data_test = data[n:(n + m), :]

    return data_calib, data_test


def plot_regions():
    return 0

