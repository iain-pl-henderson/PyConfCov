import numpy as np
from scipy.stats import multivariate_t

def gen_gaussian(n, m, k, l, mean, cov, ell = 5., sig2 = 1., nugget = 0.):
    """
    Generates iid Gaussian data with specified mean and covariance.
    :param n: (int) number of examples in full calibration set
    :param m: (int) number of examples in test set
    :param k: (int) dimension of x
    :param l: (int) dimension of y
    :param mean: (str) name of mean type for data
    :param cov: (str) name of covariance type for data
    :param ell: (float) lengthscale of covariance matrix/covariance function
    :param sig2: (float) varaince of covariance matrix/covariance function
    :param nugget: (float) value of Tichonov regularization param. in covariance
    :return: data_calib, data_test, A_theoretic
    :A_theoretic: corresponds to the matrix A_infty in Prop. 4.1 of the paper
    Supported mean are : 'zero'
    Supported cov are : 'matern-1/2', 'matern-3/2', 'matern-5/2', 'matern-7/2', 'id'
    """

    d = k + l
    
    # compute mean vector
    if mean == 'zero':
        mean = np.zeros(d)
        mean = mean.tolist()
        
    # compute covariance matrix
    if cov == 'matern-1/2':
        cov_mat = np.zeros((d, d))
        rho = np.exp(-1/ell)
        # lazy loop
        for i in range(d):
            for j in range(i+1):
                exp_h = (i - j)
                cov_mat[i, j] = rho ** exp_h
    elif cov == 'matern-3/2':
        cov_mat = np.zeros((d, d))
        rho = np.exp(-np.sqrt(3)/ell)
        S = -np.log(rho)
        # lazy loop
        for i in range(d):
            for j in range(i+1):
                exp_h = (i - j)
                cov_mat[i, j] = (1+S*exp_h) * rho ** exp_h
    elif cov == 'matern-5/2':
        cov_mat = np.zeros((d, d))
        rho = np.exp(-np.sqrt(5)/ell)
        S = -np.log(rho)
        S2 = S*S
        # lazy loop
        for i in range(d):
            for j in range(i+1):
                exp_h = (i - j)
                cov_mat[i, j] = (1+S*exp_h + S2*exp_h*exp_h/3) * rho ** exp_h
    elif cov == 'matern-7/2':
        cov_mat = np.zeros((d, d))
        rho = np.exp(-np.sqrt(7)/ell)
        S = -np.log(rho)
        S2 = S*S
        S3 = S*S*S
        # lazy loop
        for i in range(d):
            for j in range(i+1):
                exp_h = (i - j)
                exp_h2 = exp_h * exp_h
                cov_mat[i, j] = (1+S*exp_h + 2*S2*exp_h2/5 + S3*exp_h2*exp_h/15) * rho ** exp_h
    elif cov == 'id':
        cov_mat = np.eye(d)
    cov_mat = sig2 * (cov_mat + cov_mat.transpose() - np.diag(np.diag(cov_mat)))
    data = np.random.multivariate_normal(mean, cov_mat, n + m)

    data_calib = data[0:n, :]
    data_test = data[n:(n + m), :]

    am = cov_mat[k:(k+l), k:(k+l)] + nugget * np.eye(l) - np.transpose(cov_mat[0:k, k:(k+l)]) @ np.linalg.solve(cov_mat[0:k, 0:k] + nugget * np.eye(k), cov_mat[0:k, k:(k+l)])
    A_theoretic = am

    return data_calib, data_test, A_theoretic

def gen_cauchy(n, m, k, l, loc, shape, ell = 5, sig2 = 1, nugget = 0):
    """
    Generates iid Cauchy data with specified position and scattter matrix
    :param n: (int) number of examples in full calibration set
    :param m: (int) number of examples in test set
    :param k: (int) dimension of x
    :param l: (int) dimension of y
    :param loc: (str) name of mean type for data
    :param shape: (str) name of covariance type for data
    :return: data_calib, data_test, A_theoretic, m_cond
    :A_theoretic: corresponds to the matrix A_infty in Prop. 4.1 of the paper
    :m_cond: 
    """
    d = k + l
    # compute mean vector

    if loc == 'zero':
        loc = np.zeros(d)
        loc = loc.tolist()

    # compute shapeariance matrix
    if shape == 'matern-1/2':
        shape_mat = np.zeros((d, d))
        rho = np.exp(-1/ell)
        # lazy loop
        for i in range(d):
            for j in range(i+1):
                exp_h = (i - j)
                shape_mat[i, j] = rho ** exp_h
    elif shape == 'matern-3/2':
        shape_mat = np.zeros((d, d))
        rho = np.exp(-np.sqrt(3)/ell)
        S = -np.log(rho)
        # lazy loop
        for i in range(d):
            for j in range(i+1):
                exp_h = (i - j)
                shape_mat[i, j] = (1+S*exp_h) * rho ** exp_h
    elif shape == 'matern-5/2':
        shape_mat = np.zeros((d, d))
        rho = np.exp(-np.sqrt(5)/ell)
        S = -np.log(rho)
        S2 = S*S
        # lazy loop
        for i in range(d):
            for j in range(i+1):
                exp_h = (i - j)
                shape_mat[i, j] = (1+S*exp_h + S2*exp_h*exp_h/3) * rho ** exp_h
    elif shape == 'matern-7/2':
        shape_mat = np.zeros((d, d))
        rho = np.exp(-np.sqrt(7)/ell)
        S = -np.log(rho)
        S2 = S*S
        S3 = S*S*S
        # lazy loop
        for i in range(d):
            for j in range(i+1):
                exp_h = (i - j)
                exp_h2 = exp_h * exp_h
                shape_mat[i, j] = (1+S*exp_h + 2*S2*exp_h2/5 + S3*exp_h2*exp_h/15) * rho ** exp_h
    elif shape == 'id':
        shape_mat = np.eye(d)
    shape_mat = sig2 * (shape_mat + shape_mat.transpose() - np.diag(np.diag(shape_mat)))

    rv = multivariate_t(loc, shape_mat, df=1)
    data = rv.rvs(size=n+m)

    data_calib = data[0:n, :]
    data_test = data[n:(n + m), :]

    am = shape_mat[k:(k+l), k:(k+l)] + nugget * np.eye(l) - np.transpose(shape_mat[0:k, k:(k+l)]) @ np.linalg.solve(shape_mat[0:k, 0:k] + nugget * np.eye(k), shape_mat[0:k, k:(k+l)])
    A_theoretic = am
    m_cond = np.linalg.solve(shape_mat[0:k, 0:k] + nugget * np.eye(k), shape_mat[0:k, k:(k+l)])

    return data_calib, data_test, A_theoretic, m_cond


class StructType:  # MATLAB-style struct type, taken from
    # user Girardi from https://stackoverflow.com/questions/11637045/complex-matlab-like-data-structure-in-python-numpy-scipy
    def __init__(self, **kwargs):
        self.Set(**kwargs)

    def Set(self, **kwargs):
        self.__dict__.update(kwargs)

    def SetAttr(self, lab, val):
        self.__dict__[lab] = val