# difference with the regression subpackage : we can do some preprocessing

import numpy as np
from scipy.special import gamma as gamma_fun


def compute_constants(n, k, l):
    """
    :param n: number of effective examples
    :param k: number of points in the past
    :param l: number of points in the future
    :return: struct containing a vector of ones, the vector w, the L matrix
    """
    np1 = n+1

    # build v
    v = np.ones((np1, 1))
    v[n] = n
    v = v/np1
    nv2 = n/np1
    w = v/np.sqrt(nv2)
    p = k + l

    L = np.zeros((p, l))
    L[k:(p-1), :] = np.eye(l)
    constants = StructType()
    constants.v = v
    constants.nv2 = nv2
    constants.ones_h = np.ones((np1, 1))/np.sqrt(np1)
    constants.L = L
    return constants

def reshape_data():
    return 0

def compute_ellipsoid(X, R, constants, params):
    """
    The equation of the ellipsoid is (z-z0)^T*A*(z-z0) < R
    This function returns (A, z0, R) in an ellipsoid object
    X is an (n+1) x k matrix containing the input points; examples are stored in row
    R is an (n+1) x l matrix containing the residuals
    """
    ### bookkeeping
    n = len(X) - 1
    k = len(X[0])
    l = len(R[0])
    p = k + l
    np1 = n+1

    w = constants.pw
    ones_h = constants.ones_h
    L = constants.L
    nv2 = constants.nv2
    nv = np.sqrt(nv2)

    lambda_h = params.lambda_h
    m_alpha = 1 - params.alpha

    # build data matrix
    V = np.vstack((X, R))
    Vc = V[n, :]
    Xc = Vc

    # center the data and project onto the orthogonal of v
    B = V - np.outer(w, V @ w) - np.outer(ones_h, V @ ones_h)
    B_lambda = B.transpose() @ B + n*lambda_h*np.eye(p)

    dC_lambda = np.diag(B @ np.linalg.solve(B_lambda, B.transpose()))
    q = 1/(n*np1) + np.quantile(dC_lambda, m_alpha)

    Ub = B_lambda[0:(k-1), 0:(k-1)]  # k x k
    Vb = B_lambda[0:(k-1), k:(p-1)]  # k x l
    Wb = B_lambda[k:(p-1), k:(p-1)]  # l x l

    Am = Wb - np.transpose(Vb) @ np.linalg.solve(Ub, Vb) # inverse of A
    A = nv2 * np.linalg.inv(Am) # schur complement
    z0 = - Am @ L.transpose() @ np.linalg.solve(B_lambda, Vc)
    s0 = Xc @ np.linalg.solve(Ub,Xc)
    R = q/(nv2-q) - s0

    ell = Ellipsoid(A, z0, R)
    return ell


class Ellipsoid:  # use a dictionary instead?

    dim = int(3)
    A = np.zeros((dim, dim))
    z0 = np.zeros((dim, 1))
    R = -1
    vol = -1
    eccentricity = -1

    def __init__(self, A, z0, R):
        self.dim = len(A)
        self.A = A
        self.z0 = z0
        self.R = R

    def compute_vol(self):
        vl = np.pi**(self.dim/2) / gamma_fun(self.dim/2 + 1)
        self.vol = vl*np.sqrt(np.linalg.det(self.A))

    def compute_eccentricity(self):
        eig_h = np.linalg.eigh(self.A)
        r = min(eig_h)/max(eig_h)
        self.eccentricity = np.sqrt(1 - r*r)


class StructType:  # piqué d'internet
    def __init__(self,**kwargs):
        self.Set(**kwargs)

    def Set(self,**kwargs):
        self.__dict__.update(kwargs)

    def SetAttr(self,lab,val):
        self.__dict__[lab] = val