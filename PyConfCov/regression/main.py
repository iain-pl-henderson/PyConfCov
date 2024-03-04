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
    L[k:p, :] = np.eye(l)
    constants = StructType()
    constants.v = v
    constants.nv2 = nv2
    constants.ones_h = np.ones((np1, 1))/np.sqrt(np1)
    constants.L = L
    return constants


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

    nv2 = constants.nv2
    nv = np.sqrt(nv2)
    w = (constants.v)/nv
    ones_h = constants.ones_h
    L = constants.L

    lambda_h = params.lambda_h
    m_alpha = 1 - params.alpha

    # build data matrix
    V = np.hstack([X, R])

    # center the data and project onto the orthogonal of v
    W = V - np.outer(ones_h, V.transpose() @ ones_h)
    Vc = W[n, :]
    Xc = Vc[0:k]

    B = W - np.outer(w, V.transpose() @ w)
    B_lambda = B.transpose() @ B + n*lambda_h*np.eye(p)

    dC_lambda = np.diag(B @ np.linalg.solve(B_lambda, B.transpose()))
    q = 1/(n*np1) + np.quantile(dC_lambda, m_alpha)

    Ub = B_lambda[0:k, 0:k]  # (k x k)
    Vb = B_lambda[0:k, k:p]  # (k x l)
    Wb = B_lambda[k:p, k:p]  # (l x l)

    Am = Wb - np.transpose(Vb) @ np.linalg.solve(Ub, Vb)  # inverse of A, block-wise inversion (schur complement)
    A = nv2 * np.linalg.inv(Am)
    z0 = - Am @ L.transpose() @ np.linalg.solve(B_lambda, Vc)
    s0 = Xc @ np.linalg.solve(Ub,Xc)
    R = q/(nv2-q) - s0

    ell = Ellipsoid(A, z0, R)
    return ell

class Ellipsoid:
    """

    """
    dim = int(3)
    A = np.zeros((dim, dim))
    detA = -1.0
    z0 = np.zeros((dim, 1))
    R = -1.0
    vol = -1.0
    eccentricity = -1.0
    v0 = -1.0
    vl = -1.0
    axes = np.array([-1, -1, -1])

    def __init__(self, A, z0, R):
        self.dim = len(A)
        self.A = A
        self.z0 = z0
        self.R = R
        self.vl = np.pi**(self.dim/2) / gamma_fun(self.dim/2 + 1)

    def compute_axes(self):
        if self.R > 0:
            self.axes = np.sqrt(self.R / np.linalg.eigh(self.A).eigenvalues)

    def compute_vol(self):
        if self.R > 0:
            self.vol = self.vl * self.R ** (self.dim/2) / np.sqrt(np.linalg.det(self.A))
        else:
            self.vol = 0

    def compute_eccentricity(self):
        eig_h = np.linalg.eigh(self.A).eigenvalues
        if self.R > 0:
            if self.axes[0] == -1:
                self.axes = np.sqrt(self.R / eig_h)
        r_h = np.min(eig_h)/np.max(eig_h)
        self.eccentricity = np.sqrt(1 - r_h)

    def is_in_ellipsoid(self, z):
        res = 0
        if self.R > 0:
            zh = z-self.z0
            dh = zh @ (self.A @ zh)
            if dh < self.R:
                res = 1
        return res

    def set_R_v0(self):
        if self.v0 > 0:
            if self.detA == -1:  # it may have already been computed
                self.detA = np.linalg.det(self.A)
            eps2 = (self.v0 * np.sqrt(self.detA)/self.vl)**(2/self.dim)
            self.R = np.maximum(self.R, eps2)


def null_predictor():
    return 0

class StructType:  # piqué d'internet
    def __init__(self, **kwargs):
        self.Set(**kwargs)

    def Set(self, **kwargs):
        self.__dict__.update(kwargs)

    def SetAttr(self, lab, val):
        self.__dict__[lab] = val