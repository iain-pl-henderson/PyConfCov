import numpy as np
from scipy.special import gamma as gamma_fun
import utils

def compute_constants(n, k, l):
    # optimal code would do without this function, 
    # but we keep it for now...
    """
    :param n: number of effective examples
    :param k: number of points in the past
    :param l: number of points in the future
    :return: struct containing a vector of ones (used to contain the vector v and matrix L)
    """

    constants = utils.StructType()
    constants.ones_h = np.ones((n, 1)) # is of size n
    return constants


def compute_ellipsoid(X, R, constants, params):
    """
    The equation of the ellipsoid is (z-z0)^T*A*(z-z0) < R
    This function returns (A, z0, R) in an ellipsoid object (see class below)
    X is an (n+1) x k matrix containing the input points; examples are stored in row
    R is an (n+1) x l matrix containing the residuals
    """

    ### book-keeping
    n = len(X) - 1
    k = len(X[0])
    l = len(R[0])
    p = k + l

    ones_h = constants.ones_h

    lambda_h = params.lambda_h
    m_alpha = 1 - params.alpha

    Vn = np.hstack([X[0:n], R[0:n]])  # first n samples
    Vbar = np.sum(Vn, axis=0)/n
    Xbar = Vbar[0:k]
    Rbar = Vbar[k:p]
    Xnp1c = X[n,:] - Xbar

    # center the data
    Bn = Vn - np.outer(ones_h, Vbar)

    # compute elements of ellipsoid
    # first the matrix
    D_mu = Bn.transpose() @ Bn + n*lambda_h*np.eye(p)
    D11 = D_mu[0:k, 0:k]  # (k x k)
    D12 = D_mu[0:k, k:p]  # (k x l)
    D22 = D_mu[k:p, k:p]  # (l x l)
    An = (D22 - np.transpose(D12) @ np.linalg.solve(D11, D12))/n

    # then rho
    Bh = np.linalg.solve(D_mu, Bn.transpose())
    diag_Pn = [Bn[i, 0:p] @ Bh[0:p, i] for i in range(n)]
    q_n_alpha_p1 = n * np.quantile(diag_Pn, (1+1/n) * m_alpha) + 1
    rho_n_alpha = q_n_alpha_p1/(1-q_n_alpha_p1/n) - 1 - n*Xnp1c @ np.linalg.solve(D11, Xnp1c)  # can be negative

    # then the center of the ellipsoid
    Z_0n = Rbar + np.transpose(D12) @ np.linalg.solve(D11, Xnp1c)

    ell = Ellipsoid(An, Z_0n, rho_n_alpha)
    ell.q_n_alpha = n * np.quantile(diag_Pn, (1+1/n) * m_alpha)
    ell.s0 = n*Xnp1c @ np.linalg.solve(D11, Xnp1c)  # should follow a Hotelling T2 distribution in the Gaussian case
    return ell

class Ellipsoid:
    """
    This class contains the parameters (A,z0,rho) corressponding to the ellipsoid
     given by (z-z0)^T * A^{-1} * (z-z0) < rho.
     It also has methods that perform elementary operations on them.
    """
    dim = int(3)
    A = np.zeros((dim, dim))
    A_inv = np.zeros((dim, dim))
    eig_A = np.array([-1, -1, -1])
    detA = -1.0
    z0 = np.zeros((dim, 1))
    rho = -1.0
    vol = -1.0
    eccentricity = -1.0
    v0 = -1.0
    vl = -1.0
    axes = np.array([-1, -1, -1])  # corresponds to the lengths of the semi-axes
    q_n_alpha = -1
    s0 = -1
    def __init__(self, A, z0, rho):
        self.dim = len(A)
        self.A = A
        self.A_inv = np.linalg.inv(A)
        self.z0 = z0
        self.rho = rho
        self.vl = np.pi**(self.dim/2) / gamma_fun(self.dim/2 + 1)

    def compute_axes(self):
        """
        Computes the eigenvalues of sqrt(rho*A), i.e. the axes of the ellipsoid
        """
        if self.rho > 0:
            if self.eig_A[0] == -1:
                self.eig_A = np.linalg.eigh(self.A).eigenvalues
            self.axes = np.sqrt(self.rho * self.eig_A)
        else:
            self.axes = np.array([0, 0, 0])

    def compute_vol(self):
        """
        Computes the volume of the ellipsoid
        """
        if self.rho > 0:
            self.vol = self.vl * self.rho ** (self.dim/2) * np.sqrt(np.linalg.det(self.A))
        else:
            self.vol = 0

    def compute_eccentricity(self):
        """
        Computes the eccentricity of the ellipsoid (see paper for a definition)
        """
        if self.eig_A[0] == -1:
            self.eig_A = np.linalg.eigh(self.A).eigenvalues
        if self.rho > 0:
            if self.axes[0] == -1:
                self.axes = np.sqrt(self.rho * self.eig_A)
        r_h = np.min(self.eig_A)/np.max(self.eig_A)
        self.eccentricity = np.sqrt(1 - r_h)

    def is_in_ellipsoid(self, z):
        """
        Checks if z is in the ellipsoid or not
        Returns either 0 or 1
        """
        res = 0
        if self.rho > 0:
            zh = z-self.z0
            dh = zh @ (self.A_inv @ zh)
            if dh < self.rho:
                res = 1
        return res

    def set_rho_v0(self):
        """
        Reajust the value of rho so that the volume is equal to v0, following
        the method described in the paper.
        """
        if self.v0 > 0:
            if self.detA == -1:  # it may have already been computed
                self.detA = np.linalg.det(self.A)
            eps2 = (self.v0/(self.vl * np.sqrt(self.detA))) ** (2/self.dim)
            self.rho = np.maximum(self.rho, eps2)


