# packages

import PyConfCov.regression as reg
import numpy as np
import matplotlib as mpl
mpl.use('Qt5Agg')
import matplotlib.pyplot as plt
import random
import copy
from scipy.special import gamma as gamma_fun
from sklearn.model_selection import train_test_split

random.seed(42)

# gaussian example
mean = 'zero'
cov = 'matern-3/2'
n_split = 1000  # number of examples for training the predictor
n_calib = 30000  # number of points within which to choose n_resample examples
n_full_calib = n_calib + n_split
n_resample = 2000
n_test = 2000
k = 5  # dimension of input
l = 3  # dimension of output
p = k + l  # full matrix dimension
v0 = 0  # 1e-5  # minimal volume
c_l = np.pi**(l/2) / gamma_fun(l/2 + 1)

low_alpha = np.ceil(n_resample*p/(n_resample-1))/(n_resample+1)
alpha = 0.1  # np.max(0.1, low_alpha)
n_alpha = int(np.ceil((n_resample+1)*(1-alpha))) - 1  # -1 because python indexing starts at 0
nugg_lin = 1e-8
nugg_cov = 1e-10

# cauchy_tanh is deprecated for now
do_cauchy = False
do_cauchy_tanh = False
do_gaussian = True
do_plot = False
do_lin_regressor = True

# generate data
if do_cauchy or do_cauchy_tanh:
    data_full_calib, data_test = reg.utils.gen_cauchy(n_full_calib, n_test, k, l, loc=mean, shape=cov)
    if do_cauchy_tanh:
        arr_ones = np.ones((100, 100, l)) - 1e-5
elif do_gaussian:
    data_full_calib, data_test, A_theoretic = reg.utils.gen_gaussian(n_full_calib, n_test, k, l, mean, cov, nugget=nugg_cov)

# train predictor : first, split full calib into split and calib
data_split, data_calib = train_test_split(data_full_calib, train_size=n_split, test_size=n_calib)

# split data in x, y format
x_split = data_split[:, 0:k]  # (n_split x k)
y_split = data_split[:, k:(k + l)]  # (n_split x l)

if do_lin_regressor:
    # train predictor on data split : y_hat = x_hat ^T beta_hat ( no affine version for now); beta_hat is a (k x l) matrix
    beta_hat = np.linalg.solve(x_split.transpose() @ x_split + nugg_lin * np.eye(k), x_split.transpose() @ y_split)
else:
    beta_hat = np.zeros((k, l))

# split test data in x, y format
x_test = data_test[:, 0:k]
y_test = data_test[:, k:(k + l)]

# do pre-computations
constants = reg.compute_constants(n_resample, k, l)
params = reg.StructType()
params.alpha = alpha
params.lambda_h = nugg_cov

# compute predictions and residuals on calibration data set
is_in_ellipsoid = np.zeros(n_test)
vol_ellipsoid = np.zeros(n_test)
is_in_sphere = np.zeros(n_test)
vol_sphere = np.zeros(n_test)

if do_plot:
    plt.ion()
    fig = plt.figure()
    ax = plt.subplot(111, projection='3d')
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('z')
    fig.canvas.draw()

for id_test in range(n_test):

    if id_test % 10 == 0:
        print(id_test)
        print('\n')
    # use random subsample of data_calib for conformal inference
    id_resample = random.sample(range(n_calib), n_resample)
    data_resample = data_calib[id_resample, :]
    # random.sample(data_calib, n_resample)
    x_calib = data_resample[:, 0:k]
    y_calib = data_resample[:, k:(k + l)]

    # compute residuals
    y_hat_calib = x_calib @ beta_hat
    res_calib = y_calib - y_hat_calib
    norm_calib = np.sort(np.linalg.norm(res_calib,axis=1))
    q_h = norm_calib[n_alpha]
    # run conformal inference on test data
    x_h = x_test[id_test, :]
    y_h = y_test[id_test, :]
    y_h_hat = x_h @ beta_hat  # predictor
    r_h = y_h - y_h_hat
    n_h = np.linalg.norm(r_h)
    X = np.vstack([x_calib, x_h])
    R = np.vstack([res_calib, np.zeros(l)])

    if do_cauchy_tanh:
        X = np.tanh(X)
        R = np.tanh(R)
        r_h = np.tanh(r_h)

    ell = reg.compute_ellipsoid(X, R, constants, params)

    # compare with theoretical matrix
    if do_gaussian:
        A_h = n_resample * ell.A
        norm_rel_A = np.linalg.norm(A_h - A_theoretic)/np.linalg.norm(A_h)

    # prevent empty regions
    ell.v0 = v0
    ell.set_R_v0()

    # compute metrics
    ell.compute_eccentricity()
    ell.compute_vol()

    # check if data point is inside ellipsoid or not
    is_in_ellipsoid[id_test] = ell.is_in_ellipsoid(r_h)
    if do_cauchy_tanh:
        vol_ellipsoid[id_test] = -1
    else:
        vol_ellipsoid[id_test] = ell.vol

    if n_h < q_h:
        is_in_sphere[id_test] = 1
    vol_sphere[id_test] = c_l * q_h ** l

    if do_plot:
        # plot ellipsoid and sphere
        # find the rotation matrix and radii of the axes
        z0 = ell.z0
        e, v = np.linalg.eig(ell.A/ell.R)
        s_ell = v @ np.diag(1.0/np.sqrt(e)) @ v.T

        u = np.linspace(0.0, 2.0 * np.pi, 100)
        v = np.linspace(0.0, np.pi, 100)
        # make sphere
        x = np.outer(np.cos(u), np.sin(v))
        y = np.outer(np.sin(u), np.sin(v))
        z = np.outer(np.ones_like(u), np.cos(v))
        sphere = np.stack((x, y, z), axis=-1)[..., None]  # unit sphere
        s_sph = q_h*np.eye(l)

        sph_plot = (s_sph @ sphere).squeeze(-1) + y_h_hat
        c_h = z0 + y_h_hat
        if do_gaussian or do_cauchy:
            ell_plot = (s_ell @ sphere).squeeze(-1) + c_h
        elif do_cauchy_tanh:
            ell_tanh = (s_ell @ sphere).squeeze(-1) + z0
            ell_tanh = np.minimum(ell_tanh, arr_ones)
            ell_tanh = np.maximum(ell_tanh, -arr_ones)
            ell_plot = np.arctanh(ell_tanh) + y_h_hat

        # plot
        ell_wire = ax.plot_wireframe(*ell_plot.transpose(2, 0, 1), rstride=4, cstride=4, color='b', alpha=0.2)
        ell_surf = ax.plot_surface(*ell_plot.transpose(2, 0, 1), rstride=4, cstride=4, color='b', alpha=0.2)
        sph_wire = ax.plot_wireframe(*sph_plot.transpose(2, 0, 1), rstride=4, cstride=4, color='g', alpha=0.2)
        sph_surf = ax.plot_surface(*sph_plot.transpose(2, 0, 1), rstride=4, cstride=4, color='g', alpha=0.2)
        ell_center = ax.scatter(c_h[0], c_h[1], c_h[2], color='r')
        pred = ax.scatter(y_h_hat[0], y_h_hat[1], y_h_hat[2], color='g')
        tru_y = ax.scatter(y_h[0], y_h[1], y_h[2], color='black')
        x_ax = ax.get_xlim()
        y_ax = ax.get_ylim()
        z_ax = ax.get_zlim()
        y_calib_plot = ax.scatter(y_calib[:, 0], y_calib[:, 1], y_calib[:, 2], color='red', marker='+', s=10, alpha=0.3)
        ax.axes.set_xlim(x_ax, auto=True)
        ax.axes.set_ylim(y_ax, auto=True)
        ax.axes.set_zlim(z_ax, auto=True)
        ax.set_aspect('equal')

        ecc_h = "{0:.3f}".format(ell.eccentricity)
        vol_ell = "{0:.3f}".format(ell.vol)
        vol_sph = "{0:.3f}".format(vol_sphere[id_test])
        if do_gaussian:
            str_title = 'Gaussian data, eccentricity = ' + ecc_h + ', vol ell = ' + vol_ell + ', vol sph = ' + vol_sph
        elif do_cauchy or do_cauchy_tanh:
            str_title = 'Cauchy data, eccentricity = ' + ecc_h + ', vol ell = ' + vol_ell + ', vol sph = ' + vol_sph
        ax.set_title(str_title)

        plt.pause(0.01)
        fig.canvas.draw_idle()
        fig.canvas.flush_events()
        ell_wire.remove()
        ell_surf.remove()
        sph_wire.remove()
        sph_surf.remove()
        tru_y.remove()
        pred.remove()
        y_calib_plot.remove()
        ell_center.remove()


frac_in_ellipsoid = is_in_ellipsoid.sum()/n_test
frac_in_sphere = is_in_sphere.sum()/n_test
mean_vol_ellipsoid = vol_ellipsoid.sum()/n_test
mean_vol_sphere = vol_sphere.sum()/n_test

#if __name__ == '__test__':
#    test()


