# import scripts
import utils, ellipsoid
# import packages and functions
import numpy as np
import matplotlib as mpl
mpl.use('Qt5Agg')  # the package PyQt5 has to be installed
import matplotlib.pyplot as plt
import random
from scipy.special import gamma as gamma_fun
from scipy.special import hyp1f1
from scipy.stats import chi2
from sklearn.model_selection import train_test_split
from datetime import datetime


##############################################################################
#
#      Data parameters, conformal inference parameters
#
##############################################################################

# path to folder for saved data
pwd_save_data = '/home/disc/i.henderson/Documents/GitHub/PyConfCov/data/'

do_cauchy = False
do_gaussian = not do_cauchy
if do_gaussian:
    data_type = 'Gaussian'
elif do_cauchy:
    data_type = 'Cauchy'

mean = 'zero'
cov = 'matern-3/2'
ell_matern = 5  # lengthscale of covariance function
sig2_matern = 1  # variance of covariance function
k = 6  # dimension of input
l = 3  # dimension of output
p = k + l  # full matrix dimension

# additional simulation parameters for conformal inference
v0 = 0  # minimal user set volume of ellipsoid to avoid empty regions
v_l = np.pi**(l/2) / gamma_fun(l/2 + 1)  # volume of unit ball
alpha = 0.1
nugg_lin = 0  # nugget for linear predictor
nugg_cov = 0  # nugget for score function S(z)
do_lin_regressor = True  # only option available for now; alternative is null predictor

##############################################################################
#
#      Booleans to select which numerical experiment to run, and what to 
#      do/save for each experiment
#
##############################################################################

# choose which part of the code to run
do_histo = False  # run code for generating histograms
do_optimal_k = False  # run code for the k_opt table
do_validate_1F1_formulas = True  # run code for the 1F1 formulas

# Booleans for histogram plots
do_save_histo = True  # save histogram data and figures
do_compute_histo = True  # generate data and results for histograms; else, some
                         # precomputed data is loaded; careful with attached 
                         # values and parameters!
do_save_data_histo = True  # save generated data for histograms
do_live_plot = False  # plot ellipsoids during the generation of data and results for histograms

# Booleans for the k optimal table
do_save_k_opt = True  # save k_opt data
do_compute_k_opt = True  # compute k_opt data

# Boolean the empirical validation of the Gaussian expectations
do_compute_1F1_formulas = True  # generate data and results for histograms
do_save_data_1F1_formulas = True  # save generated data for histograms


##############################################################################
#
#      Code for generating histograms
#
##############################################################################
if do_histo:

    random.seed(42)
    np.random.seed(42)

    if do_compute_histo:
        # Gaussian data parameters for paper experiment : 
        # n_split = 5 000, 
        # n_calib = 200, 
        # ntest = 500, 
        # n_data_calib = 100 000, 
        # n_histo = 10 000
        
        # Cauchy data parameters for paper experiment : 
        # n_split = 3 000 000, 
        # n_data_calib = 800 000, 
        # n_calib = 10 000, 
        # n_test = 800, 
        # n_histo = 1000

        # simulation parameters
        n_split = 5000  # number of examples for training the predictor
        n_calib = 200  # corresponds to n in the paper
        n_test = 500  # number of CI procedures run for this numerical experiment
        n_data_calib = n_test*n_calib  # number of points within which to choose n_calib examples; optimally,
                               # n_data_calib >= n_calib * n_test, else there will be some redundancy
        n_full_calib = n_data_calib + n_split                       
        n_per_loop = n_calib * n_test
        n_histo = 10000 

        low_alpha = np.ceil(n_calib * p / (n_calib - 1)) / (n_calib + 1)  # minimal alpha for finite ellipsoids
        alpha = np.max([alpha, low_alpha])
        n_alpha = int(np.ceil((n_calib + 1) * (1 - alpha))) - 1  # -1 because python indexing starts at 0

        alpha_hat_ell_loop = np.zeros(n_histo)
        alpha_hat_sph_loop = np.zeros(n_histo)
        vol_ell_loop = np.zeros(n_histo)
        vol_sph_loop = np.zeros(n_histo)
        q_n_alpha_loop = np.zeros(n_histo)

        if do_save_histo:  # save parameters of simulation
            t_h = datetime.now()
            t_h_string = t_h.strftime("%m-%d-%Y-%H-%M-%S")
            fname_simu_data = pwd_save_data + t_h_string + '-simu-data.txt'
            with open(fname_simu_data, 'w') as f:
                f.write('Simulation type : histograms \n\n')
                f.write('Simulated data: \n')
                if do_gaussian:
                    f.write('Gaussian data \n')
                else:
                    f.write('Cauchy data \n')
                f.write('mean = ' + mean + '\n')
                f.write('cov = ' + cov + '\n')
                f.write('L = ' + str(ell_matern) + '\n')
                f.write('sig2 = ' + str(sig2_matern) + '\n')
                f.write('k = ' + str(k) + '\n')
                f.write('l = ' + str(l) + '\n')
                f.write('Sampling parameters: \n')
                f.write('n_split = ' + str(n_split) + '\n')
                f.write('n_data_calib = ' + str(n_data_calib) + '\n')
                f.write('n_calib = ' + str(n_calib) + '\n')
                f.write('n_test = ' + str(n_test) + '\n')
                f.write('n_histo = ' + str(n_histo) + '\n')
                f.close()

        # train predictor once and for all
        # generate split data
        if do_cauchy:
            data_split, _, A_infty, _ = utils.gen_cauchy(n_split, 0, 
                                                         k, l, mean, cov, ell=ell_matern, 
                                                         sig2=sig2_matern, nugget=nugg_cov)
        elif do_gaussian:
            data_split, _, A_infty = utils.gen_gaussian(n_split, 0, 
                                                        k, l, mean, cov, ell=ell_matern, 
                                                        sig2=sig2_matern, nugget=nugg_cov)

        # split data in x, y format
        x_split = data_split[:, 0:k]  # (n_split x k)
        y_split = data_split[:, k:(k + l)]  # (n_split x l)

        if do_lin_regressor:
            beta_hat = np.linalg.solve(x_split.transpose() @ x_split + nugg_lin * np.eye(k),
                                       x_split.transpose() @ y_split)
        else:  # do null predictor
            beta_hat = np.zeros((k, l))

        for id_loop in range(n_histo):

            # generate all the data at once
            if do_cauchy:
                data_full_calib, data_test, _, _ = utils.gen_cauchy(n_full_calib, 
                                                                    n_test, k, l, mean, cov, 
                                                                    ell=ell_matern, sig2=sig2_matern, nugget=nugg_cov)
            elif do_gaussian:
                data_full_calib, data_test, _ = utils.gen_gaussian(n_full_calib, 
                                                                   n_test, k, l, mean, cov,
                                                                   ell=ell_matern, sig2=sig2_matern, nugget=nugg_cov)

            # split full calib into split and calib
            data_split, data_calib = train_test_split(data_full_calib, train_size=n_split, test_size=n_data_calib)

            # split data in x, y format
            x_split = data_split[:, 0:k]  # (n_split x k)
            y_split = data_split[:, k:(k + l)]  # (n_split x l)

            # split test data in x, y format
            x_test = data_test[:, 0:k]
            y_test = data_test[:, k:(k + l)]

            # do pre-computations
            constants = ellipsoid.compute_constants(n_calib, k, l)
            params = utils.StructType()
            params.alpha = alpha
            params.lambda_h = nugg_cov

            # compute predictions and residuals on calibration data set
            is_in_ellipsoid = np.zeros(n_test)
            vol_ellipsoid = np.zeros(n_test)
            is_in_sphere = np.zeros(n_test)
            vol_sphere = np.zeros(n_test)

            if do_live_plot:
                plt.ion()
                fig = plt.figure()
                ax = plt.subplot(111, projection='3d')
                ax.set_xlabel('x')
                ax.set_ylabel('y')
                ax.set_zlabel('z')
                fig.canvas.draw()

            for id_test in range(n_test):

                if id_test % 200 == 0:
                    print("id_loop = ", id_loop+1, "out of", n_histo, ", id_test = ", id_test, "out of ", n_test, "\n")

                if n_per_loop <= n_data_calib:
                    id_resample = range(id_test*n_calib, (id_test+1)*n_calib)
                    data_resample = data_calib[id_resample, :]
                else:
                    # use random subsample of data_calib
                    id_resample = random.sample(range(n_data_calib), n_calib)
                    data_resample = data_calib[id_resample, :]

                x_calib = data_resample[:, 0:k]
                y_calib = data_resample[:, k:(k + l)]

                # compute residuals
                y_hat_calib = x_calib @ beta_hat
                res_calib = y_calib - y_hat_calib
                norm_calib = np.sort(np.linalg.norm(res_calib, axis=1))
                q_h = norm_calib[n_alpha]
                # run conformal inference on test data
                x_h = x_test[id_test, :]
                y_h = y_test[id_test, :]
                y_h_hat = x_h @ beta_hat  # predictor
                r_h = y_h - y_h_hat
                n_h = np.linalg.norm(r_h)
                X = np.vstack([x_calib, x_h])
                R = np.vstack([res_calib, np.zeros(l)])

                ell = ellipsoid.compute_ellipsoid(X, R, constants, params)

                # prevent empty regions if requested
                if v0 > 0:
                    ell.v0 = v0
                    ell.set_rho_v0()

                # compute metrics
                ell.compute_eccentricity()
                ell.compute_vol()

                # check if data point is inside ellipsoid or not, compute volume
                is_in_ellipsoid[id_test] = ell.is_in_ellipsoid(r_h)
                vol_ellipsoid[id_test] = ell.vol

                # check if data point is in sphere, compute volume
                if n_h < q_h:
                    is_in_sphere[id_test] = 1
                vol_sphere[id_test] = v_l * q_h ** l

                if do_live_plot:
                    # explanation at https://stackoverflow.com/questions/7819498/plotting-ellipsoid-with-matplotlib
                    # or at https://matplotlib.org/stable/gallery/mplot3d/surface3d.html
                    # plot ellipsoid and sphere
                    # find the rotation matrix and radii of the axes
                    z0 = ell.z0
                    e, v = np.linalg.eig(ell.rho * ell.A)  # np.linalg.eig(ell.A_inv/ell.rho)
                    s_ell = v @ np.diag(np.sqrt(e)) @ v.T

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
                    ell_plot = (s_ell @ sphere).squeeze(-1) + c_h

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
                    n_scatter_max = 0
                    if n_calib >= n_scatter_max:
                        id_plot_calib = random.sample(range(n_calib), n_scatter_max)
                        y_calib_visible = y_calib[id_plot_calib, :]
                    else:
                        y_calib_visible = y_calib
                    y_calib_plot = ax.scatter(y_calib_visible[:, 0], y_calib_visible[:, 1], y_calib_visible[:, 2],
                                              color='red', marker='+', s=10, alpha=0.3)
                    ax.axes.set_xlim(x_ax, auto=True)
                    ax.axes.set_ylim(y_ax, auto=True)
                    ax.axes.set_zlim(z_ax, auto=True)
                    ax.set_aspect('equal')

                    ecc_h = "{0:.3f}".format(ell.eccentricity)
                    vol_ell = "{0:.3f}".format(ell.vol)
                    vol_sph = "{0:.3f}".format(vol_sphere[id_test])
                    if do_gaussian:
                        str_title = 'Gaussian data, eccentricity = ' + ecc_h + ', vol ell = ' + vol_ell + ', vol sph = ' + vol_sph
                    elif do_cauchy:
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

            alpha_hat_ell_loop[id_loop] = is_in_ellipsoid.sum()/n_test
            alpha_hat_sph_loop[id_loop] = is_in_sphere.sum()/n_test
            vol_ell_loop[id_loop] = vol_ellipsoid.sum()/n_test  # mean
            vol_sph_loop[id_loop] = vol_sphere.sum()/n_test

        if do_save_data_histo:
            fname_vol_ell = pwd_save_data + t_h_string + '-histo-ell_vol.csv'
            fname_alpha_ell = pwd_save_data + t_h_string + '-histo-ell_alpha.csv'
            fname_vol_sph = pwd_save_data + t_h_string + '-histo-sph_vol.csv'
            fname_alpha_sph = pwd_save_data + t_h_string + '-histo-sph_alpha.csv'
            np.savetxt(fname_vol_ell, vol_ell_loop, delimiter=',')
            np.savetxt(fname_alpha_ell, alpha_hat_ell_loop, delimiter=',')
            np.savetxt(fname_vol_sph, vol_sph_loop, delimiter=',')
            np.savetxt(fname_alpha_sph, alpha_hat_sph_loop, delimiter=',')

    else:  # load precomputed data
        t_h_string = '' # enter time-stamp id of previous simulation, eg '06-13-2024-11-28-05'
        fname_simu_data = pwd_save_data + t_h_string + '-simu-data.txt'

        fname_load_ell_vol = pwd_save_data + t_h_string + '-histo-ell_vol.csv'
        fname_load_ell_alpha = pwd_save_data + t_h_string + '-histo-ell_alpha.csv'
        fname_load_sph_vol = pwd_save_data + t_h_string + '-histo-sph_vol.csv'
        fname_load_sph_alpha = pwd_save_data + t_h_string + '-histo-sph_alpha.csv'

        vol_ell_loop = np.genfromtxt(fname_load_ell_vol, delimiter=',')
        alpha_hat_ell_loop = np.genfromtxt(fname_load_ell_alpha, delimiter=',')
        vol_sph_loop = np.genfromtxt(fname_load_sph_vol, delimiter=',')
        alpha_hat_sph_loop = np.genfromtxt(fname_load_sph_alpha, delimiter=',')

        # In theory, the following could be uploaded from data txt file...
        # but one has to import them by hand for now...
        n_split = 5000  # number of examples for training the predictor
        n_data_calib = 800000  # number of points within which to choose n_calib examples; optimally,
                               # n_data_calib >= n_calib * n_test, else there will be some redundancy
        n_full_calib = n_data_calib + n_split
        n_calib = 1000
        n_test = 800
        n_per_loop = n_calib * n_test
        n_histo = 100

    # First set the figure params for histograms
    plt.rcParams.update({"text.usetex": True, "font.family": "Computer Modern Typewriter"})
    sz_tk = 30
    sz_lgd = 40
    sz_title = 30
    lw = 1.5
    plt.rc(group='xtick', labelsize=sz_tk)
    plt.rc(group='ytick', labelsize=sz_tk)
    plt.rc(group='legend', fontsize=sz_lgd)
    plt.rc(group='axes', labelsize=sz_tk)

    # The bins have to be chosen carefully because the alpha hat are averages 
    # of Bernoulli rvs, with discrete range
    # Below, choose number of bins for the histograms
    if do_gaussian:
        n_bins_vol = 100
    elif do_cauchy:  
        n_bins_vol = 50  
    n_bins_alpha = n_bins_vol
    dh = 1/n_test
    h_min = 0.85
    h_min = dh * int(h_min/dh)
    n_bins_alpha_max = int((1-h_min)*n_test)
    n_bins_alpha = np.min([n_bins_alpha, n_bins_alpha_max])
    sz_bins = (1-h_min)/n_bins_alpha
    q_bins = int(sz_bins/dh)  # quotient
    bins_alpha = []
    for i in range(n_bins_alpha):
        h_h = h_min + i*q_bins*dh
        if h_h < 1:
            bins_alpha.append(h_h)

    # histograms for alpha
    title_ell_alpha = data_type + ", ellipsoid: empirical $1-\\alpha$"
    title_ell_vol = data_type + ", ellipsoid: empirical volume"
    title_sph_alpha = data_type + ", sphere: empirical $1-\\alpha$"
    title_sph_vol = data_type + ", sphere: empirical volume"

    alpha_hat_ell_mean = alpha_hat_ell_loop.mean()
    fig1, ax1 = plt.subplots(nrows=2, ncols=2)
    ax1[0, 0].hist(alpha_hat_ell_loop, bins=bins_alpha, density=True)
    ax1[0, 0].axvline(alpha_hat_ell_loop.mean(), color='b', linewidth=lw)
    ax1[0, 0].axvline(1-alpha, color='r', linewidth=lw)
    ax1[0, 0].set_title(title_ell_alpha, fontsize=sz_title)

    alpha_hat_sph_mean = alpha_hat_sph_loop.mean()
    ax1[0, 1].hist(alpha_hat_sph_loop, bins=bins_alpha, density=True)
    ax1[0, 1].axvline(alpha_hat_sph_loop.mean(), color='b', linewidth=lw)
    ax1[0, 1].axvline(1-alpha, color='r', linewidth=lw)
    ax1[0, 1].set_title(title_sph_alpha, fontsize=sz_title)

    # histograms for the volume
    vol_ell_mean = vol_ell_loop.mean()
    ax1[1, 0].hist(vol_ell_loop, bins=n_bins_vol, density=True)
    ax1[1, 0].axvline(vol_ell_loop.mean(), color='b', linewidth=lw)
    ax1[1, 0].set_title(title_ell_vol, fontsize=sz_title)

    vol_sph_mean = vol_sph_loop.mean()
    vol_sph_sort = np.sort(vol_sph_loop)
    ax1[1, 1].hist(vol_sph_loop, bins=n_bins_vol, density=True) #, range=[0, 10000]
    ax1[1, 1].axvline(vol_sph_loop.mean(), color='b', linewidth=lw)
    ax1[1, 1].set_title(title_sph_vol, fontsize=sz_title)

    figManager = plt.get_current_fig_manager()
    figManager.window.showMaximized()

    if do_save_histo:
        plt.pause(2)
        fig1.tight_layout()
        fname_histo = pwd_save_data + t_h_string + '-histo-alpha-vol.eps'
        plt.pause(2)
        plt.savefig(fname_histo, format='eps', dpi=1200)

##############################################################################
#
#      Code for validating the expectation formulas in the Gaussian case
#      (only valid when do_gaussian = True)
#
##############################################################################
if do_validate_1F1_formulas:

    random.seed(42)
    np.random.seed(42)

    n_split = 5000  # number of examples for training the predictor
    n_data_calib = 2000000  # number of points within which to choose n_calib examples ; optimally,
                            # n_data_calib >= n_calib * n_test, else there will be some redundancy
    n_full_calib = n_data_calib + n_split
    n_calib = 50000
    n_test = 40
    n_per_loop = n_calib * n_test
    n_histo = 1000

    n_q = 3  # number of moments to compute

    if do_save_data_1F1_formulas:  # save parameters of simulation
        t_h = datetime.now()
        t_h_string = t_h.strftime("%m-%d-%Y-%H-%M-%S")
        fname_simu_data = pwd_save_data + t_h_string + '-simu-data.txt'
        with open(fname_simu_data, 'w') as f:
            f.write('Simulation type : 1F1 formulas \n\n')
            f.write('Simulated data: \n')
            if do_gaussian:
                f.write('Gaussian data \n')
            else:
                f.write('Cauchy data \n')
            f.write('mean = ' + mean + '\n')
            f.write('cov = ' + cov + '\n')
            f.write('L = ' + str(ell_matern) + '\n')
            f.write('sig2_matern = ' + str(sig2_matern) + '\n')
            f.write('k = ' + str(k) + '\n')
            f.write('l = ' + str(l) + '\n\n')
            f.write('Sampling parameters: \n')
            f.write('n_split = ' + str(n_split) + '\n')
            f.write('n_data_calib = ' + str(n_data_calib) + '\n')
            f.write('n_calib = ' + str(n_calib) + '\n')
            f.write('n_test = ' + str(n_test) + '\n')
            f.write('n_histo = ' + str(n_histo) + '\n')
            f.write('n_q = ' + str(n_q) + '\n')
            f.close()

    if do_compute_1F1_formulas:
        vol_ell_loop = np.zeros((n_q, n_histo))  # each row corresponds to a value of q

        # train predictor once and for all
        # generate split data
        if do_cauchy:
            data_split, _, A_infty, _ = utils.gen_cauchy(n_split, 0, k, l, mean, 
                                                         cov, ell=ell_matern, 
                                                         sig2=sig2_matern, nugget=nugg_cov)
        elif do_gaussian:
            data_split, _, A_infty = utils.gen_gaussian(n_split, 0, k, l, mean, 
                                                        cov, ell=ell_matern, 
                                                        sig2=sig2_matern, nugget=nugg_cov)

        # split data in x, y format
        x_split = data_split[:, 0:k]  # (n_split x k)
        y_split = data_split[:, k:(k + l)]  # (n_split x l)

        if do_lin_regressor:
            beta_hat = np.linalg.solve(x_split.transpose() @ x_split + nugg_lin * np.eye(k),
                                       x_split.transpose() @ y_split)
        else: 
            beta_hat = np.zeros((k, l))

        # first generate data and compute empirical expected volumes
        for id_loop in range(n_histo):

            # generate all the data at once
            if do_cauchy:
                data_full_calib, data_test, _, _ = utils.gen_cauchy(n_full_calib, 
                                                                    n_test, k, l, mean,
                                                                    cov, ell=ell_matern, sig2=sig2_matern, nugget=nugg_cov)
            elif do_gaussian:
                data_full_calib, data_test, _ = utils.gen_gaussian(n_full_calib, 
                                                                   n_test, k, l, mean, 
                                                                   cov, ell=ell_matern, sig2=sig2_matern, nugget=nugg_cov)

            # split full calib into split and calib
            _, data_calib = train_test_split(data_full_calib, train_size=n_split, test_size=n_data_calib)

            # split test data in x, y format
            x_test = data_test[:, 0:k]
            y_test = data_test[:, k:(k + l)]

            # do pre-computations
            constants = ellipsoid.compute_constants(n_calib, k, l)
            params = utils.StructType()
            params.alpha = alpha
            params.lambda_h = nugg_cov

            # compute predictions and residuals on calibration data set
            vol_ellipsoid = np.zeros(n_test)
            is_in_ellipsoid = np.zeros(n_test)

            q_n_alpha_test = np.zeros(n_test)  # if one wants to check the value of the empirical quantile; untouched for now
            s0_test = np.zeros(n_test)  # supposed to follow a Hotelling distribution for Gaussian data; untouched for now

            for id_test in range(n_test):

                if id_test % 20 == 0:
                    print("id_loop = ", id_loop + 1, "out of", n_histo, ", id_test = ", id_test, "out of ", n_test,
                          "\n")

                if n_per_loop <= n_data_calib:
                    id_resample = range(id_test * n_calib, (id_test + 1) * n_calib)
                    data_resample = data_calib[id_resample, :]
                else:
                    # use random subsample of data_calib
                    id_resample = random.sample(range(n_data_calib), n_calib)
                    data_resample = data_calib[id_resample, :]

                x_calib = data_resample[:, 0:k]
                y_calib = data_resample[:, k:(k + l)]

                # compute residuals
                y_hat_calib = x_calib @ beta_hat
                res_calib = y_calib - y_hat_calib
                # run conformal inference on test data
                x_h = x_test[id_test, :]
                y_h = y_test[id_test, :]
                y_h_hat = x_h @ beta_hat  # predictor
                r_h = y_h - y_h_hat
                X = np.vstack([x_calib, x_h])
                R = np.vstack([res_calib, np.zeros(l)])

                ell = ellipsoid.compute_ellipsoid(X, R, constants, params)

                # prevent empty regions if requested
                if v0 > 0:
                    ell.v0 = v0
                    ell.set_rho_v0()

                # compute metrics
                ell.compute_eccentricity()
                ell.compute_vol()

                # compute volume
                vol_ellipsoid[id_test] = ell.vol

            vol_ell_loop[0, id_loop] = vol_ellipsoid.sum() / n_test  # mean
            vol_ell_loop[1, id_loop] = sum(i ** 2 for i in vol_ellipsoid) / n_test  # mean of square
            vol_ell_loop[2, id_loop] = sum(i ** 3 for i in vol_ellipsoid) / n_test  # mean of cube

    else:  # load precomputed data
        t_h_string = '06-07-2024-22-31-44'
        fname_simu_data = pwd_save_data + t_h_string + '-simu-data.txt'
        fname_expectations = pwd_save_data + t_h_string + '-gaussian-expectation-ell_vol.csv'
        vol_ell_loop = np.genfromtxt(fname_expectations, delimiter=',')
        # get A_infty
        data_full_calib, data_test, A_infty = utils.gen_gaussian(n_full_calib, n_test, k, l, mean, cov, ell=ell_matern, sig2=sig2_matern, nugget=nugg_cov)

    # compute theoretical values in the Gaussian case
    e_theo = np.zeros(n_q)  # theoretical expectations
    err_rel = np.zeros(n_q)  # relative error
    q_infty = chi2.ppf(1-alpha, df=k+l)  # theoretical quantile
    det_schur = np.linalg.det(A_infty)  # determinant of schur complement
    empirical_means = np.zeros(n_q)
    for i_q in range(n_q):
        # thoerical expectations
        q_h = i_q+1
        hyp1F1_val = hyp1f1(k/2, (k+q_h*l)/2+1, -q_infty/2)  # value of 1F1
        const = np.pi ** (q_h*l/2) * gamma_fun(1+q_h*l/2) / (2 ** (k/2) * gamma_fun(1 + (k+q_h*l)/2) * gamma_fun(1+l/2) ** q_h)
        e_theo[i_q] = const * det_schur ** (q_h/2) * q_infty ** ((k+q_h*l)/2) * hyp1F1_val

        # compare with empirical value
        empirical_means[i_q] = vol_ell_loop[i_q, 0:n_histo].mean()
        err_rel[i_q] = np.abs((e_theo[i_q]-empirical_means[i_q]))/e_theo[i_q]

    if do_save_data_1F1_formulas:
        if not ('t_h' in locals()):
            t_h = datetime.now()
            t_h_string = t_h.strftime("%m-%d-%Y-%H-%M-%S")
        fname_vol_ell_exp_emp = pwd_save_data + t_h_string + '-gaussian-expectation-ell_vol-empirical.csv'
        fname_vol_ell_exp_theo = pwd_save_data + t_h_string + '-gaussian-expectation-ell_vol-theoretical.csv'
        fname_vol_ell_exp_relative = pwd_save_data + t_h_string + '-gaussian-expectation-ell_vol-error-rel.csv'
        np.savetxt(fname_vol_ell_exp_emp, vol_ell_loop, delimiter=',')
        np.savetxt(fname_vol_ell_exp_relative, err_rel, delimiter=',')

    # histogram of volume with theoretical volume
    fig2, ax2 = plt.subplots(nrows=1, ncols=1)
    ax2.hist(vol_ell_loop[0, 0:n_histo], bins='auto')
    ax2.axvline(e_theo[0], color='r')
    ax2.axvline(vol_ell_loop[0, 0:n_histo].mean(), color='b')

##############################################################################
#
#      Code for comparing the theoretical and pratical optimal value of k
#      for an AR(q) Gaussian process
#
##############################################################################
if do_optimal_k:

    random.seed(42)
    np.random.seed(42)

    # simulation parameters
    n_split = 5000  # number of examples for training the predictor
    n_data_calib = 2000000  # number of points within which to choose n_calib examples; optimally,
                            # n_data_calib >= n_calib * n_test, else there will be some redundancy
    n_full_calib = n_data_calib + n_split
    n_calib = 200
    n_test = 10000
    n_tot = n_calib * n_test
    n_alpha = int(np.ceil((n_calib + 1) * (1 - alpha))) - 1  # -1 because python indexing starts at 0

    k_to_test = 6
    list_k = range(k_to_test)
    list_cov = ['matern-1/2', 'matern-3/2', 'matern-5/2', 'matern-7/2']
    n_cov = len(list_cov)

    if do_save_k_opt:  # save parameters of simulation
        t_h = datetime.now()
        t_h_string = t_h.strftime("%m-%d-%Y-%H-%M-%S")
        fname_simu_data = pwd_save_data + t_h_string + '-simu-data.txt'
        with open(fname_simu_data, 'w') as f:
            f.write('Simulation type : optimal k \n\n')
            f.write('Simulated data: \n')
            if do_gaussian:
                f.write('Gaussian data \n')
            else:
                f.write('Cauchy data \n')
            f.write('mean = ' + mean + '\n')
            f.write('cov = ' + cov + '\n')
            f.write('L = ' + str(ell_matern) + '\n')
            f.write('sig2_matern = ' + str(sig2_matern) + '\n')
            f.write('k = ' + str(k) + '\n')
            f.write('l = ' + str(l) + '\n')
            f.write('Sampling parameters: \n')
            f.write('n_split = ' + str(n_split) + '\n')
            f.write('n_data_calib = ' + str(n_data_calib) + '\n')
            f.write('n_calib = ' + str(n_calib) + '\n')
            f.write('n_test = ' + str(n_test) + '\n')
            f.write('n_histo = 1 \n')
            f.write('number of k tested = ' + str(k_to_test) + '\n')
            f.close()

    if do_compute_k_opt:
        # now we do the optimal k section : we test AR(1), AR(2), AR(3) and AR(4)
        vol_mean = np.zeros((n_cov, k_to_test+1))
        for id_cov in range(n_cov):
            cov = list_cov[id_cov]
            for id_k in range(k_to_test+1):
                k_h = id_k
                # generate data
                if do_cauchy:
                    data_full_calib, data_test, A_infty, _ = utils.gen_cauchy(n_full_calib, n_test, k, l, mean, cov,
                                                                               ell=ell_matern, sig2=sig2_matern, nugget=nugg_cov)
                elif do_gaussian:
                    data_full_calib, data_test, A_infty = utils.gen_gaussian(n_full_calib, n_test, k, l, mean, cov,
                                                                                 ell=ell_matern, sig2=sig2_matern, nugget=nugg_cov)

                # train predictor : first, split full calib into split and calib
                data_split, data_calib = train_test_split(data_full_calib, train_size=n_split, test_size=n_data_calib)

                # split data in x, y format
                x_split = data_split[:, (k-k_h):k]  # (n_split x k_h)
                y_split = data_split[:, k:(k + l)]  # (n_split x l)

                if do_lin_regressor:
                    beta_hat = np.linalg.solve(x_split.transpose() @ x_split 
                                               + nugg_lin * np.eye(k_h), x_split.transpose() @ y_split)
                else:
                    beta_hat = np.zeros((k_h, l))

                # split test data in x, y format
                x_test = data_test[:, (k-k_h):k]
                y_test = data_test[:, k:(k + l)]

                # do pre-computations
                constants = ellipsoid.compute_constants(n_calib, k_h, l)
                params = utils.StructType()
                params.alpha = alpha
                params.lambda_h = nugg_cov

                # compute predictions and residuals on calibration data set
                is_in_ellipsoid = np.zeros(n_test)
                vol_ellipsoid = np.zeros(n_test)
                is_in_sphere = np.zeros(n_test)
                vol_sphere = np.zeros(n_test)

                if do_live_plot:
                    plt.ion()
                    fig = plt.figure()
                    ax = plt.subplot(111, projection='3d')
                    ax.set_xlabel('x')
                    ax.set_ylabel('y')
                    ax.set_zlabel('z')
                    fig.canvas.draw()

                for id_test in range(n_test):

                    if id_test % 200 == 0:
                        print("id_cov = ", id_cov+1, "out of ", n_cov, "| k = ", 
                              id_k, "out of ", k, "| id_test = ", id_test, "out of ", n_test, "\n")

                    if n_tot <= n_data_calib:
                        id_resample = range(id_test * n_calib, (id_test + 1) * n_calib)
                        data_resample = data_calib[id_resample, :]
                    else:
                        # use random subsample of data_calib
                        id_resample = random.sample(range(n_data_calib), n_calib)
                        data_resample = data_calib[id_resample, :]
                    x_calib = data_resample[:, (k-k_h):k]
                    y_calib = data_resample[:, k:(k + l)]

                    # compute residuals
                    y_hat_calib = x_calib @ beta_hat
                    res_calib = y_calib - y_hat_calib
                    norm_calib = np.sort(np.linalg.norm(res_calib, axis=1))
                    q_h = norm_calib[n_alpha]
                    # run conformal inference on test data
                    x_h = x_test[id_test, :]
                    y_h = y_test[id_test, :]
                    y_h_hat = x_h @ beta_hat  # predictor
                    r_h = y_h - y_h_hat
                    n_h = np.linalg.norm(r_h)
                    X = np.vstack([x_calib, x_h])
                    R = np.vstack([res_calib, np.zeros(l)])

                    ell = ellipsoid.compute_ellipsoid(X, R, constants, params)

                    # prevent empty regions if requested
                    ell.v0 = v0
                    ell.set_rho_v0()

                    # compute metrics
                    ell.compute_eccentricity()
                    ell.compute_vol()

                    # check if data point is inside ellipsoid or not
                    is_in_ellipsoid[id_test] = ell.is_in_ellipsoid(r_h)
                    vol_ellipsoid[id_test] = ell.vol

                    if n_h < q_h:
                        is_in_sphere[id_test] = 1
                    vol_sphere[id_test] = v_l * q_h ** l

                vol_mean[id_cov, id_k] = vol_ellipsoid.sum()/n_test

    # the following generates a figures from which one can visualize the 
    # numerical optimal value of k (is not present in the paper)
    ratios = np.zeros((n_cov, k))
    for i in range(n_cov):
        for j in range(k):
            ratios[i, j] = vol_mean[i, j+1]/vol_mean[i, j]

    sz_tk = 40
    sz_lgd = 40
    lw = 4
    plt.rcParams.update({
        "text.usetex": True,
        "font.family": "Computer Modern Typewriter"
    })
    plt.rc('xtick', labelsize=sz_tk)
    plt.rc('ytick', labelsize=sz_tk)
    plt.rc('legend', fontsize=sz_lgd)
    plt.rc('axes', labelsize=sz_tk)
    fig_ratios, ax_ratios = plt.subplots(nrows=1, ncols=1)
    ax_ratios.plot(range(1, k + 1), np.transpose(ratios), linewidth=lw)
    plt.legend(['AR(1)', 'AR(2)', 'AR(3)', 'AR(4)'], loc="lower right")
    #plt.xlabel("Value of k")
    #plt.ylabel("Ratio")
    for i in range(n_cov):
        ax_ratios.scatter(range(1, k+1), ratios[i, 0:k], s=200)  
        #  linestyle=':','-', '--', '-.', ':', 'None', ' ', '', 'solid', 'dashed', 'dashdot', 'dotted'
    ax_ratios.axhline(y=1, color='k', linestyle='-', linewidth=lw)
    #fig_ratios.tight_layout()

    plt.show()

    if do_save_k_opt:
        if not ('t_h' in locals()):
            pwd_save_data = '/home/ihenders/PycharmProjects/PyConfCovV3/data/'
            t_h = datetime.now()
            t_h_string = t_h.strftime("%m-%d-%Y-%H-%M-%S")

        fname_k_opt_vol_mean = pwd_save_data + t_h_string + '-k_opt_vol_mean.csv'
        fname_k_opt_vol_ratios = pwd_save_data + t_h_string + '-k_opt_vol_ratios.csv'

        np.savetxt(fname_k_opt_vol_mean, vol_mean, delimiter=',')
        np.savetxt(fname_k_opt_vol_ratios, ratios, delimiter=',')



