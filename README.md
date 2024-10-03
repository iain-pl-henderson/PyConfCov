These files can be used to reproduce the numerical experiments described in the preprint "Adaptive inference with random ellipsoids through Conformal Conditional Linear Expectation" (2024) available at https://arxiv.org/abs/2409.18508 .
This paper describes a conformity score to be used in conformal prediction for multivariate regression, that yields adaptive ellipsoidal confidence regions. Their volume is asymptotically smaller that that of the standard norm residual score, for elliptically contoured distributions.

The experiments in the paper correspond to
1) EXP1 : an assessment of the performances (coverage and volume) of our conformity score on Gaussian and Cauchy data, resulting in the two histograms presented in the paper.
2) EXP2 : a validation of the expectation formulas corresponding to the volume of our region for Gaussian data, (Table 1 in the paper).
3) EXP3 : an experiment on AR(q) Gaussian processes, whose aim is to identify the optimal number of previous time stamps to use in our conformity score (Table 2 in the paper).$

To reproduce the experiments, run the script test.py. 
Be wary that the parameters corresponding to the experiments described in the paper are time-consuming, so different values (smaller n_calib, n_test, n_histo) may be more appropriate for a first time use.

This test.py script is organised as follow.
1) A first section where the parameters of the numerical experiments are set.
2) A second section with booleans to declare which numerical experiment to run, and what to do/save for each experiment.
3) A third section with the code corresponding to EXP1 (default values correspond to Gaussian data; values corresponding to Cauchy are commented in this section). In the case of three dimensional Y_i, a 3D visualization of the confidence region is enabled by setting do_live_plot=True in the second section (this requires the PyQt5 package to be installed).
4) A fourth section with the code corresponding to EXP2 (with corresponding default values).
5) A fifth section with the code corresponding to EXP3 (with corresponding default values).

The utils.py script file contains routines that generated iid Gaussian and Cauchy data with specified location vector and dispersion matrix.
The ellipsoid.py file contains routines that compute Ellipsoid objects (corresponding to our confidence regions), as well as methods that run elementary computations on them.

Feel free to play around with the different parameters of the test.py script!
If you wish to save the results of your numerical experiments, you will need to specify the path of the folder in which to save your data and figures, using the the pwd_save_data variable (line 23 of test.py). Numerical experiments are uniquely identified thanks to their time-stamp, and are accompanied with a simu_data.txt file detailing the corresponding parameters.
   
