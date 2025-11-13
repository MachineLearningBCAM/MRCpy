import numpy as np
from .mrc_dual_lp import mrc_dual_lp_model
from .cg_utils import select
import time

def mrc_ccg_large_n_multiclass(constr_dict, X_full, tau_, lambda_, n_max=400, k_max=60, warm_start=None, eps=1e-2):
	"""
	Constraint generation algorithm for Minimax Risk Classifiers.

	Parameters:
	-----------
	F : `array`-like of shape (no_of_constraints, 2*(no_of_feature+1))
		Constraint matrix.

	b : `array`-like of shape (no_of_constraints)
		Right handside of the constraints.

	tau_ : `array`-like of shape (no_of_features)
		Mean estimates.

	lambda_ : `array`-like of shape (no_of_features)
		Standard deviation of the estimates.

	I : `list`
		List of feature indices corresponding to features in matrix M.
		This is the initialization for the constraint generation method.

	n_max : `int`, default=`100`
		Maximum number of features selected in each iteration of the algorithm

	k_max : `int`, default=`20`
		Maximum number of iterations allowed for termination of the algorithm

	warm_start : `list`, default=`None`
		Coefficients corresponding to features in I as a warm start
		for the initial problem.

	nu_init : `int`, default=`None`
		Coefficient nu corresponding to the warm start (mu)

	eps : `float`, default=`1e-4`
		Constraints' threshold. Maximum violation allowed in the constraints.

	Return:
	-------
	mu : `array`-like of shape (`n_features`) or `float`
		Parameters learnt by the algorithm.

	nu : `float`
		Parameter learnt by the algorithm.

	R : `float`
		Optimized upper bound of the MRC classifier.

	I : `list`
		List of indices of the features selected

	R_k : `list` of shape (no_of_iterations)
		List of worst-case error probabilites
		obtained for the subproblems at each iteration.
	"""

	# Generate the matrices for the linear optimization of 0-1 MRC
	# from the feature mappings.

	print('MRC-CCG with n_max = ' + str(n_max) + ', k_max = ' + str(k_max) + ', eps_1 = ' + str(eps))
	R_k = []
	n_classes = tau_.shape[0]
	d = tau_.shape[1]
	solver_times = []

	solver_init_time = time.time()
	initTime = time.time()
	# Initial optimization
	MRC_model, constr_var_dict = mrc_dual_lp_model(X_full,
												  constr_dict,
												  tau_,
												  lambda_,
												  warm_start)

	initTime = time.time() - initTime
	R_k.append(MRC_model.objVal)

	# Primal solution
	mu_plus = []
	mu_minus = []
	for y_i in range(n_classes):
		mu_plus.append([(MRC_model.getConstrByName("constr_+_" + str(d * y_i + i))).Pi for i in range(d)])
		mu_minus.append([(MRC_model.getConstrByName("constr_-_" + str(d * y_i + i))).Pi for i in range(d)])
	nu = MRC_model.getConstrByName("constr_=").Pi

	mu = np.asarray(mu_plus) - np.asarray(mu_minus)
	mu[np.isclose(mu, 0)] = 0

	print('The initial worst-case error probability : ', MRC_model.objVal)
	solver_times.append(time.time() - solver_init_time)
	last_checked = 0

	# Add the columns to the model.
	MRC_model, constr_dict, constr_var_dict, count_added, last_checked = select(MRC_model,
																			 X_full,
																			 constr_dict,
																			 constr_var_dict,
																			 n_max,
																			 mu,
																			 nu,
																			 eps,
																			 last_checked)

	k = 0
	while(k < k_max and count_added > 0):

		init_solve_time = time.time()
		# Solve the updated optimization and get the dual solution.
		MRC_model.optimize()

		# Get the primal solution
		mu_plus = []
		mu_minus = []
		for y_i in range(n_classes):
			mu_plus.append([(MRC_model.getConstrByName("constr_+_" + str(d*y_i + i))).Pi for i in range(d)])
			mu_minus.append([(MRC_model.getConstrByName("constr_-_" + str(d*y_i + i))).Pi for i in range(d)])

		nu = MRC_model.getConstrByName("constr_=").Pi
		mu = np.asarray(mu_plus) - np.asarray(mu_minus)

		print('The worst-case error probability at iteration ' + str(k) + ' is ', MRC_model.objVal)
		R_k.append(MRC_model.objVal)
		solver_times.append(time.time() - solver_init_time)

		# Select the columns/features for the next iteration.
		MRC_model, constr_dict, constr_var_dict, count_added, last_checked = select(MRC_model,
																					 X_full,
																					 constr_dict,
																					 constr_var_dict,
																					 n_max,
																					 mu,
																					 nu,
																					 eps,
																					 last_checked)
		k = k + 1

	# Obtain the final primal solution.
	R 			= MRC_model.objVal
	mu = np.reshape(mu, (mu.shape[0] * mu.shape[1],))

	return mu, nu, R, R_k, solver_times, initTime, constr_dict
