from .ccg import mrc_ccg_large_n_m
import time
import numpy as np
import itertools as it
import scipy.special as scs

from .cg_large_m.cg import alg1

def main_large_n_m(X, y, phi_ob, s, n_max, k_max, eps_1, eps_2, max_iters):
	"""
	Efficient learning of 0-1 MRCs for large number of samples and features.

	Parameters
	----------
	X : `array`-like of shape (`n_samples`, `n_features`)
		Training instances used in

		`n_samples` is the number of training samples and
		`n_features` is the number of features.

	y : `array`-like of shape (`n_samples`, 1), default = `None`
		Labels corresponding to the training instances
		used only to compute the expectation estimates.

	phi_ob : `BasePhi` instance
		This is an instance of the `BasePhi` 
		feature mapping class of MRCs. Any feature
		mapping object derived from BasePhi can be used
		to provide a new interpretation to the input data `X`.

	s : `float`, default = `0.01`
		Regularization parameter

	n_max : `int`, default=`400`
		Maximum number of features selected in each iteration of the algorithm.

	k_max : `int`, default=`60`
		Maximum number of iterations allowed for termination of the algorithm.

	eps : `float`, default=`1e-2`
		Constraints' violation threshold. Maximum violation allowed in the constraints.

	Returns
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

	totalTime : `float`
		Total time taken by the algorithm.

	initTime : `float`
		Time taken for the initialization to the algorithm.
	"""

	n = X.shape[0]
	n_classes = len(np.unique(y))

	if n_classes == 2:
		phi_xy = phi_ob.eval_xy(X, y)
		tau_ = np.average(phi_xy, axis=0)
		lambda_ = s * np.std(phi_xy, axis=0)
	elif n_classes > 2:
		# Compute tau and lambda without creating the one-hot encoded phi matrix
		X_transform = phi_ob.transform(X)
		X_transform = np.hstack(([[1]] * n, X_transform))
		feat_len = X_transform.shape[1]

		# Compute tau efficiently
		tau_mat = np.zeros((n_classes, feat_len))
		for y_i in range(n_classes):
			tau_mat[y_i, :] = np.sum(X_transform[y == y_i, :], axis=0) * (1 / n)

		# Compute lambda efficiently
		lambda_mat = np.zeros((n_classes, feat_len))
		for y_i in range(n_classes):
			not_y_i = np.sum(y!=y_i)
			lambda_mat[y_i, :] = s * np.sqrt(np.sum(np.square(np.abs(X_transform[y == y_i, :] - tau_mat[y_i, :])), axis=0) + \
											 (np.square(np.abs(tau_mat[y_i, :])) * not_y_i)) / np.sqrt(n)

		tau_ = np.reshape(tau_mat, (n_classes * feat_len, ))
		lambda_ = np.reshape(lambda_mat, (n_classes * feat_len, ))

	# Calculate the time
	# Total time taken.
	totalTime = time.time()

	init_time_1 = time.time()
	#-> Initialization.

	n_classes = phi_ob.n_classes
	y_unique, count_y = np.unique(y, return_counts=True)

	# Initialization changes based on binary or multiclass classification due to one-hot encoding.
	if phi_ob.one_hot is False:

	#---> Reduce n by using centroids
		# Obtain the centroid of each class for initialization.
		X_transformed = phi_ob.transform(X)
		tau_mat = np.zeros((n_classes, X_transformed.shape[1]))
		tau_mat[0, :] = np.sum(X_transformed[y == y_unique[0], :], axis=0) * (1 / count_y[0])
		tau_mat[1, :] = np.sum(X_transformed[y == y_unique[1], :], axis=0) * (1 / count_y[1])

		# Generate all the constraints using the centroids as instances (6 constraints for binary)
		phi_1 = phi_ob.eval_x(tau_mat)
		n = n_classes
		F_init = np.vstack(list(np.sum(phi_1[:, S, ], axis=1)
							for numVals in range(1, n_classes + 1)
							for S in it.combinations(np.arange(n_classes), numVals)))

		cardS = np.arange(1, n_classes + 1). \
			repeat([n * scs.comb(n_classes, numVals)
					for numVals in np.arange(1, n_classes + 1)])

		# Constraint coefficient matrix
		F_init = F_init / (cardS[:, np.newaxis])

		# The bounds on the constraints
		b_init = (1 / cardS) - 1

		#---> Reduce no of features based on weight tau.
		idx_cols = np.argsort(tau_)[:1000]

	else:

	# ---> Reduce n by using centroids
		# Obtain the centroid of each class for initialization.
		d = int(tau_.shape[0] / n_classes)
		tau_mat = np.reshape(tau_.copy(), (n_classes, d))
		for i in range(n_classes):
			tau_mat[i, :] = tau_mat[i, :] * (y.shape[0] / count_y[i])

		# Add some of the constraints corresponding to centroids as initialization
		# and all the centroids as instances to be checked for constraint violation.
		# Adding the centroids enables faster convergence and doesnot affect the solution.
		# Note that we cannot add all the constraints corresponding to the centroids
		# as the number of constraints grows exponentially with the number of classes.
		n = n_classes
		fit_intercept = phi_ob.fit_intercept
		phi_ob.fit_intercept = False
		phi_1 = phi_ob.eval_x(tau_mat)
		F_init = np.reshape(phi_1, (n * n_classes, phi_ob.len_))
		b_init = np.tile(np.zeros(phi_ob.n_classes), n)

	# ---> Now reduce m by using standard deviation among the taus across different classes.
		idx_cols_no_one_hot = np.argsort(np.std(tau_mat, axis = 0))[:100]
		idx_cols = idx_cols_no_one_hot.copy()
		for y_i in range(1, len(y_unique)):
			idx_cols = np.append(idx_cols, idx_cols_no_one_hot + (d * y_i))
		phi_ob.fit_intercept = fit_intercept

	# Run few iterations of mrc cg on the centroids to obtain the initial set of features.
	print('Obtaining initial set of features using method in Bondugula et al. with time limit of 1 hr')
	mu, nu, R, idx_cols, R_k = alg1(F_init.copy(), b_init.copy(), tau_, lambda_, idx_cols, k_max=50, eps=0)
	F_1 = F_init
	b_1 = b_init

	init_time_1 = time.time() - init_time_1
	idx_cols = np.asarray(idx_cols)

	#-> Run the CG code.
	# Note that the input F_1 matrix should be of full size to be selected by idx_cols
	mu, nu, R, R_k, solver_times_gurobi, solver_times, idx_cols, initTime = mrc_ccg_large_n_m(F_1,
																							b_1,
																							X,
																							phi_ob,
																							tau_,
																							lambda_,
																							idx_cols,
																							n_max,
																							k_max,
																							nu,
																							mu,
																							eps_1,
																							eps_2,
																							max_iters)
	totalTime = time.time() - totalTime
	solver_times[0] = solver_times[0] + init_time_1

	return mu, nu, R, R_k, solver_times_gurobi, solver_times, idx_cols, totalTime, initTime + init_time_1