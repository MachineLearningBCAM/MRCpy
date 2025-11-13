from .cg import mrc_ccg_large_n_multiclass
import time
import numpy as np
import itertools as it
import scipy.special as scs

def main_large_n_efficient_multiclass(X, y, phi_ob, s, n_max, k_max, eps):
	"""
	Efficient learning of 0-1 MRCs for large number of samples or
	multi-class problems.

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

	s : `float`, default = `0.3`
		Parameter that tunes the estimation of expected values
		of feature mapping function. It is used to calculate :math:`\lambda`
        (variance in the mean estimates
        for the expectations of the feature mappings) in the following way

        .. math::
            \\lambda = s * \\text{std}(\\phi(X,Y)) / \\sqrt{\\left| X \\right|}

        where (X,Y) is the dataset of training samples and their
        labels respectively and
        :math:`\\text{std}(\\phi(X,Y))` stands for standard deviation
        of :math:`\\phi(X,Y)` in the supervised dataset (X,Y).

	n_max : `int`, default=`100`
		Maximum number of features selected in each iteration of the algorithm.

	k_max : `int`, default=`20`
		Maximum number of iterations allowed for termination of the algorithm.

	eps : `float`, default=`1e-4`
		Constraints' threshold. Maximum violation allowed in the constraints.

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


	# Compute tau and lambda
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


	# Calculate the time
	# Total time taken.
	totalTime = time.time()

	init_time_1 = time.time()
	#-> Initialization.
	# 1. Using the constraints of the centroids as the initial points.
	# 	 This initialization works well and does not lead to empty uncertainty sets.
	#	 The drawback with such initialization is that it grows exponentially.

	# F_, b_ = fo_init_1(tau_, phi_ob, y)

	# 2. Adding the centroids to the constraint matrix and
	# 	 using the subset of the centroid constraints as initialization.
	# 	 This initialization grows |Y|^2 and works with objective >= 0 constraint
	y_unique, count_y = np.unique(y, return_counts=True)

	# Obtain the centroid of each class for initialization.
	X_art = []
	constr_dict = {}
	subsets_arr = []
	for i in range(n_classes):
		subsets_arr.append([i])

	for i in range(n_classes):
		constr_dict[i] = subsets_arr.copy()
		X_art.append(tau_mat[i, :] * (n / count_y[i]))

	X_art = np.asarray(X_art)

	# Add some of the constraints corresponding to centroids as initialization
	# and all the centroids as instances to be checked for constraint violation.
	# Adding the centroids enables faster convergence and doesnot affect the solution.
	# Note that we cannot add all the constraints corresponding to the centroids
	# as the number of constraints grows exponentially with the number of classes.
	# Therefore, we only add the linear constraints of the centroids as the initiallization,
	# that is, adding |Y|^2 constraints.
	X_full = np.vstack((X_art, X_transform))

	init_time_1 = time.time() - init_time_1
	#-> Run the CG code.
	mu, nu, R, R_k, solver_times, initTime, constr_dict = mrc_ccg_large_n_multiclass(constr_dict,
																					 X_full,
																					 tau_mat,
																					 lambda_mat,
																					 n_max,
																					 k_max,
																					 None,
																					 eps)

	totalTime = time.time() - totalTime
	solver_times[0] = solver_times[0] + init_time_1

	return mu, nu, R, R_k, totalTime, solver_times, initTime + init_time_1, constr_dict