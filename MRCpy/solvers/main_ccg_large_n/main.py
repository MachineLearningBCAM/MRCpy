from .cg import mrc_ccg_large_n
import time
import numpy as np
import itertools as it
import scipy.special as scs

def main_large_n(X, y, phi_ob, s, n_max, k_max, eps):
	"""
	Efficient learning of 0-1 MRCs for large number of samples. 
	This version should handle multi-class but is preferable to use 
	the efficient implementation provided in the other folder.

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

	phi_ = phi_ob.eval_x(X)
	tau_ = phi_ob.est_exp(X, y)
	lambda_ = s * (phi_ob.est_std(X, y))

	print('Shape of phi: ', phi_.shape)

	# Calculate the time
	# Total time taken.
	totalTime = time.time()

	init_time_1 = time.time()
	#-> Initialization.
	# 1. Using the constraints of the centroids as the initial points.
	# 	 This initialization works well and does not lead to empty uncertainty sets.
	#	 The drawback with such initialization is that it grows exponentially.

	# 2. Adding the centroids to the constraint matrix and
	# 	 using the subset of the centroid constraints as initialization.
	# 	 This initialization grows |Y|^2 and works with objective >= 0 constraint
	n_classes = phi_ob.n_classes
	y_unique, count_y = np.unique(y, return_counts=True)

	# Initialization changes based on binary or multiclass classification
	# as in multiclass, the number of constraints increase exponentially.
	if phi_ob.one_hot is False:

		# Obtain the centroid of each class for initialization.
		X_transformed = phi_ob.transform(X)
		tau_mat = np.zeros((n_classes, X_transformed.shape[1]))
		tau_mat[0, :] = np.sum(X_transformed[y == y_unique[0], :], axis=0) * (1 / count_y[0])
		tau_mat[1, :] = np.sum(X_transformed[y == y_unique[1], :], axis=0) * (1 / count_y[1])

		# Generate all the constraints using the centroids as instances (6 constraints for binary)
		phi_1 = phi_ob.eval_x(tau_mat)
		n = n_classes
		F_ = np.vstack(list(np.sum(phi_1[:, S, ], axis=1)
							for numVals in range(1, n_classes + 1)
							for S in it.combinations(np.arange(n_classes), numVals)))

		cardS = np.arange(1, n_classes + 1). \
			repeat([n * scs.comb(n_classes, numVals)
					for numVals in np.arange(1, n_classes + 1)])

		# Constraint coefficient matrix
		F_ = F_ / (cardS[:, np.newaxis])

		# The bounds on the constraints
		b_ = (1 / cardS) - 1
	else:
		# Obtain the centroid of each class for initialization.
		tau_mat = np.reshape(tau_.copy(), (n_classes, int(tau_.shape[0] / n_classes)))
		for i in range(n_classes):
			tau_mat[i, :] = tau_mat[i, :] * (y.shape[0] / count_y[i])

		# Add some of the constraints corresponding to centroids as initialization
		# and all the centroids as instances to be checked for constraint violation.
		# Adding the centroids enables faster convergence and doesnot affect the solution.
		# Note that we cannot add all the constraints corresponding to the centroids
		# as the number of constraints grows exponentially with the number of classes.
		# Therefore, we only add the linear constraints of the centroids as the initiallization,
		# that is, adding |Y|^2 constraints.
		n = n_classes
		fit_intercept = phi_ob.fit_intercept
		phi_ob.fit_intercept = False
		phi_1 = phi_ob.eval_x(tau_mat)
		F_ = np.reshape(phi_1, (n * phi_.shape[1], phi_.shape[2]))
		b_ = np.tile(np.zeros(phi_ob.n_classes), n)
		phi_ = np.vstack((phi_1, phi_))

		phi_ob.fit_intercept = fit_intercept

	init_time_1 = time.time() - init_time_1
	#-> Run the CG code.
	mu, nu, R, R_k, solver_times, initTime = mrc_ccg_large_n(F_,
															 b_,
															 phi_,
															 tau_,
															 lambda_,
															 n_max,
															 k_max,
															 None,
															 eps)

	totalTime = time.time() - totalTime
	solver_times[0] = solver_times[0] + init_time_1

	return mu, nu, R, R_k, totalTime, solver_times, initTime + init_time_1