from .ccg import mrc_ccg_large_n_m_sparse_binary
import time
import numpy as np
import itertools as it
import scipy.special as scs
import scipy as sp

from .cg_large_m.cg import alg1

def main_large_n_m_sparse_binary(X, y, fit_intercept, s, n_max, k_max, eps_1, eps_2, is_sparse=True, dict_nnz={}, max_iters=150):
	"""
	Efficient learning of 0-1 MRCs for sparse binary datasets with large number of samples and features.

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

	# Transformed features
	if fit_intercept is True:
		X_transformed = sp.sparse.hstack([sp.sparse.csr_matrix([[1]]*n), X]).tocsr()
		for sample_idx, nnz_arr in dict_nnz.items():
			nnz_arr_numpy = np.asarray(nnz_arr) + 1
			nnz_arr_intercept = [0]
			nnz_arr_intercept.extend(nnz_arr_numpy.tolist())
			dict_nnz[sample_idx] = nnz_arr_intercept

	# Compute the mean vector estimate
	tau_0 = X_transformed[y == 0, :].sum(axis=0)
	tau_1 = (-1) * X_transformed[y == 1, :].sum(axis=0)
	tau_ = (tau_0 + tau_1) / n

	# Compute the standard deviation
	std_mat = X_transformed.copy()
	std_mat.data **=2
	lambda_ = np.sqrt(std_mat.sum(axis=0) / n - np.square(tau_))

	# Reshape for compatibility with upcoming computations
	tau_ = np.reshape(np.asarray(tau_), (tau_.shape[1],))
	lambda_ = s * np.reshape(np.asarray(lambda_), (lambda_.shape[1],))

	# Calculate the time
	# Total time taken.
	totalTime = time.time()

	init_time_1 = time.time()
	#-> Initialization.

	n_classes = 2
	y_unique, count_y = np.unique(y, return_counts=True)

	# Initialization changes based on binary or multiclass classification due to one-hot encoding.
	idx_cols = []
	if is_sparse:
		#---> Reduce n by using centroids
		# Obtain the centroid of each class for initialization.
		centroid_0 = sp.sparse.csr_matrix(X[y == y_unique[0], :].sum(axis=0) * (1 / count_y[0]))
		centroid_1 = sp.sparse.csr_matrix(X[y == y_unique[1], :].sum(axis=0) * (1 / count_y[1]))

		if fit_intercept:
			centroid_0 = sp.sparse.hstack([sp.sparse.csr_matrix([[1]]), centroid_0]).tocsr()
			centroid_1 = sp.sparse.hstack([sp.sparse.csr_matrix([[1]]), centroid_1]).tocsr()

		# Create the constraint matrix for initialization using constraint generation
		centroid_0_arr = centroid_0.toarray()
		centroid_1_arr = centroid_1.toarray()
		phi_1 = np.asarray(
			[[centroid_0_arr[0], -1 * centroid_0_arr[0]], [centroid_1_arr[0], -1 * centroid_1_arr[0]]])

		F_init = np.vstack(list(np.sum(phi_1[:, S, ], axis=1)
								for numVals in range(1, n_classes + 1)
								for S in it.combinations(np.arange(n_classes), numVals)))

		cardS = np.arange(1, n_classes + 1). \
			repeat([n_classes * scs.comb(n_classes, numVals)
					for numVals in np.arange(1, n_classes + 1)])

		# Constraint coefficient matrix for obtaining initial set of features.
		F_init = F_init / (cardS[:, np.newaxis])

		# Coefficient vector of constraints for obtaining initial set of features.
		b_init = (1 / cardS) - 1

		# Add the samples corresponding with centroids
		X_full = sp.sparse.vstack([X_transformed, centroid_0, centroid_1])

		dict_nnz[n] = centroid_0.nonzero()[1].tolist()
		dict_nnz[n+1] = centroid_1.nonzero()[1].tolist()
		idx_samples_plus_constr = [n, n+1]
		idx_samples_minus_constr = [n, n+1]


		# #---> Now reduce m by using standard deviation among the taus across different classes.
		idx_cols = np.argsort(tau_)[:1000]

	#---> Find the corresponding intial set of features.
	print('Obtaining initial set of features using method in Bondugula et al. with time limit of 1 hr')
	mu, nu, R, idx_cols, R_k = alg1(F_init.copy(),
									b_init.copy(),
									tau_,
									lambda_,
									idx_cols,
									k_max=50,
									eps=0)

	init_time_1 = time.time() - init_time_1

	idx_cols = np.asarray(idx_cols)

	print('\n\n')

	#-> Run the CCG code.
	# Note that the input F_1 matrix should be of full size to be selected by idx_cols
	mu, nu, R, R_k, solver_times_gurobi, solver_times, idx_samples_plus_constr, idx_samples_minus_constr, idx_cols, initTime = mrc_ccg_large_n_m_sparse_binary(X_full,
																																							   idx_samples_plus_constr,
																																							   idx_samples_minus_constr,
																																							   tau_,
																																							   lambda_,
																																							   idx_cols,
																																							   n_max,
																																							   k_max,
																																							   nu_init=nu,
																																							   mu_init=mu,
																																							   eps_1=eps_1,
																																							   eps_2=eps_2,
																																							   is_sparse=is_sparse,
																																							   dict_nnz=dict_nnz,
																																							   max_iters=max_iters)
	totalTime = time.time() - totalTime
	solver_times[0] = solver_times[0] + init_time_1

	return mu, nu, R, R_k, solver_times_gurobi, solver_times, idx_cols, totalTime, initTime + init_time_1