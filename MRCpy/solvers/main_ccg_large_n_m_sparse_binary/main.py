"""
Main entry point for CCG solver for sparse binary MRC with large datasets.

This module provides the main function for learning Minimax Risk Classifiers
on sparse binary datasets with large numbers of samples and features. It
coordinates the initialization phase and the iterative CCG algorithm.
"""

from .ccg import mrc_ccg_large_n_m_sparse_binary
import time
import numpy as np
import itertools as it
import scipy.special as scs
import scipy as sp

from .cg_large_m.cg import alg1

def main_large_n_m_sparse_binary(X_transform, tau_, lambda_, n_max, k_max, eps_1, eps_2, dict_nnz={}, max_iters=150):
	"""
	Efficient learning of 0-1 MRCs for sparse binary datasets with large number 
	of samples and features.

	This function implements a two-phase approach for learning Minimax Risk
	Classifiers on high-dimensional sparse binary data:
	
	1. **Initialization Phase**: Uses a column generation algorithm to identify
	   an initial set of important features based on feature statistics (tau_).
	   
	2. **CCG Phase**: Applies the full column and constraint generation algorithm
	   to iteratively refine the solution by adding both features and sample
	   constraints.

	The algorithm is specifically optimized for sparse binary classification
	problems where both the number of samples (n) and features (m) are large,
	and the feature matrix is sparse.

	Parameters
	----------
	X_transform : scipy.sparse matrix of shape (n_samples, n_features)
		Training instances in sparse format (typically scipy.sparse.csr_matrix).
		Each row represents a training sample and each column represents a
		binary feature. The matrix should contain mostly zeros with binary
		(0 or 1) non-zero values.

	tau_ : numpy.ndarray of shape (n_features,)
		Mean estimates for each feature across the training distribution.
		These represent the expected value of each feature and are used to
		define the uncertainty set in the MRC formulation.

	lambda_ : numpy.ndarray of shape (n_features,)
		Deviation estimates (uncertainty bounds) for each feature. These
		represent the maximum deviation from tau_ allowed in the uncertainty
		set. Larger values indicate more uncertainty about the feature.

	n_max : int
		Maximum number of constraints (samples) to add per iteration during
		the CCG phase. Controls the rate at which sample constraints are
		added to the model. Typical values: 100-500.

	k_max : int
		Maximum number of features (columns) to add per iteration during
		the CCG phase. Controls the rate at which features are added to
		the model. Typical values: 100-500.

	eps_1 : float
		Constraint violation threshold for primal constraints. Constraints
		violated by more than this amount will be added to the model during
		the CCG phase. Smaller values lead to tighter solutions but more
		constraints. Typical values: 1e-3 to 1e-1.

	eps_2 : float
		Feature violation threshold for dual constraints. Features with
		dual violations exceeding this amount will be added to the model
		during the CCG phase. Smaller values lead to more features being
		considered. Typical values: 1e-6 to 1e-4.

	dict_nnz : dict, default={}
		Dictionary mapping sample indices (int) to lists of non-zero feature
		indices (list of int) for efficient sparse matrix operations. If
		empty, will be computed as needed. Pre-computing this can improve
		performance for very large datasets.
		
		Example: {0: [1, 5, 10], 1: [2, 5, 8], ...}

	max_iters : int, default=150
		Maximum number of column and constraint generation iterations in
		the CCG phase. The algorithm terminates when either no violations
		remain or this limit is reached. Typical values: 50-200.

	Returns
	-------
	mu : numpy.ndarray of shape (n_features,)
		Learned parameter vector of the MRC classifier. This is a sparse
		vector where most entries are zero. Non-zero entries correspond
		to selected features and their learned weights.

	nu : float
		Learned intercept parameter of the MRC classifier. This is the
		bias term in the linear decision function.

	R : float
		Optimized upper bound on the worst-case error probability of the
		MRC classifier. This represents the maximum error rate over all
		distributions in the uncertainty set. Lower values indicate better
		worst-case performance.

	R_k : list of float
		List of worst-case error probabilities obtained at each iteration
		of the CCG algorithm. Useful for analyzing convergence behavior.
		The length equals the number of iterations performed.

	Notes
	-----
	The algorithm uses a two-phase approach:
	
	**Phase 1 - Initialization**:
	- Constructs an initial constraint matrix based on feature means (tau_)
	- Uses column generation (alg1) to select an initial set of ~1000 features
	- Adds a centroid sample to initialize the constraint set
	
	**Phase 2 - CCG Iterations**:
	- Iteratively solves the restricted optimization problem
	- Identifies and adds violated dual constraints (features)
	- Identifies and adds violated primal constraints (samples)
	- Continues until convergence or max_iters is reached

	The algorithm requires Gurobi as the LP solver. Ensure Gurobi is properly
	installed and licensed before using this function.

	The function is designed for binary classification problems. For multi-class
	problems, use the appropriate multi-class variant.

	Examples
	--------
	>>> import numpy as np
	>>> from scipy.sparse import csr_matrix
	>>> # Create sparse binary feature matrix (1000 samples, 5000 features)
	>>> X = csr_matrix(np.random.binomial(1, 0.01, (1000, 5000)))
	>>> # Compute feature statistics
	>>> tau = np.array(X.mean(axis=0)).flatten()
	>>> lambda_ = np.array(X.std(axis=0)).flatten()
	>>> # Build non-zero index dictionary
	>>> dict_nnz = {i: X[i].nonzero()[1].tolist() for i in range(X.shape[0])}
	>>> # Run the algorithm
	>>> mu, nu, R, R_k = main_large_n_m_sparse_binary(
	...     X, tau, lambda_,
	...     n_max=200, k_max=200,
	...     eps_1=1e-2, eps_2=1e-5,
	...     dict_nnz=dict_nnz, max_iters=100
	... )
	>>> print(f"Worst-case error bound: {R:.4f}")
	>>> print(f"Number of selected features: {np.sum(mu != 0)}")

	References
	----------
	The algorithm is based on column and constraint generation techniques
	for large-scale robust optimization. See the MRCpy documentation for
	more details on the theoretical foundation.
	"""

	#-> Initialization.
	n = X_transform.shape[0]
	n_classes = 2

	# Initialization changes based on binary or multiclass classification due to one-hot encoding.
	idx_cols = []
	#---> Reduce n by using mean vector tau_
	phi_1 = tau_
	n_init = phi_1.shape[0]
	F_init = np.vstack(list(np.sum(phi_1[:, S, ], axis=1)
							for numVals in range(1, n_classes + 1)
							for S in it.combinations(np.arange(n_classes), numVals)))

	cardS = np.arange(1, n_classes + 1). \
		repeat([n_init * scs.comb(n_classes, numVals)
				for numVals in np.arange(1, n_classes + 1)])

	# Constraint coefficient matrix for obtaining initial set of features.
	F_init = F_init / (cardS[:, np.newaxis])

	# Coefficient vector of constraints for obtaining initial set of features.
	b_init = (1 / cardS) - 1

	# Add the samples corresponding with centroids
	X_full = sp.sparse.vstack([X_transform, tau_])

	dict_nnz[n] = tau_.nonzero()[1].tolist()
	idx_samples_plus_constr = [n]
	idx_samples_minus_constr = [n]

	# #---> Now reduce m by using standard deviation among the taus across different classes.
	idx_cols = np.argsort(tau_)[:1000]

	#---> Find the corresponding intial set of features.
	mu, nu, R, idx_cols, R_k = alg1(F_init.copy(),
									b_init.copy(),
									tau_,
									lambda_,
									idx_cols,
									k_max=50,
									eps=0)


	idx_cols = np.asarray(idx_cols)

	#-> Run the CCG code.
	# Note that the input F_1 matrix should be of full size to be selected by idx_cols
	mu, nu, R, R_k, idx_samples_plus_constr, idx_samples_minus_constr, idx_cols = mrc_ccg_large_n_m_sparse_binary(X_full,
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
																												dict_nnz=dict_nnz,
																												max_iters=max_iters)
	return mu, nu, R, R_k