"""
Main entry point for CCG solver for large-scale MRC.

This module provides the main function for learning Minimax Risk Classifiers
on datasets with large numbers of both samples and features. It coordinates
the initialization phase and the iterative CCG algorithm.
"""

from .ccg import mrc_ccg_large_n_m
import time
import numpy as np
import itertools as it
import scipy.special as scs

from .cg_large_m.cg import alg1

def main_large_n_m(X, tau_mat, lambda_mat, phi_ob, n_max, k_max, eps_1, eps_2, max_iters):
	"""
	Efficient learning of 0-1 MRCs for large numbers of samples and features.

	This function implements a two-phase approach for learning Minimax Risk
	Classifiers on datasets with large numbers of both samples and features:
	
	1. **Initialization Phase**: Uses centroid-based initialization and column
	   generation to identify an initial set of important features.
	   
	2. **CCG Phase**: Applies the full column and constraint generation algorithm
	   to iteratively refine the solution by adding both features and sample
	   constraints.

	The algorithm is specifically optimized for problems where both the number
	of samples (n) and features (m) are large.

	Parameters
	----------
	X : numpy.ndarray of shape (n_samples, n_features)
		Training feature matrix. Each row represents a training sample and
		each column represents a feature.

	tau_mat : numpy.ndarray of shape (n_classes, d)
		Mean estimates for each feature across each class. Each row corresponds
		to a class, and each column corresponds to a feature. For binary
		classification, shape is (1, d).

	lambda_mat : numpy.ndarray of shape (n_classes, d)
		Deviation estimates (uncertainty bounds) for each feature across each
		class. Represents the maximum deviation from tau_mat allowed in the
		uncertainty set. Shape matches tau_mat.

	phi_ob : BasePhi instance
		Feature mapping object that transforms input data. This object must
		have methods `eval_x` for evaluating feature mappings and attributes
		`n_classes`, `fit_intercept`, and `len_`.

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

	max_iters : int
		Maximum number of column and constraint generation iterations in
		the CCG phase. The algorithm terminates when either no violations
		remain or this limit is reached. Typical values: 50-200.

	Returns
	-------
	mu : numpy.ndarray of shape (n_features,)
		Learned parameter vector of the MRC classifier. These are the
		optimal weights for features in the final working set.

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
	- Constructs initial constraint matrix based on class centroids
	- Uses column generation (alg1) to select an initial set of ~100 features
	  per class based on standard deviation across classes
	- Adds centroid constraints to initialize the constraint set
	
	**Phase 2 - CCG Iterations**:
	- Iteratively solves the restricted optimization problem
	- Identifies and adds violated dual constraints (features)
	- Identifies and adds violated primal constraints (samples)
	- Continues until convergence or max_iters is reached

	The algorithm requires Gurobi as the LP solver. Ensure Gurobi is properly
	installed and licensed before using this function.

	The function temporarily disables `fit_intercept` on phi_ob during
	initialization and restores it afterward.

	Examples
	--------
	>>> import numpy as np
	>>> from MRCpy import BasePhi
	>>> # Create feature matrix
	>>> X = np.random.randn(1000, 100)
	>>> y = np.random.randint(0, 3, 1000)
	>>> # Initialize feature mapping
	>>> phi = BasePhi(n_classes=3)
	>>> # Compute feature statistics per class (shape: n_classes x d)
	>>> tau_mat = np.array([X[y==i].mean(axis=0) for i in range(3)])  # Shape: (3, 100)
	>>> lambda_mat = np.array([X[y==i].std(axis=0) for i in range(3)])  # Shape: (3, 100)
	>>> # Run CCG algorithm
	>>> mu, nu, R, R_k = main_large_n_m(
	...     X, tau_mat, lambda_mat, phi,
	...     n_max=200, k_max=200,
	...     eps_1=1e-2, eps_2=1e-5, max_iters=100
	... )
	>>> print(f"Worst-case error bound: {R:.4f}")
	>>> print(f"Converged in {len(R_k)} iterations")

	References
	----------
	The algorithm is based on column and constraint generation techniques
	for large-scale robust optimization. See Mazumdar et al. for theoretical
	details on MRC and CCG methods.
	"""

	#-> Initialization.
	n = X.shape[0]
	n_classes = tau_mat.shape[0]
	d = tau_mat.shape[1]  # Number of features

	tau_ = tau_mat.flatten()
	lambda_ = lambda_mat.flatten()

	# Initialization changes based on binary or multiclass classification due to one-hot encoding.
# ---> Reduce n by using centroids
	# Add some of the constraints corresponding to representative samples
	# for each class as initialization.
	# Adding these enables faster convergence and does not affect the solution.
	fit_intercept = phi_ob.fit_intercept
	phi_ob.fit_intercept = False
	# Build feature mapping directly from tau_mat (already in transformed
	# feature space) to avoid double transformation via eval_x.
	# Use one-hot encoding: phi_1[i, y, :] = kron(e_y, tau_mat[i, :])
	n_init = tau_mat.shape[0]
	phi_1 = np.zeros((n_init, n_classes, d * n_classes))
	for i in range(n_init):
		for y in range(n_classes):
			phi_1[i, y, :] = np.kron(np.eye(n_classes)[y], tau_mat[i, :])
	F_init = np.reshape(phi_1, (n_init * n_classes, phi_ob.len_))
	b_init = np.tile(np.zeros(phi_ob.n_classes), n_init)

# ---> Now reduce m by using standard deviation among the taus across different classes.
	idx_cols_no_one_hot = np.argsort(np.std(tau_mat, axis = 0))[:100]
	idx_cols = idx_cols_no_one_hot.copy()
	for y_i in range(1, n_classes):
		idx_cols = np.append(idx_cols, idx_cols_no_one_hot + (d * y_i))
	phi_ob.fit_intercept = fit_intercept

	# Run few iterations of mrc cg on the centroids to obtain the initial set of features.
	mu, nu, R, idx_cols, R_k = alg1(F_init.copy(), b_init.copy(), tau_, lambda_, idx_cols, k_max=50, eps=0)
	F_1 = F_init
	b_1 = b_init

	idx_cols = np.asarray(idx_cols)

	#-> Run the CG code.
	# Note that the input F_1 matrix should be of full size to be selected by idx_cols
	mu, nu, R, R_k = mrc_ccg_large_n_m(F_1,
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

	return mu, nu, R, R_k