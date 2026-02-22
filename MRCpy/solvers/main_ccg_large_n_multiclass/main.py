"""
Main entry point for CG solver for multiclass large-sample MRC.

This module provides the main function for learning Minimax Risk Classifiers
on multiclass datasets with large numbers of samples. It coordinates the
initialization phase and the iterative constraint generation algorithm.
"""

from .cg import mrc_ccg_large_n_multiclass
import time
import numpy as np
import itertools as it
import scipy.special as scs

def main_large_n_efficient_multiclass(X, tau_mat, lambda_mat, n_max, max_iters, eps):
	"""
	Efficient learning of 0-1 MRCs for large numbers of samples in multiclass
	problems.

	This function implements a constraint generation approach specifically
	optimized for multiclass classification problems with large numbers of
	samples. It uses a dual formulation with efficient constraint tracking
	and iteratively adds violated constraints to solve the MRC optimization
	problem.

	Parameters
	----------
	X : numpy.ndarray of shape (n_samples, n_features)
		Transformed training feature matrix. Each row represents a training
		sample and each column represents a feature. This should be the
		feature-transformed version of the original data.

	tau_mat : numpy.ndarray of shape (n_classes, d)
		Mean estimates for each feature across each class. tau_mat[y, i]
		represents the mean of feature i for class y. Used to define the
		uncertainty set and for centroid initialization.

	lambda_mat : numpy.ndarray of shape (n_classes, d)
		Deviation estimates (uncertainty bounds) for each feature across each
		class. lambda_mat[y, i] represents the uncertainty bound for feature i
		in class y. Used to define the uncertainty set.

	n_max : int
		Maximum number of constraints to add per iteration. Controls the rate
		at which the constraint set grows. Larger values may speed up
		convergence but increase memory usage. Typical values: 100-500.

	max_iters : int
		Maximum number of constraint generation iterations. The algorithm
		terminates when either no violations remain or this limit is reached.
		Typical values: 20-100.

	eps : float
		Constraint violation threshold. Constraints violated by more than this
		amount will be added to the model. Smaller values lead to more
		constraints being added and tighter solutions. Typical values: 1e-3
		to 1e-1.

	Returns
	-------
	mu : numpy.ndarray of shape (n_features * n_classes,)
		Learned feature coefficients flattened across all classes. The
		coefficients are reshaped from (n_classes, n_features) to a 1D array.

	nu : float
		Learned intercept parameter. This is the bias term in the linear
		classifier obtained from the dual solution.

	R : float
		Final objective value representing the optimized upper bound on the
		worst-case error probability.

	R_k : list of float
		Objective values at each iteration, tracking convergence. The length
		equals the number of iterations performed plus one (for the initial
		solution).

	constr_dict : dict
		Final dictionary of constraints mapping sample indices to lists of
		class subsets. Shows which constraints were added for each sample
		during the algorithm. Useful for analysis and debugging.

	Notes
	-----
	The algorithm uses a two-phase approach:
	
	**Phase 1 - Initialization**:
	- Creates artificial centroid samples from tau_mat (one per class)
	- Initializes constraint dictionary with linear constraints (|Y|^2 total)
	- Each centroid gets constraints for all individual classes
	- Stacks centroids with training data to form X_full
	
	**Phase 2 - CG Iterations**:
	- Iteratively solves the restricted dual optimization problem
	- Identifies and adds violated primal constraints (samples)
	- Removes over-satisfied constraints to keep model size manageable
	- Continues until convergence or max_iters is reached

	The initialization strategy avoids the empty uncertainty set problem that
	can occur with few constraints. By adding centroid constraints, the
	algorithm starts with a feasible solution and converges faster.

	The constraint dictionary (constr_dict) tracks which class subsets have
	been added as constraints for each sample. This allows efficient constraint
	management and removal of redundant constraints.

	The algorithm requires Gurobi as the LP solver. Ensure Gurobi is properly
	installed and licensed before using this function.

	For binary classification (n_classes=2), the algorithm still works but
	the general large-sample solver may be more efficient.

	Examples
	--------
	>>> import numpy as np
	>>> # Create feature matrix for 1000 samples, 50 features
	>>> X = np.random.randn(1000, 50)
	>>> y = np.random.randint(0, 3, 1000)
	>>> # Compute feature statistics per class
	>>> tau_mat = np.array([X[y==i].mean(axis=0) for i in range(3)]).T
	>>> lambda_mat = np.array([X[y==i].std(axis=0) for i in range(3)]).T
	>>> # Run CG algorithm
	>>> mu, nu, R, R_k, constr_dict = main_large_n_efficient_multiclass(
	...     X, tau_mat, lambda_mat,
	...     n_max=200, max_iters=50, eps=1e-2
	... )
	>>> print(f"Worst-case error bound: {R:.4f}")
	>>> print(f"Converged in {len(R_k)} iterations")
	>>> print(f"Total constraints added: {sum(len(v) for v in constr_dict.values())}")

	References
	----------
	The algorithm is based on constraint generation techniques for large-scale
	multiclass optimization problems. See Mazumdar et al. for theoretical
	details on MRC and constraint generation methods for multiclass problems.
	"""

	n = X.shape[0]
	if tau_mat.shape[0] == 1:
		# Binary classification
		n_classes = 2
	else:
		n_classes = tau_mat.shape[0]

	# Compute tau and lambda
	feat_len = X.shape[1]

	#-> Initialization.
	#   Adding the centroids to the constraint matrix and
	# 	using the subset of the centroid constraints as initialization.
	# 	This initialization grows |Y|^2 and works with objective >= 0 constraint

	# Obtain the centroid of each class (artifical instances) for initialization.
	# This avoids the empty uncertainty set for initial iteration corresponding
	# with few constraints.
	X_art = []
	constr_dict = {}
	subsets_arr = []
	for i in range(n_classes):
		subsets_arr.append([i])

	for i in range(n_classes):
		constr_dict[i] = subsets_arr.copy()
		X_art.append(tau_mat[i, :])

	X_art = np.asarray(X_art)

	# Add some of the constraints corresponding to centroids as initialization
	# and all the centroids as instances to be checked for constraint violation.
	# Adding the centroids enables faster convergence and doesnot affect the solution.
	# Note that we cannot add all the constraints corresponding to the centroids
	# as the number of constraints grows exponentially with the number of classes.
	# Therefore, we only add the linear constraints of the centroids as the initiallization,
	# that is, adding |Y|^2 constraints.
	X_full = np.vstack((X_art, X))

	#-> Run the CG code.
	mu, nu, R, R_k, constr_dict = mrc_ccg_large_n_multiclass(constr_dict,
															X_full,
															tau_mat,
															lambda_mat,
															n_max,
															max_iters,
															None,
															eps)

	return mu, nu, R, R_k, constr_dict