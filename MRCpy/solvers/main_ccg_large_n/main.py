"""
Main entry point for CG solver for large-sample MRC.

This module provides the main function for learning Minimax Risk Classifiers
on datasets with large numbers of samples. It coordinates the initialization
phase and the iterative constraint generation algorithm.
"""

from .cg import mrc_ccg_large_n
import numpy as np
import itertools as it
import scipy.special as scs

def main_large_n(phi, phi_ob, tau_mat, lambda_mat, n_max, max_iters, eps):
	"""
	Efficient learning of 0-1 MRCs for large numbers of samples.

	This function implements a constraint generation approach for learning
	Minimax Risk Classifiers on datasets with large numbers of samples. It
	uses a dual formulation and iteratively adds violated constraints to
	solve the MRC optimization problem efficiently.

	Note: This version handles multiclass problems but for better performance
	on multiclass problems, consider using the specialized implementation in
	main_ccg_large_n_multiclass.

	Parameters
	----------
	phi : numpy.ndarray of shape (n_samples, n_classes, n_features)
		Feature mapping matrix for all samples. For each sample, phi[i] is a
		matrix of shape (n_classes, n_features) where each row represents the
		feature vector for one class. This should be pre-computed using a
		feature mapping function.

	tau_mat : numpy.ndarray of shape (n_classes, d)
		Mean estimates for each feature across each class. Each row corresponds
		to a class, and each column corresponds to a feature. For binary
		classification, shape is (1, d).

	lambda_mat : numpy.ndarray of shape (n_classes, d)
		Deviation estimates (uncertainty bounds) for each feature across each
		class. Represents the maximum deviation from tau_mat allowed in the
		uncertainty set. Shape matches tau_mat.

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
	mu : numpy.ndarray of shape (n_features,)
		Learned feature coefficients. These are the optimal weights for
		features obtained from the dual solution.

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

	Notes
	-----
	The algorithm uses a two-phase approach:
	
	**Phase 1 - Initialization**:
	- Constructs initial constraint matrix based on class centroids (tau_mat)
	- Generates constraints for all possible class subsets from centroids
	- This avoids empty uncertainty set at initialization
	
	**Phase 2 - CG Iterations**:
	- Iteratively solves the restricted dual optimization problem
	- Identifies and adds violated primal constraints (samples)
	- Removes redundant constraints to keep model size manageable
	- Continues until convergence or max_iters is reached

	The algorithm requires Gurobi as the LP solver. Ensure Gurobi is properly
	installed and licensed before using this function.

	For binary classification (n_classes=2), the algorithm handles the feature
	mapping differently than for multiclass problems.

	The function uses a dual formulation, so the primal solution (mu, nu) is
	recovered from the dual constraints' Pi values.

	Examples
	--------
	>>> import numpy as np
	>>> from MRCpy import BasePhi
	>>> # Create feature mapping for 1000 samples, 3 classes, 50 features
	>>> X = np.random.randn(1000, 50)
	>>> y = np.random.randint(0, 3, 1000)
	>>> phi_obj = BasePhi(n_classes=3)
	>>> phi = phi_obj.eval_x(X)  # Shape: (1000, 3, 150) with one-hot
	>>> # Compute feature statistics per class
	>>> tau_mat = np.array([X[y==i].mean(axis=0) for i in range(3)])  # Shape: (3, 50)
	>>> lambda_mat = np.array([X[y==i].std(axis=0) for i in range(3)])  # Shape: (3, 50)
	>>> # Run CG algorithm
	>>> mu, nu, R, R_k = main_large_n(
	...     phi, tau_mat, lambda_mat,
	...     n_max=200, max_iters=50, eps=1e-2
	... )
	>>> print(f"Worst-case error bound: {R:.4f}")
	>>> print(f"Converged in {len(R_k)} iterations")

	References
	----------
	The algorithm is based on constraint generation techniques for large-scale
	optimization problems. See Mazumdar et al. for theoretical details on MRC
	and constraint generation methods.
	"""

	n = phi.shape[0]

	#-> Initialization.
	# 1. Using the constraints of the centroids as the initial points.
	# 	 This initialization avoids empty uncertainty set.
	if tau_mat.shape[0] == 1:
		# Binary classification
		n_classes = 2
	else:
		n_classes = tau_mat.shape[0]

	#-> Initialization.
	# Generate all the constraints using the tau_mat as instance
	# This will avoid the empty uncertainty set condition at the start.
	fit_intercept = phi_ob.fit_intercept
	phi_ob.fit_intercept = False
	phi_1 = phi_ob.eval_x(tau_mat)
	n_init = phi_1.shape[0]
	F = np.vstack(list(np.sum(phi_1[:, S, ], axis=1)
						for numVals in range(1, n_classes + 1)
						for S in it.combinations(np.arange(n_classes), numVals)))

	cardS = np.arange(1, n_classes + 1). \
		repeat([n_init * scs.comb(n_classes, numVals)
				for numVals in np.arange(1, n_classes + 1)])

	# Constraint coefficient matrix
	F = F / (cardS[:, np.newaxis])

	# The bounds on the constraints
	b = (1 / cardS) - 1
	phi_ob.fit_intercept = fit_intercept
	
	#-> Run the CG code.
	mu, nu, R, R_k = mrc_ccg_large_n(F,
									b,
									phi,
									tau_mat.flatten(),
									lambda_mat.flatten(),
									n_max,
									max_iters,
									None,
									eps)

	return mu, nu, R, R_k