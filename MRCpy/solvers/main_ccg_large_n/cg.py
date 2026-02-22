import numpy as np
from .mrc_dual_lp import mrc_dual_lp_model
from .cg_utils import select

def mrc_ccg_large_n(F_, b_, phi, tau_, lambda_, n_max=400, max_iters=60, warm_start=None, eps=1e-2):
	"""
	Constraint generation algorithm for Minimax Risk Classifiers with large
	numbers of samples.

	This function implements a dual formulation approach that iteratively adds
	violated constraints to solve the MRC optimization problem efficiently for
	datasets with large numbers of samples. The algorithm alternates between
	solving the current restricted dual problem and identifying violated primal
	constraints to add.

	Parameters
	----------
	F_ : numpy.ndarray of shape (n_constraints, n_features)
		Initial constraint coefficient matrix. Each row represents a constraint
		and each column represents a feature. This matrix is extended during
		the algorithm as new constraints are added.

	b_ : numpy.ndarray of shape (n_constraints,)
		Right-hand side values for the initial constraints. This vector is
		extended during the algorithm as new constraints are added.

	phi : numpy.ndarray of shape (n_samples, n_classes, n_features)
		Feature mapping matrix for all samples. For each sample, phi[i] is a
		matrix of shape (n_classes, n_features) where each row represents the
		feature vector for one class.

	tau_ : numpy.ndarray of shape (n_features,)
		Mean estimates for each feature across the training distribution.
		Used to define the uncertainty set in the MRC formulation.

	lambda_ : numpy.ndarray of shape (n_features,)
		Deviation estimates (uncertainty bounds) for each feature. Represents
		the maximum deviation from tau_ allowed in the uncertainty set.

	n_max : int, default=400
		Maximum number of constraints (samples) to add per iteration. Controls
		the rate at which the constraint set grows. Larger values may speed up
		convergence but increase memory usage.

	max_iters : int, default=60
		Maximum number of constraint generation iterations. The algorithm
		terminates when either no violations remain or this limit is reached.

	warm_start : numpy.ndarray, optional, default=None
		Initial values for dual variables (alpha). If provided, used as a warm
		start for the optimization. Must have length equal to F_.shape[0].

	eps : float, default=1e-2
		Constraint violation threshold. Constraints violated by more than this
		amount will be added to the model. Smaller values lead to more
		constraints being added and tighter solutions.

	Returns
	-------
	mu : numpy.ndarray of shape (n_features,)
		Learned feature coefficients. These are the optimal weights for features
		obtained from the dual solution.

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
	The algorithm modifies the input arrays `F_` and `b_` in-place by extending
	them with newly added constraints.

	The stopping criteria are:
	- No samples violate primal constraints by more than eps, OR
	- Maximum iterations (max_iters) is reached

	The algorithm uses Gurobi as the LP solver in dual formulation. Ensure
	Gurobi is properly installed and licensed.

	The function also removes redundant constraints (those with zero slack)
	during iterations to keep the model size manageable.

	Examples
	--------
	>>> import numpy as np
	>>> # Create feature mapping for 100 samples, 3 classes, 50 features
	>>> phi = np.random.randn(100, 3, 50)
	>>> # Compute feature statistics
	>>> tau = np.mean(phi.reshape(-1, 50), axis=0)
	>>> lambda_ = np.std(phi.reshape(-1, 50), axis=0)
	>>> # Initialize constraint matrix
	>>> F_init = np.random.randn(10, 50)
	>>> b_init = np.zeros(10)
	>>> # Run CG algorithm
	>>> mu, nu, R, R_k = mrc_ccg_large_n(
	...     F_init, b_init, phi, tau, lambda_,
	...     n_max=100, max_iters=50, eps=1e-2
	... )
	"""

	# Generate the matrices for the linear optimization of 0-1 MRC
	# from the feature mappings.

	R_k = []
	N_constr_dual = F_.shape[1]

	# Initial optimization
	MRC_model = mrc_dual_lp_model(F_,
							 	  b_,
							 	  tau_,
							      lambda_,
							      warm_start)

	R_k.append(MRC_model.objVal)

	# Primal solution
	mu_plus = [(MRC_model.getConstrByName("constr_+_" + str(i))).Pi for i in range(N_constr_dual)]
	mu_minus = [(MRC_model.getConstrByName("constr_-_" + str(i))).Pi for i in range(N_constr_dual)]
	nu = MRC_model.getConstrByName("constr_=").Pi

	mu = np.asarray(mu_plus) - np.asarray(mu_minus)
	mu[np.isclose(mu, 0)] = 0

	last_checked = 0

	search_indices = np.arange(phi.shape[0])

	# Add the columns to the model.
	MRC_model, F_, b_, count_added, last_checked = select(MRC_model,
														phi,
														F_,
														b_,
														n_max,
														mu,
														nu,
														eps,
														last_checked)

	# print("Total number of constraints violated: ")
	k = 0
	while(k < max_iters and count_added > 0):

		# Solve the updated optimization and get the dual solution.
		MRC_model.optimize()

		# Get the primal solution
		mu_plus = [(MRC_model.getConstrByName("constr_+_" + str(i))).Pi for i in range(N_constr_dual)]
		mu_minus = [(MRC_model.getConstrByName("constr_-_" + str(i))).Pi for i in range(N_constr_dual)]
		nu = MRC_model.getConstrByName("constr_=").Pi
		
		mu = np.asarray(mu_plus) - np.asarray(mu_minus)
		R_k.append(MRC_model.objVal)

		# Select the columns/features for the next iteration.
		MRC_model, F_, b_, count_added, last_checked = select(MRC_model,
															phi,
															F_,
															b_,
															n_max,
															mu,
															nu,
															eps,
															last_checked)

		k = k + 1

	# Obtain the final primal solution.
	R 			= MRC_model.objVal

	return mu, nu, R, R_k
