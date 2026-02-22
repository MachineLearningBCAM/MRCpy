import numpy as np
from .ccg_utils import generate_cols, generate_rows, add_constr, add_var
from .mrc_lp_large_n_m import mrc_lp_large_n_m_model_gurobi

def mrc_ccg_large_n_m(F_, b_, X, phi_ob, tau_, lambda_, idx_cols, n_max=400, m_max=400, nu_init=None, mu_init=None, eps_1=1e-2, eps_2=1e-5, max_iters=150):
	"""
	Column and constraint generation algorithm for Minimax Risk Classifiers
	with large numbers of samples and features.

	This function implements a dual approach that iteratively adds both
	features (columns) and constraints (rows) to solve the MRC optimization
	problem efficiently for high-dimensional data. The algorithm alternates
	between solving the current restricted problem, identifying violated dual
	constraints (features to add), and identifying violated primal constraints
	(samples to add).

	Parameters
	----------
	F_ : numpy.ndarray of shape (n_constraints, n_features)
		Initial constraint coefficient matrix. Each row represents a constraint
		and each column represents a feature. This matrix is extended during
		the algorithm as new constraints are added.

	b_ : numpy.ndarray of shape (n_constraints,)
		Right-hand side values for the initial constraints. This vector is
		extended during the algorithm as new constraints are added.

	X : numpy.ndarray of shape (n_samples, n_features)
		Training feature matrix. Each row represents a training sample and
		each column represents a feature.

	phi_ob : BasePhi instance
		Feature mapping object that transforms input data. This object must
		have methods `eval_x` for evaluating feature mappings and attributes
		`n_classes` and `fit_intercept`.

	tau_ : numpy.ndarray of shape (n_classes * d,)
		Flattened mean estimates for each feature across each class. This is
		obtained by flattening tau_mat of shape (n_classes, d) in C-order,
		resulting in features sequentially concatenated by classes.

	lambda_ : numpy.ndarray of shape (n_classes * d,)
		Flattened deviation estimates (uncertainty bounds) for each feature.
		This is obtained by flattening lambda_mat of shape (n_classes, d) in
		C-order. Shape matches tau_.

	idx_cols : array-like of int
		Initial list of feature indices to include in the optimization.
		These features form the starting working set. Will be converted to
		a list and extended during the algorithm.

	n_max : int, default=400
		Maximum number of constraints (samples) to add per iteration. Controls
		the rate at which the constraint set grows. Larger values may speed up
		convergence but increase memory usage.

	m_max : int, default=400
		Maximum number of features (columns) to add per iteration. Controls
		the rate at which the feature set grows. Larger values may speed up
		convergence but increase computational cost.

	nu_init : float, optional, default=None
		Initial value for the nu parameter (intercept term). If provided,
		used as a warm start for the optimization. If None, the solver
		determines the initial value.

	mu_init : numpy.ndarray, optional, default=None
		Initial values for mu parameters corresponding to features in idx_cols.
		If provided, used as a warm start for the optimization. Must have
		length equal to len(idx_cols).

	eps_1 : float, default=1e-2
		Constraint violation threshold for primal constraints. Constraints
		violated by more than this amount will be added to the model. Smaller
		values lead to more constraints being added and tighter solutions.

	eps_2 : float, default=1e-5
		Feature violation threshold for dual constraints. Features with dual
		violations exceeding this amount will be added to the model. Smaller
		values lead to more features being added and potentially better solutions.

	max_iters : int, default=150
		Maximum number of column/constraint generation iterations. The algorithm
		terminates when either no violations remain or this limit is reached.

	Returns
	-------
	mu : numpy.ndarray of shape (len(idx_cols),)
		Learned feature coefficients for the selected features. These are the
		optimal weights for features in the final working set.

	nu : float
		Learned intercept parameter. This is the bias term in the linear
		classifier.

	R : float
		Final objective value representing the optimized upper bound on the
		worst-case error probability.

	R_k : list of float
		Objective values at each iteration, tracking convergence. The length
		equals the number of iterations performed plus one (for the initial
		solution).

	Notes
	-----
	The algorithm modifies the input arrays `F_`, `b_`, and list `idx_cols`
	in-place by extending them with newly added constraints and features.

	The stopping criteria are:
	- No features violate dual constraints by more than eps_2, AND
	- No samples violate primal constraints by more than eps_1, OR
	- Maximum iterations (max_iters) is reached

	The algorithm uses Gurobi as the LP solver. Ensure Gurobi is properly
	installed and licensed.

	The function prints progress information including the worst-case error
	probability at each iteration.

	Examples
	--------
	>>> import numpy as np
	>>> from MRCpy import BasePhi
	>>> # Create feature matrix
	>>> X = np.random.randn(1000, 50)
	>>> # Initialize feature mapping
	>>> phi = BasePhi(n_classes=2)
	>>> # Compute feature statistics
	>>> tau = np.mean(X, axis=0)
	>>> lambda_ = np.std(X, axis=0)
	>>> # Initial feature set
	>>> idx_cols = list(range(10))
	>>> # Initialize constraint matrix
	>>> F_init = np.zeros((10, 50))
	>>> b_init = np.zeros(10)
	>>> # Run CCG algorithm
	>>> mu, nu, R, R_k = mrc_ccg_large_n_m(
	...     F_init, b_init, X, phi, tau, lambda_, idx_cols,
	...     n_max=100, m_max=100, eps_1=1e-2, eps_2=1e-5, max_iters=50
	... )
	"""


	# Initialization
	R_k = []
	n_tries = []
	last_checked = 0
	idx_cols = idx_cols.tolist()

	# Indices of variables not selected
	not_idx_cols = list(set(np.arange(tau_.size)) - set(idx_cols))
	# Number of exponential constraints selected for each sample
	n_constraint_xi = np.zeros(X.shape[0])

	# Solve the initial optimization.
	MRC_model = mrc_lp_large_n_m_model_gurobi(F_,
							 	  			  b_,
							 	  			  tau_,
							      			  lambda_,
											  idx_cols,
							      			  nu_init,
							      			  mu_init)

	R_k.append(MRC_model.objVal)

	# Gurobi model variables
	mu_plus = np.asarray([(MRC_model.getVarByName("mu_+_" + str(i))) for i in idx_cols])
	mu_minus = np.asarray([(MRC_model.getVarByName("mu_-_" + str(i))) for i in idx_cols])
	nu_pos = MRC_model.getVarByName("nu_+")
	nu_neg = MRC_model.getVarByName("nu_-")

	# Obtain the primal and dual solutions to generate columns and constraints.
	alpha = [constr.Pi for constr in MRC_model.getConstrs()]
	mu = np.asarray([mu_plus_i.x for mu_plus_i in mu_plus]) - np.asarray([mu_minus_i.x for mu_minus_i in mu_minus])
	nu = nu_pos.x - nu_neg.x

	# Generate the columns to be added to the model.
	cols_to_add, n_features_generated = generate_cols(F_,
													  tau_,
													  lambda_,
													  alpha,
													  not_idx_cols,
													  m_max,
													  eps_2)

	# Generate the constraints to be added to the model.
	F_new, b_new, n_constr_generated, n_tries, last_checked, n_constraint_xi = generate_rows(X,
																							 phi_ob,
																							 idx_cols,
																							 mu,
																							 nu,
																							 n_max,
																							 eps_1,
																							 last_checked,
																							 n_constraint_xi)

	if n_constr_generated > 0:
		MRC_model = add_constr(MRC_model, F_new, b_new, idx_cols, mu_plus, mu_minus, nu_pos, nu_neg)

		# Update the coefficient matrix
		if F_.shape[0] == 0:
			F_ = F_new
			b_ = b_new
		else:
			F_ = np.vstack((F_, F_new))
			b_ = np.append(b_, b_new)
		MRC_model.update()

	# Add the cols and rows to the model
	if n_features_generated > 0:
		MRC_model = add_var(MRC_model, F_, tau_, lambda_, cols_to_add)

		# Add and remove column indices from the current set		
		idx_cols.extend(cols_to_add)

		# Add and remove column indices from the set to be checked
		for i in cols_to_add:
			not_idx_cols.remove(i)

		MRC_model.update()

	k = 0
	while(n_features_generated + n_constr_generated > 0 and k < max_iters):

		# Solve the updated optimization and get the dual solution.
		MRC_model.optimize()

		R_k.append(MRC_model.objVal)
		
		# Gurobi model variables
		mu_plus = np.asarray([(MRC_model.getVarByName("mu_+_" + str(i))) for i in idx_cols])
		mu_minus = np.asarray([(MRC_model.getVarByName("mu_-_" + str(i))) for i in idx_cols])
		nu_pos = MRC_model.getVarByName("nu_+")
		nu_neg = MRC_model.getVarByName("nu_-")

		# Obtain the primal and dual solutions to generate columns and constraints.
		alpha = [constr.Pi for constr in MRC_model.getConstrs()]
		mu = np.asarray([mu_plus_i.x for mu_plus_i in mu_plus]) - np.asarray([mu_minus_i.x for mu_minus_i in mu_minus])
		nu = nu_pos.x - nu_neg.x

		# Generate the columns to be added to the model.
		cols_to_add, n_features_generated = generate_cols(F_,
														  tau_,
														  lambda_,
														  alpha,
														  not_idx_cols,
														  m_max,
														  eps_2)

		# Generate the constraints to be added to the model.
		F_new, b_new, n_constr_generated, n_tries, last_checked, n_constraint_xi = generate_rows(X,
																								 phi_ob,
																								 idx_cols,
																								 mu,
																								 nu,
																								 n_max,
																								 eps_1,
																								 last_checked,
																								 n_constraint_xi)


		if n_constr_generated > 0:
			# Gurobi model variables
			mu_plus = np.asarray([(MRC_model.getVarByName("mu_+_" + str(i))) for i in idx_cols])
			mu_minus = np.asarray([(MRC_model.getVarByName("mu_-_" + str(i))) for i in idx_cols])
			MRC_model = add_constr(MRC_model, F_new, b_new, idx_cols, mu_plus, mu_minus, nu_pos, nu_neg)

			# Update the coefficient matrix
			F_ = np.vstack((F_, F_new))
			b_ = np.append(b_, b_new)
			MRC_model.update()

		# Add the cols and rows to the model
		if n_features_generated > 0:
			MRC_model = add_var(MRC_model, F_, tau_, lambda_, cols_to_add)

			# Add and remove column indices from the current set		
			idx_cols.extend(cols_to_add)

			# Add and remove column indices from the set to be checked
			for i in cols_to_add:
				not_idx_cols.remove(i)

			MRC_model.update()

		k = k + 1

	# Obtain the final primal solution.
	if k == max_iters:
		MRC_model.optimize()
		mu_plus = np.asarray([(MRC_model.getVarByName("mu_+_" + str(i))) for i in idx_cols])
		mu_minus = np.asarray([(MRC_model.getVarByName("mu_-_" + str(i))) for i in idx_cols])
		nu_pos = MRC_model.getVarByName("nu_+")
		nu_neg = MRC_model.getVarByName("nu_-")
		mu = np.asarray([mu_plus_i.x for mu_plus_i in mu_plus]) - np.asarray(
			[mu_minus_i.x for mu_minus_i in mu_minus])
		nu = nu_pos.x - nu_neg.x
		R_k.append(MRC_model.objVal)

	R 			= MRC_model.objVal

	n_active_constr = 0
	for c in MRC_model.getConstrs():
		if c.Slack < 1e-6:
			n_active_constr = n_active_constr + 1

	return mu, nu, R, R_k
